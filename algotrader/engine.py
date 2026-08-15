"""Vectorised, look-ahead-free backtest engine.

Contract
--------
A strategy emits ``target[t]``: the exposure it wants, decided using only
information available at the close of bar ``t``. The engine holds
``position[t] = target[t - lag]`` during bar ``t`` and credits it with that
bar's close-to-close return. With the default ``lag=1`` this means "decide on
today's close, hold the position through tomorrow" -- the single place where
look-ahead could sneak in, and it is one line.

Costs are charged on exposure *changes*, so a strategy that flips daily pays
for it. Short exposure additionally accrues a borrow fee.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compute_metrics, infer_periods_per_year
from .types import BacktestResult, CostModel

__all__ = ["run_backtest", "bars_to_returns"]


def bars_to_returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close simple returns."""
    return df["close"].astype(float).pct_change().fillna(0.0)


def run_backtest(
    df: pd.DataFrame,
    target: pd.Series,
    costs: CostModel | None = None,
    lag: int = 1,
    max_leverage: float = 1.0,
    allow_short: bool = True,
    initial_capital: float = 100_000.0,
    periods_per_year: Optional[int] = None,
    rf: float = 0.0,
    meta: Optional[dict] = None,
) -> BacktestResult:
    """Run one backtest and return equity, returns and the full metric bundle."""
    if df.empty:
        raise ValueError("Cannot backtest an empty price frame")
    if lag < 1:
        raise ValueError("lag must be >= 1; lag=0 would trade on unavailable information")

    costs = costs or CostModel()
    ppy = periods_per_year or infer_periods_per_year(df.index)

    asset_ret = bars_to_returns(df)

    target = target.reindex(df.index).astype(float).fillna(0.0)
    lower = -max_leverage if allow_short else 0.0
    target = target.clip(lower, max_leverage)

    position = target.shift(lag).fillna(0.0)

    gross = position * asset_ret

    # Turnover is measured against the *drifted* weight, not the previous
    # target. Holding a full-notional long needs no rebalancing (the position
    # and the portfolio grow together), but a short does: lose 10% on a 100%
    # short and the weight drifts to -82%, so staying at -100% costs a trade.
    # See portfolio.py for the same formula in matrix form.
    growth = (1.0 + gross).replace(0.0, np.nan)
    drifted = (position * (1.0 + asset_ret)) / growth
    previous = drifted.shift(1).fillna(0.0)
    traded = position - previous
    trade_cost = traded.abs() * (costs.one_way_bps / 1e4)

    borrow_cost = position.clip(upper=0.0).abs() * (costs.short_borrow_bps / 1e4) / ppy
    total_cost = trade_cost + borrow_cost

    net = gross - total_cost
    equity = initial_capital * (1.0 + net).cumprod()
    benchmark_equity = initial_capital * (1.0 + asset_ret).cumprod()

    result = BacktestResult(
        equity=equity,
        returns=net,
        gross_returns=gross,
        position=position,
        target=target,
        costs=total_cost,
        benchmark_equity=benchmark_equity,
        metrics=compute_metrics(net, equity, position, ppy, rf),
        benchmark_metrics=compute_metrics(asset_ret, benchmark_equity, None, ppy, rf),
        meta={
            "lag": lag,
            "commission_bps": costs.commission_bps,
            "slippage_bps": costs.slippage_bps,
            "short_borrow_bps": costs.short_borrow_bps,
            "max_leverage": max_leverage,
            "allow_short": allow_short,
            "initial_capital": initial_capital,
            "periods_per_year": ppy,
            **(meta or {}),
        },
    )
    result.metrics["cost_drag_ann"] = float(total_cost.sum() / max(result.metrics.get("years", 1e-9), 1e-9))
    result.metrics["gross_sharpe"] = float(
        compute_metrics(gross, initial_capital * (1.0 + gross).cumprod(), None, ppy, rf).get("sharpe", 0.0)
    )
    return result


def fast_sharpe(
    asset_ret: np.ndarray,
    target: np.ndarray,
    one_way_bps: float,
    lag: int,
    periods_per_year: int,
) -> float:
    """Numpy-only Sharpe for hot loops (permutation tests, PBO grids).

    Mirrors :func:`run_backtest` exactly for the no-borrow case; it exists only
    because building a DataFrame 1000 times is the difference between a Space
    that answers in 4 seconds and one nobody waits for.
    """
    n = asset_ret.size
    position = np.empty(n, dtype=float)
    position[:lag] = 0.0
    position[lag:] = target[:-lag] if lag else target
    gross = position * asset_ret
    traded = np.empty(n, dtype=float)
    traded[0] = position[0]
    traded[1:] = np.diff(position)
    net = gross - np.abs(traded) * (one_way_bps / 1e4)
    net = net[np.isfinite(net)]
    if net.size < 2:
        return 0.0
    sd = net.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(net.mean() / sd * np.sqrt(periods_per_year))
