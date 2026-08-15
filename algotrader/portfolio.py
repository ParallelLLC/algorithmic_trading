"""Matrix portfolio engine.

The single-asset engine treats turnover as ``|target[t] - target[t-1]|``. That
is wrong the moment weights are fractional, because a position you did not
touch still *drifts*: hold 50% of your book in a name that doubles and you are
now at 67% without trading. Charging costs against the previous target instead
of the previous *actual* weight understates the cost of doing nothing and
overstates the cost of rebalancing.

This module models the drift explicitly and closes the gap:

    w_start[t] = target[t - lag]                     what we want to hold
    r_p[t]     = sum(w_start[t] * R[t])              portfolio return that bar
    w_end[t]   = w_start[t] * (1 + R[t]) / (1 + r_p[t])   drifted by the bar
    turnover[t] = sum |w_start[t] - w_end[t - 1]|    what we actually traded

Every step is a function of the current bar and the one before, so the whole
thing stays vectorised -- no Python loop over time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compute_metrics, infer_periods_per_year
from .panel import Panel
from .types import BacktestResult, CostModel

__all__ = ["run_portfolio_backtest", "PortfolioResult", "normalise_weights"]


class PortfolioResult(BacktestResult):
    """A :class:`BacktestResult` that also keeps the per-asset weight history."""

    def __init__(self, *args, weights: pd.DataFrame, held: pd.DataFrame, panel: Panel, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights  # requested, post-constraint
        self.held = held  # actually held during each bar
        self.panel = panel

    def attribution(self) -> pd.Series:
        """Total return contribution per symbol, largest first."""
        contrib = (self.held * self.panel.returns().fillna(0.0)).sum(axis=0)
        return contrib.sort_values(ascending=False)


def normalise_weights(
    weights: pd.DataFrame,
    listed: pd.DataFrame,
    gross_leverage: float = 1.0,
    max_weight: Optional[float] = None,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Apply the constraints a real book has, in the order a real book applies them.

    ``listed`` is "did this name have a price when the decision was made" --
    deliberately not the stricter "is a return defined over this bar" mask. A
    decision taken at Monday's close only needs Monday's price to exist; whether
    the position can actually be carried is enforced after the lag shift.
    """
    w = weights.reindex(index=listed.index, columns=listed.columns).astype(float).fillna(0.0)

    # You cannot ask for exposure to something that is not listed yet.
    w = w.where(listed, 0.0)

    if not allow_short:
        w = w.clip(lower=0.0)
    if max_weight is not None:
        w = w.clip(-abs(max_weight), abs(max_weight))

    # Scale down (never up) so gross exposure respects the leverage cap.
    gross = w.abs().sum(axis=1)
    scale = np.minimum(1.0, gross_leverage / gross.replace(0.0, np.nan))
    return w.mul(scale.fillna(1.0), axis=0)


def run_portfolio_backtest(
    panel: Panel,
    weights: pd.DataFrame,
    costs: Optional[CostModel] = None,
    lag: int = 1,
    gross_leverage: float = 1.0,
    max_weight: Optional[float] = None,
    allow_short: bool = True,
    initial_capital: float = 100_000.0,
    periods_per_year: Optional[int] = None,
    rf: float = 0.0,
    benchmark: Optional[pd.Series] = None,
    rebalance_on: Optional[pd.Series] = None,
    meta: Optional[dict] = None,
) -> PortfolioResult:
    """Backtest a ``T x N`` weight matrix against a panel.

    ``weights[t]`` is the exposure decided using information up to the close of
    bar ``t``; it is held from bar ``t + lag``. The default benchmark is the
    equal-weight universe, which is a far more honest comparison for a
    cross-sectional strategy than any single ticker.
    """
    if len(panel) < 2:
        raise ValueError("Cannot backtest a panel with fewer than two bars")
    if lag < 1:
        raise ValueError("lag must be >= 1; lag=0 would trade on unavailable information")

    costs = costs or CostModel()
    ppy = periods_per_year or infer_periods_per_year(panel.index)

    asset_returns = panel.returns().fillna(0.0)
    tradable = panel.tradable()
    listed = panel.close.notna()

    target = normalise_weights(weights, listed, gross_leverage, max_weight, allow_short)
    held = target.shift(lag).fillna(0.0)
    # Re-apply tradability after the shift: a name can delist between the
    # decision and the fill, and we must not be holding it when it does.
    held = held.where(tradable, 0.0)

    if rebalance_on is not None:
        held = _apply_rebalance_schedule(held, asset_returns, tradable, rebalance_on)

    gross_return = (held * asset_returns).sum(axis=1)

    # Weights after the bar's move, renormalised to the new portfolio value.
    growth = (1.0 + gross_return).replace(0.0, np.nan)
    drifted = (held * (1.0 + asset_returns)).div(growth, axis=0).fillna(0.0)
    previous = drifted.shift(1).fillna(0.0)

    traded = (held - previous).abs().sum(axis=1)
    trade_cost = traded * (costs.one_way_bps / 1e4)
    borrow_cost = held.clip(upper=0.0).abs().sum(axis=1) * (costs.short_borrow_bps / 1e4) / ppy
    total_cost = trade_cost + borrow_cost

    net = gross_return - total_cost
    equity = initial_capital * (1.0 + net).cumprod()

    if benchmark is None:
        # Equal weight across whatever was tradable on each bar.
        counts = tradable.sum(axis=1).replace(0, np.nan)
        equal = tradable.astype(float).div(counts, axis=0).fillna(0.0)
        benchmark = (equal.shift(lag).fillna(0.0) * asset_returns).sum(axis=1)
    benchmark = benchmark.reindex(panel.index).fillna(0.0)
    benchmark_equity = initial_capital * (1.0 + benchmark).cumprod()

    exposure = held.abs().sum(axis=1)
    metrics = compute_metrics(net, equity, exposure, ppy, rf)
    metrics.update(_portfolio_metrics(held, traded, metrics.get("years", 1.0)))

    survivorship = panel.survivorship()

    result = PortfolioResult(
        equity=equity,
        returns=net,
        gross_returns=gross_return,
        position=exposure,
        target=target.abs().sum(axis=1),
        costs=total_cost,
        benchmark_equity=benchmark_equity,
        metrics=metrics,
        benchmark_metrics=compute_metrics(benchmark, benchmark_equity, None, ppy, rf),
        meta={
            "lag": lag,
            "gross_leverage": gross_leverage,
            "max_weight": max_weight,
            "allow_short": allow_short,
            "initial_capital": initial_capital,
            "periods_per_year": ppy,
            "n_symbols": len(panel.symbols),
            "survivorship": survivorship,
            **(meta or {}),
        },
        weights=target,
        held=held,
        panel=panel,
    )
    return result


def _apply_rebalance_schedule(
    held: pd.DataFrame,
    asset_returns: pd.DataFrame,
    tradable: pd.DataFrame,
    rebalance_on: pd.Series,
) -> pd.DataFrame:
    """Trade only on rebalance bars; let the book drift in between.

    Without this, a monthly strategy whose target is constant between
    rebalances gets charged turnover every single bar for holding still --
    the model would be paying to *prevent* drift that a real book simply lets
    happen. This is the one genuinely recursive step in the engine: today's
    holding depends on yesterday's drifted holding.
    """
    schedule = rebalance_on.reindex(held.index).fillna(False).to_numpy(dtype=bool)
    target = held.to_numpy(dtype=float)
    returns = asset_returns.to_numpy(dtype=float)
    can_hold = tradable.to_numpy(dtype=bool)

    n_bars, n_assets = target.shape
    out = np.zeros((n_bars, n_assets), dtype=float)
    carried = np.zeros(n_assets, dtype=float)

    for t in range(n_bars):
        current = target[t] if schedule[t] else carried
        current = np.where(can_hold[t], current, 0.0)
        out[t] = current
        # Drift into the next bar, renormalised to the new portfolio value.
        portfolio_return = float(current @ returns[t])
        growth = 1.0 + portfolio_return
        carried = current * (1.0 + returns[t]) / growth if abs(growth) > 1e-12 else current

    return pd.DataFrame(out, index=held.index, columns=held.columns)


def rebalance_schedule(index: pd.DatetimeIndex, frequency: str = "M") -> pd.Series:
    """Boolean per-bar mask marking rebalance dates.

    ``frequency`` is ``D`` (every bar), ``W``, ``M``, ``Q``, or an integer
    number of bars as a string.
    """
    frequency = str(frequency).upper().strip()
    if frequency in ("D", "1", "B", ""):
        return pd.Series(True, index=index)
    if frequency.isdigit():
        step = max(1, int(frequency))
        mask = np.zeros(len(index), dtype=bool)
        mask[::step] = True
        return pd.Series(mask, index=index)

    periods = {"W": index.to_period("W"), "M": index.to_period("M"), "Q": index.to_period("Q")}
    if frequency not in periods:
        raise ValueError(f"Unknown rebalance frequency '{frequency}'")
    period = periods[frequency]
    # First bar of each period -- known at the time, unlike the last bar.
    return pd.Series(period != pd.Series(period, index=index).shift(1).to_numpy(), index=index)


def _portfolio_metrics(held: pd.DataFrame, traded: pd.Series, years: float) -> dict:
    """Book-level statistics a portfolio manager will look for first."""
    absolute = held.abs()
    gross = absolute.sum(axis=1)
    active = (absolute > 1e-9).sum(axis=1)

    # Herfindahl on the gross book: 1.0 is everything in one name, 1/n is even.
    shares = absolute.div(gross.replace(0.0, np.nan), axis=0)
    hhi = (shares**2).sum(axis=1)

    return {
        "gross_exposure": float(gross.mean()),
        "net_exposure": float(held.sum(axis=1).mean()),
        "max_gross_exposure": float(gross.max()),
        "avg_positions": float(active.mean()),
        "max_positions": float(active.max()),
        "concentration_hhi": float(hhi.mean(skipna=True)) if hhi.notna().any() else float("nan"),
        "turnover_ann": float(traded.sum() / years) if years > 0 else 0.0,
        "n_trades": float((traded > 1e-9).sum()),
    }
