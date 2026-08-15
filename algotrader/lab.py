"""The Lab: one call that runs a backtest and then tries to disprove it.

This is the module both the Gradio Space and the CLI drive. Keeping the whole
pipeline here means the app and the command line can never disagree about what
a Reality Score means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data import load_ohlcv
from .engine import run_backtest
from .metrics import infer_periods_per_year
from .strategies import Strategy, get_strategy, list_strategies
from .types import BacktestResult, CostModel, MarketData
from .validation.deflated_sharpe import deflated_sharpe_ratio, min_track_record_length
from .validation.pbo import probability_of_backtest_overfitting
from .validation.permutation import PermutationResult, permutation_test
from .validation.walkforward import walk_forward
from .verdict import reality_score

logger = logging.getLogger(__name__)

__all__ = ["LabConfig", "LabReport", "run_lab", "run_arena"]

ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class LabConfig:
    symbol: str = "SPY"
    start: str = "2015-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    source: str = "yahoo"

    strategy: str = "sma_cross"
    params: Dict[str, float] = field(default_factory=dict)

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps: float = 50.0
    lag: int = 1
    allow_short: bool = True
    max_leverage: float = 1.0
    capital: float = 100_000.0

    n_permutations: int = 250
    permutation_method: str = "permute"
    block_size: int = 20
    wf_folds: int = 5
    pbo_splits: int = 8
    grid_limit: int = 40
    seed: int = 0

    def costs(self, multiplier: float = 1.0) -> CostModel:
        return CostModel(
            commission_bps=self.commission_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            short_borrow_bps=self.short_borrow_bps * multiplier,
        )


@dataclass
class LabReport:
    config: LabConfig
    market: MarketData
    strategy: Strategy
    params: Dict[str, float]
    backtest: BacktestResult
    permutation: Optional[PermutationResult] = None
    dsr: Dict[str, float] = field(default_factory=dict)
    pbo: Dict[str, object] = field(default_factory=dict)
    walkforward: Dict[str, object] = field(default_factory=dict)
    trials: Dict[str, object] = field(default_factory=dict)
    verdict: Dict[str, object] = field(default_factory=dict)
    cost_stress: Dict[str, float] = field(default_factory=dict)
    benchmark_correlation: float = float("nan")


def _trial_matrix(
    df: pd.DataFrame,
    strategy: Strategy,
    cfg: LabConfig,
    progress: ProgressFn = None,
) -> tuple[np.ndarray, List[float], List[str]]:
    """Backtest every parameter combination a researcher would plausibly try.

    The resulting ``T x N`` return matrix feeds both the Deflated Sharpe (how
    many variants were tried, and how spread out were they) and PBO.
    """
    grid = strategy.grid(limit=cfg.grid_limit)
    costs = cfg.costs()
    columns, sharpes, labels = [], [], []

    for i, params in enumerate(grid):
        target = strategy.generate(df, params)
        result = run_backtest(
            df, target, costs=costs, lag=cfg.lag,
            max_leverage=cfg.max_leverage, allow_short=cfg.allow_short,
        )
        columns.append(result.returns.to_numpy(dtype=float))
        sharpes.append(result.sharpe)
        labels.append(", ".join(f"{k}={v}" for k, v in params.items()) or "default")
        if progress is not None and i % 5 == 0:
            progress((i + 1) / max(len(grid), 1), f"Variant {i + 1}/{len(grid)}")

    matrix = np.column_stack(columns) if columns else np.zeros((len(df), 0))
    return matrix, sharpes, labels


def run_lab(cfg: LabConfig, progress: ProgressFn = None) -> LabReport:
    """Run the full honesty pipeline for one strategy on one symbol."""

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), message)

    step(0.02, "Loading market data")
    market = load_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.interval, cfg.source)
    df = market.df
    if len(df) < 120:
        raise ValueError(
            f"Only {len(df)} bars available for {cfg.symbol}. "
            "Widen the date range — anything shorter cannot be validated."
        )

    strategy = get_strategy(cfg.strategy)
    params = strategy.clean(cfg.params)
    ppy = infer_periods_per_year(df.index)

    step(0.10, "Running the backtest")
    target = strategy.generate(df, params)
    backtest = run_backtest(
        df,
        target,
        costs=cfg.costs(),
        lag=cfg.lag,
        max_leverage=cfg.max_leverage,
        allow_short=cfg.allow_short,
        initial_capital=cfg.capital,
        periods_per_year=ppy,
        meta={"symbol": market.symbol, "strategy": strategy.key, "params": params},
    )

    step(0.16, "Stress-testing costs")
    stressed = run_backtest(
        df, target, costs=cfg.costs(3.0), lag=cfg.lag,
        max_leverage=cfg.max_leverage, allow_short=cfg.allow_short,
        periods_per_year=ppy,
    )
    base_sharpe = backtest.sharpe
    cost_stress_ratio = float(stressed.sharpe / base_sharpe) if base_sharpe > 1e-9 else 0.0
    cost_stress = {
        "sharpe_1x": base_sharpe,
        "sharpe_3x": stressed.sharpe,
        "ratio": cost_stress_ratio,
        "return_3x": float(stressed.metrics.get("total_return", 0.0)),
    }

    step(0.22, "Backtesting every parameter variant")
    matrix, trial_sharpes, labels = _trial_matrix(
        df, strategy, cfg, lambda f, m: step(0.22 + 0.18 * f, m)
    )
    n_trials = max(len(trial_sharpes), 1)

    step(0.42, "Deflating the Sharpe ratio for selection bias")
    dsr = deflated_sharpe_ratio(
        backtest.returns.to_numpy(dtype=float),
        sharpe_annual=base_sharpe,
        periods_per_year=ppy,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes if n_trials > 1 else None,
    )
    mtrl = min_track_record_length(
        dsr["sr_per_period"], dsr["n_obs"], dsr["skew"], dsr["kurtosis"],
        benchmark=dsr["threshold_sr_per_period"],
    )
    dsr["min_track_record_bars"] = mtrl
    dsr["min_track_record_years"] = float(mtrl / ppy) if np.isfinite(mtrl) else float("inf")

    step(0.46, "Measuring backtest overfitting")
    pbo = probability_of_backtest_overfitting(matrix, n_splits=cfg.pbo_splits, labels=labels)

    step(0.50, "Shuffling the market")
    permutation = None
    if cfg.n_permutations > 0:
        permutation = permutation_test(
            df,
            lambda frame: strategy.generate(frame, params),
            n_permutations=cfg.n_permutations,
            method=cfg.permutation_method,
            block=cfg.block_size,
            costs=cfg.costs(),
            lag=cfg.lag,
            max_leverage=cfg.max_leverage,
            allow_short=cfg.allow_short,
            seed=cfg.seed,
            observed=base_sharpe,
            progress=lambda f, m: step(0.50 + 0.32 * f, m),
        )

    step(0.84, "Walking the strategy forward")
    wf = walk_forward(
        df, strategy, n_folds=cfg.wf_folds, costs=cfg.costs(), lag=cfg.lag,
        max_leverage=cfg.max_leverage, allow_short=cfg.allow_short,
        grid_limit=min(cfg.grid_limit, 24),
        progress=lambda f, m: step(0.84 + 0.12 * f, m),
    )

    bench_corr = float(
        pd.Series(backtest.returns).corr(backtest.benchmark_equity.pct_change().fillna(0.0))
    )

    step(0.98, "Grading")
    verdict = reality_score(
        metrics=backtest.metrics,
        benchmark_metrics=backtest.benchmark_metrics,
        p_value=permutation.p_value if permutation else None,
        dsr=dsr.get("dsr"),
        pbo=pbo.get("pbo"),
        wf_efficiency=wf.get("efficiency"),
        wf_win_rate=wf.get("oos_win_rate"),
        cost_stress_ratio=cost_stress_ratio,
        benchmark_correlation=bench_corr,
    )

    step(1.0, "Done")
    return LabReport(
        config=cfg,
        market=market,
        strategy=strategy,
        params=params,
        backtest=backtest,
        permutation=permutation,
        dsr=dsr,
        pbo=pbo,
        walkforward=wf,
        trials={"n": n_trials, "sharpes": trial_sharpes, "labels": labels, "matrix_shape": matrix.shape},
        verdict=verdict,
        cost_stress=cost_stress,
        benchmark_correlation=bench_corr,
    )


def run_arena(
    cfg: LabConfig,
    strategy_keys: Optional[List[str]] = None,
    n_permutations: int = 120,
    progress: ProgressFn = None,
) -> tuple[pd.DataFrame, MarketData, Dict[str, BacktestResult]]:
    """Race every strategy on the same market, ranked by evidence not returns.

    Buy & hold and the coin flip stay in the field on purpose: a leaderboard
    without a control group is marketing, not measurement.
    """
    market = load_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.interval, cfg.source)
    df = market.df
    ppy = infer_periods_per_year(df.index)
    costs = cfg.costs()

    keys = strategy_keys or [s.key for s in list_strategies()]
    rows, curves = [], {}

    for i, key in enumerate(keys):
        strategy = get_strategy(key)
        params = strategy.defaults()
        target = strategy.generate(df, params)
        result = run_backtest(
            df, target, costs=costs, lag=cfg.lag, max_leverage=cfg.max_leverage,
            allow_short=cfg.allow_short, initial_capital=cfg.capital, periods_per_year=ppy,
        )
        curves[key] = result

        p_value = None
        if n_permutations > 0:
            p_value = permutation_test(
                df,
                lambda frame, s=strategy, p=params: s.generate(frame, p),
                n_permutations=n_permutations,
                method=cfg.permutation_method,
                block=cfg.block_size,
                costs=costs,
                lag=cfg.lag,
                max_leverage=cfg.max_leverage,
                allow_short=cfg.allow_short,
                seed=cfg.seed,
                observed=result.sharpe,
            ).p_value

        grid_size = len(strategy.grid(limit=cfg.grid_limit))
        dsr = deflated_sharpe_ratio(
            result.returns.to_numpy(dtype=float),
            sharpe_annual=result.sharpe,
            periods_per_year=ppy,
            n_trials=grid_size,
        )

        rows.append(
            {
                "Strategy": strategy.name,
                "key": key,
                "Family": strategy.family,
                "Return": result.metrics.get("total_return", 0.0),
                "CAGR": result.metrics.get("cagr", 0.0),
                "Sharpe": result.sharpe,
                "MaxDD": result.metrics.get("max_drawdown", 0.0),
                "Trades": int(result.metrics.get("n_trades", 0)),
                "p-value": p_value if p_value is not None else float("nan"),
                "DSR": dsr["dsr"],
            }
        )
        if progress is not None:
            progress((i + 1) / len(keys), f"{strategy.name} ({i + 1}/{len(keys)})")

    table = pd.DataFrame(rows)
    if not table.empty:
        # Rank by evidence: a high Sharpe with a p-value of 0.4 is not a win.
        table["Evidence"] = (1.0 - table["p-value"].fillna(0.5)) * table["DSR"]
        table = table.sort_values("Evidence", ascending=False).reset_index(drop=True)
        table.insert(0, "#", table.index + 1)
    return table, market, curves
