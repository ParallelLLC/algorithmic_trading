"""The Portfolio Lab: the honesty pipeline for cross-sectional strategies.

Same idea as :mod:`algotrader.lab`, but the questions change when you move from
one asset to many. A timing rule has to prove the market had structure. A book
that ranks names has to prove three harder things:

1. it picked the right names (cross-sectional permutation);
2. what it picked is not just a style you could buy in an ETF (attribution);
3. the universe it picked from contains the losers as well as the winners
   (survivorship).

All three are wired into the Reality Score alongside the usual selection-bias
and walk-forward machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .attribution import build_style_factors, factor_attribution
from .cross_sectional import CrossSectionalStrategy, get_xs_strategy, list_xs_strategies
from .metrics import infer_periods_per_year
from .panel import Panel, load_panel
from .portfolio import PortfolioResult, rebalance_schedule, run_portfolio_backtest
from .types import CostModel
from .validation.cross_permutation import CrossPermutationResult, cross_sectional_permutation_test
from .validation.deflated_sharpe import deflated_sharpe_ratio
from .validation.pbo import probability_of_backtest_overfitting
from .validation.walkforward import walk_forward_panel
from .verdict import reality_score

logger = logging.getLogger(__name__)

__all__ = ["PortfolioLabConfig", "PortfolioLabReport", "run_portfolio_lab", "run_portfolio_arena"]

ProgressFn = Optional[Callable[[float, str], None]]

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN",
    "META", "TSLA", "GOOGL", "GLD", "TLT", "BTC-USD",
]


@dataclass
class PortfolioLabConfig:
    symbols: Sequence[str] = tuple(DEFAULT_UNIVERSE)
    start: str = "2015-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    source: str = "auto"

    strategy: str = "xs_momentum"
    params: Dict[str, float] = field(default_factory=dict)

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps: float = 50.0
    lag: int = 1
    gross_leverage: float = 1.0
    max_weight: Optional[float] = 0.25
    allow_short: bool = True
    rebalance: str = "M"
    capital: float = 1_000_000.0

    n_permutations: int = 150
    wf_folds: int = 4
    pbo_splits: int = 8
    grid_limit: int = 24
    seed: int = 0

    def costs(self, multiplier: float = 1.0) -> CostModel:
        return CostModel(
            commission_bps=self.commission_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            short_borrow_bps=self.short_borrow_bps * multiplier,
        )


@dataclass
class PortfolioLabReport:
    config: PortfolioLabConfig
    panel: Panel
    strategy: CrossSectionalStrategy
    params: Dict[str, float]
    backtest: PortfolioResult
    permutation: Optional[CrossPermutationResult] = None
    dsr: Dict[str, float] = field(default_factory=dict)
    pbo: Dict[str, object] = field(default_factory=dict)
    walkforward: Dict[str, object] = field(default_factory=dict)
    attribution: Dict[str, object] = field(default_factory=dict)
    trials: Dict[str, object] = field(default_factory=dict)
    verdict: Dict[str, object] = field(default_factory=dict)
    cost_stress: Dict[str, float] = field(default_factory=dict)

    @property
    def survivorship(self):
        return self.panel.survivorship()


def _trial_matrix(
    panel: Panel,
    strategy: CrossSectionalStrategy,
    cfg: PortfolioLabConfig,
    schedule: pd.Series,
    progress: ProgressFn = None,
) -> tuple[np.ndarray, List[float], List[str]]:
    """Backtest every parameter variant, for the Deflated Sharpe and PBO inputs."""
    grid = strategy.grid(limit=cfg.grid_limit)
    costs = cfg.costs()
    columns, sharpes, labels = [], [], []

    for i, params in enumerate(grid):
        weights = strategy.generate(panel, params)
        result = run_portfolio_backtest(
            panel, weights, costs=costs, lag=cfg.lag,
            gross_leverage=cfg.gross_leverage, max_weight=cfg.max_weight,
            allow_short=cfg.allow_short, rebalance_on=schedule,
        )
        columns.append(result.returns.to_numpy(dtype=float))
        sharpes.append(result.sharpe)
        labels.append(", ".join(f"{k}={v}" for k, v in params.items()) or "default")
        if progress is not None and i % 3 == 0:
            progress((i + 1) / max(len(grid), 1), f"Variant {i + 1}/{len(grid)}")

    matrix = np.column_stack(columns) if columns else np.zeros((len(panel), 0))
    return matrix, sharpes, labels


def run_portfolio_lab(cfg: PortfolioLabConfig, progress: ProgressFn = None) -> PortfolioLabReport:
    """Run the full cross-sectional honesty pipeline."""

    def step(fraction: float, message: str) -> None:
        if progress is not None:
            progress(min(max(fraction, 0.0), 1.0), message)

    step(0.02, f"Loading {len(cfg.symbols)} symbols")
    panel = load_panel(cfg.symbols, cfg.start, cfg.end, cfg.interval, cfg.source)
    if len(panel) < 250:
        raise ValueError(
            f"Only {len(panel)} bars available. Widen the date range — a cross-sectional "
            "book cannot be validated on less than a year of data."
        )

    strategy = get_xs_strategy(cfg.strategy)
    params = strategy.clean(cfg.params)
    ppy = infer_periods_per_year(panel.index)
    schedule = rebalance_schedule(panel.index, cfg.rebalance)

    step(0.10, "Running the backtest")
    weights = strategy.generate(panel, params)
    backtest = run_portfolio_backtest(
        panel, weights, costs=cfg.costs(), lag=cfg.lag,
        gross_leverage=cfg.gross_leverage, max_weight=cfg.max_weight,
        allow_short=cfg.allow_short, initial_capital=cfg.capital,
        periods_per_year=ppy, rebalance_on=schedule,
        meta={"strategy": strategy.key, "params": params, "rebalance": cfg.rebalance},
    )

    step(0.16, "Stress-testing costs")
    stressed = run_portfolio_backtest(
        panel, weights, costs=cfg.costs(3.0), lag=cfg.lag,
        gross_leverage=cfg.gross_leverage, max_weight=cfg.max_weight,
        allow_short=cfg.allow_short, periods_per_year=ppy, rebalance_on=schedule,
    )
    base_sharpe = backtest.sharpe
    cost_stress = {
        "sharpe_1x": base_sharpe,
        "sharpe_3x": stressed.sharpe,
        "ratio": float(stressed.sharpe / base_sharpe) if base_sharpe > 1e-9 else 0.0,
        "return_3x": float(stressed.metrics.get("total_return", 0.0)),
    }

    step(0.22, "Backtesting every parameter variant")
    matrix, trial_sharpes, labels = _trial_matrix(
        panel, strategy, cfg, schedule, lambda f, m: step(0.22 + 0.14 * f, m)
    )
    n_trials = max(len(trial_sharpes), 1)

    step(0.38, "Deflating the Sharpe ratio for selection bias")
    dsr = deflated_sharpe_ratio(
        backtest.returns.to_numpy(dtype=float),
        sharpe_annual=base_sharpe,
        periods_per_year=ppy,
        n_trials=n_trials,
        trial_sharpes=trial_sharpes if n_trials > 1 else None,
    )

    step(0.42, "Measuring backtest overfitting")
    pbo = probability_of_backtest_overfitting(matrix, n_splits=cfg.pbo_splits, labels=labels)

    step(0.46, "Shuffling names within each date")
    permutation = None
    if cfg.n_permutations > 0:
        permutation = cross_sectional_permutation_test(
            panel, weights, n_permutations=cfg.n_permutations, lag=cfg.lag,
            gross_leverage=cfg.gross_leverage, max_weight=cfg.max_weight,
            allow_short=cfg.allow_short, rebalance_on=schedule, seed=cfg.seed,
            progress=lambda f, m: step(0.46 + 0.30 * f, m),
        )

    step(0.78, "Attributing returns to style factors")
    try:
        factors = build_style_factors(panel)
        attribution = factor_attribution(backtest.returns, factors, ppy)
    except Exception as exc:  # noqa: BLE001 - attribution must never sink a run
        logger.warning("Attribution failed: %s", exc)
        attribution = {"available": False, "note": f"Attribution unavailable: {exc}"}

    step(0.86, "Walking the strategy forward")
    wf = walk_forward_panel(
        panel, strategy, n_folds=cfg.wf_folds, costs=cfg.costs(), lag=cfg.lag,
        gross_leverage=cfg.gross_leverage, allow_short=cfg.allow_short,
        rebalance=cfg.rebalance, grid_limit=min(cfg.grid_limit, 12),
        progress=lambda f, m: step(0.86 + 0.10 * f, m),
    )

    step(0.98, "Grading")
    survivorship = panel.survivorship()
    verdict = reality_score(
        metrics=backtest.metrics,
        benchmark_metrics=backtest.benchmark_metrics,
        p_value=permutation.p_value if permutation else None,
        dsr=dsr.get("dsr"),
        pbo=pbo.get("pbo"),
        wf_efficiency=wf.get("efficiency"),
        wf_win_rate=wf.get("oos_win_rate"),
        cost_stress_ratio=cost_stress["ratio"],
        attribution=attribution,
        survivorship=survivorship,
        benchmark_name="The equal-weight universe",
        permutation_label="books with the same shape but randomly chosen names",
    )

    step(1.0, "Done")
    return PortfolioLabReport(
        config=cfg,
        panel=panel,
        strategy=strategy,
        params=params,
        backtest=backtest,
        permutation=permutation,
        dsr=dsr,
        pbo=pbo,
        walkforward=wf,
        attribution=attribution,
        trials={"n": n_trials, "sharpes": trial_sharpes, "labels": labels},
        verdict=verdict,
        cost_stress=cost_stress,
    )


def run_portfolio_arena(
    cfg: PortfolioLabConfig,
    strategy_keys: Optional[List[str]] = None,
    n_permutations: int = 80,
    progress: ProgressFn = None,
) -> tuple[pd.DataFrame, Panel, Dict[str, PortfolioResult]]:
    """Race every cross-sectional strategy on one universe, ranked by evidence."""
    panel = load_panel(cfg.symbols, cfg.start, cfg.end, cfg.interval, cfg.source)
    ppy = infer_periods_per_year(panel.index)
    schedule = rebalance_schedule(panel.index, cfg.rebalance)
    costs = cfg.costs()

    keys = strategy_keys or [s.key for s in list_xs_strategies()]
    factors = build_style_factors(panel)
    rows, books = [], {}

    for i, key in enumerate(keys):
        strategy = get_xs_strategy(key)
        weights = strategy.generate(panel, strategy.defaults())
        result = run_portfolio_backtest(
            panel, weights, costs=costs, lag=cfg.lag, gross_leverage=cfg.gross_leverage,
            max_weight=cfg.max_weight, allow_short=cfg.allow_short,
            initial_capital=cfg.capital, periods_per_year=ppy, rebalance_on=schedule,
        )
        books[key] = result

        p_value = float("nan")
        if n_permutations > 0:
            p_value = cross_sectional_permutation_test(
                panel, weights, n_permutations=n_permutations, lag=cfg.lag,
                gross_leverage=cfg.gross_leverage, max_weight=cfg.max_weight,
                allow_short=cfg.allow_short, rebalance_on=schedule, seed=cfg.seed,
                observed=result.sharpe,
            ).p_value

        dsr = deflated_sharpe_ratio(
            result.returns.to_numpy(dtype=float), sharpe_annual=result.sharpe,
            periods_per_year=ppy, n_trials=len(strategy.grid(limit=cfg.grid_limit)),
        )
        attr = factor_attribution(result.returns, factors, ppy)

        rows.append({
            "Strategy": strategy.name,
            "key": key,
            "Family": strategy.family,
            "Return": result.metrics.get("total_return", 0.0),
            "CAGR": result.metrics.get("cagr", 0.0),
            "Sharpe": result.sharpe,
            "MaxDD": result.metrics.get("max_drawdown", 0.0),
            "Turnover": result.metrics.get("turnover_ann", 0.0),
            "p-value": p_value,
            "DSR": dsr["dsr"],
            "Alpha t": attr.get("alpha_t_stat", float("nan")) if attr.get("available") else float("nan"),
        })
        if progress is not None:
            progress((i + 1) / len(keys), f"{strategy.name} ({i + 1}/{len(keys)})")

    table = pd.DataFrame(rows)
    if not table.empty:
        table["Evidence"] = (1.0 - table["p-value"].fillna(0.5)) * table["DSR"]
        table = table.sort_values("Evidence", ascending=False).reset_index(drop=True)
        table.insert(0, "#", table.index + 1)
    return table, panel, books
