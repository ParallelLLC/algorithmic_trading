"""algotrader 2.0 — a backtester that tries to prove itself wrong.

Most backtesting libraries answer "how much would this have made?". This one
answers the question that actually matters before you risk money: "how much of
that was luck?"

Quick start::

    from algotrader import LabConfig, run_lab

    report = run_lab(LabConfig(symbol="SPY", strategy="sma_cross"))
    print(report.verdict["verdict"])
"""

from .attribution import build_style_factors, factor_attribution
from .cross_sectional import XS_REGISTRY, get_xs_strategy, list_xs_strategies
from .data import load_ohlcv, simulate_ohlcv
from .engine import run_backtest
from .lab import LabConfig, LabReport, run_arena, run_lab
from .metrics import compute_metrics
from .panel import Panel, load_panel
from .portfolio import rebalance_schedule, run_portfolio_backtest
from .portfolio_lab import PortfolioLabConfig, PortfolioLabReport, run_portfolio_arena, run_portfolio_lab
from .strategies import REGISTRY, get_strategy, list_strategies
from .types import BacktestResult, CostModel, MarketData
from .verdict import reality_score

__version__ = "2.1.0"

__all__ = [
    "__version__",
    # single asset
    "LabConfig",
    "LabReport",
    "run_lab",
    "run_arena",
    "run_backtest",
    "get_strategy",
    "list_strategies",
    "REGISTRY",
    # multi asset
    "Panel",
    "load_panel",
    "run_portfolio_backtest",
    "rebalance_schedule",
    "PortfolioLabConfig",
    "PortfolioLabReport",
    "run_portfolio_lab",
    "run_portfolio_arena",
    "get_xs_strategy",
    "list_xs_strategies",
    "XS_REGISTRY",
    "build_style_factors",
    "factor_attribution",
    # shared
    "compute_metrics",
    "load_ohlcv",
    "simulate_ohlcv",
    "BacktestResult",
    "CostModel",
    "MarketData",
    "reality_score",
]
