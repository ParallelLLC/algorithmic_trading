"""algotrader 2.0 — a backtester that tries to prove itself wrong.

Most backtesting libraries answer "how much would this have made?". This one
answers the question that actually matters before you risk money: "how much of
that was luck?"

Quick start::

    from algotrader import LabConfig, run_lab

    report = run_lab(LabConfig(symbol="SPY", strategy="sma_cross"))
    print(report.verdict["verdict"])
"""

from .data import load_ohlcv, simulate_ohlcv
from .engine import run_backtest
from .lab import LabConfig, LabReport, run_arena, run_lab
from .metrics import compute_metrics
from .strategies import REGISTRY, get_strategy, list_strategies
from .types import BacktestResult, CostModel, MarketData
from .verdict import reality_score

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "LabConfig",
    "LabReport",
    "run_lab",
    "run_arena",
    "run_backtest",
    "compute_metrics",
    "load_ohlcv",
    "simulate_ohlcv",
    "get_strategy",
    "list_strategies",
    "REGISTRY",
    "BacktestResult",
    "CostModel",
    "MarketData",
    "reality_score",
]
