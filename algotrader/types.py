"""Core data types shared across the algotrader 2.0 stack.

Everything downstream (engine, validation, UI) speaks these types, so they are
deliberately small, immutable-ish and free of framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketData:
    """A validated OHLCV series plus provenance.

    Provenance matters here: the app is about honesty, so the UI always tells
    the user whether they are looking at real prices or a simulation.
    """

    symbol: str
    df: pd.DataFrame
    source: str  # "yfinance" | "bundled" | "synthetic"
    interval: str = "1d"
    note: str = ""

    @property
    def is_real(self) -> bool:
        return self.source in ("yfinance", "bundled")

    @property
    def start(self) -> pd.Timestamp:
        return self.df.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.df.index[-1]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)


@dataclass(frozen=True)
class CostModel:
    """Round-trip friction. All values are one-way, in basis points."""

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_bps: float = 50.0  # annualised, charged on short exposure

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


@dataclass
class BacktestResult:
    """Output of a single backtest run."""

    equity: pd.Series
    returns: pd.Series  # net of costs
    gross_returns: pd.Series
    position: pd.Series  # exposure actually held during each bar
    target: pd.Series  # exposure requested by the strategy
    costs: pd.Series
    benchmark_equity: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def sharpe(self) -> float:
        return float(self.metrics.get("sharpe", 0.0))

    @property
    def n_trades(self) -> int:
        return int(self.metrics.get("n_trades", 0))


@dataclass
class ValidationReport:
    """Everything we know about how much of a backtest is luck."""

    permutation_p_value: Optional[float] = None
    permutation_null: Optional[Any] = None  # np.ndarray of null Sharpes
    deflated_sharpe: Optional[float] = None
    probabilistic_sharpe: Optional[float] = None
    min_track_record_years: Optional[float] = None
    n_trials: int = 1
    pbo: Optional[float] = None
    pbo_detail: Dict[str, Any] = field(default_factory=dict)
    walkforward: Dict[str, Any] = field(default_factory=dict)
    reality_score: float = 0.0
    grade: str = "?"
    verdict: str = ""
    flags: list = field(default_factory=list)
