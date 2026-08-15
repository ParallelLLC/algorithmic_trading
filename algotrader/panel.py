"""Multi-asset price panels.

A :class:`Panel` is a set of aligned ``T x N`` frames -- one per OHLCV field,
one column per symbol. That is the shape cross-sectional work actually needs,
and it is what the portfolio engine consumes.

The important design choice here is that **missing data stays missing**. It is
tempting to forward-fill a symbol through the days it did not trade, but that
invents liquidity that never existed and quietly lets a strategy hold a
delisted stock forever. Instead the panel tracks exactly when each symbol was
tradable, which is also what makes survivorship measurable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .data import load_ohlcv
from .types import OHLCV_COLUMNS

__all__ = ["Panel", "load_panel", "SurvivorshipReport"]


@dataclass(frozen=True)
class SurvivorshipReport:
    """How much of this universe is made of winners we already know survived."""

    n_symbols: int
    n_alive_at_end: int
    n_delisted: int
    delisted_symbols: List[str]
    late_starters: List[str]
    survival_rate: float
    biased: bool
    note: str

    def as_flag(self) -> Optional[str]:
        return self.note if self.biased else None


@dataclass(frozen=True)
class Panel:
    """Aligned multi-asset OHLCV."""

    fields: Mapping[str, pd.DataFrame]
    sources: Mapping[str, str] = field(default_factory=dict)
    interval: str = "1d"
    note: str = ""

    def __post_init__(self) -> None:
        missing = [c for c in OHLCV_COLUMNS if c not in self.fields]
        if missing:
            raise ValueError(f"Panel is missing field(s): {', '.join(missing)}")
        reference = self.fields["close"]
        for name, frame in self.fields.items():
            if not frame.index.equals(reference.index) or list(frame.columns) != list(reference.columns):
                raise ValueError(f"Panel field '{name}' is not aligned with 'close'")

    # -- accessors ---------------------------------------------------------
    @property
    def close(self) -> pd.DataFrame:
        return self.fields["close"]

    @property
    def open(self) -> pd.DataFrame:
        return self.fields["open"]

    @property
    def high(self) -> pd.DataFrame:
        return self.fields["high"]

    @property
    def low(self) -> pd.DataFrame:
        return self.fields["low"]

    @property
    def volume(self) -> pd.DataFrame:
        return self.fields["volume"]

    @property
    def symbols(self) -> List[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def is_real(self) -> bool:
        return all(s in ("yfinance", "bundled") for s in self.sources.values())

    def __len__(self) -> int:
        return len(self.close)

    @property
    def shape(self) -> tuple:
        return self.close.shape

    # -- derived -----------------------------------------------------------
    def returns(self) -> pd.DataFrame:
        """Per-asset close-to-close returns, NaN where the asset was untradable."""
        rets = self.close.pct_change()
        return rets.where(self.tradable())

    def tradable(self) -> pd.DataFrame:
        """True where the asset had a price on this bar *and* the one before.

        A position can only be held over a bar whose return is defined, so this
        is the mask the engine uses to zero out impossible weights.
        """
        listed = self.close.notna()
        return listed & listed.shift(1, fill_value=False)

    def dollar_volume(self) -> pd.DataFrame:
        return (self.close * self.volume).where(self.close.notna())

    def first_valid(self) -> pd.Series:
        return self.close.apply(lambda col: col.first_valid_index())

    def last_valid(self) -> pd.Series:
        return self.close.apply(lambda col: col.last_valid_index())

    # -- survivorship ------------------------------------------------------
    def survivorship(self, tolerance_bars: int = 5) -> SurvivorshipReport:
        """Measure how many names survived to the end of the sample.

        A universe picked today and backfilled contains only survivors, and
        every backtest run on it is flattered by the companies that failed and
        were quietly excluded. We cannot fix that here, but we can refuse to
        hide it: if every single name is still trading at the end of a long
        sample, that is itself the evidence.
        """
        if not len(self):
            return SurvivorshipReport(0, 0, 0, [], [], 1.0, False, "Empty panel.")

        last = self.last_valid()
        first = self.first_valid()
        end = self.index[-1]
        start = self.index[0]
        cutoff = self.index[max(0, len(self) - 1 - tolerance_bars)]
        entry_cutoff = self.index[min(len(self) - 1, tolerance_bars)]

        delisted = sorted(str(s) for s in last.index[last < cutoff])
        late = sorted(str(s) for s in first.index[first > entry_cutoff])
        n = len(self.symbols)
        alive = n - len(delisted)
        rate = alive / n if n else 1.0

        years = len(self) / 252.0
        biased = rate >= 1.0 and years >= 3 and n >= 5
        if biased:
            note = (
                f"All {n} symbols were still trading at the end of a {years:.1f}-year sample. "
                "A universe with no failures in it was almost certainly chosen after the fact, "
                "which means these results exclude every name that went to zero. Treat the "
                "returns below as an upper bound."
            )
        elif n == 0:
            note = "Empty panel."
        else:
            note = (
                f"{len(delisted)} of {n} symbols stopped trading before the end of the sample "
                f"({rate:.0%} survived), so the universe is not made purely of winners."
            )

        return SurvivorshipReport(
            n_symbols=n,
            n_alive_at_end=alive,
            n_delisted=len(delisted),
            delisted_symbols=delisted[:25],
            late_starters=late[:25],
            survival_rate=float(rate),
            biased=bool(biased),
            note=note,
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        sources: Optional[Mapping[str, str]] = None,
        interval: str = "1d",
        note: str = "",
        min_bars: int = 2,
    ) -> "Panel":
        """Build a panel from ``{symbol: ohlcv_frame}``, aligning on the union index."""
        usable = {
            str(symbol): frame
            for symbol, frame in frames.items()
            if frame is not None and len(frame) >= min_bars
        }
        if not usable:
            raise ValueError("No symbol had enough data to build a panel")

        index = pd.DatetimeIndex([])
        for frame in usable.values():
            index = index.union(pd.DatetimeIndex(frame.index))
        index = index.sort_values()

        fields: Dict[str, pd.DataFrame] = {}
        for column in OHLCV_COLUMNS:
            fields[column] = pd.DataFrame(
                {
                    symbol: pd.to_numeric(frame[column], errors="coerce").reindex(index)
                    for symbol, frame in usable.items()
                },
                index=index,
            )

        return cls(
            fields=fields,
            sources=dict(sources or {s: "unknown" for s in usable}),
            interval=interval,
            note=note,
        )

    def select(self, symbols: Sequence[str]) -> "Panel":
        keep = [s for s in symbols if s in self.close.columns]
        if not keep:
            raise ValueError("None of the requested symbols are in this panel")
        return Panel(
            fields={name: frame.loc[:, keep] for name, frame in self.fields.items()},
            sources={s: self.sources.get(s, "unknown") for s in keep},
            interval=self.interval,
            note=self.note,
        )

    def slice(self, start=None, end=None) -> "Panel":
        return Panel(
            fields={name: frame.loc[start:end] for name, frame in self.fields.items()},
            sources=dict(self.sources),
            interval=self.interval,
            note=self.note,
        )


def load_panel(
    symbols: Iterable[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    source: str = "auto",
    min_bars: int = 120,
) -> Panel:
    """Load a panel for ``symbols``, skipping any that cannot supply enough history."""
    symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    if not symbols:
        raise ValueError("No symbols requested")

    frames: Dict[str, pd.DataFrame] = {}
    sources: Dict[str, str] = {}
    skipped: List[str] = []

    for symbol in dict.fromkeys(symbols):  # de-duplicate, keep order
        market = load_ohlcv(symbol, start, end, interval, source)
        if len(market.df) < min_bars:
            skipped.append(symbol)
            continue
        frames[symbol] = market.df
        sources[symbol] = market.source

    if not frames:
        raise ValueError(
            f"None of {len(symbols)} symbols returned at least {min_bars} bars."
        )

    simulated = sorted(s for s, src in sources.items() if src == "synthetic")
    note = ""
    if simulated:
        note = (
            f"{len(simulated)} of {len(frames)} symbols fell back to the market simulator "
            f"({', '.join(simulated[:6])}{'...' if len(simulated) > 6 else ''}). "
            "The statistics are still valid; they are measured on a simulated market."
        )
    if skipped:
        note = (note + " " if note else "") + f"Skipped for insufficient history: {', '.join(skipped[:6])}."

    return Panel.from_frames(frames, sources, interval=interval, note=note.strip())
