"""Monte-Carlo permutation test for trading rules.

The question this answers is not "did the strategy make money" but "would a
rule of this shape have made this much money on a market with no exploitable
structure?". We destroy the serial dependence in the price path while keeping
its distribution of moves intact, re-run the *same* strategy on each shuffled
market, and see where the real result lands in that null distribution.

A strategy whose Sharpe sits comfortably inside the null is not a strategy —
it is a lottery ticket that happened to win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..engine import bars_to_returns, run_backtest
from ..types import CostModel

__all__ = ["permutation_test", "PermutationResult", "permute_bars"]


@dataclass
class PermutationResult:
    observed: float
    null: np.ndarray
    p_value: float
    method: str
    n_permutations: int

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null)) if self.null.size else 0.0

    @property
    def percentile(self) -> float:
        """Where the observed Sharpe sits in the null distribution, 0-100."""
        if not self.null.size:
            return 50.0
        return float((self.null < self.observed).mean() * 100.0)


def _decompose(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Split bars into scale-free log moves that can be reshuffled safely."""
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    gap = np.log(open_[1:] / close[:-1])
    hi = np.log(np.maximum(high[1:], open_[1:]) / open_[1:])
    lo = np.log(np.minimum(low[1:], open_[1:]) / open_[1:])
    body = np.log(close[1:] / open_[1:])
    return np.column_stack([gap, hi, lo, body, volume[1:]]), float(close[0])


def _rebuild(parts: np.ndarray, anchor: float, index: pd.Index, first_row: pd.Series) -> pd.DataFrame:
    gap, hi, lo, body, volume = (parts[:, i] for i in range(5))
    n = parts.shape[0] + 1

    close = np.empty(n)
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    vol = np.empty(n)

    close[0] = anchor
    open_[0] = float(first_row["open"])
    high[0] = float(first_row["high"])
    low[0] = float(first_row["low"])
    vol[0] = float(first_row["volume"])

    # Cumulative product form: close[i] = close[0] * exp(cumsum(gap + body)).
    close[1:] = anchor * np.exp(np.cumsum(gap + body))
    open_[1:] = close[:-1] * np.exp(gap)
    high[1:] = open_[1:] * np.exp(hi)
    low[1:] = open_[1:] * np.exp(lo)
    vol[1:] = volume

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=index
    )


def permute_bars(
    df: pd.DataFrame,
    rng: np.random.Generator,
    method: str = "permute",
    block: int = 20,
) -> pd.DataFrame:
    """Return a shuffled market with the same index and bar anatomy.

    ``permute`` reshuffles individual bars, destroying all serial structure.
    ``block`` resamples contiguous blocks with replacement, which preserves
    short-horizon autocorrelation and volatility clustering — a harder null
    that trend strategies deserve to be tested against.
    """
    parts, anchor = _decompose(df)
    m = parts.shape[0]
    if m < 2:
        return df.copy()

    if method == "block":
        size = max(2, min(int(block), m))
        starts = rng.integers(0, m, size=int(np.ceil(m / size)))
        order = np.concatenate([(np.arange(s, s + size) % m) for s in starts])[:m]
    else:
        order = rng.permutation(m)

    return _rebuild(parts[order], anchor, df.index, df.iloc[0])


def permutation_test(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    n_permutations: int = 300,
    method: str = "permute",
    block: int = 20,
    costs: Optional[CostModel] = None,
    lag: int = 1,
    max_leverage: float = 1.0,
    allow_short: bool = True,
    seed: int = 0,
    observed: Optional[float] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> PermutationResult:
    """Run ``signal_fn`` against ``n_permutations`` shuffled markets.

    ``signal_fn`` must be the strategy's target-exposure generator; it is
    re-evaluated on every synthetic market, which is the whole point — a rule
    that only works because of the specific path it was tuned on will fall
    apart here.
    """
    costs = costs or CostModel()
    rng = np.random.default_rng(seed)

    def sharpe_on(frame: pd.DataFrame) -> float:
        target = signal_fn(frame)
        result = run_backtest(
            frame,
            target,
            costs=costs,
            lag=lag,
            max_leverage=max_leverage,
            allow_short=allow_short,
        )
        return result.sharpe

    if observed is None:
        observed = sharpe_on(df)

    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = sharpe_on(permute_bars(df, rng, method, block))
        if progress is not None and (i % 25 == 0 or i == n_permutations - 1):
            progress((i + 1) / n_permutations, f"Permutation {i + 1}/{n_permutations}")

    # +1 in both places: the observed result is itself one draw from the null
    # under H0, which keeps the test from ever reporting an impossible p = 0.
    p_value = float((1 + np.sum(null >= observed)) / (n_permutations + 1))

    return PermutationResult(
        observed=float(observed),
        null=null,
        p_value=p_value,
        method=method,
        n_permutations=n_permutations,
    )


def bootstrap_return_paths(returns: pd.Series, n: int = 500, seed: int = 0) -> np.ndarray:
    """Bootstrap terminal-wealth outcomes from a realised return stream.

    Useful for the "how wide is the cone of outcomes?" chart — the same edge
    can produce wildly different equity curves.
    """
    arr = np.asarray(returns.dropna(), dtype=float)
    if arr.size == 0:
        return np.zeros((n, 1))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n, arr.size), replace=True)
    return np.cumprod(1.0 + draws, axis=1)
