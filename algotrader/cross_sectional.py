"""Cross-sectional strategies.

Where a single-asset rule asks "should I be long this thing?", a
cross-sectional rule asks "which of these things should I be long, and which
short?". That difference matters for validation: a long-short book that ranks
names is exposed to entirely different failure modes than a timing rule, and it
needs its own null (see :mod:`algotrader.validation.cross_permutation`).

Each strategy is a pure function ``(panel, **params) -> T x N weights``, causal
by construction, with gross exposure of at most 1.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from .panel import Panel
from .strategies import ParamSpec

__all__ = [
    "CrossSectionalStrategy",
    "XS_REGISTRY",
    "get_xs_strategy",
    "list_xs_strategies",
    "scores_to_weights",
]

MIN_NAMES = 4  # below this, "cross-section" is not a meaningful word


def scores_to_weights(
    scores: pd.DataFrame,
    long_frac: float = 0.3,
    short_frac: float = 0.3,
    long_only: bool = False,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Turn a score matrix into a dollar-neutral (or long-only) weight matrix.

    Ranks within each date, takes the top and bottom fractions, and equal-weights
    each leg. Gross exposure is 1: half per leg when short-selling, all of it in
    the long leg otherwise.
    """
    valid = scores.notna()
    counts = valid.sum(axis=1)
    ranks = scores.rank(axis=1, pct=True, na_option="keep")

    longs = (ranks > 1.0 - long_frac) & valid
    n_long = longs.sum(axis=1).replace(0, np.nan)

    if long_only:
        weights = longs.astype(float).div(n_long, axis=0)
    else:
        shorts = (ranks <= short_frac) & valid
        n_short = shorts.sum(axis=1).replace(0, np.nan)
        weights = (
            longs.astype(float).div(n_long, axis=0) * 0.5
            - shorts.astype(float).div(n_short, axis=0) * 0.5
        )

    # A cross-section of two names is not a cross-section.
    weights = weights.where(counts >= min_names, 0.0)
    return weights.fillna(0.0)


def _rebalance_hold(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Refresh the target only every ``every`` bars, holding it in between."""
    if every <= 1:
        return weights
    out = weights.copy()
    keep = np.zeros(len(out), dtype=bool)
    keep[::every] = True
    out.iloc[~keep] = np.nan
    return out.ffill().fillna(0.0)


# --------------------------------------------------------------------------
# Strategy implementations
# --------------------------------------------------------------------------

def _xs_momentum(
    panel: Panel, lookback: int = 250, skip: int = 20, rebalance: int = 21, long_frac: float = 0.3
) -> pd.DataFrame:
    """Classic 12-1 momentum: rank on past return, skipping the most recent month.

    The skip is not decoration -- including the last month mixes in short-term
    reversal, which points the other way and muddies the signal.
    """
    close = panel.close
    scores = close.shift(skip) / close.shift(lookback) - 1.0
    return _rebalance_hold(scores_to_weights(scores, long_frac, long_frac), rebalance)


def _xs_reversal(
    panel: Panel, lookback: int = 5, rebalance: int = 5, long_frac: float = 0.3
) -> pd.DataFrame:
    """Short-term reversal: buy the recent losers, sell the recent winners."""
    scores = -(panel.close.pct_change(lookback))
    return _rebalance_hold(scores_to_weights(scores, long_frac, long_frac), rebalance)


def _low_volatility(
    panel: Panel, window: int = 60, rebalance: int = 21, long_frac: float = 0.3
) -> pd.DataFrame:
    """The low-volatility anomaly: long the calm names, short the wild ones."""
    scores = -(panel.close.pct_change().rolling(window, min_periods=window).std())
    return _rebalance_hold(scores_to_weights(scores, long_frac, long_frac), rebalance)


def _xs_value_proxy(
    panel: Panel, window: int = 250, rebalance: int = 21, long_frac: float = 0.3
) -> pd.DataFrame:
    """Distance below the long-run average price, as a crude cheapness proxy.

    This is not book-to-market -- there are no fundamentals in the panel -- so
    treat it as mean reversion over a long horizon rather than value investing.
    """
    close = panel.close
    scores = -(close / close.rolling(window, min_periods=window).mean() - 1.0)
    return _rebalance_hold(scores_to_weights(scores, long_frac, long_frac), rebalance)


def _equal_weight(panel: Panel, rebalance: int = 21) -> pd.DataFrame:
    """Own everything tradable, equally. The bar a stock picker has to clear."""
    listed = panel.close.notna()
    counts = listed.sum(axis=1).replace(0, np.nan)
    return _rebalance_hold(listed.astype(float).div(counts, axis=0).fillna(0.0), rebalance)


def _xs_random(panel: Panel, rebalance: int = 21, seed: int = 7) -> pd.DataFrame:
    """Random long-short book. The control group for cross-sectional claims."""
    rng = np.random.default_rng(int(seed))
    scores = pd.DataFrame(
        rng.standard_normal(panel.close.shape), index=panel.index, columns=panel.symbols
    ).where(panel.close.notna())
    return _rebalance_hold(scores_to_weights(scores), rebalance)


@dataclass(frozen=True)
class CrossSectionalStrategy:
    key: str
    name: str
    family: str
    description: str
    fn: Callable[..., pd.DataFrame]
    params: tuple = ()

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.cast(p.default) for p in self.params}

    def clean(self, params: Dict[str, float] | None) -> Dict[str, float]:
        merged = self.defaults()
        for spec in self.params:
            if params and spec.name in params and params[spec.name] is not None:
                merged[spec.name] = spec.cast(params[spec.name])
        return merged

    def generate(self, panel: Panel, params: Dict[str, float] | None = None) -> pd.DataFrame:
        weights = self.fn(panel, **self.clean(params))
        return (
            weights.reindex(index=panel.index, columns=panel.symbols)
            .astype(float)
            .fillna(0.0)
            .clip(-1.0, 1.0)
        )

    def grid(self, limit: int | None = None) -> List[Dict[str, float]]:
        if not self.params:
            return [{}]
        names = [p.name for p in self.params]
        combos = [dict(zip(names, v)) for v in itertools.product(*[p.grid for p in self.params])]
        combos = [c for c in combos if not ("skip" in c and "lookback" in c and c["skip"] >= c["lookback"])]
        if limit is not None and len(combos) > limit:
            step = len(combos) / limit
            combos = [combos[int(i * step)] for i in range(limit)]
        return combos


XS_REGISTRY: Dict[str, CrossSectionalStrategy] = {}


def _register(strategy: CrossSectionalStrategy) -> CrossSectionalStrategy:
    XS_REGISTRY[strategy.key] = strategy
    return strategy


_register(
    CrossSectionalStrategy(
        key="equal_weight",
        name="Equal Weight",
        family="benchmark",
        description="Own every name equally. The bar a stock picker has to clear.",
        fn=_equal_weight,
        params=(ParamSpec("rebalance", "Rebalance (bars)", 21, (5, 21, 63), "int", 1, 252, 1),),
    )
)

_register(
    CrossSectionalStrategy(
        key="xs_momentum",
        name="Cross-Sectional Momentum",
        family="momentum",
        description="Long the past winners, short the past losers, skipping the most recent month.",
        fn=_xs_momentum,
        params=(
            ParamSpec("lookback", "Lookback", 250, (60, 120, 250), "int", 20, 750, 1),
            ParamSpec("skip", "Skip recent", 20, (0, 5, 20), "int", 0, 60, 1),
            ParamSpec("rebalance", "Rebalance (bars)", 21, (5, 21, 63), "int", 1, 252, 1),
            ParamSpec("long_frac", "Leg size", 0.3, (0.1, 0.2, 0.3), "float", 0.05, 0.5, 0.05),
        ),
    )
)

_register(
    CrossSectionalStrategy(
        key="xs_reversal",
        name="Short-Term Reversal",
        family="mean-reversion",
        description="Buy this week's losers and sell its winners.",
        fn=_xs_reversal,
        params=(
            ParamSpec("lookback", "Lookback", 5, (1, 3, 5, 10, 21), "int", 1, 60, 1),
            ParamSpec("rebalance", "Rebalance (bars)", 5, (1, 5, 21), "int", 1, 252, 1),
            ParamSpec("long_frac", "Leg size", 0.3, (0.1, 0.2, 0.3), "float", 0.05, 0.5, 0.05),
        ),
    )
)

_register(
    CrossSectionalStrategy(
        key="low_volatility",
        name="Low Volatility",
        family="risk",
        description="Long the calm names, short the volatile ones.",
        fn=_low_volatility,
        params=(
            ParamSpec("window", "Vol window", 60, (20, 60, 120), "int", 5, 252, 1),
            ParamSpec("rebalance", "Rebalance (bars)", 21, (5, 21, 63), "int", 1, 252, 1),
            ParamSpec("long_frac", "Leg size", 0.3, (0.1, 0.2, 0.3), "float", 0.05, 0.5, 0.05),
        ),
    )
)

_register(
    CrossSectionalStrategy(
        key="xs_value_proxy",
        name="Long-Horizon Reversion",
        family="value-ish",
        description="Long names trading below their long-run average, short those above.",
        fn=_xs_value_proxy,
        params=(
            ParamSpec("window", "Window", 250, (120, 250, 500), "int", 30, 1000, 1),
            ParamSpec("rebalance", "Rebalance (bars)", 21, (5, 21, 63), "int", 1, 252, 1),
            ParamSpec("long_frac", "Leg size", 0.3, (0.1, 0.2, 0.3), "float", 0.05, 0.5, 0.05),
        ),
    )
)

_register(
    CrossSectionalStrategy(
        key="xs_random",
        name="Random Book (control)",
        family="control",
        description="Random long-short positions. Anything that cannot beat this is noise.",
        fn=_xs_random,
        params=(
            ParamSpec("rebalance", "Rebalance (bars)", 21, (5, 21, 63), "int", 1, 252, 1),
            ParamSpec("seed", "Seed", 7, (1, 7, 42, 123), "int", 0, 9999, 1),
        ),
    )
)


def get_xs_strategy(key: str) -> CrossSectionalStrategy:
    try:
        return XS_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown cross-sectional strategy '{key}'. Available: {', '.join(sorted(XS_REGISTRY))}"
        ) from None


def list_xs_strategies(exclude: Iterable[str] = ()) -> List[CrossSectionalStrategy]:
    skip = set(exclude)
    return [s for k, s in XS_REGISTRY.items() if k not in skip]
