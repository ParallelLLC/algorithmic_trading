"""The cross-sectional null.

For a timing rule, shuffling the price path is the right null. For a rule that
*ranks names*, it is the wrong test entirely: shuffling time destroys the
market's whole correlation structure, and the resulting null is so weak that
almost any long-short book clears it.

The question a cross-sectional strategy has to answer is narrower. Not "does
this market have structure?" but: **given these dates, these assets and this
book's shape, does the strategy put its weight on the right names?**

So we permute the *weights across assets within each date*. Every calendar
effect survives. Every correlation between names survives. The gross and net
exposure of the book on each date survives exactly. The one thing destroyed is
the link between the strategy's choice and the asset it chose.

A momentum book that beats this null is picking names. One that does not was
being paid for its market exposure, its sector tilt, or the calendar -- all of
which are available far more cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..panel import Panel
from ..portfolio import run_portfolio_backtest
from ..types import CostModel

__all__ = ["cross_sectional_permutation_test", "permute_within_dates", "CrossPermutationResult"]


@dataclass
class CrossPermutationResult:
    observed: float
    null: np.ndarray
    p_value: float
    n_permutations: int

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null)) if self.null.size else 0.0

    @property
    def percentile(self) -> float:
        if not self.null.size:
            return 50.0
        return float((self.null < self.observed).mean() * 100.0)


def permute_within_dates(
    weights: pd.DataFrame,
    investable: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reassign each date's weights among that date's investable assets.

    The multiset of weights on every row is preserved exactly -- so gross
    exposure, net exposure, leg sizes and position counts are all identical to
    the real book -- but which asset receives which weight is randomised.

    Vectorised across all dates at once: sorting each row puts the investable
    weights first and pushes non-investable slots to NaN, then a random rank per
    investable slot picks each weight exactly once.
    """
    values = weights.to_numpy(dtype=float, copy=True)
    mask = investable.to_numpy(dtype=bool)

    masked = np.where(mask, values, np.nan)
    # NaNs sort last, so the first k entries of each row are that row's real weights.
    ordered = np.sort(masked, axis=1)

    noise = np.where(mask, rng.random(values.shape), np.inf)
    # Double argsort turns random values into ranks 0..k-1 for investable slots,
    # and k..n-1 for the rest, which then index into the NaN tail.
    random_rank = np.argsort(np.argsort(noise, axis=1), axis=1)

    shuffled = np.take_along_axis(ordered, random_rank, axis=1)
    return pd.DataFrame(
        np.nan_to_num(shuffled, nan=0.0), index=weights.index, columns=weights.columns
    )


def cross_sectional_permutation_test(
    panel: Panel,
    weights: pd.DataFrame,
    n_permutations: int = 200,
    costs: Optional[CostModel] = None,
    lag: int = 1,
    gross_leverage: float = 1.0,
    max_weight: Optional[float] = None,
    allow_short: bool = True,
    rebalance_on: Optional[pd.Series] = None,
    seed: int = 0,
    observed: Optional[float] = None,
    neutralise_costs: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
) -> CrossPermutationResult:
    """Test whether a book's Sharpe survives randomising which names it picked.

    ``neutralise_costs`` defaults to True, and it matters more than it looks.
    A real momentum book holds many of the same names from one rebalance to the
    next, so it churns slowly. A shuffled book reassigns names at random every
    date, so it churns furiously and pays for it. Charging costs would penalise
    the null for turnover the strategy never had, and the strategy would look
    good by comparison for reasons that have nothing to do with skill.

    So this test asks only "did it pick the right names?" and leaves "can you
    afford to trade it?" to the cost stress test, which measures that directly.
    """
    costs = CostModel(0.0, 0.0, 0.0) if neutralise_costs else (costs or CostModel())
    rng = np.random.default_rng(seed)
    investable = panel.close.notna()

    def sharpe_of(w: pd.DataFrame) -> float:
        return run_portfolio_backtest(
            panel,
            w,
            costs=costs,
            lag=lag,
            gross_leverage=gross_leverage,
            max_weight=max_weight,
            allow_short=allow_short,
            rebalance_on=rebalance_on,
        ).sharpe

    if observed is None:
        observed = sharpe_of(weights)

    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = sharpe_of(permute_within_dates(weights, investable, rng))
        if progress is not None and (i % 10 == 0 or i == n_permutations - 1):
            progress((i + 1) / n_permutations, f"Cross-sectional shuffle {i + 1}/{n_permutations}")

    p_value = float((1 + np.sum(null >= observed)) / (n_permutations + 1))
    return CrossPermutationResult(
        observed=float(observed),
        null=null,
        p_value=p_value,
        n_permutations=n_permutations,
    )
