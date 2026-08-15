"""Probabilistic and Deflated Sharpe Ratios.

Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality".

The intuition: if you try 200 strategy variants, the best one will show a
handsome Sharpe *even when none of them has any edge*. The Deflated Sharpe
Ratio asks whether the winner beats what the luckiest of 200 coin-flippers
would have produced, and it charges extra for fat tails and negative skew --
exactly the return shapes that make naive Sharpe ratios flatter.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
]

_EULER = 0.5772156649015329


def _moments(returns: np.ndarray) -> tuple[float, float]:
    """Sample skew and *non-excess* kurtosis, as the PSR formula expects."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return 0.0, 3.0
    return float(stats.skew(arr, bias=False)), float(stats.kurtosis(arr, bias=False) + 3.0)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > ``benchmark``) given the observed Sharpe and its shape.

    ``sharpe`` and ``benchmark`` are per-observation (i.e. *not* annualised).
    """
    if n_obs < 3:
        return 0.5
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if denom <= 0:
        return 0.5
    z = (sharpe - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, variance_of_trials: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` *skill-free* strategies.

    This is the bar the winner has to clear to be interesting. It grows with
    the number of things you tried — which is why "I found a strategy with
    Sharpe 2" means nothing until you say how many you looked at.
    """
    n = max(int(n_trials), 1)
    if n == 1 or variance_of_trials <= 0:
        return 0.0
    sd = np.sqrt(variance_of_trials)
    # Bailey & López de Prado's Gumbel-based approximation.
    q1 = stats.norm.ppf(1.0 - 1.0 / n)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - _EULER) * q1 + _EULER * q2))


def deflated_sharpe_ratio(
    returns,
    sharpe_annual: float,
    periods_per_year: int,
    n_trials: int,
    trial_sharpes=None,
    variance_of_trials: float | None = None,
) -> dict:
    """Deflate an annualised Sharpe for selection bias and non-normality.

    Returns a dict with the PSR against a zero benchmark, the selection-bias
    threshold, the deflated probability, and the inputs used, so the UI can
    show its working rather than just a number.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_obs = arr.size
    sr_per_period = sharpe_annual / np.sqrt(periods_per_year)
    skew, kurt = _moments(arr)

    if variance_of_trials is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            trials = np.asarray(trial_sharpes, dtype=float) / np.sqrt(periods_per_year)
            trials = trials[np.isfinite(trials)]
            variance_of_trials = float(np.var(trials, ddof=1)) if trials.size > 1 else 0.0
        else:
            # With no trial cloud to measure, fall back to the asymptotic
            # variance of a skill-free Sharpe estimate.
            variance_of_trials = 1.0 / max(n_obs - 1, 1)

    threshold = expected_max_sharpe(n_trials, variance_of_trials)

    psr = probabilistic_sharpe_ratio(sr_per_period, n_obs, skew, kurt, 0.0)
    dsr = probabilistic_sharpe_ratio(sr_per_period, n_obs, skew, kurt, threshold)

    return {
        "psr": float(psr),
        "dsr": float(dsr),
        "sr_per_period": float(sr_per_period),
        "threshold_sr_per_period": float(threshold),
        "threshold_sr_annual": float(threshold * np.sqrt(periods_per_year)),
        "n_obs": int(n_obs),
        "n_trials": int(n_trials),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "variance_of_trials": float(variance_of_trials),
    }


def min_track_record_length(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed before the Sharpe is significant at ``confidence``.

    Inputs are per-observation. Returns ``inf`` when the edge is too small to
    ever clear the bar.
    """
    if sharpe <= benchmark:
        return float("inf")
    z = stats.norm.ppf(confidence)
    denom = (sharpe - benchmark) ** 2
    if denom <= 0:
        return float("inf")
    numer = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2
    if numer <= 0:
        return float("inf")
    return float(1.0 + numer * (z / (sharpe - benchmark)) ** 2)
