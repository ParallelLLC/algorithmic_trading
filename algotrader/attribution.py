"""Style attribution: is this alpha, or is it beta you could have bought cheaply?

The most common way a strategy is oversold is not fraud or overfitting — it is
that the "edge" is a well-known risk premium wearing a new name. A book that
loads on market beta, or on momentum, or on low-volatility, will produce a
respectable Sharpe and an exciting story, and you can buy the same exposure in
an ETF for a few basis points.

So we regress the strategy's returns on style factors built from the panel
itself and ask what is left over. If the intercept is not distinguishable from
zero, the strategy has no alpha — however good its Sharpe looked.

Factors are constructed from the universe under test rather than downloaded,
which keeps this offline and self-consistent. The tradeoff is that they are
proxies: without fundamentals there is no true value or size factor, so those
are labelled honestly as what they are.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .panel import Panel
from .portfolio import run_portfolio_backtest
from .types import CostModel

__all__ = ["build_style_factors", "factor_attribution"]

_FREE = CostModel(0.0, 0.0, 0.0)  # factors are theoretical portfolios


def _long_short(panel: Panel, scores: pd.DataFrame, rebalance: int = 21) -> pd.Series:
    from .cross_sectional import _rebalance_hold, scores_to_weights

    weights = _rebalance_hold(scores_to_weights(scores), rebalance)
    return run_portfolio_backtest(panel, weights, costs=_FREE).returns


def build_style_factors(panel: Panel, rebalance: int = 21) -> pd.DataFrame:
    """Construct market and style factor returns from the panel itself."""
    close = panel.close
    listed = close.notna()
    counts = listed.sum(axis=1).replace(0, np.nan)

    equal = listed.astype(float).div(counts, axis=0).fillna(0.0)
    market = run_portfolio_backtest(panel, equal, costs=_FREE).returns

    factors = {
        "market": market,
        "momentum": _long_short(panel, close.shift(20) / close.shift(250) - 1.0, rebalance),
        "low_vol": _long_short(panel, -close.pct_change().rolling(60, min_periods=60).std(), rebalance),
        "reversal": _long_short(panel, -close.pct_change(5), rebalance),
        # Dollar volume is a liquidity proxy, not market cap. Named accordingly.
        "liquidity": _long_short(panel, -np.log(panel.dollar_volume().replace(0, np.nan)), rebalance),
    }
    return pd.DataFrame(factors).reindex(panel.index).fillna(0.0)


def _white_standard_errors(x: np.ndarray, residuals: np.ndarray, xtx_inv: np.ndarray) -> np.ndarray:
    """Heteroskedasticity-robust (White) standard errors.

    Return series are famously heteroskedastic — volatility clusters — and
    classical standard errors would overstate the significance of alpha.
    """
    meat = x.T @ (x * (residuals**2)[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def factor_attribution(
    returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: int = 252,
    alpha_t_threshold: float = 2.0,
) -> Dict[str, object]:
    """Regress strategy returns on style factors; report annualised alpha and betas."""
    aligned = pd.concat([returns.rename("strategy"), factors], axis=1).dropna()
    if len(aligned) < 60 or factors.shape[1] == 0:
        return {"available": False, "note": "Not enough overlapping observations for attribution."}

    y = aligned["strategy"].to_numpy(dtype=float)
    names = list(factors.columns)
    x = np.column_stack([np.ones(len(aligned))] + [aligned[c].to_numpy(dtype=float) for c in names])

    # Drop factors that are constant or collinear; a singular fit is worse than
    # a smaller one.
    keep = [0] + [i + 1 for i, c in enumerate(names) if aligned[c].std() > 1e-12]
    x = x[:, keep]
    names = [names[i - 1] for i in keep[1:]]

    try:
        xtx_inv = np.linalg.pinv(x.T @ x)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv rarely fails
        return {"available": False, "note": "Factor matrix is singular."}

    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    residuals = y - fitted
    errors = _white_standard_errors(x, residuals, xtx_inv)
    t_stats = np.divide(beta, errors, out=np.zeros_like(beta), where=errors > 0)

    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    alpha_period = float(beta[0])
    alpha_annual = alpha_period * periods_per_year
    alpha_t = float(t_stats[0])
    significant = bool(abs(alpha_t) >= alpha_t_threshold and alpha_annual > 0)

    betas = {name: float(b) for name, b in zip(names, beta[1:])}
    beta_ts = {name: float(t) for name, t in zip(names, t_stats[1:])}
    dominant = max(betas, key=lambda k: abs(betas[k])) if betas else None

    if significant:
        note = (
            f"Alpha of {alpha_annual:.1%} a year survives the style regression "
            f"(t = {alpha_t:.1f}). Something here is not explained by market, momentum, "
            "low-volatility, reversal or liquidity exposure."
        )
    else:
        explained = (
            f" Most of the variation is {dominant} exposure (beta {betas[dominant]:.2f})."
            if dominant
            else ""
        )
        note = (
            f"Alpha is {alpha_annual:.1%} a year with t = {alpha_t:.1f}, which is not "
            f"distinguishable from zero. The style factors explain {r_squared:.0%} of the "
            f"returns.{explained} You can buy that exposure far more cheaply than by "
            "running this strategy."
        )

    return {
        "available": True,
        "alpha_annual": alpha_annual,
        "alpha_t_stat": alpha_t,
        "alpha_significant": significant,
        "betas": betas,
        "beta_t_stats": beta_ts,
        "r_squared": float(r_squared),
        "dominant_factor": dominant,
        "n_obs": int(len(aligned)),
        "note": note,
    }
