"""Performance and risk metrics.

All ratios are computed from *net* per-bar returns and annualised with the
periodicity inferred from the index, so daily / hourly / minute series all get
comparable numbers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "infer_periods_per_year",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "compute_metrics",
]

_SECONDS_PER_YEAR = 365.25 * 24 * 3600
_TRADING_DAYS = 252

# A return stream with dispersion below this is constant to floating-point
# noise. Without an absolute floor, a flat series divides by ~1e-19 and reports
# a Sharpe of 1e16 -- the exact kind of nonsense number this project exists to
# catch, so it must not originate here.
_DEGENERATE_SD = 1e-12


def infer_periods_per_year(index: pd.Index) -> int:
    """Guess bars-per-year from an index, defaulting to daily trading bars."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return _TRADING_DAYS
    nanos = index.to_numpy(dtype="datetime64[ns]").astype("int64")
    deltas = np.diff(nanos) / 1e9  # seconds
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return _TRADING_DAYS
    step = float(np.median(deltas))
    if step >= 20 * 3600:  # daily or slower -> use trading-day convention
        days = step / 86400.0
        return max(1, int(round(_TRADING_DAYS / max(days / 1.4, 1.0))))
    # Intraday: assume a 6.5h session, 252 days a year.
    bars_per_session = (6.5 * 3600) / step
    return max(1, int(round(bars_per_session * _TRADING_DAYS)))


def _clean(returns: pd.Series) -> np.ndarray:
    arr = np.asarray(returns, dtype=float)
    return arr[np.isfinite(arr)]


def sharpe_ratio(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    """Annualised Sharpe. ``rf`` is an annual risk-free rate."""
    arr = _clean(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - rf / periods_per_year
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < _DEGENERATE_SD:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int, rf: float = 0.0) -> float:
    arr = _clean(returns)
    if arr.size < 2:
        return 0.0
    excess = arr - rf / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = np.sqrt(np.mean(downside**2))
    if not np.isfinite(dd) or dd < _DEGENERATE_SD:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(drawdown_series(equity).min())


def _time_under_water(equity: pd.Series, periods_per_year: int) -> float:
    """Longest stretch below a prior peak, in years."""
    if equity.empty:
        return 0.0
    dd = drawdown_series(equity).to_numpy()
    longest = current = 0
    for value in dd:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest / periods_per_year


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    position: pd.Series | None = None,
    periods_per_year: int | None = None,
    rf: float = 0.0,
) -> Dict[str, float]:
    """Full metric bundle for one equity curve."""
    ppy = periods_per_year or infer_periods_per_year(returns.index)
    arr = _clean(returns)
    n = arr.size
    if n == 0 or equity.empty:
        return {"periods_per_year": float(ppy)}

    years = n / ppy
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(arr.std(ddof=1) * np.sqrt(ppy))
    mdd = max_drawdown(equity)
    sr = sharpe_ratio(returns, ppy, rf)

    out: Dict[str, float] = {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sr,
        "sortino": sortino_ratio(returns, ppy, rf),
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "max_drawdown": mdd,
        "time_under_water_yrs": _time_under_water(equity, ppy),
        "hit_rate": float((arr > 0).mean()),
        "skew": float(pd.Series(arr).skew()) if n > 2 else 0.0,
        "kurtosis": float(pd.Series(arr).kurtosis()) if n > 3 else 0.0,
        "var_95": float(np.percentile(arr, 5)),
        "cvar_95": float(arr[arr <= np.percentile(arr, 5)].mean()) if n > 20 else 0.0,
        "best_bar": float(arr.max()),
        "worst_bar": float(arr.min()),
        "n_bars": float(n),
        "years": float(years),
        "periods_per_year": float(ppy),
    }

    if position is not None and not position.empty:
        pos = position.fillna(0.0)
        turnover = pos.diff().abs().fillna(pos.abs().iloc[0] if len(pos) else 0.0)
        out["exposure"] = float(pos.abs().mean())
        out["long_share"] = float((pos > 0).mean())
        out["short_share"] = float((pos < 0).mean())
        out["turnover_ann"] = float(turnover.sum() / years) if years > 0 else 0.0
        # A "trade" is any change in sign or size of exposure.
        out["n_trades"] = float((turnover > 1e-9).sum())
    return out
