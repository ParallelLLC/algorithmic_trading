"""Vectorised technical indicators.

Every function takes and returns pandas objects aligned to the input index, and
every one of them is causal: the value at bar ``t`` uses only data up to and
including ``t``. That property is what makes the backtest engine's single
``shift`` enough to guarantee no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "donchian",
    "zscore",
    "roc",
    "realised_vol",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 leaves rs undefined: an all-gain window is RSI 100, and a
    # perfectly flat window (no gains either) is RSI 50.
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask(flat, 50.0)
    return out.where(avg_gain.notna())


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ``(macd_line, signal_line, histogram)``."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    series: pd.Series, window: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns ``(lower, middle, upper)``."""
    mid = sma(series, window)
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def donchian(df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Rolling channel excluding the current bar, so a breakout test is causal."""
    upper = df["high"].rolling(window, min_periods=window).max().shift(1)
    lower = df["low"].rolling(window, min_periods=window).min().shift(1)
    return lower, upper


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / sd.replace(0.0, np.nan)


def roc(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change(window)


def realised_vol(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)
