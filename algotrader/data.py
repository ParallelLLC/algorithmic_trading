"""Market data loading with a three-tier fallback.

Order of preference: live Yahoo download -> on-disk cache -> a deterministic
simulator. The fallback exists because a Hugging Face Space that shows a
stack trace on the first click is a Space nobody shares. When the simulator is
used, :class:`~algotrader.types.MarketData` says so and the UI shows it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .types import OHLCV_COLUMNS, MarketData

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("ALGOTRADER_CACHE", Path.home() / ".cache" / "algotrader"))
NETWORK_ENABLED = os.environ.get("ALGOTRADER_OFFLINE", "").lower() not in ("1", "true", "yes")

# Popular tickers get hand-set simulation parameters so the offline demo is at
# least in the right postcode: annual drift, annual vol, and a starting price.
@dataclass(frozen=True)
class SimProfile:
    drift: float
    vol: float
    price: float


SIM_PROFILES: dict[str, SimProfile] = {
    "AAPL": SimProfile(0.24, 0.29, 190.0),
    "MSFT": SimProfile(0.25, 0.27, 410.0),
    "NVDA": SimProfile(0.55, 0.52, 120.0),
    "TSLA": SimProfile(0.30, 0.58, 250.0),
    "AMZN": SimProfile(0.22, 0.33, 180.0),
    "GOOGL": SimProfile(0.20, 0.31, 170.0),
    "META": SimProfile(0.26, 0.40, 500.0),
    "SPY": SimProfile(0.10, 0.16, 550.0),
    "QQQ": SimProfile(0.14, 0.21, 480.0),
    "BTC-USD": SimProfile(0.45, 0.65, 65000.0),
    "ETH-USD": SimProfile(0.35, 0.75, 3000.0),
    "GLD": SimProfile(0.07, 0.14, 200.0),
    "TLT": SimProfile(0.01, 0.15, 95.0),
}

DEFAULT_UNIVERSE = ["SPY", "AAPL", "NVDA", "MSFT", "TSLA", "QQQ", "BTC-USD", "GLD"]


def _seed_for(symbol: str) -> int:
    """Stable per-symbol seed so a given ticker always simulates identically."""
    digest = hashlib.sha256(symbol.upper().encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any loader's output into a clean lowercase OHLCV frame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) for c in df.columns]
    df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    for col in missing:
        if col == "volume":
            df["volume"] = 0.0
        elif "close" in df.columns:
            df[col] = df["close"]
        else:
            raise ValueError(f"Price data is missing required column: {col}")
    df = df.loc[:, list(OHLCV_COLUMNS)].astype(float)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"] > 0].dropna(subset=["close"])
    return df


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.upper().replace("/", "_")
    return CACHE_DIR / f"{safe}_{interval}.csv"


def _read_cache(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, interval)
    if not path.exists():
        return None
    try:
        return _normalise(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception as exc:  # pragma: no cover - corrupted cache is not worth failing over
        logger.warning("Ignoring unreadable cache %s: %s", path, exc)
        return None


def _write_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_cache_path(symbol, interval))
    except Exception as exc:  # pragma: no cover - a read-only FS must not break the app
        logger.warning("Could not write cache for %s: %s", symbol, exc)


def _download(symbol: str, start: str, end: str | None, interval: str) -> Optional[pd.DataFrame]:
    if not NETWORK_ENABLED:
        return None
    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance not installed; using offline data")
        return None
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:
        logger.warning("Download failed for %s: %s", symbol, exc)
        return None
    if raw is None or len(raw) == 0:
        logger.warning("Download for %s returned no rows", symbol)
        return None
    try:
        return _normalise(raw)
    except Exception as exc:
        logger.warning("Could not normalise download for %s: %s", symbol, exc)
        return None


def simulate_ohlcv(
    symbol: str = "SIM",
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a deterministic but realistic-looking OHLCV series.

    This is not geometric Brownian motion with a straight face: it uses a
    two-state (calm / stressed) regime switch, Student-t innovations and
    GARCH-ish vol persistence, so the resulting series has fat tails and
    volatility clustering. That matters, because a strategy tested against
    naive GBM looks far better than it deserves to.
    """
    profile = SIM_PROFILES.get(symbol.upper(), SimProfile(0.08, 0.25, 100.0))
    rng = np.random.default_rng(_seed_for(symbol) if seed is None else seed)

    freq = {"1d": "B", "1wk": "W-FRI", "1h": "h"}.get(interval, "B")
    index = pd.date_range(start=start, end=end or pd.Timestamp.today().normalize(), freq=freq)
    n = len(index)
    if n < 50:
        raise ValueError("Simulated range is too short to backtest")

    ppy = 252 if freq in ("B", "h") else 52
    mu = profile.drift / ppy
    base_vol = profile.vol / np.sqrt(ppy)

    # Regime chain: calm state is sticky, stressed state is short and violent.
    p_calm_to_stress, p_stress_to_calm = 0.01, 0.06
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        flip = rng.random()
        if regime[i - 1] == 0:
            regime[i] = 1 if flip < p_calm_to_stress else 0
        else:
            regime[i] = 0 if flip < p_stress_to_calm else 1

    # Persistent vol around a regime-dependent level.
    vol = np.empty(n)
    level = np.where(regime == 1, base_vol * 2.4, base_vol * 0.9)
    vol[0] = level[0]
    for i in range(1, n):
        vol[i] = 0.92 * vol[i - 1] + 0.08 * level[i]

    shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # unit-ish variance, fat tails
    drift = np.where(regime == 1, mu - 3.0 * base_vol**2, mu)
    log_ret = drift + vol * shocks
    close = profile.price * np.exp(np.cumsum(log_ret))
    close = close * (profile.price / close[-1])  # end near the quoted level

    intrabar = vol * rng.uniform(0.3, 1.1, size=n)
    open_ = close * np.exp(-log_ret * rng.uniform(0.2, 0.8, size=n))
    high = np.maximum(open_, close) * np.exp(np.abs(intrabar))
    low = np.minimum(open_, close) * np.exp(-np.abs(intrabar))
    volume = rng.lognormal(mean=15.5, sigma=0.45, size=n) * (1.0 + 3.0 * regime)

    return _normalise(
        pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    )


def load_ohlcv(
    symbol: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    source: str = "yahoo",
) -> MarketData:
    """Load OHLCV for ``symbol``.

    Default ``source='yahoo'`` requires a Yahoo download. ``auto`` still falls
    back to cache then the simulator (Hugging Face Space). ``synthetic`` is tests only.
    """
    symbol = (symbol or "SPY").strip().upper()

    if source == "synthetic":
        df = simulate_ohlcv(symbol, start, end, interval)
        return MarketData(symbol, df, "synthetic", interval, "Simulated prices (requested).")

    if source in ("yahoo", "live", "auto"):
        df = _download(symbol, start, end, interval)
        if df is not None and len(df) > 50:
            _write_cache(symbol, interval, df)
            return MarketData(symbol, df, "yfinance", interval, "Live data from Yahoo Finance.")
        if source in ("yahoo", "live"):
            raise RuntimeError(
                f"Yahoo returned no usable bars for {symbol}. "
                "Check the ticker, date range, and network. "
                "Pass source='synthetic' only for offline tests."
            )

    cached = _read_cache(symbol, interval)
    if cached is not None and len(cached) > 50:
        window = cached.loc[str(start) : str(end)] if end else cached.loc[str(start) :]
        if len(window) > 50:
            return MarketData(symbol, window, "bundled", interval, "Cached data (network unavailable).")

    df = simulate_ohlcv(symbol, start, end, interval)
    return MarketData(
        symbol,
        df,
        "synthetic",
        interval,
        f"Live data for {symbol} was unavailable, so this run uses a deterministic "
        "market simulator with fat tails and volatility clustering. The statistics "
        "below are still valid — they are just measured on a simulated market.",
    )
