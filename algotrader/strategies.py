"""The strategy zoo.

Every strategy is a pure function ``(df, **params) -> target exposure series``
in ``[-1, 1]``, causal by construction. They never see costs, capital or
execution — that is the engine's job — which is what lets the same function be
re-run thousands of times inside the permutation and PBO machinery.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind

__all__ = ["Strategy", "ParamSpec", "REGISTRY", "get_strategy", "list_strategies"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    default: float
    grid: Sequence[float]
    kind: str = "int"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def cast(self, value):
        return int(value) if self.kind == "int" else float(value)


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    family: str
    description: str
    fn: Callable[..., pd.Series]
    params: tuple[ParamSpec, ...] = ()

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.cast(p.default) for p in self.params}

    def clean(self, params: Dict[str, float] | None) -> Dict[str, float]:
        """Fill in missing params and coerce types, ignoring unknown keys."""
        merged = self.defaults()
        for spec in self.params:
            if params and spec.name in params and params[spec.name] is not None:
                merged[spec.name] = spec.cast(params[spec.name])
        return merged

    def generate(self, df: pd.DataFrame, params: Dict[str, float] | None = None) -> pd.Series:
        target = self.fn(df, **self.clean(params))
        return target.reindex(df.index).astype(float).fillna(0.0).clip(-1.0, 1.0)

    def grid(self, limit: int | None = None) -> List[Dict[str, float]]:
        """Cartesian product of the per-parameter grids (the 'trials' a
        researcher would realistically run before picking a winner)."""
        if not self.params:
            return [{}]
        names = [p.name for p in self.params]
        combos = [
            dict(zip(names, values))
            for values in itertools.product(*[p.grid for p in self.params])
        ]
        combos = [c for c in combos if self._valid(c)]
        if limit is not None and len(combos) > limit:
            step = len(combos) / limit
            combos = [combos[int(i * step)] for i in range(limit)]
        return combos

    def _valid(self, combo: Dict[str, float]) -> bool:
        """Reject nonsensical combinations (a fast MA slower than the slow one)."""
        if "fast" in combo and "slow" in combo and combo["fast"] >= combo["slow"]:
            return False
        if "lower" in combo and "upper" in combo and combo["lower"] >= combo["upper"]:
            return False
        return True


def _hold_until_flip(raw: pd.Series) -> pd.Series:
    """Turn sparse entry/exit signals into a continuously held position."""
    return raw.ffill().fillna(0.0)


# --------------------------------------------------------------------------
# Strategy implementations
# --------------------------------------------------------------------------

def _buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def _sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    f, s = ind.sma(df["close"], fast), ind.sma(df["close"], slow)
    return pd.Series(np.where(f > s, 1.0, -1.0), index=df.index).where(s.notna())


def _ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 50) -> pd.Series:
    f, s = ind.ema(df["close"], fast), ind.ema(df["close"], slow)
    return pd.Series(np.where(f > s, 1.0, -1.0), index=df.index).where(s.notna())


def _macd_trend(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    _, _, hist = ind.macd(df["close"], fast, slow, signal)
    return pd.Series(np.sign(hist), index=df.index).where(hist.notna())


def _rsi_reversion(df: pd.DataFrame, window: int = 14, lower: int = 30, upper: int = 70) -> pd.Series:
    r = ind.rsi(df["close"], window)
    raw = pd.Series(np.nan, index=df.index)
    raw[r < lower] = 1.0
    raw[r > upper] = -1.0
    raw[(r > 45) & (r < 55)] = 0.0  # flatten in the middle of the range
    return _hold_until_flip(raw).where(r.notna())


def _bollinger_reversion(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.Series:
    low, mid, high = ind.bollinger(df["close"], window, k)
    close = df["close"]
    raw = pd.Series(np.nan, index=df.index)
    raw[close < low] = 1.0
    raw[close > high] = -1.0
    raw[(close - mid).abs() < 0.1 * (high - mid)] = 0.0
    return _hold_until_flip(raw).where(mid.notna())


def _donchian_breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    low, high = ind.donchian(df, window)
    raw = pd.Series(np.nan, index=df.index)
    raw[df["close"] > high] = 1.0
    raw[df["close"] < low] = -1.0
    return _hold_until_flip(raw).where(high.notna())


def _momentum(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    return np.sign(ind.roc(df["close"], lookback))


def _vol_target_momentum(
    df: pd.DataFrame, lookback: int = 60, vol_window: int = 20, target_vol: float = 15
) -> pd.Series:
    """Momentum sized inversely to recent volatility (targets ``target_vol`` %)."""
    signal = np.sign(ind.roc(df["close"], lookback))
    rv = ind.realised_vol(df["close"].pct_change(), vol_window)
    scale = (target_vol / 100.0) / rv.replace(0.0, np.nan)
    return (signal * scale.clip(upper=1.0)).where(rv.notna())


def _channel_trend(df: pd.DataFrame, window: int = 50, atr_window: int = 14, mult: float = 1.0) -> pd.Series:
    """Long above an ATR band around the mean, short below it, flat inside."""
    mid = ind.sma(df["close"], window)
    band = ind.atr(df, atr_window) * mult
    raw = pd.Series(np.nan, index=df.index)
    raw[df["close"] > mid + band] = 1.0
    raw[df["close"] < mid - band] = -1.0
    raw[(df["close"] - mid).abs() < 0.25 * band] = 0.0
    return _hold_until_flip(raw).where(mid.notna() & band.notna())


def _coin_flip(df: pd.DataFrame, hold: int = 5, seed: int = 7) -> pd.Series:
    """A deliberately worthless strategy: the control group.

    If your clever rule cannot beat this on the validation panel, that is the
    single most useful thing this app can tell you.
    """
    rng = np.random.default_rng(int(seed))
    n = len(df)
    draws = rng.choice([-1.0, 1.0], size=int(np.ceil(n / max(hold, 1))))
    return pd.Series(np.repeat(draws, max(hold, 1))[:n], index=df.index)


REGISTRY: Dict[str, Strategy] = {}


def _register(strategy: Strategy) -> Strategy:
    REGISTRY[strategy.key] = strategy
    return strategy


_register(
    Strategy(
        key="buy_and_hold",
        name="Buy & Hold",
        family="benchmark",
        description="Own the asset, do nothing. The bar every other strategy has to clear.",
        fn=_buy_and_hold,
    )
)

_register(
    Strategy(
        key="sma_cross",
        name="SMA Crossover",
        family="trend",
        description="Long when the fast simple moving average is above the slow one, short when below.",
        fn=_sma_cross,
        params=(
            ParamSpec("fast", "Fast MA", 20, (5, 10, 20, 30, 50), "int", 2, 100, 1),
            ParamSpec("slow", "Slow MA", 100, (50, 100, 150, 200), "int", 10, 300, 5),
        ),
    )
)

_register(
    Strategy(
        key="ema_cross",
        name="EMA Crossover",
        family="trend",
        description="Same idea as the SMA cross but with exponential averages, so it turns faster.",
        fn=_ema_cross,
        params=(
            ParamSpec("fast", "Fast EMA", 12, (5, 8, 12, 21, 34), "int", 2, 100, 1),
            ParamSpec("slow", "Slow EMA", 50, (34, 50, 89, 144, 200), "int", 10, 300, 1),
        ),
    )
)

_register(
    Strategy(
        key="macd_trend",
        name="MACD Trend",
        family="trend",
        description="Follow the sign of the MACD histogram.",
        fn=_macd_trend,
        params=(
            ParamSpec("fast", "Fast", 12, (8, 12, 16), "int", 2, 60, 1),
            ParamSpec("slow", "Slow", 26, (21, 26, 34, 50), "int", 10, 200, 1),
            ParamSpec("signal", "Signal", 9, (5, 9, 13), "int", 2, 50, 1),
        ),
    )
)

_register(
    Strategy(
        key="rsi_reversion",
        name="RSI Mean Reversion",
        family="mean-reversion",
        description="Buy oversold, sell overbought, flatten in the middle of the range.",
        fn=_rsi_reversion,
        params=(
            ParamSpec("window", "RSI window", 14, (7, 14, 21), "int", 2, 60, 1),
            ParamSpec("lower", "Oversold", 30, (20, 25, 30, 35), "int", 5, 49, 1),
            ParamSpec("upper", "Overbought", 70, (65, 70, 75, 80), "int", 51, 95, 1),
        ),
    )
)

_register(
    Strategy(
        key="bollinger_reversion",
        name="Bollinger Reversion",
        family="mean-reversion",
        description="Fade moves outside the Bollinger bands and exit back at the middle band.",
        fn=_bollinger_reversion,
        params=(
            ParamSpec("window", "Window", 20, (10, 20, 30, 50), "int", 5, 120, 1),
            ParamSpec("k", "Band width (σ)", 2.0, (1.5, 2.0, 2.5, 3.0), "float", 0.5, 4.0, 0.1),
        ),
    )
)

_register(
    Strategy(
        key="donchian_breakout",
        name="Donchian Breakout",
        family="breakout",
        description="The classic turtle rule: buy new highs, sell new lows.",
        fn=_donchian_breakout,
        params=(ParamSpec("window", "Channel", 20, (10, 20, 40, 55, 100), "int", 5, 250, 1),),
    )
)

_register(
    Strategy(
        key="momentum",
        name="Time-Series Momentum",
        family="momentum",
        description="Hold long if the asset is up over the lookback, short if it is down.",
        fn=_momentum,
        params=(ParamSpec("lookback", "Lookback", 60, (5, 10, 20, 60, 120, 250), "int", 2, 500, 1),),
    )
)

_register(
    Strategy(
        key="vol_target_momentum",
        name="Vol-Targeted Momentum",
        family="momentum",
        description="Momentum sized down when markets get volatile, so risk stays roughly constant.",
        fn=_vol_target_momentum,
        params=(
            ParamSpec("lookback", "Lookback", 60, (20, 60, 120, 250), "int", 5, 500, 1),
            ParamSpec("vol_window", "Vol window", 20, (10, 20, 60), "int", 5, 120, 1),
            ParamSpec("target_vol", "Target vol %", 15, (10, 15, 20), "float", 2, 60, 1),
        ),
    )
)

_register(
    Strategy(
        key="channel_trend",
        name="ATR Channel Trend",
        family="trend",
        description="Trade with the trend only once price clears an ATR band around its mean.",
        fn=_channel_trend,
        params=(
            ParamSpec("window", "Mean window", 50, (20, 50, 100, 200), "int", 5, 300, 1),
            ParamSpec("atr_window", "ATR window", 14, (7, 14, 28), "int", 2, 60, 1),
            ParamSpec("mult", "ATR multiple", 1.0, (0.5, 1.0, 1.5, 2.0), "float", 0.1, 5.0, 0.1),
        ),
    )
)

_register(
    Strategy(
        key="coin_flip",
        name="Coin Flip (control)",
        family="control",
        description="Random positions. The control group — anything that cannot beat this is noise.",
        fn=_coin_flip,
        params=(
            ParamSpec("hold", "Bars per flip", 5, (1, 5, 10, 20), "int", 1, 60, 1),
            ParamSpec("seed", "Seed", 7, (1, 7, 42, 123), "int", 0, 9999, 1),
        ),
    )
)


def get_strategy(key: str) -> Strategy:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"Unknown strategy '{key}'. Available: {', '.join(sorted(REGISTRY))}") from None


def list_strategies(exclude: Iterable[str] = ()) -> List[Strategy]:
    skip = set(exclude)
    return [s for k, s in REGISTRY.items() if k not in skip]
