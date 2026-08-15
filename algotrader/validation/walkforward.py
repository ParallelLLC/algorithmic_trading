"""Walk-forward analysis.

Re-tune on a training window, trade the next window blind, roll forward. The
gap between in-sample and out-of-sample Sharpe is the honest estimate of how
much of the backtest was curve-fitting.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..engine import run_backtest
from ..strategies import Strategy
from ..types import CostModel

__all__ = ["walk_forward"]


def walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    costs: Optional[CostModel] = None,
    lag: int = 1,
    max_leverage: float = 1.0,
    allow_short: bool = True,
    grid_limit: int = 40,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """Roll a train/test split forward ``n_folds`` times.

    Each fold picks the best parameters by in-sample Sharpe and reports what
    those parameters then did out of sample. The stitched OOS returns are the
    closest thing to a paper-trading record this repo can produce offline.
    """
    costs = costs or CostModel()
    grid = strategy.grid(limit=grid_limit)
    n = len(df)

    if n < 250 or n_folds < 2:
        return {"folds": [], "note": "Not enough history for walk-forward analysis."}

    # Each fold is a contiguous train+test block; blocks advance by test length.
    block = int(n / (1 + (n_folds - 1) * (1 - train_ratio)))
    block = min(block, n)
    train_len = int(block * train_ratio)
    test_len = block - train_len
    if train_len < 100 or test_len < 20:
        return {"folds": [], "note": "Not enough history for walk-forward analysis."}

    folds: List[Dict[str, object]] = []
    oos_returns: List[pd.Series] = []

    for k in range(n_folds):
        start = k * test_len
        train = df.iloc[start : start + train_len]
        test = df.iloc[start + train_len : start + train_len + test_len]
        if len(test) < 20:
            break

        best_params, best_sharpe = None, -np.inf
        for params in grid:
            target = strategy.generate(train, params)
            sr = run_backtest(
                train, target, costs=costs, lag=lag,
                max_leverage=max_leverage, allow_short=allow_short,
            ).sharpe
            if sr > best_sharpe:
                best_params, best_sharpe = params, sr

        # Generate signals on train+test so indicators are warm at the fold
        # boundary, then evaluate only the test slice.
        combined = df.iloc[start : start + train_len + len(test)]
        target = strategy.generate(combined, best_params).loc[test.index]
        oos = run_backtest(
            test, target, costs=costs, lag=lag,
            max_leverage=max_leverage, allow_short=allow_short,
        )

        folds.append(
            {
                "fold": k + 1,
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "params": best_params,
                "is_sharpe": float(best_sharpe),
                "oos_sharpe": float(oos.sharpe),
                "oos_return": float(oos.metrics.get("total_return", 0.0)),
                "oos_max_dd": float(oos.metrics.get("max_drawdown", 0.0)),
            }
        )
        oos_returns.append(oos.returns)

        if progress is not None:
            progress((k + 1) / n_folds, f"Walk-forward fold {k + 1}/{n_folds}")

    if not folds:
        return {"folds": [], "note": "Not enough history for walk-forward analysis."}

    is_sharpes = np.array([f["is_sharpe"] for f in folds], dtype=float)
    oos_sharpes = np.array([f["oos_sharpe"] for f in folds], dtype=float)
    stitched = pd.concat(oos_returns) if oos_returns else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()

    mean_is = float(np.mean(is_sharpes))
    mean_oos = float(np.mean(oos_sharpes))

    return {
        "folds": folds,
        "mean_is_sharpe": mean_is,
        "mean_oos_sharpe": mean_oos,
        # 1.0 = the edge fully survived; 0.0 = it evaporated out of sample.
        "efficiency": float(mean_oos / mean_is) if mean_is > 1e-9 else 0.0,
        "oos_win_rate": float(np.mean(oos_sharpes > 0)),
        # 1.0 means every fold chose different parameters -- a tuning process
        # that cannot make up its mind is fitting noise.
        "param_instability": float(
            len({str(f["params"]) for f in folds}) / max(len(folds), 1)
        ),
        "oos_returns": stitched,
        "oos_equity": (1.0 + stitched).cumprod() if len(stitched) else stitched,
        "note": "",
    }
