"""Probability of Backtest Overfitting via Combinatorially Symmetric CV.

Bailey, Borwein, López de Prado & Zhu (2016), "The Probability of Backtest
Overfitting".

Take the N parameter variants you tried, cut the timeline into S chunks, and
for every way of splitting those chunks half-and-half: pick the variant that
won in-sample, then look up where it ranked out-of-sample. If your selection
process has skill, the in-sample winner should keep winning. If it is fitting
noise, the winner lands in the bottom half about as often as not — and PBO
approaches 0.5.
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np

__all__ = ["probability_of_backtest_overfitting"]


def _sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (per-period, unannualised — ranks are all we need)."""
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[1])
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    # Absolute floor, not `> 0`: a flat column's std is float noise, and
    # dividing by it would hand a do-nothing variant an enormous rank.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 1e-12, mean / sd, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 8,
    labels: Sequence[str] | None = None,
    max_combinations: int = 200,
) -> Dict[str, object]:
    """Compute PBO for a ``T x N`` matrix of per-period strategy returns.

    ``n_splits`` must be even. Returns PBO, the logit distribution, the
    in-sample/out-of-sample Sharpe pairs for the selected variants, and the
    rate at which the selected variant actually loses money out of sample.
    """
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (time x strategy)")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    t_obs, n_strats = matrix.shape

    if n_strats < 2 or t_obs < 2 * n_splits:
        return {
            "pbo": float("nan"),
            "n_strategies": int(n_strats),
            "n_combinations": 0,
            "logits": np.array([]),
            "is_sharpes": np.array([]),
            "oos_sharpes": np.array([]),
            "prob_oos_loss": float("nan"),
            "performance_degradation": float("nan"),
            "note": "Not enough variants or observations to estimate PBO.",
        }

    if n_splits % 2:
        n_splits += 1
    chunks = np.array_split(np.arange(t_obs), n_splits)

    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combinations:
        step = len(combos) / max_combinations
        combos = [combos[int(i * step)] for i in range(max_combinations)]

    logits, is_sr, oos_sr, chosen = [], [], [], []
    for combo in combos:
        is_idx = np.concatenate([chunks[c] for c in combo])
        oos_idx = np.concatenate([chunks[c] for c in range(n_splits) if c not in combo])

        is_perf = _sharpe_columns(matrix[is_idx])
        oos_perf = _sharpe_columns(matrix[oos_idx])

        best = int(np.argmax(is_perf))
        chosen.append(best)
        is_sr.append(float(is_perf[best]))
        oos_sr.append(float(oos_perf[best]))

        # Relative rank of the chosen variant in the OOS ranking, in (0, 1).
        rank = float(np.sum(oos_perf <= oos_perf[best]))
        omega = rank / (n_strats + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(omega / (1.0 - omega))))

    logits_arr = np.asarray(logits)
    is_arr, oos_arr = np.asarray(is_sr), np.asarray(oos_sr)

    # Slope of OOS on IS: negative means better in-sample fits do *worse* live.
    degradation = float("nan")
    if is_arr.size > 2 and np.std(is_arr) > 1e-12:
        degradation = float(np.polyfit(is_arr, oos_arr, 1)[0])

    counts = np.bincount(chosen, minlength=n_strats)
    most_selected = int(np.argmax(counts))

    return {
        "pbo": float(np.mean(logits_arr <= 0.0)),
        "n_strategies": int(n_strats),
        "n_combinations": int(len(combos)),
        "logits": logits_arr,
        "is_sharpes": is_arr,
        "oos_sharpes": oos_arr,
        "prob_oos_loss": float(np.mean(oos_arr <= 0.0)),
        "performance_degradation": degradation,
        "most_selected_index": most_selected,
        "most_selected_label": (labels[most_selected] if labels is not None else str(most_selected)),
        "selection_stability": float(counts[most_selected] / max(len(combos), 1)),
        "note": "",
    }
