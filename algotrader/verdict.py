"""Turn a pile of statistics into one number and one sentence.

The Reality Score is deliberately harsh. Most published backtests would score
below 40, and that is the point: the score exists to be screenshotted.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

__all__ = ["reality_score", "GRADES"]

GRADES = [
    (85, "A", "Survives everything we threw at it"),
    (70, "B", "Probably a real edge, with caveats"),
    (55, "C", "Ambiguous — could go either way"),
    (40, "D", "Mostly luck"),
    (0, "F", "Indistinguishable from randomness"),
]

WEIGHTS = {
    "significance": 0.30,
    "selection": 0.25,
    "walk_forward": 0.20,
    "overfitting": 0.15,
    "robustness": 0.10,
}


def _ramp(value: float, good: float, bad: float) -> float:
    """Linear 0-100 score where ``good`` maps to 100 and ``bad`` maps to 0."""
    if not np.isfinite(value):
        return 50.0
    if good == bad:
        return 50.0
    scaled = (value - bad) / (good - bad)
    return float(np.clip(scaled, 0.0, 1.0) * 100.0)


def reality_score(
    metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
    p_value: Optional[float] = None,
    dsr: Optional[float] = None,
    pbo: Optional[float] = None,
    wf_efficiency: Optional[float] = None,
    wf_win_rate: Optional[float] = None,
    cost_stress_ratio: Optional[float] = None,
    benchmark_correlation: Optional[float] = None,
) -> Dict[str, object]:
    """Combine the validation panel into a 0-100 score, a grade and warnings."""
    components: Dict[str, float] = {}

    components["significance"] = _ramp(p_value, good=0.01, bad=0.50) if p_value is not None else 50.0
    components["selection"] = float(np.clip(dsr, 0.0, 1.0) * 100.0) if dsr is not None else 50.0

    if wf_efficiency is not None:
        wf = _ramp(wf_efficiency, good=0.8, bad=-0.2)
        if wf_win_rate is not None:
            wf = 0.7 * wf + 0.3 * float(np.clip(wf_win_rate, 0.0, 1.0) * 100.0)
        components["walk_forward"] = wf
    else:
        components["walk_forward"] = 50.0

    components["overfitting"] = _ramp(pbo, good=0.05, bad=0.50) if pbo is not None and np.isfinite(pbo) else 50.0
    components["robustness"] = (
        _ramp(cost_stress_ratio, good=0.8, bad=0.0) if cost_stress_ratio is not None else 50.0
    )

    score = float(sum(components[k] * w for k, w in WEIGHTS.items()))

    flags: List[str] = []
    n_trades = int(metrics.get("n_trades", 0))
    sharpe = float(metrics.get("sharpe", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    bench_sharpe = float(benchmark_metrics.get("sharpe", 0.0))

    # The score asks "is the measured edge real?" — if the strategy lost money,
    # there is no edge to validate, whatever the statistics say about it.
    if total_return <= 0:
        flags.append(
            f"The strategy lost money over the test period ({total_return:.1%}). "
            "There is no edge here to validate."
        )
        score = min(score, 50.0)
    if total_return <= 0 < sharpe:
        flags.append(
            "Positive Sharpe with a negative total return: the average bar was profitable but "
            "the compounding was not. Volatility drag ate the arithmetic edge."
        )
    if max_dd < -0.5:
        flags.append(
            f"Peak-to-trough drawdown of {max_dd:.0%} — an account running this would have been "
            "closed long before the recovery arrived."
        )
        score = min(score, 60.0)

    if n_trades < 20:
        flags.append(
            f"Only {n_trades} position changes — with this few decisions, "
            "the result is a handful of coin flips, not a track record."
        )
        score = min(score, 55.0)
    if p_value is not None and p_value > 0.10:
        flags.append(
            f"Permutation p-value is {p_value:.2f}: roughly {p_value * 100:.0f}% of shuffled, "
            "structure-free markets did this well or better."
        )
    if dsr is not None and dsr < 0.5:
        flags.append(
            f"Deflated Sharpe is {dsr:.2f} — once you account for how many variants were tried, "
            "the edge does not clear the selection-bias bar."
        )
    if pbo is not None and np.isfinite(pbo) and pbo > 0.3:
        flags.append(
            f"Probability of backtest overfitting is {pbo:.0%}: the in-sample winner "
            "usually lands in the bottom half out of sample."
        )
    if wf_efficiency is not None and wf_efficiency < 0.3:
        flags.append(
            f"Walk-forward efficiency is {wf_efficiency:.0%} — most of the in-sample Sharpe "
            "does not survive re-tuning and trading forward."
        )
    if cost_stress_ratio is not None and cost_stress_ratio < 0.5:
        flags.append(
            "Tripling trading costs removes more than half the Sharpe. The edge is "
            "smaller than the friction it has to pay."
        )
    if benchmark_correlation is not None and benchmark_correlation > 0.95:
        flags.append(
            f"Returns are {benchmark_correlation:.0%} correlated with buy & hold — "
            "this is mostly a repackaged long position."
        )
    if sharpe < bench_sharpe:
        flags.append(
            f"Buy & hold beat it on risk-adjusted return ({bench_sharpe:.2f} vs {sharpe:.2f} Sharpe)."
        )
    if float(metrics.get("turnover_ann", 0.0)) > 100:
        flags.append(
            f"Annual turnover of {metrics.get('turnover_ann', 0):.0f}x is far beyond what "
            "retail execution can absorb without moving the modelled fills."
        )

    grade, headline = next((g, h) for threshold, g, h in GRADES if score >= threshold)

    if score >= 70:
        verdict = (
            f"Grade {grade}. {headline}. The edge is still there after shuffling the market, "
            "after charging for every variant tried, and after walking it forward."
        )
    elif score >= 55:
        verdict = (
            f"Grade {grade}. {headline}. Parts of the panel hold up and parts do not — "
            "this is the zone where more data, not more tuning, is what settles it."
        )
    elif score >= 40:
        verdict = (
            f"Grade {grade}. {headline}. The backtest looks better than the evidence supports; "
            "the gap between the two is selection bias."
        )
    else:
        verdict = (
            f"Grade {grade}. {headline}. A randomly shuffled market produces results like this "
            "often enough that there is nothing here to trade."
        )

    return {
        "score": round(score, 1),
        "grade": grade,
        "headline": headline,
        "verdict": verdict,
        "components": {k: round(v, 1) for k, v in components.items()},
        "flags": flags,
    }
