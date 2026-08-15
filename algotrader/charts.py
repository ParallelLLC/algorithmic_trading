"""Plotly figures for the Lab.

Colour system (dark surface, validated for CVD separation):

* **blue** is always *your strategy's honest result* — the realised equity
  curve, the out-of-sample fold, the observed Sharpe.
* **orange** is always *the thing it is measured against* — buy & hold, the
  in-sample fold, the null distribution.

Holding that mapping across every figure means a reader learns it once.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

SUBJECT = "#3987e5"  # categorical slot 1
REFERENCE = "#d95926"  # categorical slot 2
NEGATIVE = "#e66767"  # negative arm of the diverging pair (drawdowns)

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_EMPTY_NOTE = "Run an analysis to populate this chart."


def _base_layout(title: str, height: int = 340, **kwargs) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left", pad=dict(b=8)),
        paper_bgcolor=PAGE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_SECONDARY),
        height=height,
        margin=dict(l=56, r=24, t=48, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(color=INK, family=FONT)),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=INK_MUTED)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=INK_SECONDARY, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        **kwargs,
    )


def empty_figure(message: str = _EMPTY_NOTE, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_base_layout("", height=height))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(color=INK_MUTED, size=13),
    )
    return fig


def equity_chart(report) -> go.Figure:
    """Strategy equity against buy & hold, both indexed to the same start."""
    bt = report.backtest
    strat = bt.equity / bt.equity.iloc[0] * 100.0
    bench = bt.benchmark_equity / bt.benchmark_equity.iloc[0] * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bench.index, y=bench.to_numpy(), name="Buy & hold", mode="lines",
            line=dict(color=REFERENCE, width=2, dash="dash"),
            hovertemplate="Buy & hold  %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=strat.index, y=strat.to_numpy(), name=report.strategy.name, mode="lines",
            line=dict(color=SUBJECT, width=2),
            hovertemplate=report.strategy.name + "  %{y:.1f}<extra></extra>",
        )
    )

    # Direct-label the two endpoints; the axis and tooltip carry everything else.
    for series, color, label in ((strat, SUBJECT, report.strategy.name), (bench, REFERENCE, "Buy & hold")):
        fig.add_annotation(
            x=series.index[-1], y=float(series.iloc[-1]),
            text=f"  {label}: {series.iloc[-1]:.0f}", showarrow=False,
            xanchor="left", font=dict(color=color, size=11),
        )

    fig.update_layout(**_base_layout("Growth of 100 (net of costs)", height=360))
    fig.update_layout(margin=dict(l=56, r=140, t=48, b=40))
    return fig


def drawdown_chart(report) -> go.Figure:
    """Underwater plot — how deep, and for how long."""
    from .metrics import drawdown_series

    dd = drawdown_series(report.backtest.equity) * 100.0
    fig = go.Figure(
        go.Scatter(
            x=dd.index, y=dd.to_numpy(), mode="lines", name="Drawdown",
            line=dict(color=NEGATIVE, width=2), fill="tozeroy",
            fillcolor="rgba(230,103,103,0.18)",
            hovertemplate="Drawdown %{y:.1f}%<extra></extra>",
        )
    )
    trough = float(dd.min())
    fig.add_annotation(
        x=dd.idxmin(), y=trough, text=f"worst {trough:.1f}%", showarrow=True,
        arrowhead=0, arrowcolor=AXIS, ay=24, font=dict(color=INK_SECONDARY, size=11),
    )
    fig.update_layout(**_base_layout("Drawdown", height=240, showlegend=False))
    fig.update_yaxes(ticksuffix="%")
    return fig


def permutation_chart(report) -> go.Figure:
    """The headline chart: your Sharpe against Sharpes from shuffled markets."""
    perm = report.permutation
    if perm is None or perm.null.size == 0:
        return empty_figure("Permutation test was skipped.", height=320)

    null = perm.null
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=null, name="Shuffled markets (no real edge)", nbinsx=44,
            marker=dict(color="rgba(217,89,38,0.55)", line=dict(color=REFERENCE, width=1)),
            hovertemplate="Sharpe %{x:.2f}<br>%{y} shuffles<extra></extra>",
        )
    )

    top = np.histogram(null, bins=44)[0].max() if null.size else 1
    fig.add_trace(
        go.Scatter(
            x=[perm.observed, perm.observed], y=[0, top * 1.08], mode="lines",
            name="Your strategy", line=dict(color=SUBJECT, width=2),
            hovertemplate="Your Sharpe %{x:.2f}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=perm.observed, y=top * 1.08, text=f"  your Sharpe {perm.observed:.2f}",
        showarrow=False, xanchor="left", font=dict(color=SUBJECT, size=11),
    )

    beats = (null >= perm.observed).mean() * 100.0
    fig.update_layout(
        **_base_layout(
            f"Permutation test — {beats:.0f}% of structure-free markets did this well or better "
            f"(p = {perm.p_value:.3f})",
            height=320,
        )
    )
    fig.update_layout(hovermode="closest", bargap=0.02)
    fig.update_xaxes(title=dict(text="Annualised Sharpe ratio", font=dict(color=INK_MUTED, size=11)))
    # Headroom so the "your Sharpe" label never collides with the plot edge.
    fig.update_yaxes(
        title=dict(text="Shuffled markets", font=dict(color=INK_MUTED, size=11)),
        range=[0, top * 1.28],
    )
    return fig


def walkforward_chart(report) -> go.Figure:
    """In-sample vs out-of-sample Sharpe, fold by fold."""
    folds = report.walkforward.get("folds") or []
    if not folds:
        return empty_figure(report.walkforward.get("note") or _EMPTY_NOTE, height=300)

    labels = [f"Fold {f['fold']}<br><span style='font-size:10px'>{f['test_start'][:7]}</span>" for f in folds]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels, y=[f["is_sharpe"] for f in folds], name="In-sample (tuned)",
            marker=dict(color=REFERENCE, line=dict(color=SURFACE, width=2)),
            hovertemplate="In-sample Sharpe %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels, y=[f["oos_sharpe"] for f in folds], name="Out-of-sample (blind)",
            marker=dict(color=SUBJECT, line=dict(color=SURFACE, width=2)),
            hovertemplate="Out-of-sample Sharpe %{y:.2f}<extra></extra>",
        )
    )
    eff = report.walkforward.get("efficiency", 0.0)
    fig.update_layout(
        **_base_layout(f"Walk-forward — {eff:.0%} of the tuned Sharpe survived out of sample", height=300)
    )
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, hovermode="x unified")
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig


def score_chart(verdict: Dict[str, object]) -> go.Figure:
    """The five components behind the Reality Score."""
    components = verdict.get("components") or {}
    if not components:
        return empty_figure(height=260)

    pretty = {
        "significance": "Beats shuffled markets",
        "selection": "Survives selection bias",
        "walk_forward": "Holds up walking forward",
        "overfitting": "Not overfit (PBO)",
        "robustness": "Survives 3x costs",
    }
    keys = list(pretty)
    values = [float(components.get(k, 0.0)) for k in keys]

    fig = go.Figure(
        go.Bar(
            x=values, y=[pretty[k] for k in keys], orientation="h",
            marker=dict(color=SUBJECT, line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0f}" for v in values], textposition="outside",
            textfont=dict(color=INK_SECONDARY, size=11),
            hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Where the score comes from", height=260, showlegend=False))
    fig.update_layout(margin=dict(l=190, r=48, t=48, b=32), hovermode="closest")
    fig.update_xaxes(range=[0, 108], tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(autorange="reversed")
    return fig


def arena_chart(table: pd.DataFrame) -> go.Figure:
    """Leaderboard bars. One measure, one colour — the table carries the rest."""
    if table is None or table.empty:
        return empty_figure(height=380)

    ordered = table.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=ordered["Sharpe"].to_numpy(), y=ordered["Strategy"].tolist(), orientation="h",
            marker=dict(color=SUBJECT, line=dict(color=SURFACE, width=2)),
            customdata=np.column_stack([ordered["p-value"].to_numpy(), ordered["DSR"].to_numpy()]),
            hovertemplate="%{y}<br>Sharpe %{x:.2f}<br>p = %{customdata[0]:.3f}"
                          "<br>Deflated Sharpe %{customdata[1]:.2f}<extra></extra>",
        )
    )
    # Direct-label only what matters: the ones that actually cleared significance.
    for _, row in ordered.iterrows():
        if np.isfinite(row["p-value"]) and row["p-value"] < 0.05:
            fig.add_annotation(
                x=row["Sharpe"], y=row["Strategy"], text="  p &lt; 0.05", showarrow=False,
                xanchor="left" if row["Sharpe"] >= 0 else "right",
                font=dict(color=INK_SECONDARY, size=10),
            )
    fig.update_layout(
        **_base_layout(
            "Strategy arena — Sharpe ratio, ordered by strength of evidence",
            height=max(300, 42 * len(table)),
            showlegend=False,
        )
    )
    fig.update_layout(margin=dict(l=180, r=96, t=48, b=32), hovermode="closest")
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    return fig


def exposure_chart(report) -> go.Figure:
    """What the strategy was actually holding, over time."""
    pos = report.backtest.position
    fig = go.Figure(
        go.Scatter(
            x=pos.index, y=pos.to_numpy(), mode="lines", name="Exposure",
            line=dict(color=SUBJECT, width=2, shape="hv"), fill="tozeroy",
            fillcolor="rgba(57,135,229,0.16)",
            hovertemplate="Exposure %{y:.2f}x<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Position held", height=200, showlegend=False))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig
