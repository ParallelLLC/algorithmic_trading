"""Backtest Reality Check — the Hugging Face Space entrypoint.

Point it at a ticker and a trading rule. It runs the backtest, then spends the
rest of its time trying to prove the result was luck: shuffled markets,
selection-bias deflation, walk-forward, and a cost stress test.

Run locally with ``python app.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import gradio as gr
import pandas as pd

from algotrader import __version__
from algotrader.charts import (
    arena_chart,
    drawdown_chart,
    empty_figure,
    equity_chart,
    exposure_chart,
    permutation_chart,
    score_chart,
    walkforward_chart,
)
from algotrader.data import DEFAULT_UNIVERSE
from algotrader.lab import LabConfig, run_arena, run_lab
from algotrader.strategies import REGISTRY, get_strategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

MAX_PARAMS = 3
STRATEGY_CHOICES = [(s.name, key) for key, s in REGISTRY.items()]

GRADE_COLORS = {
    "A": "#0ca30c",
    "B": "#3987e5",
    "C": "#fab219",
    "D": "#ec835a",
    "F": "#d03b3b",
}

CSS = """
.gradio-container { max-width: 1280px !important; }
#hero h1 { font-size: 2.4rem; line-height: 1.1; margin: 0 0 .4rem 0; letter-spacing: -.02em; }
#hero p { color: #c3c2b7; margin: 0; font-size: 1.05rem; max-width: 60ch; }
.score-card {
  display: flex; gap: 24px; align-items: center; padding: 22px 24px;
  border: 1px solid rgba(255,255,255,.10); border-radius: 14px; background: #1a1a19;
}
.score-badge {
  min-width: 132px; text-align: center; padding: 14px 10px; border-radius: 12px;
  background: #0d0d0d; border: 1px solid rgba(255,255,255,.10);
}
.score-badge .grade { font-size: 3.2rem; font-weight: 700; line-height: 1; }
.score-badge .num { font-size: .95rem; color: #898781; margin-top: 6px; }
.score-body h3 { margin: 0 0 6px 0; font-size: 1.15rem; color: #fff; }
.score-body p { margin: 0; color: #c3c2b7; line-height: 1.55; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px,1fr)); gap: 10px; margin-top: 14px; }
.tile { padding: 12px 14px; border: 1px solid rgba(255,255,255,.10); border-radius: 10px; background: #1a1a19; }
.tile .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: #898781; }
.tile .value { font-size: 1.5rem; font-weight: 600; color: #fff; margin-top: 4px; }
.tile .sub { font-size: .75rem; color: #898781; margin-top: 2px; }
.flags { margin-top: 14px; padding: 0; list-style: none; }
.flags li {
  padding: 9px 12px; margin-bottom: 7px; border-radius: 8px; background: #1a1a19;
  border-left: 3px solid #ec835a; color: #c3c2b7; font-size: .9rem; line-height: 1.5;
}
.provenance { font-size: .82rem; color: #898781; margin-top: 10px; }
.provenance.sim { color: #fab219; }
"""


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.1f}%"


def param_controls(strategy_key: str):
    """Re-label the shared sliders to match the selected strategy."""
    strategy = get_strategy(strategy_key)
    updates = []
    for i in range(MAX_PARAMS):
        if i < len(strategy.params):
            spec = strategy.params[i]
            lo = spec.minimum if spec.minimum is not None else min(spec.grid)
            hi = spec.maximum if spec.maximum is not None else max(spec.grid)
            updates.append(
                gr.update(
                    visible=True,
                    label=spec.label,
                    value=spec.cast(spec.default),
                    minimum=lo,
                    maximum=hi,
                    step=spec.step or (1 if spec.kind == "int" else 0.1),
                )
            )
        else:
            updates.append(gr.update(visible=False))
    return tuple(updates)


def collect_params(strategy_key: str, *values) -> Dict[str, float]:
    strategy = get_strategy(strategy_key)
    return {spec.name: spec.cast(values[i]) for i, spec in enumerate(strategy.params[:MAX_PARAMS])}


def _score_card(report) -> str:
    v = report.verdict
    color = GRADE_COLORS.get(v["grade"], "#898781")
    market = report.market
    provenance = (
        f'<div class="provenance">Data: {market.source} · {market.symbol} · '
        f"{market.start.date()} to {market.end.date()} · {len(market.df):,} bars</div>"
        if market.is_real
        else f'<div class="provenance sim">⚠ {market.note}</div>'
    )
    flags = "".join(f"<li>{f}</li>" for f in v["flags"])
    flags_html = f'<ul class="flags">{flags}</ul>' if flags else ""

    return f"""
<div class="score-card">
  <div class="score-badge">
    <div class="grade" style="color:{color}">{v['grade']}</div>
    <div class="num">{v['score']} / 100</div>
  </div>
  <div class="score-body">
    <h3>{report.strategy.name} on {market.symbol}</h3>
    <p>{v['verdict']}</p>
  </div>
</div>
{flags_html}
{provenance}
"""


def _tiles(report) -> str:
    m = report.backtest.metrics
    b = report.backtest.benchmark_metrics
    perm = report.permutation
    p_text = f"{perm.p_value:.3f}" if perm else "—"
    p_sub = "vs shuffled markets" if perm else "test skipped"
    pbo = report.pbo.get("pbo")
    pbo_text = f"{pbo:.0%}" if pbo is not None and pbo == pbo else "n/a"

    cells = [
        ("Total return", _fmt_pct(m.get("total_return", 0)), f"buy & hold {_fmt_pct(b.get('total_return', 0))}"),
        ("CAGR", _fmt_pct(m.get("cagr", 0)), f"over {m.get('years', 0):.1f} years"),
        ("Sharpe", f"{m.get('sharpe', 0):.2f}", f"buy & hold {b.get('sharpe', 0):.2f}"),
        ("Max drawdown", _fmt_pct(m.get("max_drawdown", 0)), f"{m.get('time_under_water_yrs', 0):.1f}y under water"),
        ("p-value", p_text, p_sub),
        ("Deflated Sharpe", f"{report.dsr.get('dsr', 0):.2f}", f"after {report.trials.get('n', 1)} variants"),
        ("Overfit prob.", pbo_text, "in-sample winner fails OOS"),
        ("Trades", f"{int(m.get('n_trades', 0)):,}", f"{m.get('turnover_ann', 0):.0f}x turnover/yr"),
    ]
    tiles = "".join(
        f'<div class="tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in cells
    )
    return f'<div class="tiles">{tiles}</div>'


def _detail_markdown(report) -> str:
    dsr, wf, pbo, stress = report.dsr, report.walkforward, report.pbo, report.cost_stress
    mtr = dsr.get("min_track_record_years", float("inf"))
    mtr_text = f"{mtr:.1f} years" if mtr == mtr and mtr != float("inf") else "never, at this effect size"

    lines = [
        "### Reading the evidence",
        "",
        f"**Selection bias.** {report.trials.get('n', 1)} parameter variants of "
        f"*{report.strategy.name}* were backtested. The luckiest skill-free variant of that many "
        f"would be expected to show an annualised Sharpe of about "
        f"**{dsr.get('threshold_sr_annual', 0):.2f}** on its own. Yours was "
        f"**{report.backtest.sharpe:.2f}**, which puts the deflated probability of a real edge at "
        f"**{dsr.get('dsr', 0):.0%}**.",
        "",
        f"**Track record needed.** To call this Sharpe significant at 95% confidence given its "
        f"skew ({dsr.get('skew', 0):.2f}) and kurtosis ({dsr.get('kurtosis', 0):.1f}), you would need "
        f"about **{mtr_text}** of live returns.",
        "",
        f"**Costs.** At the modelled friction the Sharpe is {stress.get('sharpe_1x', 0):.2f}. "
        f"Triple the costs and it becomes {stress.get('sharpe_3x', 0):.2f} "
        f"({stress.get('ratio', 0):.0%} retained).",
        "",
    ]

    if wf.get("folds"):
        lines += [
            f"**Walk-forward.** Across {len(wf['folds'])} folds the tuned in-sample Sharpe averaged "
            f"{wf.get('mean_is_sharpe', 0):.2f} and the blind out-of-sample Sharpe averaged "
            f"{wf.get('mean_oos_sharpe', 0):.2f} — an efficiency of {wf.get('efficiency', 0):.0%}. "
            f"{wf.get('oos_win_rate', 0):.0%} of folds were profitable out of sample, and the tuner "
            f"picked a different parameter set in {wf.get('param_instability', 0):.0%} of them.",
            "",
        ]
    if pbo.get("note"):
        lines += [f"**Overfitting.** {pbo['note']}", ""]
    elif pbo.get("pbo") == pbo.get("pbo"):
        lines += [
            f"**Overfitting (CSCV).** Over {pbo.get('n_combinations', 0)} in/out splits of "
            f"{pbo.get('n_strategies', 0)} variants, the in-sample winner landed in the bottom half "
            f"out of sample **{pbo.get('pbo', 0):.0%}** of the time, and lost money outright "
            f"{pbo.get('prob_oos_loss', 0):.0%} of the time. The most frequently selected variant was "
            f"`{pbo.get('most_selected_label', 'n/a')}` "
            f"({pbo.get('selection_stability', 0):.0%} of splits).",
            "",
        ]

    lines += [
        "> Past performance, simulated or otherwise, does not predict future returns. "
        "This is research tooling, not investment advice.",
    ]
    return "\n".join(lines)


def analyse(
    symbol: str,
    start: str,
    end: str,
    strategy_key: str,
    p1: float,
    p2: float,
    p3: float,
    commission: float,
    slippage: float,
    allow_short: bool,
    n_permutations: int,
    perm_method: str,
    wf_folds: int,
    progress=gr.Progress(),
):
    """Main Lab handler. Never raises into the UI — it returns a readable message."""
    try:
        cfg = LabConfig(
            symbol=symbol or "SPY",
            start=start or "2015-01-01",
            end=end or None,
            strategy=strategy_key,
            params=collect_params(strategy_key, p1, p2, p3),
            commission_bps=float(commission),
            slippage_bps=float(slippage),
            allow_short=bool(allow_short),
            n_permutations=int(n_permutations),
            permutation_method="block" if perm_method.startswith("Block") else "permute",
            wf_folds=int(wf_folds),
        )
        report = run_lab(cfg, progress=lambda f, m: progress(f, desc=m))
    except Exception as exc:  # noqa: BLE001 - the UI must always say something useful
        logger.exception("Lab run failed")
        message = f'<div class="score-card"><div class="score-body"><h3>Could not run that</h3><p>{exc}</p></div></div>'
        blank = empty_figure("No results.")
        return message, "", blank, blank, blank, blank, blank, blank, ""

    return (
        _score_card(report),
        _tiles(report),
        permutation_chart(report),
        equity_chart(report),
        drawdown_chart(report),
        exposure_chart(report),
        walkforward_chart(report),
        score_chart(report.verdict),
        _detail_markdown(report),
    )


def race(symbol: str, start: str, allow_short: bool, n_permutations: int, progress=gr.Progress()):
    try:
        cfg = LabConfig(
            symbol=symbol or "SPY",
            start=start or "2015-01-01",
            allow_short=bool(allow_short),
        )
        table, market, _ = run_arena(
            cfg,
            n_permutations=int(n_permutations),
            progress=lambda f, m: progress(f, desc=m),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Arena run failed")
        return pd.DataFrame({"Error": [str(exc)]}), empty_figure("No results."), ""

    display = table.copy()
    for col in ("Return", "CAGR", "MaxDD"):
        display[col] = display[col].map(lambda v: f"{v * 100:,.1f}%")
    for col in ("Sharpe", "DSR", "Evidence"):
        display[col] = display[col].map(lambda v: f"{v:.2f}")
    display["p-value"] = display["p-value"].map(lambda v: "—" if v != v else f"{v:.3f}")
    display = display.drop(columns=["key"])

    provenance = (
        f"Data: {market.source} · {market.symbol} · {market.start.date()} to {market.end.date()}"
        if market.is_real
        else f"⚠ {market.note}"
    )
    note = (
        f"{provenance}\n\nRanked by **evidence** — `(1 − p) × deflated Sharpe` — not by return. "
        "*Buy & Hold* and *Coin Flip* are in the field on purpose: a leaderboard without a "
        "control group is marketing, not measurement."
    )
    return display, arena_chart(table), note


HOW_IT_WORKS = """
## Why most backtests are wrong

A backtest is a measurement taken with a ruler you built after seeing the thing you
are measuring. Four failure modes do almost all the damage, and this Space tests for
each one.

### 1. The market had no structure to find — permutation test

We take the real price series and shuffle it: each bar's gap, high, low, body and
volume are kept intact, but their **order** is destroyed. The result is a market with
the same volatility and the same fat tails, and no exploitable structure whatsoever.
Then we re-run *your exact rule* on hundreds of these shuffled markets.

If your Sharpe sits inside that cloud of results, your rule found nothing that a
coin-flip market would not also have handed it. The **p-value** is the share of
shuffled markets that did as well or better.

*Block mode* resamples contiguous chunks instead of single bars, preserving
short-horizon momentum and volatility clustering. It is a harder null, and trend
strategies should be held to it.

### 2. You tried 200 things and reported the best — Deflated Sharpe Ratio

If you test 200 worthless strategies, the best of them will show a Sharpe near 1.0
purely by chance. The **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) works
out what the luckiest of *N* skill-free variants would have scored, and asks whether
yours beats that bar — with an extra penalty for negative skew and fat tails, the
return shapes that flatter naive Sharpe ratios.

This Space counts the whole parameter grid as trials, because that is what a
researcher would really have run.

### 3. The parameters were fitted to the past — PBO and walk-forward

**Probability of Backtest Overfitting** (CSCV) cuts the timeline into chunks, and for
every way of splitting them half in-sample and half out-of-sample, checks whether the
in-sample winner stayed a winner. If the winner lands in the bottom half about half
the time, PBO ≈ 50% and your selection process has no skill at all.

**Walk-forward** re-tunes on a training window and trades the next window blind,
rolling forward. Efficiency is out-of-sample Sharpe over in-sample Sharpe: 100% means
the edge survived intact, 0% means it was entirely curve-fit.

### 4. The edge is smaller than the costs — stress test

Every result here is net of commission and slippage charged on exposure changes, plus
a borrow fee on short positions. We then re-run at **triple** the friction. A real edge
degrades; a fake one disappears.

---

## The Reality Score

| Weight | Component | What it measures |
|---:|---|---|
| 30% | Significance | How far outside the shuffled-market null the result sits |
| 25% | Selection | Deflated Sharpe — does it clear the best-of-N bar |
| 20% | Walk-forward | How much of the tuned Sharpe survived trading forward |
| 15% | Overfitting | 1 − PBO, from combinatorially symmetric cross-validation |
| 10% | Robustness | Sharpe retained when costs triple |

Grades: **A** ≥ 85 · **B** ≥ 70 · **C** ≥ 55 · **D** ≥ 40 · **F** below 40.

The scale is deliberately harsh. Most strategies people post online score below 40,
and the honest response to that is not to soften the scale.

## No look-ahead, by construction

A strategy emits a target exposure at each bar's close using only data up to that
bar. The engine holds `position[t] = target[t - lag]` with `lag ≥ 1`, so a signal
computed on Tuesday's close cannot earn Tuesday's move. That is the single line where
look-ahead could enter, and the test suite asserts it directly.

## Use it from Python

```python
from algotrader import LabConfig, run_lab

report = run_lab(LabConfig(symbol="SPY", strategy="sma_cross", params={"fast": 20, "slow": 100}))
print(report.verdict["grade"], report.verdict["score"])
print(report.permutation.p_value, report.dsr["dsr"], report.pbo["pbo"])
```

Or from the command line:

```bash
python -m algotrader.cli lab --symbol SPY --strategy donchian_breakout --permutations 500
python -m algotrader.cli arena --symbol BTC-USD
```

---

*Research tooling, not investment advice. Nothing here is a recommendation to trade.*
"""


# Gradio 6 moved `css` and `theme` from the Blocks constructor to launch().
# Spaces pin their own version, so pass them wherever the installed one wants.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_STYLE_KWARGS = {"css": CSS, "theme": gr.themes.Base()}
_BLOCKS_KWARGS = {} if _GRADIO_MAJOR >= 6 else _STYLE_KWARGS
# Gradio 6 also dropped launch(show_api=...).
_LAUNCH_KWARGS = dict(_STYLE_KWARGS) if _GRADIO_MAJOR >= 6 else {"show_api": False}


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Backtest Reality Check", **_BLOCKS_KWARGS) as demo:
        with gr.Column(elem_id="hero"):
            gr.HTML(
                "<h1>Backtest Reality Check</h1>"
                "<p>Your backtest is probably lying to you. Pick a market and a trading rule — "
                "this runs it, then spends the rest of its effort trying to prove the result "
                "was luck.</p>"
            )

        with gr.Tabs():
            with gr.Tab("The Lab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        symbol = gr.Dropdown(
                            choices=DEFAULT_UNIVERSE, value="SPY", label="Ticker",
                            allow_custom_value=True,
                            info="Any Yahoo Finance symbol. Falls back to a simulated market if offline.",
                        )
                        with gr.Row():
                            start = gr.Textbox(value="2015-01-01", label="Start", scale=1)
                            end = gr.Textbox(value="", label="End (blank = today)", scale=1)

                        strategy = gr.Dropdown(
                            choices=STRATEGY_CHOICES, value="sma_cross", label="Strategy"
                        )
                        strategy_note = gr.Markdown(get_strategy("sma_cross").description)

                        param_sliders = [
                            gr.Slider(label=f"Parameter {i + 1}", visible=False, minimum=0, maximum=100)
                            for i in range(MAX_PARAMS)
                        ]

                        with gr.Accordion("Costs and testing", open=False):
                            commission = gr.Slider(0, 20, value=1, step=0.5, label="Commission (bps per trade)")
                            slippage = gr.Slider(0, 50, value=2, step=0.5, label="Slippage (bps per trade)")
                            allow_short = gr.Checkbox(value=True, label="Allow short positions")
                            n_perms = gr.Slider(
                                0, 1000, value=250, step=50, label="Shuffled markets to test against",
                                info="More is stricter and slower. 250 is plenty for a first look.",
                            )
                            perm_method = gr.Radio(
                                ["Shuffle bars (standard)", "Block bootstrap (harder)"],
                                value="Shuffle bars (standard)", label="Null market",
                            )
                            wf_folds = gr.Slider(2, 8, value=5, step=1, label="Walk-forward folds")

                        run_button = gr.Button("Run reality check", variant="primary", size="lg")

                        gr.Examples(
                            label="Or try one of these",
                            examples=[
                                ["SPY", "sma_cross"],
                                ["BTC-USD", "donchian_breakout"],
                                ["NVDA", "rsi_reversion"],
                                ["QQQ", "momentum"],
                                ["SPY", "coin_flip"],
                            ],
                            inputs=[symbol, strategy],
                        )

                    with gr.Column(scale=2):
                        verdict_html = gr.HTML(
                            '<div class="score-card"><div class="score-body">'
                            "<h3>Nothing tested yet</h3><p>Pick a market and a rule, then hit "
                            "<b>Run reality check</b>. A full run is a few seconds.</p>"
                            "</div></div>"
                        )
                        tiles_html = gr.HTML("")

                # The headline test gets the full width — it is the whole point.
                perm_plot = gr.Plot(value=empty_figure("The headline test appears here.", height=320))
                equity_plot = gr.Plot(value=empty_figure())
                with gr.Row():
                    dd_plot = gr.Plot(value=empty_figure(height=240))
                    exposure_plot = gr.Plot(value=empty_figure(height=200))
                with gr.Row():
                    wf_plot = gr.Plot(value=empty_figure(height=300))
                    components_plot = gr.Plot(value=empty_figure(height=260))
                detail_md = gr.Markdown("")

                strategy.change(
                    fn=param_controls, inputs=strategy, outputs=param_sliders
                ).then(
                    fn=lambda k: get_strategy(k).description, inputs=strategy, outputs=strategy_note
                )

                run_button.click(
                    fn=analyse,
                    inputs=[
                        symbol, start, end, strategy, *param_sliders,
                        commission, slippage, allow_short, n_perms, perm_method, wf_folds,
                    ],
                    outputs=[
                        verdict_html, tiles_html, perm_plot, equity_plot,
                        dd_plot, exposure_plot, wf_plot, components_plot, detail_md,
                    ],
                )

            with gr.Tab("Arena"):
                gr.Markdown(
                    "Race every strategy on the same market, ranked by **evidence** rather than "
                    "return. Buy & hold and a coin flip stay in the field as controls."
                )
                with gr.Row():
                    arena_symbol = gr.Dropdown(
                        choices=DEFAULT_UNIVERSE, value="SPY", label="Ticker", allow_custom_value=True
                    )
                    arena_start = gr.Textbox(value="2015-01-01", label="Start")
                    arena_short = gr.Checkbox(value=True, label="Allow shorts")
                    arena_perms = gr.Slider(0, 400, value=120, step=20, label="Shuffled markets per strategy")
                arena_button = gr.Button("Run the arena", variant="primary")
                arena_note = gr.Markdown("")
                arena_table = gr.Dataframe(interactive=False, wrap=True)
                arena_plot = gr.Plot(value=empty_figure(height=380))

                arena_button.click(
                    fn=race,
                    inputs=[arena_symbol, arena_start, arena_short, arena_perms],
                    outputs=[arena_table, arena_plot, arena_note],
                )

            with gr.Tab("How it works"):
                gr.Markdown(HOW_IT_WORKS)

        gr.Markdown(
            f"<sub>algotrader {__version__} · Apache-2.0 · "
            "Research tooling, not investment advice.</sub>"
        )

        demo.load(fn=param_controls, inputs=strategy, outputs=param_sliders)

    return demo


if __name__ == "__main__":
    build_app().queue(max_size=24).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        **_LAUNCH_KWARGS,
    )
