---
title: Backtest Reality Check
emoji: 🎲
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: true
license: apache-2.0
short_description: Your backtest is probably lying to you. This proves it.
tags:
  - finance
  - quantitative-finance
  - algorithmic-trading
  - backtesting
  - statistics
  - time-series
---

# Backtest Reality Check

**Your backtest is probably lying to you.**

Pick a market and a trading rule. This Space runs the backtest — and then spends
the rest of its effort trying to prove the result was luck.

Most backtesting tools answer *"how much would this have made?"*. That is the easy
question, and the answer is almost always flattering. This one answers the question
you need before risking money: **how much of that was luck?**

## The four ways a backtest lies, and the test for each

| The lie | The test |
|---|---|
| The market had no structure to find | **Permutation test** — re-run your rule on hundreds of shuffled markets |
| You tried 200 things and reported the best | **Deflated Sharpe Ratio** — charge for every variant you tried |
| The parameters were fitted to the past | **PBO + walk-forward** — does the in-sample winner keep winning? |
| The edge is smaller than the costs | **Cost stress test** — triple the friction and see what survives |

Each contributes to a single **Reality Score** out of 100, with a grade from A to F.
The scale is deliberately harsh. Most strategies people post online score below 40.

## Try this first

Run the **Arena** tab on `SPY`. On most markets and most date ranges, plain
**buy & hold** tops the leaderboard, and the **coin flip** control out-ranks
several respectable-looking strategies. That is not a bug in the app — it is the
finding.

## How the permutation test works

We take the real price series and shuffle it. Each bar's gap, high, low, body and
volume are kept intact, but their **order** is destroyed. The result is a market
with the same volatility and the same fat tails, and no exploitable structure at
all. Then we re-run *your exact rule* on hundreds of these shuffled markets.

If your Sharpe ratio sits comfortably inside that cloud, your rule found nothing
a coin-flip market would not also have handed it.

## No look-ahead, by construction

A strategy emits a target exposure at each bar's close using only data up to that
bar. The engine holds `position[t] = target[t - lag]` with `lag >= 1`, so a signal
computed on Tuesday's close cannot earn Tuesday's move. That is the single line
where look-ahead could enter, and the test suite asserts it directly.

## Use it from Python

```python
from algotrader import LabConfig, run_lab

report = run_lab(LabConfig(symbol="SPY", strategy="sma_cross"))
print(report.verdict["grade"], report.verdict["score"])
print(report.permutation.p_value, report.dsr["dsr"], report.pbo["pbo"])
```

Or from the command line:

```bash
python -m algotrader.cli lab --symbol SPY --strategy donchian_breakout --permutations 500
python -m algotrader.cli arena --symbol BTC-USD
```

## Data

Live prices come from Yahoo Finance. When the network is unavailable or rate-limited,
the app falls back to a deterministic market simulator with regime switching, fat
tails and volatility clustering — and says so, clearly, on every result. The
statistics remain valid; they are just measured on a simulated market.

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest Overfitting*
- Masters (2018), *Permutation and Randomization Tests for Trading System Development*

---

Apache-2.0. Research tooling, not investment advice. Nothing here is a
recommendation to trade.
