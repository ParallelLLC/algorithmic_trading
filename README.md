# Backtest Reality Check

**algotrader 2.0 — a backtester that tries to prove itself wrong.**

Most backtesting tools answer *"how much would this have made?"*. That is the easy
question, and the answer is almost always flattering. This one answers the question
you actually need before risking money: **how much of that was luck?**

```bash
pip install -r requirements-space.txt
python app.py                                   # the Gradio app on localhost:7860
python -m algotrader.cli lab --symbol SPY --strategy sma_cross
```

<sub>The v1 agentic trading system (FinRL, Alpaca, Yahoo ingest, Streamlit/Dash UIs) is
unchanged and still lives here — see [docs/AGENTIC_SYSTEM_V1.md](docs/AGENTIC_SYSTEM_V1.md).</sub>

---

## Two labs

**The Lab** validates a timing rule on one asset. **The Portfolio Lab** validates a
cross-sectional book that ranks many names — and it asks three harder questions,
because a long-short book fails in ways a timing rule cannot.

## The four ways a backtest lies

| The lie | The test | Where |
|---|---|---|
| The market had no structure to find | Monte-Carlo **permutation test** — re-run your rule on hundreds of shuffled markets | `algotrader/validation/permutation.py` |
| You tried 200 things and reported the best | **Deflated Sharpe Ratio** — charge for every variant you tried | `algotrader/validation/deflated_sharpe.py` |
| The parameters were fitted to the past | **PBO** (CSCV) and **walk-forward** | `algotrader/validation/pbo.py`, `walkforward.py` |
| The edge is smaller than the costs | **Cost stress test** at 3× friction | `algotrader/lab.py` |

Each feeds a single **Reality Score** out of 100 with a grade from A to F:

| Weight | Component | What it measures |
|---:|---|---|
| 30% | Significance | How far outside the shuffled-market null the result sits |
| 25% | Selection | Deflated Sharpe — does it clear the best-of-N bar |
| 20% | Walk-forward | How much of the tuned Sharpe survived trading forward |
| 15% | Overfitting | 1 − PBO |
| 10% | Robustness | Sharpe retained when costs triple |

The scale is deliberately harsh. On most markets, plain buy & hold beats every
strategy in the arena on evidence, and the built-in coin-flip control out-ranks
several respectable-looking rules. That is the finding, not a bug.

## The permutation test, concretely

We take the real price series and shuffle it. Each bar's gap, high, low, body and
volume are kept intact, but their **order** is destroyed. The result is a market
with the same volatility and the same fat tails, and no exploitable structure at
all. Then we re-run *your exact rule* on hundreds of these shuffled markets.

If your Sharpe sits inside that cloud, your rule found nothing that a coin-flip
market would not also have handed it. The p-value is the share of shuffled markets
that did as well or better.

Block mode resamples contiguous chunks instead of single bars, preserving
short-horizon momentum and volatility clustering — a harder null that trend
strategies deserve to be held to.

## Cross-sectional books get a harder null

Shuffling the price path is the right null for a timing rule and the *wrong* one for
a book that ranks names: it destroys the market's whole correlation structure, and
almost any long-short book clears a null that weak.

So the Portfolio Lab permutes the **weights across assets within each date**. Every
calendar effect survives. Every correlation between names survives. Each date's gross
exposure, net exposure and position count survive *exactly*. The only thing destroyed
is the link between the strategy's choice and the asset it chose.

A book that beats that null is picking names. One that doesn't was being paid for
market exposure or a style tilt — which the factor regression measures directly:

| Question | Test |
|---|---|
| Did it pick the right names? | Within-date weight permutation |
| Is it alpha, or beta you can buy for 3bps? | Style regression (market, momentum, low-vol, reversal, liquidity) with White standard errors |
| Does the universe contain the losers? | Survivorship measured, not assumed |

That last one is not optional. A universe where every name is still trading after ten
years was chosen after the fact, and every result computed on it is an upper bound.
The Panel measures survival directly and the Reality Score caps at 60 when it finds
none.

```python
from algotrader import PortfolioLabConfig, run_portfolio_lab

report = run_portfolio_lab(PortfolioLabConfig(
    symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD", "TLT"],
    strategy="xs_momentum",
    rebalance="M",
))
print(report.verdict["grade"], report.permutation.p_value)
print(report.attribution["note"])
print(report.survivorship.note)
```

```bash
python -m algotrader.cli portfolio --symbols SPY,QQQ,AAPL,MSFT,NVDA --strategy xs_momentum
```

## Turnover is measured against drift, not against the last target

Holding 50% of a book in a name that doubles leaves you at 67% without trading. A
backtest that charges turnover as `|target[t] - target[t-1]|` understates the cost of
doing nothing and overstates the cost of rebalancing. Both engines measure turnover
against the *drifted* weight instead, and a rebalance schedule (`D`/`W`/`M`/`Q`) lets a
monthly book drift between dates rather than paying daily to stand still.

## No look-ahead, by construction

A strategy emits a target exposure at each bar's close using only data up to that
bar. The engine holds `position[t] = target[t - lag]` with `lag >= 1`, so a signal
computed on Tuesday's close cannot earn Tuesday's move.

That is the single line where look-ahead could enter, and the test suite asserts it
from four directions — including that truncating the data never changes the equity
curve before the cut, and that a `lag=0` request is refused outright.

## Python API

```python
from algotrader import LabConfig, run_lab

report = run_lab(LabConfig(
    symbol="SPY",
    start="2015-01-01",
    strategy="sma_cross",
    params={"fast": 20, "slow": 100},
    commission_bps=1.0,
    slippage_bps=2.0,
    n_permutations=500,
))

print(report.verdict["grade"], report.verdict["score"])
print("p-value          ", report.permutation.p_value)
print("deflated Sharpe  ", report.dsr["dsr"])
print("overfit prob.    ", report.pbo["pbo"])
print("walk-forward eff.", report.walkforward["efficiency"])
for flag in report.verdict["flags"]:
    print(" !", flag)
```

Lower-level pieces compose on their own:

```python
from algotrader import load_ohlcv, run_backtest, get_strategy
from algotrader.types import CostModel

market = load_ohlcv("BTC-USD", "2018-01-01")
strategy = get_strategy("donchian_breakout")
result = run_backtest(
    market.df,
    strategy.generate(market.df, {"window": 55}),
    costs=CostModel(commission_bps=1, slippage_bps=5, short_borrow_bps=50),
)
print(result.metrics["sharpe"], result.metrics["max_drawdown"])
```

## CLI

```bash
python -m algotrader.cli strategies                     # list the zoo
python -m algotrader.cli lab --symbol NVDA --strategy rsi_reversion --permutations 500
python -m algotrader.cli lab --symbol SPY --param fast=10 --param slow=50 --json
python -m algotrader.cli arena --symbol BTC-USD --start 2018-01-01
```

`--source synthetic` forces the offline simulator, which makes runs fully
deterministic and network-free.

## The strategy zoo

Single asset: `buy_and_hold` · `sma_cross` · `ema_cross` · `macd_trend` ·
`rsi_reversion` · `bollinger_reversion` · `donchian_breakout` · `momentum` ·
`vol_target_momentum` · `channel_trend` · `coin_flip`

Cross-sectional: `equal_weight` · `xs_momentum` · `xs_reversal` · `low_volatility` ·
`xs_value_proxy` · `xs_random`

Buy & hold and the coin flip are controls, and they stay in the arena on purpose: a
leaderboard without a control group is marketing, not measurement.

Adding one is a function and a registry entry — see `algotrader/strategies.py`.

## Data

Live prices come from Yahoo Finance. When the network is unavailable or rate-limited,
the app falls back to a deterministic market simulator — regime switching, Student-t
innovations, persistent volatility — and says so on every result. Naive geometric
Brownian motion flatters strategies; this simulator does not.

Set `ALGOTRADER_OFFLINE=1` to skip network access entirely.

## Deploying the Hugging Face Space

The Space ships `app.py` plus the `algotrader` package and nothing else, so it builds
in well under a minute:

```bash
HF_TOKEN=hf_xxx ./scripts/deploy_hf_space.sh <your-username>/backtest-reality-check
```

Or set the `HF_TOKEN` secret and `HF_SPACE_ID` variable on the repository and let
`.github/workflows/sync-hf-space.yml` publish on every push to `main`. The workflow
runs the test suite and builds the app before it publishes anything.

`SPACE_README.md` is the Space card (with the Hugging Face YAML frontmatter);
`requirements-space.txt` is its dependency set. The root `requirements.txt` still
carries the full v1 stack for CI, Docker and the FinRL agents.

## Tests

```bash
python -m pytest tests/test_v2_*.py -q     # 162 tests, ~25s, no network
```

The validation tests check both directions, which is the part that matters: the
statistics must reject noise **and** detect a real edge. They build a market with
genuine serial correlation and assert that the permutation test finds it, that PBO
stays near 0.5 on pure noise and drops below 0.15 when one variant is genuinely
better, and that walk-forward efficiency survives.

The cross-sectional null is held to the same standard, and it is calibrated: on a
universe with no cross-sectional structure it returns p ≈ 0.5, and its power rises
monotonically with the size of the injected effect. The tests also assert the
permutation preserves each date's gross exposure, net exposure and position count
exactly — if it did not, the null would be testing something else.

## References

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality*
- Bailey, Borwein, López de Prado & Zhu (2016), *The Probability of Backtest Overfitting*
- Masters (2018), *Permutation and Randomization Tests for Trading System Development*

## License

Apache-2.0. Research tooling, not investment advice. Nothing here is a
recommendation to trade.
