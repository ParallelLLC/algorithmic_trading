# Algorithmic Trading

Parallel LLC. Two layers in one repository:

1. **algotrader 2.0** (`algotrader/`, `app.py`): a backtester that tries to prove a rule was luck (permutation, deflated Sharpe, PBO, walk-forward, cost stress).
2. **Agentic v1** (`agentic_ai_system/`): FinRL policies, Yahoo or Alpaca ingest, paper/live execution, Streamlit/Dash/Jupyter UIs, Docker.

Default market data is **Yahoo Finance** (`yfinance>=1.0`), not simulated prices. The simulator exists for offline tests (`--source synthetic` or `ALGOTRADER_OFFLINE=1` with `source=auto`). Live capital still needs a separate evaluation contract. This is research tooling, not investment advice.

---

## 1. Title and Summary

**Algorithmic Trading**  
Ingest real OHLCV, test whether a timing or cross-sectional rule survives a hostile null, optionally train a FinRL policy, size orders under position and drawdown caps, route to paper or live Alpaca.

GitHub keeps two branches: `main` (protected) and `dev` (integration).

**Design themes**

* Yahoo as the default public tape (delayed, unofficial, lookback-limited)
* Validation before belief: permutation, DSR, PBO/CSCV, walk-forward, 3× cost stress
* FinRL (PPO, A2C, DDPG, TD3) unchanged on the v1 path
* Alpaca optional for authenticated bars and orders; keys from the environment
* Synthetic GBM / regime simulator only when requested
* Secrets never in git

---

## 2. Quick start

```bash
git clone https://github.com/ParallelLLC/algorithmic_trading.git
cd algorithmic_trading
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-space.txt   # algotrader + Gradio
# or: pip install -r requirements.txt   # full v1 stack (FinRL, Dash, Docker CI)
```

```bash
python app.py                                      # Gradio, localhost:7860, Yahoo by default
python -m algotrader.cli lab --symbol SPY --strategy sma_cross
python -m algotrader.cli lab --symbol NVDA --strategy rsi_reversion --permutations 500
python -m agentic_ai_system.main --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
```

`config.yaml` defaults:

```yaml
data_source:
  type: 'yahoo'
trading:
  symbol: 'AAPL'
  timeframe: '1d'    # Yahoo 1m history is ~7 days; use 1d for multi-year windows
yahoo:
  auto_adjust: true  # raw Close turns splits into fake crashes
```

Alpaca is opt-in: `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` and `data_source.type: alpaca` or `execution.broker_api: alpaca_paper`.

---

## 3. algotrader 2.0 (validation lab)

Most backtests answer "how much would this have made?" This one asks **how much of that was luck?**

### Two labs

**The Lab** validates a timing rule on one asset. **The Portfolio Lab** validates a cross-sectional book that ranks many names.

### The four ways a backtest lies

| The lie | The test | Where |
| --- | --- | --- |
| The market had no structure to find | Monte-Carlo permutation (shuffle bar order, keep gap/high/low/body/volume) | `algotrader/validation/permutation.py` |
| You tried 200 things and reported the best | Deflated Sharpe Ratio | `algotrader/validation/deflated_sharpe.py` |
| Parameters were fitted to the past | PBO (CSCV) and walk-forward | `algotrader/validation/pbo.py`, `walkforward.py` |
| The edge is smaller than the costs | Cost stress at 3× friction | `algotrader/lab.py` |

Reality Score (0–100, grades A–F): significance 30%, selection 25%, walk-forward 20%, overfitting 15%, robustness 10%. The scale is harsh on purpose. Buy-and-hold and a coin-flip stay in the arena as controls.

Cross-sectional books use a **within-date weight permutation** so market correlation survives; path-shuffle is the wrong null for a long-short ranker. Survivorship is measured. Style regression (market, momentum, low-vol, reversal, liquidity) with White standard errors.

Look-ahead: `position[t] = target[t - lag]` with `lag >= 1`. Turnover is measured against drifted weights, not `|target[t]-target[t-1]|`.

```python
from algotrader import LabConfig, run_lab

report = run_lab(LabConfig(
    symbol="SPY",
    start="2015-01-01",
    strategy="sma_cross",
    params={"fast": 20, "slow": 100},
    source="yahoo",
    n_permutations=500,
))
print(report.verdict["grade"], report.permutation.p_value, report.dsr["dsr"])
```

```bash
python -m algotrader.cli strategies
python -m algotrader.cli lab --symbol SPY --source yahoo
python -m algotrader.cli portfolio --symbols SPY,QQQ,AAPL,MSFT,NVDA --strategy xs_momentum
python -m algotrader.cli lab --source synthetic   # offline tests only
```

Single-asset zoo: `buy_and_hold`, `sma_cross`, `ema_cross`, `macd_trend`, `rsi_reversion`, `bollinger_reversion`, `donchian_breakout`, `momentum`, `vol_target_momentum`, `channel_trend`, `coin_flip`.

Cross-sectional: `equal_weight`, `xs_momentum`, `xs_reversal`, `low_volatility`, `xs_value_proxy`, `xs_random`.

**Data:** `load_ohlcv(..., source="yahoo")` downloads from Yahoo and **raises** if the download is empty. `source="auto"` is the Space fallback (cache, then simulator). `ALGOTRADER_OFFLINE=1` disables the network.

**HF Space:** `HF_TOKEN=hf_xxx ./scripts/deploy_hf_space.sh <user>/backtest-reality-check`. Card is `SPACE_README.md`. Tests: `python -m pytest tests/test_v2_*.py -q`.

References: Bailey & López de Prado (2014) DSR; Bailey et al. (2016) PBO; Masters (2018) permutation tests for trading systems.

---

## 4. Concepts and methods (v1 ingest and execution)

| Source | Default? | Failure modes |
| ------ | -------- | ------------- |
| **Yahoo** | Yes (`config.yaml`, algotrader CLI, Gradio) | Unofficial API, ~15 min delay, 1m ≈ 7 days, split-adjustment required (`auto_adjust: true`) |
| **Alpaca** | Optional | Auth, feed, rate limits |
| **CSV** | Replay | Missing path or OHLCV columns |
| **Synthetic** | Tests / `--source synthetic` | Not tradable edge |

`agentic_ai_system.data_ingestion.load_data` dispatches on `data_source.type`. Yahoo stream: `yahoo_data_stream.py` (clamped lookback, no incomplete bars by default).

* `StrategyAgent`: SMA, RSI, Bollinger, MACD on Close (teaching rule, not an alpha claim)
* `FinRLAgent`: PPO / A2C / DDPG / TD3 via Stable-Baselines3
* `ExecutionAgent` / `AlpacaBroker`: paper simulation or Alpaca orders

v1 `run_backtest` is a single in-sample pass unless you use algotrader walk-forward. Leakage is the null hypothesis.

---

## 5. Stack

| Layer | Tools |
| ----- | ----- |
| Language | Python 3.11 (CI) |
| Validation | algotrader (permutation, DSR, PBO, walk-forward) |
| RL | FinRL / Stable-Baselines3, Gym/Gymnasium, PyTorch |
| Market data | yfinance ≥ 1.0 (default); alpaca-py optional |
| Tabular | pandas, NumPy, scikit-learn |
| UI | Gradio (`app.py`); Streamlit, Dash, Jupyter (v1) |
| Deploy | Docker Compose, GitHub Actions, Hugging Face Space |
| Tests | pytest |

---

## 6. Structure

```
algorithmic_trading/
├── algotrader/                 # 2.0 lab, engine, validation, strategies
├── app.py                      # Gradio Reality Check
├── agentic_ai_system/          # v1 FinRL, Yahoo/Alpaca ingest, execution
├── ui/                         # Streamlit, Dash, Jupyter, WebSocket
├── tests/
├── docs/AGENTIC_SYSTEM_V1.md   # v1 notes
├── config.yaml                 # default data_source.type: yahoo
├── requirements-space.txt      # Space / algotrader
├── requirements.txt            # full v1 + CI
└── scripts/deploy_hf_space.sh
```

---

## 7. Configuration

| Key | Meaning |
| --- | ------- |
| `data_source.type` | `yahoo` (default) \| `csv` \| `synthetic` \| `alpaca` |
| `trading.timeframe` | Mapped to Yahoo intervals; use `1d` for multi-year history |
| `yahoo.auto_adjust` | Split/dividend adjust (keep true) |
| `yahoo.emit_incomplete_bars` | Default false; forming bars are not closes |
| `execution.broker_api` | `paper` \| `alpaca_paper` \| `alpaca_live` |
| `finrl.algorithm` | PPO, A2C, DDPG, TD3 |
| algotrader `--source` | `yahoo` (default) \| `auto` \| `cache` \| `synthetic` |

---

## 8. Tests and ops

```bash
python -m pytest tests/test_v2_*.py -q
python -m pytest tests/test_yahoo_data_stream.py tests/test_data_ingestion.py -q
```

UI launchers and Docker: `UI_SETUP.md`, `DOCKER_HUB_SETUP.md`. Branch policy: `main` and `dev` only. Do not re-enable Dependabot.

---

**License:** Apache License 2.0  
**Organization:** [Parallel LLC](https://github.com/ParallelLLC)  
**Repository:** <https://github.com/ParallelLLC/algorithmic_trading>
