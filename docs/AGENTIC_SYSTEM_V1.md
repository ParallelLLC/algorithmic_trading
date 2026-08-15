# Algorithmic Trading

FinRL reinforcement-learning trading with Alpaca execution, plus Yahoo Finance OHLCV as the default public tape. Parallel LLC.

This is **research and paper-trading infrastructure**. Live capital requires a separate evaluation contract, feature-parity tests, and a rewritten execution path. Do not treat `paper_trading: false` as a promotion gate.

---

## 1. Title and Summary

**Algorithmic Trading**  
Ingest OHLCV, compute indicators or train a FinRL policy, size orders under position and drawdown caps, route to paper or live Alpaca.

GitHub `main` is the FinRL / Docker / Streamlit tree plus algotrader 2.0. `dev` is the integration branch. Yahoo is the default `data_source.type`.

**Design themes**

* Four ingest paths: CSV replay, synthetic GBM, Alpaca REST, Yahoo (`yfinance>=1.0`)
* FinRL policies (PPO, A2C, DDPG, TD3) on a Gymnasium-style environment
* Alpaca for authenticated market data and order routing (paper by default)
* Yahoo for delayed public bars when no broker key is available
* Secrets from environment (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`), never committed
* Tests and Docker/CI as already present on this tree

---

## 2. Concepts and Methods

### Market data

| Source | When to use | Failure modes |
| ------ | ----------- | ------------- |
| **CSV** | Offline replay; default in `config.yaml` | Missing path or OHLCV columns → `None` |
| **Synthetic** | Unit tests and demos | GBM is not tradable edge |
| **Alpaca** | Authenticated bars and live/paper orders | Auth, feed, and rate-limit failures |
| **Yahoo** | Real Close without a broker account | Unofficial API, ~15 min delay, interval lookback caps (1m ≈ 7 days). Pin `yfinance>=1.0`; 0.2.x fails against the current chart API |

`load_data` dispatches on `data_source.type`. Existing `alpaca` / `csv` / `synthetic` branches are unchanged.

### Strategy and FinRL

* `StrategyAgent`: SMA, RSI, Bollinger, MACD on Close; teaching rule, not an alpha claim
* `FinRLAgent`: PPO / A2C / DDPG / TD3 via Stable-Baselines3; persist under `models/`
* `ExecutionAgent` / `AlpacaBroker`: paper simulation or Alpaca market/limit orders

Backtests in this repo are in-sample passes unless you add a purged walk-forward yourself. Leakage is the null hypothesis.

---

## 3. Stack

| Layer | Tools |
| ----- | ----- |
| Language | Python 3.11 (CI); 3.8+ stated for local |
| RL | FinRL / Stable-Baselines3, Gym/Gymnasium, PyTorch |
| Broker | alpaca-py |
| Market data | Alpaca REST; yfinance ≥ 1.0 (Yahoo) |
| Tabular | pandas, NumPy, scikit-learn |
| UI | Streamlit, Dash, Jupyter widgets |
| Deploy | Docker Compose, GitHub Actions |
| Tests | pytest, pytest-cov |

---

## 4. Structure

```
algorithmic_trading/
├── agentic_ai_system/     # ingest, strategy, FinRL, Alpaca, Yahoo
├── ui/                    # Streamlit, Dash, Jupyter, WebSocket
├── tests/
├── models/                # trained artifacts (gitignored bodies)
├── data/                  # generated CSV (gitignored)
├── scripts/               # Docker / deploy helpers
├── .github/workflows/     # CI/CD, release, backtesting
├── config.yaml
├── requirements.txt
├── Dockerfile
└── docker-compose*.yml
```

Branch policy: **`main`** (protected) and **`dev`** only. Do not re-enable Dependabot or the Monday `dependency-updates` workflow; those created extra branches.

---

## 5. Quick start

```bash
git clone https://github.com/ParallelLLC/algorithmic_trading.git
cd algorithmic_trading
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Alpaca keys if using alpaca ingest or orders
```

Default ingest is CSV. For Yahoo daily bars without a broker:

```yaml
data_source:
  type: 'yahoo'
trading:
  symbol: 'AAPL'
  timeframe: '1d'
```

```bash
python demo.py
python -m agentic_ai_system.main --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
pytest tests/ -q
```

UI launchers and Docker are documented in `UI_SETUP.md` and `DOCKER_HUB_SETUP.md`. Paper-trade before live. Yahoo is not a SIP tape.

---

## 6. Configuration (additive Yahoo keys)

| Key | Meaning |
| --- | ------- |
| `data_source.type` | `csv` \| `synthetic` \| `alpaca` \| `yahoo` |
| `yahoo.start_date` / `end_date` | Historical window; clamped per Yahoo interval limits |
| `yahoo.auto_adjust` | Passed to `yfinance` |
| `execution.broker_api` | `paper` \| `alpaca_paper` \| `alpaca_live` |
| `finrl.algorithm` | PPO, A2C, DDPG, TD3 |

---

**License:** Apache License 2.0  
**Organization:** [Parallel LLC](https://github.com/ParallelLLC)  
**Repository:** <https://github.com/ParallelLLC/algorithmic_trading>
