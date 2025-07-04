---
language: code
license: apache-2.0
tags:
- algorithmic-trading
- reinforcement-learning
- finrl
- trading-bot
- machine-learning
- finance
- quantitative-finance
- backtesting
- risk-management
- technical-analysis
- docker
- python
datasets:
- synthetic-market-data
metrics:
- sharpe-ratio
- total-return
- drawdown
- win-rate
library_name: algorithmic-trading
paperswithcode_id: null
---

# Algorithmic Trading System with FinRL and Alpaca Integration

A sophisticated algorithmic trading system that combines reinforcement learning (FinRL) with real-time market data and order execution through Alpaca Markets. This system supports both paper trading and live trading with advanced risk management and technical analysis.

## 🚀 Features

### Core Trading System
- **Multi-source Data Ingestion**: CSV files, Alpaca Markets API, and synthetic data generation
- **Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Risk Management**: Position sizing, drawdown limits, and portfolio protection
- **Real-time Execution**: Live order placement and portfolio monitoring

### FinRL Reinforcement Learning
- **Multiple Algorithms**: PPO, A2C, DDPG, and TD3 support
- **Custom Trading Environment**: Gymnasium-compatible environment for RL training
- **Real-time Integration**: Can execute real trades during training and inference
- **Model Persistence**: Save and load trained models for consistent performance

### Alpaca Broker Integration
- **Paper Trading**: Risk-free testing with virtual money
- **Live Trading**: Real market execution (use with caution!)
- **Market Data**: Real-time and historical data from Alpaca
- **Account Management**: Portfolio monitoring and position tracking
- **Order Types**: Market orders, limit orders, and order cancellation

### Advanced Features
- **Docker Support**: Containerized deployment for consistency
- **Comprehensive Logging**: Detailed logs for debugging and performance analysis
- **Backtesting Engine**: Historical performance evaluation
- **Live Trading Simulation**: Real-time trading with configurable duration
- **Performance Metrics**: Returns, Sharpe ratio, drawdown analysis

## 📋 Prerequisites

- Python 3.8+
- Alpaca Markets account (free paper trading available)
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd algorithmic_trading
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Alpaca API Credentials
Create a `.env` file in the project root:
```bash
cp env.example .env
```

Edit `.env` with your Alpaca credentials:
```env
# Get these from https://app.alpaca.markets/paper/dashboard/overview
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here

# For live trading (use with caution!)
# ALPACA_API_KEY=your_live_api_key_here
# ALPACA_SECRET_KEY=your_live_secret_key_here
```

### 4. Configure Trading Parameters
Edit `config.yaml` to customize your trading strategy:
```yaml
# Data source configuration
data_source:
  type: 'alpaca'  # Options: 'alpaca', 'csv', 'synthetic'

# Trading parameters
trading:
  symbol: 'AAPL'
  timeframe: '1m'
  capital: 100000

# Risk management
risk:
  max_position: 100
  max_drawdown: 0.05

# Execution settings
execution:
  broker_api: 'alpaca_paper'  # Options: 'paper', 'alpaca_paper', 'alpaca_live'
  order_size: 10

# FinRL configuration
finrl:
  algorithm: 'PPO'
  learning_rate: 0.0003
  training:
    total_timesteps: 100000
    save_best_model: true
```

## 🚀 Quick Start

### 1. Run the Demo
```bash
python demo.py
```

This will:
- Test data ingestion from Alpaca
- Demonstrate FinRL training
- Show trading workflow execution
- Run backtesting on historical data

### 2. Start Paper Trading
```bash
python -m agentic_ai_system.main --mode live --duration 60
```

### 3. Run Backtesting
```bash
python -m agentic_ai_system.main --mode backtest --start-date 2024-01-01 --end-date 2024-01-31
```

## 📊 Usage Examples

### Basic Trading Workflow
```python
from agentic_ai_system.main import load_config
from agentic_ai_system.orchestrator import run

# Load configuration
config = load_config()

# Run single trading cycle
result = run(config)
print(f"Trading result: {result}")
```

### FinRL Training
```python
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
from agentic_ai_system.data_ingestion import load_data

# Load data and configuration
config = load_config()
data = load_data(config)

# Initialize FinRL agent
finrl_config = FinRLConfig(algorithm='PPO', learning_rate=0.0003)
agent = FinRLAgent(finrl_config)

# Train the agent
result = agent.train(
    data=data,
    config=config,
    total_timesteps=100000,
    use_real_broker=False  # Use simulation for training
)

print(f"Training completed: {result}")
```

### Alpaca Integration
```python
from agentic_ai_system.alpaca_broker import AlpacaBroker

# Initialize Alpaca broker
config = load_config()
broker = AlpacaBroker(config)

# Get account information
account_info = broker.get_account_info()
print(f"Account balance: ${account_info['buying_power']:,.2f}")

# Place a market order
result = broker.place_market_order(
    symbol='AAPL',
    quantity=10,
    side='buy'
)
print(f"Order result: {result}")
```

### Real-time Trading with FinRL
```python
from agentic_ai_system.finrl_agent import FinRLAgent

# Load trained model
agent = FinRLAgent(FinRLConfig())
agent.model = agent._load_model('models/finrl_best/best_model', config)

# Make predictions with real execution
result = agent.predict(
    data=recent_data,
    config=config,
    use_real_broker=True  # Execute real trades!
)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Strategy Agent │    │ Execution Agent │
│                 │    │                 │    │                 │
│ • Alpaca API    │───▶│ • Technical     │───▶│ • Alpaca Broker │
│ • CSV Files     │    │   Indicators    │    │ • Order Mgmt    │
│ • Synthetic     │    │ • Signal Gen    │    │ • Risk Control  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │    │   FinRL Agent   │    │   Portfolio     │
│                 │    │                 │    │   Management    │
│ • Validation    │    │ • PPO/A2C/DDPG  │    │ • Positions     │
│ • Indicators    │    │ • Training      │    │ • P&L Tracking  │
│ • Preprocessing │    │ • Prediction    │    │ • Risk Metrics  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Data Ingestion**: Market data from Alpaca, CSV, or synthetic sources
2. **Preprocessing**: Technical indicators, data validation, and feature engineering
3. **Strategy Generation**: Traditional technical analysis or FinRL predictions
4. **Risk Management**: Position sizing and portfolio protection
5. **Order Execution**: Real-time order placement through Alpaca
6. **Performance Tracking**: Continuous monitoring and logging

## 🔧 Configuration

### Alpaca Settings
```yaml
alpaca:
  api_key: ''  # Set via environment variable
  secret_key: ''  # Set via environment variable
  paper_trading: true
  base_url: 'https://paper-api.alpaca.markets'
  live_url: 'https://api.alpaca.markets'
  data_url: 'https://data.alpaca.markets'
  account_type: 'paper'  # 'paper' or 'live'
```

### FinRL Settings
```yaml
finrl:
  algorithm: 'PPO'  # PPO, A2C, DDPG, TD3
  learning_rate: 0.0003
  batch_size: 64
  buffer_size: 1000000
  training:
    total_timesteps: 100000
    eval_freq: 10000
    save_best_model: true
    model_save_path: 'models/finrl_best/'
  inference:
    use_trained_model: false
    model_path: 'models/finrl_best/best_model'
```

### Risk Management
```yaml
risk:
  max_position: 100
  max_drawdown: 0.05
  stop_loss: 0.02
  take_profit: 0.05
```

## 📈 Performance Monitoring

### Logging
The system provides comprehensive logging:
- `logs/trading_system.log`: Main system logs
- `logs/trading.log`: Trading-specific events
- `logs/performance.log`: Performance metrics
- `logs/finrl_tensorboard/`: FinRL training logs

### Metrics Tracked
- Portfolio value and returns
- Trade execution statistics
- Risk metrics (Sharpe ratio, drawdown)
- FinRL training progress
- Alpaca account status

### Real-time Monitoring
```python
# Get account information
account_info = broker.get_account_info()
print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")

# Get current positions
positions = broker.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} shares")

# Check market status
market_open = broker.is_market_open()
print(f"Market: {'OPEN' if market_open else 'CLOSED'}")
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the image
docker build -t algorithmic-trading .

# Run with environment variables
docker run -it --env-file .env algorithmic-trading

# Run with Jupyter Lab for development
docker-compose -f docker-compose.dev.yml up
```

### Production Deployment
```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Monitor logs
docker-compose -f docker-compose.prod.yml logs -f
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Specific Components
```bash
# Test Alpaca integration
pytest tests/test_alpaca_integration.py -v

# Test FinRL agent
pytest tests/test_finrl_agent.py -v

# Test trading workflow
pytest tests/test_integration.py -v
```

## ⚠️ Important Notes

### Paper Trading vs Live Trading
- **Paper Trading**: Uses virtual money, safe for testing
- **Live Trading**: Uses real money, use with extreme caution
- Always test strategies thoroughly in paper trading before going live

### Risk Management
- Set appropriate position limits and drawdown thresholds
- Monitor your portfolio regularly
- Use stop-loss orders to limit potential losses
- Never risk more than you can afford to lose

### API Rate Limits
- Alpaca has rate limits on API calls
- The system includes built-in delays to respect these limits
- Monitor your API usage in the Alpaca dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the logs and configuration files
- **Issues**: Report bugs and feature requests on GitHub
- **Alpaca Support**: Contact Alpaca for API-related issues
- **Community**: Join our Discord/Telegram for discussions

## 🔗 Useful Links

- [Alpaca Markets Documentation](https://alpaca.markets/docs/)
- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Disclaimer**: This software is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a financial advisor before making investment decisions. 
