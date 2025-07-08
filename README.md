---
title: Algorithmic Trading System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.46.1
app_file: streamlit_app.py
pinned: false
license: apache-2.0
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

### 🎨 Comprehensive UI System

- **Streamlit UI**: Quick prototyping and data science workflows
- **Dash UI**: Enterprise-grade interactive dashboards
- **Jupyter UI**: Interactive notebook-based interfaces
- **WebSocket API**: Real-time trading data streaming
- **Multi-interface Support**: Choose the right UI for your needs

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
git clone https://github.com/ParallelLLC/algorithmic_trading.git
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

### 1. Launch the UI (Recommended)

```bash
# Launch Streamlit UI (best for beginners)
python ui_launcher.py streamlit

# Launch Dash UI (best for production)
python ui_launcher.py dash

# Launch Jupyter Lab
python ui_launcher.py jupyter

# Launch all UIs
python ui_launcher.py all
```

### 2. Run the Demo

```bash
python demo.py
```

This will:
- Test data ingestion from Alpaca
- Demonstrate FinRL training
- Show trading workflow execution
- Run backtesting on historical data

### 3. Start Paper Trading

```bash
python -m agentic_ai_system.main --mode live --duration 60
```

### 4. Run Backtesting

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

## 📁 Project Directory Structure

```
algorithmic_trading/
├── 📄 README.md                    # Project documentation
├── 📄 LICENSE                      # Apache License 2.0
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.yaml                  # Main configuration file
├── 📄 env.example                  # Environment variables template
├── 📄 .gitignore                   # Git ignore rules
├── 📄 pytest.ini                  # Test configuration
│
├── 🐳 Docker/
│   ├── 📄 Dockerfile              # Container definition
│   ├── 📄 docker-entrypoint.sh    # Container startup script
│   ├── 📄 .dockerignore           # Docker ignore rules
│   ├── 📄 docker-compose.yml      # Default compose file
│   ├── 📄 docker-compose.dev.yml  # Development environment
│   ├── 📄 docker-compose.prod.yml # Production environment
│   ├── 📄 docker-compose.hub.yml  # Docker Hub deployment
│   └── 📄 docker-compose.prod.yml # Production environment
│
├── 🤖 agentic_ai_system/          # Core AI trading system
│   ├── 📄 main.py                 # Main entry point
│   ├── 📄 orchestrator.py         # System coordination
│   ├── 📄 agent_base.py           # Base agent class
│   ├── 📄 data_ingestion.py       # Market data processing
│   ├── 📄 strategy_agent.py       # Trading strategy logic
│   ├── 📄 execution_agent.py      # Order execution
│   ├── 📄 finrl_agent.py          # FinRL reinforcement learning
│   ├── 📄 alpaca_broker.py        # Alpaca API integration
│   ├── 📄 synthetic_data_generator.py # Test data generation
│   └── 📄 logger_config.py        # Logging configuration
│
├── 🎨 ui/                         # User interface system
│   ├── 📄 __init__.py            # UI package initialization
│   ├── 📄 streamlit_app.py       # Streamlit web application
│   ├── 📄 dash_app.py            # Dash enterprise dashboard
│   ├── 📄 jupyter_widgets.py     # Jupyter interactive widgets
│   └── 📄 websocket_server.py    # Real-time WebSocket server
│
├── 🧪 tests/                      # Test suite
│   ├── 📄 __init__.py
│   ├── 📄 test_data_ingestion.py
│   ├── 📄 test_strategy_agent.py
│   ├── 📄 test_execution_agent.py
│   ├── 📄 test_finrl_agent.py
│   ├── 📄 test_synthetic_data_generator.py
│   └── 📄 test_integration.py
│
├── 📊 data/                       # Market data storage
│   └── 📄 synthetic_market_data.csv
│
├── 🧠 models/                     # Trained AI models
│   └── 📁 finrl_best/            # Best FinRL models
│
├── 📈 plots/                      # Generated charts/visualizations
│
├── 📝 logs/                       # System logs
│   ├── 📄 trading_system.log
│   ├── 📄 trading.log
│   ├── 📄 performance.log
│   ├── 📄 errors.log
│   ├── 📁 finrl_tensorboard/     # FinRL training logs
│   └── 📁 finrl_eval/            # Model evaluation logs
│
├── 🔧 scripts/                    # Utility scripts
│   ├── 📄 docker-build.sh        # Docker build automation
│   └── 📄 docker-hub-deploy.sh   # Docker Hub deployment
│
├── 📄 demo.py                     # Main demo script
├── 📄 finrl_demo.py              # FinRL-specific demo
├── 📄 ui_launcher.py             # UI launcher script
├── 📄 UI_SETUP.md                # UI setup documentation
├── 📄 DOCKER_HUB_SETUP.md        # Docker Hub documentation
│
└── 🐍 .venv/                     # Python virtual environment
```

## 🏗️ Architecture Overview

### Core Components:

- **Data Layer**: Market data ingestion and preprocessing
- **Strategy Layer**: Technical analysis and signal generation
- **AI Layer**: FinRL reinforcement learning agents
- **Execution Layer**: Order management and broker integration
- **Orchestration**: System coordination and workflow management

### Key Features:

- **Modular Design**: Each component is independent and testable
- **Docker Support**: Complete containerization for deployment
- **Testing**: Comprehensive test suite for all components
- **Logging**: Detailed logging for monitoring and debugging
- **Configuration**: Centralized configuration management
- **Documentation**: Extensive documentation and examples

### Development Workflow:

1. **Data Ingestion** → Market data from Alpaca/CSV/synthetic sources
2. **Strategy Generation** → Technical indicators and FinRL predictions
3. **Risk Management** → Position sizing and portfolio protection
4. **Order Execution** → Real-time trading through Alpaca
5. **Performance Tracking** → Continuous monitoring and logging

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

## 🎨 User Interface System

The project includes a comprehensive UI system with multiple interface options:

### Available UIs

#### Streamlit UI (Recommended for beginners)
- **URL**: http://localhost:8501
- **Features**: Interactive widgets, real-time data visualization, easy configuration
- **Best for**: Data scientists, quick experiments, rapid prototyping

#### Dash UI (Recommended for production)
- **URL**: http://localhost:8050
- **Features**: Enterprise-grade dashboards, advanced charts, professional styling
- **Best for**: Production dashboards, real-time monitoring, complex analytics

#### Jupyter UI (For research)
- **URL**: http://localhost:8888
- **Features**: Interactive notebooks, code execution, rich documentation
- **Best for**: Research, experimentation, educational purposes

#### WebSocket API (For developers)
- **URL**: ws://localhost:8765
- **Features**: Real-time data streaming, trading signals, portfolio updates
- **Best for**: Real-time trading signals, live data streaming

### Quick UI Launch

```bash
# Launch individual UIs
python ui_launcher.py streamlit    # Streamlit UI
python ui_launcher.py dash         # Dash UI
python ui_launcher.py jupyter      # Jupyter Lab
python ui_launcher.py websocket    # WebSocket server

# Launch all UIs at once
python ui_launcher.py all
```

### UI Features

- **Real-time Data Visualization**: Live market data charts and indicators
- **Portfolio Monitoring**: Real-time portfolio value and P&L tracking
- **Trading Controls**: Start/stop trading, backtesting, risk management
- **FinRL Training**: Interactive model training and evaluation
- **Alpaca Integration**: Account management and order execution
- **Configuration Management**: Easy parameter tuning and strategy setup

For detailed UI documentation, see [UI_SETUP.md](UI_SETUP.md).

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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
