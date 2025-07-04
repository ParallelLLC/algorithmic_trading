# Algorithmic Trading System

A comprehensive algorithmic trading system with synthetic data generation, comprehensive logging, extensive testing capabilities, and FinRL reinforcement learning integration.

## Features

### Core Trading System
- **Agent-based Architecture**: Modular design with separate strategy and execution agents
- **Technical Analysis**: Built-in technical indicators (SMA, RSI, Bollinger Bands, MACD)
- **Risk Management**: Position sizing and drawdown limits
- **Order Execution**: Simulated broker integration with realistic execution delays

### FinRL Reinforcement Learning
- **Multiple RL Algorithms**: Support for PPO, A2C, DDPG, and TD3
- **Custom Trading Environment**: Gymnasium-compatible environment for RL training
- **Technical Indicators Integration**: Automatic calculation and inclusion of technical indicators
- **Portfolio Management**: Realistic portfolio simulation with transaction costs
- **Model Persistence**: Save and load trained models for inference
- **TensorBoard Integration**: Training progress visualization and monitoring
- **Comprehensive Evaluation**: Performance metrics including Sharpe ratio and total returns

### Synthetic Data Generation
- **Realistic Market Data**: Generate OHLCV data using geometric Brownian motion
- **Multiple Frequencies**: Support for 1min, 5min, 1H, and 1D data
- **Market Scenarios**: Normal, volatile, trending, and crash market conditions
- **Tick Data**: High-frequency tick data generation for testing
- **Configurable Parameters**: Volatility, trend, noise levels, and base prices

### Comprehensive Logging
- **Multi-level Logging**: Console and file-based logging
- **Rotating Log Files**: Automatic log rotation with size limits
- **Specialized Loggers**: Separate loggers for trading, performance, and errors
- **Structured Logging**: Detailed log messages with timestamps and context

### Testing Framework
- **Unit Tests**: Comprehensive tests for all components
- **Integration Tests**: End-to-end workflow testing
- **Test Coverage**: Code coverage reporting with HTML and XML outputs
- **Mock Testing**: Isolated testing with mocked dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ParallelLLC/algorithmic_trading.git
cd algorithmic_trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system is configured via `config.yaml`:

```yaml
# Data source configuration
data_source:
  type: 'synthetic'  # or 'csv'
  path: 'data/market_data.csv'

# Trading parameters
trading:
  symbol: 'AAPL'
  timeframe: '1min'
  capital: 100000

# Risk management
risk:
  max_position: 100
  max_drawdown: 0.05

# Order execution
execution:
  broker_api: 'paper'
  order_size: 10
  delay_ms: 100
  success_rate: 0.95

# Synthetic data generation
synthetic_data:
  base_price: 150.0
  volatility: 0.02
  trend: 0.001
  noise_level: 0.005
  generate_data: true
  data_path: 'data/synthetic_market_data.csv'

# Logging configuration
logging:
  log_level: 'INFO'
  log_dir: 'logs'
  enable_console: true
  enable_file: true
  max_file_size_mb: 10
  backup_count: 5
```

## Usage

### Standard Trading Mode
```bash
python -m agentic_ai_system.main
```

### Backtest Mode
```bash
python -m agentic_ai_system.main --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
```

### Live Trading Mode
```bash
python -m agentic_ai_system.main --mode live --duration 60
```

### Custom Configuration
```bash
python -m agentic_ai_system.main --config custom_config.yaml
```

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest -m unit
```

### Integration Tests Only
```bash
pytest -m integration
```

### With Coverage Report
```bash
pytest --cov=agentic_ai_system --cov-report=html
```

### Specific Test File
```bash
pytest tests/test_synthetic_data_generator.py
```

## System Architecture

### Components

1. **SyntheticDataGenerator**: Generates realistic market data for testing
2. **DataIngestion**: Loads and validates market data from various sources
3. **StrategyAgent**: Analyzes market data and generates trading signals
4. **ExecutionAgent**: Executes trading orders with broker simulation
5. **Orchestrator**: Coordinates the entire trading workflow
6. **LoggerConfig**: Manages comprehensive logging throughout the system

### Data Flow

```
Synthetic Data Generator → Data Ingestion → Strategy Agent → Execution Agent
                              ↓
                         Logging System
```

## Synthetic Data Generation

### Features
- **Geometric Brownian Motion**: Realistic price movement simulation
- **OHLCV Data**: Complete market data with open, high, low, close, and volume
- **Market Scenarios**: Different market conditions for testing
- **Configurable Parameters**: Adjustable volatility, trend, and noise levels

### Usage Examples

```python
from agentic_ai_system.synthetic_data_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(config)

# Generate OHLCV data
data = generator.generate_ohlcv_data(
    symbol='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-31',
    frequency='1min'
)

# Generate tick data
tick_data = generator.generate_tick_data(
    symbol='AAPL',
    duration_minutes=60,
    tick_interval_ms=1000
)

# Generate market scenarios
crash_data = generator.generate_market_scenarios('crash')
volatile_data = generator.generate_market_scenarios('volatile')
```

## Logging System

### Log Files
- `logs/trading_system.log`: General system logs
- `logs/trading.log`: Trading-specific logs
- `logs/performance.log`: Performance metrics
- `logs/errors.log`: Error logs

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about system operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical system failures

### Usage Examples

```python
import logging
from agentic_ai_system.logger_config import setup_logging, get_logger

# Setup logging
setup_logging(config)

# Get logger for specific module
logger = get_logger(__name__)

# Log messages
logger.info("Trading signal generated")
logger.warning("High volatility detected")
logger.error("Order execution failed", exc_info=True)
```

## FinRL Integration

### Overview
The system now includes FinRL (Financial Reinforcement Learning) integration, providing state-of-the-art reinforcement learning capabilities for algorithmic trading. The FinRL agent can learn optimal trading strategies through interaction with a simulated market environment.

### Supported Algorithms
- **PPO (Proximal Policy Optimization)**: Stable policy gradient method
- **A2C (Advantage Actor-Critic)**: Actor-critic method with advantage estimation
- **DDPG (Deep Deterministic Policy Gradient)**: Continuous action space algorithm
- **TD3 (Twin Delayed DDPG)**: Improved version of DDPG with twin critics

### Trading Environment
The custom trading environment provides:
- **Action Space**: Discrete actions (0=Buy, 1=Hold, 2=Sell)
- **Observation Space**: OHLCV data + technical indicators + portfolio state
- **Reward Function**: Portfolio return-based rewards
- **Transaction Costs**: Realistic trading fees and slippage
- **Position Limits**: Maximum position constraints

### Usage Examples

#### Basic FinRL Training
```python
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
import pandas as pd

# Create configuration
config = FinRLConfig(
    algorithm="PPO",
    learning_rate=0.0003,
    batch_size=64,
    total_timesteps=100000
)

# Initialize agent
agent = FinRLAgent(config)

# Train the agent
training_result = agent.train(
    data=market_data,
    total_timesteps=100000,
    eval_freq=10000
)

# Generate predictions
predictions = agent.predict(test_data)

# Evaluate performance
evaluation = agent.evaluate(test_data)
print(f"Total Return: {evaluation['total_return']:.2%}")
```

#### Using Configuration File
```python
from agentic_ai_system.finrl_agent import create_finrl_agent_from_config

# Create agent from config file
agent = create_finrl_agent_from_config('config.yaml')

# Train and evaluate
agent.train(market_data)
results = agent.evaluate(test_data)
```

#### Running FinRL Demo
```bash
# Run the complete FinRL demo
python finrl_demo.py

# This will:
# 1. Generate synthetic training and test data
# 2. Train a FinRL agent
# 3. Evaluate performance
# 4. Generate trading predictions
# 5. Create visualization plots
```

### Configuration
FinRL settings can be configured in `config.yaml`:

```yaml
finrl:
  algorithm: 'PPO'  # PPO, A2C, DDPG, TD3
  learning_rate: 0.0003
  batch_size: 64
  buffer_size: 1000000
  gamma: 0.99
  tensorboard_log: 'logs/finrl_tensorboard'
  training:
    total_timesteps: 100000
    eval_freq: 10000
    save_best_model: true
    model_save_path: 'models/finrl_best/'
  inference:
    use_trained_model: false
    model_path: 'models/finrl_best/best_model'
```

### Model Management
```python
# Save trained model
agent.save_model('models/my_finrl_model')

# Load pre-trained model
agent.load_model('models/my_finrl_model')

# Continue training
agent.train(more_data, total_timesteps=50000)
```

### Performance Monitoring
- **TensorBoard Integration**: Monitor training progress
- **Evaluation Metrics**: Total return, Sharpe ratio, portfolio value
- **Trading Statistics**: Buy/sell signal analysis
- **Visualization**: Price charts with trading signals

### Advanced Features
- **Multi-timeframe Support**: Train on different data frequencies
- **Feature Engineering**: Automatic technical indicator calculation
- **Risk Management**: Built-in position and drawdown limits
- **Backtesting**: Comprehensive backtesting capabilities
- **Hyperparameter Tuning**: Easy configuration for different algorithms

## Testing

### Test Structure
```
tests/
├── __init__.py
├── test_synthetic_data_generator.py
├── test_strategy_agent.py
├── test_execution_agent.py
├── test_data_ingestion.py
├── test_integration.py
├── test_finrl_agent.py
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Test system performance and scalability
- **Error Handling Tests**: Test error conditions and edge cases
- **Slow RL Tests**: RL agent training tests are marked as `@pytest.mark.slow` and use minimal timesteps for speed. These are skipped by default unless explicitly run.

### Running Specific Tests

```bash
# Run all fast tests (default)
pytest

# Run slow RL tests (FinRL agent training)
pytest -m slow

# Run tests with coverage
pytest --cov=agentic_ai_system --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v
```

## Performance Monitoring

The system includes comprehensive performance monitoring:

- **Execution Time Tracking**: Monitor workflow execution times
- **Trade Statistics**: Track successful vs failed trades
- **Performance Metrics**: Calculate returns and drawdowns
- **Resource Usage**: Monitor memory and CPU usage

## Error Handling

The system includes robust error handling:

- **Graceful Degradation**: System continues operation despite component failures
- **Error Logging**: Comprehensive error logging with stack traces
- **Fallback Mechanisms**: Automatic fallback to synthetic data when CSV files are missing
- **Validation**: Data validation at multiple levels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details. 
