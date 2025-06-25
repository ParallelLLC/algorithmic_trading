# Algorithmic Trading System

A comprehensive algorithmic trading system with synthetic data generation, comprehensive logging, and extensive testing capabilities.

## Features

### Core Trading System
- **Agent-based Architecture**: Modular design with separate strategy and execution agents
- **Technical Analysis**: Built-in technical indicators (SMA, RSI, Bollinger Bands, MACD)
- **Risk Management**: Position sizing and drawdown limits
- **Order Execution**: Simulated broker integration with realistic execution delays

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
git clone <repository-url>
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

## Testing

### Test Structure
```
tests/
├── __init__.py
├── test_synthetic_data_generator.py
├── test_strategy_agent.py
├── test_execution_agent.py
├── test_data_ingestion.py
└── test_integration.py
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows
- **Performance Tests**: Test system performance and scalability
- **Error Handling Tests**: Test error conditions and edge cases

### Running Specific Tests

```bash
# Run tests with specific markers
pytest -m unit
pytest -m integration
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

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a simulation system for educational and testing purposes. It is not intended for real trading and should not be used with real money. Always test thoroughly before using any trading system with real funds. 