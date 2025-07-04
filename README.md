# Algorithmic Trading System

A comprehensive algorithmic trading system with synthetic data generation, comprehensive logging, extensive testing capabilities, FinRL reinforcement learning integration, and full Docker support.

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

### Docker Integration
- **Multi-Environment Support**: Development, production, and testing environments
- **Container Orchestration**: Docker Compose for easy service management
- **Monitoring Stack**: Prometheus and Grafana for system monitoring
- **Development Tools**: Jupyter Lab integration for interactive development
- **Automated Testing**: Containerized test execution with coverage reporting
- **Resource Management**: CPU and memory limits for production deployment
- **Health Checks**: Built-in health monitoring for all services
- **Backup Services**: Automated backup and data persistence

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

### Option 1: Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/ParallelLLC/algorithmic_trading.git
cd algorithmic_trading
```

2. Build and run with Docker:
```bash
# Build the image
docker build -t algorithmic-trading .

# Run the trading system
docker run -p 8000:8000 algorithmic-trading
```

### Option 2: Local Installation

1. Clone the repository:
```bash
git clone https://huggingface.co/esalguero/algorithmic_trading
cd algorithmic_trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Docker Usage

### Quick Start

```bash
# Build and start development environment
./scripts/docker-build.sh dev

# Build and start production environment
./scripts/docker-build.sh prod

# Run tests in Docker
./scripts/docker-build.sh test

# Stop all containers
./scripts/docker-build.sh stop
```

### Development Environment

```bash
# Start development environment with Jupyter Lab
docker-compose -f docker-compose.dev.yml up -d

# Access services:
# - Jupyter Lab: http://localhost:8888
# - Trading System: http://localhost:8000
# - TensorBoard: http://localhost:6006
```

### Production Environment

```bash
# Start production environment with monitoring
docker-compose -f docker-compose.prod.yml up -d

# Access services:
# - Trading System: http://localhost:8000
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Custom Commands

```bash
# Run a specific command in the container
./scripts/docker-build.sh run 'python demo.py'

# Run FinRL training
./scripts/docker-build.sh run 'python finrl_demo.py'

# Run backtesting
./scripts/docker-build.sh run 'python -m agentic_ai_system.main --mode backtest'

# Show logs
./scripts/docker-build.sh logs trading-system
```

### Docker Compose Services

#### Development (`docker-compose.dev.yml`)
- **trading-dev**: Jupyter Lab environment with hot reload
- **finrl-training-dev**: FinRL training with TensorBoard
- **testing**: Automated test execution
- **linting**: Code quality checks

#### Production (`docker-compose.prod.yml`)
- **trading-system**: Main trading system with resource limits
- **monitoring**: Prometheus metrics collection
- **grafana**: Data visualization dashboard
- **backup**: Automated backup service

#### Standard (`docker-compose.yml`)
- **trading-system**: Basic trading system
- **finrl-training**: FinRL training service
- **backtesting**: Backtesting service
- **development**: Development environment

### Docker Features

#### Health Checks
All services include health checks to ensure system reliability:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

#### Resource Management
Production services include resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

#### Volume Management
Persistent data storage with named volumes:
- `trading_data`: Market data and configuration
- `trading_logs`: System logs
- `trading_models`: Trained models
- `prometheus_data`: Monitoring metrics
- `grafana_data`: Dashboard configurations

#### Logging
Structured logging with rotation:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
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

# FinRL configuration
finrl:
  algorithm: 'PPO'
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

### Docker Testing
```bash
# Run all tests in Docker
./scripts/docker-build.sh test

# Run tests with coverage
docker run --rm -v $(pwd):/app algorithmic-trading:latest pytest --cov=agentic_ai_system --cov-report=html
```

## System Architecture

### Components

1. **SyntheticDataGenerator**: Generates realistic market data for testing
2. **DataIngestion**: Loads and validates market data from various sources
3. **StrategyAgent**: Analyzes market data and generates trading signals
4. **ExecutionAgent**: Executes trading orders with broker simulation
5. **Orchestrator**: Coordinates the entire trading workflow
6. **LoggerConfig**: Manages comprehensive logging throughout the system
7. **FinRLAgent**: Reinforcement learning agent for advanced trading strategies

### Data Flow

```
Synthetic Data Generator → Data Ingestion → Strategy Agent → Execution Agent
                              ↓
                         Logging System
                              ↓
                    FinRL Agent (Optional)
```

### Docker Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Development   │    │   Production    │    │    Monitoring   │
│   Environment   │    │   Environment   │    │     Stack       │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Jupyter Lab   │    │ • Trading Sys   │    │ • Prometheus    │
│ • Hot Reload    │    │ • Resource Mgmt │    │ • Grafana       │
│ • TensorBoard   │    │ • Health Checks │    │ • Metrics       │
│ • Testing       │    │ • Logging       │    │ • Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Monitoring and Observability

### Prometheus Metrics
- Trading performance metrics
- System resource usage
- Error rates and response times
- Custom business metrics

### Grafana Dashboards
- Real-time trading performance
- System health monitoring
- Historical data analysis
- Alert management

### Health Checks
- Service availability monitoring
- Dependency health verification
- Automatic restart on failure
- Performance degradation detection

## Deployment

### Local Development
```bash
# Start development environment
./scripts/docker-build.sh dev

# Access Jupyter Lab
open http://localhost:8888
```

### Production Deployment
```bash
# Deploy to production
./scripts/docker-build.sh prod

# Monitor system health
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Cloud Deployment
The Docker setup is compatible with:
- **AWS ECS/Fargate**: For serverless container deployment
- **Google Cloud Run**: For scalable containerized applications
- **Azure Container Instances**: For managed container deployment
- **Kubernetes**: For orchestrated container management

### Environment Variables
```bash
# Development
LOG_LEVEL=DEBUG
PYTHONDONTWRITEBYTECODE=1

# Production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

## Troubleshooting

### Common Docker Issues

#### Build Failures
```bash
# Clean build cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t algorithmic-trading .
```

#### Container Startup Issues
```bash
# Check container logs
docker logs algorithmic-trading

# Check container status
docker ps -a
```

#### Volume Mount Issues
```bash
# Check volume permissions
docker run --rm -v $(pwd):/app algorithmic-trading:latest ls -la /app

# Fix volume permissions
chmod -R 755 data logs models
```

### Performance Optimization

#### Resource Tuning
```yaml
# Adjust resource limits in docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 4G  # Increase for heavy workloads
      cpus: '2.0' # Increase for CPU-intensive tasks
```

#### Logging Optimization
```yaml
# Reduce log verbosity in production
logging:
  driver: "json-file"
  options:
    max-size: "5m"   # Smaller log files
    max-file: "2"    # Fewer log files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (including Docker tests)
5. Submit a pull request

### Development Workflow
```bash
# Start development environment
./scripts/docker-build.sh dev

# Make changes and test
./scripts/docker-build.sh test

# Run linting
docker-compose -f docker-compose.dev.yml run linting

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature-branch
```

## License

This project is licensed under the Apache License, Version 2.0 - see the LICENSE file for details.

## About

A comprehensive, production-ready algorithmic trading system with real-time market data streaming, multi-symbol trading, advanced technical analysis, robust risk management capabilities, and full Docker containerization support.

[Medium Article](https://medium.com/@edwinsalguero/data-pipeline-design-in-an-algorithmic-trading-system-ac0d8109c4b9) 
