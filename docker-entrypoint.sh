#!/bin/bash
set -e

# Function to wait for dependencies
wait_for_dependencies() {
    echo "Waiting for dependencies to be ready..."
    sleep 5
}

# Function to initialize directories
init_directories() {
    echo "Initializing directories..."
    mkdir -p /app/data
    mkdir -p /app/logs
    mkdir -p /app/models
    chmod 755 /app/data /app/logs /app/models
}

# Function to generate synthetic data if needed
generate_data_if_needed() {
    if [ ! -f "/app/data/synthetic_market_data.csv" ]; then
        echo "Generating synthetic market data..."
        python -c "
from agentic_ai_system.synthetic_data_generator import SyntheticDataGenerator
import yaml

with open('/app/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

generator = SyntheticDataGenerator(config)
data = generator.generate_ohlcv_data(
    symbol='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-31',
    frequency='1min'
)
data.to_csv('/app/data/synthetic_market_data.csv', index=True)
print('Synthetic data generated successfully')
"
    else
        echo "Synthetic data already exists"
    fi
}

# Function to run health check
health_check() {
    echo "Running health check..."
    python -c "
import sys
from agentic_ai_system.logger_config import setup_logging
try:
    setup_logging({})
    print('Health check passed')
except Exception as e:
    print(f'Health check failed: {e}')
    sys.exit(1)
"
}

# Main execution
main() {
    echo "Starting Algorithmic Trading System..."
    
    # Initialize
    init_directories
    wait_for_dependencies
    generate_data_if_needed
    health_check
    
    echo "System initialized successfully"
    
    # Execute the main command
    exec "$@"
}

# Run main function with all arguments
main "$@" 