#!/usr/bin/env python3
"""
Demonstration script for the Algorithmic Trading System

This script demonstrates the key features of the system:
- Synthetic data generation
- Trading workflow execution
- Backtesting
- Logging
"""

import yaml
import pandas as pd
from agentic_ai_system.synthetic_data_generator import SyntheticDataGenerator
from agentic_ai_system.logger_config import setup_logging
from agentic_ai_system.orchestrator import run, run_backtest
from agentic_ai_system.main import load_config

def main():
    """Main demonstration function"""
    print("🚀 Algorithmic Trading System Demo")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Setup logging
    setup_logging(config)
    print("✅ Logging system initialized")
    
    # Demo 1: Synthetic Data Generation
    print("\n📊 Demo 1: Synthetic Data Generation")
    print("-" * 30)
    
    try:
        generator = SyntheticDataGenerator(config)
        
        # Generate OHLCV data
        print("Generating OHLCV data...")
        ohlcv_data = generator.generate_ohlcv_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-02',
            frequency='1H'
        )
        print(f"✅ Generated {len(ohlcv_data)} OHLCV data points")
        
        # Show sample data
        print("\nSample OHLCV data:")
        print(ohlcv_data.head())
        
        # Generate different market scenarios
        print("\nGenerating market scenarios...")
        scenarios = ['normal', 'volatile', 'trending', 'crash']
        
        for scenario in scenarios:
            scenario_data = generator.generate_market_scenarios(scenario)
            avg_price = scenario_data['close'].mean()
            print(f"  {scenario.capitalize()} market: {len(scenario_data)} points, avg price: ${avg_price:.2f}")
        
    except Exception as e:
        print(f"❌ Error in synthetic data generation: {e}")
    
    # Demo 2: Trading Workflow
    print("\n🤖 Demo 2: Trading Workflow")
    print("-" * 30)
    
    try:
        print("Running trading workflow...")
        result = run(config)
        
        if result['success']:
            print("✅ Trading workflow completed successfully")
            print(f"  Data loaded: {result['data_loaded']}")
            print(f"  Signal generated: {result['signal_generated']}")
            print(f"  Order executed: {result['order_executed']}")
            print(f"  Execution time: {result['execution_time']:.2f} seconds")
            
            if result['order_executed'] and result['execution_result']:
                exec_result = result['execution_result']
                print(f"  Order details: {exec_result['action']} {exec_result['quantity']} {exec_result['symbol']} @ ${exec_result['price']:.2f}")
        else:
            print("❌ Trading workflow failed")
            print(f"  Errors: {result['errors']}")
            
    except Exception as e:
        print(f"❌ Error in trading workflow: {e}")
    
    # Demo 3: Backtesting
    print("\n📈 Demo 3: Backtesting")
    print("-" * 30)
    
    try:
        print("Running backtest...")
        backtest_result = run_backtest(config, '2024-01-01', '2024-01-07')
        
        if backtest_result['success']:
            print("✅ Backtest completed successfully")
            print(f"  Initial capital: ${backtest_result['initial_capital']:,.2f}")
            print(f"  Final value: ${backtest_result['final_value']:,.2f}")
            print(f"  Total return: {backtest_result['total_return']:.2%}")
            print(f"  Total trades: {backtest_result['total_trades']}")
            print(f"  Positions: {backtest_result['positions']}")
        else:
            print("❌ Backtest failed")
            print(f"  Error: {backtest_result['error']}")
            
    except Exception as e:
        print(f"❌ Error in backtesting: {e}")
    
    # Demo 4: System Statistics
    print("\n📊 Demo 4: System Statistics")
    print("-" * 30)
    
    try:
        # Show configuration summary
        print("Configuration Summary:")
        print(f"  Trading symbol: {config['trading']['symbol']}")
        print(f"  Timeframe: {config['trading']['timeframe']}")
        print(f"  Capital: ${config['trading']['capital']:,.2f}")
        print(f"  Max position: {config['risk']['max_position']}")
        print(f"  Max drawdown: {config['risk']['max_drawdown']:.1%}")
        print(f"  Broker API: {config['execution']['broker_api']}")
        
        # Show synthetic data parameters
        print("\nSynthetic Data Parameters:")
        print(f"  Base price: ${config['synthetic_data']['base_price']:.2f}")
        print(f"  Volatility: {config['synthetic_data']['volatility']:.3f}")
        print(f"  Trend: {config['synthetic_data']['trend']:.3f}")
        print(f"  Noise level: {config['synthetic_data']['noise_level']:.3f}")
        
    except Exception as e:
        print(f"❌ Error showing statistics: {e}")
    
    print("\n🎉 Demo completed!")
    print("\n📝 Check the logs directory for detailed logs:")
    print("  - logs/trading_system.log")
    print("  - logs/trading.log")
    print("  - logs/performance.log")
    print("  - logs/errors.log")

if __name__ == '__main__':
    main() 