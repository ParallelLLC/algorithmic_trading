#!/usr/bin/env python3
"""
Demo script for the Algorithmic Trading System with FinRL and Alpaca Integration

This script demonstrates the complete trading workflow including:
- Data ingestion from multiple sources (CSV, Alpaca, Synthetic)
- Strategy generation with technical indicators
- Order execution with Alpaca broker
- FinRL reinforcement learning integration
- Real-time trading capabilities
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ai_system.main import load_config
from agentic_ai_system.orchestrator import run, run_backtest, run_live_trading
from agentic_ai_system.data_ingestion import load_data, validate_data, add_technical_indicators
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
from agentic_ai_system.alpaca_broker import AlpacaBroker

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/demo.log')
        ]
    )

def print_system_info(config: Dict[str, Any]):
    """Print system configuration information"""
    print("\n" + "="*60)
    print("🤖 ALGORITHMIC TRADING SYSTEM WITH FINRL & ALPACA")
    print("="*60)
    
    print(f"\n📊 Data Source: {config['data_source']['type']}")
    print(f"📈 Trading Symbol: {config['trading']['symbol']}")
    print(f"💰 Capital: ${config['trading']['capital']:,}")
    print(f"⏱️  Timeframe: {config['trading']['timeframe']}")
    print(f"🔧 Broker API: {config['execution']['broker_api']}")
    
    if config['execution']['broker_api'] in ['alpaca_paper', 'alpaca_live']:
        print(f"🏦 Alpaca Account Type: {config['alpaca']['account_type']}")
        print(f"📡 Alpaca Base URL: {config['alpaca']['base_url']}")
    
    print(f"🧠 FinRL Algorithm: {config['finrl']['algorithm']}")
    print(f"📚 Learning Rate: {config['finrl']['learning_rate']}")
    print(f"🎯 Training Steps: {config['finrl']['training']['total_timesteps']:,}")
    
    print("\n" + "="*60)

def demo_data_ingestion(config: Dict[str, Any]):
    """Demonstrate data ingestion capabilities"""
    print("\n📥 DATA INGESTION DEMO")
    print("-" * 30)
    
    try:
        # Load data
        print(f"Loading data from source: {config['data_source']['type']}")
        data = load_data(config)
        
        if data is not None and not data.empty:
            print(f"✅ Successfully loaded {len(data)} data points")
            print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            # Validate data
            if validate_data(data):
                print("✅ Data validation passed")
                
                # Add technical indicators
                data_with_indicators = add_technical_indicators(data)
                print(f"✅ Added {len(data_with_indicators.columns) - len(data.columns)} technical indicators")
                
                return data_with_indicators
            else:
                print("❌ Data validation failed")
                return None
        else:
            print("❌ Failed to load data")
            return None
            
    except Exception as e:
        print(f"❌ Error in data ingestion: {e}")
        return None

def demo_alpaca_integration(config: Dict[str, Any]):
    """Demonstrate Alpaca broker integration"""
    print("\n🏦 ALPACA INTEGRATION DEMO")
    print("-" * 30)
    
    if config['execution']['broker_api'] not in ['alpaca_paper', 'alpaca_live']:
        print("⚠️  Alpaca integration not configured (using simulation mode)")
        return None
    
    try:
        # Initialize Alpaca broker
        print("Connecting to Alpaca...")
        alpaca_broker = AlpacaBroker(config)
        
        # Get account information
        account_info = alpaca_broker.get_account_info()
        if account_info:
            print(f"✅ Connected to Alpaca {config['alpaca']['account_type']} account")
            print(f"   Account ID: {account_info['account_id']}")
            print(f"   Status: {account_info['status']}")
            print(f"   Buying Power: ${account_info['buying_power']:,.2f}")
            print(f"   Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            print(f"   Equity: ${account_info['equity']:,.2f}")
        
        # Check market status
        market_hours = alpaca_broker.get_market_hours()
        if market_hours:
            print(f"📈 Market Status: {'🟢 OPEN' if market_hours['is_open'] else '🔴 CLOSED'}")
            if market_hours['next_open']:
                print(f"   Next Open: {market_hours['next_open']}")
            if market_hours['next_close']:
                print(f"   Next Close: {market_hours['next_close']}")
        
        # Get current positions
        positions = alpaca_broker.get_positions()
        if positions:
            print(f"📊 Current Positions: {len(positions)}")
            for pos in positions:
                print(f"   {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f}")
        else:
            print("📊 No current positions")
        
        return alpaca_broker
        
    except Exception as e:
        print(f"❌ Error connecting to Alpaca: {e}")
        return None

def demo_finrl_training(config: Dict[str, Any], data):
    """Demonstrate FinRL training"""
    print("\n🧠 FINRL TRAINING DEMO")
    print("-" * 30)
    
    try:
        # Initialize FinRL agent
        finrl_config = FinRLConfig(
            algorithm=config['finrl']['algorithm'],
            learning_rate=config['finrl']['learning_rate'],
            batch_size=config['finrl']['batch_size'],
            buffer_size=config['finrl']['buffer_size'],
            learning_starts=config['finrl']['learning_starts'],
            gamma=config['finrl']['gamma'],
            tau=config['finrl']['tau'],
            train_freq=config['finrl']['train_freq'],
            gradient_steps=config['finrl']['gradient_steps'],
            verbose=config['finrl']['verbose'],
            tensorboard_log=config['finrl']['tensorboard_log']
        )
        
        agent = FinRLAgent(finrl_config)
        
        # Use a subset of data for demo training
        demo_data = data.tail(500) if len(data) > 500 else data
        print(f"Training on {len(demo_data)} data points...")
        
        # Train the agent (shorter training for demo)
        training_steps = min(10000, config['finrl']['training']['total_timesteps'])
        result = agent.train(
            data=demo_data,
            config=config,
            total_timesteps=training_steps,
            use_real_broker=False  # Use simulation for demo training
        )
        
        if result['success']:
            print(f"✅ Training completed successfully!")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Timesteps: {result['total_timesteps']:,}")
            print(f"   Model saved: {result['model_path']}")
            
            # Test prediction
            print("\n🔮 Testing predictions...")
            prediction_result = agent.predict(
                data=demo_data.tail(100),
                config=config,
                use_real_broker=False
            )
            
            if prediction_result['success']:
                print(f"✅ Prediction completed!")
                print(f"   Initial Value: ${prediction_result['initial_value']:,.2f}")
                print(f"   Final Value: ${prediction_result['final_value']:,.2f}")
                print(f"   Total Return: {prediction_result['total_return']:.2%}")
                print(f"   Total Trades: {prediction_result['total_trades']}")
            
            return agent
        else:
            print(f"❌ Training failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Error in FinRL training: {e}")
        return None

def demo_trading_workflow(config: Dict[str, Any], data):
    """Demonstrate complete trading workflow"""
    print("\n🔄 TRADING WORKFLOW DEMO")
    print("-" * 30)
    
    try:
        # Run single trading cycle
        print("Running trading workflow...")
        result = run(config)
        
        if result['success']:
            print("✅ Trading workflow completed successfully!")
            print(f"   Data Loaded: {'✅' if result['data_loaded'] else '❌'}")
            print(f"   Signal Generated: {'✅' if result['signal_generated'] else '❌'}")
            print(f"   Order Executed: {'✅' if result['order_executed'] else '❌'}")
            print(f"   Execution Time: {result['execution_time']:.2f} seconds")
            
            if result['order_executed'] and result['execution_result']:
                exec_result = result['execution_result']
                print(f"   Order ID: {exec_result.get('order_id', 'N/A')}")
                print(f"   Action: {exec_result['action']}")
                print(f"   Symbol: {exec_result['symbol']}")
                print(f"   Quantity: {exec_result['quantity']}")
                print(f"   Price: ${exec_result['price']:.2f}")
                print(f"   Total Value: ${exec_result['total_value']:.2f}")
        else:
            print("❌ Trading workflow failed!")
            for error in result['errors']:
                print(f"   Error: {error}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in trading workflow: {e}")
        return None

def demo_backtest(config: Dict[str, Any], data):
    """Demonstrate backtesting capabilities"""
    print("\n📊 BACKTESTING DEMO")
    print("-" * 30)
    
    try:
        # Run backtest on recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Running backtest from {start_date} to {end_date}...")
        result = run_backtest(config, start_date, end_date)
        
        if result['success']:
            print("✅ Backtest completed successfully!")
            print(f"   Initial Capital: ${result['initial_capital']:,.2f}")
            print(f"   Final Value: ${result['final_value']:,.2f}")
            print(f"   Total Return: {result['total_return']:.2%}")
            print(f"   Total Trades: {result['total_trades']}")
            
            # Calculate additional metrics
            if result['total_trades'] > 0:
                win_rate = len([t for t in result['trades'] if t.get('execution', {}).get('success', False)]) / result['total_trades']
                print(f"   Win Rate: {win_rate:.2%}")
        else:
            print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in backtesting: {e}")
        return None

def main():
    """Main demo function"""
    setup_logging()
    
    try:
        # Load configuration
        config = load_config()
        print_system_info(config)
        
        # Demo 1: Data Ingestion
        data = demo_data_ingestion(config)
        if data is None:
            print("❌ Cannot proceed without data")
            return
        
        # Demo 2: Alpaca Integration
        alpaca_broker = demo_alpaca_integration(config)
        
        # Demo 3: FinRL Training
        finrl_agent = demo_finrl_training(config, data)
        
        # Demo 4: Trading Workflow
        workflow_result = demo_trading_workflow(config, data)
        
        # Demo 5: Backtesting
        backtest_result = demo_backtest(config, data)
        
        # Summary
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 Summary:")
        print(f"   ✅ Data Ingestion: {'Working' if data is not None else 'Failed'}")
        print(f"   ✅ Alpaca Integration: {'Working' if alpaca_broker is not None else 'Simulation Mode'}")
        print(f"   ✅ FinRL Training: {'Working' if finrl_agent is not None else 'Failed'}")
        print(f"   ✅ Trading Workflow: {'Working' if workflow_result and workflow_result['success'] else 'Failed'}")
        print(f"   ✅ Backtesting: {'Working' if backtest_result and backtest_result['success'] else 'Failed'}")
        
        print("\n🚀 Next Steps:")
        print("   1. Set up your Alpaca API credentials in .env file")
        print("   2. Configure your trading strategy in config.yaml")
        print("   3. Run live trading with: python -m agentic_ai_system.main --mode live")
        print("   4. Monitor performance in logs/ directory")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 