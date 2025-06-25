import yaml
import logging
import sys
from typing import Dict, Any
from .orchestrator import run, run_backtest, run_live_trading
from .logger_config import setup_logging

def main():
    """Main entry point for the trading system"""
    try:
        # Load configuration
        config = load_config()
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting algorithmic trading system")
        
        # Run the trading workflow
        result = run(config)
        
        if result['success']:
            logger.info("Trading workflow completed successfully")
            if result['order_executed']:
                logger.info(f"Order executed: {result['execution_result']}")
        else:
            logger.error(f"Trading workflow failed: {result['errors']}")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def run_backtest_mode(config_path: str = 'config.yaml', 
                     start_date: str = '2024-01-01', 
                     end_date: str = '2024-12-31'):
    """Run the system in backtest mode"""
    try:
        config = load_config(config_path)
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Running in backtest mode")
        result = run_backtest(config, start_date, end_date)
        
        if result['success']:
            logger.info(f"Backtest completed: {result['total_return']:.2%} return")
            logger.info(f"Total trades: {result['total_trades']}")
        else:
            logger.error(f"Backtest failed: {result['error']}")
            
    except Exception as e:
        print(f"Backtest error: {e}")
        sys.exit(1)

def run_live_mode(config_path: str = 'config.yaml', duration_minutes: int = 60):
    """Run the system in live trading mode"""
    try:
        config = load_config(config_path)
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Running in live trading mode")
        result = run_live_trading(config, duration_minutes)
        
        if result['success']:
            logger.info(f"Live trading completed: {result['total_trades']} trades")
        else:
            logger.error(f"Live trading failed: {result['error']}")
            
    except Exception as e:
        print(f"Live trading error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithmic Trading System')
    parser.add_argument('--mode', choices=['standard', 'backtest', 'live'], 
                       default='standard', help='Run mode')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--start-date', default='2024-01-01', help='Backtest start date')
    parser.add_argument('--end-date', default='2024-12-31', help='Backtest end date')
    parser.add_argument('--duration', type=int, default=60, help='Live trading duration (minutes)')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        run_backtest_mode(args.config, args.start_date, args.end_date)
    elif args.mode == 'live':
        run_live_mode(args.config, args.duration)
    else:
        main()
