import logging
import time
import pandas as pd
from typing import Dict, Any, Optional
from .data_ingestion import load_data, validate_data
from .strategy_agent import StrategyAgent
from .execution_agent import ExecutionAgent

logger = logging.getLogger(__name__)

def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main orchestration function that coordinates the trading workflow.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing execution results and statistics
    """
    start_time = time.time()
    logger.info("Starting trading system orchestration")
    
    try:
        # Initialize workflow results
        workflow_result = {
            'success': False,
            'data_loaded': False,
            'signal_generated': False,
            'order_executed': False,
            'execution_result': None,
            'errors': [],
            'execution_time': 0
        }
        
        # Step 1: Load market data
        logger.info("Step 1: Loading market data")
        data = load_data(config)
        
        if data is not None and not data.empty:
            workflow_result['data_loaded'] = True
            logger.info(f"Successfully loaded {len(data)} data points")
            
            # Validate data quality
            if validate_data(data):
                logger.info("Data validation passed")
            else:
                logger.warning("Data validation failed, but continuing with workflow")
        else:
            logger.error("Failed to load market data")
            workflow_result['errors'].append("Failed to load market data")
            return workflow_result
        
        # Step 2: Generate trading signal
        logger.info("Step 2: Generating trading signal")
        strategy_agent = StrategyAgent(config)
        signal = strategy_agent.act(data)
        
        if signal and signal.get('action') != 'hold':
            workflow_result['signal_generated'] = True
            logger.info(f"Generated signal: {signal['action']} {signal['quantity']} {signal['symbol']}")
        else:
            logger.info("No actionable signal generated (hold)")
            workflow_result['signal_generated'] = True  # Hold is still a valid signal
        
        # Step 3: Execute order
        logger.info("Step 3: Executing order")
        execution_agent = ExecutionAgent(config)
        execution_result = execution_agent.act(signal)
        
        if execution_result['success']:
            workflow_result['order_executed'] = True
            workflow_result['execution_result'] = execution_result
            logger.info("Order executed successfully")
        else:
            logger.error(f"Order execution failed: {execution_result.get('error', 'Unknown error')}")
            workflow_result['errors'].append(f"Order execution failed: {execution_result.get('error')}")
        
        # Calculate execution time
        workflow_result['execution_time'] = time.time() - start_time
        workflow_result['success'] = workflow_result['data_loaded'] and workflow_result['signal_generated']
        
        logger.info(f"Trading workflow completed in {workflow_result['execution_time']:.2f} seconds")
        
        return workflow_result
        
    except Exception as e:
        logger.error(f"Error in trading workflow: {e}", exc_info=True)
        workflow_result = {
            'success': False,
            'data_loaded': False,
            'signal_generated': False,
            'order_executed': False,
            'execution_result': None,
            'errors': [str(e)],
            'execution_time': time.time() - start_time
        }
        return workflow_result

def run_backtest(config: Dict[str, Any], start_date: str = '2024-01-01', end_date: str = '2024-12-31') -> Dict[str, Any]:
    """
    Run backtesting simulation over historical data.
    
    Args:
        config: Configuration dictionary
        start_date: Start date for backtest
        end_date: End date for backtest
        
    Returns:
        Dictionary containing backtest results
    """
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    try:
        # Load historical data
        data = load_data(config)
        
        if data is None or data.empty:
            logger.error("No data available for backtest")
            return {'success': False, 'error': 'No data available'}
        
        # Filter data for backtest period
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
        backtest_data = data.loc[mask]
        
        if backtest_data.empty:
            logger.error("No data available for specified backtest period")
            return {'success': False, 'error': 'No data for backtest period'}
        
        logger.info(f"Running backtest on {len(backtest_data)} data points")
        
        # Initialize agents
        strategy_agent = StrategyAgent(config)
        execution_agent = ExecutionAgent(config)
        
        # Track backtest results
        trades = []
        portfolio_value = config['trading']['capital']
        positions = {}
        
        # Run simulation
        for i in range(len(backtest_data)):
            current_data = backtest_data.iloc[:i+1]
            
            if len(current_data) < 50:  # Need minimum data for indicators
                continue
            
            # Generate signal
            signal = strategy_agent.act(current_data)
            
            # Execute if not hold
            if signal['action'] != 'hold':
                execution_result = execution_agent.act(signal)
                trades.append({
                    'timestamp': current_data.index[-1],
                    'signal': signal,
                    'execution': execution_result
                })
                
                # Update portfolio (simplified)
                if execution_result['success']:
                    symbol = signal['symbol']
                    if signal['action'] == 'buy':
                        positions[symbol] = positions.get(symbol, 0) + signal['quantity']
                        portfolio_value -= execution_result['total_value']
                    elif signal['action'] == 'sell':
                        positions[symbol] = positions.get(symbol, 0) - signal['quantity']
                        portfolio_value += execution_result['total_value']
        
        # Calculate final portfolio value
        final_value = portfolio_value
        for symbol, quantity in positions.items():
            if quantity > 0:
                final_price = backtest_data['close'].iloc[-1]
                final_value += quantity * final_price
        
        # Calculate performance metrics
        total_return = (final_value - config['trading']['capital']) / config['trading']['capital']
        
        backtest_results = {
            'success': True,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': config['trading']['capital'],
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'trades': trades,
            'positions': positions
        }
        
        logger.info(f"Backtest completed: {total_return:.2%} return over {len(trades)} trades")
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

def run_live_trading(config: Dict[str, Any], duration_minutes: int = 60) -> Dict[str, Any]:
    """
    Run live trading simulation for a specified duration.
    
    Args:
        config: Configuration dictionary
        duration_minutes: Duration to run live trading in minutes
        
    Returns:
        Dictionary containing live trading results
    """
    logger.info(f"Starting live trading simulation for {duration_minutes} minutes")
    
    try:
        import time
        from datetime import datetime, timedelta
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        trades = []
        
        while datetime.now() < end_time:
            # Run single trading cycle
            result = run(config)
            
            if result['order_executed'] and result['execution_result']['success']:
                trades.append(result['execution_result'])
            
            # Wait before next cycle
            time.sleep(60)  # Wait 1 minute between cycles
        
        live_results = {
            'success': True,
            'duration_minutes': duration_minutes,
            'total_trades': len(trades),
            'trades': trades,
            'start_time': datetime.now() - timedelta(minutes=duration_minutes),
            'end_time': datetime.now()
        }
        
        logger.info(f"Live trading completed: {len(trades)} trades executed")
        return live_results
        
    except Exception as e:
        logger.error(f"Error in live trading: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
