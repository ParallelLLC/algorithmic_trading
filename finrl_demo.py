#!/usr/bin/env python3
"""
FinRL Demo Script

This script demonstrates the integration of FinRL with the algorithmic trading system.
It shows how to train a reinforcement learning agent and use it for trading decisions.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig, create_finrl_agent_from_config
from agentic_ai_system.synthetic_data_generator import SyntheticDataGenerator
from agentic_ai_system.logger_config import setup_logging


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Setup logging
config = load_config()
setup_logging(config)
logger = logging.getLogger(__name__)


def generate_training_data(config: dict) -> pd.DataFrame:
    """Generate synthetic data for training"""
    logger.info("Generating synthetic training data")
    
    generator = SyntheticDataGenerator(config)
    
    # Generate training data (longer period)
    train_data = generator.generate_ohlcv_data(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2023-12-31',
        frequency='1H'
    )
    
    # Add technical indicators
    train_data['sma_20'] = train_data['close'].rolling(window=20).mean()
    train_data['sma_50'] = train_data['close'].rolling(window=50).mean()
    train_data['rsi'] = calculate_rsi(train_data['close'])
    bb_upper, bb_lower = calculate_bollinger_bands(train_data['close'])
    train_data['bb_upper'] = bb_upper
    train_data['bb_lower'] = bb_lower
    train_data['macd'] = calculate_macd(train_data['close'])
    
    # Fill NaN values
    train_data = train_data.fillna(method='bfill').fillna(0)
    
    logger.info(f"Generated {len(train_data)} training samples")
    return train_data


def generate_test_data(config: dict) -> pd.DataFrame:
    """Generate synthetic data for testing"""
    logger.info("Generating synthetic test data")
    
    generator = SyntheticDataGenerator(config)
    
    # Generate test data (shorter period)
    test_data = generator.generate_ohlcv_data(
        symbol='AAPL',
        start_date='2024-01-01',
        end_date='2024-03-31',
        frequency='1H'
    )
    
    # Add technical indicators
    test_data['sma_20'] = test_data['close'].rolling(window=20).mean()
    test_data['sma_50'] = test_data['close'].rolling(window=50).mean()
    test_data['rsi'] = calculate_rsi(test_data['close'])
    bb_upper, bb_lower = calculate_bollinger_bands(test_data['close'])
    test_data['bb_upper'] = bb_upper
    test_data['bb_lower'] = bb_lower
    test_data['macd'] = calculate_macd(test_data['close'])
    
    # Fill NaN values
    test_data = test_data.fillna(method='bfill').fillna(0)
    
    logger.info(f"Generated {len(test_data)} test samples")
    return test_data


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line


def train_finrl_agent(config: dict, train_data: pd.DataFrame, test_data: pd.DataFrame) -> FinRLAgent:
    """Train the FinRL agent"""
    logger.info("Starting FinRL agent training")
    
    # Create FinRL agent
    finrl_config = FinRLConfig(**{k: v for k, v in config['finrl'].items() if k in FinRLConfig.__dataclass_fields__})
    agent = FinRLAgent(finrl_config)
    
    # Train the agent
    training_result = agent.train(
        data=train_data,
        config=config,
        total_timesteps=config['finrl']['training']['total_timesteps']
    )
    
    logger.info(f"Training completed: {training_result}")
    
    # Save the model
    if config['finrl']['training']['save_best_model']:
        model_path = config['finrl']['training']['model_save_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)
    
    return agent


def evaluate_agent(agent: FinRLAgent, test_data: pd.DataFrame, config: dict) -> dict:
    """Evaluate the trained agent"""
    logger.info("Evaluating FinRL agent")
    
    # Evaluate on test data
    evaluation_results = agent.evaluate(test_data, config)
    
    logger.info(f"Evaluation results: {evaluation_results}")
    
    return evaluation_results


def generate_predictions(agent: FinRLAgent, test_data: pd.DataFrame, config: dict) -> list:
    """Generate trading predictions"""
    logger.info("Generating trading predictions")
    
    prediction_results = agent.predict(test_data, config)
    
    if prediction_results['success']:
        predictions = prediction_results['actions']
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    else:
        logger.error(f"Prediction failed: {prediction_results['error']}")
        return []


def plot_results(test_data: pd.DataFrame, predictions: list, evaluation_results: dict):
    """Plot trading results"""
    logger.info("Creating visualization plots")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price and predictions
    axes[0].plot(test_data.index, test_data['close'], label='Close Price', alpha=0.7)
    
    # Mark buy/sell signals only if predictions are available
    if predictions:
        buy_signals = [i for i, pred in enumerate(predictions) if pred == 2]
        sell_signals = [i for i, pred in enumerate(predictions) if pred == 0]
        
        if buy_signals:
            axes[0].scatter(test_data.index[buy_signals], test_data['close'].iloc[buy_signals], 
                           color='green', marker='^', s=100, label='Buy Signal', alpha=0.8)
        if sell_signals:
            axes[0].scatter(test_data.index[sell_signals], test_data['close'].iloc[sell_signals], 
                           color='red', marker='v', s=100, label='Sell Signal', alpha=0.8)
    
    axes[0].set_title('Price Action and Trading Signals')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Technical indicators
    axes[1].plot(test_data.index, test_data['close'], label='Close Price', alpha=0.7)
    axes[1].plot(test_data.index, test_data['sma_20'], label='SMA 20', alpha=0.7)
    axes[1].plot(test_data.index, test_data['sma_50'], label='SMA 50', alpha=0.7)
    axes[1].plot(test_data.index, test_data['bb_upper'], label='BB Upper', alpha=0.5)
    axes[1].plot(test_data.index, test_data['bb_lower'], label='BB Lower', alpha=0.5)
    
    axes[1].set_title('Technical Indicators')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: RSI
    axes[2].plot(test_data.index, test_data['rsi'], label='RSI', color='purple')
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[2].set_title('RSI Indicator')
    axes[2].set_ylabel('RSI')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/finrl_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Plots saved to plots/finrl_trading_results.png")


def print_summary(evaluation_results: dict, predictions: list):
    """Print trading summary"""
    print("\n" + "="*60)
    print("FINRL TRADING SYSTEM SUMMARY")
    print("="*60)
    
    if evaluation_results.get('success', False):
        print(f"Algorithm: {evaluation_results.get('algorithm', 'Unknown')}")
        print(f"Total Return: {evaluation_results.get('total_return', 0):.2%}")
        print(f"Final Portfolio Value: ${evaluation_results.get('final_portfolio_value', 0):,.2f}")
        print(f"Total Reward: {evaluation_results.get('total_reward', 0):.4f}")
        print(f"Sharpe Ratio: {evaluation_results.get('sharpe_ratio', 0):.4f}")
        print(f"Number of Trading Steps: {evaluation_results.get('steps', 0)}")
        print(f"Max Drawdown: {evaluation_results.get('max_drawdown', 0):.2%}")
    else:
        print(f"Evaluation failed: {evaluation_results.get('error', 'Unknown error')}")
    
    # Trading statistics
    if predictions:
        buy_signals = sum(1 for pred in predictions if pred == 2)
        sell_signals = sum(1 for pred in predictions if pred == 0)
        hold_signals = sum(1 for pred in predictions if pred == 1)
        
        print(f"\nTrading Signals:")
        print(f"  Buy signals: {buy_signals}")
        print(f"  Sell signals: {sell_signals}")
        print(f"  Hold signals: {hold_signals}")
        print(f"  Total signals: {len(predictions)}")
    else:
        print(f"\nNo trading predictions available")
    
    print("\n" + "="*60)


def main():
    """Main function to run the FinRL demo"""
    logger.info("Starting FinRL Demo")
    
    try:
        # Load configuration
        config = load_config()
        
        # Generate data
        train_data = generate_training_data(config)
        test_data = generate_test_data(config)
        
        # Train FinRL agent
        agent = train_finrl_agent(config, train_data, test_data)
        
        # Evaluate agent
        evaluation_results = evaluate_agent(agent, test_data, config)
        
        # Generate predictions
        predictions = generate_predictions(agent, test_data, config)
        
        # Create visualizations
        plot_results(test_data, predictions, evaluation_results)
        
        # Print summary
        print_summary(evaluation_results, predictions)
        
        logger.info("FinRL Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in FinRL demo: {str(e)}")
        raise


if __name__ == "__main__":
    main() 