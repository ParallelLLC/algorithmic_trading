"""
FinRL Agent for Algorithmic Trading

This module provides a FinRL-based reinforcement learning agent that can be integrated
with the existing algorithmic trading system. It supports various RL algorithms
including PPO, A2C, DDPG, and TD3.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class FinRLConfig:
    """Configuration for FinRL agent"""
    algorithm: str = "PPO"  # PPO, A2C, DDPG, TD3
    learning_rate: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 1000000
    learning_starts: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    verbose: int = 1
    tensorboard_log: str = "logs/finrl_tensorboard"


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for FinRL
    
    This environment simulates a trading scenario where the agent can:
    - Buy, sell, or hold positions
    - Use technical indicators for decision making
    - Manage portfolio value and risk
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000, 
                 transaction_fee: float = 0.001, max_position: int = 100):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        
        # Reset state
        self.reset()
        
        # Define action space: [-1, 0, 1] for sell, hold, buy
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # Features: OHLCV + technical indicators + portfolio state
        n_features = len(self._get_features(self.data.iloc[0]))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
    
    def _get_features(self, row: pd.Series) -> np.ndarray:
        """Extract features from market data row"""
        features = []
        
        # Price features
        features.extend([
            row['open'], row['high'], row['low'], row['close'], row['volume']
        ])
        
        # Technical indicators (if available)
        for indicator in ['sma_20', 'sma_50', 'rsi', 'bb_upper', 'bb_lower', 'macd']:
            if indicator in row.index:
                features.append(row[indicator])
            else:
                features.append(0.0)
        
        # Portfolio state
        features.extend([
            self.balance,
            self.position,
            self.portfolio_value,
            self.total_return
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price = self.data.iloc[self.current_step]['close']
        return self.balance + (self.position * current_price)
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on portfolio performance"""
        current_value = self._calculate_portfolio_value()
        previous_value = self.previous_portfolio_value
        
        # Calculate return
        if previous_value > 0:
            return (current_value - previous_value) / previous_value
        else:
            return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute action
        if action == 0:  # Sell
            if self.position > 0:
                shares_to_sell = min(self.position, self.max_position)
                sell_value = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.balance += sell_value
                self.position -= shares_to_sell
        elif action == 2:  # Buy
            if self.balance > 0:
                max_shares = min(
                    int(self.balance / current_price),
                    self.max_position - self.position
                )
                if max_shares > 0:
                    buy_value = max_shares * current_price * (1 + self.transaction_fee)
                    self.balance -= buy_value
                    self.position += max_shares
        
        # Update portfolio value
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self._calculate_portfolio_value()
        self.total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get observation
        if not done:
            observation = self._get_features(self.data.iloc[self.current_step])
        else:
            # Use last available data for final observation
            observation = self._get_features(self.data.iloc[-1])
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'total_return': self.total_return,
            'current_price': current_price
        }
        
        return observation, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.total_return = 0.0
        
        observation = self._get_features(self.data.iloc[self.current_step])
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'total_return': self.total_return
        }
        
        return observation, info


class FinRLAgent:
    """
    FinRL-based reinforcement learning agent for algorithmic trading
    """
    
    def __init__(self, config: FinRLConfig):
        self.config = config
        self.model = None
        self.env = None
        self.eval_env = None
        self.callback = None
        
        logger.info(f"Initializing FinRL agent with algorithm: {config.algorithm}")
    
    def create_environment(self, data: pd.DataFrame, initial_balance: float = 100000) -> TradingEnvironment:
        """Create trading environment from market data"""
        return TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            transaction_fee=0.001,
            max_position=100
        )
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with technical indicators for FinRL"""
        df = data.copy()
        
        # Add technical indicators if not present
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        if 'rsi' not in df.columns:
            df['rsi'] = self._calculate_rsi(df['close'])
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
        if 'macd' not in df.columns:
            df['macd'] = self._calculate_macd(df['close'])
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        return macd_line
    
    def train(self, data: pd.DataFrame, total_timesteps: int = 100000, 
              eval_freq: int = 10000, eval_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the FinRL agent"""
        
        logger.info("Starting FinRL agent training")
        
        # Prepare data
        train_data = self.prepare_data(data)
        
        # Create training environment
        self.env = DummyVecEnv([lambda: self.create_environment(train_data)])
        
        # Create evaluation environment if provided
        if eval_data is not None:
            eval_data = self.prepare_data(eval_data)
            self.eval_env = DummyVecEnv([lambda: self.create_environment(eval_data)])
            self.callback = EvalCallback(
                self.eval_env,
                best_model_save_path="models/finrl_best/",
                log_path="logs/finrl_eval/",
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
        
        # Initialize model based on algorithm
        if self.config.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log
            )
        elif self.config.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log
            )
        elif self.config.algorithm == "DDPG":
            self.model = DDPG(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                gamma=self.config.gamma,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log
            )
        elif self.config.algorithm == "TD3":
            self.model = TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                learning_starts=self.config.learning_starts,
                gamma=self.config.gamma,
                tau=self.config.tau,
                train_freq=self.config.train_freq,
                gradient_steps=self.config.gradient_steps,
                target_update_interval=self.config.target_update_interval,
                verbose=self.config.verbose,
                tensorboard_log=self.config.tensorboard_log
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Train the model
        callbacks = [self.callback] if self.callback else None
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        logger.info("FinRL agent training completed")
        
        return {
            'algorithm': self.config.algorithm,
            'total_timesteps': total_timesteps,
            'model_path': f"models/finrl_{self.config.algorithm.lower()}"
        }
    
    def predict(self, data: pd.DataFrame) -> List[int]:
        """Generate trading predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        test_data = self.prepare_data(data)
        
        # Create test environment
        test_env = self.create_environment(test_data)
        
        predictions = []
        obs, _ = test_env.reset()
        
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            predictions.append(action)
            obs, _, done, _, _ = test_env.step(action)
        
        return predictions
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        test_data = self.prepare_data(data)
        
        # Create test environment
        test_env = self.create_environment(test_data)
        
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward
            steps += 1
        
        # Calculate metrics
        final_portfolio_value = info['portfolio_value']
        initial_balance = test_env.initial_balance
        total_return = (final_portfolio_value - initial_balance) / initial_balance
        
        return {
            'total_reward': total_reward,
            'total_return': total_return,
            'final_portfolio_value': final_portfolio_value,
            'steps': steps,
            'sharpe_ratio': total_reward / steps if steps > 0 else 0
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        if self.config.algorithm == "PPO":
            self.model = PPO.load(path)
        elif self.config.algorithm == "A2C":
            self.model = A2C.load(path)
        elif self.config.algorithm == "DDPG":
            self.model = DDPG.load(path)
        elif self.config.algorithm == "TD3":
            self.model = TD3.load(path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        logger.info(f"Model loaded from {path}")


def create_finrl_agent_from_config(config_path: str) -> FinRLAgent:
    """Create FinRL agent from configuration file"""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    finrl_config = FinRLConfig(**config_data.get('finrl', {}))
    return FinRLAgent(finrl_config) 