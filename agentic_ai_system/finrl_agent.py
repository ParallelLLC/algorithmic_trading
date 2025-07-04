"""
FinRL Agent for Algorithmic Trading

This module provides a FinRL-based reinforcement learning agent that can be integrated
with the existing algorithmic trading system. It supports various RL algorithms
including PPO, A2C, DDPG, and TD3, and can work with Alpaca broker for real trading.
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
import inspect

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
    - Integrate with Alpaca broker for real trading
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], 
                 initial_balance: float = 100000, transaction_fee: float = 0.001, 
                 max_position: int = 100, use_real_broker: bool = False):
        super().__init__()
        
        self.data = data
        self.config = config
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        self.use_real_broker = use_real_broker
        
        # Initialize Alpaca broker if using real trading
        self.alpaca_broker = None
        if use_real_broker:
            try:
                from .alpaca_broker import AlpacaBroker
                self.alpaca_broker = AlpacaBroker(config)
                logger.info("Alpaca broker initialized for FinRL environment")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca broker: {e}")
                self.use_real_broker = False
        
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
                
                if self.use_real_broker and self.alpaca_broker:
                    # Execute real order with Alpaca
                    result = self.alpaca_broker.place_market_order(
                        symbol=self.config['trading']['symbol'],
                        quantity=shares_to_sell,
                        side='sell'
                    )
                    
                    if result['success']:
                        sell_value = result['filled_avg_price'] * shares_to_sell * (1 - self.transaction_fee)
                        self.balance += sell_value
                        self.position -= shares_to_sell
                        logger.info(f"Real sell order executed: {result['order_id']}")
                    else:
                        logger.warning(f"Real sell order failed: {result['error']}")
                else:
                    # Simulate order execution
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
                    if self.use_real_broker and self.alpaca_broker:
                        # Execute real order with Alpaca
                        result = self.alpaca_broker.place_market_order(
                            symbol=self.config['trading']['symbol'],
                            quantity=max_shares,
                            side='buy'
                        )
                        
                        if result['success']:
                            buy_value = result['filled_avg_price'] * max_shares * (1 + self.transaction_fee)
                            self.balance -= buy_value
                            self.position += max_shares
                            logger.info(f"Real buy order executed: {result['order_id']}")
                        else:
                            logger.warning(f"Real buy order failed: {result['error']}")
                    else:
                        # Simulate order execution
                        buy_value = max_shares * current_price * (1 + self.transaction_fee)
                        self.balance -= buy_value
                        self.position += max_shares
        
        # Update portfolio value
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self._calculate_portfolio_value()
        self.total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get observation for next step
        if not done:
            observation = self._get_features(self.data.iloc[self.current_step])
        else:
            observation = self._get_features(self.data.iloc[-1])
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'total_return': self.total_return,
            'position': self.position,
            'balance': self.balance,
            'step': self.current_step
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
        
        # Get initial observation
        observation = self._get_features(self.data.iloc[0])
        
        return observation, {}


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
        
        logger.info(f"Initializing FinRL agent with algorithm: {self.config.algorithm}")

    def _get_valid_kwargs(self, algo_class):
        """Return a dict of config fields valid for the given algorithm class, excluding tensorboard_log."""
        sig = inspect.signature(algo_class.__init__)
        valid_keys = set(sig.parameters.keys())
        # Exclude 'self', 'policy', and 'tensorboard_log' which are always passed explicitly
        valid_keys.discard('self')
        valid_keys.discard('policy')
        valid_keys.discard('tensorboard_log')
        # Build kwargs from config dataclass
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__ if k in valid_keys}
    
    def create_environment(self, data: pd.DataFrame, config: Dict[str, Any], 
                          initial_balance: float = 100000, use_real_broker: bool = False) -> TradingEnvironment:
        """Create trading environment from market data"""
        return TradingEnvironment(
            data=data,
            config=config,
            initial_balance=initial_balance,
            transaction_fee=0.001,
            max_position=100,
            use_real_broker=use_real_broker
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
    
    def train(self, data: pd.DataFrame, config: Dict[str, Any], 
              total_timesteps: int = 100000, use_real_broker: bool = False) -> Dict[str, Any]:
        """
        Train the FinRL agent
        
        Args:
            data: Market data for training
            config: Configuration dictionary
            total_timesteps: Number of timesteps for training
            use_real_broker: Whether to use real Alpaca broker during training
            
        Returns:
            Training results dictionary
        """
        try:
            # Prepare data
            prepared_data = self.prepare_data(data)
            
            # Create environment
            self.env = self.create_environment(prepared_data, config, use_real_broker=use_real_broker)
            
            # Create evaluation environment (without real broker)
            eval_data = prepared_data.copy()
            self.eval_env = self.create_environment(eval_data, config, use_real_broker=False)
            
            # Create callback for evaluation
            finrl_config = config.get('finrl', {})
            training_config = finrl_config.get('training', {})
            
            model_save_path = training_config.get('model_save_path', 'models/finrl')
            tensorboard_log = finrl_config.get('tensorboard_log', self.config.tensorboard_log)
            eval_freq = training_config.get('eval_freq', 1000)
            
            self.callback = EvalCallback(
                self.eval_env,
                best_model_save_path=model_save_path,
                log_path=tensorboard_log,
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            
            # Initialize model based on algorithm
            if self.config.algorithm == "PPO":
                algo_kwargs = self._get_valid_kwargs(PPO)
                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    **algo_kwargs,
                    tensorboard_log=self.config.tensorboard_log
                )
            elif self.config.algorithm == "A2C":
                algo_kwargs = self._get_valid_kwargs(A2C)
                self.model = A2C(
                    "MlpPolicy",
                    self.env,
                    **algo_kwargs,
                    tensorboard_log=self.config.tensorboard_log
                )
            elif self.config.algorithm == "DDPG":
                algo_kwargs = self._get_valid_kwargs(DDPG)
                self.model = DDPG(
                    "MlpPolicy",
                    self.env,
                    **algo_kwargs,
                    tensorboard_log=self.config.tensorboard_log
                )
            elif self.config.algorithm == "TD3":
                algo_kwargs = self._get_valid_kwargs(TD3)
                self.model = TD3(
                    "MlpPolicy",
                    self.env,
                    **algo_kwargs,
                    tensorboard_log=self.config.tensorboard_log
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
            
            # Train the model
            logger.info(f"Starting training with {total_timesteps} timesteps")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callback,
                progress_bar=True
            )
            
            # Save the final model
            model_path = f"{model_save_path}/final_model"
            self.model.save(model_path)
            logger.info(f"Training completed. Model saved to {model_path}")
            
            return {
                'success': True,
                'algorithm': self.config.algorithm,
                'total_timesteps': total_timesteps,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, data: pd.DataFrame, config: Dict[str, Any], 
                use_real_broker: bool = False) -> Dict[str, Any]:
        """
        Make predictions using the trained model
        
        Args:
            data: Market data for prediction
            config: Configuration dictionary
            use_real_broker: Whether to use real Alpaca broker for execution
            
        Returns:
            Prediction results dictionary
        """
        try:
            if self.model is None:
                # Try to load model
                finrl_config = config.get('finrl', {})
                inference_config = finrl_config.get('inference', {})
                
                model_path = inference_config.get('model_path', 'models/finrl/final_model')
                use_trained_model = inference_config.get('use_trained_model', True)
                
                if use_trained_model:
                    self.model = self._load_model(model_path, config)
                    if self.model is None:
                        return {'success': False, 'error': 'No trained model available'}
                else:
                    return {'success': False, 'error': 'No model available for prediction'}
            
            # Prepare data
            prepared_data = self.prepare_data(data)
            
            # Create environment
            env = self.create_environment(prepared_data, config, use_real_broker=use_real_broker)
            
            # Run prediction
            obs, _ = env.reset()
            done = False
            actions = []
            rewards = []
            portfolio_values = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                
                actions.append(action)
                rewards.append(reward)
                portfolio_values.append(info['portfolio_value'])
            
            # Calculate final metrics
            initial_value = config.get('trading', {}).get('capital', 100000)
            final_value = portfolio_values[-1] if portfolio_values else initial_value
            total_return = (final_value - initial_value) / initial_value
            
            return {
                'success': True,
                'actions': actions,
                'rewards': rewards,
                'portfolio_values': portfolio_values,
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': len([a for a in actions if a != 1])  # Count non-hold actions
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate(self, data: pd.DataFrame, config: Dict[str, Any], 
                 use_real_broker: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data
        
        Args:
            data: Market data for evaluation
            config: Configuration dictionary
            use_real_broker: Whether to use real Alpaca broker for execution
            
        Returns:
            Evaluation results dictionary
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            # Prepare data
            prepared_data = self.prepare_data(data)
            
            # Create environment
            env = self.create_environment(prepared_data, config, use_real_broker=use_real_broker)
            
            # Run evaluation
            obs, _ = env.reset()
            done = False
            actions = []
            rewards = []
            portfolio_values = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                
                actions.append(action)
                rewards.append(reward)
                portfolio_values.append(info['portfolio_value'])
            
            # Calculate evaluation metrics
            initial_value = config.get('trading', {}).get('capital', 100000)
            final_value = portfolio_values[-1] if portfolio_values else initial_value
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate additional metrics
            total_trades = len([a for a in actions if a != 1])  # Count non-hold actions
            avg_reward = np.mean(rewards) if rewards else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            return {
                'success': True,
                'total_return': total_return,
                'total_trades': total_trades,
                'avg_reward': avg_reward,
                'max_drawdown': max_drawdown,
                'final_portfolio_value': final_value,
                'initial_portfolio_value': initial_value,
                'actions': actions,
                'rewards': rewards,
                'portfolio_values': portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_model(self, model_path: str) -> bool:
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """
        Load a trained model
        
        Args:
            model_path: Path to the model
            config: Configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = self._load_model(model_path, config)
            if self.model is None:
                return False
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values"""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _load_model(self, model_path: str, config: Dict[str, Any]):
        """Load a trained model"""
        try:
            # Get algorithm from config or use default
            finrl_config = config.get('finrl', {})
            algorithm = finrl_config.get('algorithm', self.config.algorithm)
            
            if algorithm == "PPO":
                return PPO.load(model_path)
            elif algorithm == "A2C":
                return A2C.load(model_path)
            elif algorithm == "DDPG":
                return DDPG.load(model_path)
            elif algorithm == "TD3":
                return TD3.load(model_path)
            else:
                logger.error(f"Unsupported algorithm for model loading: {algorithm}")
                return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
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
        macd = ema_fast - ema_slow
        return macd


def create_finrl_agent_from_config(config: FinRLConfig) -> FinRLAgent:
    """Create a FinRL agent from configuration"""
    return FinRLAgent(config) 