"""
Tests for FinRL Agent

This module contains comprehensive tests for the FinRL agent functionality.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os
from unittest.mock import Mock, patch

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_ai_system.finrl_agent import (
    FinRLAgent, 
    FinRLConfig, 
    TradingEnvironment,
    create_finrl_agent_from_config
)


class TestFinRLConfig:
    """Test FinRL configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = FinRLConfig()
        
        assert config.algorithm == "PPO"
        assert config.learning_rate == 0.0003
        assert config.batch_size == 64
        assert config.gamma == 0.99
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = FinRLConfig(
            algorithm="A2C",
            learning_rate=0.001,
            batch_size=128
        )
        
        assert config.algorithm == "A2C"
        assert config.learning_rate == 0.001
        assert config.batch_size == 128


class TestTradingEnvironment:
    """Test trading environment"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100),
            'sma_20': np.random.uniform(100, 200, 100),
            'sma_50': np.random.uniform(100, 200, 100),
            'rsi': np.random.uniform(0, 100, 100),
            'bb_upper': np.random.uniform(100, 200, 100),
            'bb_lower': np.random.uniform(100, 200, 100),
            'macd': np.random.uniform(-10, 10, 100)
        }, index=dates)
        return data
    
    def test_environment_initialization(self, sample_data):
        """Test environment initialization"""
        env = TradingEnvironment(sample_data)
        
        assert env.initial_balance == 100000
        assert env.transaction_fee == 0.001
        assert env.max_position == 100
        assert env.action_space.n == 3
        assert len(env.observation_space.shape) == 1
    
    def test_environment_reset(self, sample_data):
        """Test environment reset"""
        env = TradingEnvironment(sample_data)
        obs, info = env.reset()
        
        assert env.current_step == 0
        assert env.balance == env.initial_balance
        assert env.position == 0
        assert env.portfolio_value == env.initial_balance
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
    
    def test_environment_step(self, sample_data):
        """Test environment step"""
        env = TradingEnvironment(sample_data)
        obs, info = env.reset()
        
        # Test hold action
        obs, reward, done, truncated, info = env.step(1)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.current_step == 1
    
    def test_buy_action(self, sample_data):
        """Test buy action"""
        env = TradingEnvironment(sample_data, initial_balance=10000)
        obs, info = env.reset()
        
        initial_balance = env.balance
        initial_position = env.position
        
        # Buy action
        obs, reward, done, truncated, info = env.step(2)
        
        assert env.position > initial_position
        assert env.balance < initial_balance
    
    def test_sell_action(self, sample_data):
        """Test sell action"""
        env = TradingEnvironment(sample_data, initial_balance=10000)
        obs, info = env.reset()
        
        # First buy some shares
        obs, reward, done, truncated, info = env.step(2)
        initial_position = env.position
        initial_balance = env.balance
        
        # Then sell
        obs, reward, done, truncated, info = env.step(0)
        
        assert env.position < initial_position
        assert env.balance > initial_balance
    
    def test_portfolio_value_calculation(self, sample_data):
        """Test portfolio value calculation"""
        env = TradingEnvironment(sample_data)
        obs, info = env.reset()
        
        # Buy some shares
        obs, reward, done, truncated, info = env.step(2)
        
        expected_value = env.balance + (env.position * sample_data.iloc[env.current_step]['close'])
        assert abs(env.portfolio_value - expected_value) < 1e-6


class TestFinRLAgent:
    """Test FinRL agent"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def finrl_config(self):
        """Create FinRL configuration"""
        return FinRLConfig(
            algorithm="PPO",
            learning_rate=0.0003,
            batch_size=32,
            total_timesteps=1000
        )
    
    def test_agent_initialization(self, finrl_config):
        """Test agent initialization"""
        agent = FinRLAgent(finrl_config)
        
        assert agent.config == finrl_config
        assert agent.model is None
        assert agent.env is None
    
    def test_prepare_data(self, finrl_config, sample_data):
        """Test data preparation"""
        agent = FinRLAgent(finrl_config)
        prepared_data = agent.prepare_data(sample_data)
        
        # Check that technical indicators were added
        assert 'sma_20' in prepared_data.columns
        assert 'sma_50' in prepared_data.columns
        assert 'rsi' in prepared_data.columns
        assert 'bb_upper' in prepared_data.columns
        assert 'bb_lower' in prepared_data.columns
        assert 'macd' in prepared_data.columns
        
        # Check that no NaN values remain
        assert not prepared_data.isnull().any().any()
    
    def test_create_environment(self, finrl_config, sample_data):
        """Test environment creation"""
        agent = FinRLAgent(finrl_config)
        env = agent.create_environment(sample_data)
        
        assert isinstance(env, TradingEnvironment)
        assert env.data.equals(sample_data)
    
    def test_technical_indicators_calculation(self, finrl_config):
        """Test technical indicators calculation"""
        agent = FinRLAgent(finrl_config)
        
        # Test RSI calculation
        prices = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
        rsi = agent._calculate_rsi(prices, period=3)
        assert len(rsi) == len(prices)
        assert not rsi.isnull().all()
        
        # Test Bollinger Bands calculation
        bb_upper, bb_lower = agent._calculate_bollinger_bands(prices, period=3)
        assert len(bb_upper) == len(prices)
        assert len(bb_lower) == len(prices)
        assert (bb_upper >= bb_lower).all()
        
        # Test MACD calculation
        macd = agent._calculate_macd(prices)
        assert len(macd) == len(prices)
    
    @patch('agentic_ai_system.finrl_agent.PPO')
    def test_training_ppo(self, mock_ppo, finrl_config, sample_data):
        """Test PPO training"""
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        
        agent = FinRLAgent(finrl_config)
        result = agent.train(sample_data, total_timesteps=100)
        
        assert result['algorithm'] == 'PPO'
        assert result['total_timesteps'] == 100
        mock_model.learn.assert_called_once()
    
    @patch('agentic_ai_system.finrl_agent.A2C')
    def test_training_a2c(self, mock_a2c):
        """Test A2C training"""
        config = FinRLConfig(algorithm="A2C")
        mock_model = Mock()
        mock_a2c.return_value = mock_model
        
        agent = FinRLAgent(config)
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        result = agent.train(sample_data, total_timesteps=100)
        
        assert result['algorithm'] == 'A2C'
        mock_model.learn.assert_called_once()
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm handling"""
        config = FinRLConfig(algorithm="INVALID")
        agent = FinRLAgent(config)
        sample_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            agent.train(sample_data, total_timesteps=100)
    
    def test_predict_without_training(self, finrl_config, sample_data):
        """Test prediction without training"""
        agent = FinRLAgent(finrl_config)
        
        with pytest.raises(ValueError, match="Model not trained"):
            agent.predict(sample_data)
    
    def test_evaluate_without_training(self, finrl_config, sample_data):
        """Test evaluation without training"""
        agent = FinRLAgent(finrl_config)
        
        with pytest.raises(ValueError, match="Model not trained"):
            agent.evaluate(sample_data)
    
    @patch('agentic_ai_system.finrl_agent.PPO')
    def test_save_and_load_model(self, mock_ppo, finrl_config, sample_data):
        """Test model saving and loading"""
        # Mock the PPO model
        mock_model = Mock()
        mock_ppo.return_value = mock_model
        mock_ppo.load.return_value = mock_model
        
        agent = FinRLAgent(finrl_config)
        
        # Train the agent
        agent.train(sample_data, total_timesteps=100)
        
        # Test saving
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            agent.save_model(tmp_file.name)
            mock_model.save.assert_called_once_with(tmp_file.name)
        
        # Test loading
        agent.load_model(tmp_file.name)
        mock_ppo.load.assert_called_once_with(tmp_file.name)
        
        # Clean up
        os.unlink(tmp_file.name)


class TestFinRLIntegration:
    """Test FinRL integration with configuration"""
    
    def test_create_agent_from_config(self):
        """Test creating agent from configuration file"""
        config_data = {
            'finrl': {
                'algorithm': 'PPO',
                'learning_rate': 0.001,
                'batch_size': 128,
                'gamma': 0.95
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            agent = create_finrl_agent_from_config(tmp_file_path)
            
            assert agent.config.algorithm == 'PPO'
            assert agent.config.learning_rate == 0.001
            assert agent.config.batch_size == 128
            assert agent.config.gamma == 0.95
        finally:
            os.unlink(tmp_file_path)
    
    def test_create_agent_from_config_missing_finrl(self):
        """Test creating agent from config without finrl section"""
        config_data = {
            'trading': {
                'symbol': 'AAPL',
                'capital': 100000
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_file_path = tmp_file.name
        
        try:
            agent = create_finrl_agent_from_config(tmp_file_path)
            
            # Should use default values
            assert agent.config.algorithm == 'PPO'
            assert agent.config.learning_rate == 0.0003
        finally:
            os.unlink(tmp_file_path)


if __name__ == "__main__":
    pytest.main([__file__]) 