import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agentic_ai_system.strategy_agent import StrategyAgent

class TestStrategyAgent:
    """Test cases for StrategyAgent"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing"""
        return {
            'trading': {
                'symbol': 'AAPL',
                'timeframe': '1min',
                'capital': 100000
            },
            'risk': {
                'max_position': 100,
                'max_drawdown': 0.05
            },
            'execution': {
                'broker_api': 'paper',
                'order_size': 10
            }
        }
    
    @pytest.fixture
    def strategy_agent(self, config):
        """Create a StrategyAgent instance"""
        return StrategyAgent(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        # Generate realistic price data
        base_price = 150.0
        prices = []
        for i in range(100):
            # Add some trend and noise
            price = base_price + (i * 0.1) + np.random.normal(0, 2)
            prices.append(max(price, 1))  # Ensure positive prices
        
        data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate OHLC from close price
            noise = np.random.normal(0, 1)
            open_price = close_price + noise
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 2))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 2))
            volume = np.random.randint(1000, 100000)
            
            data.append({
                'timestamp': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, strategy_agent, config):
        """Test agent initialization"""
        assert strategy_agent.symbol == config['trading']['symbol']
        assert strategy_agent.capital == config['trading']['capital']
        assert strategy_agent.max_position == config['risk']['max_position']
        assert strategy_agent.max_drawdown == config['risk']['max_drawdown']
    
    def test_act_with_valid_data(self, strategy_agent, sample_data):
        """Test signal generation with valid data"""
        signal = strategy_agent.act(sample_data)
        
        # Check signal structure
        assert isinstance(signal, dict)
        assert 'action' in signal
        assert 'symbol' in signal
        assert 'quantity' in signal
        assert 'price' in signal
        assert 'confidence' in signal
        
        # Check action values
        assert signal['action'] in ['buy', 'sell', 'hold']
        assert signal['symbol'] == strategy_agent.symbol
        assert signal['quantity'] >= 0
        assert signal['price'] > 0
        assert 0 <= signal['confidence'] <= 1
    
    def test_act_with_empty_data(self, strategy_agent):
        """Test signal generation with empty data"""
        empty_data = pd.DataFrame()
        signal = strategy_agent.act(empty_data)
        
        assert signal['action'] == 'hold'
        assert signal['quantity'] == 0
        assert signal['confidence'] == 0.0
    
    def test_calculate_indicators(self, strategy_agent, sample_data):
        """Test technical indicator calculations"""
        indicators = strategy_agent._calculate_indicators(sample_data)
        
        # Check that indicators are calculated
        expected_indicators = ['sma_20', 'sma_50', 'rsi', 'bb_upper', 'bb_lower', 'macd', 'macd_signal']
        for indicator in expected_indicators:
            assert indicator in indicators
        
        # Check that indicators have reasonable values
        if len(indicators['sma_20']) > 0:
            assert indicators['sma_20'][-1] > 0
        
        if len(indicators['rsi']) > 0:
            rsi_value = indicators['rsi'][-1]
            assert 0 <= rsi_value <= 100
    
    def test_calculate_sma(self, strategy_agent):
        """Test Simple Moving Average calculation"""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Test SMA with window 3
        sma = strategy_agent._calculate_sma(prices, 3)
        expected_sma = np.array([101, 102, 103, 104, 105, 106, 107, 108])
        
        np.testing.assert_array_almost_equal(sma, expected_sma, decimal=2)
        
        # Test with insufficient data
        short_prices = np.array([100, 101])
        sma_short = strategy_agent._calculate_sma(short_prices, 3)
        assert len(sma_short) == 0
    
    def test_calculate_rsi(self, strategy_agent):
        """Test RSI calculation"""
        # Create price data with known pattern
        prices = np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
        
        rsi = strategy_agent._calculate_rsi(prices, window=3)
        
        # RSI should be between 0 and 100
        if len(rsi) > 0:
            assert 0 <= rsi[-1] <= 100
    
    def test_calculate_bollinger_bands(self, strategy_agent):
        """Test Bollinger Bands calculation"""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        bb_upper, bb_lower = strategy_agent._calculate_bollinger_bands(prices, window=5)
        
        if len(bb_upper) > 0 and len(bb_lower) > 0:
            # Upper band should be above lower band
            assert bb_upper[-1] > bb_lower[-1]
    
    def test_calculate_position_size(self, strategy_agent):
        """Test position size calculation"""
        price = 150.0
        
        # Test normal case
        quantity = strategy_agent._calculate_position_size(price)
        expected_quantity = int((strategy_agent.capital * 0.1) / price)
        expected_quantity = min(expected_quantity, strategy_agent.max_position)
        
        assert quantity == expected_quantity
        assert quantity >= 1
        
        # Test with very high price
        high_price = 10000.0
        quantity_high = strategy_agent._calculate_position_size(high_price)
        assert quantity_high == 1  # Minimum quantity
    
    def test_generate_no_action_signal(self, strategy_agent):
        """Test no-action signal generation"""
        signal = strategy_agent._generate_no_action_signal()
        
        assert signal['action'] == 'hold'
        assert signal['quantity'] == 0
        assert signal['price'] == 0
        assert signal['confidence'] == 0.0
        assert signal['symbol'] == strategy_agent.symbol
    
    def test_signal_generation_logic(self, strategy_agent, sample_data):
        """Test signal generation logic with different market conditions"""
        # Test with upward trending data (should generate buy signal)
        upward_data = sample_data.copy()
        upward_data['close'] = upward_data['close'] * 1.1  # 10% increase
        
        signal_up = strategy_agent.act(upward_data)
        
        # Test with downward trending data (should generate sell signal)
        downward_data = sample_data.copy()
        downward_data['close'] = downward_data['close'] * 0.9  # 10% decrease
        
        signal_down = strategy_agent.act(downward_data)
        
        # Both should be valid signals
        assert signal_up['action'] in ['buy', 'sell', 'hold']
        assert signal_down['action'] in ['buy', 'sell', 'hold']
    
    def test_error_handling(self, strategy_agent):
        """Test error handling in signal generation"""
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        # Should not raise exception, should return hold signal
        signal = strategy_agent.act(invalid_data)
        assert signal['action'] == 'hold'
    
    def test_technical_indicators_edge_cases(self, strategy_agent):
        """Test technical indicators with edge cases"""
        # Test with constant prices
        constant_prices = np.ones(50) * 100
        rsi_constant = strategy_agent._calculate_rsi(constant_prices)
        
        # Test with all increasing prices
        increasing_prices = np.arange(100, 150)
        rsi_increasing = strategy_agent._calculate_rsi(increasing_prices)
        
        # Test with all decreasing prices
        decreasing_prices = np.arange(150, 100, -1)
        rsi_decreasing = strategy_agent._calculate_rsi(decreasing_prices)
        
        # All should return valid arrays (possibly empty)
        assert isinstance(rsi_constant, np.ndarray)
        assert isinstance(rsi_increasing, np.ndarray)
        assert isinstance(rsi_decreasing, np.ndarray)
    
    def test_macd_calculation(self, strategy_agent):
        """Test MACD calculation"""
        prices = np.array([100 + i * 0.1 + np.random.normal(0, 1) for i in range(50)])
        
        macd, signal = strategy_agent._calculate_macd(prices)
        
        # Both should be numpy arrays
        assert isinstance(macd, np.ndarray)
        assert isinstance(signal, np.ndarray)
        
        # If we have enough data, both should have values
        if len(prices) >= 26:
            assert len(macd) > 0
            if len(macd) >= 9:  # Need enough data for signal line
                assert len(signal) > 0
    
    def test_ema_calculation(self, strategy_agent):
        """Test Exponential Moving Average calculation"""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        ema = strategy_agent._calculate_ema(prices, window=5)
        
        assert isinstance(ema, np.ndarray)
        if len(ema) > 0:
            assert ema[-1] > 0  # EMA should be positive 