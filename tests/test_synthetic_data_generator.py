import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from agentic_ai_system.synthetic_data_generator import SyntheticDataGenerator

class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing"""
        return {
            'synthetic_data': {
                'base_price': 100.0,
                'volatility': 0.02,
                'trend': 0.001,
                'noise_level': 0.005
            },
            'trading': {
                'symbol': 'AAPL',
                'timeframe': '1min'
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """Create a SyntheticDataGenerator instance"""
        return SyntheticDataGenerator(config)
    
    def test_initialization(self, generator, config):
        """Test generator initialization"""
        assert generator.base_price == config['synthetic_data']['base_price']
        assert generator.volatility == config['synthetic_data']['volatility']
        assert generator.trend == config['synthetic_data']['trend']
        assert generator.noise_level == config['synthetic_data']['noise_level']
    
    def test_generate_ohlcv_data(self, generator):
        """Test OHLCV data generation"""
        df = generator.generate_ohlcv_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-02',
            frequency='1min'
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns
        
        # Check data types
        assert df['timestamp'].dtype == 'datetime64[ns]'
        assert df['symbol'].dtype == 'object'
        assert df['open'].dtype in ['float64', 'float32']
        assert df['high'].dtype in ['float64', 'float32']
        assert df['low'].dtype in ['float64', 'float32']
        assert df['close'].dtype in ['float64', 'float32']
        assert df['volume'].dtype in ['int64', 'int32']
        
        # Check data validity
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        assert (df['volume'] >= 0).all()
        assert (df['open'] > 0).all()
        assert (df['close'] > 0).all()
    
    def test_generate_tick_data(self, generator):
        """Test tick data generation"""
        df = generator.generate_tick_data(
            symbol='AAPL',
            duration_minutes=10,
            tick_interval_ms=1000
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'price', 'volume']
        for col in required_columns:
            assert col in df.columns
        
        # Check data validity
        assert (df['price'] > 0).all()
        assert (df['volume'] >= 0).all()
        assert df['symbol'].iloc[0] == 'AAPL'
    
    def test_generate_price_series(self, generator):
        """Test price series generation"""
        length = 100
        prices = generator._generate_price_series(length)
        
        assert isinstance(prices, np.ndarray)
        assert len(prices) == length
        assert (prices > 0).all()  # All prices should be positive
    
    def test_save_to_csv(self, generator):
        """Test saving data to CSV"""
        df = generator.generate_ohlcv_data(
            symbol='AAPL',
            start_date='2024-01-01',
            end_date='2024-01-01',
            frequency='1H'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            filepath = tmp_file.name
        
        try:
            generator.save_to_csv(df, filepath)
            
            # Check if file exists and has content
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            # Load and verify data
            loaded_df = pd.read_csv(filepath)
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)
            
        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_market_scenarios(self, generator):
        """Test different market scenarios"""
        scenarios = ['normal', 'volatile', 'trending', 'crash']
        
        for scenario in scenarios:
            df = generator.generate_market_scenarios(scenario)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
            # Check that crash scenario has lower prices on average
            if scenario == 'crash':
                avg_price = df['close'].mean()
                assert avg_price < generator.base_price * 0.9  # Should be significantly lower
    
    def test_invalid_frequency(self, generator):
        """Test handling of invalid frequency"""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            generator.generate_ohlcv_data(frequency='invalid')
    
    def test_invalid_scenario(self, generator):
        """Test handling of invalid scenario"""
        with pytest.raises(ValueError, match="Unknown scenario type"):
            generator.generate_market_scenarios('invalid_scenario')
    
    def test_empty_date_range(self, generator):
        """Test handling of empty date range"""
        df = generator.generate_ohlcv_data(
            start_date='2024-01-01',
            end_date='2024-01-01',
            frequency='1D'
        )
        
        # Should generate at least one data point
        assert len(df) >= 1
    
    def test_different_symbols(self, generator):
        """Test data generation for different symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        for symbol in symbols:
            df = generator.generate_ohlcv_data(symbol=symbol)
            assert df['symbol'].iloc[0] == symbol
    
    def test_price_consistency(self, generator):
        """Test that generated prices are consistent"""
        df = generator.generate_ohlcv_data(
            start_date='2024-01-01',
            end_date='2024-01-02',
            frequency='1H'
        )
        
        # Check that prices are within reasonable bounds
        max_price = df[['open', 'high', 'low', 'close']].max().max()
        min_price = df[['open', 'high', 'low', 'close']].min().min()
        
        # Prices should be within 50% of base price
        assert min_price > generator.base_price * 0.5
        assert max_price < generator.base_price * 1.5
    
    def test_volume_correlation(self, generator):
        """Test that volume correlates with price movement"""
        df = generator.generate_ohlcv_data(
            start_date='2024-01-01',
            end_date='2024-01-02',
            frequency='1H'
        )
        
        # Calculate price movement
        df['price_movement'] = abs(df['close'] - df['open'])
        
        # Check that volume is correlated with price movement
        correlation = df['volume'].corr(df['price_movement'])
        assert not np.isnan(correlation)  # Should have some correlation 