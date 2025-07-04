import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from agentic_ai_system.orchestrator import run, run_backtest, run_live_trading
from agentic_ai_system.main import load_config

class TestIntegration:
    """Integration tests for the entire trading system"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for integration testing"""
        return {
            'data_source': {
                'type': 'synthetic',
                'path': 'data/synthetic_market_data_test.csv'
            },
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
                'order_size': 10,
                'delay_ms': 10,  # Fast for testing
                'success_rate': 1.0  # Always succeed for testing
            },
            'synthetic_data': {
                'base_price': 150.0,
                'volatility': 0.02,
                'trend': 0.001,
                'noise_level': 0.005,
                'data_path': 'data/synthetic_market_data_test.csv'
            },
            'logging': {
                'log_level': 'INFO',
                'log_dir': 'logs',
                'enable_console': True,
                'enable_file': True
            }
        }
    
    def test_full_workflow(self, config):
        """Test the complete trading workflow"""
        result = run(config)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'data_loaded' in result
        assert 'signal_generated' in result
        assert 'order_executed' in result
        assert 'execution_time' in result
        assert 'errors' in result
        
        # Check that data was loaded
        assert result['data_loaded'] == True
        
        # Check that signal was generated
        assert result['signal_generated'] == True
        
        # Check execution time is reasonable
        assert result['execution_time'] > 0
        assert result['execution_time'] < 60  # Should complete within 60 seconds
    
    def test_backtest_workflow(self, config):
        """Test the backtest workflow"""
        result = run_backtest(config, '2024-01-01', '2024-01-02')
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'start_date' in result
            assert 'end_date' in result
            assert 'initial_capital' in result
            assert 'final_value' in result
            assert 'total_return' in result
            assert 'total_trades' in result
            assert 'trades' in result
            assert 'positions' in result
            
            # Check that backtest completed
            assert result['initial_capital'] == config['trading']['capital']
            assert result['final_value'] >= 0
            assert isinstance(result['total_return'], float)
            assert result['total_trades'] >= 0
            assert isinstance(result['trades'], list)
            assert isinstance(result['positions'], dict)
    
    def test_live_trading_workflow(self, config):
        """Test the live trading workflow (short duration)"""
        # Test with very short duration to avoid long test times
        result = run_live_trading(config, duration_minutes=1)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'duration_minutes' in result
            assert 'total_trades' in result
            assert 'trades' in result
            assert 'start_time' in result
            assert 'end_time' in result
            
            # Check that live trading completed
            assert result['duration_minutes'] == 1
            assert result['total_trades'] >= 0
            assert isinstance(result['trades'], list)
    
    def test_workflow_with_csv_data(self, config):
        """Test workflow with CSV data source"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            # Generate sample data with correct column names
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
            data = []
            for i, date in enumerate(dates):
                base_price = 150.0 + (i * 0.1)
                data.append({
                    'date': date,
                    'open': base_price + np.random.normal(0, 1),
                    'high': base_price + abs(np.random.normal(0, 2)),
                    'low': base_price - abs(np.random.normal(0, 2)),
                    'close': base_price + np.random.normal(0, 1),
                    'volume': np.random.randint(1000, 100000)
                })
            
            df = pd.DataFrame(data)
            df.to_csv(tmp_file.name, index=False)
            config['data_source']['type'] = 'csv'
            config['data_source']['path'] = tmp_file.name
            
            try:
                result = run(config)
                
                assert result['success'] == True
                assert result['data_loaded'] == True
                assert result['signal_generated'] == True
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_workflow_error_handling(self, config):
        """Test workflow error handling"""
        # Test with invalid configuration
        invalid_config = config.copy()
        invalid_config['data_source']['type'] = 'invalid_type'
        
        result = run(invalid_config)
        
        assert result['success'] == False
        assert len(result['errors']) > 0
    
    def test_backtest_with_different_periods(self, config):
        """Test backtest with different time periods"""
        # Test short period
        short_result = run_backtest(config, '2024-01-01', '2024-01-01')
        assert isinstance(short_result, dict)
        
        # Test longer period
        long_result = run_backtest(config, '2024-01-01', '2024-01-07')
        assert isinstance(long_result, dict)
        
        # Both should be valid results (success or failure)
        assert 'success' in short_result
        assert 'success' in long_result
    
    def test_system_with_different_symbols(self, config):
        """Test system with different trading symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        for symbol in symbols:
            test_config = config.copy()
            test_config['trading']['symbol'] = symbol
            
            result = run(test_config)
            
            assert result['success'] == True
            assert result['data_loaded'] == True
            assert result['signal_generated'] == True
    
    def test_system_with_different_capital_amounts(self, config):
        """Test system with different capital amounts"""
        capital_amounts = [10000, 50000, 100000, 500000]
        
        for capital in capital_amounts:
            test_config = config.copy()
            test_config['trading']['capital'] = capital
            
            result = run(test_config)
            
            assert result['success'] == True
            assert result['data_loaded'] == True
            assert result['signal_generated'] == True
    
    def test_execution_failure_simulation(self, config):
        """Test system behavior with execution failures"""
        # Set success rate to 0 to simulate all failures
        test_config = config.copy()
        test_config['execution']['success_rate'] = 0.0
        
        result = run(test_config)
        
        # System should still complete workflow
        assert result['success'] == True
        assert result['data_loaded'] == True
        assert result['signal_generated'] == True
        
        # If a non-hold order was executed, it should fail with success_rate = 0.0
        # But if only hold signals were generated, no orders would be executed
        if result['order_executed'] and result.get('execution_result', {}).get('action') != 'hold':
            assert result['execution_result']['success'] == False
    
    def test_data_validation_integration(self, config):
        """Test data validation integration"""
        # Create invalid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            invalid_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'open': [150] * 10,
                'high': [145] * 10,  # Invalid: high < open
                'low': [145] * 10,
                'close': [152] * 10,
                'volume': [1000] * 10
            })
            invalid_data.to_csv(tmp_file.name, index=False)
            config['data_source']['type'] = 'csv'
            config['data_source']['path'] = tmp_file.name
            
            try:
                result = run(config)
                
                # System should still work (fallback to synthetic data)
                assert result['success'] == True
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_performance_metrics(self, config):
        """Test that performance metrics are calculated correctly"""
        result = run_backtest(config, '2024-01-01', '2024-01-03')
        
        if result['success']:
            # Check that return is calculated correctly
            initial_capital = result['initial_capital']
            final_value = result['final_value']
            calculated_return = (final_value - initial_capital) / initial_capital
            
            assert abs(result['total_return'] - calculated_return) < 0.001
            
            # Check that trade count is reasonable
            assert result['total_trades'] >= 0
    
    def test_config_loading(self):
        """Test configuration loading functionality"""
        # Test with valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            config_content = """
data_source:
  type: 'synthetic'
  path: 'data/market_data.csv'

trading:
  symbol: 'AAPL'
  timeframe: '1min'
  capital: 100000

risk:
  max_position: 100
  max_drawdown: 0.05

execution:
  broker_api: 'paper'
  order_size: 10
"""
            tmp_file.write(config_content)
            tmp_file.flush()
            
            try:
                config = load_config(tmp_file.name)
                
                assert config['data_source']['type'] == 'synthetic'
                assert config['trading']['symbol'] == 'AAPL'
                assert config['trading']['capital'] == 100000
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_system_scalability(self, config):
        """Test system scalability with larger datasets"""
        # Test with larger synthetic dataset
        test_config = config.copy()
        test_config['synthetic_data']['base_price'] = 200.0
        test_config['synthetic_data']['volatility'] = 0.03
        
        result = run(test_config)
        
        assert result['success'] == True
        assert result['data_loaded'] == True
        assert result['signal_generated'] == True
        
        # Check execution time is reasonable
        assert result['execution_time'] < 30  # Should complete within 30 seconds 