import pytest
import time
from unittest.mock import patch, MagicMock
from agentic_ai_system.execution_agent import ExecutionAgent

class TestExecutionAgent:
    """Test cases for ExecutionAgent"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration for testing"""
        return {
            'execution': {
                'broker_api': 'paper',
                'order_size': 10,
                'delay_ms': 50,
                'success_rate': 0.95
            },
            'trading': {
                'symbol': 'AAPL',
                'timeframe': '1min',
                'capital': 100000
            },
            'risk': {
                'max_position': 100,
                'max_drawdown': 0.05
            }
        }
    
    @pytest.fixture
    def execution_agent(self, config):
        """Create an ExecutionAgent instance"""
        return ExecutionAgent(config)
    
    @pytest.fixture
    def valid_signal(self):
        """Create a valid trading signal"""
        return {
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': 10,
            'price': 150.0,
            'confidence': 0.8
        }
    
    def test_initialization(self, execution_agent, config):
        """Test agent initialization"""
        assert execution_agent.broker_api == config['execution']['broker_api']
        assert execution_agent.order_size == config['execution']['order_size']
        assert execution_agent.execution_delay == config['execution']['delay_ms']
        assert execution_agent.success_rate == config['execution']['success_rate']
    
    def test_act_with_valid_signal(self, execution_agent, valid_signal):
        """Test order execution with valid signal"""
        result = execution_agent.act(valid_signal)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'order_id' in result
        assert 'status' in result
        assert 'action' in result
        assert 'symbol' in result
        assert 'quantity' in result
        assert 'price' in result
        assert 'execution_time' in result
        assert 'commission' in result
        assert 'total_value' in result
        assert 'success' in result
        assert 'error' in result
        
        # Check values
        assert result['action'] == valid_signal['action']
        assert result['symbol'] == valid_signal['symbol']
        assert result['quantity'] == valid_signal['quantity']
        assert result['price'] > 0
        assert result['execution_time'] > 0
        assert result['commission'] >= 0
        assert result['total_value'] >= 0
    
    def test_act_with_hold_signal(self, execution_agent):
        """Test order execution with hold signal"""
        hold_signal = {
            'action': 'hold',
            'symbol': 'AAPL',
            'quantity': 0,
            'price': 0,
            'confidence': 0.0
        }
        
        result = execution_agent.act(hold_signal)
        
        assert result['action'] == 'hold'
        assert result['quantity'] == 0
        assert result['success'] == True  # Hold should always succeed
    
    def test_validate_signal_valid(self, execution_agent, valid_signal):
        """Test signal validation with valid signal"""
        assert execution_agent._validate_signal(valid_signal) == True
    
    def test_validate_signal_missing_fields(self, execution_agent):
        """Test signal validation with missing fields"""
        invalid_signal = {'action': 'buy'}  # Missing symbol and quantity
        
        assert execution_agent._validate_signal(invalid_signal) == False
    
    def test_validate_signal_invalid_action(self, execution_agent):
        """Test signal validation with invalid action"""
        invalid_signal = {
            'action': 'invalid_action',
            'symbol': 'AAPL',
            'quantity': 10
        }
        
        assert execution_agent._validate_signal(invalid_signal) == False
    
    def test_validate_signal_invalid_quantity(self, execution_agent):
        """Test signal validation with invalid quantity"""
        invalid_signal = {
            'action': 'buy',
            'symbol': 'AAPL',
            'quantity': -5  # Negative quantity
        }
        
        assert execution_agent._validate_signal(invalid_signal) == False
    
    def test_validate_signal_invalid_symbol(self, execution_agent):
        """Test signal validation with invalid symbol"""
        invalid_signal = {
            'action': 'buy',
            'symbol': '',  # Empty symbol
            'quantity': 10
        }
        
        assert execution_agent._validate_signal(invalid_signal) == False
    
    def test_calculate_commission(self, execution_agent):
        """Test commission calculation"""
        # Test buy order
        buy_signal = {'action': 'buy', 'quantity': 10}
        commission_buy = execution_agent._calculate_commission(buy_signal)
        
        # Base commission ($1) + per share commission ($0.01 * 10) = $1.10
        expected_commission = 1.0 + (10 * 0.01)
        assert commission_buy == expected_commission
        
        # Test sell order
        sell_signal = {'action': 'sell', 'quantity': 5}
        commission_sell = execution_agent._calculate_commission(sell_signal)
        
        expected_commission = 1.0 + (5 * 0.01)
        assert commission_sell == expected_commission
        
        # Test hold order (no commission)
        hold_signal = {'action': 'hold', 'quantity': 0}
        commission_hold = execution_agent._calculate_commission(hold_signal)
        
        assert commission_hold == 0.0
    
    def test_generate_order_id(self, execution_agent):
        """Test order ID generation"""
        order_id = execution_agent._generate_order_id()
        
        assert isinstance(order_id, str)
        assert order_id.startswith('ORD_')
        assert len(order_id) == 12  # 'ORD_' + 8 hex characters
    
    def test_simulate_successful_execution(self, execution_agent, valid_signal):
        """Test successful execution simulation"""
        result = execution_agent._simulate_successful_execution(valid_signal)
        
        assert result['status'] == 'filled'
        assert result['success'] == True
        assert result['error'] is None
        assert result['order_id'] is not None
        assert result['price'] > 0
        assert result['total_value'] > 0
    
    def test_simulate_failed_execution(self, execution_agent, valid_signal):
        """Test failed execution simulation"""
        result = execution_agent._simulate_failed_execution(valid_signal)
        
        assert result['status'] == 'rejected'
        assert result['success'] == False
        assert result['error'] is not None
        assert result['order_id'] is None
        assert result['price'] == 0
        assert result['total_value'] == 0
    
    def test_generate_execution_result(self, execution_agent, valid_signal):
        """Test execution result generation"""
        # Test successful result
        success_result = execution_agent._generate_execution_result(valid_signal, True)
        
        assert success_result['status'] == 'filled'
        assert success_result['success'] == True
        assert success_result['order_id'] is not None
        
        # Test failed result
        failed_result = execution_agent._generate_execution_result(valid_signal, False, "Test error")
        
        assert failed_result['status'] == 'rejected'
        assert failed_result['success'] == False
        assert failed_result['error'] == "Test error"
        assert failed_result['order_id'] is None
    
    def test_execution_delay(self, execution_agent, valid_signal):
        """Test that execution delay is applied"""
        start_time = time.time()
        
        with patch('time.sleep') as mock_sleep:
            execution_agent._execute_order(valid_signal)
            mock_sleep.assert_called_once()
            
            # Check that sleep was called with the correct delay
            call_args = mock_sleep.call_args[0][0]
            expected_delay = execution_agent.execution_delay / 1000.0
            assert abs(call_args - expected_delay) < 0.001
    
    def test_success_rate_simulation(self, execution_agent, valid_signal):
        """Test success rate simulation"""
        # Set success rate to 0.0 (should always fail)
        execution_agent.success_rate = 0.0
        
        with patch('random.random', return_value=0.5):  # Always above 0.0
            result = execution_agent._execute_order(valid_signal)
            assert result['success'] == False
        
        # Set success rate to 1.0 (should always succeed)
        execution_agent.success_rate = 1.0
        
        with patch('random.random', return_value=0.5):  # Always below 1.0
            result = execution_agent._execute_order(valid_signal)
            assert result['success'] == True
    
    def test_error_handling_in_execution(self, execution_agent, valid_signal):
        """Test error handling during execution"""
        # Mock _simulate_successful_execution to raise an exception
        with patch.object(execution_agent, '_simulate_successful_execution', side_effect=Exception("Test error")):
            result = execution_agent._execute_order(valid_signal)
            
            assert result['success'] == False
            assert "Test error" in result['error']
    
    def test_get_execution_statistics(self, execution_agent):
        """Test execution statistics retrieval"""
        stats = execution_agent.get_execution_statistics()
        
        expected_keys = [
            'total_orders', 'successful_orders', 'failed_orders',
            'success_rate', 'average_execution_time', 'total_commission'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check default values
        assert stats['total_orders'] == 0
        assert stats['successful_orders'] == 0
        assert stats['failed_orders'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time'] == 0.0
        assert stats['total_commission'] == 0.0
    
    def test_price_slippage_simulation(self, execution_agent, valid_signal):
        """Test price slippage simulation"""
        # Mock random.uniform to return a known slippage value
        with patch('random.uniform', return_value=0.001):  # 0.1% slippage
            result = execution_agent._simulate_successful_execution(valid_signal)
            
            # Price should be slightly different from original
            original_price = valid_signal['price']
            executed_price = result['price']
            
            # Should be within 0.2% of original price
            price_diff = abs(executed_price - original_price) / original_price
            assert price_diff <= 0.002
    
    def test_commission_calculation_edge_cases(self, execution_agent):
        """Test commission calculation edge cases"""
        # Test with zero quantity
        zero_signal = {'action': 'buy', 'quantity': 0}
        commission_zero = execution_agent._calculate_commission(zero_signal)
        assert commission_zero == 1.0  # Only base commission
        
        # Test with very large quantity
        large_signal = {'action': 'sell', 'quantity': 10000}
        commission_large = execution_agent._calculate_commission(large_signal)
        expected_large = 1.0 + (10000 * 0.01)
        assert commission_large == expected_large
    
    def test_order_id_uniqueness(self, execution_agent):
        """Test that order IDs are unique"""
        order_ids = set()
        
        for _ in range(100):
            order_id = execution_agent._generate_order_id()
            order_ids.add(order_id)
        
        # All order IDs should be unique
        assert len(order_ids) == 100 