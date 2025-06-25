import logging
import time
from typing import Dict, Any, Optional
from .agent_base import Agent

class ExecutionAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.broker_api = config['execution']['broker_api']
        self.order_size = config['execution']['order_size']
        self.execution_delay = config.get('execution', {}).get('delay_ms', 100)
        self.success_rate = config.get('execution', {}).get('success_rate', 0.95)
        self.logger.info(f"Execution agent initialized with {self.broker_api} broker")
    
    def act(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trading signal by sending order to broker.
        
        Args:
            signal: Dictionary containing trading signal
            
        Returns:
            Dictionary containing execution result
        """
        try:
            self.logger.info(f"Processing execution signal: {signal['action']} {signal['quantity']} {signal['symbol']}")
            
            # Validate signal
            if not self._validate_signal(signal):
                self.logger.warning("Invalid signal received, skipping execution")
                return self._generate_execution_result(signal, success=False, error="Invalid signal")
            
            # Simulate order execution
            execution_result = self._execute_order(signal)
            
            # Log execution result
            self.log_action(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.log_error(e, "Error in order execution")
            return self._generate_execution_result(signal, success=False, error=str(e))
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        try:
            required_fields = ['action', 'symbol', 'quantity']
            
            # Check required fields
            for field in required_fields:
                if field not in signal:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate action
            if signal['action'] not in ['buy', 'sell', 'hold']:
                self.logger.error(f"Invalid action: {signal['action']}")
                return False
            
            # Validate quantity
            if signal['quantity'] <= 0 and signal['action'] != 'hold':
                self.logger.error(f"Invalid quantity: {signal['quantity']}")
                return False
            
            # Validate symbol
            if not signal['symbol'] or not isinstance(signal['symbol'], str):
                self.logger.error(f"Invalid symbol: {signal['symbol']}")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(e, "Error validating signal")
            return False
    
    def _execute_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order with broker simulation"""
        try:
            # Simulate execution delay
            time.sleep(self.execution_delay / 1000.0)
            
            # Simulate execution success/failure
            import random
            success = random.random() < self.success_rate
            
            if signal['action'] == 'hold':
                success = True  # Hold actions always succeed
            
            if success:
                return self._simulate_successful_execution(signal)
            else:
                return self._simulate_failed_execution(signal)
                
        except Exception as e:
            self.log_error(e, "Error in order execution simulation")
            return self._generate_execution_result(signal, success=False, error=str(e))
    
    def _simulate_successful_execution(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate successful order execution"""
        try:
            # Generate execution details
            execution_price = signal.get('price', 0)
            if execution_price == 0:
                # Simulate price slippage
                import random
                slippage = random.uniform(-0.001, 0.001)  # ±0.1% slippage
                execution_price = signal.get('price', 100) * (1 + slippage)
            
            execution_time = time.time()
            
            # Calculate fees (simplified)
            commission = self._calculate_commission(signal)
            
            result = {
                'order_id': self._generate_order_id(),
                'status': 'filled',
                'action': signal['action'],
                'symbol': signal['symbol'],
                'quantity': signal['quantity'],
                'price': round(execution_price, 4),
                'execution_time': execution_time,
                'commission': commission,
                'total_value': round(signal['quantity'] * execution_price, 2),
                'success': True,
                'error': None
            }
            
            self.logger.info(f"Order executed successfully: {result['order_id']} - "
                           f"{result['action']} {result['quantity']} {result['symbol']} @ {result['price']}")
            
            return result
            
        except Exception as e:
            self.log_error(e, "Error in successful execution simulation")
            return self._generate_execution_result(signal, success=False, error=str(e))
    
    def _simulate_failed_execution(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate failed order execution"""
        error_reasons = [
            "Insufficient funds",
            "Market closed",
            "Invalid order",
            "Network timeout",
            "Broker error"
        ]
        
        import random
        error_reason = random.choice(error_reasons)
        
        result = self._generate_execution_result(signal, success=False, error=error_reason)
        
        self.logger.warning(f"Order execution failed: {error_reason}")
        
        return result
    
    def _generate_execution_result(self, signal: Dict[str, Any], success: bool, error: Optional[str] = None) -> Dict[str, Any]:
        """Generate execution result"""
        return {
            'order_id': self._generate_order_id() if success else None,
            'status': 'filled' if success else 'rejected',
            'action': signal.get('action', 'unknown'),
            'symbol': signal.get('symbol', 'unknown'),
            'quantity': signal.get('quantity', 0),
            'price': signal.get('price', 0) if success else 0,  # Price is 0 for failed executions
            'execution_time': time.time(),
            'commission': 0,
            'total_value': 0,
            'success': success,
            'error': error
        }
    
    def _calculate_commission(self, signal: Dict[str, Any]) -> float:
        """Calculate commission for the order"""
        try:
            # Simple commission calculation
            base_commission = 1.0  # $1 base commission
            per_share_commission = 0.01  # $0.01 per share
            
            if signal['action'] == 'hold':
                return 0.0
            
            commission = base_commission + (signal['quantity'] * per_share_commission)
            return round(commission, 2)
            
        except Exception as e:
            self.log_error(e, "Error calculating commission")
            return 0.0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        import uuid
        return f"ORD_{uuid.uuid4().hex[:8].upper()}"
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        # This would typically track real execution statistics
        # For now, return placeholder data
        return {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'success_rate': 0.0,
            'average_execution_time': 0.0,
            'total_commission': 0.0
        }
