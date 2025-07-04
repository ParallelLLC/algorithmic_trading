"""
Alpaca Broker Integration for Algorithmic Trading

This module provides integration with Alpaca Markets for real trading capabilities,
including paper trading and live trading support.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment
    from alpaca.account import Account
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca SDK not available. Install with: pip install alpaca-py")

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Alpaca broker integration for algorithmic trading
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpaca broker connection
        
        Args:
            config: Configuration dictionary containing Alpaca settings
        """
        self.config = config
        self.alpaca_config = config.get('alpaca', {})
        
        # Get API credentials from environment or config
        self.api_key = os.getenv('ALPACA_API_KEY') or self.alpaca_config.get('api_key', '')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY') or self.alpaca_config.get('secret_key', '')
        
        # Determine if using paper or live trading
        self.paper_trading = self.alpaca_config.get('paper_trading', True)
        self.account_type = self.alpaca_config.get('account_type', 'paper')
        
        # Set URLs based on account type
        if self.account_type == 'live':
            self.base_url = self.alpaca_config.get('live_url', 'https://api.alpaca.markets')
            self.data_url = self.alpaca_config.get('data_url', 'https://data.alpaca.markets')
        else:
            self.base_url = self.alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
            self.data_url = self.alpaca_config.get('data_url', 'https://data.alpaca.markets')
        
        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.account = None
        
        # Initialize connection
        self._initialize_connection()
        
        logger.info(f"Alpaca broker initialized for {self.account_type} trading")
    
    def _initialize_connection(self):
        """Initialize Alpaca API connections"""
        if not ALPACA_AVAILABLE:
            logger.error("Alpaca SDK not available")
            return False
        
        if not self.api_key or not self.secret_key:
            logger.error("Alpaca API credentials not provided")
            return False
        
        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_trading
            )
            
            # Initialize data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            # Get account information
            self.account = self.trading_client.get_account()
            
            logger.info(f"Connected to Alpaca {self.account_type} account: {self.account.id}")
            logger.info(f"Account status: {self.account.status}")
            logger.info(f"Buying power: ${self.account.buying_power}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca connection: {e}")
            return False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.account:
            return {}
        
        return {
            'account_id': self.account.id,
            'status': self.account.status,
            'buying_power': float(self.account.buying_power),
            'cash': float(self.account.cash),
            'portfolio_value': float(self.account.portfolio_value),
            'equity': float(self.account.equity),
            'daytrade_count': self.account.daytrade_count,
            'trading_blocked': self.account.trading_blocked,
            'transfers_blocked': self.account.transfers_blocked,
            'account_blocked': self.account.account_blocked
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        if not self.trading_client:
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'current_price': float(pos.current_price)
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_market_data(self, symbol: str, timeframe: str = '1Min', 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get historical market data from Alpaca
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (1Min, 5Min, 15Min, 1Hour, 1Day)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        if not self.data_client:
            logger.error("Data client not initialized")
            return None
        
        try:
            # Convert timeframe string to TimeFrame enum
            tf_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame.Minute_5,
                '15Min': TimeFrame.Minute_15,
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day
            }
            
            time_frame = tf_map.get(timeframe, TimeFrame.Minute)
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().isoformat()
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=time_frame,
                start=start_date,
                end=end_date,
                adjustment=Adjustment.ALL,
                limit=limit
            )
            
            # Get data
            bars = self.data_client.get_stock_bars(request)
            
            if bars and symbol in bars:
                # Convert to DataFrame
                df = bars[symbol].df
                df = df.reset_index()
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
                
                # Select relevant columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                
                logger.info(f"Retrieved {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol: str, quantity: int, side: str) -> Dict[str, Any]:
        """
        Place a market order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            
        Returns:
            Order result dictionary
        """
        if not self.trading_client:
            return {'success': False, 'error': 'Trading client not initialized'}
        
        try:
            # Convert side to Alpaca enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )
            
            # Place order
            order = self.trading_client.submit_order(order_request)
            
            # Wait for order to be processed
            time.sleep(1)
            
            # Get updated order status
            order = self.trading_client.get_order_by_id(order.id)
            
            result = {
                'success': order.status == 'filled',
                'order_id': order.id,
                'status': order.status,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side.value,
                'filled_quantity': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                'error': None
            }
            
            if order.status == 'rejected':
                result['error'] = 'Order rejected'
            
            logger.info(f"Market order placed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return {
                'success': False,
                'order_id': None,
                'status': 'error',
                'error': str(e)
            }
    
    def place_limit_order(self, symbol: str, quantity: int, side: str, 
                         limit_price: float) -> Dict[str, Any]:
        """
        Place a limit order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price for the order
            
        Returns:
            Order result dictionary
        """
        if not self.trading_client:
            return {'success': False, 'error': 'Trading client not initialized'}
        
        try:
            # Convert side to Alpaca enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Create order request
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            # Place order
            order = self.trading_client.submit_order(order_request)
            
            result = {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'symbol': order.symbol,
                'quantity': int(order.qty),
                'side': order.side.value,
                'limit_price': float(order.limit_price),
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'error': None
            }
            
            logger.info(f"Limit order placed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return {
                'success': False,
                'order_id': None,
                'status': 'error',
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        if not self.trading_client:
            return {'success': False, 'error': 'Trading client not initialized'}
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return {'success': True, 'order_id': order_id, 'status': 'cancelled'}
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {'success': False, 'order_id': order_id, 'error': str(e)}
    
    def get_orders(self, status: str = 'all') -> List[Dict[str, Any]]:
        """Get order history"""
        if not self.trading_client:
            return []
        
        try:
            orders = self.trading_client.get_orders(status=status)
            return [
                {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'quantity': int(order.qty),
                    'side': order.side.value,
                    'status': order.status,
                    'order_type': order.order_type.value,
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if order.filled_at else None
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        if not self.trading_client:
            return False
        
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours information"""
        if not self.trading_client:
            return {}
        
        try:
            clock = self.trading_client.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat() if clock.next_open else None,
                'next_close': clock.next_close.isoformat() if clock.next_close else None,
                'timestamp': clock.timestamp.isoformat() if clock.timestamp else None
            }
        except Exception as e:
            logger.error(f"Error getting market hours: {e}")
            return {} 