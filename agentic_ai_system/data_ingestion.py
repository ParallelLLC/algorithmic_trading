import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load market data based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with market data or None if error
    """
    try:
        data_source = config['data_source']['type']
        logger.info(f"Loading data from source: {data_source}")
        
        if data_source == 'alpaca':
            return _load_alpaca_data(config)
        elif data_source == 'csv':
            return _load_csv_data(config)
        elif data_source == 'synthetic':
            return _load_synthetic_data(config)
        else:
            logger.error(f"Unsupported data source: {data_source}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def _load_alpaca_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load market data from Alpaca"""
    try:
        from .alpaca_broker import AlpacaBroker
        
        # Initialize Alpaca broker
        alpaca_broker = AlpacaBroker(config)
        
        # Get symbol and timeframe from config
        symbol = config['trading']['symbol']
        timeframe = config['trading']['timeframe']
        
        # Convert timeframe to Alpaca format
        tf_map = {
            '1m': '1Min',
            '5m': '5Min', 
            '15m': '15Min',
            '1h': '1Hour',
            '1d': '1Day'
        }
        alpaca_timeframe = tf_map.get(timeframe, '1Min')
        
        # Get market data
        data = alpaca_broker.get_market_data(
            symbol=symbol,
            timeframe=alpaca_timeframe,
            limit=1000
        )
        
        if data is not None and not data.empty:
            logger.info(f"Loaded {len(data)} data points from Alpaca for {symbol}")
            return data
        else:
            logger.error(f"No data returned from Alpaca for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading Alpaca data: {e}")
        return None

def _load_csv_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load market data from CSV file"""
    try:
        file_path = config['data_source']['path']
        
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            return None
        
        # Load CSV data
        data = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded {len(data)} data points from CSV: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

def _load_synthetic_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load or generate synthetic market data"""
    try:
        synthetic_config = config.get('synthetic_data', {})
        data_path = synthetic_config.get('data_path', 'data/synthetic_market_data.csv')
        
        # Check if synthetic data file exists
        if os.path.exists(data_path):
            logger.info(f"Loading existing synthetic data from: {data_path}")
            return _load_csv_data({'data_source': {'path': data_path}})
        
        # Generate new synthetic data
        logger.info("Generating new synthetic market data")
        from .synthetic_data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(config)
        data = generator.generate_data()
        
        # Save generated data
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        data.to_csv(data_path, index=False)
        logger.info(f"Saved synthetic data to: {data_path}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading synthetic data: {e}")
        return None

def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate market data quality.
    
    Args:
        data: DataFrame with market data
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        if data is None or data.empty:
            logger.error("Data is None or empty")
            return False
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values
        nan_counts = data[required_columns].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values: {nan_counts.to_dict()}")
            # Remove rows with NaN values
            data.dropna(subset=required_columns, inplace=True)
            logger.info(f"Removed {nan_counts.sum()} rows with NaN values")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = data[price_columns] < 0
        if negative_prices.any().any():
            logger.error("Found negative prices in data")
            return False
        
        # Check for zero volumes
        zero_volumes = data['volume'] == 0
        if zero_volumes.sum() > len(data) * 0.5:  # More than 50% zero volumes
            logger.warning("High percentage of zero volumes detected")
        
        # Check OHLC consistency
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['open'] > data['high']) |
            (data['close'] > data['high']) |
            (data['open'] < data['low']) |
            (data['close'] < data['low'])
        )
        
        if invalid_ohlc.any():
            logger.error("Found invalid OHLC relationships")
            return False
        
        # Check timestamp consistency
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            if not timestamps.is_monotonic_increasing:
                logger.warning("Timestamps are not in ascending order")
                data = data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data validation passed: {len(data)} valid records")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to market data.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators added
    """
    try:
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        
        logger.info("Technical indicators added successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        return data

def get_latest_data(data: pd.DataFrame, n_periods: int = 100) -> pd.DataFrame:
    """
    Get the latest n periods of data.
    
    Args:
        data: DataFrame with market data
        n_periods: Number of periods to return
        
    Returns:
        DataFrame with latest n periods
    """
    try:
        if len(data) <= n_periods:
            return data
        
        return data.tail(n_periods).reset_index(drop=True)
        
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return data
