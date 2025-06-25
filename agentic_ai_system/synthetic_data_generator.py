import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Generates synthetic market data for testing and development purposes.
    Creates realistic price movements with volatility, trends, and market noise.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_price = config.get('synthetic_data', {}).get('base_price', 100.0)
        self.volatility = config.get('synthetic_data', {}).get('volatility', 0.02)
        self.trend = config.get('synthetic_data', {}).get('trend', 0.001)
        self.noise_level = config.get('synthetic_data', {}).get('noise_level', 0.005)
        
        logger.info(f"Initialized SyntheticDataGenerator with base_price={self.base_price}, "
                   f"volatility={self.volatility}, trend={self.trend}")
    
    def generate_ohlcv_data(self, 
                           symbol: str = 'AAPL',
                           start_date: str = '2024-01-01',
                           end_date: str = '2024-12-31',
                           frequency: str = '1min') -> pd.DataFrame:
        """
        Generate synthetic OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('1min', '5min', '1H', '1D')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Generating synthetic OHLCV data for {symbol} from {start_date} to {end_date}")
        
        # Create datetime range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate timestamps based on frequency
        if frequency == '1min' or frequency == '1m':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='1min')
        elif frequency == '5min' or frequency == '5m':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='5min')
        elif frequency == '1H' or frequency == '1h':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='1H')
        elif frequency == '1D' or frequency == '1d':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='1D')
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Generate price data
        prices = self._generate_price_series(len(timestamps))
        
        # Generate OHLCV data
        data = []
        current_price = self.base_price
        
        for i, timestamp in enumerate(timestamps):
            # Add trend and noise
            trend_component = self.trend * i
            noise = np.random.normal(0, self.noise_level)
            
            # Generate OHLC from current price
            open_price = current_price * (1 + noise)
            close_price = open_price * (1 + np.random.normal(0, self.volatility))
            
            # Generate high and low
            price_range = abs(close_price - open_price) * np.random.uniform(1.5, 3.0)
            high_price = max(open_price, close_price) + price_range * np.random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - price_range * np.random.uniform(0, 0.5)
            
            # Generate volume (correlated with price movement)
            volume = np.random.randint(1000, 100000) * (1 + abs(close_price - open_price) / open_price)
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
            
            current_price = close_price
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} data points for {symbol}")
        return df
    
    def generate_tick_data(self, 
                          symbol: str = 'AAPL',
                          duration_minutes: int = 60,
                          tick_interval_ms: int = 1000) -> pd.DataFrame:
        """
        Generate high-frequency tick data for testing.
        
        Args:
            symbol: Stock symbol
            duration_minutes: Duration in minutes
            tick_interval_ms: Interval between ticks in milliseconds
            
        Returns:
            DataFrame with tick data
        """
        logger.info(f"Generating tick data for {symbol} for {duration_minutes} minutes")
        
        num_ticks = (duration_minutes * 60 * 1000) // tick_interval_ms
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=num_ticks,
            freq=f'{tick_interval_ms}ms'
        )
        
        # Generate price series with more noise for tick data
        base_prices = self._generate_price_series(num_ticks, volatility=self.volatility * 2)
        
        data = []
        for i, (timestamp, base_price) in enumerate(zip(timestamps, base_prices)):
            # Add micro-movements
            tick_price = base_price * (1 + np.random.normal(0, self.noise_level * 0.5))
            
            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'price': round(tick_price, 4),
                'volume': np.random.randint(1, 100)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} tick data points for {symbol}")
        return df
    
    def _generate_price_series(self, length: int, volatility: Optional[float] = None) -> np.ndarray:
        """
        Generate a realistic price series using geometric Brownian motion.
        
        Args:
            length: Number of price points
            volatility: Price volatility (if None, uses self.volatility)
            
        Returns:
            Array of prices
        """
        if volatility is None:
            volatility = self.volatility
        
        # Geometric Brownian motion parameters
        mu = self.trend  # drift
        sigma = volatility  # volatility
        
        # Generate random walks
        dt = 1.0 / length
        t = np.linspace(0, 1, length)
        
        # Brownian motion
        dW = np.random.normal(0, np.sqrt(dt), length)
        W = np.cumsum(dW)
        
        # Geometric Brownian motion
        S = self.base_price * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
        
        return S
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save generated data to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the CSV file
        """
        df.to_csv(filepath, index=False)
        logger.info(f"Saved synthetic data to {filepath}")
    
    def generate_market_scenarios(self, scenario_type: str = 'normal') -> pd.DataFrame:
        """
        Generate data for different market scenarios.
        
        Args:
            scenario_type: Type of scenario ('normal', 'volatile', 'trending', 'crash')
            
        Returns:
            DataFrame with scenario-specific data
        """
        logger.info(f"Generating {scenario_type} market scenario")
        
        if scenario_type == 'normal':
            return self.generate_ohlcv_data()
        elif scenario_type == 'volatile':
            # High volatility scenario
            self.volatility *= 3
            data = self.generate_ohlcv_data()
            self.volatility /= 3  # Reset
            return data
        elif scenario_type == 'trending':
            # Strong upward trend
            self.trend *= 5
            data = self.generate_ohlcv_data()
            self.trend /= 5  # Reset
            return data
        elif scenario_type == 'crash':
            # Market crash scenario
            original_volatility = self.volatility
            original_trend = self.trend
            
            self.volatility *= 5
            self.trend = -0.01  # Strong downward trend
            
            try:
                data = self.generate_ohlcv_data()
            finally:
                # Reset parameters
                self.volatility = original_volatility
                self.trend = original_trend
            
            return data
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}") 