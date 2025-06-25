import pandas as pd
import logging
import os
from typing import Dict, Any
from .synthetic_data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load market data from file or generate synthetic data if needed.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with market data
    """
    logger.info("Starting data ingestion process")
    
    try:
        data_source = config['data_source']
        data_type = data_source['type']
        
        if data_type == 'csv':
            return _load_csv_data(config)
        elif data_type == 'synthetic':
            return _generate_synthetic_data(config)
        else:
            raise ValueError(f"Unsupported data source type: {data_type}")
            
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}", exc_info=True)
        raise

def _load_csv_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data from CSV file"""
    path = config['data_source']['path']
    
    if not os.path.exists(path):
        logger.warning(f"CSV file not found at {path}, generating synthetic data instead")
        return _generate_synthetic_data(config)
    
    logger.info(f"Loading data from CSV: {path}")
    df = pd.read_csv(path)
    
    # Validate data
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns in CSV: {missing_columns}")
        logger.info("Generating synthetic data instead")
        return _generate_synthetic_data(config)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Successfully loaded {len(df)} data points from CSV")
    return df

def _generate_synthetic_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Generate synthetic data using the SyntheticDataGenerator"""
    logger.info("Generating synthetic market data")
    
    try:
        # Create data directory if it doesn't exist
        data_path = config['synthetic_data']['data_path']
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Initialize synthetic data generator
        generator = SyntheticDataGenerator(config)
        
        # Generate OHLCV data
        df = generator.generate_ohlcv_data(
            symbol=config['trading']['symbol'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            frequency=config['trading']['timeframe']
        )
        
        # Save to CSV if configured
        if config['synthetic_data'].get('generate_data', True):
            generator.save_to_csv(df, data_path)
            logger.info(f"Saved synthetic data to {data_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}", exc_info=True)
        raise

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded data for required fields and data quality.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    logger.info("Validating data quality")
    
    try:
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts.to_dict()}")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = df[price_columns].lt(0).any().any()
        if negative_prices:
            logger.error("Found negative prices in data")
            return False
        
        # Check for negative volumes
        if (df['volume'] < 0).any():
            logger.error("Found negative volumes in data")
            return False
        
        # Check OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_ohlc.any():
            logger.error(f"Found {invalid_ohlc.sum()} rows with invalid OHLC data")
            return False
        
        logger.info("Data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error during data validation: {e}", exc_info=True)
        return False
