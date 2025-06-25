import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict, Optional

def setup_logging(config: Dict, log_level: str = 'INFO') -> None:
    """
    Set up comprehensive logging for the trading system.
    
    Args:
        config: Configuration dictionary
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    all_logs_file = os.path.join(log_dir, 'trading_system.log')
    file_handler = logging.handlers.RotatingFileHandler(
        all_logs_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file
    error_log_file = os.path.join(log_dir, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Trading-specific log file
    trading_log_file = os.path.join(log_dir, 'trading.log')
    trading_handler = logging.handlers.RotatingFileHandler(
        trading_log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(detailed_formatter)
    
    # Create trading logger
    trading_logger = logging.getLogger('trading')
    trading_logger.addHandler(trading_handler)
    trading_logger.setLevel(logging.INFO)
    trading_logger.propagate = False
    
    # Performance log file
    performance_log_file = os.path.join(log_dir, 'performance.log')
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(detailed_formatter)
    
    # Create performance logger
    performance_logger = logging.getLogger('performance')
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = False
    
    logging.info(f"Logging system initialized. Log files in: {log_dir}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_trade(logger: logging.Logger, trade_data: Dict) -> None:
    """
    Log trade execution details.
    
    Args:
        logger: Logger instance
        trade_data: Dictionary containing trade information
    """
    logger.info(f"TRADE EXECUTED: {trade_data}")

def log_performance(logger: logging.Logger, performance_data: Dict) -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        performance_data: Dictionary containing performance metrics
    """
    perf_logger = logging.getLogger('performance')
    perf_logger.info(f"PERFORMANCE: {performance_data}")

def log_error(logger: logging.Logger, error: Exception, context: Optional[str] = None) -> None:
    """
    Log errors with context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_msg = f"ERROR: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    logger.error(error_msg, exc_info=True) 