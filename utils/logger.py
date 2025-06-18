"""
Logging configuration for the pairs trading system.
"""
import logging
import logging.handlers
import os
from datetime import datetime

def setup_logger(name='pairs_trading', log_level='INFO'):
    """Setup logger with file and console handlers."""
    
    # Default configuration to avoid circular imports
    log_file = './logs/pairs_trading.log'
    max_file_size = 10 * 1024 * 1024  # 10MB
    backup_count = 5
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue with console only
        pass
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def update_logger_config(config):
    """Update logger configuration after config is loaded."""
    try:
        from config.settings import CONFIG
        logger = logging.getLogger('pairs_trading')
        
        # Update log level
        logger.setLevel(getattr(logging, CONFIG['logging']['log_level'].upper()))
        
        # Update file handler if needed
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.close()
                logger.removeHandler(handler)
                break
        
        # Add new file handler with updated config
        file_handler = logging.handlers.RotatingFileHandler(
            CONFIG['logging']['log_file'], 
            maxBytes=CONFIG['logging']['max_file_size'], 
            backupCount=CONFIG['logging']['backup_count']
        )
        file_handler.setLevel(getattr(logging, CONFIG['logging']['log_level'].upper()))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"Warning: Could not update logger config: {e}")

# Global logger instance
logger = setup_logger()