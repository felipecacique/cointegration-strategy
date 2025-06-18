"""
Simple logger module to avoid circular imports.
"""
import logging
import sys
import os

def get_logger(name='pairs_trading'):
    """Get a simple logger instance."""
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Try to add file handler if possible
        try:
            if not os.path.exists('logs'):
                os.makedirs('logs')
            
            file_handler = logging.FileHandler('logs/pairs_trading.log')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except:
            # File logging not available, continue with console only
            pass
    
    return logger

# Global logger instance
logger = get_logger()