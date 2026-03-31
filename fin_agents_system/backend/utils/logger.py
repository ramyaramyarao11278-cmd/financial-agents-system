import logging
import os
from concurrent_log_handler import ConcurrentRotatingFileHandler  # 修改这里

from config import config


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger if it doesn't exist
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Set logger level
        logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if LOG_FILE is configured
        if config.LOG_FILE:
            # Ensure log directory exists
            log_dir = os.path.dirname(config.LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Create rotating file handler with UTF-8 encoding
            file_handler = ConcurrentRotatingFileHandler(
                config.LOG_FILE,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


# Example usage
if __name__ == "__main__":
    logger = get_logger("example")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
