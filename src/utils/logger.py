"""Logging utility for Travel Assistant"""

import logging
import sys
from pathlib import Path
from src.config import Config


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(Config.LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(Config.LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = Config.LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(Config.LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
