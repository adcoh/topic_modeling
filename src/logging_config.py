"""
Logging configuration for WCSVNtm pipeline.

This module provides centralized logging configuration with support for
both console and file output, configurable log levels, and third-party
library noise suppression.
"""

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Set up logging configuration with both console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output

    Raises:
        ValueError: If the provided logging level is invalid

    Example:
        >>> from WCSVNtm.logging_config import setup_logging
        >>> setup_logging(level="DEBUG", log_file="wcsvntm.log")
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    logger = logging.getLogger()
    try:
        log_level = getattr(logging, level.upper())
    except AttributeError as e:
        raise ValueError(f"Invalid logging level: {level}") from e
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set specific logger levels to reduce noise
    logging.getLogger("nltk").setLevel(logging.WARNING)
    logging.getLogger("networkx").setLevel(logging.WARNING)
    logging.getLogger("polars").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for quick setup
def quick_setup(level: str = "INFO") -> None:
    """
    Quick setup with default console logging only.

    Args:
        level: Logging level (default: INFO)
    """
    setup_logging(level=level)
