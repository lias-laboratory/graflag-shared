"""
Logging utilities for GraFlag methods.

Provides consistent logging functions across all methods.
"""

import logging as _logging
import os

# Configure logging once at module import
_method_name = os.environ.get("METHOD_NAME", "unknown_method")
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)
_logger = _logging.getLogger(_method_name)


def debug(msg, *args, **kwargs):
    """Log a debug message."""
    _logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log an info message."""
    _logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a warning message."""
    _logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log an error message."""
    _logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log a critical message."""
    _logger.critical(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    """Log an exception message with traceback."""
    _logger.exception(msg, *args, **kwargs)
