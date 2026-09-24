"""
Logging utilities for GraFlag methods.

Provides consistent logging functions across all methods.
"""

import logging as _logging
import os

# Configure logging once at module import
_method_name = os.environ.get("METHOD_NAME", "unknown_method")
# force=True removes and closes every existing root handler. This module is
# imported by graflag_runner/__init__.py, so `from graflag_runner import
# ResultWriter` silently destroyed the logging a method had already set up --
# a FileHandler, a DEBUG level. Configure only when nothing else has.
if not _logging.getLogger().handlers:
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
