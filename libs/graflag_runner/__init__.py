"""
GraFlag Runner - Framework for executing graph anomaly detection methods.

This package provides:
- MethodRunner: Main execution wrapper with resource monitoring
- ResourceMonitor: Real-time CPU, memory, and GPU tracking
- ResultWriter: Simple API for methods to save standardized results
- StreamableArray: Wrapper for memory-efficient streaming of large arrays
- subprocess_utils: Utilities for running subprocesses with real-time output
- logging: Simple logging functions (debug, info, warning, error, critical, exception)
"""

from .runner import MethodRunner
from .results import ResultWriter
from .streaming import StreamableArray, stream_write_json
from .subprocess_utils import (
    run_with_realtime_output,
    run_command_list,
    save_output_to_file
)
from .logging_utils import debug, info, warning, error, critical, exception

__version__ = "1.0.0"
__all__ = [
    "MethodRunner",
    "ResultWriter",
    "run_with_realtime_output",
    "run_command_list",
    "save_output_to_file",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception"
]
