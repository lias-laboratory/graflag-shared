"""
GraFlag Runner - Framework for executing graph anomaly detection methods.

This package provides:
- MethodRunner: Main execution wrapper with resource monitoring
- ResourceMonitor: Real-time CPU, memory, and GPU tracking
- ResultWriter: Simple API for methods to save standardized results
- StreamableArray: Wrapper for memory-efficient streaming of large arrays
- subprocess_utils: Utilities for running subprocesses with real-time output
- logging: Simple logging functions (debug, info, warning, error, critical, exception)
- method: Helpers for integration scripts (params, apply_params, device,
  paths, upstream, load_dataset, seed_all) -- see graflag_runner/method.py
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
from .method import (
    ExperimentPaths,
    apply_params,
    injected_params,
    params,
    device,
    paths,
    upstream,
    load_attributed_graph,
    load_dataset,
    load_snapshots,
    snapshot_files,
    split_test_edges,
    seed_all,
    write_mat,
    read_mat,
)

__version__ = "1.1.1"
__all__ = [
    "MethodRunner",
    "ResultWriter",
    "StreamableArray",
    "stream_write_json",
    "run_with_realtime_output",
    "run_command_list",
    "save_output_to_file",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "ExperimentPaths",
    "apply_params",
    "injected_params",
    "params",
    "device",
    "paths",
    "upstream",
    "load_attributed_graph",
    "load_dataset",
    "load_snapshots",
    "snapshot_files",
    "split_test_edges",
    "seed_all",
    "write_mat",
    "read_mat",
]
