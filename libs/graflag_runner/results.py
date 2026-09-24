"""Result management and standardization."""

import json
import csv
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from collections import OrderedDict
import logging

from .serialization import json_default, sanitize
from .streaming import StreamableArray, stream_write_json

logger = logging.getLogger(__name__)


class ResultWriter:
    """
    Simple API for methods to save standardized results.
    
    Usage in method code:
        from graflag_runner import ResultWriter
        
        writer = ResultWriter()
        writer.save_scores(
            result_type="TEMPORAL_NODE_SCORES",
            scores=[[0.1, 0.2], [0.3, 0.4]],
            timestamps=[0, 1],
            node_ids=[0, 1]
        )
        writer.add_metadata(method_name="TADDY", dataset="uci")
        writer.finalize()
    """
    
    VALID_RESULT_TYPES = {
        "NODE_ANOMALY_SCORES",
        "EDGE_ANOMALY_SCORES",
        "GRAPH_ANOMALY_SCORES",
        "TEMPORAL_NODE_ANOMALY_SCORES",
        "TEMPORAL_EDGE_ANOMALY_SCORES",
        "TEMPORAL_GRAPH_ANOMALY_SCORES",
        "NODE_STREAM_ANOMALY_SCORES",
        "EDGE_STREAM_ANOMALY_SCORES",
        "GRAPH_STREAM_ANOMALY_SCORES",
    }
    
    def __init__(self, output_dir=None):
        """
        Initialize result writer.

        Args:
            output_dir: Directory to save results.json. Defaults to $EXP.

        The signature used to take no arguments while the docstring
        documented output_dir, so code following the docstring raised
        TypeError; and an unset $EXP produced Path(None) -> TypeError:
        argument should be a str or an os.PathLike, not NoneType.
        """
        target = output_dir if output_dir is not None else os.environ.get("EXP")
        if not target:
            raise ValueError(
                "ResultWriter needs an output directory: set the EXP "
                "environment variable (GraFlag does this for you) or pass "
                "output_dir explicitly."
            )
        self.output_dir = Path(target)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "result_type": None,
            "scores": None,
            "metadata": {}
        }
        
        # Schema tracking for spot() method
        self._spot_schemas: Dict[str, OrderedDict] = {}
    
    def save_scores(
        self,
        result_type: str,
        scores: Union[List, StreamableArray, Iterator],
        **kwargs
    ):
        """
        Save anomaly scores with specified result type.
        
        Supports both regular lists and streaming for large datasets:
        - Regular list/array: scores = [[...], [...], ...]
        - Generator: scores = StreamableArray(generate_rows())
        - Raw iterator: Will be wrapped in StreamableArray automatically
        
        Args:
            result_type: One of VALID_RESULT_TYPES
            scores: Anomaly scores (list, StreamableArray, or generator)
                   Can be a generator for memory-efficient handling of large arrays
            **kwargs: Additional fields (timestamps, node_ids, edges, etc.)
        """
        if result_type not in self.VALID_RESULT_TYPES:
            raise ValueError(
                f"Invalid result_type: {result_type}. "
                f"Must be one of {self.VALID_RESULT_TYPES}"
            )
        
        self.results["result_type"] = result_type
        
        # Wrap raw generators/iterators in StreamableArray
        if hasattr(scores, '__iter__') and hasattr(scores, '__next__'):
            if not isinstance(scores, StreamableArray):
                scores = StreamableArray(scores)
                logger.info("[INFO] Wrapped generator in StreamableArray for streaming")
        
        self.results["scores"] = scores
        
        # Add optional fields
        for key, value in kwargs.items():
            self.results[key] = value
        
        if isinstance(scores, StreamableArray):
            logger.info(f"[OK] Streamable scores registered: {result_type}")
        else:
            logger.info(f"[OK] Scores saved: {result_type}")
    
    def add_metadata(self, **kwargs):
        """
        Add metadata fields.
        
        Args:
            **kwargs: Metadata key-value pairs (method_name, dataset, etc.)
        """
        self.results["metadata"].update(kwargs)
    
    def add_resource_metrics(
        self,
        exec_time_ms: float,
        peak_memory_mb: float,
        peak_gpu_mb: Optional[float] = None
    ):
        """
        Add resource consumption metrics.
        
        Args:
            exec_time_ms: Execution time in milliseconds
            peak_memory_mb: Peak memory usage in MB
            peak_gpu_mb: Peak GPU memory in MB (optional)
        """
        self.results["metadata"]["exec_time_ms"] = round(exec_time_ms, 2)
        self.results["metadata"]["peak_memory_mb"] = round(peak_memory_mb, 2)
        if peak_gpu_mb is not None:
            self.results["metadata"]["peak_gpu_mb"] = round(peak_gpu_mb, 2)
    
    def finalize(self) -> Path:
        """
        Write results to results.json file.
        
        Uses streaming for large score arrays to avoid memory issues.
        Regular lists are written normally, StreamableArray objects are
        written row-by-row without loading the entire array into memory.
        
        Returns:
            Path to results.json
        """
        output_file = self.output_dir / "results.json"

        # Validation
        if self.results["result_type"] is None:
            raise ValueError("No scores saved. Call save_scores() first.")

        # Detect streamables anywhere, not just under "scores": the writer
        # already handles every streamable key, and a StreamableArray passed as
        # ground_truth or edges used to reach json.dump and raise part-way
        # through an already-truncated file.
        has_streamable = any(
            isinstance(v, StreamableArray) for v in self.results.values()
        )

        # Write to a sibling temp file and rename. open(..., 'w') truncates
        # immediately and json.dump streams, so a value it cannot serialize
        # (a numpy scalar or array, which save_scores stores verbatim) used to
        # leave a truncated results.json on disk. That file still counted as a
        # result, so a failed run was reported as completed and the evaluator
        # then crashed on it. os.replace is atomic within a directory.
        tmp_file = output_file.with_name(output_file.name + ".tmp")
        try:
            if has_streamable:
                logger.info("[INFO] Writing results with streaming (large data)...")
                stream_write_json(self.results, tmp_file)
            else:
                logger.info("[INFO] Writing results (standard)...")
                payload, replaced = sanitize(self.results)
                if replaced:
                    logger.warning(
                        f"[WARN] Replaced {replaced} non-finite value(s) with "
                        f"null; NaN/Infinity are not valid JSON"
                    )
                with open(tmp_file, 'w') as f:
                    json.dump(payload, f, indent=2, default=json_default)
            os.replace(tmp_file, output_file)
        except BaseException:
            tmp_file.unlink(missing_ok=True)
            raise

        logger.info(f"[OK] Results written to: {output_file}")
        return output_file
    
    @staticmethod
    def _read_spot_header(csv_file: Path):
        """Column names of an existing spot CSV (minus 'timestamp'), or None."""
        if not csv_file.is_file():
            return None
        try:
            with open(csv_file, newline='') as f:
                header = next(csv.reader(f), None)
        except OSError:
            return None
        if not header:
            return None
        return [c for c in header if c != 'timestamp']

    def spot(self, metric_key: str, **metrics):
        """
        Track real-time metrics to a CSV file with schema validation.
        
        This method is used for monitoring progress during training/execution:
        - Creates a CSV file named "{metric_key}.csv" in the output directory
        - First column is always "timestamp" (Unix timestamp)
        - Subsequent columns are the metric keys provided in **metrics
        - Schema is locked after first call - subsequent calls must have same keys
        - Automatically appends new rows on each call
        
        Args:
            metric_key: Identifier for the metric group (e.g., "training", "validation", "resources")
                       Used as the CSV filename: "{metric_key}.csv"
            **metrics: Metric key-value pairs to record (e.g., loss=0.5, auc=0.85)
        
        Raises:
            ValueError: If schema changes after first call (different metric keys)
        
        Examples:
            # Track training metrics
            writer.spot("training", epoch=1, loss=0.5, auc=0.85)
            writer.spot("training", epoch=2, loss=0.3, auc=0.90)  # Must have same keys
            
            # Track resource usage
            writer.spot("resources", memory_mb=512.5, gpu_mb=2048.0)
            
            # Track validation metrics separately
            writer.spot("validation", epoch=1, val_loss=0.6, val_auc=0.82)
        """
        if not metrics:
            raise ValueError("At least one metric must be provided to spot()")
        
        # Get CSV file path
        csv_file = self.output_dir / f"{metric_key}.csv"
        
        # Get current schema (ordered dict to preserve column order)
        current_schema = OrderedDict(sorted(metrics.items()))
        
        # Check if this is the first call for this metric_key
        if metric_key not in self._spot_schemas:
            # The schema lock is per-instance, so "first call for this object"
            # is not the same as "file does not exist". Opening with 'w'
            # unconditionally destroyed every row an earlier writer had
            # appended -- a second ResultWriter in the same method, or a
            # re-run into an existing $EXP, silently lost the history.
            existing_header = self._read_spot_header(csv_file)

            if existing_header is None:
                self._spot_schemas[metric_key] = current_schema
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp'] + list(current_schema.keys()))
                logger.debug(f"[INFO] Created spot metric file: {csv_file}")
                logger.debug(f"   Schema: {list(current_schema.keys())}")
            elif set(existing_header) == set(current_schema.keys()):
                # Same columns: adopt the file's order and append to it.
                self._spot_schemas[metric_key] = OrderedDict(
                    (k, current_schema[k]) for k in existing_header
                )
                logger.debug(f"[INFO] Appending to existing spot file: {csv_file}")
            else:
                raise ValueError(
                    f"Schema mismatch for metric '{metric_key}'.\n"
                    f"Existing file columns: {existing_header}\n"
                    f"Provided keys: {list(current_schema.keys())}\n"
                    f"Refusing to overwrite {csv_file}; remove it or use a "
                    f"different metric_key."
                )
        else:
            # Validate schema matches
            expected_schema = self._spot_schemas[metric_key]
            if set(current_schema.keys()) != set(expected_schema.keys()):
                raise ValueError(
                    f"Schema mismatch for metric '{metric_key}'.\n"
                    f"Expected keys: {list(expected_schema.keys())}\n"
                    f"Provided keys: {list(current_schema.keys())}\n"
                    f"All spot() calls for the same metric_key must have identical metric keys."
                )
        
        # Append row to CSV
        timestamp = time.time()
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            # Use the established schema order
            schema = self._spot_schemas[metric_key]
            row = [timestamp] + [metrics[key] for key in schema.keys()]
            writer.writerow(row)
    
    @staticmethod
    def load_results(results_file: str) -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            results_file: Path to results.json
            
        Returns:
            Results dictionary
        """
        with open(results_file, 'r') as f:
            return json.load(f)
