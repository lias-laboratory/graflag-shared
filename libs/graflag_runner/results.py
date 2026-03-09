"""Result management and standardization."""

import json
import csv
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from collections import OrderedDict
import logging

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
    
    def __init__(self):
        """
        Initialize result writer.
        
        Args:
            output_dir: Directory to save results.json
        """
        self.output_dir = Path(os.environ.get("EXP"))
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
        
        # Check if we need streaming
        has_streamable = isinstance(self.results.get("scores"), StreamableArray)
        
        if has_streamable:
            logger.info("[INFO] Writing results with streaming (large data)...")
            stream_write_json(self.results, output_file)
        else:
            # Regular JSON dump for small data
            logger.info("[INFO] Writing results (standard)...")
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        logger.info(f"[OK] Results written to: {output_file}")
        return output_file
    
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
            # First call - establish schema
            self._spot_schemas[metric_key] = current_schema
            
            # Create CSV file with header
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['timestamp'] + list(current_schema.keys())
                writer.writerow(header)
            
            logger.debug(f"[INFO] Created spot metric file: {csv_file}")
            logger.debug(f"   Schema: {list(current_schema.keys())}")
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
