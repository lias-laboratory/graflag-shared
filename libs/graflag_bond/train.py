#!/usr/bin/env python3
"""
Generic PyGOD Bond Training Script

This script trains any PyGOD detector based on METHOD_NAME environment variable.
"""

import os
import sys
import time
from pathlib import Path

import psutil
import torch

# Import graflag_runner utilities
from graflag_runner import ResultWriter
from graflag_runner import info, warning, error

# Import PyGOD
from pygod.utils import load_data

# Import bond utilities
from graflag_bond.detectors import BondDetector
from graflag_bond.utils import get_all_parameters


def load_graph_data(data_dir):
    """Load graph data from PyGOD datasets."""
    
    supported_data = os.environ.get("SUPPORTED_DATA", "").split(", ")
    dataset_name = data_dir.name
    
    if supported_data and dataset_name not in supported_data:
        warning(f"Dataset '{dataset_name}' may not be officially tested. Supported: {supported_data}")
    
    info(f"Loading dataset: {dataset_name} from {data_dir}")

    # Load data using PyGOD's load_data
    data = load_data(dataset_name, cache_dir=data_dir)
    info(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    
    return data


def train_detector(method_name, data, exp_dir, writer):
    """Train PyGOD detector."""
    
    # Get detector name and class dynamically
    detector_name = BondDetector.from_method_name(method_name)
    detector_class = BondDetector.get_detector_class(detector_name)
    
    # Get parameters from environment with type hints from detector signature
    params = get_all_parameters(detector_class)
    
    info("=" * 60)
    info(f"Training {detector_name.upper()} Model")
    info("=" * 60)
    
    # Log key parameters
    info(f"Detector: {detector_class.__name__}")
    if "hid_dim" in params:
        info(f"Architecture: hid_dim={params['hid_dim']}, num_layers={params.get('num_layers', 'N/A')}")
    if "epoch" in params:
        info(f"Training: epochs={params['epoch']}, lr={params.get('lr', 'N/A')}")
    if "contamination" in params:
        info(f"Contamination: {params['contamination']}")
    
    # Initialize model
    info(f"Initializing {detector_name.upper()} detector...")
    model = detector_class(**params)
    
    # Train model
    info("Starting training...")
    start_time = time.time()
    model.fit(data)
    training_time = time.time() - start_time
    
    # Log training metrics
    writer.spot("training", 
                epochs=params.get('epoch', 'N/A'),
                training_time_sec=training_time)

    info(f"Training completed in {training_time:.2f}s")

    return model


def save_results(model, data, exp_dir, writer, method_name, dataset_name,
                 exec_time_ms, peak_memory_mb, peak_gpu_mb=None):
    """Save results with metadata and resource metrics."""
    info("=" * 60)
    info("Generating Results")
    info("=" * 60)

    # Get anomaly scores
    scores = model.decision_score_

    # Get ground truth labels from data (binarize: 0=normal, any non-zero=anomaly)
    gt_raw = data.y.cpu() if hasattr(data.y, 'cpu') else data.y
    ground_truth = [1 if label != 0 else 0 for label in gt_raw]

    # Save results using ResultWriter
    writer.save_scores(
        result_type="NODE_ANOMALY_SCORES",
        scores=scores.tolist(),
        ground_truth=ground_truth,
        node_ids=list(range(len(scores)))
    )

    # Get detector info
    detector_name = BondDetector.from_method_name(method_name)
    detector_class = BondDetector.get_detector_class(detector_name)
    params = get_all_parameters(detector_class)

    # Convert params to JSON-safe strings (some values are Python types/functions)
    safe_params = {}
    for k, v in params.items():
        if callable(v) or isinstance(v, type):
            safe_params[k] = f"{v.__module__}.{v.__qualname__}" if hasattr(v, '__module__') else str(v)
        else:
            safe_params[k] = v

    # Add metadata
    writer.add_metadata(
        exp_name=os.path.basename(os.environ.get("EXP", "experiment")),
        method_name=method_name,
        dataset=dataset_name,
        method_parameters=safe_params,
        threshold=None,
        summary={
            "description": f"PyGOD {detector_name.upper()} detector",
            "task": "node_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_nodes": data.num_nodes,
                "num_edges": data.num_edges,
                "num_features": data.num_features,
                "num_anomalies": sum(ground_truth),
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_ms,
        peak_memory_mb=peak_memory_mb,
        peak_gpu_mb=peak_gpu_mb,
    )

    # Finalize results
    writer.finalize()

    info(f"Results saved to {exp_dir}")


def main():
    # Get environment variables
    method_name = os.environ.get("METHOD_NAME")
    if not method_name:
        error("METHOD_NAME environment variable not set!")
        sys.exit(1)

    data_dir = Path(os.environ.get("DATA"))
    exp_dir = Path(os.environ.get("EXP"))

    info("=" * 60)
    info(f"PyGOD Bond: {method_name.upper()}")
    info("=" * 60)
    info(f"Dataset: {data_dir}")
    info(f"Output: {exp_dir}")
    info("")

    # Create experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Start resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize ResultWriter
    writer = ResultWriter()

    try:
        # Load data
        data = load_graph_data(data_dir)

        # Track memory
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Train model
        model = train_detector(method_name, data, exp_dir, writer)

        # Track memory after training
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Calculate resource metrics
        end_time = time.time()
        exec_time_ms = (end_time - start_time) * 1000

        # Track GPU memory if available
        peak_gpu_mb = None
        if torch.cuda.is_available():
            gpu_bytes = torch.cuda.max_memory_allocated()
            if gpu_bytes > 0:
                peak_gpu_mb = gpu_bytes / (1024 * 1024)

        # Save results with metadata and resource metrics
        save_results(model, data, exp_dir, writer, method_name, data_dir.name,
                     exec_time_ms, peak_memory_mb, peak_gpu_mb)

        info("")
        info(f"[INFO] Resource Usage:")
        info(f"   [INFO] Execution time: {exec_time_ms/1000:.2f}s")
        info(f"   [INFO] Peak memory: {peak_memory_mb:.2f}MB")
        if peak_gpu_mb is not None:
            info(f"   [INFO] Peak GPU memory: {peak_gpu_mb:.2f}MB")

        info("")
        info("=" * 60)
        info(f"{method_name.upper()} execution completed successfully!")
        info("=" * 60)

    except Exception as e:
        error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
