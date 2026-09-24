#!/usr/bin/env python3
"""
Generic PyGOD Bond Training Script

This script trains any PyGOD detector based on METHOD_NAME environment variable.
"""

import fnmatch
import os
import sys
import time
from pathlib import Path

# psutil and torch are gone with the sampling block they fed. The
# runner measures exec time, peak memory and peak GPU from outside the
# method and its numbers win (runner._merge_runtime_metadata), so the four
# point samples taken here were overwritten on every run -- and understated,
# because they saw only this process, not the tree. `time` stays: it measures
# the fit itself, which goes to training.csv as a spot metric, not to the
# metadata fields the runner owns.

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
    
    dataset_name = data_dir.name

    # The .env key is SUPPORTED_DATASETS and its values are fnmatch patterns
    # ("bond_*"), the same dialect tests/test_methods.py validates. Reading
    # SUPPORTED_DATA and comparing for equality made this warn on every run:
    # "".split(", ") is [""], which is truthy, and "bond_gen_100" is never
    # literally equal to "bond_*".
    patterns = [p.strip() for p in
                os.environ.get("SUPPORTED_DATASETS", "").split(",") if p.strip()]
    if patterns and not any(fnmatch.fnmatchcase(dataset_name, p) for p in patterns):
        warning(f"Dataset '{dataset_name}' may not be officially tested. "
                f"Supported: {', '.join(patterns)}")

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


def save_results(model, data, exp_dir, writer, method_name, dataset_name):
    """Save results with metadata.

    Resource metrics are not written here. graflag_runner records exec time,
    peak memory and peak GPU for the whole process tree and overwrites
    whatever the method put in those fields.
    """
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
                # PyGOD's detectors fit on the graph they score: every node,
                # the BOND protocol.
                "scored_split": "all_nodes",
                "scored_samples": len(scores),
            },
        },
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

    # Initialize ResultWriter
    writer = ResultWriter()

    try:
        # Load data
        data = load_graph_data(data_dir)

        # Train model
        model = train_detector(method_name, data, exp_dir, writer)

        # Save results with metadata
        save_results(model, data, exp_dir, writer, method_name, data_dir.name)

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
