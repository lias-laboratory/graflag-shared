"""
==============================================================================
GraFlag Method Integration Template
==============================================================================

This template demonstrates how to integrate a Graph Anomaly Detection (GAD)
method with GraFlag. Copy this file and modify it for your method.

Key Components:
    1. Argument parsing (receives parameters from .env via --pass-env-args)
    2. Data loading (multiple format support)
    3. Model training (your method implementation)
    4. Result saving (using ResultWriter for standardized output)

Usage:
    ./graflag_cli.py benchmark -m your_method -d dataset_name --build

Environment Variables (set automatically by GraFlag):
    DATA: Path to dataset directory (e.g., /shared/datasets/email_snapshot)
    EXP:  Path to experiment output directory

==============================================================================
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import psutil

# ==============================================================================
# REQUIRED: Import GraFlag ResultWriter
# ==============================================================================
# ResultWriter handles standardized result output for the evaluation framework
from graflag_runner import ResultWriter


# ==============================================================================
# Step 1: Argument Parsing
# ==============================================================================
# Parameters from .env (prefixed with _) are converted to CLI arguments.
# Example: _LEARNING_RATE=0.001 becomes --learning_rate 0.001

def parse_args():
    """
    Parse command line arguments.

    These arguments come from two sources:
    1. Default values defined here
    2. .env file values (via --pass-env-args conversion)
    3. User overrides via --params flag

    The --pass-env-args mechanism converts:
        _LEARNING_RATE=0.001 -> --learning_rate 0.001
    """
    parser = argparse.ArgumentParser(
        description='Template Method for GraFlag Integration'
    )

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimization')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='Embedding dimension')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


# ==============================================================================
# Step 2: Data Loading Functions
# ==============================================================================
# GraFlag supports multiple data formats. Implement loaders for formats
# your method supports.

def load_data(data_path):
    """
    Load dataset from the given path.

    GraFlag datasets can be in different formats:

    1. Edge Stream Format (Data.csv + Label.csv):
       - Data.csv: src,dst,timestamp (no header)
       - Label.csv: binary labels (0=normal, 1=anomaly)

    2. Snapshot Format (acc_*.npy + split.npz):
       - acc_*.npy: Accumulated adjacency matrices [T, N, N]
       - split.npz: Train/test edge splits with labels

    3. Raw Edge List (edges.txt):
       - Simple edge list: src dst [timestamp]

    Args:
        data_path: Path to dataset directory

    Returns:
        data_df: DataFrame with columns [src, dst, timestamp]
        labels: numpy array of binary labels (0=normal, 1=anomaly)
    """
    data_dir = Path(data_path)

    # ----- Format 1: Data.csv + Label.csv (e.g., AnoGraph datasets) -----
    data_file = data_dir / 'Data.csv'
    label_file = data_dir / 'Label.csv'

    if data_file.exists() and label_file.exists():
        print(f"Loading Data.csv/Label.csv format from {data_dir}")
        data_df = pd.read_csv(data_file, header=None,
                              names=['src', 'dst', 'timestamp'])
        labels = pd.read_csv(label_file, header=None,
                             names=['label'])['label'].values
        return data_df, labels

    # ----- Format 2: Snapshot Format (e.g., StrGNN datasets) -----
    graph_file = None
    for pattern in ['acc_*.npy', 'graph.npy']:
        matches = list(data_dir.glob(pattern))
        if matches:
            for m in matches:
                if 'sta_' not in m.name:  # Skip static graphs
                    graph_file = m
                    break
            if graph_file:
                break

    split_file = None
    for pattern in ['split.npz', '*.npz']:
        matches = list(data_dir.glob(pattern))
        if matches:
            split_file = matches[0]
            break

    if graph_file and split_file:
        print(f"Loading snapshot format from {data_dir}")
        return load_snapshot_format(graph_file, split_file)

    # ----- Format 3: Raw Edge List (known names) -----
    for edge_file in ['edges.txt', 'edges.csv', 'edge_list.txt']:
        if (data_dir / edge_file).exists():
            print(f"Loading edge list from {data_dir / edge_file}")
            return load_edge_list(data_dir / edge_file)

    # ----- Format 4: Raw Edge List (dataset-named file, e.g., digg/digg) -----
    dataset_name = data_dir.name
    dataset_file = data_dir / dataset_name
    if dataset_file.exists() and dataset_file.is_file():
        print(f"Loading edge list from {dataset_file}")
        return load_edge_list(dataset_file)

    # ----- Fallback: Try any text file in the directory -----
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and f.suffix in ('', '.txt', '.csv', '.tsv') and f.name != 'README.md':
            print(f"Loading edge list from {f}")
            return load_edge_list(f)

    raise ValueError(f"Could not find valid data format in {data_path}")


def load_snapshot_format(graph_file, split_file):
    """
    Load data from snapshot format (numpy arrays).

    This format is used by methods like StrGNN, AddGraph that work
    with discrete graph snapshots.
    """
    net = np.load(graph_file, allow_pickle=True)
    split_data = np.load(split_file, allow_pickle=True)

    # Determine number of snapshots
    if net.dtype == object:  # Array of sparse matrices
        num_snapshots = len(net)
    else:  # Dense 3D array [T, N, N]
        num_snapshots = net.shape[0]

    # Extract edges with timestamps
    edges = []
    for t in range(num_snapshots):
        if net.dtype == object:
            adj = net[t].toarray() if hasattr(net[t], 'toarray') else net[t]
        else:
            adj = net[t]

        rows, cols = np.where(adj > 0)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates for undirected graphs
                edges.append([int(i), int(j), t])

    data_df = pd.DataFrame(edges, columns=['src', 'dst', 'timestamp'])

    # Extract labels from split data
    # test_neg contains negative (anomalous) edges
    test_neg = split_data['test_neg']
    if test_neg.shape[0] == 2:
        test_neg = test_neg.T

    labels = np.zeros(len(data_df))

    # Get timestamps for test edges if available
    if 'test_neg_id' in split_data:
        test_neg_id = split_data['test_neg_id']
    else:
        test_neg_id = np.full(len(test_neg), num_snapshots - 1)

    # Mark anomalous edges
    for idx, (src, dst) in enumerate(test_neg):
        t = test_neg_id[idx]
        mask = ((data_df['src'] == src) & (data_df['dst'] == dst) &
                (data_df['timestamp'] == t))
        if mask.any():
            labels[mask.values] = 1
        else:
            # Add anomalous edge if not in graph
            data_df = pd.concat([
                data_df,
                pd.DataFrame([[src, dst, t]], columns=['src', 'dst', 'timestamp'])
            ], ignore_index=True)
            labels = np.append(labels, 1)

    return data_df, labels


def load_edge_list(file_path):
    """Load simple edge list format."""
    edges = []
    with open(file_path) as f:
        for line in f:
            if line.startswith('%') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = int(parts[0]), int(parts[1])
                timestamp = int(parts[2]) if len(parts) > 2 else len(edges)
                edges.append([src, dst, timestamp])

    data_df = pd.DataFrame(edges, columns=['src', 'dst', 'timestamp'])
    labels = np.zeros(len(data_df))  # No labels in this format

    return data_df, labels


# ==============================================================================
# Step 3: Your Method Implementation
# ==============================================================================
# Replace this with your actual anomaly detection method.

class YourModel:
    """
    Template model class.

    Replace this with your actual model implementation.
    Your model should:
    1. Learn from the graph structure
    2. Produce anomaly scores for edges (higher = more anomalous)
    """

    def __init__(self, config):
        """Initialize model with configuration."""
        self.config = config
        self.is_trained = False

    def train(self, data_df, writer=None):
        """
        Train the model on the data.

        Args:
            data_df: DataFrame with [src, dst, timestamp] columns
            writer: ResultWriter for logging training metrics (optional)
        """
        print(f"Training model for {self.config['epochs']} epochs...")

        for epoch in range(self.config['epochs']):
            # ----- Your training logic here -----
            loss = 1.0 / (epoch + 1)  # Placeholder

            # Log training metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.config['epochs']}, Loss: {loss:.6f}")

                # Use ResultWriter.spot() to log training metrics
                # These are saved to training.csv for later analysis
                if writer:
                    writer.spot("training", epoch=epoch + 1, loss=loss)

        self.is_trained = True
        print("Training complete!")

    def predict(self, data_df):
        """
        Generate anomaly scores for each edge.

        Args:
            data_df: DataFrame with [src, dst, timestamp] columns

        Returns:
            scores: numpy array of anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        print("Generating anomaly scores...")

        # ----- Your prediction logic here -----
        # This placeholder generates random scores
        # Replace with your actual scoring mechanism
        num_edges = len(data_df)
        scores = np.random.rand(num_edges)

        return scores


# ==============================================================================
# Step 4: Main Function
# ==============================================================================

def main():
    """
    Main execution function.

    This function orchestrates the entire pipeline:
    1. Parse arguments
    2. Load data
    3. Train model
    4. Generate predictions
    5. Save results
    """

    # -------------------------------------------------------------------------
    # 4.1: Parse arguments and setup
    # -------------------------------------------------------------------------
    args = parse_args()
    config = vars(args)  # Convert to dictionary

    # Get paths from environment (set by GraFlag)
    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    # Print configuration
    print("=" * 60)
    print("Template Method Configuration")
    print("=" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    # If using PyTorch: torch.manual_seed(args.seed)
    # If using TensorFlow: tf.random.set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 4.2: Initialize tracking
    # -------------------------------------------------------------------------
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # -------------------------------------------------------------------------
    # 4.3: Initialize ResultWriter (REQUIRED)
    # -------------------------------------------------------------------------
    # ResultWriter handles all result output for GraFlag
    writer = ResultWriter()

    # Extract dataset name for metadata
    dataset_name = Path(data_path).name

    # Add initial metadata
    writer.add_metadata(
        method_name="example",  # Must match METHOD_NAME in .env
        dataset=dataset_name,
        seed=args.seed,
    )

    # -------------------------------------------------------------------------
    # 4.4: Load data
    # -------------------------------------------------------------------------
    print(f"Loading data from {data_path}...")
    data_df, labels = load_data(data_path)

    num_edges = len(data_df)
    num_anomalies = int(labels.sum()) if labels is not None else 0

    # Get unique nodes
    all_nodes = set(data_df['src'].values) | set(data_df['dst'].values)
    num_nodes = len(all_nodes)

    print(f"Loaded {num_edges} edges, {num_nodes} nodes, {num_anomalies} anomalies")

    # Track memory
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # -------------------------------------------------------------------------
    # 4.5: Train model
    # -------------------------------------------------------------------------
    print("\nTraining model...")
    model = YourModel(config)
    model.train(data_df, writer=writer)

    # Track memory after training
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # -------------------------------------------------------------------------
    # 4.6: Generate predictions
    # -------------------------------------------------------------------------
    print("\nGenerating predictions...")
    scores = model.predict(data_df)

    # Normalize scores to [0, 1] range (recommended)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # -------------------------------------------------------------------------
    # 4.7: Calculate metrics
    # -------------------------------------------------------------------------
    if labels is not None and len(labels) == len(scores) and labels.sum() > 0:
        try:
            auc = roc_auc_score(labels, scores)
            print(f"\nTest AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            auc = 0.0
    else:
        auc = 0.0
        print("\nNo labels available for AUC calculation")

    # -------------------------------------------------------------------------
    # 4.8: Save results (REQUIRED)
    # -------------------------------------------------------------------------
    print("\nSaving results...")

    # Prepare data for output
    all_edges = data_df[['src', 'dst']].values.tolist()
    all_timestamps = data_df['timestamp'].values.tolist()
    all_labels = labels.tolist() if labels is not None else [0] * len(scores)

    # Save scores using ResultWriter
    # result_type must be one of the valid GraFlag result types:
    #   - NODE_ANOMALY_SCORES, EDGE_ANOMALY_SCORES, GRAPH_ANOMALY_SCORES
    #   - TEMPORAL_NODE_ANOMALY_SCORES, TEMPORAL_EDGE_ANOMALY_SCORES, etc.
    #   - NODE_STREAM_ANOMALY_SCORES, EDGE_STREAM_ANOMALY_SCORES, etc.
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",  # Choose appropriate type
        scores=scores.tolist(),
        edges=all_edges,
        timestamps=all_timestamps,
        ground_truth=all_labels,
    )

    # -------------------------------------------------------------------------
    # 4.9: Calculate execution time and finalize
    # -------------------------------------------------------------------------
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    # Final memory check
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    print(f"\nResource Usage:")
    print(f"  Execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")

    # Add final metadata
    writer.add_metadata(
        exp_name=os.path.basename(exp_path),
        method_name="example",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "Template method for GraFlag integration",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_edges": num_edges,
                "num_nodes": num_nodes,
                "num_anomalies": num_anomalies,
            },
            "results": {
                "auc": float(auc),
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
    )

    # Finalize and write results.json
    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"Test AUC: {auc:.4f}")


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    main()
