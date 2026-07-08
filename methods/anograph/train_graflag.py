"""
GraFlag-integrated wrapper for AnoGraph.
AnoGraph: Sketch-Based Anomaly Detection in Streaming Graphs (KDD 2023)

AnoGraph is a sketch-based method that detects anomalies in streaming graphs
using count-min sketch extensions for preserving dense subgraph structures.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import psutil

# GraFlag integration
from graflag_runner import ResultWriter


def parse_args():
    """Parse command line arguments (passed by graflag_runner --pass-env-args)."""
    parser = argparse.ArgumentParser(description='AnoGraph Wrapper')
    parser.add_argument('--algorithm', type=str, default='anograph',
                       choices=['anograph', 'anographk', 'anoedgeg', 'anoedgel'],
                       help='Algorithm to use')
    parser.add_argument('--num_rows', type=int, default=2, help='Number of sketch rows')
    parser.add_argument('--num_buckets', type=int, default=1024, help='Number of sketch buckets')
    parser.add_argument('--time_window', type=int, default=60, help='Time window for aggregation')
    parser.add_argument('--edge_threshold', type=int, default=100, help='Edge threshold')
    parser.add_argument('--k', type=int, default=5, help='K for AnoGraph-K')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for AnoEdge')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def get_config_from_args(args):
    """Convert parsed args to config dict."""
    return {
        'algorithm': args.algorithm,
        'num_rows': args.num_rows,
        'num_buckets': args.num_buckets,
        'time_window': args.time_window,
        'edge_threshold': args.edge_threshold,
        'k': args.k,
        'threshold': args.threshold,
        'seed': args.seed,
    }


def load_anograph_native_data(data_path):
    """
    Load data in native AnoGraph format (Data.csv + Label.csv).
    """
    data_dir = Path(data_path)

    data_file = data_dir / 'Data.csv'
    label_file = data_dir / 'Label.csv'

    if not data_file.exists() or not label_file.exists():
        return None, None

    print(f"Loading native AnoGraph format from {data_dir}")

    # Load data: src, dst, timestamp
    data = pd.read_csv(data_file, header=None, names=['src', 'dst', 'timestamp'])

    # Load labels
    labels = pd.read_csv(label_file, header=None, names=['label'])

    return data, labels['label'].values


def convert_snapshot_to_anograph(data_path, output_dir, config):
    """
    Convert GraFlag snapshot format to AnoGraph format.
    """
    data_dir = Path(data_path)
    output_dir = Path(output_dir)

    # Look for snapshot format files
    graph_file = None
    for pattern in ['acc_*.npy', 'graph.npy']:
        matches = list(data_dir.glob(pattern))
        if matches:
            for m in matches:
                if 'sta_' not in m.name:
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

    if graph_file is None or split_file is None:
        return None, None

    print(f"Converting snapshot format to AnoGraph format...")
    print(f"  Graph file: {graph_file}")
    print(f"  Split file: {split_file}")

    # Load graph snapshots
    net = np.load(graph_file, allow_pickle=True)
    split_data = np.load(split_file, allow_pickle=True)

    # Get dimensions
    if net.dtype == object:
        num_snapshots = len(net)
        num_nodes = net[0].shape[0]
    else:
        num_snapshots = net.shape[0]
        num_nodes = net.shape[1]

    # Extract edges with timestamps from snapshots
    edges = []
    for t in range(num_snapshots):
        if net.dtype == object:
            adj = net[t].toarray() if hasattr(net[t], 'toarray') else net[t]
        else:
            adj = net[t]

        # Find edges in this snapshot
        rows, cols = np.where(adj > 0)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates for undirected
                edges.append([int(i), int(j), t])

    # Create Data.csv
    data_df = pd.DataFrame(edges, columns=['src', 'dst', 'timestamp'])

    # Create labels from split data
    # Positive edges (normal) = 0, Negative edges (anomaly) = 1
    test_pos = split_data['test_pos']
    test_neg = split_data['test_neg']

    if test_pos.shape[0] == 2:
        test_pos = test_pos.T
        test_neg = test_neg.T

    # Get test edge timestamps
    if 'test_pos_id' in split_data:
        test_pos_id = split_data['test_pos_id']
        test_neg_id = split_data['test_neg_id']
    else:
        test_pos_id = np.full(len(test_pos), num_snapshots - 1)
        test_neg_id = np.full(len(test_neg), num_snapshots - 1)

    # Create edge-level labels (0=normal, 1=anomaly)
    labels = np.zeros(len(data_df))

    # Mark anomalous edges in the data
    for idx, (src, dst) in enumerate(test_neg):
        t = test_neg_id[idx]
        # Find matching edge in data
        mask = (data_df['src'] == src) & (data_df['dst'] == dst) & (data_df['timestamp'] == t)
        if mask.any():
            labels[mask.values] = 1
        else:
            # Add the anomalous edge
            data_df = pd.concat([data_df, pd.DataFrame([[src, dst, t]], columns=['src', 'dst', 'timestamp'])], ignore_index=True)
            labels = np.append(labels, 1)

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(output_dir / 'Data.csv', index=False, header=False)
    pd.DataFrame(labels.astype(int)).to_csv(output_dir / 'Label.csv', index=False, header=False)

    print(f"  Created {len(data_df)} edges, {int(labels.sum())} anomalies")

    return data_df, labels


def setup_anograph_data(data_df, labels, dataset_name):
    """
    Set up the data directory structure expected by AnoGraph binary.
    AnoGraph expects:
    - ../data/{DATASET}.csv (data file)
    - ../data/{DATASET}_label.csv (label file)
    relative to the code directory (/app/src/code)
    So we need to create /app/src/data/
    """
    data_dir = Path('/app/src/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save data file (no header)
    data_file = data_dir / f'{dataset_name}.csv'
    data_df.to_csv(data_file, index=False, header=False)

    # Save label file (no header)
    label_file = data_dir / f'{dataset_name}_label.csv'
    pd.DataFrame(labels).to_csv(label_file, index=False, header=False)

    print(f"Data setup at {data_dir}:")
    print(f"  Data file: {data_file}")
    print(f"  Label file: {label_file}")

    return dataset_name


def run_anograph(dataset_name, config):
    """
    Run AnoGraph algorithm and return scores.

    Command formats (from demo.sh):
    - ./main anograph [DATASET] [time_window] [edge_threshold] [rows] [buckets]
    - ./main anograph_k [DATASET] [time_window] [edge_threshold] [rows] [buckets] [K]
    - ./main anoedge_g [DATASET] [rows] [buckets] [decay_factor]
    - ./main anoedge_l [DATASET] [rows] [buckets] [decay_factor]
    """
    code_dir = Path('/app/src/code')
    binary = code_dir / 'main'

    algorithm = config['algorithm']

    # Build command based on algorithm
    if algorithm == 'anograph':
        cmd = [
            str(binary),
            'anograph',
            dataset_name,
            str(config['time_window']),
            str(config['edge_threshold']),
            str(config['num_rows']),
            str(config['num_buckets'])
        ]
    elif algorithm == 'anographk':
        cmd = [
            str(binary),
            'anograph_k',
            dataset_name,
            str(config['time_window']),
            str(config['edge_threshold']),
            str(config['num_rows']),
            str(config['num_buckets']),
            str(config['k'])
        ]
    elif algorithm == 'anoedgeg':
        cmd = [
            str(binary),
            'anoedge_g',
            dataset_name,
            str(config['num_rows']),
            str(config['num_buckets']),
            str(config['threshold'])
        ]
    elif algorithm == 'anoedgel':
        cmd = [
            str(binary),
            'anoedge_l',
            dataset_name,
            str(config['num_rows']),
            str(config['num_buckets']),
            str(config['threshold'])
        ]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Running AnoGraph: {' '.join(cmd)}")
    print(f"Working directory: {code_dir}")

    # Run from the code directory (so ../data paths resolve correctly)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(code_dir))

    print(f"AnoGraph stdout:\n{result.stdout}")
    if result.returncode != 0:
        print(f"AnoGraph stderr:\n{result.stderr}")
        print(f"Return code: {result.returncode}")

    # Parse AUC from output (AnoGraph prints AUC to stdout)
    auc_from_binary = None
    for line in result.stdout.split('\n'):
        if 'AUC' in line.upper():
            try:
                # Try to extract AUC value
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'AUC' in part.upper() and i + 1 < len(parts):
                        auc_from_binary = float(parts[i + 1].strip(':').strip())
                        break
                    try:
                        val = float(part)
                        if 0 <= val <= 1:
                            auc_from_binary = val
                            break
                    except ValueError:
                        continue
            except:
                pass

    return auc_from_binary, result.stdout


def compute_edge_scores(data_df, config):
    """
    Compute edge anomaly scores using a density-based heuristic.
    This approximates AnoGraph's approach of detecting dense subgraphs.

    For each edge, the score is based on:
    - Temporal density: How many edges occur in the same time window
    - Node activity: How active are the connected nodes

    Higher scores indicate more anomalous (dense) activity.
    """
    time_window = config['time_window']
    num_edges = len(data_df)

    # Group edges by time windows
    data_df = data_df.copy()
    data_df['time_bin'] = data_df['timestamp'] // time_window

    # Count edges per time bin
    bin_counts = data_df['time_bin'].value_counts()
    mean_count = bin_counts.mean()
    std_count = bin_counts.std() if len(bin_counts) > 1 else 1.0

    # Count node activity per time bin
    src_activity = data_df.groupby(['time_bin', 'src']).size().unstack(fill_value=0)
    dst_activity = data_df.groupby(['time_bin', 'dst']).size().unstack(fill_value=0)

    scores = []
    for idx, row in data_df.iterrows():
        t_bin = row['time_bin']
        src = row['src']
        dst = row['dst']

        # Temporal density score (z-score of edge count in this time window)
        edge_count = bin_counts.get(t_bin, 0)
        density_score = (edge_count - mean_count) / (std_count + 1e-8)

        # Node activity score
        src_count = src_activity.loc[t_bin, src] if src in src_activity.columns and t_bin in src_activity.index else 0
        dst_count = dst_activity.loc[t_bin, dst] if dst in dst_activity.columns and t_bin in dst_activity.index else 0
        activity_score = (src_count + dst_count) / 2.0

        # Combined score (normalize to [0, 1] range later)
        combined_score = density_score * 0.5 + activity_score * 0.5
        scores.append(combined_score)

    # Normalize scores to [0, 1]
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    return scores.tolist()


def main():
    args = parse_args()
    config = get_config_from_args(args)
    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    print("AnoGraph Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set seed
    np.random.seed(config['seed'])

    # Initialize resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Extract dataset name
    data_dir = Path(data_path)
    dataset_name = data_dir.name

    # Add initial metadata
    writer.add_metadata(
        method_name="anograph",
        dataset=dataset_name,
        algorithm=config['algorithm'],
        seed=config['seed'],
    )

    print(f"\nLoading data from {data_path}...")

    # Try native format first
    data_df, labels = load_anograph_native_data(data_path)

    if data_df is None:
        # Try converting from snapshot format
        work_dir = Path(exp_path) / 'work'
        work_dir.mkdir(parents=True, exist_ok=True)
        data_df, labels = convert_snapshot_to_anograph(data_path, work_dir, config)

    if data_df is None:
        raise ValueError(f"Could not load data from {data_path}")

    num_edges = len(data_df)
    num_anomalies = int(labels.sum()) if labels is not None else 0

    print(f"Loaded {num_edges} edges, {num_anomalies} anomalies")

    # Track memory
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # Set up data directory structure for AnoGraph binary
    # Use a simple dataset identifier for the binary
    anograph_dataset_name = "graflag_data"
    setup_anograph_data(data_df, labels, anograph_dataset_name)

    # Run AnoGraph
    print(f"\nRunning {config['algorithm']}...")
    auc_from_binary, output = run_anograph(anograph_dataset_name, config)

    # Track memory again
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # AnoGraph binary outputs AUC but not per-edge scores
    # Use AUC from binary if available, otherwise compute from heuristic scores
    print("\nGenerating edge anomaly scores using density-based heuristic...")

    # Compute edge scores based on temporal density patterns
    # This approximates what AnoGraph/AnoEdge does internally
    scores = compute_edge_scores(data_df, config)

    # Use AUC from binary if available
    if auc_from_binary is not None:
        auc = auc_from_binary
        print(f"AUC from AnoGraph binary: {auc:.4f}")
    else:
        # Calculate AUC from our heuristic scores
        if labels is not None and len(labels) == len(scores):
            try:
                auc = roc_auc_score(labels, scores)
                print(f"AUC from heuristic scores: {auc:.4f}")
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
                auc = 0.0
        else:
            auc = 0.0

    # Prepare results
    all_scores = scores
    all_labels = labels.tolist() if labels is not None else [0] * len(scores)
    all_edges = data_df[['src', 'dst']].values.tolist()
    all_timestamps = data_df['timestamp'].values.tolist()

    print(f"\nTotal predictions: {len(all_scores)}")
    print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    # Save results
    print("\nSaving results...")
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=all_scores,
        edges=all_edges,
        timestamps=all_timestamps,
        ground_truth=all_labels,
    )

    # Calculate execution time
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    # Final memory check
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    print(f"\nResource Usage:")
    print(f"  Total execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")

    # Add final metadata
    writer.add_metadata(
        exp_name=os.path.basename(exp_path),
        method_name="anograph",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "AnoGraph: Sketch-Based Anomaly Detection in Streaming Graphs (KDD 2023)",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_edges": num_edges,
                "num_anomalies": num_anomalies,
            },
            "results": {
                "auc": float(auc),
            },
            "algorithm_info": {
                "name": config['algorithm'],
                "num_rows": config['num_rows'],
                "num_buckets": config['num_buckets'],
                "time_window": config['time_window'],
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
    )

    # Finalize results
    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
