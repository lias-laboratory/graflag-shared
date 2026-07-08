#!/usr/bin/env python3
"""
Convert edge-list datasets to StrGNN format.

StrGNN requires:
- acc_*.npy: Accumulated adjacency matrices (T, N, N) - cumulative graph at each snapshot
- sta_*.npy: Static adjacency (N, N) - final aggregated graph
- split.npz: Train/test split with pos_train, neg_train, pos_test, neg_test

Each contains edges as (source, target) pairs.
"""

import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import scipy.sparse as sp


def load_bitcoin_csv(filepath):
    """Load Bitcoin dataset: source,target,rating,timestamp"""
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                src, tgt, rating = int(parts[0]), int(parts[1]), int(parts[2])
                ts = int(float(parts[3]))  # Handle float timestamps
                # Only use positive ratings (trust edges)
                if rating > 0:
                    edges.append((src, tgt, ts))
    return edges


def load_uci_format(filepath):
    """Load UCI format: source target weight timestamp"""
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('%') or not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                src, tgt, weight, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                edges.append((src, tgt, ts))
    return edges


def create_strgnn_dataset(edges, output_dir, n_snapshots=10, anomaly_ratio=0.05, test_ratio=0.2, window=5):
    """
    Convert edges to StrGNN format.

    Args:
        edges: List of (src, tgt, timestamp) tuples
        output_dir: Output directory path
        n_snapshots: Number of temporal snapshots
        anomaly_ratio: Ratio of negative (anomalous) edges to inject
        test_ratio: Ratio of edges for test set
        window: Window size for StrGNN (train starts from snapshot[window])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp
    edges = sorted(edges, key=lambda x: x[2])

    # Remap node IDs to 0-indexed
    nodes = set()
    for src, tgt, _ in edges:
        nodes.add(src)
        nodes.add(tgt)

    node_map = {n: i for i, n in enumerate(sorted(nodes))}
    n_nodes = len(nodes)

    print(f"Dataset stats: {len(edges)} edges, {n_nodes} nodes")

    # Remap edges with snapshot assignment
    timestamps = [e[2] for e in edges]
    min_ts, max_ts = min(timestamps), max(timestamps)
    time_range = max_ts - min_ts
    snapshot_size = time_range / n_snapshots

    # edges_with_snapshot: (src, tgt, snapshot_idx)
    edges_with_snapshot = []
    for src, tgt, ts in edges:
        snapshot_idx = min(int((ts - min_ts) / snapshot_size), n_snapshots - 1)
        edges_with_snapshot.append((node_map[src], node_map[tgt], snapshot_idx))

    # Create accumulated adjacency matrices
    acc_adj = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=np.float32)

    for src, tgt, snapshot_idx in edges_with_snapshot:
        # Accumulate: add to current and all future snapshots
        for t in range(snapshot_idx, n_snapshots):
            acc_adj[t, src, tgt] = 1
            acc_adj[t, tgt, src] = 1  # Undirected

    # Static adjacency (final accumulated state)
    sta_adj = acc_adj[-1].copy()

    # Save adjacency matrices
    np.save(output_dir / 'acc_graph.npy', acc_adj)
    np.save(output_dir / 'sta_graph.npy', sta_adj)

    print(f"Saved acc_graph.npy: shape {acc_adj.shape}")
    print(f"Saved sta_graph.npy: shape {sta_adj.shape}")

    # Create train/test split with snapshot IDs
    # Group edges by snapshot
    edges_by_snapshot = defaultdict(list)
    for src, tgt, snap in edges_with_snapshot:
        edges_by_snapshot[snap].append((src, tgt))

    # Split snapshots: train from window to n_train, test from n_train to end
    n_train_snapshots = int(n_snapshots * (1 - test_ratio))
    n_train_snapshots = max(window + 1, n_train_snapshots)  # Ensure at least one train snapshot

    train_pos = []
    train_pos_id = []
    test_pos = []
    test_pos_id = []

    for snap_idx in range(window, n_snapshots):
        snap_edges = edges_by_snapshot[snap_idx]
        for src, tgt in snap_edges:
            if snap_idx < n_train_snapshots:
                train_pos.append((src, tgt))
                train_pos_id.append(snap_idx)
            else:
                test_pos.append((src, tgt))
                test_pos_id.append(snap_idx)

    # Remove duplicates while preserving snapshot assignment
    seen_train = set()
    unique_train_pos = []
    unique_train_pos_id = []
    for edge, snap_id in zip(train_pos, train_pos_id):
        if edge not in seen_train:
            seen_train.add(edge)
            unique_train_pos.append(edge)
            unique_train_pos_id.append(snap_id)

    seen_test = set()
    unique_test_pos = []
    unique_test_pos_id = []
    for edge, snap_id in zip(test_pos, test_pos_id):
        if edge not in seen_test:
            seen_test.add(edge)
            unique_test_pos.append(edge)
            unique_test_pos_id.append(snap_id)

    train_pos = unique_train_pos
    train_pos_id = unique_train_pos_id
    test_pos = unique_test_pos
    test_pos_id = unique_test_pos_id

    # Generate negative edges (non-existing edges as anomalies)
    n_neg_train = int(len(train_pos) * anomaly_ratio)
    n_neg_test = int(len(test_pos) * anomaly_ratio)

    all_edges = set((src, tgt) for src, tgt, _ in edges_with_snapshot)
    all_edges.update((tgt, src) for src, tgt, _ in edges_with_snapshot)

    train_neg = []
    train_neg_id = []
    test_neg = []
    test_neg_id = []

    max_attempts = (n_neg_train + n_neg_test) * 20
    attempts = 0
    neg_generated = set()

    # Generate negative train edges
    while len(train_neg) < n_neg_train and attempts < max_attempts:
        src = np.random.randint(0, n_nodes)
        tgt = np.random.randint(0, n_nodes)
        snap = np.random.randint(window, n_train_snapshots)
        if src != tgt and (src, tgt) not in all_edges and (src, tgt) not in neg_generated:
            train_neg.append((src, tgt))
            train_neg_id.append(snap)
            neg_generated.add((src, tgt))
        attempts += 1

    # Generate negative test edges
    while len(test_neg) < n_neg_test and attempts < max_attempts:
        src = np.random.randint(0, n_nodes)
        tgt = np.random.randint(0, n_nodes)
        snap = np.random.randint(n_train_snapshots, n_snapshots)
        if src != tgt and (src, tgt) not in all_edges and (src, tgt) not in neg_generated:
            test_neg.append((src, tgt))
            test_neg_id.append(snap)
            neg_generated.add((src, tgt))
        attempts += 1

    # Save split with IDs (StrGNN format)
    np.savez(output_dir / 'split.npz',
             train_pos=np.array(train_pos),
             train_neg=np.array(train_neg),
             test_pos=np.array(test_pos),
             test_neg=np.array(test_neg),
             train_pos_id=np.array(train_pos_id),
             train_neg_id=np.array(train_neg_id),
             test_pos_id=np.array(test_pos_id),
             test_neg_id=np.array(test_neg_id))

    print(f"Train: {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"Test: {len(test_pos)} pos, {len(test_neg)} neg")
    print(f"Saved split.npz")

    total_edges = len(set((src, tgt) for src, tgt, _ in edges_with_snapshot))

    # Create README
    readme = f"""# StrGNN Dataset

Converted from edge-list format to StrGNN format.

## Statistics
- Nodes: {n_nodes}
- Total unique edges: {total_edges}
- Snapshots: {n_snapshots}
- Window size: {window}
- Train snapshots: {window} to {n_train_snapshots-1}
- Test snapshots: {n_train_snapshots} to {n_snapshots-1}
- Anomaly ratio: {anomaly_ratio}

## Files
- `acc_graph.npy`: Accumulated adjacency matrices, shape ({n_snapshots}, {n_nodes}, {n_nodes})
- `sta_graph.npy`: Static adjacency matrix, shape ({n_nodes}, {n_nodes})
- `split.npz`: Train/test split with snapshot IDs
  - train_pos: {len(train_pos)} positive training edges
  - train_neg: {len(train_neg)} negative training edges (anomalies)
  - test_pos: {len(test_pos)} positive test edges
  - test_neg: {len(test_neg)} negative test edges (anomalies)
  - *_id: Snapshot indices for each edge
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme)

    print(f"\nDataset created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert edge-list to StrGNN format')
    parser.add_argument('input', help='Input file path')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--format', choices=['bitcoin', 'uci'], default='bitcoin',
                       help='Input format (default: bitcoin)')
    parser.add_argument('--snapshots', type=int, default=10,
                       help='Number of temporal snapshots (default: 10)')
    parser.add_argument('--anomaly-ratio', type=float, default=0.05,
                       help='Ratio of anomalous edges (default: 0.05)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio of test edges (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--window', type=int, default=5,
                       help='StrGNN window size (default: 5)')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load edges
    if args.format == 'bitcoin':
        edges = load_bitcoin_csv(args.input)
    else:
        edges = load_uci_format(args.input)

    # Convert
    create_strgnn_dataset(
        edges,
        args.output,
        n_snapshots=args.snapshots,
        anomaly_ratio=args.anomaly_ratio,
        test_ratio=args.test_ratio,
        window=args.window
    )


if __name__ == '__main__':
    main()
