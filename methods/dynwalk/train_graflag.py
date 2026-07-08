"""
GraFlag-integrated implementation of NetWalk.
NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks (KDD 2018)

This implementation follows the NetWalk paper's approach:
1. Random walks on dynamic graph snapshots
2. Deep autoencoder for node embeddings
3. K-means clustering for anomaly detection
4. Distance-based anomaly scoring
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graflag_runner import ResultWriter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DynWalk (NetWalk) Anomaly Detection')
    parser.add_argument('--representation_size', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--walk_length', type=int, default=5, help='Random walk length')
    parser.add_argument('--number_walks', type=int, default=10, help='Number of walks per node')
    parser.add_argument('--init_percent', type=float, default=0.5, help='Initial graph percentage')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of K-means clusters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


class Autoencoder(nn.Module):
    """Deep autoencoder for node embedding."""

    def __init__(self, input_size, hidden_size, embedding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)


def random_walk(graph, start_node, walk_length):
    """Perform a random walk starting from start_node."""
    walk = [start_node]
    current = start_node

    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        walk.append(current)

    return walk


def generate_walks(graph, num_walks, walk_length):
    """Generate random walks for all nodes."""
    walks = []
    nodes = list(graph.nodes())

    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            if graph.degree(node) > 0:
                walk = random_walk(graph, node, walk_length)
                walks.append(walk)

    return walks


def create_node_features(graph, walks, num_nodes, feature_dim=128):
    """Create node features from graph structure and walks."""
    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    # Degree-based features
    for node in range(num_nodes):
        if node in graph:
            degree = graph.degree(node)
            features[node, 0] = np.log1p(degree)

            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_degrees = [graph.degree(n) for n in neighbors[:20]]
                features[node, 1] = np.mean(neighbor_degrees)
                features[node, 2] = np.max(neighbor_degrees)

    # Walk co-occurrence features (sparse)
    cooccur = defaultdict(lambda: defaultdict(float))
    for walk in walks:
        for i, node in enumerate(walk):
            for j in range(max(0, i - 2), min(len(walk), i + 3)):
                if i != j:
                    cooccur[node][walk[j]] += 1

    # Add top co-occurring neighbors as features
    for node in range(num_nodes):
        if node in cooccur:
            sorted_neighbors = sorted(cooccur[node].items(), key=lambda x: -x[1])[:feature_dim - 10]
            for idx, (neighbor, count) in enumerate(sorted_neighbors):
                if 10 + idx < feature_dim:
                    features[node, 10 + idx] = np.log1p(count)

    # Normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms

    return features


def load_edge_data(data_path):
    """Load edge data from various formats."""
    data_dir = Path(data_path)

    # Try edge list format (NetWalk native)
    for edge_file in ['edges.txt', 'edge_list.txt', 'email-Eu-core-sub.txt']:
        file_path = data_dir / edge_file
        if file_path.exists():
            print(f"Loading edge list from {file_path}")
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
            labels = np.zeros(len(data_df))
            return data_df, labels

    # Try Data.csv + Label.csv format (AnoGraph style)
    data_file = data_dir / 'Data.csv'
    label_file = data_dir / 'Label.csv'
    if data_file.exists() and label_file.exists():
        print(f"Loading Data.csv/Label.csv format from {data_dir}")
        data = pd.read_csv(data_file, header=None, names=['src', 'dst', 'timestamp'])
        labels = pd.read_csv(label_file, header=None, names=['label'])
        return data, labels['label'].values

    # Try snapshot format
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

    if graph_file and split_file:
        print(f"Loading snapshot format from {data_dir}")
        return load_snapshot_format(graph_file, split_file)

    raise ValueError(f"Could not find valid data format in {data_path}")


def load_snapshot_format(graph_file, split_file):
    """Load data from snapshot format."""
    net = np.load(graph_file, allow_pickle=True)
    split_data = np.load(split_file, allow_pickle=True)

    if net.dtype == object:
        num_snapshots = len(net)
    else:
        num_snapshots = net.shape[0]

    edges = []
    for t in range(num_snapshots):
        if net.dtype == object:
            adj = net[t].toarray() if hasattr(net[t], 'toarray') else net[t]
        else:
            adj = net[t]

        rows, cols = np.where(adj > 0)
        for i, j in zip(rows, cols):
            if i < j:
                edges.append([int(i), int(j), t])

    data_df = pd.DataFrame(edges, columns=['src', 'dst', 'timestamp'])

    test_neg = split_data['test_neg']
    if test_neg.shape[0] == 2:
        test_neg = test_neg.T

    labels = np.zeros(len(data_df))

    if 'test_neg_id' in split_data:
        test_neg_id = split_data['test_neg_id']
    else:
        test_neg_id = np.full(len(test_neg), num_snapshots - 1)

    for idx, (src, dst) in enumerate(test_neg):
        t = test_neg_id[idx]
        mask = (data_df['src'] == src) & (data_df['dst'] == dst) & (data_df['timestamp'] == t)
        if mask.any():
            labels[mask.values] = 1
        else:
            data_df = pd.concat([data_df, pd.DataFrame([[src, dst, t]],
                                columns=['src', 'dst', 'timestamp'])], ignore_index=True)
            labels = np.append(labels, 1)

    return data_df, labels


def compute_edge_embedding(node_embeddings, src, dst):
    """Compute edge embedding using Hadamard product."""
    return node_embeddings[src] * node_embeddings[dst]


def main():
    args = parse_args()
    config = vars(args)

    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    print("DynWalk (NetWalk) Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize ResultWriter
    writer = ResultWriter()

    data_dir = Path(data_path)
    dataset_name = data_dir.name

    writer.add_metadata(
        method_name="dynwalk",
        dataset=dataset_name,
        seed=args.seed,
    )

    # Load data
    print(f"\nLoading data from {data_path}...")
    data_df, labels = load_edge_data(data_path)

    num_edges = len(data_df)
    num_anomalies = int(labels.sum()) if labels is not None else 0
    print(f"Loaded {num_edges} edges, {num_anomalies} anomalies")

    # Remap node IDs
    all_nodes = set(data_df['src'].values) | set(data_df['dst'].values)
    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    num_nodes = len(node_to_idx)

    data_df['src_idx'] = data_df['src'].map(node_to_idx)
    data_df['dst_idx'] = data_df['dst'].map(node_to_idx)

    print(f"Number of nodes: {num_nodes}")

    # Memory check
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # Split data
    init_size = int(num_edges * args.init_percent)
    init_edges = data_df.iloc[:init_size]

    print(f"\nInitial graph: {len(init_edges)} edges")

    # Build initial graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for _, row in init_edges.iterrows():
        G.add_edge(row['src_idx'], row['dst_idx'])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Generate walks
    print(f"\nGenerating random walks...")
    walks = generate_walks(G, args.number_walks, args.walk_length)
    print(f"Generated {len(walks)} walks")

    # Create features
    print("Creating node features...")
    feature_dim = min(128, num_nodes)
    features = create_node_features(G, walks, num_nodes, feature_dim)
    print(f"Feature shape: {features.shape}")

    # Memory check
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # Train autoencoder
    print(f"\nTraining autoencoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Autoencoder(feature_dim, args.hidden_size, args.representation_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    features_tensor = torch.FloatTensor(features).to(device)
    dataset = TensorDataset(features_tensor, features_tensor)
    dataloader = DataLoader(dataset, batch_size=min(256, num_nodes), shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}")
            writer.spot("training", epoch=epoch + 1, loss=avg_loss)

    # Get embeddings
    model.eval()
    node_embeddings = model.get_embedding(features_tensor).cpu().numpy()
    print(f"Node embeddings: {node_embeddings.shape}")

    # Train K-means on initial edges
    print("\nTraining K-means clustering...")
    train_edge_embeddings = []
    for _, row in init_edges.iterrows():
        emb = compute_edge_embedding(node_embeddings, row['src_idx'], row['dst_idx'])
        train_edge_embeddings.append(emb)
    train_edge_embeddings = np.array(train_edge_embeddings)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    kmeans.fit(train_edge_embeddings)

    # Get threshold from training
    train_distances = kmeans.transform(train_edge_embeddings).min(axis=1)
    threshold = np.percentile(train_distances, 95)

    # Score all edges
    print("\nScoring all edges...")
    all_scores = []
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Scoring"):
        emb = compute_edge_embedding(node_embeddings, row['src_idx'], row['dst_idx'])
        dist = kmeans.transform(emb.reshape(1, -1)).min()
        all_scores.append(dist)

    # Normalize scores
    all_scores = np.array(all_scores)
    if all_scores.max() > all_scores.min():
        all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())

    # Calculate AUC
    if labels is not None and len(labels) == len(all_scores) and labels.sum() > 0:
        try:
            auc = roc_auc_score(labels, all_scores)
            print(f"\nTest AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
            auc = 0.0
    else:
        auc = 0.0

    # Prepare results
    all_edges = data_df[['src', 'dst']].values.tolist()
    all_timestamps = data_df['timestamp'].values.tolist()
    all_labels = labels.tolist() if labels is not None else [0] * len(all_scores)

    print(f"\nTotal predictions: {len(all_scores)}")
    print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

    # Save results
    print("\nSaving results...")
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=all_scores.tolist(),
        edges=all_edges,
        timestamps=all_timestamps,
        ground_truth=all_labels,
    )

    # Timing
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    # Track GPU memory if available
    peak_gpu_mb = None
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\nResource Usage:")
    print(f"  Execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")
    if peak_gpu_mb is not None:
        print(f"  Peak GPU memory: {peak_gpu_mb:.2f}MB")

    # Final metadata
    writer.add_metadata(
        exp_name=os.path.basename(exp_path),
        method_name="dynwalk",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "NetWalk: Deep Embedding for Anomaly Detection in Dynamic Networks (KDD 2018)",
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
        peak_gpu_mb=peak_gpu_mb,
    )

    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
