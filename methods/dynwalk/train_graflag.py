"""
GraFlag-integrated implementation of NetWalk.
NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks (KDD 2018)

This implementation follows the NetWalk paper's approach:
1. Random walks on dynamic graph snapshots
2. Deep autoencoder for node embeddings
3. K-means clustering for anomaly detection
4. Distance-based anomaly scoring
"""

import random
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from graflag_runner import (
    ResultWriter, device, info, load_dataset, params, paths, seed_all, warning,
)


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    This replaces an argparse parser that restated every key in .env. The
    annotations are what params() coerces to, so _HIDDEN_SIZE=64 arrives as
    an int, and a stale .env key is dropped here instead of making argparse
    exit 2 on an unrecognized argument.
    """

    representation_size: int = 32   # embedding dimension
    walk_length: int = 5            # random walk length
    number_walks: int = 10          # walks per node
    init_percent: float = 0.5       # fraction of the stream used to seed the graph
    learning_rate: float = 0.001
    epochs: int = 50
    hidden_size: int = 64
    n_clusters: int = 5             # K-means clusters over the embeddings
    seed: int = 42


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


def compute_edge_embedding(node_embeddings, src, dst):
    """Compute edge embedding using Hadamard product."""
    return node_embeddings[src] * node_embeddings[dst]


def main():
    run = paths()
    args = Config(**params(Config))
    config = asdict(args)

    info(f"[INFO] DynWalk (NetWalk) on {run.dataset}: {config}")
    seed_all(args.seed)

    writer = ResultWriter()

    data_df, labels = load_dataset(run.data)

    num_edges = len(data_df)
    num_anomalies = int(labels.sum())
    info(f"[INFO] Loaded {num_edges} edges, {num_anomalies} anomalies")

    # Remap node IDs
    all_nodes = set(data_df['src'].values) | set(data_df['dst'].values)
    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    num_nodes = len(node_to_idx)

    data_df['src_idx'] = data_df['src'].map(node_to_idx)
    data_df['dst_idx'] = data_df['dst'].map(node_to_idx)

    print(f"Number of nodes: {num_nodes}")

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
    print("\nGenerating random walks...")
    walks = generate_walks(G, args.number_walks, args.walk_length)
    print(f"Generated {len(walks)} walks")

    # Create features
    print("Creating node features...")
    feature_dim = min(128, num_nodes)
    features = create_node_features(G, walks, num_nodes, feature_dim)
    print(f"Feature shape: {features.shape}")

    # Train autoencoder
    print("\nTraining autoencoder...")
    # device() reads _GPU, which `graflag run --no-gpu` sets to -1. This used
    # to take a GPU whenever one was visible, including on a service Swarm
    # scheduled without reserving one.
    dev = device()

    model = Autoencoder(feature_dim, args.hidden_size, args.representation_size).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    features_tensor = torch.FloatTensor(features).to(dev)
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

    # No threshold is derived here. A 95th-percentile cut over the training
    # distances used to be computed and then dropped on the floor -- the
    # scores are min-max normalised below, so a raw-distance cut is not on
    # their scale and reporting it would have been misleading.

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

    # Calculate AUC. None, not 0.0: 0.0 is a real AUC -- the score of a
    # perfectly inverted ranking -- so reporting it for "could not be
    # computed" made an unmeasurable run look like the worst possible one.
    auc = None
    if labels is None or len(labels) != len(all_scores):
        warning("[WARN] No usable labels for this stream; reporting auc null")
    elif not 0 < labels.sum() < len(labels):
        warning(f"[WARN] Only one class present ({int(labels.sum())} of "
                f"{len(labels)} edges anomalous); reporting auc null")
    else:
        auc = float(roc_auc_score(labels, all_scores))
        print(f"\nTest AUC: {auc:.4f}")

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

    # No timing or psutil sampling here: graflag_runner measures exec time,
    # peak memory and peak GPU from outside the method and its numbers win
    # (runner._merge_runtime_metadata). The block that used to be here
    # reported 409 MB against the monitor's 907 MB.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="dynwalk",
        dataset=run.dataset,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "NetWalk: Deep Embedding for Anomaly Detection in Dynamic Networks (KDD 2018)",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": run.dataset,
                "num_edges": num_edges,
                "num_nodes": num_nodes,
                "num_anomalies": num_anomalies,
            },
            "results": {
                "auc": auc,
            },
        },
    )

    shown = "n/a" if auc is None else f"{auc:.4f}"
    info(f"[OK] AUC {shown}, results written to {writer.finalize()}")


if __name__ == "__main__":
    main()
