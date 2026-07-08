"""
GraFlag-integrated training script for AddGraph.
AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN (IJCAI 2019)

AddGraph detects anomalous EDGES in dynamic graphs using attention-based temporal
graph convolutional networks with a GRU for sequence modeling.
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import psutil

# Add AddGraph source paths
sys.path.insert(0, '/app/src/UCI_D_Addgraph/framwork')

# GraFlag integration
from graflag_runner import ResultWriter


def parse_args():
    """Parse command line arguments (passed by graflag_runner --pass-env-args)."""
    parser = argparse.ArgumentParser(description='AddGraph Training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0008, help='Learning rate')
    parser.add_argument('--learning_rate_score', type=float, default=0.0005, help='LR for score network')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--window_size', type=int, default=2, help='Temporal window size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta for score function')
    parser.add_argument('--mui', type=float, default=0.3, help='Mui (bias) for score function')
    parser.add_argument('--gamma', type=float, default=0.6, help='Gamma for margin loss')
    parser.add_argument('--training_ratio', type=float, default=0.5, help='Training ratio')
    parser.add_argument('--anomaly_rate', type=float, default=0.1, help='Anomaly injection rate')
    return parser.parse_args()


def get_config_from_args(args):
    """Convert parsed args to config dict."""
    return {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'learning_rate_score': args.learning_rate_score,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'window_size': args.window_size,
        'seed': args.seed,
        'gpu': args.gpu,
        'beta': args.beta,
        'mui': args.mui,
        'gamma': args.gamma,
        'training_ratio': args.training_ratio,
        'anomaly_rate': args.anomaly_rate,
    }


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============== Model Definitions ==============
# Reimplemented from AddGraph source for better control

class SpGraphAttentionLayer(nn.Module):
    """Sparse Graph Attention Layer."""

    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        Args:
            x: Node features [N, in_features]
            adj: Sparse adjacency matrix [N, N]
        """
        N = x.size(0)
        h = torch.mm(x, self.W)  # [N, out_features]

        # Compute attention coefficients
        edge_index = adj.coalesce().indices()
        edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_e = self.leakyrelu(torch.matmul(edge_h, self.a.t()).squeeze())

        # Sparse softmax
        attention = torch.sparse_coo_tensor(edge_index, edge_e, (N, N))
        attention = torch.sparse.softmax(attention, dim=1)
        attention = attention.coalesce()

        # Apply dropout
        attention_values = self.dropout(attention.values())
        attention = torch.sparse_coo_tensor(attention.indices(), attention_values, (N, N))

        # Aggregate
        h_prime = torch.sparse.mm(attention, h)

        if self.concat:
            return nn.functional.elu(h_prime)
        else:
            return h_prime


class SpGAT(nn.Module):
    """Sparse Graph Attention Network."""

    def __init__(self, nfeat, nhid, nout, dropout, nheads):
        super().__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True)
            for _ in range(nheads)
        ])

        self.out_att = SpGraphAttentionLayer(nhid * nheads, nout, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x


class HCA(nn.Module):
    """Hierarchical Context Attention."""

    def __init__(self, hidden):
        super().__init__()
        self.Q = nn.Parameter(torch.zeros(hidden, hidden))
        nn.init.xavier_uniform_(self.Q.data)
        self.r = nn.Parameter(torch.zeros(hidden))
        nn.init.uniform_(self.r.data)
        self.dropout = nn.Dropout(0.2)

    def forward(self, C):
        """
        Args:
            C: Context tensor [window, N, hidden]
        Returns:
            Aggregated context [N, hidden]
        """
        # C: [w, N, h]
        w, N, h = C.shape

        # Compute attention weights
        # QC: [w, N, h]
        QC = torch.matmul(C, self.Q)
        # scores: [w, N]
        scores = torch.matmul(QC, self.r)
        # attention: [w, N]
        attention = torch.softmax(scores, dim=0)
        attention = self.dropout(attention)

        # Weighted sum: [N, h]
        output = torch.sum(attention.unsqueeze(-1) * C, dim=0)
        return output


class GRUCell(nn.Module):
    """Custom GRU cell for combining current and short-term states."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden

        # Update gate
        self.Wp = nn.Parameter(torch.zeros(hidden, hidden))
        self.Up = nn.Parameter(torch.zeros(hidden, hidden))
        self.bp = nn.Parameter(torch.zeros(hidden))

        # Reset gate
        self.Wr = nn.Parameter(torch.zeros(hidden, hidden))
        self.Ur = nn.Parameter(torch.zeros(hidden, hidden))
        self.br = nn.Parameter(torch.zeros(hidden))

        # Candidate state
        self.Wh = nn.Parameter(torch.zeros(hidden, hidden))
        self.Uh = nn.Parameter(torch.zeros(hidden, hidden))
        self.bh = nn.Parameter(torch.zeros(hidden))

        self._init_weights()

    def _init_weights(self):
        for param in [self.Wp, self.Up, self.Wr, self.Ur, self.Wh, self.Uh]:
            nn.init.xavier_uniform_(param.data)

    def forward(self, current, short):
        """
        Args:
            current: Current state [N, hidden]
            short: Short-term context [N, hidden]
        """
        # Update gate
        p = torch.sigmoid(torch.mm(current, self.Wp) + torch.mm(short, self.Up) + self.bp)
        # Reset gate
        r = torch.sigmoid(torch.mm(current, self.Wr) + torch.mm(short, self.Ur) + self.br)
        # Candidate
        h_tilde = torch.tanh(torch.mm(current, self.Wh) + torch.mm(r * short, self.Uh) + self.bh)
        # Output
        h = (1 - p) * short + p * h_tilde
        return h


class ScoreNetwork(nn.Module):
    """Score network for edge anomaly detection."""

    def __init__(self, hidden, beta=1.0, mui=0.3):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.zeros(hidden))
        nn.init.uniform_(self.a.data)
        nn.init.uniform_(self.b.data)
        self.beta = beta
        self.mui = mui

    def forward(self, hi, hj):
        """
        Args:
            hi: Source node embeddings [batch, hidden] or [hidden]
            hj: Target node embeddings [batch, hidden] or [hidden]
        Returns:
            Anomaly scores [batch] or scalar
        """
        s = torch.sum(self.a * hi + self.b * hj, dim=-1)
        score = torch.sigmoid(self.beta * s + self.mui)
        return score


def load_data(data_path, config):
    """
    Load data from GraFlag dataset format.

    Expected files:
    - acc_*.npy or graph.npy: Graph snapshots
    - split.npz: Train/test split
    """
    data_dir = Path(data_path)

    # Look for graph file
    graph_file = None
    for pattern in ['acc_*.npy', 'graph.npy', '*.npy']:
        matches = list(data_dir.glob(pattern))
        if matches:
            for m in matches:
                if 'sta_' not in m.name:
                    graph_file = m
                    break
            if graph_file:
                break

    if graph_file is None:
        raise FileNotFoundError(f"No graph .npy file found in {data_dir}")

    # Load graph
    print(f"Loading graph from: {graph_file}")
    net = np.load(graph_file, allow_pickle=True)

    # Convert to list of sparse matrices if needed
    if net.dtype != object:
        # Dense tensor - convert to sparse
        num_snapshots = net.shape[0]
        num_nodes = net.shape[1]
        print(f"Graph shape: {net.shape}")
        snapshots = []
        for t in range(num_snapshots):
            snapshots.append(sp.csr_matrix(net[t]))
        net = snapshots
    else:
        num_snapshots = len(net)
        num_nodes = net[0].shape[0]
        snapshots = list(net)

    print(f"Loaded {num_snapshots} snapshots, {num_nodes} nodes")

    # Load split file
    split_file = None
    for pattern in ['split.npz', '*.npz']:
        matches = list(data_dir.glob(pattern))
        if matches:
            split_file = matches[0]
            break

    if split_file:
        print(f"Loading split from: {split_file}")
        split_data = np.load(split_file, allow_pickle=True)

        train_pos = split_data['train_pos']
        train_neg = split_data['train_neg']
        test_pos = split_data['test_pos']
        test_neg = split_data['test_neg']

        # Handle (2, N) vs (N, 2) format
        if train_pos.shape[0] == 2 and len(train_pos.shape) == 2:
            train_pos = train_pos.T
            train_neg = train_neg.T
            test_pos = test_pos.T
            test_neg = test_neg.T

        # Get snapshot IDs if available
        if 'train_pos_id' in split_data:
            train_pos_id = split_data['train_pos_id']
            train_neg_id = split_data['train_neg_id']
            test_pos_id = split_data['test_pos_id']
            test_neg_id = split_data['test_neg_id']
        else:
            # Assign to last snapshots
            n_train = int(num_snapshots * config['training_ratio'])
            train_pos_id = np.full(len(train_pos), n_train - 1)
            train_neg_id = np.full(len(train_neg), n_train - 1)
            test_pos_id = np.full(len(test_pos), num_snapshots - 1)
            test_neg_id = np.full(len(test_neg), num_snapshots - 1)
    else:
        raise FileNotFoundError(f"No split file found in {data_dir}")

    return {
        'snapshots': snapshots,
        'num_nodes': num_nodes,
        'num_snapshots': num_snapshots,
        'train_pos': train_pos,
        'train_neg': train_neg,
        'test_pos': test_pos,
        'test_neg': test_neg,
        'train_pos_id': train_pos_id,
        'train_neg_id': train_neg_id,
        'test_pos_id': test_pos_id,
        'test_neg_id': test_neg_id,
    }


def sparse_mx_to_torch_sparse(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def main():
    args = parse_args()
    config = get_config_from_args(args)
    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    print("AddGraph Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Set seeds
    set_seed(config['seed'])

    # Initialize resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0
    peak_gpu_memory_mb = None

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Load data
    print("\nLoading data...")
    data = load_data(data_path, config)

    # Extract dataset name
    data_dir = Path(data_path)
    dataset_name = data_dir.name

    # Add initial metadata
    writer.add_metadata(
        method_name="addgraph",
        dataset=dataset_name,
        seed=config['seed'],
        num_epochs=config['num_epochs'],
        hidden_dim=config['hidden_dim'],
        learning_rate=config['learning_rate'],
    )

    num_nodes = data['num_nodes']
    num_snapshots = data['num_snapshots']
    hidden_dim = config['hidden_dim']
    window_size = config['window_size']

    print(f"Nodes: {num_nodes}, Snapshots: {num_snapshots}")
    print(f"Train: {len(data['train_pos'])} pos, {len(data['train_neg'])} neg")
    print(f"Test: {len(data['test_pos'])} pos, {len(data['test_neg'])} neg")

    # Initialize models
    print("\nInitializing models...")

    # Node features: one-hot encoding
    nfeat = num_nodes

    net1 = SpGAT(nfeat=nfeat, nhid=hidden_dim // config['num_heads'],
                 nout=hidden_dim, dropout=config['dropout'],
                 nheads=config['num_heads']).to(device)
    net2 = HCA(hidden_dim).to(device)
    net3 = GRUCell(hidden_dim).to(device)
    net4 = ScoreNetwork(hidden_dim, beta=config['beta'], mui=config['mui']).to(device)

    # Optimizers
    optimizer1 = optim.Adam(list(net1.parameters()) + list(net2.parameters()) + list(net3.parameters()),
                           lr=config['learning_rate'], weight_decay=1e-5)
    optimizer2 = optim.Adam(net4.parameters(), lr=config['learning_rate_score'], weight_decay=1e-5)

    # Initialize node features (identity matrix)
    X = torch.eye(num_nodes, device=device)

    # Convert adjacency matrices to torch sparse
    adj_list = [sparse_mx_to_torch_sparse(adj).to(device) for adj in data['snapshots']]

    # Training
    print("\nStarting training...")
    best_auc = 0.0
    best_epoch = 0
    H_history = []  # Store node embeddings history

    for epoch in range(config['num_epochs']):
        # Track memory
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        if torch.cuda.is_available():
            current_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if peak_gpu_memory_mb is None:
                peak_gpu_memory_mb = current_gpu_mb
            else:
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_gpu_mb)

        net1.train()
        net2.train()
        net3.train()
        net4.train()

        total_loss = 0.0
        H_history = []

        # Process each snapshot
        for t in range(num_snapshots):
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # Forward through SpGAT
            H_current = net1(X, adj_list[t])

            # Context aggregation if we have history
            if len(H_history) >= window_size:
                C = torch.stack(H_history[-window_size:], dim=0)
                H_short = net2(C)
                H = net3(H_current, H_short)
            else:
                H = H_current

            H_history.append(H.detach())

            # Compute loss on training edges for this snapshot
            # Get training edges for this snapshot
            train_mask = data['train_pos_id'] == t
            if train_mask.sum() > 0:
                pos_edges = data['train_pos'][train_mask]
                n_pos = len(pos_edges)

                # Sample negative edges (match positive count for balanced loss)
                neg_mask = data['train_neg_id'] == t
                if neg_mask.sum() > 0:
                    available_neg = data['train_neg'][neg_mask]
                    # Sample with replacement if needed
                    if len(available_neg) >= n_pos:
                        neg_idx = np.random.choice(len(available_neg), n_pos, replace=False)
                    else:
                        neg_idx = np.random.choice(len(available_neg), n_pos, replace=True)
                    neg_edges = available_neg[neg_idx]
                else:
                    # Random negative sampling
                    neg_src = np.random.randint(0, num_nodes, n_pos)
                    neg_dst = np.random.randint(0, num_nodes, n_pos)
                    neg_edges = np.stack([neg_src, neg_dst], axis=1)

                # Compute scores
                pos_scores = net4(H[pos_edges[:, 0]], H[pos_edges[:, 1]])
                neg_scores = net4(H[neg_edges[:, 0]], H[neg_edges[:, 1]])

                # Margin loss: max(0, gamma - pos_score + neg_score)
                margin_loss = torch.mean(torch.clamp(config['gamma'] - pos_scores + neg_scores, min=0))

                # Regularization
                reg_loss = 1e-5 * (sum(p.pow(2).sum() for p in net1.parameters()) +
                                   sum(p.pow(2).sum() for p in net2.parameters()) +
                                   sum(p.pow(2).sum() for p in net3.parameters()) +
                                   sum(p.pow(2).sum() for p in net4.parameters()))

                loss = margin_loss + reg_loss
                loss.backward()

                optimizer1.step()
                optimizer2.step()

                total_loss += loss.item()

        # Evaluation on test set
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()

        with torch.no_grad():
            # Get final embeddings
            H_history_eval = []
            for t in range(num_snapshots):
                H_current = net1(X, adj_list[t])
                if len(H_history_eval) >= window_size:
                    C = torch.stack(H_history_eval[-window_size:], dim=0)
                    H_short = net2(C)
                    H = net3(H_current, H_short)
                else:
                    H = H_current
                H_history_eval.append(H)

            # Compute test scores
            test_scores = []
            test_labels = []

            # Positive test edges (normal - label 0)
            for i, (src, dst) in enumerate(data['test_pos']):
                t = data['test_pos_id'][i]
                score = net4(H_history_eval[t][src], H_history_eval[t][dst])
                test_scores.append(1 - score.item())  # Invert: low score = anomaly
                test_labels.append(0)  # Normal

            # Negative test edges (anomalous - label 1)
            for i, (src, dst) in enumerate(data['test_neg']):
                t = data['test_neg_id'][i]
                score = net4(H_history_eval[t][src], H_history_eval[t][dst])
                test_scores.append(1 - score.item())  # Invert: low score = anomaly
                test_labels.append(1)  # Anomaly

            test_auc = roc_auc_score(test_labels, test_scores) if len(set(test_labels)) > 1 else 0.0

        avg_loss = total_loss / max(1, num_snapshots)
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {avg_loss:.4f}, Test AUC: {test_auc:.4f}")

        # Track training metrics
        writer.spot(
            "training",
            epoch=epoch + 1,
            loss=avg_loss,
            test_auc=test_auc,
        )

        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch + 1
            print(f"  -> New best AUC: {best_auc:.4f}")

    print(f"\nTraining completed! Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Generate final predictions
    print("\nGenerating final predictions...")
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()

    with torch.no_grad():
        # Get final embeddings
        H_history_final = []
        for t in range(num_snapshots):
            H_current = net1(X, adj_list[t])
            if len(H_history_final) >= window_size:
                C = torch.stack(H_history_final[-window_size:], dim=0)
                H_short = net2(C)
                H = net3(H_current, H_short)
            else:
                H = H_current
            H_history_final.append(H)

        all_scores = []
        all_labels = []
        all_edges = []
        all_timestamps = []

        # Positive test edges
        for i, (src, dst) in enumerate(data['test_pos']):
            t = data['test_pos_id'][i]
            score = net4(H_history_final[t][src], H_history_final[t][dst])
            all_scores.append(1 - score.item())
            all_labels.append(0)
            all_edges.append([int(src), int(dst)])
            all_timestamps.append(int(t))

        # Negative test edges
        for i, (src, dst) in enumerate(data['test_neg']):
            t = data['test_neg_id'][i]
            score = net4(H_history_final[t][src], H_history_final[t][dst])
            all_scores.append(1 - score.item())
            all_labels.append(1)
            all_edges.append([int(src), int(dst)])
            all_timestamps.append(int(t))

    print(f"Total predictions: {len(all_scores)}")
    print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    final_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
    print(f"Final Test AUC: {final_auc:.4f}")

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

    if torch.cuda.is_available():
        final_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if peak_gpu_memory_mb is not None:
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, final_gpu_mb)

    print(f"\nResource Usage:")
    print(f"  Total execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")
    if peak_gpu_memory_mb is not None:
        print(f"  Peak GPU memory: {peak_gpu_memory_mb:.2f}MB")

    # Add final metadata
    writer.add_metadata(
        exp_name=os.path.basename(exp_path),
        method_name="addgraph",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN (IJCAI 2019)",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_snapshots": num_snapshots,
                "num_nodes": num_nodes,
                "train_edges": len(data['train_pos']) + len(data['train_neg']),
                "test_edges": len(data['test_pos']) + len(data['test_neg']),
            },
            "training_info": {
                "best_auc": float(best_auc),
                "best_epoch": best_epoch,
                "final_auc": float(final_auc),
                "total_epochs": config['num_epochs'],
            },
            "model_info": {
                "architecture": ["SpGAT", "HCA", "GRU", "Score"],
                "hidden_dim": config['hidden_dim'],
                "num_heads": config['num_heads'],
                "window_size": config['window_size'],
                "dropout": config['dropout'],
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
        peak_gpu_mb=peak_gpu_memory_mb,
    )

    # Finalize results
    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"Best Test AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
