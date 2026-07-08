"""
GraFlag-integrated training script for StrGNN.
StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs (CIKM 2021)

StrGNN detects anomalous EDGES in dynamic graphs by learning structural patterns
from temporal graph snapshots.
"""

import os
import sys
import time
import random
import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as ssp
import torch
import torch.optim as optim
import psutil

# Add StrGNN source paths
sys.path.insert(0, '/app/src/detection')
sys.path.insert(0, '/app/src/pytorch_DGCNN')

# GraFlag integration
from graflag_runner import ResultWriter


def parse_args():
    """Parse command line arguments (passed by graflag_runner --pass-env-args)."""
    parser = argparse.ArgumentParser(description='StrGNN Training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio')
    parser.add_argument('--window', type=int, default=5, help='Window size')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--hop', type=int, default=1, help='Enclosing subgraph hop')
    parser.add_argument('--max_nodes_per_hop', type=int, default=100, help='Max nodes per hop')
    parser.add_argument('--use_embedding', type=int, default=0, help='Use node2vec embeddings')
    parser.add_argument('--sortpooling_k', type=float, default=0.6, help='SortPooling k')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=int, default=1, help='Use dropout (1=True, 0=False)')
    return parser.parse_args()


def get_config_from_args(args):
    """Convert parsed args to config dict."""
    return {
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'window': args.window,
        'gpu': args.gpu,
        'hop': args.hop,
        'max_nodes_per_hop': args.max_nodes_per_hop if args.max_nodes_per_hop > 0 else None,
        'use_embedding': args.use_embedding == 1,
        'sortpooling_k': args.sortpooling_k,
        'hidden': args.hidden,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'dropout': args.dropout == 1,
    }


def load_strgnn_data(data_path):
    """
    Load data in StrGNN format.

    Expected files:
    - graph.npy or acc_*.npy: Graph snapshots with shape (T, N, N)
    - split.npz or *.npz: Train/test split with edge indices

    Or convert from other formats.
    """
    data_dir = Path(data_path)

    # Look for graph file
    graph_file = None
    for pattern in ['graph.npy', 'acc_*.npy', '*.npy']:
        matches = list(data_dir.glob(pattern))
        if matches:
            # Prefer non-sta (accumulated) files
            for m in matches:
                if 'sta_' not in m.name:
                    graph_file = m
                    break
            if graph_file is None:
                graph_file = matches[0]
            break

    if graph_file is None:
        raise FileNotFoundError(f"No graph .npy file found in {data_dir}")

    # Look for split file
    split_file = None
    for pattern in ['split.npz', '*.npz']:
        matches = list(data_dir.glob(pattern))
        if matches:
            split_file = matches[0]
            break

    print(f"Loading graph from: {graph_file}")
    net = np.load(graph_file, allow_pickle=True)

    # Handle different data formats
    # Format 1: Array of sparse matrices (T,) where each is (N, N)
    # Format 2: Dense 3D tensor (T, N, N) - needs conversion to sparse
    if net.dtype == object:
        # Array of sparse matrices - already in correct format
        num_snapshots = len(net)
        num_nodes = net[0].shape[0]
        print(f"Graph: {num_snapshots} sparse snapshots, {num_nodes} nodes each")
    else:
        # Dense 3D tensor - convert to array of sparse matrices for StrGNN
        num_snapshots = net.shape[0]
        num_nodes = net.shape[1]
        print(f"Graph shape: {net.shape} (T={num_snapshots} snapshots, N={num_nodes} nodes)")
        print("Converting dense tensor to sparse matrices...")
        sparse_net = np.empty(num_snapshots, dtype=object)
        for t in range(num_snapshots):
            sparse_net[t] = ssp.csr_matrix(net[t])
        net = sparse_net
        print(f"Converted to {num_snapshots} sparse matrices")

    if split_file:
        print(f"Loading split from: {split_file}")
        split_data = np.load(split_file, allow_pickle=True)
        return net, split_data, graph_file.stem
    else:
        print("No split file found - will generate train/test split")
        return net, None, graph_file.stem


def generate_split(net, test_ratio, window_size):
    """Generate train/test split from graph snapshots."""
    from util_functions import sample_neg

    num_graphs = len(net)
    num_train = int(math.ceil(num_graphs * (1 - test_ratio)))

    train_id = []
    train_pos_list = []
    train_neg_list = []

    for i in range(window_size, num_train):
        # Convert to sparse matrix
        adj = ssp.csr_matrix(net[i])
        train_pos, train_neg, _, _ = sample_neg(adj, 0)
        ids = np.ones(len(train_pos[0]), dtype=int) * i
        train_id.append(ids)
        train_pos_list.append(np.array(train_pos).T)
        train_neg_list.append(np.array(train_neg).T)

    train_pos_id = np.concatenate(train_id)
    train_pos = np.concatenate(train_pos_list, axis=0)
    train_neg_id = train_pos_id.copy()
    train_neg = np.concatenate(train_neg_list, axis=0)

    test_id = []
    test_pos_list = []
    test_neg_list = []

    for i in range(num_train, num_graphs):
        adj = ssp.csr_matrix(net[i])
        test_pos, test_neg, _, _ = sample_neg(adj, 0)
        ids = np.ones(len(test_pos[0]), dtype=int) * i
        test_id.append(ids)
        test_pos_list.append(np.array(test_pos).T)
        test_neg_list.append(np.array(test_neg).T)

    test_pos_id = np.concatenate(test_id)
    test_pos = np.concatenate(test_pos_list, axis=0)
    test_neg_id = test_pos_id.copy()
    test_neg = np.concatenate(test_neg_list, axis=0)

    return {
        'train_pos_id': train_pos_id,
        'train_neg_id': train_neg_id,
        'test_pos_id': test_pos_id,
        'test_neg_id': test_neg_id,
        'train_pos': train_pos,
        'train_neg': train_neg,
        'test_pos': test_pos,
        'test_neg': test_neg,
    }


def main():
    args = parse_args()
    config = get_config_from_args(args)
    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    print("StrGNN Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    # Set seeds
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Initialize resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0
    peak_gpu_memory_mb = None

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Load data
    print("\nLoading data...")
    net, split_data, dataset_name = load_strgnn_data(data_path)

    # Add initial metadata
    writer.add_metadata(
        method_name="strgnn",
        dataset=dataset_name,
        seed=config['seed'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        window=config['window'],
        hop=config['hop'],
    )

    # Import StrGNN modules after setting paths
    from util_functions import dyn_links2subgraphs
    from main import Classifier, loop_dataset, cmd_args
    from util import GNNGraph

    # Get or generate split
    if split_data is not None:
        train_pos_id = split_data['train_pos_id']
        train_neg_id = split_data['train_neg_id']
        test_pos_id = split_data['test_pos_id']
        test_neg_id = split_data['test_neg_id']
        train_pos = split_data['train_pos']
        train_neg = split_data['train_neg']
        test_pos = split_data['test_pos']
        test_neg = split_data['test_neg']

        # Handle (2, N) vs (N, 2) format - ensure we have (N, 2)
        if train_pos.shape[0] == 2 and len(train_pos.shape) == 2:
            train_pos = train_pos.T
            train_neg = train_neg.T
            test_pos = test_pos.T
            test_neg = test_neg.T
    else:
        print("Generating train/test split...")
        split = generate_split(net, config['test_ratio'], config['window'])
        train_pos_id = split['train_pos_id']
        train_neg_id = split['train_neg_id']
        test_pos_id = split['test_pos_id']
        test_neg_id = split['test_neg_id']
        train_pos = split['train_pos']
        train_neg = split['train_neg']
        test_pos = split['test_pos']
        test_neg = split['test_neg']

    print(f"Train positive edges: {len(train_pos)}")
    print(f"Train negative edges: {len(train_neg)}")
    print(f"Test positive edges: {len(test_pos)}")
    print(f"Test negative edges: {len(test_neg)}")

    # Check for cached subgraphs
    cache_file = Path(exp_path) / f"subgraphs_h{config['hop']}.pkl"

    if cache_file.exists():
        print(f"Loading cached subgraphs from {cache_file}")
        with open(cache_file, 'rb') as f:
            train_graphs, test_graphs, max_n_label = pickle.load(f)
    else:
        print("Extracting enclosing subgraphs...")
        train_graphs, test_graphs, max_n_label = dyn_links2subgraphs(
            net, config['window'],
            train_pos_id, (train_pos[:, 0], train_pos[:, 1]),
            train_neg_id, (train_neg[:, 0], train_neg[:, 1]),
            test_pos_id, (test_pos[:, 0], test_pos[:, 1]),
            test_neg_id, (test_neg[:, 0], test_neg[:, 1]),
            h=config['hop'],
            max_nodes_per_hop=config['max_nodes_per_hop'],
            node_information=None
        )
        # Cache for future runs
        print(f"Caching subgraphs to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump([train_graphs, test_graphs, max_n_label], f, protocol=4)

    print(f"# train graphs: {len(train_graphs)}, # test graphs: {len(test_graphs)}")
    print(f"Max node label: {max_n_label}")

    # Configure DGCNN
    cmd_args.gm = 'DGCNN'
    cmd_args.sortpooling_k = config['sortpooling_k']
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = config['hidden']
    cmd_args.out_dim = 0
    cmd_args.dropout = config['dropout']
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu' if torch.cuda.is_available() else 'cpu'
    cmd_args.num_epochs = config['num_epochs']
    cmd_args.learning_rate = config['learning_rate']
    cmd_args.batch_size = config['batch_size']
    cmd_args.printAUC = True
    cmd_args.feat_dim = max_n_label + 1
    cmd_args.attr_dim = 0
    cmd_args.edge_feat_dim = 0
    cmd_args.window = config['window']
    cmd_args.conv1d_activation = 'ReLU'

    # Calculate sortpooling_k if it's a fraction
    if cmd_args.sortpooling_k <= 1:
        A = []
        for i in train_graphs:
            A.append(i[-1])
        for i in test_graphs:
            A.append(i[-1])
        num_nodes_list = sorted([g.num_nodes for g in A])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print(f"SortPooling k: {cmd_args.sortpooling_k}")

    # Create classifier
    print("\nBuilding model...")
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    # Training loop
    print("\nStarting training...")
    train_idxes = list(range(len(train_graphs)))
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(cmd_args.num_epochs):
        # Track memory
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        if torch.cuda.is_available():
            current_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if peak_gpu_memory_mb is None:
                peak_gpu_memory_mb = current_gpu_mb
            else:
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_gpu_mb)

        # Train
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        train_loss, train_acc, train_auc = avg_loss[0], avg_loss[1], avg_loss[2]

        # Evaluate
        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        test_loss_val, test_acc, test_auc = test_loss[0], test_loss[1], test_loss[2]
        avg_precision = test_loss[3] if len(test_loss) > 3 else 0.0
        pr_auc = test_loss[4] if len(test_loss) > 4 else 0.0

        print(f"Epoch {epoch+1}/{cmd_args.num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | "
              f"Test Loss: {test_loss_val:.4f}, Test AUC: {test_auc:.4f}, PR-AUC: {pr_auc:.4f}")

        # Track training metrics
        writer.spot(
            "training",
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            train_auc=train_auc,
            test_loss=test_loss_val,
            test_acc=test_acc,
            test_auc=test_auc,
            avg_precision=avg_precision,
            pr_auc=pr_auc,
        )

        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch + 1
            print(f"  -> New best AUC: {best_auc:.4f}")

    print(f"\nTraining completed! Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    # Get predictions for all test edges
    print("\nGenerating predictions for test edges...")
    classifier.eval()

    all_scores = []
    all_labels = []
    all_edges = []
    all_timestamps = []

    # Process test graphs to get predictions
    with torch.no_grad():
        for i, graph_list in enumerate(test_graphs):
            # Get the label (1 for positive edge, 0 for negative)
            label = graph_list[-1].label

            # Get prediction score
            # This is simplified - in practice you'd batch these
            batch = [graph_list]
            output = classifier(batch)

            # Handle tuple output (logits, loss) from classifier
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]

            all_scores.append(float(prob))
            all_labels.append(int(label))

            # Determine if this is from positive or negative test set
            if i < len(test_pos):
                edge = [int(test_pos[i, 0]), int(test_pos[i, 1])]
                timestamp = int(test_pos_id[i])
            else:
                idx = i - len(test_pos)
                edge = [int(test_neg[idx, 0]), int(test_neg[idx, 1])]
                timestamp = int(test_neg_id[idx])

            all_edges.append(edge)
            all_timestamps.append(timestamp)

    print(f"Total test predictions: {len(all_scores)}")
    print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

    # Calculate final metrics
    from sklearn.metrics import roc_auc_score
    final_auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
    print(f"Final Test AUC: {final_auc:.4f}")

    # Save results
    print("\nSaving results in EDGE_STREAM_ANOMALY_SCORES format...")
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
        method_name="strgnn",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs (CIKM 2021)",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_snapshots": len(net) if net.dtype == object else net.shape[0],
                "num_nodes": net[0].shape[0] if net.dtype == object else net.shape[1],
                "train_edges": len(train_pos) + len(train_neg),
                "test_edges": len(test_pos) + len(test_neg),
            },
            "training_info": {
                "best_auc": float(best_auc),
                "best_epoch": best_epoch,
                "final_auc": float(final_auc),
                "total_epochs": cmd_args.num_epochs,
            },
            "model_info": {
                "type": "DGCNN",
                "sortpooling_k": cmd_args.sortpooling_k,
                "hidden": config['hidden'],
                "window": config['window'],
                "hop": config['hop'],
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
