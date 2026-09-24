"""
GraFlag-integrated training script for StrGNN.
StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs (CIKM 2021)

StrGNN detects anomalous EDGES in dynamic graphs by learning structural patterns
from temporal graph snapshots.
"""

import math
import pickle
import random
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as ssp
import torch
import torch.optim as optim

from graflag_runner import (
    ResultWriter, device, info, params, paths, seed_all, snapshot_files,
    split_test_edges, upstream,
)

# The StrGNN clone. upstream() anchors on this file's directory, so the
# script no longer hardcodes /app and is runnable wherever the Dockerfile
# puts the checkout; it raises if the clone is missing instead of failing
# later with an unexplained ImportError.
upstream("src/detection")
upstream("src/pytorch_DGCNN")


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    Replaces an argparse parser plus a get_config_from_args() that restated
    every key a third time. The three values that are not plain scalars are
    derived in __post_init__, which is where that translation belongs.
    """

    seed: int = 1
    test_ratio: float = 0.2
    window: int = 5
    gpu: int = 0                    # GPU index; -1 means CPU
    hop: int = 1                    # enclosing subgraph hop
    max_nodes_per_hop: int = 100    # <= 0 means unlimited
    use_embedding: int = 0          # node2vec embeddings, 1 to enable
    sortpooling_k: float = 0.6
    hidden: int = 128
    num_epochs: int = 50
    learning_rate: float = 0.0001
    batch_size: int = 32
    dropout: int = 1                # 1 to enable

    def __post_init__(self):
        # Upstream reads these three as None / bool, not as the ints .env
        # can express.
        self.max_nodes_per_hop = self.max_nodes_per_hop if self.max_nodes_per_hop > 0 else None
        self.use_embedding = self.use_embedding == 1
        self.dropout = self.dropout == 1


def load_strgnn_data(data_path):
    """
    Load data in StrGNN format.

    Expected files:
    - graph.npy or acc_*.npy: Graph snapshots with shape (T, N, N)
    - split.npz or *.npz: Train/test split with edge indices

    Or convert from other formats.
    """
    graph_file, split_file = snapshot_files(data_path)

    if graph_file is None:
        raise FileNotFoundError(f"No graph .npy file found in {data_path}")

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
        return net, np.load(split_file, allow_pickle=True)

    print("No split file found - will generate train/test split")
    return net, None


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
    run = paths()
    config = asdict(Config(**params(Config)))

    info(f"[INFO] StrGNN on {run.dataset}: {config}")

    # This used to assign CUDA_VISIBLE_DEVICES *after* `import torch`, which
    # only works while nothing has touched CUDA yet, and then decided the
    # mode from torch.cuda.is_available() rather than from _GPU. device()
    # reads _GPU (-1 means CPU), and set_device pins the index so the bare
    # .cuda() calls inside pytorch_DGCNN land on the right one.
    dev = device(config['gpu'])
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)

    seed_all(config['seed'])

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Load data
    print("\nLoading data...")
    net, split_data = load_strgnn_data(run.data)

    # From the clone that upstream() put on sys.path. `cmd_args` is the
    # namespace upstream builds at import time; the block below fills it
    # in from this method's own Config.
    from util_functions import dyn_links2subgraphs
    from main import Classifier, loop_dataset, cmd_args

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
    cache_file = run.exp / f"subgraphs_h{config['hop']}.pkl"

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
    cmd_args.mode = 'gpu' if dev.type == 'cuda' else 'cpu'
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

    # dyn_links2subgraphs concatenates test_pos then test_neg, which is the
    # order split_test_edges returns -- so the scores below line up with it.
    all_edges, all_timestamps, all_labels = split_test_edges({
        'test_pos': test_pos, 'test_neg': test_neg,
        'test_pos_id': test_pos_id, 'test_neg_id': test_neg_id,
    })
    if len(test_graphs) != len(all_edges):
        raise RuntimeError(
            f"dyn_links2subgraphs returned {len(test_graphs)} test subgraphs "
            f"for {len(all_edges)} test edges"
        )

    all_scores = []

    with torch.no_grad():
        for i, graph_list in enumerate(test_graphs):
            # Upstream labels test_pos 1 and test_neg 0 -- real edge against
            # sampled non-edge -- so its class 1 is the *normal* one, the
            # opposite of the anomaly label. Check it rather than trust it: if
            # upstream ever reorders or relabels, both the scores and the
            # ground truth invert together and auc_roc still looks right, so
            # nothing downstream would notice.
            upstream_label = int(graph_list[-1].label)
            if upstream_label != 1 - all_labels[i]:
                raise RuntimeError(
                    f"test subgraph {i} carries upstream label "
                    f"{upstream_label}, expected {1 - all_labels[i]}: "
                    f"dyn_links2subgraphs no longer puts test_pos first or no "
                    f"longer labels it 1"
                )

            batch = [graph_list]
            output = classifier(batch)

            # Handle tuple output (logits, loss) from classifier
            logits = output[0] if isinstance(output, tuple) else output

            # softmax(...)[:, 1] is P(upstream class 1) = P(normal). The
            # contract wants a score that rises with anomalousness.
            normal = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]
            all_scores.append(float(1.0 - normal))

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

    # graflag_runner measures exec time, peak memory and peak GPU from
    # outside the method and its numbers win (_merge_runtime_metadata), so
    # the psutil and torch.cuda sampling that used to be here was recorded
    # and then overwritten.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="strgnn",
        # run.dataset, not the graph file's stem: a run on uci_snapshot used
        # to record itself as "acc_graph".
        dataset=run.dataset,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs (CIKM 2021)",
            "task": "edge_anomaly_detection",
            "dataset_info": {
                "name": run.dataset,
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

    info(f"[OK] Best test AUC {best_auc:.4f}, results written to {writer.finalize()}")


if __name__ == "__main__":
    main()
