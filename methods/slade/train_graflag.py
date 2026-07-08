"""
GraFlag-integrated training script for SLADE.
SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning
"""

import math
import logging
import time
import sys
import random
import os
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model.SLADE_TGN import SLADE_TGN
from utils.utils import get_neighbor_finder
from utils.data_processing import Data
from evaluation.evaluation import eval_anomaly_node_detection

# GraFlag integration
from graflag_runner import ResultWriter


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(data_path, training_ratio=0.85):
    """
    Load data from GraFlag dataset directory.
    Expected format: ml_{dataset_name}.csv with columns: u, i, ts, label, idx
    """
    # Find the CSV file in the data directory
    data_dir = Path(data_path)
    csv_files = list(data_dir.glob("ml_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No ml_*.csv file found in {data_dir}")

    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")

    graph_df = pd.read_csv(csv_file)

    # Handle different column naming conventions
    if 'u' in graph_df.columns:
        sources = graph_df.u.values
    elif 'source' in graph_df.columns:
        sources = graph_df.source.values
    else:
        sources = graph_df.iloc[:, 0].values

    if 'i' in graph_df.columns:
        destinations = graph_df.i.values
    elif 'destination' in graph_df.columns:
        destinations = graph_df.destination.values
    else:
        destinations = graph_df.iloc[:, 1].values

    if 'ts' in graph_df.columns:
        timestamps = graph_df.ts.values
    elif 'timestamp' in graph_df.columns:
        timestamps = graph_df.timestamp.values
    else:
        timestamps = graph_df.iloc[:, 2].values

    if 'label' in graph_df.columns:
        labels = graph_df.label.values
    elif 'labels' in graph_df.columns:
        labels = graph_df.labels.values
    else:
        labels = graph_df.iloc[:, 3].values

    if 'idx' in graph_df.columns:
        edge_idxs = graph_df.idx.values
    else:
        edge_idxs = np.arange(len(sources))

    # Split based on training ratio
    test_time = np.quantile(timestamps, training_ratio)

    train_mask = timestamps <= test_time
    test_mask = timestamps > test_time

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(
        sources[train_mask], destinations[train_mask], timestamps[train_mask],
        edge_idxs[train_mask], labels[train_mask]
    )
    test_data = Data(
        sources[test_mask], destinations[test_mask], timestamps[test_mask],
        edge_idxs[test_mask], labels[test_mask]
    )

    return full_data, train_data, test_data, graph_df


def parse_args():
    """Parse command line arguments (passed by graflag_runner --pass-env-args)."""
    parser = argparse.ArgumentParser(description='SLADE Training')
    parser.add_argument('--bs', type=int, default=100, help='Batch size')
    parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--message_dim', type=int, default=128, help='Message dimension')
    parser.add_argument('--memory_dim', type=int, default=256, help='Memory dimension')
    parser.add_argument('--agg_type', type=str, default='TGAT', help='Aggregation type')
    parser.add_argument('--negative_memory_type', type=str, default='train', help='Negative memory type')
    parser.add_argument('--message_updater', type=str, default='mlp', help='Message updater type')
    parser.add_argument('--memory_updater', type=str, default='gru', help='Memory updater type')
    parser.add_argument('--training_ratio', type=float, default=0.85, help='Training ratio')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--srf', type=float, default=0.1, help='Source recovery factor')
    parser.add_argument('--drf', type=float, default=0.1, help='Drift recovery factor')
    parser.add_argument('--only_drift_loss_score', type=int, default=0, help='Only drift loss score')
    parser.add_argument('--only_recovery_loss_score', type=int, default=0, help='Only recovery loss score')
    parser.add_argument('--only_drift_score', type=int, default=0, help='Only drift score')
    parser.add_argument('--only_rec_score', type=int, default=0, help='Only rec score')
    return parser.parse_args()


def get_config_from_args(args):
    """Convert parsed args to config dict."""
    return {
        'bs': args.bs,
        'n_degree': args.n_degree,
        'n_head': args.n_head,
        'n_epoch': args.n_epoch,
        'lr': args.lr,
        'n_runs': args.n_runs,
        'seed': args.seed,
        'drop_out': args.drop_out,
        'gpu': args.gpu,
        'message_dim': args.message_dim,
        'memory_dim': args.memory_dim,
        'agg_type': args.agg_type,
        'negative_memory_type': args.negative_memory_type,
        'message_updater': args.message_updater,
        'memory_updater': args.memory_updater,
        'training_ratio': args.training_ratio,
        'lr_decay': args.lr_decay,
        'weight_decay': args.weight_decay,
        'srf': args.srf,
        'drf': args.drf,
        'only_drift_loss_score': args.only_drift_loss_score == 1,
        'only_recovery_loss_score': args.only_recovery_loss_score == 1,
        'only_drift_score': args.only_drift_score == 1,
        'only_rec_score': args.only_rec_score == 1,
    }


def main():
    # Get configuration
    args = parse_args()
    config = get_config_from_args(args)
    data_path = os.environ.get("DATA")

    print(f"SLADE Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set random seed
    set_seed(config['seed'])

    # Setup device
    device_string = f"cuda:{config['gpu']}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    print(f"Using device: {device}")

    # Initialize resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0
    peak_gpu_memory_mb = None

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Extract dataset name from path
    data_dir = Path(data_path)
    dataset_name = data_dir.name.replace('slade_', '')

    # Add initial metadata
    writer.add_metadata(
        method_name="slade",
        dataset=dataset_name,
        seed=config['seed'],
        n_epochs=config['n_epoch'],
        batch_size=config['bs'],
        learning_rate=config['lr'],
        message_dim=config['message_dim'],
        memory_dim=config['memory_dim'],
        n_degree=config['n_degree'],
        n_head=config['n_head'],
        training_ratio=config['training_ratio'],
        srf=config['srf'],
        drf=config['drf'],
    )

    # Load data
    print("\nLoading data...")
    full_data, train_data, test_data, graph_df = load_data(data_path, config['training_ratio'])

    print(f"Full data: {full_data.n_interactions} interactions, {full_data.n_unique_nodes} unique nodes")
    print(f"Train data: {train_data.n_interactions} interactions")
    print(f"Test data: {test_data.n_interactions} interactions")
    print(f"Anomaly ratio: {np.mean(full_data.labels):.4f}")

    # Get maximum node index
    max_idx = max(full_data.unique_nodes)

    # Build neighbor finders
    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_idx)

    # Pre-compute neighbors for training data
    print("Pre-computing neighbors...")
    src_neighbors, _, src_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.sources, train_data.timestamps, config['n_degree']
    )
    dst_neighbors, _, dst_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.destinations, train_data.timestamps, config['n_degree']
    )

    # Run multiple runs if specified
    all_test_aucs = []
    all_pred_scores = []

    for run_idx in range(config['n_runs']):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{config['n_runs']}")
        print(f"{'='*60}")

        # Initialize model
        model = SLADE_TGN(
            neighbor_finder=train_ngh_finder,
            n_nodes=full_data.n_unique_nodes,
            n_edges=full_data.n_interactions,
            device=device,
            n_layers=1,  # Only 1 hop neighbor aggregation
            n_heads=config['n_head'],
            dropout=config['drop_out'],
            message_dimension=config['message_dim'],
            memory_dimension=config['memory_dim'],
            n_neighbors=config['n_degree'],
            memory_agg_type=config['agg_type'],
            negative_memory_type=config['negative_memory_type'],
            message_updater=config['message_updater'],
            memory_updater=config['memory_updater'],
            src_reg_factor=config['srf'],
            dst_reg_factor=config['drf'],
            only_drift_loss=config['only_drift_loss_score'],
            only_recovery_loss=config['only_recovery_loss_score']
        )
        model = model.to(device)

        # Prepare training data tensors
        train_data_sources = torch.from_numpy(train_data.sources).long().to(device)
        train_data_destinations = torch.from_numpy(train_data.destinations).long().to(device)
        train_data_timestamps = torch.from_numpy(train_data.timestamps).float().to(device)
        train_data_src_neighbors = torch.from_numpy(src_neighbors).long().to(device)
        train_data_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(device)
        train_data_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(device)
        train_data_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / config['bs'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

        negative_train_nodes = torch.from_numpy(
            np.array(list(set(train_data.destinations) | set(train_data.sources)))
        ).long().to(device)

        best_val_auc = 0.0
        val_aucs = []

        for epoch in range(config['n_epoch']):
            # Track memory
            current_memory_mb = process.memory_info().rss / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, current_memory_mb)

            if torch.cuda.is_available():
                current_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if peak_gpu_memory_mb is None:
                    peak_gpu_memory_mb = current_gpu_mb
                else:
                    peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_gpu_mb)

            # Reset memory at start of each epoch
            model.memory.__init_memory__()
            model.set_neighbor_finder(train_ngh_finder)

            m_loss = []

            # Training loop
            for k in tqdm(range(num_batch), desc=f"Epoch {epoch+1}/{config['n_epoch']}"):
                optimizer.zero_grad()
                s_idx = k * config['bs']
                e_idx = min(num_instance, s_idx + config['bs'])

                sources_batch = train_data_sources[s_idx:e_idx]
                destinations_batch = train_data_destinations[s_idx:e_idx]
                timestamps_batch = train_data_timestamps[s_idx:e_idx]
                src_neighbors_batch = train_data_src_neighbors[s_idx:e_idx]
                dst_neighbors_batch = train_data_dst_neighbors[s_idx:e_idx]
                src_neighbors_time_batch = train_data_src_neighbors_time[s_idx:e_idx]
                dst_neighbors_time_batch = train_data_dst_neighbors_time[s_idx:e_idx]

                model.train()
                _, _, _, _, contrastive_loss = model.compute_node_diff_score(
                    sources_batch, destinations_batch, timestamps_batch,
                    src_neighbors_batch, dst_neighbors_batch,
                    src_neighbors_time_batch, dst_neighbors_time_batch,
                    config['n_degree'], negative_train_nodes
                )

                loss = contrastive_loss

                # Skip gradient update on first batch for drift-only mode
                if config['only_drift_loss_score'] and k == 0:
                    continue

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())
                model.memory.detach_memory()

            scheduler.step()

            # Evaluation
            model.set_neighbor_finder(full_ngh_finder)

            val_auc, pred_score, _ = eval_anomaly_node_detection(
                model, test_data, config['bs'],
                n_neighbors=config['n_degree'],
                device=device,
                only_rec_score=config['only_rec_score'] or config['only_recovery_loss_score'],
                only_drift_score=config['only_drift_loss_score'] or config['only_drift_score']
            )

            avg_loss = sum(m_loss) / len(m_loss) if m_loss else 0.0
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

            val_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                print(f"  -> New best AUC: {best_val_auc:.4f}")

            # Track training metrics
            writer.spot(
                "training",
                epoch=epoch + 1,
                run=run_idx + 1,
                loss=avg_loss,
                val_auc=val_auc,
            )

        # Final evaluation on test data
        final_auc = val_aucs[-1] if val_aucs else 0.0
        all_test_aucs.append(final_auc)

        print(f"\nRun {run_idx + 1} - Final Test AUC: {final_auc:.4f}")

    # Get predictions for ALL data using the last model
    print("\n" + "="*60)
    print("Generating predictions for entire dataset...")
    print("="*60)

    # Re-initialize model memory and use full neighbor finder
    model.memory.__init_memory__()
    model.set_neighbor_finder(full_ngh_finder)

    # Get predictions for full data
    pred_scores = np.zeros(len(full_data.sources))
    num_instance = len(full_data.sources)
    num_batch = math.ceil(num_instance / config['bs'])

    with torch.no_grad():
        model.eval()
        for k in tqdm(range(num_batch), desc="Predicting"):
            s_idx = k * config['bs']
            e_idx = min(num_instance, s_idx + config['bs'])

            sources_batch = torch.from_numpy(full_data.sources[s_idx:e_idx]).long().to(device)
            destinations_batch = torch.from_numpy(full_data.destinations[s_idx:e_idx]).long().to(device)
            timestamps_batch = torch.from_numpy(full_data.timestamps[s_idx:e_idx]).float().to(device)

            # Get neighbors
            src_neighbors_np, _, src_neighbors_time_np = full_ngh_finder.get_temporal_neighbor(
                full_data.sources[s_idx:e_idx], full_data.timestamps[s_idx:e_idx], config['n_degree']
            )
            dst_neighbors_np, _, dst_neighbors_time_np = full_ngh_finder.get_temporal_neighbor(
                full_data.destinations[s_idx:e_idx], full_data.timestamps[s_idx:e_idx], config['n_degree']
            )

            src_neighbors_batch = torch.from_numpy(src_neighbors_np).long().to(device)
            dst_neighbors_batch = torch.from_numpy(dst_neighbors_np).long().to(device)
            src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_np).long().to(device)
            dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_np).long().to(device)

            positive_memory_score, drift_score, _, _ = model.compute_anomaly_score(
                sources_batch, destinations_batch, timestamps_batch,
                src_neighbors_batch, dst_neighbors_batch,
                src_neighbors_time_batch, dst_neighbors_time_batch,
                config['n_degree']
            )

            # Compute final score based on configuration
            if config['only_drift_loss_score'] or config['only_drift_score']:
                batch_scores = (-(drift_score).reshape(-1).cpu().numpy() + 1) / 2
            elif config['only_recovery_loss_score'] or config['only_rec_score']:
                batch_scores = (-(positive_memory_score).reshape(-1).cpu().numpy() + 1) / 2
            else:
                batch_scores = (-(drift_score).reshape(-1).cpu().numpy() - (positive_memory_score).reshape(-1).cpu().numpy() + 2) / 4

            pred_scores[s_idx:e_idx] = batch_scores

    # Prepare edge list and timestamps
    edge_list = [[int(s), int(d)] for s, d in zip(full_data.sources, full_data.destinations)]
    timestamps_list = full_data.timestamps.tolist()
    ground_truth = full_data.labels.tolist()

    print(f"\nTotal samples: {len(pred_scores)}")
    print(f"Score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")

    # Calculate final metrics
    from sklearn.metrics import roc_auc_score
    final_auc_all = roc_auc_score(ground_truth, pred_scores) if len(np.unique(ground_truth)) > 1 else 0.0
    print(f"Final AUC (all data): {final_auc_all:.4f}")

    # Save results using EDGE_STREAM format
    print("\nSaving results in EDGE_STREAM_ANOMALY_SCORES format...")
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=pred_scores.tolist(),
        edges=edge_list,
        timestamps=timestamps_list,
        ground_truth=ground_truth,
    )

    # Calculate execution time and resource usage
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    # Final memory check
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    if torch.cuda.is_available():
        final_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if peak_gpu_memory_mb is not None:
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, final_gpu_mb)

    print(f"\nResource Usage:")
    print(f"  Total execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")
    if peak_gpu_memory_mb is not None:
        print(f"  Peak GPU memory: {peak_gpu_memory_mb:.2f}MB")

    # Compute statistics across runs
    mean_auc = np.mean(all_test_aucs)
    std_auc = np.std(all_test_aucs) if len(all_test_aucs) > 1 else 0.0

    # Add final metadata
    writer.add_metadata(
        exp_name=os.path.basename(os.environ.get("EXP", "experiment")),
        method_name="slade",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning",
            "task": "edge_stream_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "total_samples": len(pred_scores),
                "n_anomalies": int(np.sum(ground_truth)),
                "anomaly_ratio": float(np.mean(ground_truth)),
                "n_unique_nodes": full_data.n_unique_nodes,
            },
            "training_info": {
                "n_runs": config['n_runs'],
                "mean_test_auc": float(mean_auc),
                "std_test_auc": float(std_auc),
                "final_auc_all_data": float(final_auc_all),
            },
            "model_architecture": {
                "type": "SLADE_TGN",
                "message_dim": config['message_dim'],
                "memory_dim": config['memory_dim'],
                "n_heads": config['n_head'],
                "n_degree": config['n_degree'],
                "dropout": config['drop_out'],
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
        peak_gpu_mb=peak_gpu_memory_mb,
    )

    # Finalize and write results
    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"Mean Test AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Final AUC (all data): {final_auc_all:.4f}")


if __name__ == "__main__":
    main()
