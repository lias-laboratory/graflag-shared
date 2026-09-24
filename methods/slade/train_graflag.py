"""
GraFlag-integrated training script for SLADE.
SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning
"""

import math
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from graflag_runner import (
    ResultWriter, device, info, params, paths, seed_all, upstream,
)

# The SLADE clone. upstream() anchors on this file's directory and raises if
# the checkout is missing, instead of failing later with an unexplained
# ImportError.
upstream("src")

from model.SLADE_TGN import SLADE_TGN                            # noqa: E402
from utils.utils import get_neighbor_finder                      # noqa: E402
from utils.data_processing import Data                           # noqa: E402
from evaluation.evaluation import eval_anomaly_node_detection    # noqa: E402


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

    # test_mask is returned, not recomputed by the caller: it is what says
    # which rows of full_data the published scores may cover, and a second
    # copy of `timestamps > np.quantile(...)` elsewhere is a boundary that can
    # drift from this one without anything failing.
    return full_data, train_data, test_data, graph_df, test_mask


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    Replaces an argparse parser plus a get_config_from_args() that restated
    every key a third time. The four score-selection switches are the only
    values that are not read as declared -- .env can only say 0 or 1, and the
    model wants booleans, so __post_init__ does that one translation.
    """

    bs: int = 100                              # batch size
    n_degree: int = 20                         # neighbours sampled per node
    n_head: int = 2
    n_epoch: int = 10
    lr: float = 3e-6
    n_runs: int = 1
    seed: int = 0
    drop_out: float = 0.1
    gpu: int = 0                               # GPU index; -1 means CPU
    message_dim: int = 128
    memory_dim: int = 256
    agg_type: str = "TGAT"
    negative_memory_type: str = "train"
    message_updater: str = "mlp"
    memory_updater: str = "gru"
    training_ratio: float = 0.85
    lr_decay: float = 0.8
    weight_decay: float = 0.0001
    srf: float = 0.1                           # source recovery factor
    drf: float = 0.1                           # drift recovery factor
    only_drift_loss_score: int = 0             # 1 to enable
    only_recovery_loss_score: int = 0          # 1 to enable
    only_drift_score: int = 0                  # 1 to enable
    only_rec_score: int = 0                    # 1 to enable

    def __post_init__(self):
        for switch in ("only_drift_loss_score", "only_recovery_loss_score",
                       "only_drift_score", "only_rec_score"):
            setattr(self, switch, getattr(self, switch) == 1)


def main():
    run = paths()
    config = asdict(Config(**params(Config)))

    info(f"[INFO] SLADE on {run.dataset}: {config}")

    seed_all(config['seed'])
    dev = device(config['gpu'])

    writer = ResultWriter()

    # The dataset directory names itself after the method; report the bare name.
    dataset_name = run.dataset.replace('slade_', '')

    print("\nLoading data...")
    full_data, train_data, test_data, graph_df, test_mask = load_data(
        run.data, config['training_ratio'])

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
            device=dev,
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
        model = model.to(dev)

        # Prepare training data tensors
        train_data_sources = torch.from_numpy(train_data.sources).long().to(dev)
        train_data_destinations = torch.from_numpy(train_data.destinations).long().to(dev)
        train_data_timestamps = torch.from_numpy(train_data.timestamps).float().to(dev)
        train_data_src_neighbors = torch.from_numpy(src_neighbors).long().to(dev)
        train_data_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(dev)
        train_data_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(dev)
        train_data_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(dev)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / config['bs'])

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])

        negative_train_nodes = torch.from_numpy(
            np.array(list(set(train_data.destinations) | set(train_data.sources)))
        ).long().to(dev)

        # Named for the split it is measured on. SLADE has two splits, not
        # three: eval_anomaly_node_detection below is handed test_data, so a
        # column called val_auc in training.csv -- and in the legend of
        # training_curves.png -- would be the test AUC under another name.
        best_test_auc = 0.0
        test_aucs = []

        for epoch in range(config['n_epoch']):
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

            test_auc, pred_score, _ = eval_anomaly_node_detection(
                model, test_data, config['bs'],
                n_neighbors=config['n_degree'],
                device=dev,
                only_rec_score=config['only_rec_score'] or config['only_recovery_loss_score'],
                only_drift_score=config['only_drift_loss_score'] or config['only_drift_score']
            )

            avg_loss = sum(m_loss) / len(m_loss) if m_loss else 0.0
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Test AUC: {test_auc:.4f}")

            test_aucs.append(test_auc)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
                print(f"  -> New best AUC: {best_test_auc:.4f}")

            # Track training metrics
            writer.spot(
                "training",
                epoch=epoch + 1,
                run=run_idx + 1,
                loss=avg_loss,
                test_auc=test_auc,
            )

        # Final evaluation on test data
        # The last epoch's, not the best: no checkpoint is restored, so the
        # model that produces the published scores is the one training ended
        # on. best_test_auc is reported, never selected on.
        final_auc = test_aucs[-1] if test_aucs else 0.0
        all_test_aucs.append(final_auc)

        print(f"\nRun {run_idx + 1} - Final Test AUC: {final_auc:.4f}")

    # Replay the whole stream with the last model, then publish the test
    # tail alone.
    #
    # The replay has to cover every edge: SLADE is a streaming method whose
    # memory is updated by each interaction it sees, so a test edge scored
    # without the training prefix ahead of it would be scored against an empty
    # memory. That part is deliberate, and it is the same reason gady swaps in
    # full_ngh_finder for evaluation.
    #
    # What was wrong was publishing the result of the whole replay. The
    # training edges are the ones the model was fitted on, so scores over them
    # are not a held-out measurement, and RESULTS_STANDARD.md asks for the test
    # split. With training_ratio=0.8 that made four fifths of every published
    # score and of every AUC computed from it in-sample.
    print("\n" + "="*60)
    print("Generating predictions for the full stream (scoring the test split)...")
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

            sources_batch = torch.from_numpy(full_data.sources[s_idx:e_idx]).long().to(dev)
            destinations_batch = torch.from_numpy(full_data.destinations[s_idx:e_idx]).long().to(dev)
            timestamps_batch = torch.from_numpy(full_data.timestamps[s_idx:e_idx]).float().to(dev)

            # Get neighbors
            src_neighbors_np, _, src_neighbors_time_np = full_ngh_finder.get_temporal_neighbor(
                full_data.sources[s_idx:e_idx], full_data.timestamps[s_idx:e_idx], config['n_degree']
            )
            dst_neighbors_np, _, dst_neighbors_time_np = full_ngh_finder.get_temporal_neighbor(
                full_data.destinations[s_idx:e_idx], full_data.timestamps[s_idx:e_idx], config['n_degree']
            )

            src_neighbors_batch = torch.from_numpy(src_neighbors_np).long().to(dev)
            dst_neighbors_batch = torch.from_numpy(dst_neighbors_np).long().to(dev)
            src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_np).long().to(dev)
            dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_np).long().to(dev)

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

    # Keep the test split. test_mask indexes full_data row for row, so this
    # selects the same edges test_data holds however the stream is ordered --
    # it does not assume the split is a contiguous tail the way slicing would.
    if len(test_mask) != len(pred_scores):
        raise RuntimeError(
            f"the split mask ({len(test_mask)}) does not cover the scored "
            f"stream ({len(pred_scores)})")
    pred_scores = pred_scores[test_mask]
    edge_list = [[int(s), int(d)] for s, d in
                 zip(full_data.sources[test_mask], full_data.destinations[test_mask])]
    timestamps_list = full_data.timestamps[test_mask].tolist()
    ground_truth = full_data.labels[test_mask].tolist()

    print(f"\nTest samples: {len(pred_scores)} of {len(test_mask)} in the stream")
    print(f"Score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")

    # Calculate final metrics
    from sklearn.metrics import roc_auc_score
    final_test_auc = roc_auc_score(ground_truth, pred_scores) if len(np.unique(ground_truth)) > 1 else 0.0
    print(f"Final AUC (test split): {final_test_auc:.4f}")

    # Save results using EDGE_STREAM format
    print("\nSaving results in EDGE_STREAM_ANOMALY_SCORES format...")
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=pred_scores.tolist(),
        edges=edge_list,
        timestamps=timestamps_list,
        ground_truth=ground_truth,
    )

    mean_auc = np.mean(all_test_aucs)
    std_auc = np.std(all_test_aucs) if len(all_test_aucs) > 1 else 0.0

    # Execution time and memory are measured by graflag_runner around this
    # process and merged into metadata afterwards -- see the runner's
    # _merge_runtime_metadata.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="slade",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via Self-Supervised Learning",
            "task": "edge_stream_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "scored_split": "test",
                "scored_samples": len(pred_scores),
                "n_anomalies": int(np.sum(ground_truth)),
                "anomaly_ratio": float(np.mean(ground_truth)),
                "n_unique_nodes": full_data.n_unique_nodes,
            },
            "training_info": {
                "n_runs": config['n_runs'],
                "mean_test_auc": float(mean_auc),
                "std_test_auc": float(std_auc),
                "final_test_auc": float(final_test_auc),
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

    info(f"[OK] Mean test AUC {mean_auc:.4f} +/- {std_auc:.4f}, AUC over the "
         f"published test split {final_test_auc:.4f}, results written to "
         f"{writer.finalize()}")


if __name__ == "__main__":
    main()
