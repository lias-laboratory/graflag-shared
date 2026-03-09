"""
GraFlag-integrated training script for GeneralDyG.
This wrapper runs the full dataset (train+test) to get anomaly scores for all nodes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
import datasets as dataset
import torch.utils.data
from sklearn.metrics import roc_auc_score
import numpy as np
import psutil

from model.CensNet import CensNet
from model.Combine import CombinedModel
from model.Transformer import TransformerBinaryClassifier
from option import args
from utils import EarlyStopMonitor
import random

# GraFlag integration
from graflag_runner import ResultWriter

# Import patched dataset for loading all data
from dataset_all import DygDatasetAll

from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def criterion(logits, labels):
    loss_classify = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss_classify = torch.mean(loss_classify)
    return loss_classify


def eval_epoch(data_loader, model, device):
    """Evaluate and return predictions and labels."""
    m_loss, m_pred, m_label = np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        model.eval()
        for batch_sample in data_loader:
            input_nodes_feature = batch_sample["input_nodes_feature"]
            input_edges_feature = batch_sample["input_edges_feature"]
            input_edges_pad = batch_sample["input_edges_pad"]
            labels = batch_sample["labels"]
            Tmats = batch_sample["Tmats"]
            adjs = batch_sample["adjs"]
            eadjs = batch_sample["eadjs"]
            mask_edge = batch_sample["mask_edge"]

            input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
            input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
            Tmats = [tensor.to(device) for tensor in Tmats]
            adjs = [tensor.to(device) for tensor in adjs]
            eadjs = [tensor.to(device) for tensor in eadjs]

            logits = model(
                input_nodes_feature,
                input_edges_feature,
                input_edges_pad.to(device),
                eadjs,
                adjs,
                Tmats,
                mask_edge.to(device),
            )
            y = labels.to(device)
            y = y.to(torch.float32)

            c_loss = np.array([criterion(logits, y).cpu()])
            pred_score = logits.cpu().numpy().flatten()
            y = y.cpu().numpy().flatten()
            m_loss = np.concatenate((m_loss, c_loss))
            m_pred = np.concatenate((m_pred, pred_score))
            m_label = np.concatenate((m_label, y))

        auc_roc = roc_auc_score(m_label, m_pred) if len(np.unique(m_label)) > 1 else 0.0
    return np.mean(m_loss), auc_roc, m_pred, m_label


def get_all_predictions(data_loader, model, device):
    """Get predictions for entire dataset."""
    all_preds = []
    all_labels = []
    all_node_ids = []

    with torch.no_grad():
        model.eval()
        for batch_sample in data_loader:
            input_nodes_feature = batch_sample["input_nodes_feature"]
            input_edges_feature = batch_sample["input_edges_feature"]
            input_edges_pad = batch_sample["input_edges_pad"]
            labels = batch_sample["labels"]
            Tmats = batch_sample["Tmats"]
            adjs = batch_sample["adjs"]
            eadjs = batch_sample["eadjs"]
            mask_edge = batch_sample["mask_edge"]

            input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
            input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
            Tmats = [tensor.to(device) for tensor in Tmats]
            adjs = [tensor.to(device) for tensor in adjs]
            eadjs = [tensor.to(device) for tensor in eadjs]

            logits = model(
                input_nodes_feature,
                input_edges_feature,
                input_edges_pad.to(device),
                eadjs,
                adjs,
                Tmats,
                mask_edge.to(device),
            )

            pred_score = logits.cpu().numpy().flatten()
            y = labels.cpu().numpy().flatten()

            all_preds.extend(pred_score.tolist())
            all_labels.extend(y.tolist())

    return np.array(all_preds), np.array(all_labels)


def main():
    config = args

    config.dir_data = os.environ.get("DATA")
    skip_pkl = False

    dir_data = Path(config.dir_data)
    if dir_data.name == "generaldyg_btc_alpha":
        config.data_set = "btc_alpha"
    elif dir_data.name == "generaldyg_btc_otc":
        config.data_set = "btc_otc"

    pkl_path = dir_data / f"{config.data_set}.pkl"
    if pkl_path.exists():
        skip_pkl = True
    

    if not skip_pkl:
        import subprocess
        subprocess.run(
            [
                "python3",
                "src/generate_datasets.py",
                "--dir_data",
                str(config.dir_data),
                "--data_set",
                config.data_set,
                "--neg",
                str(config.neg),
            ],
            check=True,
        )

    set_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start tracking execution time
    start_time = time.time()

    # Initialize resource tracking
    process = psutil.Process()
    peak_memory_mb = 0.0
    peak_gpu_memory_mb = None

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Add metadata
    writer.add_metadata(
        method_name="generaldyg",
        dataset=config.data_set,
        seed=config.seed,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layer=config.n_layer,
        drop_out=config.drop_out,
    )

    # Load datasets
    print("Loading datasets...")
    dataset_train = dataset.DygDataset(config, "train")
    dataset_test = dataset.DygDataset(config, "test")

    collate_fn = dataset.Collate(config)

    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_data_workers,
        pin_memory=True,
        collate_fn=collate_fn.dyg_collate_fn,
    )

    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        shuffle=False,  # Don't shuffle for final predictions
        num_workers=config.num_data_workers,
        collate_fn=collate_fn.dyg_collate_fn,
    )

    # Build model
    print("Building model...")
    GNN = CensNet(config.input_dim, config.drop_out)
    transformer = TransformerBinaryClassifier(
        config, device, hidden_size=config.hidden_dim
    )
    backbone = CombinedModel(GNN, transformer)
    model = backbone.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("Starting training...")
    max_test_auc = 0.0
    best_model_state = None
    early_stopper = EarlyStopMonitor(higher_better=True)

    for epoch in range(config.n_epochs):
        # Track memory usage
        current_memory_mb = process.memory_info().rss / (1024 * 1024)
        peak_memory_mb = max(peak_memory_mb, current_memory_mb)

        # Track GPU memory if available
        if torch.cuda.is_available():
            current_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (
                1024 * 1024
            )
            if peak_gpu_memory_mb is None:
                peak_gpu_memory_mb = current_gpu_memory_mb
            else:
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, current_gpu_memory_mb)

        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, batch_sample in enumerate(loader_train):
            input_nodes_feature = batch_sample["input_nodes_feature"]
            input_edges_feature = batch_sample["input_edges_feature"]
            input_edges_pad = batch_sample["input_edges_pad"]
            labels = batch_sample["labels"]
            Tmats = batch_sample["Tmats"]
            adjs = batch_sample["adjs"]
            eadjs = batch_sample["eadjs"]
            mask_edge = batch_sample["mask_edge"]

            input_nodes_feature = [tensor.to(device) for tensor in input_nodes_feature]
            input_edges_feature = [tensor.to(device) for tensor in input_edges_feature]
            Tmats = [tensor.to(device) for tensor in Tmats]
            adjs = [tensor.to(device) for tensor in adjs]
            eadjs = [tensor.to(device) for tensor in eadjs]

            optimizer.zero_grad()
            logits = model(
                input_nodes_feature,
                input_edges_feature,
                input_edges_pad.to(device),
                eadjs,
                adjs,
                Tmats,
                mask_edge.to(device),
            )
            y = labels.to(device).to(torch.float32)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        # Validation
        test_loss, test_auc, _, _ = eval_epoch(loader_test, model, device)

        print(
            f"Epoch {epoch+1}/{config.n_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {test_loss:.4f}, Val AUC: {test_auc:.4f}"
        )

        # Track metrics with spot()
        writer.spot(
            "training",
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=test_loss,
            val_auc=test_auc,
        )

        # Save best model
        if test_auc > max_test_auc:
            max_test_auc = test_auc
            best_model_state = model.state_dict().copy()
            print(f"  [OK] New best AUC: {max_test_auc:.4f}")
            
        # Early stopping
        if early_stopper.early_stop_check(test_auc):
            print(
                f"Early stopping after {epoch+1} epochs (no improvement over {early_stopper.max_round} rounds)"
            )
            break

    print(f"\nTraining completed! Best validation AUC: {max_test_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model for final predictions")

    # Get predictions for ALL data (entire dataset without split)
    print("\nGenerating predictions for entire dataset...")

    # Load the original CSV to get edge pairs and timestamps
    import pandas as pd
    csv_path = dir_data / f"{config.data_set}_0.5_0.{config.neg}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    graph_df = pd.read_csv(csv_path)
    # Extract edge information: u (source), i (destination), id (timestamp/edge_id)
    edge_list = graph_df[['u', 'i']].values.tolist()  # [[src, dst], ...]
    timestamps = graph_df['id'].values.tolist()  # Edge IDs serve as timestamps
    
    print(f"Loaded {len(edge_list)} edges from CSV")

    # Create dataset with all data (no train/test split)
    dataset_all = DygDatasetAll(config)
    collate_fn_all = dataset.Collate(config)

    loader_all = torch.utils.data.DataLoader(
        dataset=dataset_all,
        batch_size=config.batch_size,
        shuffle=False,  # Keep original order
        num_workers=config.num_data_workers,
        collate_fn=collate_fn_all.dyg_collate_fn,
    )

    # Get predictions for entire dataset
    all_scores, all_labels = get_all_predictions(loader_all, model, device)

    print(f"Total samples: {len(all_scores)}")
    print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")

    # Calculate final AUC
    final_auc = (
        roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    )
    print(f"Final AUC (all data): {final_auc:.4f}")

    print("[OK] Using EDGE_STREAM format (1D) - most memory efficient")

    # Save results using GraFlag format
    # GeneralDyG is a temporal edge anomaly detection method for streaming edges
    # Format: 1D arrays where each index represents one edge occurrence
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=all_scores.tolist(),  # 1D array of scores
        edges=edge_list,  # [[src, dst], ...] - one per score
        timestamps=timestamps,  # [t0, t1, ...] - one per score
        ground_truth=all_labels.tolist(),
    )

    # Calculate total execution time
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    # Final memory check
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    # Final GPU memory check
    if torch.cuda.is_available():
        final_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if peak_gpu_memory_mb is None:
            peak_gpu_memory_mb = final_gpu_memory_mb
        else:
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, final_gpu_memory_mb)

    print(f"\n[INFO] Resource Usage:")
    print(f"   [INFO] Total execution time: {exec_time_seconds:.2f}s")
    print(f"   [INFO] Peak memory: {peak_memory_mb:.2f}MB")
    if peak_gpu_memory_mb is not None:
        print(f"   [INFO] Peak GPU memory: {peak_gpu_memory_mb:.2f}MB")

    # Extract all config parameters dynamically for method_parameters
    method_parameters = {}
    for attr_name in dir(config):
        if not attr_name.startswith("_"):  # Skip private attributes
            attr_value = getattr(config, attr_name)
            # Only include simple types (not methods or complex objects)
            if isinstance(attr_value, (int, float, str, bool, type(None))):
                method_parameters[attr_name] = attr_value

    # Add final metrics to metadata (following result_types.md specification)
    writer.add_metadata(
        exp_name=os.path.basename(os.environ.get("EXP", "experiment")),
        method_name="generaldyg",
        dataset=config.data_set,
        method_parameters=method_parameters,
        threshold=None,  # No explicit threshold used
        summary={
            "description": "A Generalizable Anomaly Detection Method in Dynamic Graphs (AAAI 2025)",
            "task": "temporal_edge_anomaly_detection",
            "dataset_info": {
                "name": config.data_set,
                "total_samples": len(all_scores),
                "n_anomalies": int(np.sum(all_labels)),
                "anomaly_ratio": float(np.sum(all_labels) / len(all_labels)),
            },
            "training_info": {
                "total_epochs": epoch + 1,
                "early_stopped": epoch + 1 < config.n_epochs,
                "best_val_auc": float(max_test_auc),
                "final_auc_all_data": float(final_auc),
            },
            "model_architecture": {
                "gnn": "CensNet",
                "temporal": "Transformer",
                "input_dim": config.input_dim,
                "hidden_dim": config.hidden_dim,
                "n_heads": config.n_heads,
                "n_layers": config.n_layer,
                "dropout": config.drop_out,
            },
        },
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
        peak_gpu_mb=peak_gpu_memory_mb,
    )

    # Finalize and write results.json
    results_file = writer.finalize()
    print(f"\n[OK] Results saved to: {results_file}")
    print(f"Best validation AUC: {max_test_auc:.4f}")
    print(f"Final AUC (all data): {final_auc:.4f}")


if __name__ == "__main__":
    main()
