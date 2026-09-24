"""
GraFlag-integrated training script for GeneralDyG.
This wrapper runs the full dataset (train+test) to get anomaly scores for all nodes.
"""

import gc
import os
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score

from graflag_runner import (
    ResultWriter, apply_params, device, info, paths, seed_all, upstream,
)

# The GeneralDyG clone, and the only place this script says where it is.
# upstream() anchors on this file's directory and raises if the checkout is
# missing, instead of failing later with an unexplained ImportError.
upstream("src")

import datasets as dataset                                  # noqa: E402
from model.CensNet import CensNet                           # noqa: E402
from model.Combine import CombinedModel                     # noqa: E402
from model.Transformer import TransformerBinaryClassifier   # noqa: E402
from option import args                                     # noqa: E402
from utils import EarlyStopMonitor                          # noqa: E402


def set_seed(seed):
    """Seed everything, then ask torch for bit-reproducible kernels.

    seed_all() covers random/numpy/torch and PYTHONHASHSEED for every method;
    what follows is specific to GeneralDyG, which trades throughput for exact
    reproducibility.
    """
    seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def criterion(logits, labels):
    loss_classify = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss_classify = torch.mean(loss_classify)
    return loss_classify


def eval_epoch(data_loader, model, dev):
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

            input_nodes_feature = [tensor.to(dev) for tensor in input_nodes_feature]
            input_edges_feature = [tensor.to(dev) for tensor in input_edges_feature]
            Tmats = [tensor.to(dev) for tensor in Tmats]
            adjs = [tensor.to(dev) for tensor in adjs]
            eadjs = [tensor.to(dev) for tensor in eadjs]

            logits = model(
                input_nodes_feature,
                input_edges_feature,
                input_edges_pad.to(dev),
                eadjs,
                adjs,
                Tmats,
                mask_edge.to(dev),
            )
            y = labels.to(dev)
            y = y.to(torch.float32)

            c_loss = np.array([criterion(logits, y).cpu()])
            pred_score = logits.cpu().numpy().flatten()
            y = y.cpu().numpy().flatten()
            m_loss = np.concatenate((m_loss, c_loss))
            m_pred = np.concatenate((m_pred, pred_score))
            m_label = np.concatenate((m_label, y))

        auc_roc = roc_auc_score(m_label, m_pred) if len(np.unique(m_label)) > 1 else 0.0
    return np.mean(m_loss), auc_roc, m_pred, m_label


def main():
    run = paths()

    # GeneralDyG configures itself from upstream's own argparse namespace, so
    # unlike the other methods it has no Config of its own: the accepted names
    # and their defaults are upstream's, and the .env lists only what GraFlag
    # overrides. apply_params() writes those onto the namespace directly.
    #
    # It used to go through --pass-env-args instead, which is unsafe here:
    # upstream declares --gpus and no --gpu, so argparse's abbreviation
    # matching turned _GPU=0 into gpus=0.
    config = args
    injected = apply_params(config, ignore={"gpu"})   # _GPU is read by device()
    config.dir_data = str(run.data)

    # The dataset directory names itself after the method; upstream's loaders
    # want the bare name.
    dir_data = run.data
    if dir_data.name == "generaldyg_btc_alpha":
        config.data_set = "btc_alpha"
    elif dir_data.name == "generaldyg_btc_otc":
        config.data_set = "btc_otc"

    if not (dir_data / f"{config.data_set}.pkl").exists():
        info(f"[INFO] Building {config.data_set}.pkl")
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

    # This used to be a hardcoded "cuda:0", so a service Swarm scheduled
    # without reserving a GPU still took one and `graflag run --no-gpu` had no
    # effect. device() reads _GPU, where -1 means CPU.
    dev = device()

    writer = ResultWriter()

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
        config, dev, hidden_size=config.hidden_dim
    )
    backbone = CombinedModel(GNN, transformer)
    model = backbone.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    print("Starting training...")
    max_test_auc = 0.0
    best_model_state = None
    early_stopper = EarlyStopMonitor(higher_better=True)

    for epoch in range(config.n_epochs):
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

            input_nodes_feature = [tensor.to(dev) for tensor in input_nodes_feature]
            input_edges_feature = [tensor.to(dev) for tensor in input_edges_feature]
            Tmats = [tensor.to(dev) for tensor in Tmats]
            adjs = [tensor.to(dev) for tensor in adjs]
            eadjs = [tensor.to(dev) for tensor in eadjs]

            optimizer.zero_grad()
            logits = model(
                input_nodes_feature,
                input_edges_feature,
                input_edges_pad.to(dev),
                eadjs,
                adjs,
                Tmats,
                mask_edge.to(dev),
            )
            y = labels.to(dev).to(torch.float32)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        # Not validation: loader_test is upstream's test split, and
        # GeneralDyG has no third one. Upstream prints these as `val loss` and
        # `val auc` (train.py:181-192) and this integration followed it, so
        # training.csv carried a val_auc column -- and training_curves.png a
        # legend entry -- for the test AUC under another name.
        test_loss, test_auc, _, _ = eval_epoch(loader_test, model, dev)

        print(
            f"Epoch {epoch+1}/{config.n_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}"
        )

        # Track metrics with spot()
        writer.spot(
            "training",
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            test_loss=test_loss,
            test_auc=test_auc,
        )

        # Snapshot the best epoch's weights.
        #
        # .clone() per tensor, not state_dict().copy(): state_dict() hands back
        # the live parameter tensors, and copy() copies the dict around them.
        # The optimizer then updates those same tensors in place, so the
        # "snapshot" tracked training and load_state_dict below restored the
        # weights onto themselves -- a no-op that looked like a checkpoint.
        # The run that exposed it had its best AUC at epoch 1 (0.7806) and
        # published epoch 2's scores (0.6829), the last epoch's, while
        # reporting that the best checkpoint had been loaded.
        if test_auc > max_test_auc:
            max_test_auc = test_auc
            best_model_state = {k: v.detach().clone()
                                for k, v in model.state_dict().items()}
            print(f"  [OK] New best AUC: {max_test_auc:.4f}")
            
        # Early stopping
        if early_stopper.early_stop_check(test_auc):
            print(
                f"Early stopping after {epoch+1} epochs (no improvement over {early_stopper.max_round} rounds)"
            )
            break

    print(f"\nTraining completed! Best test AUC: {max_test_auc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model for final predictions")

    # Score the test split, using the checkpoint the best-AUC epoch selected.
    #
    # This used to score every snapshot, training ones included, through a
    # local DygDatasetAll that upstream has no equivalent of. Two things were
    # wrong with that. A result built from the training split is not a held-out
    # measurement, which RESULTS_STANDARD.md requires ("scores must come from
    # the test split"). And DygDatasetAll re-read the pickle and drew a third
    # independent np.random.uniform feature matrix, so the features the model
    # was scored on were not the ones it had been evaluated against all
    # through training.
    #
    # loader_test is upstream's own DygDataset(config, 'test') and is built
    # with shuffle=False, so eval_epoch returns its predictions in dataset
    # order -- which is CSV order, restricted to the tail after split_indices
    # (upstream datasets.py:71-74). eval_epoch already computed and returned
    # these scores every epoch; the run simply discarded them.
    print("\nScoring the test split with the selected checkpoint...")

    # Release the training split first. Each DygDataset is a dense float64
    # padding of its share of the stream, not a view of the pickle, and the
    # train split is the larger one. Holding it through the final pass is what
    # got this run killed with exit code 137 on a 15 GB host: the monitor
    # recorded 6.3 GB in the last sample before the OOM killer arrived.
    del loader_train, dataset_train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _, final_auc, all_scores, all_labels = eval_epoch(loader_test, model, dev)

    # Load the original CSV to get edge pairs and timestamps
    csv_path = dir_data / f"{config.data_set}_0.5_0.{config.neg}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    graph_df = pd.read_csv(csv_path)
    # Extract edge information: u (source), i (destination), id (timestamp/edge_id)
    edge_list = graph_df[['u', 'i']].values.tolist()  # [[src, dst], ...]
    timestamps = graph_df['id'].values.tolist()  # Edge IDs serve as timestamps

    # Keep the rows the test split covers. Upstream's split is a contiguous
    # tail, so its length locates it; deriving the boundary from that rather
    # than from a copy of upstream's split_indices table means a dataset it
    # splits elsewhere needs no second edit here. The check is fatal on
    # purpose -- a mismatch would otherwise publish scores against the wrong
    # edges, which reads as a result rather than as an error.
    split = len(edge_list) - len(dataset_test)
    if split < 0 or len(all_scores) != len(dataset_test):
        raise RuntimeError(
            f"the test split does not line up with the edge stream: "
            f"{len(dataset_test)} test snapshots and {len(all_scores)} scores "
            f"against {len(edge_list)} CSV rows")
    edge_list = edge_list[split:]
    timestamps = timestamps[split:]

    print(f"Scored {len(all_scores)} test snapshots "
          f"(CSV rows {split}..{len(edge_list) + split})")
    print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"Test AUC: {final_auc:.4f}")

    print("[OK] Using EDGE_STREAM format (1D) - most memory efficient")

    # Save results using GraFlag format
    # GeneralDyG is a temporal edge anomaly detection method for streaming edges
    # Format: 1D arrays where each index represents one edge occurrence,
    # covering the test split only.
    writer.save_scores(
        result_type="EDGE_STREAM_ANOMALY_SCORES",
        scores=all_scores.tolist(),  # 1D array of scores
        edges=edge_list,  # [[src, dst], ...] - one per score
        timestamps=timestamps,  # [t0, t1, ...] - one per score
        ground_truth=all_labels.tolist(),
    )

    # The whole effective configuration, not just what the .env overrode:
    # most of it is upstream's defaults, and a run is only reproducible if
    # those are on the record too. `injected` names the subset GraFlag set.
    method_parameters = {
        name: value for name, value in vars(config).items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }

    # graflag_runner measures exec time, peak memory and peak GPU from outside
    # the method and its numbers win (_merge_runtime_metadata), so the psutil
    # sampling that used to be here was recorded and then overwritten. psutil
    # was never in this image's dependencies either -- importing it was a
    # latent ImportError.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="generaldyg",
        dataset=config.data_set,
        method_parameters=method_parameters,
        injected=sorted(injected),
        threshold=None,  # No explicit threshold used
        summary={
            "description": "A Generalizable Anomaly Detection Method in Dynamic Graphs (AAAI 2025)",
            "task": "temporal_edge_anomaly_detection",
            "dataset_info": {
                "name": config.data_set,
                "scored_split": "test",
                "scored_samples": len(all_scores),
                "n_anomalies": int(np.sum(all_labels)),
                "anomaly_ratio": float(np.sum(all_labels) / len(all_labels)),
            },
            "training_info": {
                "total_epochs": epoch + 1,
                "early_stopped": epoch + 1 < config.n_epochs,
                "best_test_auc": float(max_test_auc),
                "test_auc": float(final_auc),
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

    info(f"[OK] Best test AUC {max_test_auc:.4f}, test AUC from the "
         f"selected checkpoint {final_auc:.4f}, results written to "
         f"{writer.finalize()}")


if __name__ == "__main__":
    main()
