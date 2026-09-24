"""
GraFlag-integrated training script for TADDY.
TADDY: Anomaly Detection in Dynamic Graphs via Transformer (TKDE 2021)

TADDY scores EDGES per snapshot of a dynamic graph with a transformer over
spatial-temporal node encodings.
"""

import sys
import os
import time
import shutil
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from graflag_runner import (
    ResultWriter, device, info, params, paths, seed_all, upstream,
)

# The TADDY clone. upstream() anchors on this file's directory and raises if
# the checkout is missing, instead of failing later with an unexplained
# ImportError.
upstream("src")

from codes.DynamicDatasetLoader import DynamicDatasetLoader     # noqa: E402
from codes.Component import MyConfig                            # noqa: E402
from codes.DynADModel import DynADModel                         # noqa: E402
from codes.Settings import Settings                             # noqa: E402
from codes.AnomalyGeneration import anomaly_generation          # noqa: E402
from scipy import sparse                                        # noqa: E402


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    Replaces an argparse parser that ran at import time, so importing this
    module no longer depends on argv. The instance is also what upstream's
    DynADModel is handed as its ``args``: it reads ``print_freq`` from it (its
    own ``train_model`` reads ``print_feq``, but this file overrides that
    method), and nothing else.
    """

    anomaly_per: float = 0.1
    train_per: float = 0.5
    neighbor_num: int = 5
    window_size: int = 2
    embedding_dim: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    max_epoch: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    seed: int = 1
    print_freq: int = 10
    gpu: int = 0                # GPU index; -1 means CPU


def read_edge_list(path):
    """Whitespace-separated `src dst ...` with `%` comments (uci, digg)."""
    edges = np.loadtxt(path, dtype=float, comments='%', delimiter=' ')
    return edges[:, 0:2].astype(int)


def read_rating_csv(path):
    """`src,dst,rating,timestamp` (the two bitcoin sets), in timestamp order."""
    with open(path) as f:
        rows = [[float(value) for value in line.split(',')]
                for line in f.read().splitlines()]
    edges = np.array(rows)
    return edges[edges[:, 3].argsort()][:, 0:2].astype(int)


@dataclass(frozen=True)
class Dataset:
    """Everything that varies per dataset, in one place.

    Adding a dataset is a row here: previously the same four datasets were
    spelled out three times -- an alias chain, a filename chain in
    setup_data_directories and a snap_size dict in preprocess_data -- and they
    had to be kept in step by hand.

    `aliases` are matched as substrings against the mounted directory's name.
    `raw_file` is both the name under the mount and the name TADDY's
    preprocessing reads under `data/raw/`. `snap_size` is how many edges go
    into one snapshot.
    """

    name: str
    raw_file: str
    snap_size: int
    read: object
    aliases: tuple


DATASETS = (
    Dataset('uci', 'uci', 1000, read_edge_list, ('uci',)),
    Dataset('digg', 'digg', 6000, read_edge_list, ('digg',)),
    Dataset('btc_alpha', 'soc-sign-bitcoinalpha.csv', 1000, read_rating_csv,
            ('btc_alpha', 'bitcoinalpha')),
    Dataset('btc_otc', 'soc-sign-bitcoinotc.csv', 2000, read_rating_csv,
            ('btc_otc', 'bitcoinotc')),
)


# Custom model class that captures predictions
class DynADModelWithResults(DynADModel):
    """Extended TADDY model that captures and saves predictions."""
    
    def __init__(self, *args, result_writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_preds = None
        self.final_labels = None
        self.snap_ids = None
        self.result_writer = result_writer  # For spot() tracking
    
    def train_model(self, max_epoch):
        """
        Override train_model to capture final predictions.

        This method:
        1. Trains the model for max_epoch epochs
        2. Validates on test snapshots periodically (every print_feq epochs)
        3. After training, generates predictions on training snapshots for final results
        """
        import torch.optim as optim
        import torch.nn.functional as F
        import time

        # Detect device from model parameters
        device = next(self.parameters()).device

        # Initialize optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Generate embeddings for positive samples (created on CPU by dicts_to_embeddings)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

        # Setup negative sampling function
        ns_function = self.negative_sampling

        # Training loop
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # Generate negative samples and their embeddings
            negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
            raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
                time_embeddings_neg = self.generate_embedding(negatives)

            # Set model to training mode
            self.train()

            # Train on all snapshots
            loss_train = 0
            num_valid_snaps = 0

            for snap in self.data['snap_train']:
                # Skip snapshots without embeddings (due to window_size constraint)
                if wl_embeddings[snap] is None:
                    continue

                # Get positive sample embeddings and move to device
                int_embedding_pos = int_embeddings[snap].to(device)
                hop_embedding_pos = hop_embeddings[snap].to(device)
                time_embedding_pos = time_embeddings[snap].to(device)
                y_pos = self.data['y'][snap].float().to(device)

                # Get negative sample embeddings and move to device
                int_embedding_neg = int_embeddings_neg[snap].to(device)
                hop_embedding_neg = hop_embeddings_neg[snap].to(device)
                time_embedding_neg = time_embeddings_neg[snap].to(device)
                y_neg = torch.ones(int_embedding_neg.size()[0], device=device)

                # Combine positive and negative samples
                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                y = torch.hstack((y_pos, y_neg))

                # Forward pass and backpropagation
                optimizer.zero_grad()
                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.detach().item()
                num_valid_snaps += 1

            # Calculate average loss
            loss_train = loss_train / num_valid_snaps if num_valid_snaps > 0 else 0
            epoch_time = time.time() - t_epoch_begin

            # Log training progress
            print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(
                epoch + 1, loss_train, epoch_time
            ))

            # Track training metrics with spot()
            if self.result_writer:
                self.result_writer.spot(
                    "training",
                    epoch=epoch + 1,
                    loss=loss_train,
                    time_sec=epoch_time
                )

            # Periodic validation on test snapshots
            if ((epoch + 1) % self.args.print_freq) == 0:
                self._validate_on_test_snapshots(
                    epoch,
                    int_embeddings,
                    hop_embeddings,
                    time_embeddings
                )

        # After training completes, generate final predictions on TRAINING snapshots
        self._generate_final_predictions(
            wl_embeddings,
            int_embeddings,
            hop_embeddings,
            time_embeddings
        )
    
    def _validate_on_test_snapshots(self, epoch, int_embeddings, hop_embeddings, time_embeddings):
        """
        Validate model performance on test snapshots.

        Args:
            epoch: Current epoch number
            int_embeddings: Interaction embeddings
            hop_embeddings: Hop embeddings
            time_embeddings: Temporal embeddings
        """
        device = next(self.parameters()).device
        self.eval()

        # Generate predictions for test snapshots
        preds = []
        for snap in self.data['snap_test']:
            int_embedding = int_embeddings[snap].to(device)
            hop_embedding = hop_embeddings[snap].to(device)
            time_embedding = time_embeddings[snap].to(device)

            with torch.no_grad():
                output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                output = torch.sigmoid(output)
            pred = output.squeeze().cpu().numpy()
            preds.append(pred)

        # Get ground truth labels
        y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test']) + 1]
        y_test = [y_snap.numpy() for y_snap in y_test]

        # Evaluate performance
        aucs, auc_full = self.evaluate(y_test, preds)

        # Log results
        for i in range(len(self.data['snap_test'])):
            print("Snap: %02d | AUC: %.4f" % (self.data['snap_test'][i], aucs[i]))
        print('TOTAL AUC: {:.4f}'.format(auc_full))
        
        # Track validation metrics with spot()
        if self.result_writer:
            self.result_writer.spot("validation", epoch=epoch + 1, auc=auc_full)
    
    def _generate_final_predictions(self, wl_embeddings, int_embeddings, hop_embeddings, time_embeddings):
        """
        Generate final predictions on TEST snapshots for results.
        
        Note: We use test snapshots because they contain injected anomalies,
        allowing for meaningful AUC-ROC/PR metrics. Training snapshots have
        all-zero labels (no anomalies).
        
        Args:
            wl_embeddings: Weisfeiler-Lehman embeddings
            int_embeddings: Interaction embeddings
            hop_embeddings: Hop embeddings
            time_embeddings: Temporal embeddings
        """
        print('\n=== Generating Final Predictions on Test Snapshots ===')
        device = next(self.parameters()).device
        self.eval()

        preds_test = []
        labels_test = []
        valid_test_snaps = []

        for snap in self.data['snap_test']:
            int_embedding = int_embeddings[snap].to(device)
            hop_embedding = hop_embeddings[snap].to(device)
            time_embedding = time_embeddings[snap].to(device)

            with torch.no_grad():
                output = self.forward(int_embedding, hop_embedding, time_embedding, None)
                output = torch.sigmoid(output)
            pred = output.squeeze().cpu().numpy()
            preds_test.append(pred)

            # Get ground truth labels
            labels = self.data['y'][snap].numpy()
            labels_test.append(labels)
            
            valid_test_snaps.append(snap)

        # Store TEST predictions for final results (test data has anomaly labels)
        self.final_preds = preds_test
        self.final_labels = labels_test
        self.snap_ids = valid_test_snaps
        
        print(f'[OK] Generated predictions for {len(valid_test_snaps)} test snapshots')
        print(f'   Snapshot IDs: {valid_test_snaps}')
        
        # Print anomaly stats
        total_edges = sum(len(l) for l in labels_test)
        total_anomalies = sum(np.sum(l) for l in labels_test)
        print(f'   Total edges: {total_edges}, Anomalies: {int(total_anomalies)} ({100*total_anomalies/total_edges:.2f}%)')


def setup_data_directories(data_path: Path, dataset):
    """Put TADDY's expected `data/raw/<file>` in place, pointing at the mount.

    Args:
        data_path: the mounted dataset directory (e.g. /shared/datasets/taddy_uci)
        dataset: the Dataset resolved from it
    """
    print(f'\n=== Setting Up Data Directories ===')
    print(f'Source: {data_path}')
    print(f'Dataset: {dataset.name}')

    for directory in ('data/raw', 'data/interim', 'data/percent', 'data/eigen',
                      'result/WL', 'result/Hop', 'result/Batch'):
        os.makedirs(directory, exist_ok=True)

    source_file = data_path / dataset.raw_file
    if not source_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_file}")

    # Symlink rather than copy: btc_otc is the largest of these and there is no
    # reason to duplicate it inside the container.
    dest_file = Path('data/raw') / dataset.raw_file
    if dest_file.exists() or dest_file.is_symlink():
        dest_file.unlink()
    dest_file.symlink_to(source_file.absolute())

    print(f'[OK] Symlinked: {source_file.name} -> {dest_file}')
    print(f'[OK] Data directories ready\n')


def preprocess_data(dataset, anomaly_per: float, train_per: float):
    """
    Run TADDY's preprocessing pipeline to generate snapshots.

    Args:
        dataset: the Dataset to preprocess
        anomaly_per: Anomaly percentage
        train_per: Training data percentage
    """
    dataset_name = dataset.name
    snap_size = dataset.snap_size

    print(f'\n=== Preprocessing Dataset: {dataset_name} ===')
    print(f'Snap size: {snap_size}')
    print(f'Anomaly %: {anomaly_per}')
    print(f'Train %: {train_per}')

    # Step 1: Preprocess raw data to interim format
    t0 = time.time()
    edges = dataset.read(Path('data/raw') / dataset.raw_file)
    
    # Remove self-loops and duplicates
    for ii in range(len(edges)):
        x0 = edges[ii][0]
        x1 = edges[ii][1]
        if x0 > x1:
            edges[ii][0] = x1
            edges[ii][1] = x0
    
    edges = edges[np.nonzero([x[0] != x[1] for x in edges])].tolist()
    aa, idx = np.unique(edges, return_index=True, axis=0)
    edges = np.array(edges)
    edges = edges[np.sort(idx)]
    
    # Relabel vertices
    vertexs, edges = np.unique(edges, return_inverse=True)
    edges = np.reshape(edges, [-1, 2])
    print(f'Vertices: {len(vertexs)}, Edges: {len(edges)}')
    
    # Save interim data
    np.savetxt(
        f'data/interim/{dataset_name}',
        X=edges,
        delimiter=' ',
        comments='%',
        fmt='%d')
    print(f'[OK] Preprocess finished! Time: {time.time() - t0:.2f}s')
    
    # Step 2: Generate anomalies and create snapshots
    print(f'\n=== Generating Anomalies ===')
    t0 = time.time()
    m = len(edges)
    n = len(vertexs)
    
    synthetic_test, train_mat, train = anomaly_generation(
        train_per, anomaly_per, edges, n, m, seed=1
    )
    print(f'[OK] Anomaly generation finished! Time: {time.time() - t0:.2f}s')
    
    # Step 3: Build snapshots
    print(f'\n=== Building Snapshots ===')
    t0 = time.time()
    
    train_mat = (train_mat + train_mat.transpose() + sparse.eye(n)).tolil()
    headtail = train_mat.rows
    del train_mat
    
    train_size = int(len(train) / snap_size + 0.5)
    test_size = int(len(synthetic_test) / snap_size + 0.5)
    print(f'Train: {len(train)} edges, {train_size} snaps')
    print(f'Test: {len(synthetic_test)} edges, {test_size} snaps')
    
    rows = []
    cols = []
    weis = []
    labs = []
    
    # Training snapshots
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size
        
        row = np.array(train[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(train[start_loc:end_loc, 1], dtype=np.int32)
        lab = np.zeros_like(row, dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)
        
        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)
    
    print(f'[OK] Training snapshots created! Time: {time.time() - t0:.2f}s')
    
    # Test snapshots
    t0 = time.time()
    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size
        
        row = np.array(synthetic_test[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(synthetic_test[start_loc:end_loc, 1], dtype=np.int32)
        lab = np.array(synthetic_test[start_loc:end_loc, 2], dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)
        
        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)
    
    print(f'[OK] Test snapshots created! Time: {time.time() - t0:.2f}s')
    
    # Save processed data
    output_file = f'data/percent/{dataset_name}_{train_per}_{anomaly_per}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump((rows, cols, labs, weis, headtail, train_size, test_size, n, m), f, pickle.HIGHEST_PROTOCOL)
    print(f'[OK] Saved: {output_file}\n')


def resolve_dataset(directory):
    """The Dataset the mounted directory holds, matched on its name."""
    name = directory.name
    if name.startswith('taddy_'):
        name = name[len('taddy_'):]

    for dataset in DATASETS:
        if any(alias in name for alias in dataset.aliases):
            return dataset

    raise ValueError(
        f"Unknown dataset: {name}. Supported: "
        f"{', '.join(d.name for d in DATASETS)}"
    )


def main():
    run = paths()
    config = Config(**params(Config))
    dataset = resolve_dataset(run.data)

    info(f"[INFO] TADDY on {dataset.name}: {asdict(config)}")

    # Setup data directories
    setup_data_directories(run.data, dataset)

    # Check if preprocessed data exists, if not, preprocess it. The cache key is
    # the two ratios that shape it; the injected anomalies do not depend on
    # _SEED, which is why it is not part of the name -- see the README.
    processed_file = (f'data/percent/{dataset.name}_{config.train_per}'
                      f'_{config.anomaly_per}.pkl')
    if not os.path.exists(processed_file):
        preprocess_data(dataset, config.anomaly_per, config.train_per)
    else:
        print(f'[OK] Using existing preprocessed data: {processed_file}\n')

    seed_all(config.seed)

    # Initialize ResultWriter
    writer = ResultWriter()
    
    # Load dataset
    print('=== Loading Dataset ===')
    data_obj = DynamicDatasetLoader()
    data_obj.dataset_name = dataset.name
    data_obj.k = config.neighbor_num
    data_obj.window_size = config.window_size
    data_obj.anomaly_per = config.anomaly_per
    data_obj.train_per = config.train_per
    data_obj.load_all_tag = False
    data_obj.compute_s = True
    
    # Configure model
    print('=== Configuring Model ===')
    my_config = MyConfig(
        k=config.neighbor_num,
        window_size=config.window_size,
        hidden_size=config.embedding_dim,
        intermediate_size=config.embedding_dim,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        weight_decay=config.weight_decay
    )

    # Initialize model with ResultWriter for spot() tracking. `config` is also
    # the `args` object upstream keeps a reference to.
    print('=== Initializing Model ===')
    method_obj = DynADModelWithResults(my_config, config, result_writer=writer)
    method_obj.spy_tag = True
    method_obj.max_epoch = config.max_epoch
    method_obj.lr = config.learning_rate

    method_obj = method_obj.to(device(config.gpu))

    # Prepare and run training
    print('=== Starting Training ===\n')
    setting_obj = Settings()
    setting_obj.prepare(data_obj, method_obj)
    setting_obj.run()

    print('\n[OK] Training Completed')

    # Execution time and memory are measured by graflag_runner around this
    # process and merged into metadata afterwards, so there is nothing to
    # record here -- see _merge_runtime_metadata in the runner.
    save_results(method_obj, writer, run, dataset, config)


def save_results(method_obj, writer, run, dataset, config):
    """
    Save model predictions and metadata to results file.

    Args:
        method_obj: Trained model object with final predictions
        writer: ResultWriter instance
        run: the experiment's ExperimentPaths
        dataset: the Dataset this run used
        config: the Config this run used
    """
    print('\n=== Saving Results ===')

    if method_obj.final_preds is None:
        print('[WARN] No predictions captured - training may have failed')
        sys.exit(1)

    # Convert predictions to list format (each snap is a list of scores)
    scores = [pred.tolist() for pred in method_obj.final_preds]
    print(f'   Predictions for {len(scores)} snapshots')
    print(f'   Total edges: {sum(len(s) for s in scores)}')

    # Convert ground truth labels to list format
    ground_truth = [labels.tolist() for labels in method_obj.final_labels]
    print(f'   Ground truth for {len(ground_truth)} snapshots')

    # Save as TEMPORAL_EDGE_ANOMALY_SCORES (scores per edge per time snapshot)
    writer.save_scores(
        result_type="TEMPORAL_EDGE_ANOMALY_SCORES",
        scores=scores,
        ground_truth=ground_truth,
        timestamps=method_obj.snap_ids,
    )

    # Add metadata
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="TADDY",
        dataset=dataset.name,
        method_parameters=asdict(config),
    )

    info(f"[OK] Results written to {writer.finalize()}")


if __name__ == "__main__":
    main()
