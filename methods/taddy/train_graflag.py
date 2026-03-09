"""
GraFlag-integrated training script for TADDY.
This wrapper runs the TADDY model with GraFlag standardized results.
"""

import sys
import os
import time
import shutil
import pickle
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import psutil

from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Component import MyConfig
from codes.DynADModel import DynADModel
from codes.Settings import Settings
from codes.AnomalyGeneration import anomaly_generation
from scipy import sparse

# GraFlag integration
from graflag_runner import ResultWriter


# Parse command line arguments (provided by graflag_runner from _* env vars)
parser = argparse.ArgumentParser()
parser.add_argument('--anomaly_per', type=float, default=0.1)
parser.add_argument('--train_per', type=float, default=0.5)
parser.add_argument('--neighbor_num', type=int, default=5)
parser.add_argument('--window_size', type=int, default=2)
parser.add_argument('--embedding_dim', type=int, default=32)
parser.add_argument('--num_hidden_layers', type=int, default=2)
parser.add_argument('--num_attention_heads', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


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


def setup_data_directories(data_path: Path, dataset_name: str):
    """
    Setup TADDY's expected directory structure from GraFlag dataset mounts.
    
    TADDY expects:
    - data/raw/{dataset_name} or .csv for bitcoin datasets
    
    GraFlag provides:
    - /data/datasets/taddy_{dataset}/...
    
    Args:
        data_path: Path to mounted dataset (e.g., /data/datasets/taddy_uci)
        dataset_name: Dataset name (uci, digg, btc_alpha, btc_otc)
    """
    print(f'\n=== Setting Up Data Directories ===')
    print(f'Source: {data_path}')
    print(f'Dataset: {dataset_name}')
    
    # Create directory structure
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('data/percent', exist_ok=True)
    os.makedirs('result/WL', exist_ok=True)
    os.makedirs('result/Hop', exist_ok=True)
    os.makedirs('result/Batch', exist_ok=True)
    os.makedirs('data/eigen', exist_ok=True)
    
    # Map dataset to expected file structure
    if dataset_name == 'uci':
        source_file = data_path / 'uci'
        dest_file = Path('data/raw/uci')
    elif dataset_name == 'digg':
        source_file = data_path / 'digg'
        dest_file = Path('data/raw/digg')
    elif dataset_name == 'btc_alpha':
        source_file = data_path / 'soc-sign-bitcoinalpha.csv'
        dest_file = Path('data/raw/soc-sign-bitcoinalpha.csv')
    elif dataset_name == 'btc_otc':
        source_file = data_path / 'soc-sign-bitcoinotc.csv'
        dest_file = Path('data/raw/soc-sign-bitcoinotc.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create symlink to raw data file
    if not source_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_file}")
    
    # Remove existing symlink/file if it exists
    if dest_file.exists() or dest_file.is_symlink():
        dest_file.unlink()
    
    dest_file.symlink_to(source_file.absolute())
    print(f'[OK] Symlinked: {source_file.name} -> {dest_file}')
    print(f'[OK] Data directories ready\n')


def preprocess_data(dataset_name: str, anomaly_per: float, train_per: float):
    """
    Run TADDY's preprocessing pipeline to generate snapshots.
    
    Args:
        dataset_name: Dataset name (uci, digg, btc_alpha, btc_otc)
        anomaly_per: Anomaly percentage
        train_per: Training data percentage
    """
    snap_size_dict = {'uci': 1000, 'digg': 6000, 'btc_alpha': 1000, 'btc_otc': 2000}
    snap_size = snap_size_dict[dataset_name]
    
    print(f'\n=== Preprocessing Dataset: {dataset_name} ===')
    print(f'Snap size: {snap_size}')
    print(f'Anomaly %: {anomaly_per}')
    print(f'Train %: {train_per}')
    
    # Step 1: Preprocess raw data to interim format
    t0 = time.time()
    if dataset_name in ['digg', 'uci']:
        edges = np.loadtxt(
            f'data/raw/{dataset_name}',
            dtype=float,
            comments='%',
            delimiter=' ')
        edges = edges[:, 0:2].astype(dtype=int)
    elif dataset_name in ['btc_alpha', 'btc_otc']:
        if dataset_name == 'btc_alpha':
            file_name = 'data/raw/soc-sign-bitcoinalpha.csv'
        elif dataset_name == 'btc_otc':
            file_name = 'data/raw/soc-sign-bitcoinotc.csv'
        with open(file_name) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = np.array(edges)
        edges = edges[edges[:, 3].argsort()]  # Sort by timestamp
        edges = edges[:, 0:2].astype(dtype=int)
    
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


def main():
    # Get dataset path from environment (set by graflag)
    data_dir = os.environ.get("DATA")
    if not data_dir:
        raise ValueError("DATA environment variable not set")
    
    data_path = Path(data_dir)
    
    # Infer dataset name from directory name
    # Supports both prefixed (taddy_uci) and unprefixed (uci) names
    dir_name = data_path.name

    # Remove taddy_ prefix if present
    if dir_name.startswith('taddy_'):
        dir_name = dir_name[6:]  # Remove 'taddy_'

    # Map directory names to TADDY dataset names
    if dir_name == 'uci' or 'uci' in dir_name:
        dataset_name = 'uci'
    elif dir_name == 'digg' or 'digg' in dir_name:
        dataset_name = 'digg'
    elif dir_name == 'btc_alpha' or 'bitcoinalpha' in dir_name:
        dataset_name = 'btc_alpha'
    elif dir_name == 'btc_otc' or 'bitcoinotc' in dir_name:
        dataset_name = 'btc_otc'
    else:
        raise ValueError(f"Unknown dataset: {dir_name}. Supported: uci, digg, btc_alpha, btc_otc")
    
    print('--- TADDY + GraFlag Integration ---')

    print(f'[INFO] Configuration:')
    print(f'   Dataset: {dataset_name}')
    print(f'   Anomaly %: {args.anomaly_per}')
    print(f'   Train %: {args.train_per}')
    print(f'   Seed: {args.seed}')
    print(f'   Max epochs: {args.max_epoch}')
    print(f'   Learning rate: {args.learning_rate}')
    print(f'   GPU: {args.gpu}')
    
    # Setup data directories
    setup_data_directories(data_path, dataset_name)
    
    # Check if preprocessed data exists, if not, preprocess it
    processed_file = f'data/percent/{dataset_name}_{args.train_per}_{args.anomaly_per}.pkl'
    if not os.path.exists(processed_file):
        preprocess_data(dataset_name, args.anomaly_per, args.train_per)
    else:
        print(f'[OK] Using existing preprocessed data: {processed_file}\n')
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Start resource tracking
    import psutil
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize ResultWriter
    writer = ResultWriter()
    
    # Load dataset
    print('=== Loading Dataset ===')
    data_obj = DynamicDatasetLoader()
    data_obj.dataset_name = dataset_name
    data_obj.k = args.neighbor_num
    data_obj.window_size = args.window_size
    data_obj.anomaly_per = args.anomaly_per
    data_obj.train_per = args.train_per
    data_obj.load_all_tag = False
    data_obj.compute_s = True
    
    # Configure model
    print('=== Configuring Model ===')
    my_config = MyConfig(
        k=args.neighbor_num,
        window_size=args.window_size,
        hidden_size=args.embedding_dim,
        intermediate_size=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        weight_decay=args.weight_decay
    )
    print(f'   Embedding dim: {args.embedding_dim}')
    print(f'   Hidden layers: {args.num_hidden_layers}')
    print(f'   Attention heads: {args.num_attention_heads}\n')
    
    # Initialize model with ResultWriter for spot() tracking
    print('=== Initializing Model ===')
    method_obj = DynADModelWithResults(my_config, args, result_writer=writer)
    method_obj.spy_tag = True
    method_obj.max_epoch = args.max_epoch
    method_obj.lr = args.learning_rate

    # Move model to selected device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        method_obj = method_obj.to(device)
        print(f'   Using GPU: cuda:{args.gpu}')
    else:
        print('   Using CPU')
    
    # Prepare and run training
    print('=== Starting Training ===\n')
    setting_obj = Settings()
    setting_obj.prepare(data_obj, method_obj)
    setting_obj.run()

    print('\n[OK] Training Completed')

    # Calculate resource metrics
    end_time = time.time()
    exec_time_ms = (end_time - start_time) * 1000
    peak_memory_mb = process.memory_info().rss / (1024 * 1024)

    # Track GPU memory if used
    peak_gpu_mb = None
    if args.gpu >= 0 and torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated(device=f'cuda:{args.gpu}') / (1024 * 1024)

    print(f'\n[INFO] Resource Usage:')
    print(f'   [INFO] Execution time: {exec_time_ms/1000:.2f}s')
    print(f'   [INFO] Peak memory: {peak_memory_mb:.2f}MB')
    if peak_gpu_mb is not None:
        print(f'   [INFO] Peak GPU memory: {peak_gpu_mb:.2f}MB')

    # Save results using ResultWriter
    save_results(method_obj, writer, dataset_name, exec_time_ms, peak_memory_mb, peak_gpu_mb)

    print('\n--- All Done! ---')


def save_results(method_obj, writer, dataset_name, exec_time_ms, peak_memory_mb, peak_gpu_mb=None):
    """
    Save model predictions and metadata to results file.

    Args:
        method_obj: Trained model object with final predictions
        writer: ResultWriter instance
        dataset_name: Dataset name
        exec_time_ms: Execution time in milliseconds
        peak_memory_mb: Peak memory usage in MB
        peak_gpu_mb: Peak GPU memory usage in MB (optional)
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
        method_name="TADDY",
        dataset=dataset_name,
        anomaly_per=args.anomaly_per,
        train_per=args.train_per,
        neighbor_num=args.neighbor_num,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        max_epoch=args.max_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    # Add resource metrics
    writer.add_resource_metrics(
        exec_time_ms=exec_time_ms,
        peak_memory_mb=peak_memory_mb,
        peak_gpu_mb=peak_gpu_mb,
    )

    results_file = writer.finalize()
    print(f'[OK] Results saved to: {results_file}')


if __name__ == "__main__":
    main()
