"""
GADY Data Loader - Handles dataset setup and loading for GraFlag integration.

This module provides functions to:
1. Setup data directories by creating symlinks to GraFlag dataset mounts
2. Run GADY's prepare_data.py to generate anomaly datasets
3. Run GADY's preproc_new.py to compute positional features

GADY expects:
- Raw data files in ./data/ folder
- Processed .npy files from prepare_data.py
- Positional features from preproc_new.py

GraFlag provides:
- /data/datasets/gady_{dataset}/... containing raw files
"""

import os
import subprocess
import sys
from pathlib import Path


# Dataset configuration: maps GraFlag dataset names to GADY internal names and file patterns
DATASET_CONFIG = {
    'uci': {
        'gady_name': 'uci',
        'raw_file': 'uci',  # Space-separated: src dst label ts
        'file_format': 'space',
    },
    'btc_alpha': {
        'gady_name': 'btc_alpha', 
        'raw_file': 'soc-sign-bitcoinalpha.csv',  # CSV: src,dst,rating,ts
        'file_format': 'csv',
    },
    'email_dnc': {
        'gady_name': 'email_dnc',
        'raw_file': 'email-dnc.edges',  # CSV: src,dst,ts
        'file_format': 'csv_edges',
    },
}


def get_dataset_name_from_path(data_path: Path) -> str:
    """
    Extract dataset name from GraFlag mount path.
    
    Examples:
        /data/datasets/gady_uci -> uci
        /data/datasets/gady_btc_alpha -> btc_alpha
        /data/datasets/gady_email_dnc -> email_dnc
    """
    folder_name = data_path.name
    if folder_name.startswith('gady_'):
        return folder_name[5:]  # Remove 'gady_' prefix
    return folder_name


def setup_data_directories(data_path: Path, dataset_name: str):
    """
    Setup GADY's expected directory structure from GraFlag dataset mounts.
    
    GADY expects:
    - data/{raw_file} for input data
    - pos_features/ for positional features output
    - results/ for output
    
    Args:
        data_path: Path to mounted dataset (e.g., /data/datasets/gady_uci)
        dataset_name: Dataset name (uci, btc_alpha, email_dnc)
    """
    print(f'\n=== Setting Up Data Directories ===')
    print(f'Source: {data_path}')
    print(f'Dataset: {dataset_name}')
    
    # Create directory structure
    os.makedirs('data', exist_ok=True)
    os.makedirs('pos_features', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('log', exist_ok=True)
    
    # Get dataset configuration
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Supported: {list(DATASET_CONFIG.keys())}")
    
    # Setup raw data file symlink
    source_file = data_path / config['raw_file']
    dest_file = Path('data') / config['raw_file']
    
    if not source_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_file}")
    
    # Remove existing symlink/file if it exists
    if dest_file.exists() or dest_file.is_symlink():
        dest_file.unlink()
    
    dest_file.symlink_to(source_file.absolute())
    print(f'[OK] Symlinked: {source_file.name} -> {dest_file}')
    print(f'[OK] Data directories ready\n')


def run_prepare_data(dataset_name: str, anomaly_per: float = 0.1, 
                     train_per: float = 0.7, batch_size: int = 200):
    """
    Run GADY's prepare_data.py to generate train/test splits with anomalies.
    
    This creates:
    - data/{dataset}.csv - preprocessed edge list
    - data/{dataset}{anomaly_per}train.npy - training data with labels
    - data/{dataset}{anomaly_per}test.npy - test data with injected anomalies
    - data/ml_{dataset}_node.npy - node features (random 172-dim)
    
    Args:
        dataset_name: Dataset name (uci, btc_alpha, email_dnc)
        anomaly_per: Anomaly rate (0.01, 0.05, 0.1)
        train_per: Train split ratio
        batch_size: Batch size for processing
    """
    print(f'\n=== Running GADY Data Preparation ===')
    print(f'Dataset: {dataset_name}, Anomaly Rate: {anomaly_per}')
    
    # Check if already processed
    train_file = Path(f'data/{dataset_name}{anomaly_per}train.npy')
    test_file = Path(f'data/{dataset_name}{anomaly_per}test.npy')
    node_file = Path(f'data/ml_{dataset_name}_node.npy')
    
    if train_file.exists() and test_file.exists() and node_file.exists():
        print(f'[OK] Data already prepared, skipping...')
        return
    
    # Run prepare_data.py from GADY source
    cmd = [
        sys.executable, 'src/prepare_data.py',
        '--dataset', dataset_name,
        '--anomaly_per', str(anomaly_per),
        '--train_per', str(train_per),
        '--batch_size', str(batch_size)
    ]
    
    print(f'Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f'[ERROR] prepare_data.py failed:')
        print(result.stderr)
        raise RuntimeError(f"Data preparation failed: {result.stderr}")
    
    print(result.stdout)
    print(f'[OK] Data preparation complete\n')


def run_preproc_positional_features(dataset_name: str, anomaly_per: float = 0.1,
                                     r_dim: int = 4, batch_size: int = 200,
                                     gpu: int = 0):
    """
    Run GADY's preproc_new.py to compute positional features (V, R matrices).
    
    This precomputes positional features for both train and test splits.
    Creates pos_features/{dataset}_nextVR_part_* files.
    
    Args:
        dataset_name: Dataset name (uci, btc_alpha, email_dnc)
        anomaly_per: Anomaly rate
        r_dim: Dimension of positional features (default 4)
        batch_size: Batch size
        gpu: GPU index to use
    """
    print(f'\n=== Computing Positional Features ===')
    print(f'Dataset: {dataset_name}, R-dim: {r_dim}')
    
    # Run for both train and test splits
    for split in ['train', 'test']:
        print(f'\nProcessing {split} split...')
        
        cmd = [
            sys.executable, 'src/preproc_new.py',
            '--data', dataset_name,
            '--gpu', str(gpu),
            '--r_dim', str(r_dim),
            '--data_split', split,
            '--anomaly_per', str(anomaly_per),
            '--bs', str(batch_size)
        ]
        
        print(f'Running: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'[ERROR] preproc_new.py failed for {split}:')
            print(result.stderr)
            raise RuntimeError(f"Preprocessing failed: {result.stderr}")
        
        print(result.stdout)
    
    print(f'[OK] Positional features computed\n')


def ensure_data_ready(data_path: Path, dataset_name: str, 
                      anomaly_per: float = 0.1, train_per: float = 0.7,
                      batch_size: int = 200, r_dim: int = 4, gpu: int = 0):
    """
    Ensure all data is ready for GADY training.
    
    This is the main entry point that:
    1. Sets up data directories with symlinks
    2. Runs prepare_data.py if needed
    3. Runs preproc_new.py if needed
    
    Args:
        data_path: Path to GraFlag dataset mount
        dataset_name: Dataset name
        anomaly_per: Anomaly rate
        train_per: Train split ratio
        batch_size: Batch size
        r_dim: Positional feature dimension
        gpu: GPU index
    """
    # Step 1: Setup directories
    setup_data_directories(data_path, dataset_name)
    
    # Step 2: Prepare data with anomaly injection
    run_prepare_data(dataset_name, anomaly_per, train_per, batch_size)
    
    # Step 3: Compute positional features
    run_preproc_positional_features(dataset_name, anomaly_per, r_dim, batch_size, gpu)
    
    print('=' * 50)
    print('[OK] All data preparation complete!')
    print('=' * 50)


if __name__ == '__main__':
    # Test with UCI dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--anomaly_per', type=float, default=0.1)
    parser.add_argument('--train_per', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--r_dim', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    dataset_name = get_dataset_name_from_path(data_path)
    
    ensure_data_ready(
        data_path, dataset_name,
        args.anomaly_per, args.train_per, args.batch_size,
        args.r_dim, args.gpu
    )
