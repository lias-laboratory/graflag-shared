"""
GADY GraFlag Integration - Train wrapper for GADY method.

GADY: Unsupervised Anomaly Detection on Dynamic Graphs (WSDM 2024)
https://github.com/mufeng-74/GADY

This script integrates GADY with the GraFlag benchmarking framework by:
1. Setting up data directories from GraFlag mounts
2. Running GADY's data preparation and preprocessing
3. Running GADY training and evaluation
4. Outputting results in GraFlag format using ResultWriter
"""

import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import psutil

# Add GADY source to path
sys.path.insert(0, 'src')

# GraFlag integration
from graflag_runner import ResultWriter

from data_loader import (
    DATASET_CONFIG,
    get_dataset_name_from_path,
    setup_data_directories,
    run_prepare_data,
    run_preproc_positional_features
)


def str2bool(v):
    """Convert string to boolean for argparse compatibility with GraFlag runner."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', ''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('GADY GraFlag Integration')
    
    # GraFlag standard args
    parser.add_argument('--data', type=str, help='Dataset name', default='uci')
    
    # GADY training args
    parser.add_argument('--seed', type=int, default=142, help='Random seed')
    parser.add_argument('--bs', type=int, default=200, help='Batch size')
    parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    
    # Model architecture
    parser.add_argument('--node_dim', type=int, default=100, help='Node embedding dim')
    parser.add_argument('--time_dim', type=int, default=1, help='Time embedding dim')
    parser.add_argument('--memory_dim', type=int, default=172, help='Memory dim')
    parser.add_argument('--message_dim', type=int, default=100, help='Message dim')
    
    # Memory settings (use str2bool for GraFlag runner compatibility)
    parser.add_argument('--use_memory', type=str2bool, nargs='?', const=True, default=False, help='Use memory')
    parser.add_argument('--memory_update_at_end', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--message_function', type=str, default='identity')
    parser.add_argument('--memory_updater', type=str, default='gru')
    parser.add_argument('--aggregator', type=str, default='last')
    
    # GADY specific
    parser.add_argument('--mode', type=int, default=0, help='GADY mode (0=normal, 1=ablation)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter')
    parser.add_argument('--betaa', type=float, default=10, help='Beta parameter')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma parameter')
    parser.add_argument('--beta', type=float, default=0.00001, help='Positional feature beta')
    parser.add_argument('--r_dim', type=int, default=4, help='Positional feature dim')
    parser.add_argument('--lr_g', '--lr_G', type=float, default=0.000001, help='Generator LR')
    parser.add_argument('--lr_d', '--lr_D', type=float, default=0.000001, help='Discriminator LR')
    
    # Data settings
    parser.add_argument('--anomaly_per', type=float, default=0.1, help='Anomaly rate')
    parser.add_argument('--train_per', type=float, default=0.7, help='Train split')
    
    # Other flags (use str2bool for GraFlag runner compatibility)
    parser.add_argument('--uniform', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--randomize_features', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_destination_embedding_in_message', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_source_embedding_in_message', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--different_new_nodes', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--prefix', type=str, nargs='?', const='', default='')
    
    return parser.parse_args()


def get_graflag_env_config():
    """Get configuration from GraFlag environment variables."""
    config = {}
    
    # Data path from GraFlag
    data_path = os.environ.get('DATA')
    if data_path:
        config['data_path'] = Path(data_path)
        config['dataset'] = get_dataset_name_from_path(config['data_path'])
    
    # Map environment variables to arguments
    env_mappings = {
        'ANOMALY_PER': ('anomaly_per', float),
        'TRAIN_PER': ('train_per', float),
        'BATCH_SIZE': ('bs', int),
        'N_LAYER': ('n_layer', int),
        'N_DEGREE': ('n_degree', int),
        'MEMORY_DIM': ('memory_dim', int),
        'MESSAGE_DIM': ('message_dim', int),
        'NODE_DIM': ('node_dim', int),
        'TIME_DIM': ('time_dim', int),
        'USE_MEMORY': ('use_memory', lambda x: x.lower() == 'true'),
        'MEMORY_UPDATE_AT_END': ('memory_update_at_end', lambda x: x.lower() == 'true'),
        'MESSAGE_FUNCTION': ('message_function', str),
        'MEMORY_UPDATER': ('memory_updater', str),
        'AGGREGATOR': ('aggregator', str),
        'R_DIM': ('r_dim', int),
        'BETA': ('beta', float),
        'N_EPOCH': ('n_epoch', int),
        'LR': ('lr', float),
        'LR_G': ('lr_G', float),
        'LR_D': ('lr_D', float),
        'PATIENCE': ('patience', int),
        'SEED': ('seed', int),
        'ALPHA': ('alpha', float),
        'BETAA': ('betaa', float),
        'GAMMA': ('gamma', float),
        'MODE': ('mode', int),
        'USE_DESTINATION_EMBEDDING_IN_MESSAGE': ('use_destination_embedding_in_message', lambda x: x.lower() == 'true'),
        'USE_SOURCE_EMBEDDING_IN_MESSAGE': ('use_source_embedding_in_message', lambda x: x.lower() == 'true'),
        'RANDOMIZE_FEATURES': ('randomize_features', lambda x: x.lower() == 'true'),
        'UNIFORM': ('uniform', lambda x: x.lower() == 'true'),
    }
    
    for env_key, (arg_key, converter) in env_mappings.items():
        value = os.environ.get(env_key)
        if value is not None:
            try:
                config[arg_key] = converter(value)
            except (ValueError, TypeError):
                pass
    
    return config


def setup_logging(data_name: str, anomaly_per: float):
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f'log/{data_name}_{anomaly_per}_{time.time()}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def run_gady_training(args, logger, writer):
    """
    Run GADY training and evaluation.

    Args:
        args: Command line arguments
        logger: Logger instance
        writer: ResultWriter instance for tracking metrics

    Returns:
        dict: Results including AUC-ROC scores, predictions, and edge-level scores
    """
    # Import GADY modules
    from modules.GAN import Generator
    from evaluation.evaluation import eval_edge_prediction
    from model.tgn import TGN
    from utils.utils import (EarlyStopMonitor, RandEdgeSampler,
                             get_neighbor_finder, get_data_settings,
                             GenFGANLoss, DiscFGANLoss)
    from utils.data_processing import get_data, compute_time_statistics

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(args)
    
    # Load data
    node_features, edge_features, full_data, train_data, test_data = get_data(
        args.data,
        different_new_nodes_between_val_and_test=args.different_new_nodes,
        randomize_features=args.randomize_features,
        anomaly_per=args.anomaly_per
    )
    
    # Initialize neighbor finders
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
    
    if args.mode == 1:  # Ablation study
        train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    
    # Set device
    device_string = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)
    logger.info(f'Using device: {device}')
    
    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    
    # Store best results across runs
    best_auc = 0
    best_ap = 0
    all_results = []
    
    # Training loop
    train_generator = Generator(n_neighbors=args.n_degree, batch_size=args.bs, device=device)
    
    for run_idx in range(args.n_runs):
        logger.info(f'\n=== Run {run_idx + 1}/{args.n_runs} ===')
        
        results_path = f"results/GADY-{args.data}_{run_idx}.pkl" if args.prefix == '' else f"results/{args.prefix}_{run_idx}.pkl"
        Path("results/").mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        discriminator = TGN(
            neighbor_finder=train_ngh_finder, 
            node_features=node_features,
            edge_features=edge_features, 
            device=device,
            n_layers=args.n_layer, 
            use_memory=args.use_memory,
            message_dimension=args.message_dim, 
            memory_dimension=args.memory_dim,
            memory_update_at_start=not args.memory_update_at_end,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=args.n_degree,
            mean_time_shift_src=mean_time_shift_src, 
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, 
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            beta=args.beta,
            r_dim=args.r_dim,
            lr_G=args.lr_g,
            lr_D=args.lr_d
        )
        
        # Setup loss functions
        criterion_gen = GenFGANLoss().to(device)
        criterion_disc = DiscFGANLoss(args.betaa, args.gamma).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        
        # Early stopping
        early_stopper = EarlyStopMonitor(max_round=args.patience)
        
        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / args.bs)
        partition_size, _ = get_data_settings(args.data)
        
        # Training epochs
        for epoch in range(args.n_epoch):
            logger.info(f'Epoch {epoch + 1}/{args.n_epoch}')
            
            discriminator.memory.__init_memory__()
            
            epoch_loss = 0
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.bs
                end_idx = min(num_instance, start_idx + args.bs)
                
                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                
                size = len(sources_batch)
                
                with torch.no_grad():
                    pos_label = torch.zeros(size, dtype=torch.float, device=device)
                    neg_label = torch.ones(size, dtype=torch.float, device=device)
                
                discriminator.train()
                
                # Load positional features
                prt = batch_idx // partition_size
                try:
                    next_V, next_R = torch.load(
                        f'pos_features/{args.data}_nextVR_part_{prt}_bs_{args.bs}_rdim_{args.r_dim}{args.anomaly_per}'
                    )
                    for c in range(len(next_V)):
                        next_V[c] = next_V[c].to(device)
                        next_R[c] = next_R[c].to(device)
                except FileNotFoundError:
                    logger.warning(f'Positional features not found for partition {prt}')
                    continue
                
                idx = batch_idx % partition_size
                
                # Forward pass
                if args.mode == 0:
                    discriminator.Generator.eval()
                    pos_prob = discriminator.compute_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch, 
                        edge_idxs_batch, args.n_degree,
                        update_memory=True,
                        next_V=discriminator.V + next_V[idx].to_dense(),
                        next_R=discriminator.R + next_R[idx].to_dense()
                    )
                    
                    neg_prob2, _ = discriminator.compute_neg_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree
                    )
                    
                    # Discriminator loss
                    loss = criterion_disc(pos_prob, neg_prob2, args.alpha)
                    
                else:  # Ablation mode
                    pos_prob = discriminator.compute_edge_probabilities(
                        sources_batch, destinations_batch, timestamps_batch,
                        edge_idxs_batch, args.n_degree,
                        update_memory=True
                    )
                    loss = -torch.mean(torch.log(pos_prob + 1e-8))
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Detach memory
                discriminator.memory.detach_memory()
            
            avg_loss = epoch_loss / num_batch
            logger.info(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')

            # Evaluation
            discriminator.eval()
            discriminator.memory.__init_memory__()

            test_auc, test_ap = eval_edge_prediction(
                model=discriminator,
                test_data=test_data,
                train_data=train_data,
                args=args,
                test_rand_sampler=test_rand_sampler,
                partition_size=partition_size,
                device=device
            )

            logger.info(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

            # Track metrics with spot()
            writer.spot("training",
                        epoch=epoch + 1,
                        loss=avg_loss,
                        test_auc=test_auc,
                        test_ap=test_ap)

            # Early stopping
            if early_stopper.early_stop_check(test_auc):
                logger.info(f'Early stopping at epoch {epoch + 1}')
                break

            if test_auc > best_auc:
                best_auc = test_auc
                best_ap = test_ap

        run_result = {
            'run': run_idx,
            'auc': test_auc,
            'ap': test_ap
        }
        all_results.append(run_result)
        logger.info(f'Run {run_idx + 1} - AUC: {test_auc:.4f}, AP: {test_ap:.4f}')

    # Collect edge-level scores from test data for results
    # Generate predictions for all test edges
    discriminator.eval()
    discriminator.memory.__init_memory__()

    all_scores = []
    all_labels = []
    all_edges = []
    all_timestamps = []

    test_sources = test_data.sources
    test_destinations = test_data.destinations
    test_timestamps = test_data.timestamps
    test_edge_idxs = test_data.edge_idxs
    test_labels = test_data.labels

    # Process in batches
    num_test = len(test_sources)
    for i in range(0, num_test, args.bs):
        end_idx = min(i + args.bs, num_test)
        src_batch = test_sources[i:end_idx]
        dst_batch = test_destinations[i:end_idx]
        ts_batch = test_timestamps[i:end_idx]
        edge_idx_batch = test_edge_idxs[i:end_idx]

        with torch.no_grad():
            prob = discriminator.compute_edge_probabilities(
                src_batch, dst_batch, ts_batch, edge_idx_batch, args.n_degree
            )
            # Anomaly score = 1 - probability (higher prob = normal, lower = anomaly)
            scores = (1 - prob.cpu().numpy()).tolist()

        all_scores.extend(scores)
        all_labels.extend(test_labels[i:end_idx].tolist())
        all_edges.extend([[int(s), int(d)] for s, d in zip(src_batch, dst_batch)])
        all_timestamps.extend(ts_batch.tolist())

    # Final results
    final_results = {
        'method': 'GADY',
        'dataset': args.data,
        'anomaly_rate': args.anomaly_per,
        'best_auc': best_auc,
        'best_ap': best_ap,
        'all_runs': all_results,
        'mean_auc': np.mean([r['auc'] for r in all_results]),
        'std_auc': np.std([r['auc'] for r in all_results]),
        'mean_ap': np.mean([r['ap'] for r in all_results]),
        'std_ap': np.std([r['ap'] for r in all_results]),
        'scores': all_scores,
        'labels': all_labels,
        'edges': all_edges,
        'timestamps': all_timestamps,
    }

    return final_results


def main():
    """Main entry point for GADY GraFlag integration."""
    print('=' * 60)
    print('GADY - Unsupervised Anomaly Detection on Dynamic Graphs')
    print('GraFlag Integration')
    print('=' * 60)

    # Parse args and merge with env config
    args = parse_args()
    env_config = get_graflag_env_config()

    # Override args with env config
    for key, value in env_config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # Get data path and dataset name
    if 'data_path' in env_config:
        data_path = env_config['data_path']
        args.data = env_config['dataset']
    else:
        # Fallback: assume data is in current directory
        data_path = Path(os.environ.get('DATA', '.'))
        args.data = get_dataset_name_from_path(data_path) if data_path.exists() else args.data

    print(f'\nConfiguration:')
    print(f'  Dataset: {args.data}')
    print(f'  Data Path: {data_path}')
    print(f'  Anomaly Rate: {args.anomaly_per}')
    print(f'  GPU: {args.gpu}')
    print(f'  Epochs: {args.n_epoch}')
    print(f'  Runs: {args.n_runs}')

    # Setup logging
    logger = setup_logging(args.data, args.anomaly_per)

    # Start resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize ResultWriter
    writer = ResultWriter()

    try:
        # Step 1: Setup data directories
        print('\n--- Step 1: Setting up data directories ---')
        setup_data_directories(data_path, args.data)

        # Track memory
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Step 2: Prepare data with anomaly injection
        print('\n--- Step 2: Preparing data with anomaly injection ---')
        run_prepare_data(args.data, args.anomaly_per, args.train_per, args.bs)

        # Track memory
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Step 3: Compute positional features
        print('\n--- Step 3: Computing positional features ---')
        run_preproc_positional_features(
            args.data, args.anomaly_per, args.r_dim, args.bs, args.gpu
        )

        # Track memory
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Step 4: Run GADY training
        print('\n--- Step 4: Running GADY training ---')
        results = run_gady_training(args, logger, writer)

        # Calculate resource metrics
        end_time = time.time()
        exec_time_ms = (end_time - start_time) * 1000
        peak_memory_mb = max(peak_memory_mb, process.memory_info().rss / (1024 * 1024))

        # Track GPU memory if available
        peak_gpu_mb = None
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Save scores using ResultWriter
        writer.save_scores(
            result_type="EDGE_STREAM_ANOMALY_SCORES",
            scores=results['scores'],
            edges=results['edges'],
            timestamps=results['timestamps'],
            ground_truth=results['labels'],
        )

        # Build method parameters dict
        method_params = {
            'seed': args.seed,
            'batch_size': args.bs,
            'n_degree': args.n_degree,
            'n_epoch': args.n_epoch,
            'n_layer': args.n_layer,
            'lr': args.lr,
            'patience': args.patience,
            'n_runs': args.n_runs,
            'node_dim': args.node_dim,
            'time_dim': args.time_dim,
            'memory_dim': args.memory_dim,
            'message_dim': args.message_dim,
            'use_memory': args.use_memory,
            'mode': args.mode,
            'alpha': args.alpha,
            'betaa': args.betaa,
            'gamma': args.gamma,
            'anomaly_per': args.anomaly_per,
            'train_per': args.train_per,
        }

        # Add metadata
        writer.add_metadata(
            exp_name=os.path.basename(os.environ.get("EXP", "experiment")),
            method_name="gady",
            dataset=args.data,
            method_parameters=method_params,
            threshold=None,
            summary={
                "description": "GADY: Unsupervised Anomaly Detection on Dynamic Graphs (WSDM 2024)",
                "task": "edge_stream_anomaly_detection",
                "dataset_info": {
                    "name": args.data,
                    "anomaly_rate": args.anomaly_per,
                    "total_test_edges": len(results['scores']),
                    "n_anomalies": sum(results['labels']),
                },
                "training_info": {
                    "n_runs": args.n_runs,
                    "best_auc": float(results['best_auc']),
                    "best_ap": float(results['best_ap']),
                    "mean_auc": float(results['mean_auc']),
                    "std_auc": float(results['std_auc']),
                    "mean_ap": float(results['mean_ap']),
                    "std_ap": float(results['std_ap']),
                },
            },
        )

        # Add resource metrics
        writer.add_resource_metrics(
            exec_time_ms=exec_time_ms,
            peak_memory_mb=peak_memory_mb,
            peak_gpu_mb=peak_gpu_mb,
        )

        # Finalize results
        results_file = writer.finalize()

        print(f'\n[INFO] Resource Usage:')
        print(f'   [INFO] Execution time: {exec_time_ms/1000:.2f}s')
        print(f'   [INFO] Peak memory: {peak_memory_mb:.2f}MB')
        if peak_gpu_mb is not None:
            print(f'   [INFO] Peak GPU memory: {peak_gpu_mb:.2f}MB')

        print(f'\n[OK] GADY completed successfully!')
        print(f'   Best AUC: {results["best_auc"]:.4f}')
        print(f'   Best AP: {results["best_ap"]:.4f}')
        print(f'   Results saved to: {results_file}')

    except Exception as e:
        logger.error(f'Error during GADY training: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
