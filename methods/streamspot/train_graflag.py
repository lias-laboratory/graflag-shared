"""
GraFlag-integrated training script for StreamSpot.
StreamSpot: Graph-Based Anomaly Detection in System Provenance Data

StreamSpot detects anomalies at the GRAPH level - each graph represents
a complete system provenance trace from a scenario (e.g., YouTube browsing,
Gmail usage, or drive-by-download attack).
"""

import os
import sys
import time
import subprocess
import re
import argparse
from pathlib import Path
import numpy as np
import psutil

# GraFlag integration
from graflag_runner import ResultWriter

# Pre-computed bootstrap clusters from the original paper
# Source: https://gist.github.com/emaadmanzoor/118846a642727a0bf704
# Format: first line = "num_clusters\tglobal_threshold"
#         subsequent lines = "cluster_threshold\tgraph_id\tgraph_id\t..."
BOOTSTRAP_CLUSTERS_DATA = {
    "all": {
        "header": (10, 0.4823),
        "clusters": [
            (0.4341, [80,79,25,15,39,40,53,17,57,50,18,69,87,16,47,3,38,52,8,34,44,72,59,91,98,14,21,12,58,82,95,86,76,54,90,42,32,23,37,62,9,1,45,75,55,81,92,99,36,56,13,46,27,24,28,65,7,88,61,97,77,73,63,29,0,51,10,74,67,66,60,84,85,30,89,115,273,278,277,213,224,280,286,211,237,227,272,229,292,268,258,285,206,209,298,261,282,216,251,207,200,270,256,239,234,230,263,294,220,248,284,244,228,293,217,214,208,281,210,225,297,291,205,202,222,231,218,249,215,241,265,295,204,232,243,279,276,274,254,266,233,287,219,221,253,212,246,264,203,235,283,542,535,552]),
            (0.0300, [465,473,498,452,479,466,437,486,462,476,467,497,496,472,566,507,508,527]),
            (0.7231, [150,173,120,169,110,125,191,187,128,189,164,132,183,134,197,119,151,144,163,180,179,171,140,102,185,113,104,126,155,116,158,176,196,157,174,162,114,133,455,416]),
            (0.1182, [160,112,145,105,108,139,188,181,131,199,123,182,124,166,193,175,129,154,186,168,138,184,137,101,143,152,148,149,161,147,109,136,106,194,177,130]),
            (0.0014, [442,491,434,485,412,492,420,448,495,463,407,422,429,402,411,417,431,470,449,428,421,406,446,409,458,460,403,435,440,401,419,487,484,405,499,418,444,477,408,461,469,427,424,413]),
            (0.1967, [505,530,588,578,591,579,555,574,514,502,547,548,506,519,524,550,531]),
            (0.0041, [438,464,468,447,430,439,475,423,459,450]),
            (0.4854, [509,569,595,573,583,523,510,517,541,544,533,526,554,558,571,534,537,511,584,540,587,593,594,585,515,560,522,543,516,568,572,546,559,599,556,582,538,561,504,549,525]),
            (0.0046, [489,481,482,493,443]),
            (0.2012, [580,521,518,501,590,567,596,589,539,529]),
        ]
    },
    "ydc": {
        "header": (5, 0.9742),
        "clusters": [
            (0.6076, [53,72,63,22,68,24,40,87,21,74,52,34,69,44,54,25,37,55,13,78,33,26,9,83,77,0,43,46,12,2,17,67,38,23,86,80,3,93,16,84,97,47,29,36,6,57,4,11,95,14]),
            (0.4788, [82,66,75,96,90,30,70,41,35,85,61,7,1,8,79]),
            (0.3526, [432,491,460,437,403,488,478,430,411,499,493,439,425,451,417,421,483,466,461,445,473,443,420,469,467,447,414,436,481,456,485,407,424,497,405,471,406,413]),
            (1.1399, [542,551,568,537,503,574,527,556,590,522,593,595,510,530,555,509,512,519,585,529,546,567,504,580,596,578,582,599,515,506,571,575,531,564,508,516]),
            (0.9571, [99,71,50,51,42,39,88,89,10,65,59,49,32,58,76,19,18,64,45,92,31,94,20,5,98,15,27,48,91,56,60,81,28,73,62]),
        ]
    },
    "gfc": {
        "header": (10, 1.0288),
        "clusters": [
            (0.6287, [187,147,172,180,124,114,115,107,127,199,160,149,103,169,112,148,163,144,190,196,138,185,129,194,175,105,186,132,137,116,198,158,120,134,152,192,193,133,173,121,170,179,135,189,109,143,183,197,178,195,145,181,168,161]),
            (0.4611, [139,141,136,123,106,162,130,176,174,117,188,154,110,119,104,156]),
            (0.3811, [275,261,239,207,274,231,219,212,289,271,255,221,280,264,214,256,295,247,291,233,210,267,273,215,226,204,223,236,205,209,265,278,262,260,218,253,237,211,283,281,263,293,242]),
            (0.7336, [151,171,155,128,159,118,102,182,125,126,111,165,108]),
            (0.7069, [299,228,235,277,230,284,213,234,240,250,229,285,248,294,292,252,287,288,249,244,217,232,251,259,290,286,297,254]),
            (1.1069, [584,533,596,594,555,579,560,512,503,562,595,570,553,572,541,599,515,506,501,516,535,527,577,510,540,532,514,509,528,508]),
            (1.1893, [519,578,507,571,534,586,521,518,538,583,558,566,526,539,522,568,517,556,520,569,580,554,500,543,523,581,544,546,573,591,565,563,576]),
            (0.4652, [225,220,208,246,241,270,238,272,269,202,224,296,257,298,200,279,243,266,268,203,282,206,227,201]),
            (0.6001, [590,505,504,549,531,529,589,597,564,537,536,511,557,542,545,559,524,547,550,598,525,552,502,513,575,587,588,561,574,593,592,582]),
            (1.2903, [166,113,191,167,184,142,146,153,122,131,177,157,140,164,101,100]),
        ]
    }
}

# Ground truth: graphs 300-399 are attacks (drive-by-download)
# All other graphs (0-99, 100-199, 200-299, 400-499, 500-599) are benign
ATTACK_GRAPH_IDS = set(range(300, 400))


def parse_args():
    """Parse command line arguments (passed by graflag_runner --pass-env-args)."""
    parser = argparse.ArgumentParser(description='StreamSpot Training')
    parser.add_argument('--chunk_length', type=int, default=10, help='Chunk length for shingle construction')
    parser.add_argument('--num_parallel_graphs', type=int, default=10, help='Number of parallel graphs to process')
    parser.add_argument('--max_num_edges', type=int, default=-1, help='Max edges per graph (-1 for unlimited)')
    parser.add_argument('--dataset', type=str, default='all', help='Dataset name')
    parser.add_argument('--training_ratio', type=float, default=0.5, help='Training ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--global_threshold', type=float, default=0.6, help='Global threshold for anomaly detection')
    return parser.parse_args()


def get_config_from_args(args):
    """Convert parsed args to config dict."""
    return {
        'chunk_length': args.chunk_length,
        'num_parallel_graphs': args.num_parallel_graphs,
        'max_num_edges': args.max_num_edges,
        'dataset': args.dataset.lower(),
        'training_ratio': args.training_ratio,
        'seed': args.seed,
        'global_threshold': args.global_threshold,
    }


def prepare_bootstrap_clusters(config, output_path):
    """
    Write pre-computed bootstrap clusters to a file with proper tab-separated format.
    """
    dataset = config['dataset']
    if dataset not in BOOTSTRAP_CLUSTERS_DATA:
        raise ValueError(f"Unknown dataset: {dataset}. Must be one of: {list(BOOTSTRAP_CLUSTERS_DATA.keys())}")

    data = BOOTSTRAP_CLUSTERS_DATA[dataset]
    num_clusters, global_threshold = data["header"]
    clusters = data["clusters"]

    # Write file with explicit tab separation
    with open(output_path, 'w') as f:
        # Header line: num_clusters<tab>global_threshold
        f.write(f"{num_clusters}\t{global_threshold}\n")
        # Cluster lines: threshold<tab>gid<tab>gid<tab>...
        for threshold, graph_ids in clusters:
            line = str(threshold) + "\t" + "\t".join(str(gid) for gid in graph_ids)
            f.write(line + "\n")

    # Collect training graph IDs
    train_gids = set()
    for threshold, graph_ids in clusters:
        train_gids.update(graph_ids)

    print(f"Bootstrap clusters: {num_clusters} clusters, {len(train_gids)} training graphs")
    print(f"Global threshold: {global_threshold}")

    return train_gids


def get_ground_truth(dataset, num_graphs=600):
    """
    Get ground truth labels for graphs.
    Attack graphs (300-399) are labeled 1, others are 0.
    """
    if dataset == 'all':
        # All 600 graphs
        labels = [1 if i in ATTACK_GRAPH_IDS else 0 for i in range(num_graphs)]
    elif dataset == 'ydc':
        # YouTube (0-99), Download (400-499), CNN (500-599) + attacks (300-399)
        labels = []
        for i in range(num_graphs):
            scenario = i // 100
            if scenario in [0, 3, 4, 5]:  # YouTube, Attack, Download, CNN
                labels.append(1 if i in ATTACK_GRAPH_IDS else 0)
    elif dataset == 'gfc':
        # GMail (100-199), VGame (200-299), CNN (500-599) + attacks (300-399)
        labels = []
        for i in range(num_graphs):
            scenario = i // 100
            if scenario in [1, 2, 3, 5]:  # GMail, VGame, Attack, CNN
                labels.append(1 if i in ATTACK_GRAPH_IDS else 0)
    else:
        labels = [1 if i in ATTACK_GRAPH_IDS else 0 for i in range(num_graphs)]

    return labels


def parse_streamspot_output(output_text, num_graphs=600):
    """
    Parse StreamSpot output to extract anomaly scores.

    The output contains lines like:
    "Iterations N"
    followed by N pairs of lines:
    - anomaly scores (space-separated)
    - cluster assignments (space-separated)

    We use the final iteration's anomaly scores.
    """
    lines = output_text.strip().split('\n')

    # Find the "Iterations N" line
    iterations_idx = None
    num_iterations = 0
    for i, line in enumerate(lines):
        if line.startswith('Iterations'):
            iterations_idx = i
            num_iterations = int(line.split()[1])
            break

    if iterations_idx is None:
        raise ValueError("Could not find 'Iterations' line in StreamSpot output")

    # Get the last iteration's anomaly scores
    # Each iteration has 2 lines: scores and cluster assignments
    # Last iteration is at: iterations_idx + 1 + (num_iterations-1) * 2
    last_scores_idx = iterations_idx + 1 + (num_iterations - 1) * 2

    if last_scores_idx >= len(lines):
        raise ValueError(f"Output doesn't have enough lines. Expected scores at line {last_scores_idx}")

    scores_line = lines[last_scores_idx]
    scores = []
    for s in scores_line.strip().split():
        try:
            score = float(s)
            scores.append(score)
        except ValueError:
            continue

    if len(scores) != num_graphs:
        print(f"Warning: Expected {num_graphs} scores, got {len(scores)}")

    return scores


def main():
    args = parse_args()
    config = get_config_from_args(args)
    data_path = os.environ.get("DATA")
    exp_path = os.environ.get("EXP")

    print("StreamSpot Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Initialize resource tracking
    start_time = time.time()
    process = psutil.Process()
    peak_memory_mb = 0.0

    # Initialize GraFlag ResultWriter
    writer = ResultWriter()

    # Extract dataset name
    data_dir = Path(data_path)
    dataset_name = data_dir.name.replace('streamspot_', '')

    # Add initial metadata
    writer.add_metadata(
        method_name="streamspot",
        dataset=dataset_name,
        seed=config['seed'],
        chunk_length=config['chunk_length'],
        num_parallel_graphs=config['num_parallel_graphs'],
        dataset_subset=config['dataset'],
    )

    # Find edges file
    edges_file = None
    for candidate in ['all.tsv', 'edges.tsv', 'all.txt', 'edges.txt']:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            edges_file = candidate_path
            break

    if edges_file is None:
        # Try to find any tsv or txt file
        tsv_files = list(data_dir.glob('*.tsv'))
        txt_files = list(data_dir.glob('*.txt'))
        if tsv_files:
            edges_file = tsv_files[0]
        elif txt_files:
            edges_file = txt_files[0]

    if edges_file is None:
        raise FileNotFoundError(f"No edges file found in {data_dir}")

    print(f"\nUsing edges file: {edges_file}")

    # Count edges
    with open(edges_file, 'r') as f:
        num_edges = sum(1 for _ in f)
    print(f"Total edges: {num_edges:,}")

    # Prepare bootstrap clusters file
    bootstrap_file = Path(exp_path) / "bootstrap_clusters.txt"
    train_gids = prepare_bootstrap_clusters(config, bootstrap_file)

    # Build StreamSpot command
    streamspot_binary = "/app/sbustreamspot-core/streamspot"

    # Always pass --max-num-edges to avoid docopt's "inf" default which can't be parsed as long
    max_edges = config['max_num_edges'] if config['max_num_edges'] > 0 else num_edges

    cmd = [
        streamspot_binary,
        f"--edges={edges_file}",
        f"--bootstrap={bootstrap_file}",
        f"--chunk-length={config['chunk_length']}",
        f"--num-parallel-graphs={config['num_parallel_graphs']}",
        f"--max-num-edges={max_edges}",
        f"--dataset={config['dataset']}",
    ]

    print(f"\nRunning StreamSpot:")
    print(f"  Command: {' '.join(cmd)}")

    # Run StreamSpot
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            print(f"StreamSpot failed with return code {result.returncode}")
            print(f"STDERR: {stderr}")
            raise RuntimeError(f"StreamSpot execution failed: {stderr}")

        print("\nStreamSpot output (first 50 lines):")
        for line in stdout.split('\n')[:50]:
            print(f"  {line}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("StreamSpot execution timed out")

    # Track memory
    current_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, current_memory_mb)

    # Parse output to get anomaly scores
    print("\nParsing StreamSpot output...")

    num_graphs = 600
    anomaly_scores = parse_streamspot_output(stdout, num_graphs)

    print(f"Parsed {len(anomaly_scores)} anomaly scores")
    print(f"Score range: [{min(anomaly_scores):.4f}, {max(anomaly_scores):.4f}]")

    # Get ground truth
    ground_truth = get_ground_truth(config['dataset'], num_graphs)

    # Handle dataset subsets
    if config['dataset'] != 'all':
        # Filter to only include graphs in the subset
        if config['dataset'] == 'ydc':
            valid_scenarios = {0, 3, 4, 5}  # YouTube, Attack, Download, CNN
        elif config['dataset'] == 'gfc':
            valid_scenarios = {1, 2, 3, 5}  # GMail, VGame, Attack, CNN
        else:
            valid_scenarios = set(range(6))

        filtered_scores = []
        filtered_labels = []
        filtered_graph_ids = []

        for i in range(num_graphs):
            scenario = i // 100
            if scenario in valid_scenarios:
                filtered_scores.append(anomaly_scores[i])
                filtered_labels.append(1 if i in ATTACK_GRAPH_IDS else 0)
                filtered_graph_ids.append(i)

        anomaly_scores = filtered_scores
        ground_truth = filtered_labels
        graph_ids = filtered_graph_ids
    else:
        graph_ids = list(range(num_graphs))

    print(f"\nFinal dataset:")
    print(f"  Graphs: {len(anomaly_scores)}")
    print(f"  Anomalies: {sum(ground_truth)}")
    print(f"  Anomaly ratio: {sum(ground_truth)/len(ground_truth):.4f}")

    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(ground_truth, anomaly_scores) if len(set(ground_truth)) > 1 else 0.0
    print(f"  AUC-ROC: {auc:.4f}")

    # Save results using GRAPH_ANOMALY_SCORES format
    print("\nSaving results in GRAPH_ANOMALY_SCORES format...")
    writer.save_scores(
        result_type="GRAPH_ANOMALY_SCORES",
        scores=anomaly_scores,
        graph_ids=graph_ids,
        ground_truth=ground_truth,
    )

    # Calculate execution time
    end_time = time.time()
    exec_time_seconds = end_time - start_time

    # Final memory check
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    peak_memory_mb = max(peak_memory_mb, final_memory_mb)

    print(f"\nResource Usage:")
    print(f"  Total execution time: {exec_time_seconds:.2f}s")
    print(f"  Peak memory: {peak_memory_mb:.2f}MB")

    # Add final metadata
    writer.add_metadata(
        exp_name=os.path.basename(exp_path),
        method_name="streamspot",
        dataset=dataset_name,
        method_parameters=config,
        threshold=None,
        summary={
            "description": "StreamSpot: Graph-Based Anomaly Detection in System Provenance Data",
            "task": "graph_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "total_graphs": len(anomaly_scores),
                "n_anomalies": sum(ground_truth),
                "anomaly_ratio": float(sum(ground_truth) / len(ground_truth)),
                "total_edges": num_edges,
            },
            "detection_info": {
                "auc_roc": float(auc),
                "num_training_graphs": len(train_gids),
                "chunk_length": config['chunk_length'],
            },
            "scenarios": {
                "0-99": "YouTube (benign)",
                "100-199": "GMail (benign)",
                "200-299": "VGame (benign)",
                "300-399": "Drive-by-download (ATTACK)",
                "400-499": "Download (benign)",
                "500-599": "CNN (benign)",
            }
        },
    )

    # Add resource metrics (StreamSpot is CPU-only, no GPU)
    writer.add_resource_metrics(
        exec_time_ms=exec_time_seconds * 1000,
        peak_memory_mb=peak_memory_mb,
    )

    # Finalize results
    results_file = writer.finalize()
    print(f"\nResults saved to: {results_file}")
    print(f"AUC-ROC: {auc:.4f}")


if __name__ == "__main__":
    main()
