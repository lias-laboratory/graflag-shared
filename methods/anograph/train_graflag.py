"""
GraFlag-integrated wrapper for AnoGraph.
AnoGraph: Sketch-Based Anomaly Detection in Streaming Graphs (KDD 2023)

AnoGraph is a sketch-based method that detects anomalies in streaming graphs
using count-min sketch extensions for preserving dense subgraph structures.

The published scores are upstream's own. The binary writes them to
`../results/<algorithm>_<dataset>[_<tw>_<et>]_score.csv`, one `score label`
pair per line, which is the same file upstream's metrics.py reads to produce
the AUCs in the paper.
"""

import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from graflag_runner import (
    ResultWriter, info, params, paths, seed_all, snapshot_files,
    split_test_edges, warning,
)

# The cloned AnoGraph checkout. Its C++ is built at image build time, and the
# binary reads its input from `../data` relative to the code directory, so both
# paths are fixed by upstream's layout rather than chosen here.
SRC = Path("/app/src")
CODE_DIR = SRC / "code"
BINARY = CODE_DIR / "main"
BINARY_DATA_DIR = SRC / "data"

# The directory the binary writes into. utils.hpp fixes it as "../results/"
# relative to the code directory, and nothing creates it: `fopen(path, "w")`
# in a directory that does not exist returns NULL, and writeScoresAndLabels
# fprintf()s to it without checking.
RESULTS_DIR = SRC / "results"

# Algorithm -> the binary's subcommand, what it scores, and the positional
# arguments it takes in order, as upstream's demo.sh spells them. Adding an
# algorithm is a row.
#
# `scores` is the granularity, and it decides both the result type and which
# file name upstream writes: the two graph algorithms aggregate the stream into
# time windows and put the window parameters in the name (anograph.cpp:46),
# the two edge algorithms score every edge and do not (anoedgeglobal.cpp:48).
ALGORITHMS = {
    'anograph': ('anograph', 'graph',
                 lambda c: [c.time_window, c.edge_threshold, c.num_rows, c.num_buckets]),
    'anographk': ('anograph_k', 'graph',
                  lambda c: [c.time_window, c.edge_threshold, c.num_rows, c.num_buckets, c.k]),
    'anoedgeg': ('anoedge_g', 'edge',
                 lambda c: [c.num_rows, c.num_buckets, c.threshold]),
    'anoedgel': ('anoedge_l', 'edge',
                 lambda c: [c.num_rows, c.num_buckets, c.threshold]),
}

#: What each granularity is called in results.json.
RESULT_TYPES = {
    'graph': 'GRAPH_ANOMALY_SCORES',
    'edge': 'EDGE_STREAM_ANOMALY_SCORES',
}


@dataclass
class Config:
    """The parameters this method accepts, and their defaults."""

    algorithm: str = 'anograph'
    num_rows: int = 2
    num_buckets: int = 1024
    time_window: int = 60
    edge_threshold: int = 100
    k: int = 5                  # anographk only
    threshold: float = 0.9      # anoedgeg / anoedgel only
    seed: int = 42

    def __post_init__(self):
        # argparse enforced this with choices=; without the check a typo would
        # reach the binary and come back as an unexplained non-zero exit.
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm!r}. "
                f"Supported: {', '.join(ALGORITHMS)}"
            )


def load_anograph_native_data(data_path):
    """
    Load data in native AnoGraph format (Data.csv + Label.csv).
    """
    data_dir = Path(data_path)

    data_file = data_dir / 'Data.csv'
    label_file = data_dir / 'Label.csv'

    if not data_file.exists() or not label_file.exists():
        return None, None

    print(f"Loading native AnoGraph format from {data_dir}")

    # Load data: src, dst, timestamp
    data = pd.read_csv(data_file, header=None, names=['src', 'dst', 'timestamp'])

    # Load labels
    labels = pd.read_csv(label_file, header=None, names=['label'])

    return data, labels['label'].values


def convert_snapshot_to_anograph(data_path, output_dir, config):
    """
    Convert GraFlag snapshot format to AnoGraph format.

    Returns (data_df, labels) with one row per edge of the accumulated
    snapshots plus any injected anomaly not already present, labelled 1.
    """
    data_dir = Path(data_path)
    output_dir = Path(output_dir)

    graph_file, split_file = snapshot_files(data_dir)
    if graph_file is None or split_file is None:
        return None, None

    print("Converting snapshot format to AnoGraph format...")
    print(f"  Graph file: {graph_file}")
    print(f"  Split file: {split_file}")

    # Load graph snapshots
    net = np.load(graph_file, allow_pickle=True)
    split_data = np.load(split_file, allow_pickle=True)

    # Get dimensions
    if net.dtype == object:
        num_snapshots = len(net)
    else:
        num_snapshots = net.shape[0]

    # Extract edges with timestamps from snapshots
    edges = []
    for t in range(num_snapshots):
        if net.dtype == object:
            adj = net[t].toarray() if hasattr(net[t], 'toarray') else net[t]
        else:
            adj = net[t]

        # Find edges in this snapshot
        rows, cols = np.where(adj > 0)
        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicates for undirected
                edges.append([int(i), int(j), t])

    # Built through an int array so an empty graph still yields int columns --
    # object-dtype columns would fail the merge below.
    data_df = pd.DataFrame(np.array(edges, dtype=int).reshape(-1, 3),
                           columns=['src', 'dst', 'timestamp'])

    # The injected anomalies are the split's test_neg -- see split_test_edges
    # for why that is the positive class here. test_pos needs no marking: every
    # edge already in data_df starts at 0.
    test_edges, test_times, test_labels = split_test_edges(
        split_data, default_snapshot=num_snapshots - 1)
    anomaly_rows = [(src, dst, t) for (src, dst), t, label
                    in zip(test_edges, test_times, test_labels) if label]
    # Deduplicated: a repeated (src, dst, timestamp) would otherwise duplicate
    # rows in the left merge below and leave labels longer than data_df.
    anomalies = pd.DataFrame(
        np.array(anomaly_rows, dtype=int).reshape(-1, 3),
        columns=['src', 'dst', 'timestamp']).drop_duplicates()

    # One merge, not a full-DataFrame boolean mask per anomaly: on the larger
    # snapshot sets that loop was the slowest part of the conversion.
    marked = data_df.merge(anomalies.assign(_anomaly=1), how='left',
                           on=['src', 'dst', 'timestamp'])
    labels = marked['_anomaly'].fillna(0).to_numpy()

    # An injected anomaly is usually a non-edge, so it is absent from the
    # accumulated snapshots and has to be appended.
    missing = anomalies.merge(data_df, how='left', on=['src', 'dst', 'timestamp'],
                              indicator=True)
    missing = missing[missing['_merge'] == 'left_only'][['src', 'dst', 'timestamp']]
    if len(missing):
        data_df = pd.concat([data_df, missing], ignore_index=True)
        labels = np.append(labels, np.ones(len(missing)))

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(output_dir / 'Data.csv', index=False, header=False)
    pd.DataFrame(labels.astype(int)).to_csv(output_dir / 'Label.csv', index=False, header=False)

    print(f"  Created {len(data_df)} edges, {int(labels.sum())} anomalies")

    return data_df, labels


def setup_anograph_data(data_df, labels, dataset_name):
    """Write the input where upstream's binary actually looks for it.

    ReadUtils reads `../data/<name>/Data.csv` and `../data/<name>/Label.csv`
    relative to the code directory (utils.cpp:26,43,57). This used to write
    `../data/<name>.csv` and `../data/<name>_label.csv` instead -- paths
    upstream never opens.

    Nothing reported that. ReadUtils answers a NULL from fopen with `exit(0)`,
    so a run that found no input at all ended in success with an empty results
    directory, and the integration went on to publish a locally computed score
    in place of the one the binary never wrote.
    """
    data_dir = BINARY_DATA_DIR / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / 'Data.csv'
    label_file = data_dir / 'Label.csv'
    data_df.to_csv(data_file, index=False, header=False)
    pd.DataFrame(np.asarray(labels).astype(int)).to_csv(
        label_file, index=False, header=False)

    print(f"Data setup at {data_dir}:")
    print(f"  Data file:  {data_file} ({len(data_df)} edges)")
    print(f"  Label file: {label_file}")


def check_stream_is_window_aligned(data_df, config):
    """The two graph algorithms need a time-ordered stream starting in bin 0.

    loadGraphData walks the file once and closes the current graph whenever
    `t/time_window` differs from the previous record's, starting from
    `cur_time = 0` (utils.cpp:79-90). process_data.py instead groups by the
    distinct values of the same quotient. The two agree on one graph per window
    only when the stream is sorted by time and its first record falls in bin 0
    -- otherwise loadGraphData emits a leading empty graph, or re-opens a
    window it already closed, and the counts diverge.

    Upstream's own assert catches the divergence, as SIGABRT with no message.
    Saying it here costs one pass over a column and names the cause.
    """
    times = data_df['timestamp'].to_numpy()
    if len(times) == 0:
        raise ValueError("the edge stream is empty")
    if not np.all(np.diff(times) >= 0):
        raise ValueError(
            "the edge stream is not sorted by timestamp, which the graph "
            "algorithms require: upstream's loadGraphData would split it into "
            "more windows than process_data.py labels. Sort Data.csv by its "
            "third column, or use an anoedge* algorithm, which scores edges "
            "and does not window the stream.")
    first_bin = int(times[0]) // config.time_window
    if first_bin != 0:
        raise ValueError(
            f"the first edge is at t={times[0]}, which falls in time bin "
            f"{first_bin} rather than 0. loadGraphData starts at bin 0, so it "
            f"would close an empty leading graph and end up with one more "
            f"graph than there are labels. Shift the timestamps so the stream "
            f"starts below _TIME_WINDOW={config.time_window}.")


def prepare_graph_labels(dataset_name, config):
    """Run upstream's process_data.py to label each time window.

    The graph algorithms read `Label_<tw>_<et>.csv`, which is not shipped with
    any dataset: demo.sh generates it per (time_window, edge_threshold) before
    every run. It marks a window anomalous when the number of anomalous edges
    inside it reaches edge_threshold, so the same stream yields different
    ground truth at different settings -- which is why this is regenerated
    here rather than cached.
    """
    script = CODE_DIR / "process_data.py"
    if not script.exists():
        raise FileNotFoundError(f"upstream's process_data.py is missing at {script}")

    cmd = [sys.executable, script.name, dataset_name,
           str(config.time_window), str(config.edge_threshold)]
    print(f"Preprocessing graph labels: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))
    if result.returncode != 0:
        raise RuntimeError(
            f"process_data.py exited {result.returncode}\n{result.stdout}\n{result.stderr}")

    label_file = (BINARY_DATA_DIR / dataset_name /
                  f"Label_{config.time_window}_{config.edge_threshold}.csv")
    if not label_file.exists():
        raise FileNotFoundError(
            f"process_data.py reported success but wrote no {label_file}")

    window_labels = np.loadtxt(label_file, dtype=int, ndmin=1)
    positives = int(window_labels.sum())
    print(f"  {len(window_labels)} time windows, {positives} anomalous "
          f"(edge_threshold={config.edge_threshold})")

    # Fail here rather than publish a result nothing can be computed from. A
    # window is anomalous only once it holds edge_threshold anomalous edges, so
    # a stream that is short, or finely windowed, or simply cleaner than the
    # paper's, produces a single class -- and a single-class ground truth makes
    # every metric in RESULTS_STANDARD.md undefined.
    if len(window_labels) < 2 or positives in (0, len(window_labels)):
        raise ValueError(
            f"_TIME_WINDOW={config.time_window} and "
            f"_EDGE_THRESHOLD={config.edge_threshold} leave {len(window_labels)} "
            f"window(s) of which {positives} are anomalous, so the ground truth "
            f"has one class and no metric is defined over it. A window is "
            f"labelled anomalous only once it contains {config.edge_threshold} "
            f"anomalous edges: lower _EDGE_THRESHOLD, or lower _TIME_WINDOW so "
            f"the anomalies concentrate into fewer windows.")
    return window_labels


def score_file_for(dataset_name, config):
    """Where the binary will write this run's scores.

    Two shapes, because upstream names them differently: the graph algorithms
    put the window parameters in the file name (anograph.cpp:46) and the edge
    algorithms do not (anoedgeglobal.cpp:48).
    """
    subcommand, granularity, _ = ALGORITHMS[config.algorithm]
    if granularity == 'graph':
        stem = (f"{subcommand}_{dataset_name}_"
                f"{config.time_window}_{config.edge_threshold}")
    else:
        stem = f"{subcommand}_{dataset_name}"
    return RESULTS_DIR / f"{stem}_score.csv"


def run_anograph(dataset_name, config):
    """Run the binary and return the (scores, labels) it wrote.

    The exit status is not sufficient evidence that it ran: ReadUtils exits 0
    when it cannot open its input. So the score file is removed first and its
    reappearance is what counts as success -- otherwise a failed run would
    read back the previous run's file and publish it as this one's result.
    """
    if not BINARY.exists():
        raise FileNotFoundError(
            f"AnoGraph binary not found at {BINARY} -- the image's `make` step "
            f"did not produce it"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    score_file = score_file_for(dataset_name, config)
    if score_file.exists():
        score_file.unlink()

    subcommand, _, positional = ALGORITHMS[config.algorithm]
    cmd = [str(BINARY), subcommand, dataset_name,
           *(str(value) for value in positional(config))]

    print(f"Running AnoGraph: {' '.join(cmd)}")
    print(f"Working directory: {CODE_DIR}")

    # Run from the code directory (so ../data and ../results resolve correctly)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))

    print(f"AnoGraph stdout:\n{result.stdout}")
    if result.stderr:
        print(f"AnoGraph stderr:\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(
            f"the AnoGraph binary exited {result.returncode}. A negative code "
            f"is a signal: -6 is the assert in loadGraphData, which fires when "
            f"the stream splits into a different number of graphs than "
            f"process_data.py labelled.")

    if not score_file.exists():
        raise FileNotFoundError(
            f"the binary exited 0 but wrote no {score_file}. Its readers call "
            f"exit(0) when they cannot open their input, so this is what a "
            f"missing or misnamed input file looks like: check "
            f"{BINARY_DATA_DIR / dataset_name}.")

    return read_scores_and_labels(score_file)


def read_scores_and_labels(score_file):
    """Parse upstream's `"%.4f %d\n"` score file (utils.cpp:8-15).

    This is the same file metrics.py reads to produce the AUCs upstream
    publishes, so the numbers here are the ones the paper reports on.
    """
    table = np.loadtxt(score_file, ndmin=2)
    if table.shape[1] != 2:
        raise ValueError(
            f"{score_file} has {table.shape[1]} columns, expected 2 "
            f"(score, label)")
    return table[:, 0], table[:, 1].astype(int)


def main():
    run = paths()
    config = Config(**params(Config))
    dataset_name = run.dataset

    info(f"[INFO] AnoGraph on {dataset_name}: {asdict(config)}")

    seed_all(config.seed)
    writer = ResultWriter()

    print(f"\nLoading data from {run.data}...")

    # Native AnoGraph format (Data.csv + Label.csv) if the dataset ships it,
    # otherwise convert a GraFlag snapshot series into it.
    data_df, labels = load_anograph_native_data(run.data)
    if data_df is None:
        work_dir = run.exp / 'work'
        work_dir.mkdir(parents=True, exist_ok=True)
        data_df, labels = convert_snapshot_to_anograph(run.data, work_dir, config)

    if data_df is None:
        raise ValueError(f"Could not load data from {run.data}")

    num_edges = len(data_df)
    num_anomalies = int(labels.sum()) if labels is not None else 0
    print(f"Loaded {num_edges} edges, {num_anomalies} anomalies")

    # The binary addresses its input by name under ../data, so the name is an
    # identifier for the binary, not the GraFlag dataset.
    binary_dataset_name = "graflag_data"
    setup_anograph_data(data_df, labels, binary_dataset_name)

    subcommand, granularity, _ = ALGORITHMS[config.algorithm]

    window_labels = None
    if granularity == 'graph':
        check_stream_is_window_aligned(data_df, config)
        window_labels = prepare_graph_labels(binary_dataset_name, config)

    print(f"\nRunning {config.algorithm}...")
    scores, ground_truth = run_anograph(binary_dataset_name, config)

    # The binary asserts these agree before it writes anything, but the assert
    # is compiled out under -DNDEBUG and this is the claim the published
    # result rests on, so it is checked here too.
    expected = len(window_labels) if granularity == 'graph' else len(data_df)
    if len(scores) != expected:
        raise RuntimeError(
            f"{config.algorithm} returned {len(scores)} scores for "
            f"{expected} {'time windows' if granularity == 'graph' else 'edges'}")

    published_auc = None
    if len(np.unique(ground_truth)) > 1:
        published_auc = float(roc_auc_score(ground_truth, scores))
        print(f"AUC of the published scores: {published_auc:.4f}")
    else:
        warning("[WARN] the ground truth upstream wrote has a single class; "
                "no AUC is defined over it")

    print(f"\nTotal predictions: {len(scores)}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\nSaving results...")
    extra = {}
    if granularity == 'edge':
        # One score per edge, in stream order, so the edges and timestamps
        # from the input line up with them row for row.
        extra = {
            'edges': data_df[['src', 'dst']].values.tolist(),
            'timestamps': data_df['timestamp'].values.tolist(),
        }
    else:
        # One score per time window. The window index is the identifier;
        # there is no edge to attach it to.
        extra = {'graph_ids': list(range(len(scores)))}

    writer.save_scores(
        result_type=RESULT_TYPES[granularity],
        scores=scores.tolist(),
        ground_truth=ground_truth.tolist(),
        **extra,
    )

    # Execution time and memory are measured by graflag_runner around this
    # process and merged into metadata afterwards, so there is nothing to
    # record here -- see _merge_runtime_metadata in the runner.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="anograph",
        dataset=dataset_name,
        method_parameters=asdict(config),
        summary={
            "description": "AnoGraph: Sketch-Based Anomaly Detection in Streaming Graphs (KDD 2023)",
            "task": ("graph_anomaly_detection" if granularity == 'graph'
                     else "edge_anomaly_detection"),
            "scores_are": (
                f"upstream's own, read back from "
                f"{score_file_for(binary_dataset_name, config).name}"),
            "dataset_info": {
                "name": dataset_name,
                "num_edges": num_edges,
                "num_anomalous_edges": num_anomalies,
                # The whole stream: AnoGraph is an online sketch method that
                # fits nothing, so there is no training split to leave out.
                "scored_split": ("all_windows" if granularity == 'graph'
                                 else "all_edges"),
                "scored_samples": len(scores),
                "scored_unit": "time_window" if granularity == 'graph' else "edge",
            },
            "results": {
                "auc_published_scores": published_auc,
            },
            "algorithm_info": {
                "name": config.algorithm,
                "subcommand": subcommand,
                "num_rows": config.num_rows,
                "num_buckets": config.num_buckets,
                "time_window": config.time_window,
                "edge_threshold": config.edge_threshold,
            },
        },
    )

    info(f"[OK] Results written to {writer.finalize()}")


if __name__ == "__main__":
    main()
