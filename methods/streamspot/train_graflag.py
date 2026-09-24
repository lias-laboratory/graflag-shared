"""
GraFlag-integrated wrapper for StreamSpot.
StreamSpot: Graph-Based Anomaly Detection in System Provenance Data (KDD 2016)

StreamSpot scores whole graphs: each graph is one system provenance trace from
a scenario (YouTube browsing, GMail, a drive-by-download attack, ...). The
upstream C++ binary consumes an edge stream and prints one anomaly score per
graph per iteration; this wrapper feeds it the paper's pre-computed bootstrap
clusters and publishes the final iteration's scores.

NOTE: the ground truth is the paper's, not the data's. `all.tsv` carries no
label column -- its six fields are source/type, destination/type, edge type and
graph id -- so which graphs are attacks cannot be read from the stream and is
taken from the published scenario layout instead. That makes the labels an
assumption about *which* stream is mounted, and SUPPORTED_DATASETS accepts any
`streamspot_*`, so the assumption is checked against the mounted file before
the binary runs: a stream whose graph ids are not exactly 0..599 is refused
rather than scored against labels invented for a different dataset. See
README.md, "Only the paper's dataset".
"""

import subprocess
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from sklearn.metrics import roc_auc_score

from graflag_runner import ResultWriter, info, params, paths, warning

# The StreamSpot binary, compiled from upstream's C++ in the image's first
# build stage. Checking for it turns a failed `make` into one clear message
# instead of a bare FileNotFoundError out of subprocess.
BINARY = Path("/app/sbustreamspot-core/streamspot")

# The paper's dataset: 600 graphs, 100 per scenario, scenario = graph_id // 100.
NUM_GRAPHS = 600
SCENARIOS = {
    0: "YouTube (benign)",
    1: "GMail (benign)",
    2: "VGame (benign)",
    3: "Drive-by-download (ATTACK)",
    4: "Download (benign)",
    5: "CNN (benign)",
}
#: Derived, not written down a second time. SCENARIOS already says which
#: scenario is the attack; a literal `range(300, 400)` beside it was a copy
#: that could disagree with the table it was supposed to follow.
ATTACK_SCENARIOS = frozenset(s for s, name in SCENARIOS.items() if "ATTACK" in name)
ATTACK_GRAPH_IDS = frozenset(
    gid for gid in range(NUM_GRAPHS) if gid // 100 in ATTACK_SCENARIOS)

# Pre-computed bootstrap clusters from the original paper
# Source: https://gist.github.com/emaadmanzoor/118846a642727a0bf704
# Format: first line = "num_clusters<TAB>global_threshold"
#         subsequent lines = "cluster_threshold<TAB>graph_id<TAB>graph_id..."
#
# `scenarios` is the subset each entry covers. It used to live in an if/elif
# chain in get_ground_truth() and in a second, separately written copy of the
# same chain in main() -- which is how the first copy came to be dead code
# without anyone noticing.
BOOTSTRAP_CLUSTERS_DATA = {
    "all": {
        "scenarios": frozenset(range(6)),
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
        "scenarios": frozenset({0, 3, 4, 5}),   # YouTube, Attack, Download, CNN
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
        "scenarios": frozenset({1, 2, 3, 5}),   # GMail, VGame, Attack, CNN
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


@dataclass
class Config:
    """The parameters this method accepts, and their defaults.

    Three of them are accepted and recorded but never read -- see INERT_PARAMS
    and README.md, "Parameters that do nothing".
    """

    chunk_length: int = 10
    num_parallel_graphs: int = 10
    max_num_edges: int = -1
    dataset: str = 'all'
    training_ratio: float = 0.5
    seed: int = 42
    global_threshold: float = 0.6

    def __post_init__(self):
        # get_config_from_args lowercased this; keep that. The membership
        # check is new: an unknown subset used to travel as far as
        # prepare_bootstrap_clusters, after the run had already started.
        self.dataset = self.dataset.lower()
        if self.dataset not in BOOTSTRAP_CLUSTERS_DATA:
            raise ValueError(
                f"Unknown dataset subset: {self.dataset!r}. "
                f"Supported: {', '.join(BOOTSTRAP_CLUSTERS_DATA)}"
            )


INERT_PARAMS = {
    'training_ratio': "the bootstrap clusters are the paper's pre-computed "
                      "ones, so no train/test split is drawn here",
    'global_threshold': "the bootstrap file carries the paper's own global "
                        "threshold (0.4823 for `all`), and that is the one "
                        "the binary reads",
    'seed': "nothing here draws a random number, and the binary takes no seed",
}


def warn_about_inert_params(config):
    """Say out loud which parameters were changed but will not be used."""
    defaults = {f.name: f.default for f in fields(Config)}
    for name, reason in INERT_PARAMS.items():
        if getattr(config, name) != defaults[name]:
            warning(f"[WARN] _{name.upper()} was set but has no effect: {reason}")


def prepare_bootstrap_clusters(config, output_path):
    """Write the paper's bootstrap clusters where the binary reads them.

    Returns the set of graph ids the clustering was bootstrapped from.
    """
    data = BOOTSTRAP_CLUSTERS_DATA[config.dataset]
    num_clusters, global_threshold = data["header"]
    clusters = data["clusters"]

    with open(output_path, 'w') as f:
        f.write(f"{num_clusters}\t{global_threshold}\n")
        for threshold, graph_ids in clusters:
            f.write("\t".join([str(threshold), *(str(g) for g in graph_ids)]) + "\n")

    train_gids = {gid for _, graph_ids in clusters for gid in graph_ids}
    info(f"[INFO] Bootstrap: {num_clusters} clusters, {len(train_gids)} training "
         f"graphs, global threshold {global_threshold}")
    return train_gids


def select_scenarios(dataset, scores):
    """Keep the graphs belonging to this subset: (graph_ids, scores, labels).

    This replaces two implementations of the same rule: get_ground_truth()
    built labels from one if/elif chain over scenarios, and main() then threw
    that result away and rebuilt it from a second chain -- except for
    `dataset='all'`, where the second block did not run at all. The two chains
    agreed, so the duplication was invisible; one of them drifting would not
    have been.
    """
    keep = BOOTSTRAP_CLUSTERS_DATA[dataset]["scenarios"]
    graph_ids = [gid for gid in range(NUM_GRAPHS) if gid // 100 in keep]
    return (
        graph_ids,
        [scores[gid] for gid in graph_ids],
        [1 if gid in ATTACK_GRAPH_IDS else 0 for gid in graph_ids],
    )


def find_edges_file(data_dir):
    """The edge stream in the dataset directory."""
    for candidate in ('all.tsv', 'edges.tsv', 'all.txt', 'edges.txt'):
        path = data_dir / candidate
        if path.exists():
            return path

    # sorted(), not glob()'s arbitrary order: which file was picked when a
    # directory held several used to depend on the filesystem.
    for pattern in ('*.tsv', '*.txt'):
        found = sorted(data_dir.glob(pattern))
        if found:
            return found[0]

    raise FileNotFoundError(f"No edges file (.tsv or .txt) found in {data_dir}")


def scan_edge_stream(path):
    """One pass over the stream: (edge count, the set of graph ids in it).

    The count was already being taken with `sum(1 for _ in f)`; reading the
    graph-id column in the same pass makes the check below free rather than a
    second walk over a 2 GB file.
    """
    graph_ids = set()
    num_edges = 0
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            num_edges += 1
            fields = line.split('\t')
            if len(fields) < 6:
                raise ValueError(
                    f"{path}:{lineno} has {len(fields)} tab-separated fields, "
                    f"expected 6 (source-id, source-type, destination-id, "
                    f"destination-type, edge-type, graph-id): {line[:120]!r}")
            try:
                graph_ids.add(int(fields[5]))
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno} has a non-numeric graph id "
                    f"{fields[5]!r}") from None
    return num_edges, graph_ids


def verify_paper_layout(graph_ids, path):
    """Refuse a stream the published labels do not describe.

    ATTACK_GRAPH_IDS comes from the paper's scenario layout, so it is only
    ground truth for the paper's own dataset. Nothing upstream of here
    establishes that: SUPPORTED_DATASETS is `streamspot_*`, and the runner
    matches that pattern against the folder name alone. Without this check the
    method would run happily on some other provenance stream and publish 100
    graphs labelled as attacks because of where they sat in the id space --
    a result, not an error, and one no downstream consumer could tell apart
    from a real one.
    """
    expected = set(range(NUM_GRAPHS))
    if graph_ids == expected:
        return
    missing = sorted(expected - graph_ids)
    extra = sorted(graph_ids - expected)
    raise ValueError(
        f"{path} is not the StreamSpot paper's dataset: its labels are the "
        f"published scenario layout ({NUM_GRAPHS} graphs, ids 0..{NUM_GRAPHS - 1}, "
        f"scenario = id // 100, attack = scenario "
        f"{sorted(ATTACK_SCENARIOS)}), and this stream holds "
        f"{len(graph_ids)} distinct graph ids"
        + (f"; missing {missing[:5]}{'...' if len(missing) > 5 else ''}" if missing else "")
        + (f"; unexpected {extra[:5]}{'...' if len(extra) > 5 else ''}" if extra else "")
        + ". Scoring it would publish labels invented for a different dataset.")


def parse_streamspot_output(output_text, num_graphs=NUM_GRAPHS):
    """The final iteration's per-graph anomaly scores.

    The binary prints "Iterations N" followed by N pairs of lines: the anomaly
    scores, then the cluster assignments.
    """
    lines = output_text.strip().split('\n')

    iterations_idx = None
    num_iterations = 0
    for i, line in enumerate(lines):
        if line.startswith('Iterations'):
            iterations_idx = i
            num_iterations = int(line.split()[1])
            break

    if iterations_idx is None:
        raise ValueError("Could not find 'Iterations' line in StreamSpot output")

    # Two lines per iteration, so the last scores line is:
    last_scores_idx = iterations_idx + 1 + (num_iterations - 1) * 2
    if last_scores_idx >= len(lines):
        raise ValueError(
            f"StreamSpot output ends at line {len(lines)}; the last "
            f"iteration's scores were expected at line {last_scores_idx}"
        )

    scores = []
    for token in lines[last_scores_idx].strip().split():
        try:
            scores.append(float(token))
        except ValueError:
            continue

    # This was a printed warning, which let a short score line through. The
    # run then died much later -- inside sklearn ("inconsistent numbers of
    # samples") or on an IndexError in the scenario filter -- and by then the
    # mismatched scores had already been handed to save_scores, which does not
    # check that scores and ground_truth are the same length.
    if len(scores) != num_graphs:
        raise ValueError(
            f"StreamSpot printed {len(scores)} scores, but this integration is "
            f"wired for {num_graphs} graphs. The edge stream is not the "
            f"paper's dataset -- see README.md, \"Only the paper's dataset\"."
        )

    return scores


def run_streamspot(config, edges_file, bootstrap_file, num_edges):
    """Run the binary and return its stdout."""
    if not BINARY.exists():
        raise FileNotFoundError(
            f"StreamSpot binary not found at {BINARY} -- the image's "
            f"`make optimized` step did not produce it"
        )

    # --max-num-edges is always passed: docopt's default is "inf", which the
    # binary cannot parse as a long.
    max_edges = config.max_num_edges if config.max_num_edges > 0 else num_edges

    cmd = [
        str(BINARY),
        f"--edges={edges_file}",
        f"--bootstrap={bootstrap_file}",
        f"--chunk-length={config.chunk_length}",
        f"--num-parallel-graphs={config.num_parallel_graphs}",
        f"--max-num-edges={max_edges}",
        f"--dataset={config.dataset}",
    ]
    info(f"[INFO] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        raise RuntimeError("StreamSpot execution timed out after 2 hours")

    if result.returncode != 0:
        print(f"StreamSpot stderr:\n{result.stderr}")
        raise RuntimeError(f"StreamSpot exited {result.returncode}")

    print("StreamSpot output (first 50 lines):")
    for line in result.stdout.split('\n')[:50]:
        print(f"  {line}")

    return result.stdout


def main():
    run = paths()
    config = Config(**params(Config))
    warn_about_inert_params(config)

    # The mount is `streamspot_all`, which the method knows as `all`.
    # config.dataset is a different thing: which scenario subset to score.
    dataset_name = run.dataset.replace('streamspot_', '')
    info(f"[INFO] StreamSpot on {dataset_name}: {asdict(config)}")

    writer = ResultWriter()

    edges_file = find_edges_file(run.data)
    num_edges, graph_ids_present = scan_edge_stream(edges_file)
    verify_paper_layout(graph_ids_present, edges_file)
    info(f"[INFO] Edge stream {edges_file}: {num_edges:,} edges over "
         f"{len(graph_ids_present)} graphs, matching the paper's layout")

    bootstrap_file = run.exp / "bootstrap_clusters.txt"
    train_gids = prepare_bootstrap_clusters(config, bootstrap_file)

    stdout = run_streamspot(config, edges_file, bootstrap_file, num_edges)
    graph_ids, scores, ground_truth = select_scenarios(
        config.dataset, parse_streamspot_output(stdout))

    num_anomalies = sum(ground_truth)
    anomaly_ratio = num_anomalies / len(ground_truth)
    print("\nFinal dataset:")
    print(f"  Graphs: {len(scores)}")
    print(f"  Anomalies: {num_anomalies} ({anomaly_ratio:.4f})")
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")

    # None, not 0.0: a single-class run has no AUC, and 0.0 reads as a
    # perfectly inverted classifier. The evaluator computes its own AUC from
    # the published scores either way; this is a summary field.
    auc = None
    if len(set(ground_truth)) > 1:
        auc = float(roc_auc_score(ground_truth, scores))
        print(f"  AUC-ROC: {auc:.4f}")
    else:
        warning("[WARN] Only one class present; no AUC computed")

    writer.save_scores(
        result_type="GRAPH_ANOMALY_SCORES",
        scores=scores,
        graph_ids=graph_ids,
        ground_truth=ground_truth,
    )

    # Execution time and memory are measured by graflag_runner around this
    # process and merged into metadata afterwards, so there is nothing to
    # record here -- see _merge_runtime_metadata in the runner.
    writer.add_metadata(
        exp_name=run.experiment,
        method_name="streamspot",
        dataset=dataset_name,
        method_parameters=asdict(config),
        threshold=None,
        summary={
            "description": "StreamSpot: Graph-Based Anomaly Detection in "
                           "System Provenance Data",
            "task": "graph_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "subset": config.dataset,
                "total_graphs": len(scores),
                "n_anomalies": num_anomalies,
                "anomaly_ratio": anomaly_ratio,
                "total_edges": num_edges,
                # Every graph of the selected scenarios, the benign ones the
                # bootstrap clusters were drawn from included.
                "scored_split": "all_graphs",
                "scored_samples": len(scores),
            },
            "detection_info": {
                "auc_roc": auc,
                "num_training_graphs": len(train_gids),
                "chunk_length": config.chunk_length,
            },
            "scenarios": {f"{s * 100}-{s * 100 + 99}": name
                          for s, name in SCENARIOS.items()},
        },
    )

    results_file = writer.finalize()
    info(f"[INFO] Results saved to {results_file}")


if __name__ == "__main__":
    main()
