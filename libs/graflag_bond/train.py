#!/usr/bin/env python3
"""
Generic PyGOD Bond Training Script

This script trains any PyGOD detector based on METHOD_NAME environment variable.
"""

import fnmatch
import os
import sys
import time
from pathlib import Path

# psutil and torch are gone with the sampling block they fed. The
# runner measures exec time, peak memory and peak GPU from outside the
# method and its numbers win (runner._merge_runtime_metadata), so the four
# point samples taken here were overwritten on every run -- and understated,
# because they saw only this process, not the tree. `time` stays: it measures
# the fit itself, which goes to training.csv as a spot metric, not to the
# metadata fields the runner owns.

# Import graflag_runner utilities
from graflag_runner import ResultWriter, seed_all
from graflag_runner import info, warning, error

# Import PyGOD
from pygod.utils import load_data

# Import bond utilities
from graflag_bond.detectors import BondDetector
from graflag_bond.utils import get_all_parameters


#: The seed when `_SEED` is not set. Every other integration script seeds
#: through `seed_all`; this one did not, so two runs of the same bond method
#: with the same configuration trained from different initial weights and
#: published different numbers, with nothing in the record to say why.
DEFAULT_SEED = 42


def run_seed() -> int:
    """The seed for this run: `_SEED` if set, else `DEFAULT_SEED`.

    Not a detector argument -- PyGOD's constructors take no seed, and
    `get_all_parameters` drops a key the signature does not accept -- so it
    is read here. A value that is not an integer raises instead of falling
    back to the default: a run that silently ignored its seed would look
    reproducible and not be.
    """
    raw = os.environ.get("_SEED", "").strip()
    if not raw:
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"_SEED={raw!r} is not an integer") from None


def load_graph_data(data_dir):
    """Load graph data from PyGOD datasets."""
    
    dataset_name = data_dir.name

    # The .env key is SUPPORTED_DATASETS and its values are fnmatch patterns
    # ("bond_*"), the same dialect tests/test_methods.py validates. Reading
    # SUPPORTED_DATA and comparing for equality made this warn on every run:
    # "".split(", ") is [""], which is truthy, and "bond_gen_100" is never
    # literally equal to "bond_*".
    patterns = [p.strip() for p in
                os.environ.get("SUPPORTED_DATASETS", "").split(",") if p.strip()]
    if patterns and not any(fnmatch.fnmatchcase(dataset_name, p) for p in patterns):
        warning(f"Dataset '{dataset_name}' may not be officially tested. "
                f"Supported: {', '.join(patterns)}")

    info(f"Loading dataset: {dataset_name} from {data_dir}")

    # Load data using PyGOD's load_data
    data = load_data(dataset_name, cache_dir=data_dir)
    info(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    
    return data


#: Detectors that are given the graph with repeated (u, v) pairs removed.
#:
#: GAAN trains with binary cross-entropy against the dense adjacency matrix
#: (pygod.nn.functional.double_recon_loss), which requires every entry to be
#: 0 or 1. to_dense_adj sums repeated pairs, and PyGOD's own inj_cora repeats
#: 6 of its 11,060 edges, so the matrix holds 2s and PyGOD 1.1.0 fails on its
#: own benchmark graph: "Assertion `target_val >= zero && target_val <= one`
#: failed" on a GPU, the same check as a ValueError on a CPU. Only GAAN gets
#: the deduplicated graph, so every other detector keeps running on exactly
#: the graph it ran on before; the count removed is published in results.json.
DEDUPLICATE_EDGES = frozenset({"gaan"})


def deduplicate_edges(data) -> int:
    """Drop repeated (u, v) pairs from ``data.edge_index`` in place.

    Keeps the first occurrence of each pair and the original order, so the
    only change is the repeats. (u, v) and (v, u) are different pairs: an
    undirected graph stored both ways is not touched. Raises on a graph with
    edge attributes or weights, whose duplicates would need a merge rule this
    function has no business choosing.

    Returns the number of edges removed.
    """
    import numpy as np
    for attr in ("edge_attr", "edge_weight"):
        if getattr(data, attr, None) is not None:
            raise ValueError(
                f"the graph has {attr}; repeated edges cannot be dropped without "
                f"deciding how to merge their {attr}")
    ei = data.edge_index
    pairs = ei.cpu().numpy()
    key = pairs[0].astype(np.int64) * int(data.num_nodes) + pairs[1].astype(np.int64)
    _, first = np.unique(key, return_index=True)
    keep = np.sort(first)
    removed = int(ei.size(1) - keep.size)
    if removed:
        import torch
        data.edge_index = ei[:, torch.as_tensor(keep, device=ei.device)]
    return removed


def prepare_graph(detector_name: str, data) -> dict:
    """Apply a detector's input adjustment, if it has one; describe what was done."""
    if detector_name.lower() not in DEDUPLICATE_EDGES:
        return {}
    in_file = int(data.edge_index.size(1))
    removed = deduplicate_edges(data)
    info(f"[INFO] {detector_name.upper()}: removed {removed} repeated edge(s) "
         f"of {in_file}; see DEDUPLICATE_EDGES")
    return {"num_edges_in_file": in_file, "duplicate_edges_removed": removed}


def describe_graph(data) -> dict:
    """The graph's size and labels, read before any detector touches it.

    PyGOD's detectors may change the Data object they are fitted on. On
    inj_cora, DMGD replaces data.x (1,433 feature columns become 2,708),
    GADNR adds 2,796 edges to edge_index and rewrites x, CARD adds 88 edges,
    and AdONE adds data.s. results.json used to be filled from the graph
    after fit(), so bond_dmgd reported `num_features: 2708`, bond_gadnr
    `num_edges: 13856` and bond_card `num_edges: 11148` for a Cora with 1,433
    and 11,060. Everything published about the input is taken from here
    instead, labels included.
    """
    labels = data.y.cpu() if hasattr(data.y, "cpu") else data.y
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "ground_truth": [1 if label != 0 else 0 for label in labels],
    }


def train_detector(method_name, data, exp_dir, writer, seed=DEFAULT_SEED):
    """Train PyGOD detector.

    Seeds immediately before the detector is built, so weight initialisation
    and every draw made during `fit` (dropout, negative sampling, PyGOD's own
    subsampling) start from `seed`.
    """
    
    # Get detector name and class dynamically
    detector_name = BondDetector.from_method_name(method_name)
    detector_class = BondDetector.get_detector_class(detector_name)
    
    # Get parameters from environment with type hints from detector signature
    params = get_all_parameters(detector_class)
    
    info("=" * 60)
    info(f"Training {detector_name.upper()} Model")
    info("=" * 60)
    
    # Log key parameters
    info(f"Detector: {detector_class.__name__}")
    if "hid_dim" in params:
        info(f"Architecture: hid_dim={params['hid_dim']}, num_layers={params.get('num_layers', 'N/A')}")
    if "epoch" in params:
        info(f"Training: epochs={params['epoch']}, lr={params.get('lr', 'N/A')}")
    if "contamination" in params:
        info(f"Contamination: {params['contamination']}")
    
    # Initialize model
    seed_all(seed)
    info(f"Initializing {detector_name.upper()} detector...")
    model = detector_class(**params)
    
    # Train model
    info("Starting training...")
    start_time = time.time()
    model.fit(data)
    training_time = time.time() - start_time
    
    # Log training metrics
    writer.spot("training", 
                epochs=params.get('epoch', 'N/A'),
                training_time_sec=training_time)

    info(f"Training completed in {training_time:.2f}s")

    return model


def save_results(model, graph, exp_dir, writer, method_name, dataset_name,
                 seed=DEFAULT_SEED, graph_changes=None):
    """Save results with metadata.

    Resource metrics are not written here. graflag_runner records exec time,
    peak memory and peak GPU for the whole process tree and overwrites
    whatever the method put in those fields.
    """
    info("=" * 60)
    info("Generating Results")
    info("=" * 60)

    # Get anomaly scores
    scores = model.decision_score_

    # `graph` is describe_graph()'s record, taken before fit(): labels
    # binarized, 0 = normal and any non-zero = anomaly.
    ground_truth = graph["ground_truth"]

    # Save results using ResultWriter
    writer.save_scores(
        result_type="NODE_ANOMALY_SCORES",
        scores=scores.tolist(),
        ground_truth=ground_truth,
        node_ids=list(range(len(scores)))
    )

    # Get detector info
    detector_name = BondDetector.from_method_name(method_name)
    detector_class = BondDetector.get_detector_class(detector_name)
    params = get_all_parameters(detector_class)

    # Convert params to JSON-safe strings (some values are Python types/functions)
    safe_params = {}
    for k, v in params.items():
        if callable(v) or isinstance(v, type):
            safe_params[k] = f"{v.__module__}.{v.__qualname__}" if hasattr(v, '__module__') else str(v)
        else:
            safe_params[k] = v
    safe_params["seed"] = seed

    # Add metadata
    writer.add_metadata(
        exp_name=os.path.basename(os.environ.get("EXP", "experiment")),
        method_name=method_name,
        dataset=dataset_name,
        method_parameters=safe_params,
        threshold=None,
        summary={
            "description": f"PyGOD {detector_name.upper()} detector",
            "task": "node_anomaly_detection",
            "dataset_info": {
                "name": dataset_name,
                "num_nodes": graph["num_nodes"],
                "num_edges": graph["num_edges"],
                "num_features": graph["num_features"],
                "num_anomalies": sum(ground_truth),
                # PyGOD's detectors fit on the graph they score: every node,
                # the BOND protocol.
                "scored_split": "all_nodes",
                "scored_samples": len(scores),
                # What the detector was given differs from the file only for
                # the detectors in DEDUPLICATE_EDGES, and then says by how much.
                **(graph_changes or {}),
            },
        },
    )

    # Finalize results
    writer.finalize()

    info(f"Results saved to {exp_dir}")


def main():
    # Get environment variables
    method_name = os.environ.get("METHOD_NAME")
    if not method_name:
        error("METHOD_NAME environment variable not set!")
        sys.exit(1)

    data_dir = Path(os.environ.get("DATA"))
    exp_dir = Path(os.environ.get("EXP"))

    info("=" * 60)
    info(f"PyGOD Bond: {method_name.upper()}")
    info("=" * 60)
    info(f"Dataset: {data_dir}")
    info(f"Output: {exp_dir}")
    info("")

    # Create experiment directory
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ResultWriter
    writer = ResultWriter()

    try:
        seed = run_seed()

        # Load data
        data = load_graph_data(data_dir)
        graph_changes = prepare_graph(BondDetector.from_method_name(method_name), data)
        graph = describe_graph(data)   # before fit(), which may change `data`

        # Train model
        model = train_detector(method_name, data, exp_dir, writer, seed=seed)

        # Save results with metadata
        save_results(model, graph, exp_dir, writer, method_name, data_dir.name,
                     seed=seed, graph_changes=graph_changes)

        info("")
        info("=" * 60)
        info(f"{method_name.upper()} execution completed successfully!")
        info("=" * 60)

    except Exception as e:
        error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
