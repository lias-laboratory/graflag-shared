"""ADA-GAD (AAAI 2024) under GraFlag.

Unsupervised node-level anomaly detection. Nothing here reimplements the
method: upstream's ``build_args()`` and ``main()`` are called as written, and
the only change to the clone is ``patches/expose-scores.patch``, which makes
``main()`` return the score vector it already computed instead of only
printing an AUC over it.

The dataset is handed to the method the way the method expects to find it
rather than the other way round. ADA-GAD calls ``pygod.utils.load_data(name)``
with no ``cache_dir``, which resolves to ``~/.pygod/data/<name>.pt``, so the
GraFlag dataset is staged there under the bare name upstream uses. Forcing it
through a GraFlag loader instead would mean patching upstream's data path,
which is a change to what runs.
"""

import os
import shutil
from pathlib import Path

from graflag_runner import (ResultWriter, apply_params, device, paths,
                            upstream, seed_all, info, warning)

UPSTREAM = upstream("src")          # raises if the clone is missing

# Appended, not inserted: the Dockerfile puts /app/src/pygod ahead of this on
# PYTHONPATH so `import pygod` finds the vendored package. Inserting /app/src
# at position 0 would shadow it with the __init__-less directory of the same
# name, and `pygod.utils` would stop existing.
import sys                          # noqa: E402
sys.path.append(str(UPSTREAM))
from main import build_args, main as ada_gad_main   # noqa: E402
from model.utils import load_best_configs                # noqa: E402


# Upstream names the BOND graphs without the bond_ prefix GraFlag stores them
# under. load_data() takes a bare name, so the staged file has to carry it.
def upstream_name(dataset_dir: Path) -> str:
    return dataset_dir.name[len("bond_"):] if dataset_dir.name.startswith("bond_") \
        else dataset_dir.name


def stage_dataset(dataset_dir: Path) -> str:
    """Put the graph where ``pygod.utils.load_data`` will look for it.

    Raises rather than falling back to a download: load_data() fetches an
    unpinned copy from the internet when the file is absent, which would make
    the run silently use a different graph from the one GraFlag mounted.
    """
    source = dataset_dir / f"{dataset_dir.name}.pt"
    if not source.is_file():
        candidates = sorted(dataset_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"no .pt graph in {dataset_dir}; contents: "
                f"{sorted(q.name for q in dataset_dir.iterdir())}")
        source = candidates[0]

    name = upstream_name(dataset_dir)
    cache = Path.home() / ".pygod" / "data"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}.pt"
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(source, target)
    info(f"[INFO] Staged {source} -> {target} (upstream name '{name}')")
    return name


def main():
    p = paths()
    dataset_name = stage_dataset(p.data)

    args = build_args()                       # upstream's own argparse defaults
    args.dataset = dataset_name

    # Load the authors' per-dataset configuration, which is what `--use_cfg`
    # does at main.py:264 and is how every number in the paper was produced.
    # Skipping it does not merely leave hyperparameters untuned: the argparse
    # default for --activation is "prelu", and the class at
    # pygod/models/basic_nn.py:218 resolves a string activation with
    # `eval('F.' + act)`, so "prelu" becomes the functional F.prelu and raises
    # `prelu() missing 1 required positional arguments: "weight"` on the first
    # forward pass. Every dataset section in the yml sets a different
    # activation, so the crash only ever appears when the config is bypassed.
    config_file = UPSTREAM / "config_ada-gad.yml"
    if not config_file.is_file():
        raise FileNotFoundError(f"upstream config missing: {config_file}")
    before = dict(vars(args))
    args = load_best_configs(args, str(config_file))
    if dict(vars(args)) == before:
        # load_best_configs logs "Best args not found" and returns args
        # unchanged when the dataset has no section. Running on the argparse
        # defaults instead would crash on prelu, and if it did not, the number
        # would not be the method's tuned performance.
        raise KeyError(
            f"'{dataset_name}' has no section in config_ada-gad.yml, so there "
            "is no configuration the authors tuned for it. Sections present "
            "are the BOND graphs; add one before running a new dataset.")

    # GraFlag's own overrides go on last so `--params` still wins over the
    # authors' defaults, which is the contract everywhere else.
    injected = apply_params(args, ignore={"gpu", "seed", "seeds"})

    # Replicate upstream's `if __name__ == "__main__"` block. main() is not
    # self-contained: two normalisations live in the entry-point block below
    # it, so calling main(args) directly skips them.
    #
    # The first one is not cosmetic. config_ada-gad.yml stores alpha_f for
    # several datasets as the *string* "None", which upstream converts to real
    # None before calling main(). Without it, alpha reaches
    # pygod/models/adanet.py:489 as "None" and `self.alpha * attribute_errors`
    # becomes str * Tensor, which fails with "only integer tensors of a single
    # element can be converted to an index" -- a message that points nowhere
    # near the yml it came from.
    if args.alpha_f == 'None':
        args.alpha_f = None
    if args.all_encoder_layers != 0:
        args.node_encoder_num_layers = args.all_encoder_layers
        args.edge_encoder_num_layers = args.all_encoder_layers
        args.subgraph_encoder_num_layers = args.all_encoder_layers

    seed = int(os.environ.get("_SEED", 42))
    seed_all(seed)
    dev = device()                            # honours _GPU, -1 means CPU
    args.device = dev.index if dev.type == "cuda" else -1
    n_seeds = int(os.environ.get("_SEEDS", 1))
    args.seeds = [seed + i for i in range(n_seeds)]

    info(f"[INFO] ADA-GAD on '{dataset_name}', device={dev}, seeds={args.seeds}")

    score_runs, auc_runs, graph = ada_gad_main(args)
    if not score_runs:
        raise RuntimeError(
            "ADA-GAD returned no scores. The patch that makes main() return "
            "them applied, or the build would have failed, so this is an "
            "upstream path that never reached god_evaluation().")

    import numpy as np
    scores = np.asarray(score_runs[0], dtype=float).ravel()
    truth = graph.y.cpu().numpy().ravel()
    # BOND labels encode outlier type in bits (1 contextual, 2 structural,
    # 3 both); the result contract wants binary.
    truth = (truth > 0).astype(int)

    if scores.shape[0] != truth.shape[0]:
        raise ValueError(
            f"ADA-GAD returned {scores.shape[0]} scores for a graph with "
            f"{truth.shape[0]} nodes -- publishing them would misalign every "
            "score with its label.")

    writer = ResultWriter()          # defaults to $EXP
    for i, auc in enumerate(auc_runs):
        writer.spot("runs", run=i, seed=args.seeds[i], upstream_auc=float(auc))

    writer.save_scores(result_type="NODE_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="ada_gad", summary={
        "dataset_info": {
            "dataset": dataset_name,
            "scored_split": "all_nodes",
            "scored_samples": int(scores.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            "upstream_auc_first_run": float(auc_runs[0]),
            "upstream_auc_mean": float(np.mean(auc_runs)),
            "runs": len(auc_runs),
            "seeds": args.seeds,
        },
        "injected": injected,
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} node scores "
         f"({int(truth.sum())} anomalies), upstream AUC "
         f"{auc_runs[0]:.4f}")


if __name__ == "__main__":
    main()
