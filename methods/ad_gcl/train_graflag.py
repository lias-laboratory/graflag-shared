"""AD-GCL (AAAI 2025) under GraFlag.

Unsupervised node-level anomaly detection by graph contrastive learning,
corrected for the structural imbalance between high- and low-degree nodes.
Upstream's ``run.py`` is executed as written; the only change to the clone is
``patches/export-scores.patch``, which writes out the score vector run.py
already computes.

The dataset is used in the form the method reads. AD-GCL's ``load_mat()``
opens ``./Data/<name>.mat``, and GraFlag stores these datasets as exactly that
``.mat`` -- so the file is placed where upstream looks and nothing is
converted.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graflag_runner import (ResultWriter, params, device, paths, upstream,
                            seed_all, info, warning)

UPSTREAM = upstream("src")

# Upstream gives no argparse default for --lr or --num_epoch: both are
# `type=` with no `default=`, so they arrive as None and the run fails inside
# the optimiser. The values below are the ones its README publishes, one
# command per dataset, and they are what every number in the paper came from.
# Running without them is not "untuned", it is broken.
AUTHOR_CONFIG = {
    "cora":       {"lr": 5e-3, "num_epoch": 200, "threshold": 7},
    "citeseer":   {"lr": 3e-3, "num_epoch": 200, "threshold": 6},
    "pubmed":     {"lr": 4e-3, "num_epoch": 100, "threshold": 8},
    "bitcoinotc": {"lr": 4e-4, "num_epoch": 100, "threshold": 8},
    "bitotc":     {"lr": 5e-4, "num_epoch": 100, "threshold": 7},
    "bitalpha":   {"lr": 5e-3, "num_epoch": 100, "threshold": 8},
}


@dataclass
class Config:
    embedding_dim: int = 64
    drop_prob: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 300
    subgraph_size: int = 4
    readout: str = "avg"
    negsamp_ratio: int = 1
    degree: int = 6
    auc_test_rounds: int = 256
    seed: int = 1
    # Overridable, but defaulted from AUTHOR_CONFIG when left unset.
    lr: float = None
    num_epoch: int = None
    threshold: int = None


def stage_dataset(data_dir: Path) -> str:
    """Put the mounted .mat where upstream's load_mat() opens it.

    ``load_mat`` does ``sio.loadmat("./Data/{}.mat".format(dataset))`` --
    relative to the working directory, which is the clone root. The dataset
    is copied rather than symlinked because the clone lives in the image and
    the share is a different mount.
    """
    name = data_dir.name[len("gad_"):] if data_dir.name.startswith("gad_") \
        else data_dir.name
    source = data_dir / f"{name}.mat"
    if not source.is_file():
        found = sorted(data_dir.glob("*.mat"))
        if not found:
            raise FileNotFoundError(
                f"no .mat in {data_dir}; AD-GCL reads Network/Attributes/Label "
                f"from one. Contents: {sorted(q.name for q in data_dir.iterdir())}")
        source, name = found[0], found[0].stem

    target_dir = UPSTREAM / "Data"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / f"{name}.mat")
    info(f"[INFO] Staged {source} -> {target_dir / f'{name}.mat'}")
    return name


def main():
    import numpy as np

    config = Config(**params(Config))
    p = paths()
    dataset = stage_dataset(p.data)

    if dataset not in AUTHOR_CONFIG:
        raise KeyError(
            f"'{dataset}' is not one of the six datasets AD-GCL publishes "
            f"hyperparameters for ({', '.join(sorted(AUTHOR_CONFIG))}). "
            "--lr and --num_epoch have no argparse defaults, so there is "
            "nothing to fall back on but None.")
    authors = AUTHOR_CONFIG[dataset]
    lr = config.lr if config.lr is not None else authors["lr"]
    num_epoch = config.num_epoch if config.num_epoch is not None else authors["num_epoch"]
    threshold = config.threshold if config.threshold is not None else authors["threshold"]
    info(f"[INFO] {dataset}: lr={lr} num_epoch={num_epoch} threshold={threshold} "
         f"(authors' values unless overridden by --params)")

    seed_all(config.seed)
    dev = device()

    exp = Path(os.environ["EXP"])
    score_out = exp / "ad_gcl_scores.txt"
    env = dict(os.environ, GRAFLAG_SCORE_OUT=str(score_out))

    command = [
        "python3", "AD-GCL/run.py",
        "--dataset", dataset,
        "--lr", str(lr),
        "--num_epoch", str(num_epoch),
        "--threshold", str(threshold),
        "--embedding_dim", str(config.embedding_dim),
        "--drop_prob", str(config.drop_prob),
        "--weight_decay", str(config.weight_decay),
        "--batch_size", str(config.batch_size),
        "--subgraph_size", str(config.subgraph_size),
        "--readout", config.readout,
        "--negsamp_ratio", str(config.negsamp_ratio),
        "--degree", str(config.degree),
        "--auc_test_rounds", str(config.auc_test_rounds),
        "--seed", str(config.seed),
        "--gpu_id", str(dev.index if dev.type == "cuda" else 0),
    ]
    info(f"[INFO] AD-GCL: {' '.join(command[1:])}")
    result = subprocess.run(command, cwd=str(UPSTREAM), env=env,
                            capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    if stdout:
        info(stdout[-3000:])
    if result.returncode != 0:
        raise RuntimeError(
            f"AD-GCL exited {result.returncode}: "
            f"{(result.stderr or '').strip()[-3000:] or 'no stderr'}")

    if not score_out.is_file():
        raise FileNotFoundError(
            f"AD-GCL exited 0 but wrote no {score_out}. export-scores.patch "
            "writes it right after the AUC is computed, and `git apply "
            "--verbose` would have failed the build if it had not applied.")

    scores = np.loadtxt(score_out, dtype=float).ravel()
    truth = np.loadtxt(str(score_out) + ".label", dtype=float).ravel()
    truth = (truth > 0).astype(int)
    if scores.shape[0] != truth.shape[0]:
        raise ValueError(
            f"AD-GCL wrote {scores.shape[0]} scores and {truth.shape[0]} "
            "labels; publishing them would misalign every score.")

    method_auc = None
    for line in stdout.splitlines():
        if line.startswith("AUC:"):
            method_auc = float(line.split(":", 1)[1])

    writer = ResultWriter()
    writer.save_scores(result_type="NODE_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="ad_gcl", summary={
        "dataset_info": {
            "dataset": dataset,
            "scored_split": "all_nodes",
            "scored_samples": int(scores.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            "lr": lr, "num_epoch": num_epoch, "threshold": threshold,
            "auc_test_rounds": config.auc_test_rounds,
            "seed": config.seed,
            "auc_roc": method_auc,
            "hyperparameters_from": "upstream README" if config.lr is None else "--params",
        },
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} node scores "
         f"({int(truth.sum())} anomalous), method AUC {method_auc}")


if __name__ == "__main__":
    main()
