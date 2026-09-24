"""HUGE-GAD (AAAI 2025) under GraFlag.

Label-free unsupervised graph fraud detection. Upstream's ``main.py`` runs as
written; the only change to the clone is ``patches/export-scores.patch``, which
writes out the score vector main.py already computes.

Two kinds of dataset reach the same method here. HUGE-GAD's ``load_mat()``
opens ``./datasets/<name>.mat``, so a GraFlag dataset stored as ``.mat``
(``gad_*``) is copied straight through, and one stored as a PyG ``.pt``
(``bond_*``) is rendered into that form with ``graflag_runner.write_mat``.
Neither case patches the authors' loader.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graflag_runner import (ResultWriter, params, device, paths, upstream,
                            seed_all, load_attributed_graph, write_mat,
                            info, warning)

UPSTREAM = upstream("src")


@dataclass
class Config:
    lr: float = 5e-4
    epoch: int = 300
    kd_param: float = 0.5
    weight_decay: float = 0.0
    hidden: int = 128
    batch_size_heterophily: int = 8192
    batch_size_sampling: int = 8192
    heterophily: str = "ours"
    seed: int = 0


def stage_dataset(data_dir: Path) -> str:
    """Put the graph where upstream's load_mat() opens it.

    ``load_mat`` does ``sio.loadmat("./datasets/<name>.mat")``, relative to the
    clone root. A dataset already stored as ``.mat`` is copied unchanged; one
    stored as a PyG ``.pt`` is rendered by ``write_mat``, which is the whole
    reason that helper exists -- it means the bond_* graphs are available to a
    method that has never heard of PyG, without a second copy of them being
    stored anywhere.
    """
    target_dir = UPSTREAM / "datasets"
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(data_dir.glob("*.mat"))
    if existing:
        source = existing[0]
        name = source.stem
        shutil.copy2(source, target_dir / f"{name}.mat")
        info(f"[INFO] Copied {source} -> {target_dir / f'{name}.mat'} (already .mat)")
        return name

    # No .mat: render one from the canonical PyG form.
    data = load_attributed_graph(data_dir)
    name = data_dir.name
    written = write_mat(data, target_dir / f"{name}.mat")
    info(f"[INFO] Rendered {data_dir.name} ({data.num_nodes} nodes) -> {written}")
    return name


def main():
    import numpy as np

    config = Config(**params(Config))
    seed_all(config.seed)
    dev = device()

    p = paths()
    dataset = stage_dataset(p.data)

    # load_dataset() routes the four names upstream ships through load_mat and
    # raises "Unimplemented dataset" for anything else, so a rendered graph
    # has to arrive under one of them.
    if dataset not in ("Amazon", "Facebook", "Reddit", "YelpChi"):
        warning(
            f"[WARN] '{dataset}' is not one of upstream's four .mat names "
            "(Amazon, Facebook, Reddit, YelpChi); modules/utils.load_dataset "
            "raises for anything else. Renaming the staged file to one of "
            "them is the supported route -- see README.md.")

    exp = Path(os.environ["EXP"])
    score_out = exp / "huge_gad_scores.txt"
    env = dict(os.environ, GRAFLAG_SCORE_OUT=str(score_out))

    command = [
        "python3", "main.py",
        "--dataset", dataset,
        "--lr", str(config.lr),
        "--epoch", str(config.epoch),
        "--kd_param", str(config.kd_param),
        "--weight_decay", str(config.weight_decay),
        "--hidden", str(config.hidden),
        "--batch_size_heterophily", str(config.batch_size_heterophily),
        "--batch_size_sampling", str(config.batch_size_sampling),
        "--heterophily", config.heterophily,
        "--seed", str(config.seed),
    ]
    info(f"[INFO] HUGE-GAD: {' '.join(command[1:])}")
    result = subprocess.run(command, cwd=str(UPSTREAM), env=env,
                            capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    if stdout:
        info(stdout[-3000:])
    if result.returncode != 0:
        raise RuntimeError(
            f"HUGE-GAD exited {result.returncode}: "
            f"{(result.stderr or '').strip()[-3000:] or 'no stderr'}")

    if not score_out.is_file():
        raise FileNotFoundError(
            f"HUGE-GAD exited 0 but wrote no {score_out}. "
            "export-scores.patch writes it where the AUC is computed, and "
            "`git apply --verbose` would have failed the build otherwise.")

    scores = np.loadtxt(score_out, dtype=float).ravel()
    truth = np.loadtxt(str(score_out) + ".label", dtype=float).ravel()
    truth = (truth > 0).astype(int)
    if scores.shape[0] != truth.shape[0]:
        raise ValueError(
            f"HUGE-GAD wrote {scores.shape[0]} scores and {truth.shape[0]} "
            "labels; publishing them would misalign every score.")

    method_auc = None
    for line in stdout.splitlines():
        if line.startswith("auc_roc:"):
            method_auc = float(line.split()[1])

    writer = ResultWriter()
    writer.save_scores(result_type="NODE_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="huge_gad", summary={
        "dataset_info": {
            "dataset": p.data.name,
            "staged_as": dataset,
            "scored_split": "all_nodes",
            "scored_samples": int(scores.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            "lr": config.lr, "epoch": config.epoch,
            "kd_param": config.kd_param, "heterophily": config.heterophily,
            "seed": config.seed,
            "auc_roc": method_auc,
        },
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} node scores "
         f"({int(truth.sum())} anomalous), method AUC {method_auc}")


if __name__ == "__main__":
    main()
