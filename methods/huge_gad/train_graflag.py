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
                            info)

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


# The names upstream's modules/utils.load_dataset routes through load_mat. It
# raises "Unimplemented dataset" for any other, after the image has started
# and the graph has been rendered, so a name outside this set is refused here.
UPSTREAM_MAT_NAMES = ("Amazon", "Facebook", "Reddit", "YelpChi")

# bond_* graphs that are one of those datasets, staged under upstream's name.
# graflag_runner's tests/test_mat.py checks that Reddit.mat round-trips to
# bond_reddit. bond_books, bond_disney, bond_enron and bond_weibo have no
# loader upstream, so they are not here and the method does not declare them.
RENDERED_AS = {"bond_reddit": "Reddit"}


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
        if name not in UPSTREAM_MAT_NAMES:
            raise ValueError(
                f"{source.name}: upstream's load_dataset accepts only "
                f"{', '.join(UPSTREAM_MAT_NAMES)} and raises for '{name}'")
        shutil.copy2(source, target_dir / f"{name}.mat")
        info(f"[INFO] Copied {source} -> {target_dir / f'{name}.mat'} (already .mat)")
        return name

    # No .mat: render one from the canonical PyG form, under upstream's name.
    name = RENDERED_AS.get(data_dir.name)
    if name is None:
        raise ValueError(
            f"{data_dir.name} has no .mat and is not one of upstream's datasets "
            f"({', '.join(f'{k} as {v}' for k, v in RENDERED_AS.items())}); "
            f"upstream's load_dataset accepts only {', '.join(UPSTREAM_MAT_NAMES)}")
    data = load_attributed_graph(data_dir)
    written = write_mat(data, target_dir / f"{name}.mat")
    info(f"[INFO] Rendered {data_dir.name} ({data.num_nodes} nodes) as {name} -> {written}")
    return name


def require_dgl_sparse() -> None:
    """Load DGL's sparse library now, or fail before upstream starts.

    The conda build links the driver's libcuda.so.1, which only a running
    container has (the NVIDIA runtime mounts it), so the image build cannot
    check this; see the Dockerfile. Without it, dgl.sparse raises deep inside
    upstream's model code instead.
    """
    try:
        import dgl.sparse  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"DGL's sparse library does not load ({exc}); this container needs "
            f"the NVIDIA runtime's libcuda.so.1") from exc


def main():
    import numpy as np

    require_dgl_sparse()

    config = Config(**params(Config))
    seed_all(config.seed)
    dev = device()

    p = paths()
    dataset = stage_dataset(p.data)

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
