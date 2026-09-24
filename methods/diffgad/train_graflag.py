"""DiffGAD (ICLR 2025) under GraFlag.

Unsupervised node-level anomaly detection in a latent diffusion space. Upstream's
own ``DiffGAD`` transform is constructed and called as written; the only change
to the clone is ``patches/expose-scores.patch``, which keeps the score vector
the sampling loop already computes.

The dataset is staged where the method looks for it rather than redirected:
``forward()`` calls ``pygod.utils.load_data(self.dataset)`` with no
``cache_dir``, which resolves to ``~/.pygod/data/<name>.pt``.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from graflag_runner import (ResultWriter, params, paths, upstream,
                            seed_all, info, warning)

UPSTREAM = upstream("src")

import sys                                            # noqa: E402
sys.path.insert(0, str(UPSTREAM))
from DiffGAD import DiffGAD                           # noqa: E402


@dataclass
class Config:
    """Only what GraFlag may override on top of the authors' config.

    The per-dataset values -- ae_lr, ae_dropout, ae_alpha, proto_alpha,
    weight, hid_dim -- come from upstream's own configs/<dataset>.yaml and are
    not defaulted here, because a default would silently replace a tuned value
    with a generic one. `--params AE_ALPHA=0.5` still wins; see main().
    """
    ae_epochs: int = 300
    diff_epochs: int = 800
    lr: float = 0.004          # main.py hardcodes this, not the yaml
    sample_steps: int = 50
    patience: int = 100


def stage_dataset(dataset_dir: Path) -> str:
    """Copy the mounted graph to ``~/.pygod/data/<bare name>.pt``.

    Raises when the directory holds no ``.pt``. ``load_data()`` downloads the
    dataset when the file is absent, so a silent failure here would train on an
    unpinned copy fetched from the internet instead of the mounted graph, and
    the run would look entirely normal.
    """
    source = dataset_dir / f"{dataset_dir.name}.pt"
    if not source.is_file():
        found = sorted(dataset_dir.glob("*.pt"))
        if not found:
            raise FileNotFoundError(
                f"no .pt graph in {dataset_dir}; contents: "
                f"{sorted(q.name for q in dataset_dir.iterdir())}")
        source = found[0]

    name = dataset_dir.name[len("bond_"):] if dataset_dir.name.startswith("bond_") \
        else dataset_dir.name
    cache = Path.home() / ".pygod" / "data"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}.pt"
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(source, target)
    info(f"[INFO] Staged {source} -> {target} (upstream name '{name}')")
    return name


def main():
    import numpy as np
    import torch

    # DiffGAD calls .cuda() unconditionally throughout its own code, so there
    # is no CPU path to fall back to. Saying so here is the difference between
    # a clear message and a CUDA error from inside the authors' code.
    gpu = int(os.environ.get("_GPU", 0))
    if gpu < 0:
        raise RuntimeError(
            "DiffGAD has no CPU path: its sampling loop calls .cuda() directly. "
            "Run it with _GPU>=0, or use a bond_* method for a CPU comparison.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DiffGAD requires CUDA and torch.cuda.is_available() is False.")

    config = Config(**params(Config))
    seed = int(os.environ.get("_SEED", 42))
    seed_all(seed)

    p = paths()
    dataset_name = stage_dataset(p.data)

    # Upstream ships one tuned config per dataset and main.py reads it before
    # constructing anything. The values differ materially between datasets
    # (books uses ae_alpha 0.5 and weight 2.0, weibo 0.8 and 1.0), so running
    # on generic defaults would not be running DiffGAD as published.
    import yaml
    config_file = UPSTREAM / "configs" / f"{dataset_name}.yaml"
    if not config_file.is_file():
        available = sorted(q.stem for q in (UPSTREAM / "configs").glob("*.yaml"))
        raise FileNotFoundError(
            f"no upstream config for '{dataset_name}' at {config_file}. "
            f"Upstream ships configs for: {', '.join(available)}. Running "
            "without one would publish a number from an untuned configuration.")
    cfg = yaml.load(config_file.read_text(), Loader=yaml.Loader)
    info(f"[INFO] Loaded upstream config {config_file.name}: {cfg}")

    # hid_dim is deliberately empty in every shipped yaml: DiffGAD.forward()
    # derives it from the feature count (2 ** int(log2(num_features) - 1)).
    # Passing a number here would replace that, so None is forwarded as-is.
    kwargs = dict(hid_dim=cfg.get("hid_dim"),
                  ae_dropout=cfg["ae_dropout"], ae_lr=cfg["ae_lr"],
                  ae_alpha=cfg["ae_alpha"], proto_alpha=cfg["proto_alpha"],
                  weight=cfg["weight"],
                  ae_epochs=config.ae_epochs, diff_epochs=config.diff_epochs,
                  lr=config.lr, sample_steps=config.sample_steps,
                  patience=config.patience)

    # GraFlag's explicit overrides go on last, so `--params AE_ALPHA=0.5` wins
    # over the authors' config -- the contract everywhere else in GraFlag.
    for key, value in params().items():
        if key in kwargs:
            kwargs[key] = value
            info(f"[INFO] Override from --params/.env: {key}={value}")

    model = DiffGAD(**kwargs)

    info(f"[INFO] DiffGAD on '{dataset_name}', seed={seed}")
    model(dataset_name)                      # BaseTransform.__call__ -> forward

    scores = getattr(model, "selected_scores", None)
    if scores is None:
        raise RuntimeError(
            "DiffGAD produced no score vector. expose-scores.patch sets "
            "self.selected_scores inside sample(), and `git apply --verbose` "
            "would have failed the build if it had not applied -- so sample() "
            "was never reached.")

    scores = np.asarray(scores, dtype=float).ravel()

    # Ground truth comes from the same file the method loaded. BOND encodes the
    # outlier type in the label bits (1 contextual, 2 structural, 3 both), and
    # the result contract wants binary.
    staged = Path.home() / ".pygod" / "data" / f"{dataset_name}.pt"
    truth = torch.load(staged, weights_only=False).y.cpu().numpy().ravel()
    truth = (truth > 0).astype(int)

    if scores.shape[0] != truth.shape[0]:
        raise ValueError(
            f"DiffGAD returned {scores.shape[0]} scores for a graph with "
            f"{truth.shape[0]} nodes -- publishing them would misalign every "
            "score with its label.")

    writer = ResultWriter()          # defaults to $EXP
    writer.save_scores(result_type="NODE_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="diffgad", summary={
        "dataset_info": {
            "dataset": dataset_name,
            "scored_split": "all_nodes",
            "scored_samples": int(scores.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            # Upstream picks this timestep by argmax of the AUC it measured
            # against the labels. See README.md -- it is not model selection
            # on a held-out split.
            "selected_timestep": int(model.selected_timestep),
            "timestep_selected_on": "test_labels",
            "seed": seed,
        },
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} node scores "
         f"({int(truth.sum())} anomalies) from timestep "
         f"{model.selected_timestep}")


if __name__ == "__main__":
    main()
