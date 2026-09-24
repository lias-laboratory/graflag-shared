"""MIDAS (AAAI 2020) under GraFlag.

Unsupervised anomaly detection in edge streams. The algorithm is upstream's,
unmodified: MIDAS is header-only C++ and ``patches/graflag-driver.patch``
touches only ``example/Demo.cpp``, which is a driver in upstream too.

The dataset needs no conversion. MIDAS reads a header-less CSV of
``source,destination,timestamp``, which is exactly what GraFlag stores as
``Data.csv`` -- ``anograph_darpa/Data.csv`` is byte-identical to the DARPA copy
the MIDAS repository itself ships, 4,554,344 records and labels included.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graflag_runner import (ResultWriter, params, paths, upstream,
                            seed_all, info, warning)

UPSTREAM = upstream("src")          # raises if the clone is missing
DEMO = UPSTREAM / "build" / "Demo"


@dataclass
class Config:
    core: str = "filtering"         # normal | relational | filtering
    num_row: int = 2
    num_column: int = 1024
    factor: float = 0.5
    threshold: float = 1000.0
    seed: int = 42


CORES = {"normal": "MIDAS", "relational": "MIDAS-R", "filtering": "MIDAS-F"}


def main():
    import numpy as np

    config = Config(**params(Config))
    if config.core not in CORES:
        raise ValueError(
            f"_CORE must be one of {sorted(CORES)}, got {config.core!r}. "
            "These are the three published variants; a typo would otherwise "
            "reach the driver and fail there with less context.")
    seed_all(config.seed)

    p = paths()
    data_file = p.data / "Data.csv"
    label_file = p.data / "Label.csv"
    for required in (data_file, label_file):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing. MIDAS scores an edge stream and "
                f"{p.data.name} does not provide one in GraFlag's Data.csv / "
                "Label.csv form; nothing here can substitute for it.")

    # The driver needs the record count up front -- upstream keeps it in a
    # shape file its PreprocessData.py writes. Counting the mounted file is
    # better than trusting a sidecar that can disagree with it.
    with data_file.open("rb") as fh:
        n_records = sum(1 for _ in fh)
    info(f"[INFO] {p.data.name}: {n_records} edge records")

    if not DEMO.is_file():
        raise FileNotFoundError(
            f"{DEMO} is missing: the image did not build the driver. The "
            "Dockerfile's `test -x` should have failed the build first.")

    score_file = Path(os.environ.get("EXP", ".")) / "midas_scores.txt"
    command = [str(DEMO), str(data_file), str(n_records), str(score_file),
               config.core, str(config.num_row), str(config.num_column),
               str(config.factor), str(config.threshold), str(config.seed),
               str(label_file)]
    info(f"[INFO] {CORES[config.core]}: {' '.join(command[1:])}")

    result = subprocess.run(command, capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    if stdout:
        info(stdout)
    # The driver prints upstream's own AUROC over the same score array it
    # wrote. Recording it lets graflag verify cross-check the evaluator's AUC
    # against the method's; a mismatch means the published vector is not the
    # one the method measured.
    method_auc = None
    for line in stdout.splitlines():
        if line.startswith("ROC-AUC = "):
            method_auc = float(line.split("=", 1)[1])
    if result.returncode != 0:
        raise RuntimeError(
            f"MIDAS driver exited {result.returncode}: "
            f"{(result.stderr or '').strip() or 'no stderr'}")

    scores = np.loadtxt(score_file, dtype=float).ravel()
    truth = np.loadtxt(label_file, dtype=float).ravel()
    truth = (truth > 0).astype(int)

    if scores.shape[0] != truth.shape[0]:
        raise ValueError(
            f"MIDAS produced {scores.shape[0]} scores for {truth.shape[0]} "
            "labelled edges -- publishing them would misalign every score "
            "with its label.")
    if scores.shape[0] != n_records:
        raise ValueError(
            f"MIDAS produced {scores.shape[0]} scores for {n_records} input "
            "records; the driver stopped early.")

    writer = ResultWriter()          # defaults to $EXP
    writer.save_scores(result_type="EDGE_STREAM_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="midas", summary={
        "dataset_info": {
            "dataset": p.data.name,
            "scored_split": "all_edges",
            "scored_samples": int(scores.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            # MIDAS fits nothing: it is a streaming detector that scores each
            # edge from the sketch state built by the edges before it. There
            # is no train/test split to report, and no epoch.
            "variant": CORES[config.core],
            "core": config.core,
            "num_row": config.num_row,
            "num_column": config.num_column,
            "factor": config.factor,
            "threshold": config.threshold if config.core == "filtering" else None,
            "seed": config.seed,
            "auc_roc": method_auc,
        },
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} edge scores "
         f"({int(truth.sum())} anomalous) from {CORES[config.core]}")


if __name__ == "__main__":
    main()
