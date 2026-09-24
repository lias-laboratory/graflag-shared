"""F-FADE (WSDM 2021) under GraFlag.

Unsupervised anomaly detection in edge streams by frequency factorization.
Upstream's ``main.py`` is run as written -- it already writes its score vector
to ``score.txt``, so nothing here is patched.

The dataset is rendered into the layout F-FADE reads rather than F-FADE being
redirected at GraFlag's: its ``Dataset`` parses whitespace-separated
``timestamp source destination label`` from one file, while GraFlag stores
``source,destination,timestamp`` in ``Data.csv`` with labels alongside in
``Label.csv``. Reordering four columns in this script is a smaller and more
honest change than patching the authors' loader.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from graflag_runner import (ResultWriter, params, device, paths, upstream,
                            seed_all, info, warning)

UPSTREAM = upstream("src")


@dataclass
class Config:
    embedding_size: int = 200
    batch_size: int = 32
    t_setup: int = 8000
    w_upd: int = 720
    alpha: float = 0.999
    t_th: int = 120
    m: int = 100
    epochs: int = 5
    online_train_steps: int = 10
    seed: int = 42


def render_dataset(data_dir: Path, target: Path) -> int:
    """Write GraFlag's edge stream in F-FADE's column order.

    GraFlag: ``Data.csv`` of ``src,dst,timestamp`` plus ``Label.csv``.
    F-FADE:  one whitespace-separated file of ``timestamp src dst label``.

    Streamed line by line rather than loaded: anograph_darpa is 4.5M records
    and holding three parallel arrays of it to reorder columns is memory the
    container does not need to spend.
    """
    data_file, label_file = data_dir / "Data.csv", data_dir / "Label.csv"
    for required in (data_file, label_file):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing. F-FADE scores an edge stream and "
                f"{data_dir.name} does not provide one in GraFlag's "
                "Data.csv / Label.csv form.")

    written = 0
    with data_file.open() as edges, label_file.open() as labels, \
            target.open("w") as out:
        for edge_line, label_line in zip(edges, labels):
            edge_line, label_line = edge_line.strip(), label_line.strip()
            if not edge_line:
                continue
            parts = edge_line.split(",")
            if len(parts) < 3:
                raise ValueError(
                    f"{data_file}:{written + 1} is {parts!r}, not "
                    "src,dst,timestamp -- refusing to guess the columns.")
            src, dst, timestamp = parts[0], parts[1], parts[2]
            label = int(float(label_line or 0))
            out.write(f"{timestamp} {src} {dst} {1 if label else 0}\n")
            written += 1

    # zip() stops at the shorter file, which would silently drop the tail of
    # the stream and publish scores for a prefix while calling it the whole.
    edge_count = sum(1 for _ in data_file.open())
    label_count = sum(1 for _ in label_file.open())
    if not (written == edge_count == label_count):
        raise ValueError(
            f"{data_dir.name}: wrote {written} records from {edge_count} "
            f"edges and {label_count} labels -- the two files disagree.")
    info(f"[INFO] Rendered {written} edges to {target} (ts src dst label)")
    return written


def main():
    import numpy as np

    config = Config(**params(Config))
    seed_all(config.seed)
    dev = device()                      # honours _GPU; -1 means CPU

    p = paths()
    exp = Path(os.environ["EXP"])
    rendered = exp / "f_fade_input.txt"
    n_records = render_dataset(p.data, rendered)

    model_dir = exp / "f_fade_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python3", "main.py",
        "--dataset", str(rendered),
        "--model_dir", str(model_dir) + os.sep,   # upstream concatenates, no join
        "--embedding_size", str(config.embedding_size),
        "--batch_size", str(config.batch_size),
        "--t_setup", str(config.t_setup),
        "--W_upd", str(config.w_upd),
        "--alpha", str(config.alpha),
        "--T_th", str(config.t_th),
        "--M", str(config.m),
        "--epochs", str(config.epochs),
        "--online_train_steps", str(config.online_train_steps),
        "--gpu", str(dev.index if dev.type == "cuda" else 0),
    ]
    info(f"[INFO] F-FADE: {' '.join(command[1:])}")
    result = subprocess.run(command, cwd=str(UPSTREAM), capture_output=True,
                            text=True)
    stdout = (result.stdout or "").strip()
    if stdout:
        info(stdout[-2000:])
    # main.py prints the AUC it measured over the same score vector it wrote.
    # Recording it lets graflag verify cross-check that number against the one
    # the evaluator computes from results.json; a mismatch means the published
    # scores are not the ones the method scored.
    method_auc = None
    for line in stdout.splitlines():
        if line.startswith("AUC:"):
            method_auc = float(line.split(":", 1)[1])
    if result.returncode != 0:
        raise RuntimeError(
            f"F-FADE exited {result.returncode}: "
            f"{(result.stderr or '').strip()[-2000:] or 'no stderr'}")

    score_file = model_dir / "score.txt"
    if not score_file.is_file():
        raise FileNotFoundError(
            f"F-FADE exited 0 but wrote no {score_file}. main.py saves the "
            "score vector before computing its AUC, so reaching the end "
            "without one means it did not reach the end.")

    scores = np.loadtxt(score_file, dtype=float).ravel()
    truth_all = np.loadtxt(p.data / "Label.csv", dtype=float).ravel()
    truth_all = (truth_all > 0).astype(int)

    # F-FADE scores a *suffix*: nothing before --t_setup is scored at all, and
    # upstream aligns with `dataset.label[-len(F_FADE):]`. Publishing the full
    # label column against a shorter score vector would silently pair each
    # score with the wrong edge.
    if scores.shape[0] > truth_all.shape[0]:
        raise ValueError(
            f"F-FADE produced {scores.shape[0]} scores for {truth_all.shape[0]} "
            "labelled edges, which cannot be a suffix of the stream.")
    truth = truth_all[-scores.shape[0]:]
    skipped = truth_all.shape[0] - scores.shape[0]

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        # Published as computed. main.py writes score.txt first and only then
        # replaces NaN with 0 for its own AUC, so zero-filling here would
        # publish a vector the method did not produce. The evaluator excludes
        # non-finite scores and reports the count under `filtering`.
        warning(f"[WARN] {n_nan} of {scores.shape[0]} scores are NaN; "
                "published as computed. Upstream's own AUC replaces them "
                "with 0, so its number and this one differ by that much.")

    writer = ResultWriter()
    writer.save_scores(result_type="EDGE_STREAM_ANOMALY_SCORES",
                       scores=scores.tolist(),
                       ground_truth=truth.tolist())
    writer.add_metadata(method_name="f_fade", summary={
        "dataset_info": {
            "dataset": p.data.name,
            "scored_split": "stream_suffix",
            "scored_samples": int(scores.shape[0]),
            "skipped_setup_edges": int(skipped),
            "total_edges": int(truth_all.shape[0]),
            "anomalies": int(truth.sum()),
        },
        "training_info": {
            "t_setup": config.t_setup,
            "embedding_size": config.embedding_size,
            "epochs": config.epochs,
            "nan_scores": n_nan,
            "seed": config.seed,
            # Upstream's own AUC, computed with NaN replaced by 0. GraFlag's
            # excludes non-finite scores instead, so where nan_scores > 0 the
            # two differ by design -- see README.md.
            "auc_roc": method_auc,
        },
    })
    writer.finalize()
    info(f"[OK] Published {scores.shape[0]} edge scores "
         f"({int(truth.sum())} anomalous), {skipped} setup edges unscored")


if __name__ == "__main__":
    main()
