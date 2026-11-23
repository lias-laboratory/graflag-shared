#!/usr/bin/env python3
"""Compare a GraFlag experiment's published scores with a reference vector.

    python3 compare_scores.py EXP_DIR REFERENCE [--labels LABELS]

EXP_DIR is an experiment directory on the share; REFERENCE is a .npy array or
a text file with one score per line, in the same order as the experiment's
flattened `scores`. Prints one JSON object: whether the two vectors are
identical, the largest absolute difference, and the AUC-ROC of each side
against the experiment's ground truth (or LABELS, one per line), computed
here with a rank-based estimator so that neither GraFlag's evaluator nor the
reference's own metric code is involved.

Needs only NumPy and SciPy, so it runs on the manager.
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata


def flatten(values):
    out = []
    for v in values:
        if isinstance(v, list):
            out.extend(flatten(v))
        else:
            out.append(v)
    return out


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    n_pos, n_neg = labels.sum(), (~labels).sum()
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_vector(path):
    """A score vector: .npy, plain text, or another run's results.json
    (or its experiment directory), which is how repeated runs are compared."""
    if os.path.isdir(path):
        path = os.path.join(path, "results.json")
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64).ravel()
    if path.endswith(".json"):
        with open(path) as fh:
            return np.asarray(flatten(json.load(fh)["scores"]), dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("reference")
    ap.add_argument("--labels")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="also report agreement within this tolerance")
    args = ap.parse_args()

    with open(f"{args.exp_dir}/results.json") as fh:
        res = json.load(fh)
    ours = np.asarray(flatten(res["scores"]), dtype=np.float64)
    gt = (np.loadtxt(args.labels).ravel() if args.labels
          else np.asarray(flatten(res["ground_truth"]), dtype=np.float64))
    ref = load_vector(args.reference)

    row = {"experiment": args.exp_dir.rstrip("/").rsplit("/", 1)[-1],
           "n_graflag": int(ours.size), "n_reference": int(ref.size)}
    if ours.size == ref.size:
        diff = np.abs(ours - ref)
        both = np.isfinite(diff)
        row["identical"] = bool(np.array_equal(ours, ref, equal_nan=True))
        row["max_abs_diff"] = float(diff[both].max()) if both.any() else None
        row["n_differing"] = int((ours != ref).sum())
        if args.atol:
            row[f"within_{args.atol:g}"] = bool(np.allclose(ours, ref, atol=args.atol, rtol=0))
        row["auc_graflag"] = auc(gt, ours)
        row["auc_reference"] = auc(gt, ref)
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
