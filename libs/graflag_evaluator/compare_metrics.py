"""Recompute metrics with the old and new implementations and diff them.

The metric fixes change reported numbers, so anything already published needs
checking. This runs both implementations over the same ``results.json`` and
prints what moves. It reads only; nothing is written and no cluster is needed.

    python3 -m graflag_evaluator.compare_metrics /shared/experiments/exp__*
    python3 -m graflag_evaluator.compare_metrics ./downloaded_experiment --json

The "old" implementation below is a verbatim copy of the pre-fix code, kept
here deliberately so the comparison does not depend on git history.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn import metrics as skm

from .metrics import compute_classification_metrics as new_metrics

# Metrics whose definition changed; shown even when numerically equal.
CHANGED_BY_DESIGN = {"auc_pr", "precision_at_k", "recall_at_k", "f1_at_k"}


def old_metrics(scores: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """The pre-fix implementation, preserved verbatim for comparison."""
    if scores.dtype == object or (
        scores.ndim == 1 and len(scores) > 0
        and isinstance(scores[0], (list, np.ndarray))
    ):
        scores_flat = np.concatenate([np.asarray(s).flatten() for s in scores])
        gt_flat = np.concatenate([np.asarray(g).flatten() for g in ground_truth])
    else:
        scores_flat = scores.flatten()
        gt_flat = ground_truth.flatten()

    valid_mask = (
        (scores_flat >= 0) & (scores_flat <= 1)
        if np.max(scores_flat) <= 1
        else scores_flat > -2
    )
    scores_valid = scores_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]

    if len(np.unique(gt_valid)) < 2:
        return {"auc_roc": None, "auc_pr": None}

    auc_roc = skm.roc_auc_score(gt_valid, scores_valid)
    precision, recall, thresholds = skm.precision_recall_curve(gt_valid, scores_valid)
    auc_pr = skm.auc(recall, precision)

    k = int(np.sum(gt_valid))
    top_k_indices = np.argsort(scores_valid)[-k:]
    predictions_at_k = np.zeros_like(gt_valid)
    predictions_at_k[top_k_indices] = 1

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1 = np.max(f1_scores)
    best_thr = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else None

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "precision_at_k": round(float(skm.precision_score(gt_valid, predictions_at_k, zero_division=0)), 4),
        "recall_at_k": round(float(skm.recall_score(gt_valid, predictions_at_k, zero_division=0)), 4),
        "f1_at_k": round(float(skm.f1_score(gt_valid, predictions_at_k, zero_division=0)), 4),
        "best_f1": round(float(best_f1), 4),
        "best_f1_threshold": round(float(best_thr), 4) if best_thr else None,
        "num_anomalies": int(k),
        "num_samples": int(len(gt_valid)),
        "anomaly_ratio": round(float(k / len(gt_valid)), 4),
    }


def _load(results_path: Path):
    with results_path.open() as fh:
        data = json.load(fh)
    scores = data.get("scores")
    gt = data.get("ground_truth")
    if scores is None or gt is None:
        raise ValueError("results.json has no 'scores' or no 'ground_truth'")
    return np.array(scores, dtype=object), np.array(gt, dtype=object), data


def compare_one(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    """Return a comparison for one experiment directory, or None to skip."""
    results_path = experiment_dir / "results.json"
    if not results_path.is_file():
        return None

    try:
        scores, gt, raw = _load(results_path)
    except Exception as exc:  # malformed / truncated results.json
        return {"experiment": experiment_dir.name, "error": str(exc)}

    entry: Dict[str, Any] = {
        "experiment": experiment_dir.name,
        "result_type": raw.get("result_type"),
    }
    for label, fn in (("old", old_metrics), ("new", new_metrics)):
        try:
            entry[label] = fn(scores, gt)
        except Exception as exc:
            entry[label] = {"error": f"{type(exc).__name__}: {exc}"}

    recorded = experiment_dir / "eval" / "evaluation.json"
    if recorded.is_file():
        try:
            with recorded.open() as fh:
                entry["recorded"] = json.load(fh).get("metrics", {})
        except Exception:
            pass
    return entry


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(entries: List[Dict[str, Any]]) -> bool:
    """Print the diff. Returns True if any metric moved."""
    any_moved = False

    for entry in entries:
        print(f"\n{entry['experiment']}  [{entry.get('result_type', '?')}]")
        if "error" in entry:
            print(f"  ERROR: {entry['error']}")
            continue

        old, new = entry.get("old", {}), entry.get("new", {})
        if "error" in old or "error" in new:
            print(f"  old: {old.get('error', 'ok')}")
            print(f"  new: {new.get('error', 'ok')}")

        keys = [k for k in ("auc_roc", "auc_pr", "precision_at_k", "recall_at_k",
                            "f1_at_k", "best_f1", "best_f1_threshold",
                            "num_samples", "num_anomalies")
                if k in old or k in new]

        print(f"  {'metric':<20} {'old':>12} {'new':>12}   change")
        print(f"  {'-' * 20} {'-' * 12} {'-' * 12}   ------")
        for key in keys:
            o, n = old.get(key), new.get(key)
            moved = o != n
            any_moved = any_moved or moved
            if moved and isinstance(o, (int, float)) and isinstance(n, (int, float)):
                delta = n - o
                note = (f"{delta:+d}" if isinstance(o, int) and isinstance(n, int)
                        else f"{delta:+.4f}")
            elif moved:
                note = "CHANGED"
            elif key in CHANGED_BY_DESIGN:
                note = "(same)"
            else:
                note = ""
            print(f"  {key:<20} {_fmt(o):>12} {_fmt(n):>12}   {note}")

        filtering = new.get("filtering")
        if filtering and filtering.get("kept") != filtering.get("total"):
            print(f"  excluded {filtering['total'] - filtering['kept']} of "
                  f"{filtering['total']} samples "
                  f"(unknown={filtering['dropped_unknown']}, "
                  f"inactive={filtering['dropped_inactive']}, "
                  f"non-finite={filtering['dropped_non_finite']})")

    return any_moved


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff old vs new evaluator metrics for existing experiments.",
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Experiment directories (each containing results.json)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = parser.parse_args(argv)

    dirs: List[Path] = []
    for path in args.paths:
        if (path / "results.json").is_file():
            dirs.append(path)
        elif path.is_dir():
            dirs.extend(sorted(d for d in path.iterdir()
                               if (d / "results.json").is_file()))

    if not dirs:
        print("[WARN] No experiment directories with results.json found", file=sys.stderr)
        return 2

    entries = [e for e in (compare_one(d) for d in dirs) if e]

    if args.json:
        print(json.dumps(entries, indent=2, default=str))
        return 0

    moved = print_report(entries)
    print(f"\n[INFO] Compared {len(entries)} experiment(s).")
    if moved:
        print("[WARN] Some metrics changed. Published figures derived from the "
              "previous implementation need reviewing.")
    else:
        print("[OK] No metric changed for these experiments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
