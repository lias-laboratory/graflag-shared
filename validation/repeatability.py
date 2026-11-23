#!/usr/bin/env python3
"""Repeated-run consistency: each replay's scores against the original run's.

    python3 validation/repeatability.py --jsonl results/repeat.jsonl \
        --cfg-dir validation/cfg --out results/repeat-summary.json

For every method in the replay log, the original experiment is the one its
cfg/<method>.json was copied from. Each run's results.json and
eval/evaluation.json are read from the share over ssh; the comparison is done
here, on the score vectors themselves: element count, max |difference| from
the original, whether every score is bit-identical, the evaluator's AUC-ROC,
and the node each run was placed on.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

from clusterenv import alias, aliases, assert_private, cluster_env

KEY = str(Path.home() / ".ssh/id_vps")


MANAGER = f"root@{cluster_env()['MANAGER']}"


def remote_json(path):
    done = subprocess.run(["ssh", "-i", KEY, MANAGER, f"cat {path}"],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if done.returncode != 0:
        return None
    return json.loads(done.stdout)


def flatten(x):
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(flatten(v))
        return out
    return [x]


def load(exp):
    base = f"/shared/experiments/{exp}"
    res = remote_json(f"{base}/results.json") or {}
    ev = remote_json(f"{base}/eval/evaluation.json") or {}
    meta = res.get("metadata") or {}
    return {"experiment": exp,
            "scores": [float(v) for v in flatten(res.get("scores") or [])],
            "auc_roc": (ev.get("metrics") or ev).get("auc_roc"),
            "node": alias(meta.get("node"), aliases()),
            "exec_time_ms": meta.get("exec_time_ms")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cfg-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    replays = {}
    for line in open(args.jsonl):
        row = json.loads(line)
        if row.get("experiment"):
            replays.setdefault(row["method"], []).append(row)

    summary = {}
    for method, rows in sorted(replays.items()):
        cfg = json.loads((Path(args.cfg_dir) / f"{method}.json").read_text())
        original = load(cfg["experiment_name"])
        runs = []
        for row in rows:
            run = load(row["experiment"])
            a, b = original["scores"], run["scores"]
            same_len = len(a) == len(b)
            diffs = [abs(x - y) for x, y in zip(a, b)
                     if math.isfinite(x) and math.isfinite(y)] if same_len else []
            runs.append({
                "experiment": run["experiment"], "node": run["node"],
                "status": row.get("status"), "verify": row.get("verify"),
                "n_scores": len(b), "same_length": same_len,
                "max_abs_diff": max(diffs) if diffs else None,
                "identical": same_len and a == b,
                "auc_roc": run["auc_roc"], "exec_time_ms": run["exec_time_ms"],
            })
        aucs = [r["auc_roc"] for r in runs if r["auc_roc"] is not None]
        if original["auc_roc"] is not None:
            aucs.append(original["auc_roc"])
        summary[method] = {
            "original": {k: original[k] for k in ("experiment", "node", "auc_roc", "exec_time_ms")}
                        | {"n_scores": len(original["scores"])},
            "replays": runs,
            "auc_range": [min(aucs), max(aucs)] if aucs else None,
            "all_identical": all(r["identical"] for r in runs),
            "nodes": sorted({r["node"] for r in runs if r["node"]} | ({original["node"]} if original["node"] else set())),
        }
        print(f"{method:14} runs={len(runs)+1} identical={summary[method]['all_identical']} "
              f"max_diff={max((r['max_abs_diff'] or 0) for r in runs):.3g} "
              f"auc={summary[method]['auc_range']} nodes={summary[method]['nodes']}")
    text = json.dumps(summary, indent=1)
    assert_private(text, args.out)
    Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
