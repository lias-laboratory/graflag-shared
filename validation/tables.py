#!/usr/bin/env python3
"""Reduce collected campaign rows to one row per method, and count them.

    python3 validation/tables.py results/coverage-all.json \
        --out results/coverage-final.json [--markdown results/coverage-final.md]

A method is **configured** when its directory has a `.env` (the annotated
`example/` template is not a method and is left out), **executed** when one of
its attempts completed on the cluster, and **verified** when a completed
attempt passed `graflag verify` with no failed check. The row kept for each
method is its first verified attempt, else its last completed one, else its
last attempt; `attempts` counts all of them, so a method that needed four
tries says so.
"""

import argparse
import json
import os
from pathlib import Path

# The paper's grouping (Table 1 of the OSP paper), not the result type: a
# method's result type says what it publishes, e.g. AnoGraph's per-window
# GRAPH_ANOMALY_SCORES, not which kind of input it detects on. "streaming"
# covers the paper's edge streams and its one graph stream.
SETTING = {
    "ad_gcl": "static", "ada_gad": "static", "diffgad": "static",
    "huge_gad": "static", "rare": "static",
    "taddy": "temporal", "addgraph": "temporal", "strgnn": "temporal",
    "dynwalk": "temporal",
    "generaldyg": "streaming", "gady": "streaming", "slade": "streaming",
    "f_fade": "streaming", "anograph": "streaming", "midas": "streaming",
    "streamspot": "streaming",
}


def setting(method):
    if method.startswith("bond_"):
        return "static"
    if method not in SETTING:
        raise KeyError(f"{method} has no setting in tables.SETTING; add it")
    return SETTING[method]



def verified(row):
    v = row.get("verify") or {}
    return row.get("status") == "completed" and v.get("failed") == 0


def pick(attempts):
    attempts = sorted(attempts, key=lambda r: r.get("experiment") or "")
    for test in (verified, lambda r: r.get("status") == "completed", lambda r: True):
        chosen = [r for r in attempts if test(r)]
        if chosen:
            return chosen[0] if test is verified else chosen[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("collected")
    ap.add_argument("--methods-dir", default=str(Path(__file__).resolve().parents[1] / "methods"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown")
    args = ap.parse_args()

    rows = json.loads(Path(args.collected).read_text())
    configured = sorted(p.name for p in Path(args.methods_dir).iterdir()
                        if (p / ".env").is_file() and p.name != "example")
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    table = []
    for m in configured:
        attempts = by_method.get(m, [])
        row = pick(attempts) if attempts else {}
        v = row.get("verify") or {}
        metrics = row.get("metrics") or {}
        table.append({
            "method": m,
            "attempts": len(attempts),
            "experiment": row.get("experiment"),
            "dataset": row.get("dataset"),
            "setting": setting(m),
            "result_type": row.get("result_type"),
            "device": None if not row else ("gpu" if row.get("gpu") else "cpu"),
            "node": row.get("node"),
            "status": row.get("status"),
            "executed": row.get("status") == "completed",
            "verified": verified(row) if row else False,
            "verify": {k: v.get(k) for k in ("failed", "warned", "passed")} if v else None,
            "auc_roc": metrics.get("auc_roc"),
            "auc_pr": metrics.get("auc_pr"),
            "exec_time_s": None if row.get("exec_time_ms") is None
            else round(row["exec_time_ms"] / 1000.0, 1),
            "peak_memory_mb": row.get("peak_memory_mb"),
            "peak_gpu_mb": row.get("peak_gpu_mb"),
            "scored_split": row.get("scored_split"),
        })

    counts = {
        "configured": len(table),
        "executed": sum(t["executed"] for t in table),
        "verified": sum(t["verified"] for t in table),
        "by_setting": {},
    }
    for t in table:
        s = counts["by_setting"].setdefault(t["setting"],
                                            {"configured": 0, "executed": 0, "verified": 0})
        s["configured"] += 1
        s["executed"] += t["executed"]
        s["verified"] += t["verified"]
    counts["not_executed"] = [t["method"] for t in table if not t["executed"]]
    counts["executed_not_verified"] = [t["method"] for t in table
                                       if t["executed"] and not t["verified"]]

    out = {"source": str(args.collected), "counts": counts, "methods": table}
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(f"[OK] {counts['configured']} configured, {counts['executed']} executed, "
          f"{counts['verified']} verified -> {args.out}")

    if args.markdown:
        def cell(x, fmt="{}"):
            return "--" if x is None else fmt.format(x)
        lines = ["| Method | Setting | Dataset | Device | Attempts | Status | verify (F/W/P) "
                 "| AUC-ROC | Time (s) | Peak RAM (MB) |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for t in table:
            v = t["verify"]
            lines.append("| `{}` | {} | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                t["method"], cell(t["setting"]), cell(t["dataset"]), cell(t["device"]),
                t["attempts"], cell(t["status"]),
                "--" if not v else f"{v['failed']}/{v['warned']}/{v['passed']}",
                cell(t["auc_roc"], "{:.4f}"), cell(t["exec_time_s"], "{:,.1f}"),
                cell(t["peak_memory_mb"], "{:,.0f}")))
        Path(args.markdown).write_text("\n".join(lines) + "\n")
        print(f"[OK] markdown -> {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
