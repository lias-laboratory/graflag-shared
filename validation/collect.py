#!/usr/bin/env python3
"""Join a campaign log with what its experiments wrote on the share.

    python3 validation/collect.py LOG.jsonl [LOG.jsonl ...] --config CLUSTER.env
        --out TABLE.json [--csv TABLE.csv]

Reads every experiment named in the logs in **one** remote call (the probe
below runs on the manager) and never moves score arrays to the client. For
each experiment it reports the lifecycle status, the node the task ran on,
the metrics `graflag evaluate` computed, the runner's resource numbers, and
the scheduling overhead: the time from the service's creation on the manager
to the method starting, derived from two clocks on the cluster (the manager's
service timestamp and the runner's completion timestamp minus its measured
execution time), both NTP-synchronised.
"""

import argparse
import csv
import json
import os
import shlex
from pathlib import Path

from graflag.config import GraflagConfig as Config
from graflag.ssh import SSHManager

from clusterenv import alias, aliases, assert_private

PROBE = r'''
import datetime, json, os, subprocess, sys
shared = sys.argv[1]
names = json.loads(sys.argv[2])
nodes = {}
try:
    out = subprocess.run(["docker", "node", "ls", "--format", "{{.ID}} {{.Hostname}}"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        nid, host = line.split()
        nodes[nid] = host
except Exception:
    pass

def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None

def ts(value):
    if not value:
        return None
    value = value.rstrip("Z")
    if "." in value:
        head, frac = value.split(".", 1)
        tz = ""
        for sep in ("+", "-"):
            if sep in frac:
                frac, tz = frac.split(sep, 1)
                tz = sep + tz
                break
        value = head + "." + frac[:6] + tz
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()

# Runs that predate GRAFLAG_NODE: placements.sh samples `docker service ps`
# every 20 s; the last node a task was seen Running on is where it ran.
sampled = {}
try:
    with open("/root/placements.log") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 4 and parts[3] == "Running":
                sampled[parts[1]] = parts[2]
except OSError:
    pass

rows = {}
for name in names:
    d = os.path.join(shared, "experiments", name)
    row = {"exists": os.path.isdir(d)}
    st = load(os.path.join(d, "status.json")) or {}
    row["status"] = st.get("status")
    row["status_error"] = st.get("error")
    res = load(os.path.join(d, "results.json"))
    if res:
        md = res.get("metadata", {})
        row["exec_time_ms"] = md.get("exec_time_ms")
        row["peak_memory_mb"] = md.get("peak_memory_mb")
        row["peak_gpu_mb"] = md.get("peak_gpu_mb")
        row["result_type"] = res.get("result_type")
        summ = md.get("summary") or {}
        info = summ.get("dataset_info") or {}
        row["scored_split"] = summ.get("scored_split", info.get("scored_split"))
        row["scored_samples"] = summ.get("scored_samples", info.get("scored_samples"))
        row["method_parameters"] = md.get("method_parameters")
    ev = load(os.path.join(d, "eval", "evaluation.json"))
    if ev:
        m = ev.get("metrics") or {}
        row["metrics"] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
    sd = load(os.path.join(d, "service_details.json")) or {}
    w = sd.get("worker") or {}
    md = (res or {}).get("metadata", {}) if res else {}
    for source, value in (("runner", st.get("node") or md.get("node")),
                          ("service", w.get("node_hostname") or nodes.get(w.get("node_id"))),
                          ("sampled", sampled.get(name))):
        if value:
            row["node"], row["node_source"] = value, source
            break
    created = ts(sd.get("created_at_docker"))
    done = ts(st.get("timestamp")) if st.get("status") in ("completed", "failed") else None
    first = None
    try:
        import re as _re
        with open(os.path.join(d, "method_output.txt"), errors="replace") as fh:
            for _ in range(200):
                line = fh.readline()
                if not line:
                    break
                m = _re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})", line)
                if m:
                    first = ts(m.group(1).replace(" ", "T") + "." + m.group(2) + "000")
                    break
    except OSError:
        pass
    if created and first:
        # Service creation to the method's first log line: queueing for a
        # GPU, image pull, container start and interpreter start-up.
        row["first_log_after_create_s"] = round(first - created, 2)
    if created and done and row.get("exec_time_ms") is not None:
        start = done - row["exec_time_ms"] / 1000.0
        row["service_created_ts"] = created
        row["method_start_ts"] = start
        row["method_end_ts"] = done
        row["startup_overhead_s"] = round(start - created, 2)
    rows[name] = row
print(json.dumps(rows))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    args = ap.parse_args()

    entries = []
    for log in args.logs:
        for line in Path(log).read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    names = [e["experiment"] for e in entries if e.get("experiment")]

    cfg = Config(os.path.expanduser(args.config))
    ssh = SSHManager(manager_ip=cfg.manager_ip, ssh_port=cfg.ssh_port,
                     ssh_key=cfg.ssh_key, batch_mode=True)
    cmd = (f"python3 -c {shlex.quote(PROBE)} "
           f"{shlex.quote(cfg.remote_shared_dir)} {shlex.quote(json.dumps(names))}")
    result = ssh.execute(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] probe failed: {result.stderr}")
    remote = json.loads(result.stdout)

    names_of = aliases()
    table = []
    for e in entries:
        row = dict(e)
        row.update(remote.get(e.get("experiment"), {}))
        row["node"] = alias(row.get("node"), names_of)
        table.append(row)
    text = json.dumps(table, indent=1, sort_keys=True)
    assert_private(text, args.out)
    Path(args.out).write_text(text)

    if args.csv:
        cols = ["label", "method", "dataset", "repeat_index", "experiment", "status",
                "node", "gpu", "auc_roc", "auc_pr", "best_f1", "exec_s",
                "peak_memory_mb", "peak_gpu_mb", "startup_overhead_s",
                "first_log_after_create_s",
                "run_wall_s", "verify_failed", "verify_warned", "error"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in table:
                m = r.get("metrics") or {}
                v = r.get("verify") or {}
                w.writerow([
                    r.get("label"), r["method"], r["dataset"], r.get("repeat_index"),
                    r.get("experiment"), r.get("status"), r.get("node"), r.get("gpu"),
                    m.get("auc_roc"), m.get("auc_pr"), m.get("best_f1"),
                    None if r.get("exec_time_ms") is None else round(r["exec_time_ms"] / 1000, 1),
                    r.get("peak_memory_mb"), r.get("peak_gpu_mb"),
                    r.get("startup_overhead_s"), r.get("first_log_after_create_s"),
                    r.get("run_wall_s"),
                    v.get("failed"), v.get("warned"),
                    " ".join((r.get("run_error") or r.get("status_error") or "").split())[:300],
                ])
    print(f"[OK] {len(table)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
