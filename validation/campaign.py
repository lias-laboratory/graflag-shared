#!/usr/bin/env python3
"""Run a list of GraFlag experiments end to end: run, evaluate, verify.

    python3 validation/campaign.py JOBS.json --config CLUSTER.env --out LOG.jsonl
        [--workers 4] [--only METHOD ...]

JOBS.json is a list of objects:

    {"method": "bond_dominant", "dataset": "bond_inj_cora",
     "params": {"SEED": 1}, "gpu": true, "build": true, "repeat": 1}

or, to replay a finished experiment through the CLI exactly as a user would
(`graflag run --from-config`), with "from_config" naming a local copy of its
service_config.json:

    {"method": "taddy", "dataset": "uci", "from_config": "cfg/taddy.json",
     "repeat": 5, "label": "repeat"}

Every job goes through the public client API (`graflag.core.GraFlag`), the
same calls `graflag run`, `graflag evaluate` and `graflag verify` make, with
`follow=False` so that several run at once. Jobs that share an image (the
seventeen `bond_*` methods share `bond_base`) wait for the first of them to
build it. Each finished job appends one JSON line to --out with its outcome;
`collect.py` later joins those lines with what the experiments wrote on the
share. A job that fails is recorded with its error and the campaign goes on:
the point is to know which methods run, not to stop at the first that does
not.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

from graflag.core import GraFlag

LOCK = threading.Lock()
NAME_LOCK = threading.Lock()
_last_name = {"value": None}


def image_key(shared_methods: Path, method: str) -> str:
    """The image a method runs in: its IMAGE= key, else its own name."""
    env = shared_methods / method / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("IMAGE="):
                return line.split("=", 1)[1].strip()
    return method


def unique_name(method: str, dataset: str) -> str:
    """A fresh experiment name; names carry a timestamp to the second."""
    with NAME_LOCK:
        while True:
            name = GraFlag.make_experiment_name(method, dataset)
            if name != _last_name["value"]:
                _last_name["value"] = name
                return name
            time.sleep(0.3)


def record(out: Path, row: dict) -> None:
    with LOCK:
        with out.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


EXP_DIR_LINE = re.compile(r"Experiment directory: \S*/experiments/(exp__\S+)")


def replay(config: str, path: str, log: Path, row: dict) -> str:
    """`graflag run --from-config PATH`, the command a user replays with.

    The CLI names the experiment itself, to the second, so launches are
    serialised until it has printed the directory it chose.
    """
    cmd = ["graflag", "--config", os.path.expanduser(config), "run",
           "--from-config", path]
    with NAME_LOCK:
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        name = None
        for line in proc.stdout:
            fh.write(line)
            m = EXP_DIR_LINE.search(line)
            if m:
                name = m.group(1)
                break
        time.sleep(1.2)
    row["experiment"] = name
    for line in proc.stdout:
        fh.write(line)
    proc.wait()
    fh.close()
    row["run_ok"] = proc.returncode == 0
    if proc.returncode != 0:
        row["run_error"] = f"graflag run --from-config exited {proc.returncode}; see {log}"
    return name


def run_job(gf: GraFlag, job: dict, build: bool, config: str = "",
            logdir: Path = Path(".")) -> dict:
    row = {"method": job["method"], "dataset": job["dataset"],
           "params": job.get("params") or {}, "gpu": job.get("gpu", True),
           "build": build, "label": job.get("label", "")}
    t0 = time.time()
    if job.get("from_config"):
        row["from_config"] = job["from_config"]
        log = logdir / f"replay_{job['method']}_{job.get('repeat_index', 0)}_{int(t0)}.log"
        name = replay(config, job["from_config"], log, row)
        if name is None:
            row["run_wall_s"] = round(time.time() - t0, 1)
            return row
    else:
        name = unique_name(job["method"], job["dataset"])
        row["experiment"] = name
        try:
            gf.run(job["method"], job["dataset"], build=build,
                   gpu=job.get("gpu", True), method_params=row["params"] or None,
                   exp_name=name, follow=False, tag=job.get("tag", "latest"))
            row["run_ok"] = True
        except Exception as exc:  # recorded, not raised: see the module docstring
            row["run_ok"] = False
            row["run_error"] = f"{type(exc).__name__}: {exc}"
    row["run_wall_s"] = round(time.time() - t0, 1)

    try:
        info = gf.get_experiment(name) if hasattr(gf, "get_experiment") else None
        row["status"] = getattr(info, "status", None)
    except Exception as exc:
        row["status_error"] = str(exc)

    if row["run_ok"] and row.get("status") == "completed" and not job.get("no_eval"):
        t1 = time.time()
        try:
            gf.evaluate(name, follow=False)
            row["eval_ok"] = True
        except Exception as exc:
            row["eval_ok"] = False
            row["eval_error"] = f"{type(exc).__name__}: {exc}"
        row["eval_wall_s"] = round(time.time() - t1, 1)
        try:
            report = gf.verify(name)
            row["verify"] = {"failed": report.failed, "warned": report.warned,
                             "passed": report.passed,
                             "findings": report.findings}
        except Exception as exc:
            row["verify_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--label-suffix", default="",
                    help="appended to every job's label (scaling.sh: -n1, -n2, -n3)")
    ap.add_argument("--methods-dir", default=str(Path(__file__).resolve().parents[1] / "methods"),
                    help="checkout used to read IMAGE= keys")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    if args.only:
        jobs = [j for j in jobs if j["method"] in args.only]
    expanded = []
    for job in jobs:
        for i in range(int(job.get("repeat", 1))):
            expanded.append(dict(job, repeat_index=i,
                                 label=job.get("label", "") + args.label_suffix))

    methods_dir = Path(args.methods_dir)
    builders = {}      # image -> Event set once its first job has built it
    first = {}         # image -> index of the job that builds it
    for idx, job in enumerate(expanded):
        key = image_key(methods_dir, job["method"])
        job["_image"] = key
        if job.get("build", False) and key not in first:
            first[key] = idx
            builders[key] = threading.Event()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for idx, job in enumerate(expanded):
        q.put((idx, job))

    def worker(n):
        gf = GraFlag(config_file=os.path.expanduser(args.config), interactive=False)
        while True:
            try:
                idx, job = q.get_nowait()
            except queue.Empty:
                return
            key = job["_image"]
            build = first.get(key) == idx
            if key in builders and not build:
                builders[key].wait()
            t = time.strftime("%H:%M:%S")
            print(f"[INFO] {t} worker {n}: {job['method']} on {job['dataset']}"
                  f"{' (build)' if build else ''}", flush=True)
            try:
                row = run_job(gf, job, build, config=args.config,
                              logdir=out.parent)
            except Exception:
                row = {"method": job["method"], "dataset": job["dataset"],
                       "run_ok": False, "run_error": traceback.format_exc()}
            finally:
                if build:
                    builders[key].set()
            row["repeat_index"] = job.get("repeat_index", 0)
            row["worker"] = n
            record(out, row)
            v = row.get("verify") or {}
            print(f"[INFO] {time.strftime('%H:%M:%S')} done {row.get('experiment')}: "
                  f"status={row.get('status')} run_ok={row.get('run_ok')} "
                  f"verify_failed={v.get('failed')} "
                  f"{row.get('run_error', '')[:200]}", flush=True)

    threads = [threading.Thread(target=worker, args=(n,), daemon=True)
               for n in range(args.workers)]
    for t in threads:
        t.start()
        time.sleep(2)
    for t in threads:
        t.join()
    print("[OK] campaign finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
