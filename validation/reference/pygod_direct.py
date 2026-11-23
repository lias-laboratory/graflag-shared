#!/usr/bin/env python3
"""Reference for the static-graph check: PyGOD called directly, no GraFlag.

Runs inside the `bond_base` image (the same PyGOD build GraFlag's bond methods
use) but bypasses everything GraFlag adds -- the graflag_bond adapter, the
graflag_runner wrapper, ResultWriter and the evaluator. For each detector it
reads the method's `.env` defaults with a parser written here (not
graflag_bond's), builds the PyGOD detector from them, seeds Python, NumPy and
torch, fits on the full graph as BOND does, and scores with scikit-learn.

    python3 pygod_direct.py --dataset-dir /shared/datasets/bond_inj_cora \
        --methods-dir /shared/methods --methods bond_dominant bond_cola ... \
        --seed 42 --gpu -1 --out /work/pygod_direct

Writes <out>/<method>.npy (the scores) and <out>/summary.json.
"""

import argparse
import copy
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from pygod import detector as pygod_detector
from pygod.utils import load_data
from sklearn.metrics import average_precision_score, roc_auc_score


def env_defaults(path: Path) -> dict:
    """`_KEY=value` lines of a .env, as {key: raw string}."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("_") and "=" in line:
            key, value = line.split("=", 1)
            out[key[1:].lower()] = value.strip()
    return out


def convert(value: str):
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("None", "none"):
        return None
    if value.startswith("torch.nn.functional."):
        return getattr(F, value.rsplit(".", 1)[1])
    if value.startswith("torch_geometric.nn."):
        return getattr(pyg_nn, value.rsplit(".", 1)[1])
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def like_default(value, default):
    """`value` in the type of the parameter's documented default.

    A `.env` writes `_DROPOUT=0` for PyGOD's `dropout=0.`; read as the int 0
    it reaches torch_geometric's MLP, which accepts a float or a list and
    raises `TypeError: object of type 'int' has no len()` on an int. So an int
    whose default is a float becomes that float; nothing else is changed.
    """
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def detector_class(method: str):
    """bond_anomalydae -> pygod.detector.AnomalyDAE, matched case-insensitively."""
    wanted = method[len("bond_"):].lower()
    for name in dir(pygod_detector):
        obj = getattr(pygod_detector, name)
        if inspect.isclass(obj) and name.lower() == wanted:
            return obj
    raise SystemExit(f"[ERROR] no PyGOD detector for {method}")


def without_repeated_edges(data):
    """A copy of `data` keeping the first occurrence of each (u, v) pair.

    GraFlag gives GAAN, and only GAAN, the graph without the 6 repeated edges
    of inj_cora (graflag_bond.train.DEDUPLICATE_EDGES), because PyGOD 1.1.0's
    GAAN cannot run on the file as it is. The reference does the same, written
    independently here, so the two score the same graph.
    """
    ei = data.edge_index.numpy()
    key = ei[0].astype(np.int64) * int(data.num_nodes) + ei[1]
    _, first = np.unique(key, return_index=True)
    copy = data.clone()
    copy.edge_index = data.edge_index[:, np.sort(first)]
    return copy


DEDUPLICATED = {"bond_gaan"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--methods-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.dataset_dir)
    data = load_data(data_dir.name, cache_dir=data_dir)
    y = (data.y != 0).long().numpy()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"dataset": data_dir.name, "seed": args.seed, "gpu": args.gpu,
               "num_nodes": int(data.num_nodes), "num_anomalies": int(y.sum()),
               "torch": torch.__version__, "methods": {}}
    for method in args.methods:
        cls = detector_class(method)
        signature = {n: p for n, p in inspect.signature(cls.__init__).parameters.items()
                     if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        kwargs = {k: like_default(convert(v), signature[k].default) for k, v in
                  env_defaults(Path(args.methods_dir) / method / ".env").items()
                  if k in signature}
        accepted = set(signature)
        if "gpu" in accepted:
            kwargs["gpu"] = args.gpu
        # Each detector gets its own copy of the graph. PyGOD's DMGD replaces
        # data.x in place (1,433 Cora features -> a 2,708 x 2,708 matrix) and
        # GADNR adds 2,796 edges and rewrites x, so sharing one Data object
        # handed every detector after them a different graph: DOMINANT and
        # DONE, run after DMGD in one group, scored 0.666 and 0.739 instead
        # of the 0.767 and 0.828 they score alone (bit-identical with
        # GraFlag). GraFlag runs each method in its own process from the file.
        fresh = copy.deepcopy(data)
        graph = without_repeated_edges(fresh) if method in DEDUPLICATED else fresh
        row = {"detector": cls.__name__,
               "kwargs": {k: (v.__name__ if callable(v) else v) for k, v in kwargs.items()},
               "num_edges": int(graph.edge_index.size(1))}
        try:
            seed_everything(args.seed)
            model = cls(**kwargs)
            t0 = time.time()
            model.fit(graph)
            row["fit_s"] = round(time.time() - t0, 2)
            scores = model.decision_score_
            scores = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
            np.save(out / f"{method}.npy", scores.astype(np.float64))
            finite = np.isfinite(scores)
            row["auc_roc"] = float(roc_auc_score(y[finite], scores[finite]))
            row["auc_pr"] = float(average_precision_score(y[finite], scores[finite]))
            row["n_scores"] = int(scores.size)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        summary["methods"][method] = row
        print(json.dumps({method: row}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
