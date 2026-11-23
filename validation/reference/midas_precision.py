"""Where GraFlag's MIDAS scores and upstream's Score.txt differ, and by how much relative to the score."""
import json, sys
import numpy as np
from scipy.stats import rankdata
exp, ref = sys.argv[1], sys.argv[2]
g = np.asarray(json.load(open(f"{exp}/results.json"))["scores"], dtype=np.float64).ravel()
r = np.loadtxt(ref, dtype=np.float64)
d = np.abs(g - r)
nz = d > 0
rel = d[nz] / np.maximum(np.abs(r[nz]), 1e-300)
print(json.dumps({
    "max_score": float(r.max()), "median_score": float(np.median(r)),
    "max_rel_diff": float(rel.max()) if rel.size else 0.0,
    "p99_rel_diff": float(np.quantile(rel, 0.99)) if rel.size else 0.0,
    "float32_equal": bool(np.array_equal(g.astype(np.float32), r.astype(np.float32))),
    "rel_within_float32_eps": bool((rel <= np.finfo(np.float32).eps).all()) if rel.size else True,
    "ranks_identical": bool(np.array_equal(rankdata(g), rankdata(r))),
    "examples": [[float(g[i]), float(r[i])] for i in np.flatnonzero(nz)[:3]],
}))
