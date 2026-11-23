"""Pinned host-to-GPU copies of batches shaped like GeneralDyG's, no method code.

Each batch is a dict like its collate function returns -- a 128 x 343 x 128
float64 tensor (45 MB), a 128 x 343 float64 mask, and lists of 128 float32
and float64 matrices -- pinned by the DataLoader's own pin_memory() and then
moved with a plain synchronous .to("cuda"), as GeneralDyG's training loop does.
"""
import sys
import torch
from torch.utils.data._utils.pin_memory import pin_memory

torch.manual_seed(0)
dev = torch.device("cuda:0")
batches = int(sys.argv[1]) if len(sys.argv) > 1 else 20
big = sys.argv[2] != "small" if len(sys.argv) > 2 else True


def make_batch():
    nodes = torch.randint(20, 60, (128,)).tolist()
    edges = torch.randint(30, 170, (128,)).tolist()
    b = {
        "input_nodes_feature": [torch.rand(n, 128, dtype=torch.float64) for n in nodes],
        "input_edges_feature": [torch.rand(e, 128, dtype=torch.float64) for e in edges],
        "Tmats": [torch.rand(n, e) for n, e in zip(nodes, edges)],
        "adjs": [torch.rand(n, n) for n in nodes],
        "eadjs": [torch.rand(e, e) for e in edges],
        "mask_edge": torch.rand(128, 343, dtype=torch.float64),
    }
    if big:
        b["input_edges_pad"] = torch.rand(128, 343, 128, dtype=torch.float64)
    return b


def flat(b):
    return {k: (v if isinstance(v, list) else [v]) for k, v in b.items()}


bad = {}
for pin in (True, False):
    n_bad = 0
    for i in range(batches):
        host = make_batch()
        src = pin_memory(host) if pin else host
        moved = {k: [t.to(dev) for t in v] for k, v in flat(src).items()}
        wrong = [k for k in host if not all(torch.equal(m.cpu(), h) for m, h in zip(moved[k], flat(host)[k]))]
        n_bad += bool(wrong)
        if wrong and n_bad <= 3:
            print(f"   {'pinned' if pin else 'pageable'} batch {i}: corrupted {wrong}", flush=True)
    bad[pin] = n_bad
print(f"torch {torch.__version__}, big tensor {big}: pinned {bad[True]} of {batches} batches corrupted, "
      f"pageable {bad[False]} of {batches}", flush=True)
