# bond_gaan

PyGOD's
[GAAN](https://docs.pygod.org/en/latest/generated/pygod.detector.GAAN.html#pygod.detector.GAAN)
detector. Generative Adversarial Attributed Network Anomaly Detection.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

## Parameters

| Key | Value |
|---|---|
| `_HID_DIM` | `64` |
| `_NUM_LAYERS` | `4` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0` |
| `_ACT` | `torch.nn.functional.relu` |
| `_BACKBONE` | `None` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_WEIGHT` | `0.5` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Different from the rest of the family: `_BACKBONE=None` (the others use
`torch_geometric.nn.GCN`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".

## Runs on a deduplicated graph -- unlike the other sixteen

GAAN is the one bond method that is **not** given the dataset file exactly
as it is: `graflag_bond` removes repeated edges from the graph before GAAN
sees it (`DEDUPLICATE_EDGES` in `libs/graflag_bond/train.py`). Every other
`bond_*` method runs on the file unchanged. Read GAAN's row in a results
table with that in mind: it was measured on a graph a handful of edges
smaller than the one the others were measured on.

| Dataset | Edges in the file | Repeated (u, v) pairs removed |
|---|---|---|
| `bond_inj_cora` | 11,060 | 6 |
| `bond_gen_100` | 618 | 4 |

Every run records what was done, under `metadata.summary.dataset_info` in
`results.json`: `num_edges_in_file` and `duplicate_edges_removed` (and
`num_edges` is then the count GAAN trained on). The removal keeps the first
occurrence of each pair and the original order; `(u, v)` and `(v, u)` are
different pairs, so an undirected graph stored both ways is left as it is. A
graph with edge attributes or weights is refused rather than merged.

### What it scores

It runs, and the number it gives is near chance. On a three-node GPU cluster
(2026-11-17, `bond_inj_cora`, seed 42, GPU) GraFlag's run published AUC-ROC
0.498 and passed `graflag verify`. PyGOD's `GAAN` called directly, outside
GraFlag, with the same constructor arguments and the same deduplicated graph
(`validation/reference/gaan_direct.py`, CPU) gave 0.4980 for seed 42 and
0.4750 for seed 1 -- so the 0.5 is PyGOD 1.1.0's GAAN with these defaults on
this graph, not something the integration adds. The BOND paper reports
74.2 +/- 0.9 for GAAN on `inj_cora`, as a mean over 20 trials with
hyperparameters drawn from a grid, on an earlier PyGOD; this file's defaults
are one point, and the detector code has changed since.

### Why

Without it GAAN cannot run on these graphs at all, on the GPU or the CPU,
for an upstream reason. On the CPU the message is legible:

    pygod/detector/gaan.py:185, in GAAN.forward_model
    pygod/nn/functional.py:85, in double_recon_loss
    RuntimeError: all elements of target should be between 0 and 1

On the GPU the same defect surfaces as a device-side assert, the same check
inside CUDA (seen on a three-node GPU cluster, 2026-11-17):

    Loss.cu:95: operator(): block: [2430,0,0], thread: [108,0,0]
    Assertion `target_val >= zero && target_val <= one` failed.

The target that is out of range is the dense adjacency matrix. GAAN is the
only detector in the family that hands it to `binary_cross_entropy` as a
*target* rather than comparing scores against it, so it is the only one that
requires the adjacency to be strictly 0/1 -- and the published BOND graphs
are not:

    >>> data = torch.load('gen_100.pt')          # the dataset, one graph
    >>> data.edge_index.shape[1]                 # 618 edges
    >>> torch.unique(data.edge_index, dim=1).shape[1]   # 614 unique
    >>> to_dense_adj(data.edge_index).max()      # tensor(2.)

A pair that appears twice becomes a cell of `2` once `to_dense_adj` sums
duplicates, and a `2` is not a probability.

The dataset files themselves are unchanged: `datasets/bond_gen_100` and
`datasets/bond_inj_cora` are downloaded verbatim from the BOND benchmark's
own archives (`https://github.com/pygod-team/data`). The deduplication
happens in memory, for GAAN only. It was limited to GAAN deliberately:
applying it to the whole family would have changed the graph under sixteen
methods whose results had already been run and verified on the file as
published. The fix that would remove the difference altogether is upstream,
in `double_recon_loss`, which should coalesce (or clamp) its target.
