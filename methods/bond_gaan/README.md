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

## Does not run on `bond_gen_100` or `bond_inj_cora`

GAAN fails on both datasets it was tried on, on the GPU and on the CPU, for
the same upstream reason. On the CPU the message is legible:

    pygod/detector/gaan.py:185, in GAAN.forward_model
    pygod/nn/functional.py:85, in double_recon_loss
    RuntimeError: all elements of target should be between 0 and 1

On the GPU the same defect surfaces as a device-side assert, which is the
same check inside CUDA:

    Loss.cu:95: operator(): block: [0,0,0], thread: [0,0,0]
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

Four edges appear twice, `to_dense_adj` sums duplicates, and a cell of `2`
is not a probability. `bond_inj_cora` has the same property.

This is data, not integration. `datasets/bond_gen_100` is downloaded
verbatim from `https://github.com/pygod-team/data/raw/main/gen_100.pt.zip`,
the BOND benchmark's own published archive, and GraFlag does not modify it.
Deduplicating the edge index here -- in the dataset loader or in
`graflag_bond` -- would silently benchmark GAAN on a graph the other
sixteen methods did not see, which is worse than reporting it as untested.

So it is **flagged, not fixed**: `bond_gaan` has no result in the matrix.
Two ways out, neither taken here because both are decisions about the
benchmark rather than about GraFlag:

- fix it upstream (`double_recon_loss` should clamp or `coalesce` its
  target), or
- publish a deduplicated variant of the affected datasets as a separate
  dataset name, so the change is visible in the results table.
