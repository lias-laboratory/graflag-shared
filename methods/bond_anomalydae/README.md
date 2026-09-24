# bond_anomalydae

PyGOD's
[ANOMALYDAE](https://docs.pygod.org/en/latest/generated/pygod.detector.AnomalyDAE.html#pygod.detector.AnomalyDAE)
detector. Dual Autoencoder for Anomaly Detection on Attributed Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

## Parameters

| Key | Value |
|---|---|
| `_EMB_DIM` | `64` |
| `_HID_DIM` | `64` |
| `_NUM_LAYERS` | `4` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0` |
| `_ACT` | `torch.nn.functional.relu` |
| `_BACKBONE` | `None` |
| `_ALPHA` | `0.5` |
| `_THETA` | `1` |
| `_ETA` | `1` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Declared by no other bond method: `_EMB_DIM`, `_THETA`.

Different from the rest of the family: `_BACKBONE=None` (the others use
`torch_geometric.nn.GCN`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".
