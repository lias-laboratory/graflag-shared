# bond_guide

PyGOD's
[GUIDE](https://docs.pygod.org/en/latest/generated/pygod.detector.GUIDE.html#pygod.detector.GUIDE)
detector. Higher-order Structure based Anomaly Detection on Attributed
Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

## Parameters

| Key | Value |
|---|---|
| `_HID_A` | `64` |
| `_HID_S` | `4` |
| `_NUM_LAYERS` | `4` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0` |
| `_ACT` | `torch.nn.functional.relu` |
| `_BACKBONE` | `None` |
| `_ALPHA` | `0.5` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_GRAPHLET_SIZE` | `4` |
| `_SELECTED_MOTIF` | `True` |
| `_CACHE_DIR` | `None` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Declared by no other bond method: `_GRAPHLET_SIZE`, `_SELECTED_MOTIF`,
`_CACHE_DIR`.

Different from the rest of the family: `_BACKBONE=None` (the others use
`torch_geometric.nn.GCN`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".
