# bond_card

PyGOD's
[CARD](https://docs.pygod.org/en/latest/generated/pygod.detector.CARD.html#pygod.detector.CARD)
detector. Community-Guided Contrastive Learning with Anomaly-Aware
Reconstruction for Anomaly Detection on Attributed Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

CARD is the most recently added detector of the seventeen, so it is the one
most likely to be missing from an older PyGOD. If the run fails at
`Unsupported detector: card`, the installed PyGOD predates it -- the `.env` is
not at fault.

## Parameters

| Key | Value |
|---|---|
| `_HID_DIM` | `64` |
| `_NUM_LAYERS` | `2` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0` |
| `_ACT` | `torch.nn.functional.relu` |
| `_BACKBONE` | `torch_geometric.nn.GCN` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_SUBGRAPH_NUM_NEIGH` | `4` |
| `_FP` | `0.6` |
| `_GAMA` | `0.5` |
| `_ALPHA` | `0.1` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Declared by no other bond method: `_SUBGRAPH_NUM_NEIGH`, `_FP`, `_GAMA`.

Different from the rest of the family: `_NUM_LAYERS=2` (the others use `4`),
`_ALPHA=0.1` (the others use `0.5`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".
