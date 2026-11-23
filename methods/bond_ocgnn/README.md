# bond_ocgnn

PyGOD's
[OCGNN](https://docs.pygod.org/en/latest/generated/pygod.detector.OCGNN.html#pygod.detector.OCGNN)
detector. One-Class Graph Neural Networks for Anomaly Detection in Attributed
Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

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
| `_BETA` | `0.5` |
| `_WARMUP` | `2` |
| `_EPS` | `0.001` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Different from the rest of the family: `_NUM_LAYERS=2` (the others use `4`),
`_BETA=0.5` (the others use `1`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".

## Constant scores on `bond_inj_cora`

With the defaults above, OCGNN gives every node of `bond_inj_cora` the same
score, so its AUC is exactly 0.5 and ranks nothing. `graflag verify` fails the
run for that reason (`every score is the same value`), which is the intended
outcome: the 0.5 is not a measurement.

It is PyGOD's behaviour, not the integration's. On a three-node GPU cluster
(2026-11-17) the GraFlag run on a GPU published -4.03e-05 for all 2,708
nodes, and PyGOD's `OCGNN` called directly, outside GraFlag
(`validation/reference/pygod_direct.py`, same constructor arguments, CPU),
gave AUC 0.5 for seeds 42, 1 and 7 alike. On a CPU the two are
bit-identical: `exp__bond_ocgnn__bond_inj_cora__20261117_134733` and the
direct fit publish the same constant vector
(`validation/results/reference/pygod_direct_compare.json`). One-class objectives have a known
trivial solution -- every embedding collapses onto the hypersphere centre --
and these defaults reach it on this graph. On `bond_gen_100` the same
defaults gave 0.7581 (devcluster, before runs were seeded).
