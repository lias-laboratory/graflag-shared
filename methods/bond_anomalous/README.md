# bond_anomalous

PyGOD's
[ANOMALOUS](https://docs.pygod.org/en/latest/generated/pygod.detector.ANOMALOUS.html#pygod.detector.ANOMALOUS)
detector. A Joint Modeling Approach for Anomaly Detection on Attributed
Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

Its parameter set is identical to `bond_radar`'s: no backbone, no batching, no
layer count. Both are matrix-factorisation detectors rather than neural ones.

## Parameters

| Key | Value |
|---|---|
| `_GAMMA` | `1` |
| `_WEIGHT_DECAY` | `0` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_CONTAMINATION` | `0.1` |
| `_VERBOSE` | `0` |

`_CONTAMINATION` and `_VERBOSE` are accepted and recorded but change nothing
in `results.json` -- see BOND.md, "`_CONTAMINATION` does nothing to the
results".
