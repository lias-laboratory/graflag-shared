# bond_one

PyGOD's
[ONE](https://docs.pygod.org/en/latest/generated/pygod.detector.ONE.html#pygod.detector.ONE)
detector. Outlier Aware Network Embedding for Attributed Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

`_EPOCH=5` where every other trained detector here uses 100. Read its numbers
with that in mind before comparing them with the rest of the family.

## Parameters

| Key | Value |
|---|---|
| `_HID_A` | `36` |
| `_HID_S` | `36` |
| `_ALPHA` | `1` |
| `_BETA` | `1` |
| `_GAMMA` | `1` |
| `_WEIGHT_DECAY` | `0` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `5` |
| `_GPU` | `0` |
| `_VERBOSE` | `0` |

Different from the rest of the family: `_ALPHA=1` (the others use `0.5`),
`_EPOCH=5` (the others use `100`).

`_CONTAMINATION` and `_VERBOSE` are accepted and recorded but change nothing
in `results.json` -- see BOND.md, "`_CONTAMINATION` does nothing to the
results".
