# bond_scan

PyGOD's
[SCAN](https://docs.pygod.org/en/latest/generated/pygod.detector.SCAN.html#pygod.detector.SCAN)
detector. Structural Clustering Algorithm for Networks.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

SCAN is a structural clustering algorithm, not a trained model, which is why
this is the only bond method with no `_GPU`, no `_LR` and no `_EPOCH`. Run it
with `--no-gpu` so Swarm does not reserve a card it cannot use.

## Parameters

| Key | Value |
|---|---|
| `_EPS` | `.5` |
| `_MU` | `2` |
| `_CONTAMINATION` | `0.1` |
| `_VERBOSE` | `0` |

Declared by no other bond method: `_MU`.

`_CONTAMINATION` and `_VERBOSE` are accepted and recorded but change nothing
in `results.json` -- see BOND.md, "`_CONTAMINATION` does nothing to the
results".
