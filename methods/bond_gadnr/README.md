# bond_gadnr

PyGOD's
[GADNR](https://docs.pygod.org/en/latest/generated/pygod.detector.GADNR.html#pygod.detector.GADNR)
detector. Graph Anomaly Detection via Neighborhood Reconstruction.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

The only bond method that does not use the family's `_LR=0.004` and
`_WEIGHT_DECAY=0`; it declares its own decoder depths and neighbour loss
instead. Those are PyGOD's defaults for this detector, not GraFlag choices.

## Parameters

| Key | Value |
|---|---|
| `_HID_DIM` | `64` |
| `_NUM_LAYERS` | `1` |
| `_DEG_DEC_LAYERS` | `4` |
| `_FEA_DEC_LAYERS` | `3` |
| `_BACKBONE` | `torch_geometric.nn.GCN` |
| `_SAMPLE_SIZE` | `2` |
| `_SAMPLE_TIME` | `3` |
| `_NEIGH_LOSS` | `KL` |
| `_LAMBDA_LOSS1` | `1e-2` |
| `_LAMBDA_LOSS2` | `1e-3` |
| `_LAMBDA_LOSS3` | `1e-4` |
| `_REAL_LOSS` | `True` |
| `_LR` | `0.01` |
| `_EPOCH` | `100` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0.0003` |
| `_ACT` | `torch.nn.functional.relu` |
| `_GPU` | `-1` (CPU -- see below) |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_CONTAMINATION` | `0.1` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Declared by no other bond method: `_DEG_DEC_LAYERS`, `_FEA_DEC_LAYERS`,
`_SAMPLE_SIZE`, `_SAMPLE_TIME`, `_NEIGH_LOSS`, `_LAMBDA_LOSS1`,
`_LAMBDA_LOSS2`, `_LAMBDA_LOSS3`, `_REAL_LOSS`.

Different from the rest of the family: `_NUM_LAYERS=1` (the others use `4`),
`_LR=0.01` (the others use `0.004`), `_WEIGHT_DECAY=0.0003` (the others use
`0`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".

## Runs on the CPU

`_GPU=-1` is not a performance choice. At the PyGOD commit this image pins
(`c84dcca`), GAD-NR cannot run on a GPU at all:

    pygod/nn/gadnr.py:337, in forward
    RuntimeError: Expected all tensors to be on the same device, but found
                  at least two devices, cuda:0 and cpu! (when checking
                  argument for argument mat1 in method wrapper_CUDA_addmm)

One of the model's submodules is never moved to the device the rest of the
model is on. The fix belongs upstream, so it is declared in `.env` rather
than patched into `graflag_bond`.

Verified on `bond_gen_100`: fails on GPU, completes on CPU
(`auc_roc` 0.7981). The graphs this family is benchmarked on are small
enough that CPU is not a constraint.

Drop the line if a later PyGOD release fixes it -- and re-read this file
when bumping `SOURCE_REF` in `images/bond_base/Dockerfile`.
