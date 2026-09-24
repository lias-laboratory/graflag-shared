# bond_dmgd

PyGOD's
[DMGD](https://docs.pygod.org/en/latest/generated/pygod.detector.DMGD.html#pygod.detector.DMGD)
detector. Deep Multiclass Graph Description.

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
| `_GPU` | `-1` (CPU -- see below) |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_ALPHA` | `1` |
| `_BETA` | `1` |
| `_GAMMA` | `1` |
| `_WARMUP` | `2` |
| `_K` | `2` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Different from the rest of the family: `_NUM_LAYERS=2` (the others use `4`),
`_ALPHA=1` (the others use `0.5`).

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".

## Runs on the CPU

`_GPU=-1` is not a performance choice. At the PyGOD commit this image pins
(`c84dcca`), DMGD cannot run on a GPU at all:

    pygod/nn/dmgd.py:168
        KMeans(n_clusters=self.k, n_init='auto').fit(emb.detach())
    TypeError: can't convert cuda:0 device type tensor to numpy.
               Use Tensor.cpu() to copy the tensor to host memory first.

DMGD clusters its embedding with scikit-learn each epoch, and scikit-learn
cannot read a CUDA tensor. The missing `.cpu()` is upstream's, so it is
declared in `.env` rather than patched into `graflag_bond` -- a wrapper that
silently moved tensors for one detector would be a second place where device
handling lives, and the next PyGOD bump would leave it there unnoticed.

Verified on `bond_gen_100`: fails on GPU, completes on CPU
(`auc_roc` 0.2087). The graphs this family is benchmarked on are small
enough that CPU is not a constraint.

Drop the line if a later PyGOD release fixes it -- and re-read this file
when bumping `SOURCE_REF` in `images/bond_base/Dockerfile`.
