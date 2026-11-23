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

## DMGD changes the graph it is given

PyGOD's `DMGD.fit()` replaces `data.x` in place: on `bond_inj_cora` the 1,433
feature columns are 2,708 when it returns. Inside GraFlag that reaches no
other detector, since each run is its own process reading the dataset file,
but it reached two other places:

- **This run's own metadata.** `graflag_bond` described the graph after
  `fit()`, so `results.json` reported `num_features: 2708` for Cora
  (`exp__bond_dmgd__bond_inj_cora__20261117_131720`). It now records the
  graph's size and labels before training (`describe_graph`,
  `graflag_bond/tests/test_describe.py`). The scores were never affected.
- **A reference that shares one graph.** The direct-PyGOD check
  (`validation/reference/pygod_direct.py`) fitted several detectors on one
  `Data` object, so DOMINANT and DONE, fitted after DMGD, trained on DMGD's
  matrix and disagreed with GraFlag. It now gives each detector a copy. Any
  script that benchmarks several PyGOD detectors in one process has the same
  trap.

Run alone on its own copy, PyGOD's DMGD agrees with GraFlag's run to within
1.6e-6 (AUC 0.76010 both), and that is as close as DMGD comes to itself: two
direct runs with the same seed, on two CPU nodes, differ by up to 2.1e-6 on
2,644 of the 2,708 scores. The sixteen other PyGOD detectors match GraFlag
bit for bit (`validation/results/reference/pygod_direct_compare.json`).
