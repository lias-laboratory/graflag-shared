# bond_conad

PyGOD's
[CONAD](https://docs.pygod.org/en/latest/generated/pygod.detector.CONAD.html#pygod.detector.CONAD)
detector. Contrastive Attributed Network Anomaly Detection.

How a bond method is wired -- how `METHOD_NAME` picks the detector, how
`_FOO` becomes a constructor argument, what the scores mean, which
parameters do nothing -- is in [BOND.md](../BOND.md). This file is only
what is specific to this one.

## Parameters

| Key | Value |
|---|---|
| `_HID_DIM` | `64` |
| `_NUM_LAYERS` | `4` |
| `_DROPOUT` | `0` |
| `_WEIGHT_DECAY` | `0` |
| `_ACT` | `torch.nn.functional.relu` |
| `_SIGMOID_S` | `False` |
| `_BACKBONE` | `torch_geometric.nn.GCN` |
| `_CONTAMINATION` | `0.1` |
| `_LR` | `0.004` |
| `_EPOCH` | `100` |
| `_GPU` | `0` |
| `_BATCH_SIZE` | `0` |
| `_NUM_NEIGH` | `-1` |
| `_WEIGHT` | `0.5` |
| `_ETA` | `0.5` |
| `_MARGIN` | `0.5` |
| `_R` | `0.2` |
| `_M` | `50` |
| `_K` | `50` |
| `_F` | `10` |
| `_VERBOSE` | `0` |
| `_SAVE_EMB` | `False` |
| `_COMPILE_MODEL` | `False` |

Declared by no other bond method: `_MARGIN`, `_R`, `_M`, `_F`.

`_CONTAMINATION`, `_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are accepted
and recorded but change nothing in `results.json` -- see BOND.md,
"`_CONTAMINATION` does nothing to the results".

## It trains as DOMINANT

In the PyGOD commit `bond_base` pins (`c84dcca`), CONAD's contrastive term
is a constant. `pygod/detector/conad.py`, line 215:

```python
margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
```

`MarginRankingLoss(x1, x2, y)` is `max(0, -y * (x1 - x2) + margin)`. With
`x1` and `x2` both `h`, that is `margin` for every node, and its gradient
with respect to `h` and to `h_aug` is zero (checked with torch: loss 0.5,
both gradients 0.0). The released PyGOD 1.1.0 has the same line. Nothing the augmentation produces reaches the weights.
The model is DOMINANT's (`DOMINANTBase`), the score is DOMINANT's
reconstruction error, and the loss is that error times `_ETA` plus a
constant -- which Adam, being invariant to the scale of the loss up to its
epsilon, follows almost exactly as it follows DOMINANT's.

On the cluster, `bond_inj_cora`, seed and defaults alike:

| | AUC-ROC | AUC-PR | Largest score difference |
|---|---|---|---|
| `exp__bond_conad__bond_inj_cora__20261117_122351` | 0.7673 | 0.1841 | |
| `exp__bond_dominant__bond_inj_cora__20261117_121219` | 0.7673 | 0.1841 | 9.7e-5 against CONAD |

So `_MARGIN`, `_R`, `_M`, `_K` and `_F`, which only shape that term, change
nothing that is learned, and `_ETA` only rescales the loss. GraFlag runs
PyGOD's CONAD faithfully -- its scores are PyGOD's own -- but the number it
publishes is DOMINANT's, not the CONAD paper's method. A comparison that
ranks the two is ranking one detector against itself.
