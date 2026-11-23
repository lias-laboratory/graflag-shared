# DiffGAD

A Diffusion-based Unsupervised Graph Anomaly Detector (ICLR 2025).

Upstream: https://github.com/fortunato-all/LJH_DiffGAD

## What runs

Upstream's code at the commit `SOURCE_REF` pins. `train_graflag.py` constructs
the authors' `DiffGAD` transform and calls it; it does not reimplement the
method. The method is **unsupervised** — no label is used to train the
autoencoder or the diffusion model.

## The authors' per-dataset configuration is loaded

Upstream ships one tuned config per dataset under `configs/` and `main.py`
reads it before constructing anything. The values differ materially between
datasets -- `books` uses `ae_alpha: 0.5` and `weight: 2.0`, `weibo` uses `0.8`
and `1.0` -- so `train_graflag.py` loads `configs/<dataset>.yaml` the same way
rather than running on generic defaults. A missing config raises, naming the
datasets upstream ships one for; publishing a number from an untuned
configuration would not be publishing DiffGAD.

`hid_dim` is deliberately empty in every shipped yaml, because
`DiffGAD.forward()` derives it from the feature count
(`2 ** int(log2(num_features) - 1)`). It is forwarded as `None` so that
derivation still happens. The `.env` therefore declares only the knobs the
authors' config does not own; `--params AE_ALPHA=0.5` still wins, applied on
top of the loaded config.

## The dataset is staged, not converted

`forward()` calls `pygod.utils.load_data(self.dataset)` with no `cache_dir`,
which resolves to `~/.pygod/data/<name>.pt`. The integration copies the mounted
GraFlag dataset there under the bare name upstream uses (`bond_weibo` becomes
`weibo.pt`) rather than patching upstream's data path.

`stage_dataset()` raises when the mounted directory holds no `.pt`, because
`load_data()` **downloads** a dataset that is absent — a silent failure would
train on an unpinned copy from the internet while looking entirely normal.

## The published number is not a clean held-out measurement

This is the thing to read before comparing DiffGAD's AUC with anything else.

`sample()` walks 500 diffusion timesteps. At each one it decodes, computes a
per-node score, and measures that score's AUC **against the labels**. What it
reports is `np.max` over those 500 AUCs, and the scores this integration
publishes are the ones from `argmax` of the same list.

So the timestep is selected using the labels the AUC is then measured on. There
is no validation split anywhere in the method to select it on instead. That is
upstream's protocol and changing it would change what DiffGAD does, so it
stands — but the number is an upper bound over 500 candidates, not a held-out
measurement, and comparing it against a method that selects nothing on test is
not comparing like with like.

`metadata.summary.training_info` records `selected_timestep` and
`timestep_selected_on: "test_labels"` so the property travels with the result
instead of living only here, and `auc_roc`, the AUC upstream printed for the
published scores, which is what `graflag verify` checks them against.

**And it is one trial of twenty.** `forward()` runs `sample()` in a loop of
`num_trial = 20`: each trial retrains both diffusion models on the shared
autoencoder, selects its own best timestep against the labels, and contributes
that maximum. What upstream prints, and what its paper reports, is the mean
and standard deviation over the 20 maxima, with the best one in parentheses
(`Final AUC: 0.6715±0.0230 (0.7228)` on `bond_books`). `selected_scores` is
overwritten by every trial, so the scores GraFlag publishes are the **last
trial's**, at its selected timestep. Their AUC is that trial's maximum, one
draw from the distribution upstream averages -- not upstream's headline
number, and not its best trial. `training_info.trial_auc_roc` records all 20
maxima and `published_trial` which one was published, so a reader can compute
upstream's summary and see where the published trial sits in it.

## The patch

`patches/expose-scores.patch` adds five lines to `sample()` -- a list, an
append, and three assignments onto `self` (the selected timestep, its scores
and the AUC upstream measured for them) -- and one to `forward()`, which keeps
the 20 per-trial AUCs before they become a tensor. Upstream keeps only
`np.max` of the metrics and discards every score vector behind them; GraFlag
publishes the scores a method computed rather than recomputing them.

Applied with `git apply --verbose`, which exits non-zero when the context no
longer matches, so upstream drift fails the build rather than producing an
image whose `selected_scores` is never set.

## No CPU path

DiffGAD calls `.cuda()` unconditionally inside its own sampling loop, so
`_GPU=-1` raises with a message saying so instead of failing later inside the
authors' code with a CUDA error. `graflag run --no-gpu` sets `_GPU=-1` for
DiffGAD as for every method that declares `_GPU`, so that is the run this
check stops: one scheduled without a GPU, which DiffGAD cannot use.

## Labels

BOND encodes the outlier type in the label bits — 1 contextual, 2 structural,
3 both — so `y` is not binary. The published `ground_truth` is `y > 0`, the
convention BOND and `graflag_bond` use.

## Verification

**Verified on `bond_books`** (three-node GPU cluster, 2026-11-17, defaults
above, the authors' `configs/books.yaml`, seed 42, GPU):

| Gate | Result |
|---|---|
| 1. Contract | `tests/test_methods.py` passes |
| 2. Build and run | `exp__diffgad__bond_books__20261117_111400`: completed in 2,253.7 s (20 trials of 500 timesteps), peak RAM 1,337 MB, peak GPU 175 MB |
| 3. `graflag evaluate` | AUC-ROC **0.6743**, AUC-PR 0.0403, over all 1,418 nodes (28 anomalous) |
| 4. `graflag verify` | 0 failed, 2 warned, 2 passed. One warning is `scored_split: all_nodes` (see "Labels" above). The other is that the run records no AUC of its own to cross-check against |

That cross-check was done by hand from the run's log instead: in the 20th
trial, at timestep 92 -- the `selected_timestep` the run recorded -- upstream
printed `pyg_AUC: 0.6743`, the evaluator's number to the four digits upstream
prints. Upstream's own summary for the same run was `Final AUC:
0.6715±0.0230 (0.7228)`; see "And it is one trial of twenty" for why the
published number is neither of those.

That run predates `training_info.auc_roc`. The image rebuilt with it ran the
same configuration on 2026-11-18 (`exp__diffgad__bond_books__20261118_200637`,
node-2, 2,262.4 s, peak RAM 1,342 MB, peak GPU 175 MB; record
`validation/results/diffgad-auc-record.json`), and `graflag verify` now makes
the check itself: 0 failed, 1 warned (`all_nodes`), 3 passed, "evaluator AUC
0.6743 matches the method's training_info.auc_roc".

What that run records about its trials:

| | |
|---|---|
| Published trial | 20 of 20, selected timestep 92 |
| Its AUC-ROC | 0.6743 (upstream measured 0.674293) |
| The 20 trials | mean 0.6729, standard deviation 0.0253, min 0.6427, max 0.7355 |
| Rank of the published trial | 9th of 20 |

Against the first run, same seed: the same selected timestep and the same AUC
to four digits, but not the same scores -- they differ by up to 0.047 -- and
not the same trials (upstream's summary then was 0.6715 +/- 0.0230, max
0.7228). DiffGAD on a GPU does not repeat bit for bit, so its published number
is one draw twice over: one trial of twenty, from one run of many.

Not yet done: the other datasets upstream ships configs for.
