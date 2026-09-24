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
instead of living only here.

## The patch

`patches/expose-scores.patch` adds three lines to `sample()`: a list, an
append, and two assignments onto `self`. Upstream keeps only `np.max` of the
metrics and discards every score vector behind them; GraFlag publishes the
scores a method computed rather than recomputing them.

Applied with `git apply --verbose`, which exits non-zero when the context no
longer matches, so upstream drift fails the build rather than producing an
image whose `selected_scores` is never set.

## No CPU path

DiffGAD calls `.cuda()` unconditionally inside its own sampling loop, so
`_GPU=-1` raises with a message saying so instead of failing later inside the
authors' code with a CUDA error. `--no-gpu` does not rewrite `_GPU` for
non-`bond_*` methods, so this is the only place that check can live.

## Labels

BOND encodes the outlier type in the label bits — 1 contextual, 2 structural,
3 both — so `y` is not binary. The published `ground_truth` is `y > 0`, the
convention BOND and `graflag_bond` use.
