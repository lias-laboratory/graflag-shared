# ADA-GAD

Anomaly-Denoised Autoencoders for Graph Anomaly Detection (AAAI 2024).

Upstream: https://github.com/jweihe/ADA-GAD

## What runs

Upstream's code, at the commit `SOURCE_REF` pins. `train_graflag.py` calls
upstream's own `build_args()` and `main()`; it does not reimplement the method.
Parameters declared in `.env` are written onto upstream's argparse namespace
with `apply_params()`, so the accepted names and their defaults are upstream's
and `.env` only lists the subset GraFlag overrides.

The method is **unsupervised**: no label is read during training, and nothing
in this integration selects a model using `y`.

## The authors' per-dataset configuration is loaded, and has to be

`main.py:264` loads `config_ada-gad.yml` when `--use_cfg` is passed, and that
file holds a tuned section per dataset (`inj_cora`, `weibo`, `books`, `disney`,
`enron`, `reddit`, `inj_amazon`). Every number in the paper comes from that
path, so `train_graflag.py` calls `load_best_configs()` the same way rather than
running on the argparse defaults.

Skipping it does not merely leave the method untuned. The argparse default for
`--activation` is `prelu`, and the class at `pygod/models/basic_nn.py:218`
resolves a string activation with `eval('F.' + act)` -- so `prelu` becomes the
*functional* `F.prelu`, which takes a mandatory `weight`, and the first forward
pass dies with `prelu() missing 1 required positional arguments: "weight"`.
Every dataset section in the yml sets an activation that is not `prelu`
(`inj_cora` uses `relu`), so the crash only ever appears when the config is
bypassed. The same file has `create_activation()` mapping `"prelu"` to
`torch.nn.PReLU()`, and `Vanilla_GCN` does the same -- the `eval('F.'+act)`
branch is simply the one place that disagrees.

`load_best_configs()` returns `args` unchanged and logs "Best args not found"
when the dataset has no section, which is a fail-open: the run would continue
on defaults and crash, or worse, not crash and publish an untuned number.
`train_graflag.py` raises instead.

Order matters: the authors' config is applied first, then GraFlag's `--params`
on top, so an explicit override still wins.

## Known upstream quirk

`pygod/models/adanet.py:447` is `assert(False,'wrong loss func')` -- a tuple,
which is always truthy, so that error path can never fire. Left alone: it is
upstream's code and changing it would change which inputs the method rejects.

## The dataset is staged, not converted

ADA-GAD calls `pygod.utils.load_data(name)` with no `cache_dir`, which resolves
to `~/.pygod/data/<name>.pt`. Rather than redirect that call — which would mean
patching upstream's data path, a change to what runs — the integration copies
the mounted GraFlag dataset there under the bare name upstream uses
(`bond_weibo` is staged as `weibo.pt`).

`stage_dataset()` raises when the mounted directory holds no `.pt`. That matters
more than it looks: `load_data()` **downloads** the dataset when the file is
absent, so a failure to stage would not error — it would quietly train on an
unpinned copy fetched from the internet instead of the graph GraFlag mounted,
and the run would look entirely normal.

## The patch

`patches/expose-scores.patch` adds three lines: a list, an append, and a
`return`. Upstream's `main()` computes `final_outlier` — the per-node score
vector `god_evaluation()` produced — prints an AUC over it and discards it.
GraFlag publishes the scores the method computed rather than recomputing them,
so `main()` has to hand them back.

It is applied with `git apply --verbose`, which exits non-zero when the context
no longer matches. A `sed -i` here would exit 0 on no match and produce an image
whose `main()` returns `None`, which fails much later and much less clearly.

Nothing the method computes changes: the returned scores are the same tensor the
printed AUC was measured on.

## Vendored PyGOD

The repository ships its own PyGOD under `src/pygod` and imports
`pygod.metrics`, which the released PyGOD renamed to `pygod.metric`. The
Dockerfile installs the vendored copy. Installing the published wheel instead
would import cleanly and fail at the first metric call.

## Labels

BOND encodes the outlier type in the label bits — 1 contextual, 2 structural,
3 both — so `graph.y` is not binary. The published `ground_truth` is `y > 0`,
which is the convention `graflag_bond` and the BOND paper use for the AUC.

## Scored split: all nodes, and what that does and does not mean

`metadata.summary.dataset_info.scored_split` is `all_nodes`, and
`graflag verify` warns about it. The warning is right to ask, so: ADA-GAD *does*
train, and it scores the same nodes it trained on.

That is not label leakage. Training is unsupervised -- reconstruction over a
denoised graph -- and no `y` is read until the AUC is computed. It is
transductive, which is the standard protocol for unsupervised GAD and exactly
what BOND does and what all seventeen `bond_*` methods in this repository do:
fit the whole graph, score the whole graph, measure AUC over every node. The
number is therefore comparable with them and is *not* comparable with a
method that holds nodes out.

What it is not is an estimate of performance on unseen nodes. Nothing here
measures that, and no split in the shipped BOND graphs would support it: only
three of the fourteen carry masks at all, and `bond_weibo`'s val and test
splits contain no anomalies whatsoever.

`_SEEDS` > 1 runs upstream's multi-seed loop; the published scores are the
first seed's, and every seed's upstream AUC is recorded in `runs.csv`.

## Verification

Smoke run, `bond_inj_cora`, `MAX_EPOCH=2 MAX_EPOCH_F=2 SEEDS=1`:

| Gate | Result |
|---|---|
| 1. Contract | 78 tests pass |
| 2. Build and run | `completed`, exit 0, 16.6 s, peak 1408 MB / 821 MB GPU |
| 3. Evaluation | `auc_roc` 0.8466, `auc_pr` 0.2017 |
| 4. Result integrity | 0 failed, 1 warned (the `all_nodes` note above), 3 passed |

The evaluator's 0.8466 equals upstream's own printed `final_auc: 84.66`, which
is the check that matters: the published scores are the vector the method
measured itself on, not a recomputation. 2708 nodes, 138 anomalous (5.10%).

These numbers come from a two-epoch smoke run and are not ADA-GAD's reported
performance; they demonstrate the integration runs end to end.
