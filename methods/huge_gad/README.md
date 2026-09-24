# HUGE-GAD

A Label-Free Heterophily-Guided Approach for Unsupervised Graph Fraud
Detection (AAAI 2025).

Upstream: https://github.com/CampanulaBells/HUGE-GAD

## What runs

Upstream's `main.py` at the commit `SOURCE_REF` pins. The method is
**unsupervised and label-free**: it estimates neighbour heterophily without
labels, uses that to build a ranking objective, and distils between an MLP and
a GNN view. No `y` is read during training.

## One method, two dataset forms

HUGE-GAD's `load_mat()` opens `./datasets/<name>.mat`. GraFlag reaches it from
both directions without patching that loader:

- **`gad_amazon`, `gad_facebook`, `gad_yelpchi`** are stored as `.mat` already
  — the same files this repository ships, pinned by commit SHA and `sha256`.
  They are copied straight through, unparsed.
- **`bond_*`** are stored as PyG `.pt`. `graflag_runner.write_mat` renders one
  into `Network`/`Attributes`/`Label` at run time, inside the container.

The second case is what that helper exists for: it makes the `bond_*` library
available to a method that has never heard of PyG, without a second copy of
any dataset being stored anywhere. The conversion is verified in
`graflag_runner/tests/test_mat.py` against ground truth — `Reddit.mat` must
round-trip to `bond_reddit` and `Disney.mat` to `bond_disney`.

**Upstream only accepts four names.** `modules/utils.load_dataset` routes
`Amazon`, `Facebook`, `Reddit` and `YelpChi` through `load_mat` and raises
`Unimplemented dataset` for anything else, so a rendered graph has to be staged
under one of them. `train_graflag.py` warns when the staged name is outside
that set rather than letting the run fail deeper in with less context.

## Hyperparameters

Unusually, upstream's argparse defaults and its README commands agree:
`--lr 5e-4`, `--epoch 300`, `--kd_param 0.5` for Amazon, Facebook, Reddit and
YelpChi alike. There is no per-dataset config table to carry, so the `.env`
simply declares those values.

`AmazonFull` and `YelpChiFull` do differ (`--kd_param 1.0 --epoch 10` and
`--lr 1e-5 --kd_param 3 --epoch 5`), but those are the DGL-format `.zip`
variants and are not registered as GraFlag datasets.

## The patch

`patches/export-scores.patch` adds three lines that write `scores` and
`ano_label` when `GRAFLAG_SCORE_OUT` is set. main.py otherwise computes the
vector, measures `auc_roc` and `auc_prc` on it, prints both and discards it.

Guarded by the environment variable, so the patched tree behaves exactly as
upstream when it is unset.

`main.py` has **CRLF** line terminators and the patch preserves them.
Rewriting the file with LF would have turned a three-line change into a
169-line diff that breaks on any upstream edit — which is what the first
attempt did before it was caught.

## networkx is pinned below 3.0

Same reason as `ad_gcl`: `nx.from_scipy_sparse_matrix()` was removed in
networkx 3.0 in favour of `from_scipy_sparse_array()`. Pinning restores the API
the authors wrote against rather than editing their code.

## Scored split

`all_nodes` — unsupervised and transductive, the same protocol as the `bond_*`
methods, ADA-GAD and AD-GCL, and comparable with them.

## Verification

**Not verified: never run on a cluster.** The definition passes gate 1 (the
contract tests); gates 2 to 4 are outstanding, so HUGE-GAD is defined and
pinned, not integrated.
