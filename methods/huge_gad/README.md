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
`Unimplemented dataset` for anything else. So the method declares exactly the
datasets that reach one of them: the three `gad_*` files, which are those
`.mat`s, and `bond_reddit`, staged as `Reddit` -- the round-trip test above is
what makes that upstream's own Reddit. `stage_dataset` raises for any other
name before upstream starts.

It used to warn instead, and declare `bond_books`, `bond_disney`, `bond_enron`
and `bond_weibo` too, rendering them under their own names. Upstream has no
loader for any of the four, so a run on one could only fail: the first run
here that got past the image build did exactly that, `Exception:
Unimplemented dataset: bond_books`.

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

**Verified on `gad_facebook`** (three-node GPU cluster, 2026-11-17, defaults above,
seed 0, GPU):

| Dataset | Gate 1 | Gate 2 | Gate 3 (`auc_roc`) | Gate 4 |
|---|---|---|---|---|
| `gad_facebook` | contract tests | completed on `node-2`, 31.7 s, 1,896 MB, 703 MB GPU | **0.9766** | 0 failed, 1 warned (`all_nodes`, see above), 3 passed |

`exp__huge_gad__gad_facebook__20261117_143047`. Upstream's own `main.py`
printed `auc_roc: 0.9765530303030303` over the 1,081 nodes (25 anomalous),
and the evaluator computes 0.9766 independently from the published vector;
`graflag verify` checks that the two agree. The four attempts before it
failed, each one step further:

| Run | Failure | Changed since |
|---|---|---|
| `exp__huge_gad__bond_books__20261117_111402` | build: `No matching distribution found for dgl==2.4.0+cu121` -- data.dgl.ai was down, and PyPI has only 0.1.x and 2.1.0 | DGL installed from the DGL team's conda channel, SHA-256 pinned (`install_dgl.py`) |
| `exp__huge_gad__bond_books__20261117_125152` | build: `import dgl.sparse` -- `libcuda.so.1` missing | checked with `ldd` at build time, imported at run time (`require_dgl_sparse`) |
| `exp__huge_gad__bond_books__20261117_130738` | `stage_dataset`: `No module named 'torch_geometric'` unpickling the `.pt` | `torch-geometric==2.6.1` in the image, for loading the dataset only |
| `exp__huge_gad__bond_books__20261117_135034` | `Exception: Unimplemented dataset: bond_books` in upstream's `load_dataset` | `bond_books` is no longer declared; only datasets upstream can load are, and any other is refused before upstream starts |

Not yet run on `gad_amazon`, `gad_yelpchi` or `bond_reddit`.
