# AD-GCL

Revisiting Graph Contrastive Learning on Anomaly Detection from a Structural
Imbalance Perspective (AAAI 2025).

Upstream: https://github.com/yimingxu24/AD-GCL

## What runs

Upstream's `AD-GCL/run.py` at the commit `SOURCE_REF` pins. The method is
**unsupervised**: contrastive learning over RWR subgraphs, with no label read
during training.

## The dataset is used as stored, not converted

AD-GCL's `load_mat()` does `sio.loadmat("./Data/<name>.mat")`. GraFlag stores
these six datasets as exactly that `.mat` — `Network`, `Attributes`, `Label` —
so `stage_dataset()` copies the mounted file to where upstream looks and
nothing is converted, parsed or re-encoded. That is why they are registered in
`.mat` form rather than as PyG `.pt`.

The six ship with the repository and are pinned by commit SHA plus `sha256` in
each dataset's `metadata.json`, so the data is pinned exactly as the code is.

## The authors' hyperparameters are required, not optional

`--lr` and `--num_epoch` are declared as `type=float` / `type=int` with **no
`default=`**. Run without them and they arrive as `None`, and the failure lands
inside the optimiser rather than at the argument. There is nothing to fall
back on.

`train_graflag.py` therefore carries the table upstream's README publishes, one
command per dataset:

| dataset | `--lr` | `--num_epoch` | `--threshold` |
|---|---|---|---|
| `gad_cora` | 5e-3 | 200 | 7 |
| `gad_citeseer` | 3e-3 | 200 | 6 |
| `gad_pubmed` | 4e-3 | 100 | 8 |
| `gad_bitcoinotc` | 4e-4 | 100 | 8 |
| `gad_bitotc` | 5e-4 | 100 | 7 |
| `gad_bitalpha` | 5e-3 | 100 | 8 |

`--params LR=...` overrides, and `training_info.hyperparameters_from` records
which of the two the run used. A dataset outside the table raises rather than
guessing.

## The patch

`patches/export-scores.patch` adds four lines that write `ano_score_final` —
the per-node mean score over `--auc_test_rounds` rounds — and its labels, when
`GRAFLAG_SCORE_OUT` is set. run.py otherwise computes the vector, measures an
AUC on it, prints the AUC and keeps the vector only for a printed degree-split
breakdown.

Guarded by the environment variable so the patched tree behaves exactly as
upstream when it is unset: running the repository's own README commands gives
byte-identical output to the unpatched clone.

The labels are written beside the scores rather than read from the dataset
separately, because `ano_label` comes from upstream's own `load_mat` and
pairing against a separately-read label column would depend on both agreeing
about node order.

## networkx is pinned below 3.0

`adj_to_dgl_graph()` calls `nx.from_scipy_sparse_matrix()`, which networkx 3.0
removed in favour of `from_scipy_sparse_array()`. The Dockerfile pins
`networkx<3` rather than rewriting the call: restoring the API the authors
wrote against is the smaller change, and it keeps the clone identical to the
pinned tree apart from the score export.

## Scored split

`all_nodes`. AD-GCL trains on the graph it scores, unsupervised and
transductive — the same protocol as the `bond_*` methods and ADA-GAD, and
comparable with them. It is not an estimate of performance on unseen nodes.
