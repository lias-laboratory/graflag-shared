# strgnn

StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in
Dynamic Graphs (CIKM 2021). Scores **edges** by classifying the enclosing
subgraph extracted around each one over a window of snapshots.

## Upstream's code runs here

The Dockerfile clones https://github.com/KnowledgeDiscovery/StrGNN into
`/app/src` and `train_graflag.py` imports `dyn_links2subgraphs` from
`util_functions` and `Classifier`, `loop_dataset`, `cmd_args` from `main`.
`graflag_runner.upstream()` puts `src/detection` and `src/pytorch_DGCNN` on
`sys.path`, anchored on this file's directory, and raises if the checkout is
missing rather than failing later with an unexplained `ImportError`.

`cmd_args` is the namespace upstream builds at import time; this method's
`Config` is written onto it instead of a second parser being declared.

## Parameters

Declared as a `Config` dataclass in `train_graflag.py`; the `.env` supplies the
defaults and `--params NAME=value` overrides one. `results.json` records the
effective values under `metadata.method_parameters`.

`_GPU` is the GPU index and **-1 means CPU**, which is what `graflag run
--no-gpu` sets. It is read by `graflag_runner.device()`, not by the model.

Three values cannot be expressed in a `.env` the way upstream wants to read
them, so `Config.__post_init__` translates them: `_MAX_NODES_PER_HOP` at or
below 0 becomes `None` (unlimited), and `_USE_EMBEDDING` and `_DROPOUT` become
booleans.

`_TEST_RATIO` is only consulted when the dataset carries no `split.npz`; with
one, the split's own train/test assignment is used.

## Datasets

`SUPPORTED_DATASETS=*_snapshot`. `graflag_runner.snapshot_files()` finds the
snapshot series (`acc_graph.npy`) and the split (`split.npz`); `sta_graph.npy`,
the static graph, is deliberately not used.

Enclosing-subgraph extraction is the expensive step, so its result is cached in
the experiment directory as `subgraphs_h<hop>.pkl` and reused on a re-run with
the same `_HOP`.

## Scores

`result_type` is `EDGE_STREAM_ANOMALY_SCORES`, over the **test** split only.

The published score is `1 - P(upstream class 1)`. Upstream is a link predictor:
`dyn_links2subgraphs` labels `test_pos` 1 and `test_neg` 0 -- real edge against
sampled non-edge -- so its class 1 is the *normal* one, while `test_neg` is
exactly the set of anomalies `datasets/convert_to_strgnn.py` injected.

This integration used to publish `softmax(logits)[:, 1]` as the anomaly score
and upstream's label as the ground truth, which inverted both sides at once.
`auc_roc` is invariant under that double flip, so the reported AUC looked
correct while `precision_at_k`, `recall_at_k`, the anomaly counts and
`score_distribution.png` all described the normal class. Both sides now come
from `graflag_runner.split_test_edges()`, which is the one place the convention
is stated, and a guard raises if upstream's own label stops agreeing with the
position rather than letting the inversion return silently.

The per-epoch `avg_precision` and `pr_auc` in `training.csv` come from
upstream's `loop_dataset` and are still measured against upstream's convention,
so they are precision for the *normal* class. `test_auc` in the same file is
unaffected, being flip-invariant. `eval/evaluation.json` is the number to read.
