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

## The one patch: STRGNN on a CPU

`patches/cpu-labels.patch` changes one line of `pytorch_DGCNN/main.py`.
Upstream moves features, labels and the classifier to the GPU only when
`cmd_args.mode == 'gpu'`, except where it builds the labels for the MLP
(line 129): that one calls `.cuda()` unconditionally, so STRGNN could not run
on a CPU at all. With `_GPU=-1` and the GPU hidden from the container
(`graflag run --no-gpu`) the first batch raised `Found no NVIDIA driver`; with
the GPU still visible, as it was before GraFlag hid it, `nll_loss` raised
`Expected all tensors to be on the same device`. The patch adds upstream's own
guard. On a GPU it changes nothing, so GPU runs from before it are unaffected.

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

## Repeated runs: exact on a CPU, not on a GPU

Same configuration, same seed, `uci_snapshot`, three-node GPU cluster, 2026-11-17:

| Runs | Device | Scores | AUC-ROC |
|---|---|---|---|
| 4 (the original and three `run --from-config` replays) | GPU | differ by up to 0.415 | 0.9027, 0.9077, 0.9060, 0.9034 |
| 2 (`exp__strgnn__uci_snapshot__20261117_141244`, `..._151728`) | CPU | **bit-identical**, all 777 | 0.9109, 0.9109 |

So STRGNN's own randomness is fully seeded: on a CPU two runs produce the
same 777 scores, and the spread comes from the GPU side. All three replays ran
on node-2, one of the two nodes whose pinned-memory copies are corrupted
(`validation/platform/README.md`), but STRGNN never pins host memory, so that
fault is not it. The likely cause is GPU kernels whose floating-point
summation order is not fixed, amplified by training; which kernel has not been
isolated. In practice a
STRGNN number from a GPU run is one draw from a band about 0.005 wide (mean
0.905, standard deviation 0.002 over these four), and the CPU is the way to
get an exactly repeatable one. The CPU runs need the patch above.
