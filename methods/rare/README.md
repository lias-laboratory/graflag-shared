# rare

RARE: Rarity-based Anomaly Detection in Graphs via Order-Embedding Subgraph
Mining (WISE 2026). Scores the **nodes** of a static graph by how rare the
subgraph patterns anchored at them are.

A graph matcher pretrained on synthetic graphs (DSAN, an order-embedding GNN)
estimates how often a pattern occurs in the target graph. Nodes whose sampled
neighbourhoods are outliers in that embedding space seed a beam search that
grows the rarest patterns, and a pattern is kept ("verified") when its
estimated frequency stays below the threshold T_F. A node's score comes from
the rarest pattern verified at it. Nothing is trained on the target graph.

## Upstream's code runs here

The Dockerfile clones https://github.com/gbay7/rare-gad at `SOURCE_REF` into
`/app/src`, and `upstream("src")` puts it on the path. The pipeline
(`RAREPipeline`), the configuration loader, the per-dataset configurations
under `configs/` and the pretrained matcher `ckpt_order_glass/best_model.pt`
all come from the clone. `train_graflag.py` configures and runs that pipeline;
no part of the detection is reimplemented, and nothing in the clone is edited
or patched.

What the script does around it, all from outside upstream's code:

- **Data.** Upstream's loader reads `<cache_dir>/<name>.pt` and downloads the
  file when it is missing. The script links the share's copy in under the
  name the loader expects, so upstream never downloads, and a dataset missing
  from the share stops the run instead of turning into a download.
- **Device.** `graflag_runner.device()` decides, and `_GPU=-1` means CPU.
  Upstream's model-based detector and batching helpers ask `get_device()`,
  which answers cuda whenever a GPU is visible; the script seeds its cache
  with the chosen device so they follow the same decision.
- **Cache.** Upstream caches embeddings and starting nodes under
  `/tmp/rare_savings`, keyed by dataset name and parameters but not by the
  graph. The script points it at a fresh directory, so no run reuses another
  run's detection.
- **Report.** The script sets upstream's `_search_only`, the switch its own
  grid study uses. It skips `anomalies.json`, the diagnostic plots and the
  pattern renderings, which would otherwise be timed as part of detection.
  The same content, every verified pattern with its anchor, nodes, frequency
  and score, is written to `rare_results.json` in the experiment directory.

Three checks surround upstream's code, because each failure would
otherwise pass silently:

- **The matcher is restored in full.** Upstream loads the checkpoint with
  `load_state_dict(strict=False)`, which neither restores nor reports a key
  the model and the checkpoint do not share, so a mismatched checkpoint leaves
  the matcher on its random initialisation and the run still completes. At
  the pinned commit all 45 keys match for the three benchmark configurations.
  The script compares the key sets after upstream's load and raises if they
  differ.
- **The published scores are the ones upstream evaluated** (see Scores).
- **A run that verified nothing is refused.** With no pattern verified every
  node scores 0, the process still exits 0, and the evaluator would report
  AUC 0.5: chance by construction, which reads as a measurement of a method
  that ranked nothing. The script writes `rare_results.json` first, then
  raises. See "What the published number means" for when this happens.

## Environment

RARE was developed and evaluated natively, not in the Dockerfile its
repository ships, which builds a different stack (Python 3.11, torch 2.4.0,
PyG 2.6.1). The image reproduces the native environment instead:

- Python 3.12.4, and torch 2.6.0+cu124 from pip, with the CUDA 12.4 runtime,
  cuDNN 9.1.0, cuBLAS and triton builds that wheel pins;
- PyG 2.7.0 with torch-scatter 2.1.2 and torch-sparse 0.6.18 (CUDA builds) and
  without pyg-lib, torch-cluster or torch-spline-conv, since PyG picks its
  implementation of some operations by which of these are installed;
- numpy 1.26.4, scipy 1.17.0, scikit-learn 1.7.2 (with joblib 1.5.2 and
  threadpoolctl 3.6.0), networkx 3.5, and the rest of what RARE imports at the
  native versions.

It also leaves unset `CUBLAS_WORKSPACE_CONFIG` and `PYTORCH_CUDA_ALLOC_CONF`,
which upstream's Dockerfile sets and the native environment does not. Given
how far rounding moves this method (see below), the environment is part of
its configuration. Run natively and inside the image on the same GPU at seeds
42, 43 and 44, the two gave the same AUROC and AP at each seed. That is
agreement between single runs, not reproducibility: GPU runs of this method
vary from one to the next, as the table below shows.

The base is `python:3.12.4-slim`, so the image sets `NVIDIA_VISIBLE_DEVICES`
and `NVIDIA_DRIVER_CAPABILITIES` itself. The cluster's workers hand a
container the GPU only when these are set: without them the same base image
finds no driver, and the run would fall back to the CPU with only an `[INFO]`
line to say so.

## Parameters

Declared as a `Config` dataclass in `train_graflag.py`; `--params NAME=value`
overrides one. `results.json` records the values passed under
`metadata.method_parameters`, together with the full upstream configuration
the run resolved to (`resolved`).

| Parameter | Default | Meaning |
|---|---|---|
| `_CONFIG` | `auto` | Upstream configuration, `configs/<name>.yaml`. `auto` picks the benchmark configuration for the dataset (table below). |
| `_TASK` | `struct-anomaly` | Which injected anomalies are ground truth: `struct-anomaly`, `context-anomaly` or `all-anomaly`. |
| `_SEED` | `42` | Upstream's `seed`; its pipeline seeds `random`, NumPy and torch from it. |
| `_GPU` | `0` | GPU index; **-1 means CPU**, which is what `graflag run --no-gpu` sets. |

Five more are accepted but deliberately left out of the `.env`, because each
upstream configuration sets its own value and a default there would override
all three at once. Pass one with `--params` to depart from upstream's setting;
each is applied through upstream's `merge_cli_overrides()`, exactly as
`python -m rare --config ... search.max_freq=20` would apply it.

| Parameter | Upstream key |
|---|---|
| `MAX_FREQ` | `search.max_freq` (the paper's T_F) |
| `OUTLIER_MAX_FREQ` | `search.outlier_max_freq` |
| `MAX_STEPS` | `search.max_steps` |
| `N_BEAMS` | `search.n_beams` |
| `N_NEIGHBORHOODS` | `sampling.n_neighborhoods` |

A configuration that enables training is refused: this integration measures
detection with the pretrained matcher. Training is upstream's
`python -m rare --train`.

## Datasets

`SUPPORTED_DATASETS=bond_inj_cora,bond_inj_amazon,bond_inj_flickr`, the three
datasets upstream ships a benchmark configuration for, each run here under
the configuration upstream's README lists as its benchmark run:

| GraFlag dataset | Upstream name | Configuration (`_CONFIG=auto`) |
|---|---|---|
| `bond_inj_cora` | `inj_cora` | `cora_order_glass_fast` |
| `bond_inj_amazon` | `inj_amazon` | `amazon_order_glass_canon` |
| `bond_inj_flickr` | `inj_flickr` | `flickr_order_glass_v2` |

One `graflag run` of each, at the default seed, through all four gates of the
integration checklist:

| Dataset | Nodes | Verified anchors (anomalies) | AUROC | AP | Run time | Peak GPU memory |
|---|---|---|---|---|---|---|
| `bond_inj_cora` | 2,708 | 67 (67) | 0.979 | 0.958 | 2.1 min | 10.5 GB |
| `bond_inj_amazon` | 13,752 | 441 (285) | 0.904 | 0.664 | 18.5 min | 5.8 GB |
| `bond_inj_flickr` | 89,250 | 1,963 (1,843) | 0.911 | 0.777 | 13.2 min | 8.4 GB |

Most of Flickr's time goes before the search: its configuration samples
175,000 neighbourhoods (Cora's samples 10,000), and scoring them against each
other took 220 s of the run.

The `.pt` files on the share come from the same PyGOD archives upstream's
loader downloads (`github.com/pygod-team/data`). The configurations are the
authors' per-dataset settings; nothing here re-tunes them.

The other `bond_*` datasets are not supported. Upstream has no configuration
for them, and the organic ones (`bond_weibo`, `bond_reddit`, `bond_disney`,
`bond_enron`, `bond_books`) carry a single 0/1 label, so the structural bit
RARE is evaluated against is never set there.

## Scores

`result_type` is `NODE_ANOMALY_SCORES`: one score per node, in PyG node order.

The vector is the one upstream's `get_stat_results()` builds and hands to
`roc_auc_score` for its own AUROC: `1 - beam.score` at each verified anchor
(the highest, when several patterns share an anchor) and `0` at every other
node. Higher is more anomalous, the direction `graflag_evaluator` assumes.
`get_stat_results()` returns only the metrics, so the script records the
arguments of that call rather than rebuilding the vector from the patterns,
which would be a second copy of upstream's scoring rule. It then checks that
the recorded call is the one that produced the metrics upstream reported, and
raises if it is not.

Ground truth is the label vector upstream evaluates against in that same
call. `metadata.summary.detection_info` carries upstream's own metrics
(`rare_auc_roc`, `rare_ap`, precision, recall and F1 of the verified set), so
`verify_run.py` can check that the evaluator reproduces them.

## What the published number means

**Every node is scored, and that is the protocol.** RARE is unsupervised and
transductive: the matcher was trained on synthetic graphs and nothing is fitted
on the target graph, so there is no split to hold out. `scored_split` is `all`,
and `verify_run.py` warns about it. That warning is expected here.

**The positives are the structural anomalies only.** With the default
`_TASK=struct-anomaly` the ground truth is bit 1 of PyGOD's `y`, the task
every upstream benchmark configuration sets. The `bond_*` methods on the same
datasets count any non-zero `y`, contextual anomalies included, so their AUC
and RARE's are not computed over the same positives: on `bond_inj_cora` that
is 70 against 138. `--params TASK=all-anomaly` evaluates RARE against the
`bond_*` labels.

**Unverified nodes tie at 0.** Only verified anchors receive a score, and
every other node scores exactly 0, so below the verified set the ranking is a
tie. On Cora every verified anchor also scored the same value (0.9993 or
0.9998, one shape of pattern per run), so the ranking there is
verified-or-not and the AUROC is (1 + recall)/2 whenever nothing false is
verified. On Amazon the 441 verified anchors took 18 distinct values, and on
Flickr the 1,963 took 47. That is upstream's scoring, published as is.

**Whether anything is verified depends on the seed.** The Cora benchmark
configuration through this script on the cluster's RTX A2000, natively, in
the image, and as a GraFlag run on the cluster:

| Environment | Seed | Starting nodes | Verified | AUROC | AP |
|---|---|---|---|---|---|
| native | 42 | 2,144 | 64 | 0.957 | 0.917 |
| image | 42 | 2,143 | 64 | 0.957 | 0.917 |
| cluster (`graflag run`) | 42 | 2,151 | 67 | 0.979 | 0.958 |
| native | 43 | 2,116 | 0 | refused | |
| image | 43 | 2,118 | 0 | refused | |
| native | 44 | 2,130 | 62 | 0.943 | 0.889 |
| image | 44 | 2,128 | 62 | 0.943 | 0.889 |
| native, CPU | 42 | 2,121 | 0 | refused | |

Everything verified was a true anomaly (70 on Cora). Whether the beam search
reaches any pattern below T_F at all is another matter: at seed 43 it reaches
none, natively and in the image. Upstream's own `python -m rare` ends the same
way at seed 43, and at seed 42 gave exactly this script's metrics when both
ran with the variables upstream's Dockerfile sets (AUROC 0.950, 63 verified).

The CPU verifies nothing even at seed 42. Nothing in upstream's code branches
on the device (TF32 is off, autocast is used only in training). The CPU run
differs in floating-point rounding alone, which upstream's hard thresholds
amplify: a reference contains a pattern when its order-embedding violation is
below a fixed threshold, and a pattern is verified when its estimated
frequency is below T_F. Its frequency estimates already differed from the
GPU's at the first search step (0.9591 against 0.9607), and at the last step
its rarest candidate sat at 0.0070, above T_F's 0.0055, where the GPU run's
sat at 0.0007. So `--no-gpu` does not give a CPU measurement of the same
method.

GPU runs are not bit-identical either, since upstream leaves torch's
deterministic algorithms off (its comment gives the cost as a 100x slowdown).
The native and image runs above verified the same number of anchors at each
seed, 61 of 64 and 59 of 62 of them the same nodes; two GPU runs at seed 42
with identical settings (this script and upstream's CLI) had 59 of 63 in
common. The image reproduces the native runs to within that variation, and
the variation is not small: the five seed-42 GPU runs verified 63 to 67
anchors, AUROC 0.950 to 0.979.

Upstream's README gives AUROC 0.986 for Cora at this operating point. At the
pinned commit the closest run here is the cluster's 0.979 at seed 42; others
at seeds 42 and 44 gave 0.943 to 0.957, and seed 43 verified nothing.
