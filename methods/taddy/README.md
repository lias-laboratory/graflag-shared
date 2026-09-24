# taddy

TADDY: Anomaly Detection in Dynamic Graphs via Transformer (TKDE 2021). Scores
**edges** per snapshot of a dynamic graph, using a transformer over
spatial-temporal node encodings.

## Upstream's code runs here

The Dockerfile clones https://github.com/yuetan031/TADDY_pytorch into
`/app/src`, and `upstream("src")` puts it on the path -- raising if the checkout
is missing instead of failing later with an unexplained `ImportError`.
`DynamicDatasetLoader`, `DynADModel`, `MyConfig`, `Settings` and
`anomaly_generation` all come from there.

`DynADModelWithResults` subclasses upstream's `DynADModel` and overrides
`train_model`, so the training loop in this file is the one that runs;
upstream's is not called. The `Config` below is also the object upstream is
handed as its `args`, which it keeps a reference to and reads `print_freq` from.
(Upstream's own `train_model` reads `print_feq` -- its spelling -- but that
method is the one being overridden, so the typo is never reached.)

## Parameters

Declared as a `Config` dataclass in `train_graflag.py`; the `.env` supplies the
defaults and `--params NAME=value` overrides one. `results.json` records the
effective values under `metadata.method_parameters`. This replaced an argparse
parser that ran at *import* time, so importing this module no longer depends on
`argv`.

`_GPU` is the GPU index and **-1 means CPU**, which is what `graflag run
--no-gpu` sets. It is read by `graflag_runner.device()`, not by the model.

`_SEED` seeds the model and the training loop. It does **not** affect which
edges become anomalies: `anomaly_generation` is called with a fixed `seed=1`, so
the injected set is the same for every run. That is also why the preprocessing
cache key is `<dataset>_<train_per>_<anomaly_per>.pkl` with no seed in it -- a
seed in the key would suggest a dependence that is not there.

## Datasets

`SUPPORTED_DATASETS=uci,btc_alpha,btc_otc,digg`. Everything that varies per
dataset -- the names it answers to, the raw file, the edges-per-snapshot size
and which of the two readers parses it -- is one row of the `DATASETS` table:

| dataset | raw file | edges/snapshot | format |
|---|---|---|---|
| `uci` | `uci` | 1000 | whitespace edge list |
| `digg` | `digg` | 6000 | whitespace edge list |
| `btc_alpha` | `soc-sign-bitcoinalpha.csv` | 1000 | rating CSV, timestamp-ordered |
| `btc_otc` | `soc-sign-bitcoinotc.csv` | 2000 | rating CSV, timestamp-ordered |

Those four were previously spelled out three times -- an alias chain, a filename
chain and a `snap_size` dict -- which had to be kept in step by hand. Adding a
dataset is now one row.

The mounted directory is matched on its name, with a `taddy_` prefix stripped
first, so `taddy_btc_alpha` and a bare `soc-sign-bitcoinalpha` both resolve.
The raw file is symlinked to `data/raw/`, not copied.

**This method injects its own anomalies.** Unlike the `*_snapshot` datasets,
which carry an injected `test_neg`, TADDY takes a clean edge list and calls
upstream's `anomaly_generation` to plant `_ANOMALY_PER` of the test edges as
fake cross-cluster edges, splitting train/test at `_TRAIN_PER`. The result is
cached in the container at
`data/percent/<dataset>_<train_per>_<anomaly_per>.pkl` and reused when the same
two ratios come round again.

## Scores

`result_type` is `TEMPORAL_EDGE_ANOMALY_SCORES`: one list of scores per test
snapshot, with `timestamps` carrying the snapshot ids.

The published score is `sigmoid(model(edge))`. The model is trained with
`binary_cross_entropy_with_logits` against labels that are **1 for an injected
anomaly** (and 1 for the sampled negatives added each epoch), so a higher score
already means more anomalous -- the direction `graflag_evaluator` assumes. No
inversion is applied, and none is needed. Ground truth is upstream's own
per-edge label, 1 for an injected edge.

Scores cover the **test** snapshots only. The training snapshots are all-normal
by construction, so publishing them would put a single class in front of the
evaluator.

`validation.csv` holds a per-`_PRINT_FREQ` AUC, and it is computed on those same
test snapshots -- it tracks progress, it is not a held-out number. `training.csv`
holds per-epoch loss and wall time.
