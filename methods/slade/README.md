# slade

SLADE: Detecting Dynamic Anomalies in Edge Streams without Labels via
Self-Supervised Learning (KDD 2024). Scores **edges** in a timestamped edge
stream, trained without labels.

## Upstream's code runs here

Unlike `addgraph`, this integration is a thin wrapper: the Dockerfile clones
https://github.com/jhsk777/SLADE into `/app/src` and `train_graflag.py` imports
`SLADE_TGN`, `get_neighbor_finder`, `Data` and `eval_anomaly_node_detection`
from it. `graflag_runner.upstream("src")` puts the checkout on `sys.path`,
anchored on this file's directory, and raises if it is missing rather than
failing later with an unexplained `ImportError`.

The integration supplies the training loop, the data loading and the result
serialization; the model and its scoring are upstream's.

## Parameters

Declared as a `Config` dataclass in `train_graflag.py`; the `.env` supplies the
defaults and `--params NAME=value` overrides one. `results.json` records the
effective values under `metadata.method_parameters`.

`_GPU` is the GPU index and **-1 means CPU**, which is what `graflag run
--no-gpu` sets. It is read by `graflag_runner.device()`, not by the model.

The four ablation switches -- `_ONLY_DRIFT_LOSS_SCORE`,
`_ONLY_RECOVERY_LOSS_SCORE`, `_ONLY_DRIFT_SCORE`, `_ONLY_REC_SCORE` -- are
declared as `int` because a `.env` can only say `0` or `1`, and `Config`'s
`__post_init__` turns them into the booleans the model wants. That is the only
value in the contract that is not used exactly as declared.

`_N_RUNS` repeats the whole train/eval cycle; the reported AUC is the mean over
runs, with the standard deviation alongside it. The published scores come from
the last run.

## Datasets

`SUPPORTED_DATASETS=slade_*`. The loader wants a single `ml_*.csv` with columns
`u, i, ts, label, idx` (it falls back to `source`/`destination`/`timestamp` and
to positional columns when those names are absent). The dataset directory names
itself after the method, so `slade_bitcoinalpha` is reported as
`bitcoinalpha` in the metadata.

`_TRAINING_RATIO` splits on a timestamp quantile: edges at or before it train,
the rest are the per-epoch evaluation set.

## Scores

`result_type` is `EDGE_STREAM_ANOMALY_SCORES`. `compute_anomaly_score` returns
a memory recovery score and a drift score, both higher when the edge looks
*normal*; the published score is `(2 - drift - recovery) / 4`, which negates and
rescales them so higher means more anomalous -- the direction
`graflag_evaluator` assumes. The ablation switches publish `(1 - drift) / 2` or
`(1 - recovery) / 2` instead of the mean of the two.

**Scores cover the test split.** They did not always: `pred_scores` is sized
to `full_data`, and the whole array used to be published, so `auc_roc` in
`evaluation.json` was measured partly on edges the model had been fit on.

The forward pass itself is still over the full stream, and has to be. SLADE is
a streaming method with memory state: an edge's score depends on everything the
model has already seen, so replaying the training prefix is what puts the memory
in the right state by the time the test edges arrive. Scoring only the test
edges in isolation would give different -- and wrong -- numbers. What changed is
the *publish*: `test_mask`, the same boolean mask `load_data` splits on, now
selects which of the computed scores reach `results.json`, along with their
edges, timestamps and labels.

`metadata.summary` records `scored_split: "test"` with `scored_samples`, and
`final_test_auc` is the AUC of exactly what was published.

**There is no validation split, and the per-epoch column says so now.**
`load_data` makes two splits, train and test, on a timestamp quantile; the
per-epoch evaluation calls `eval_anomaly_node_detection(model, test_data, ...)`.
That column in `training.csv` was called `val_auc`, and its curve in
`training_curves.png` was labelled the same -- a test-set number under a name
that promises it is not one. It is `test_auc`, and the printed line is `Test
AUC`. A `training.csv` from an older run has the old column name.

What this method does **not** do is select on it. `best_test_auc` is tracked
and printed, no checkpoint is restored from it, and the published scores come
from whatever weights training ended on (`final_auc = test_aucs[-1]`). So the
name was misleading but the number was not inflated by selection -- unlike
`generaldyg` and `gady`, which do pick their published checkpoint by the test
score.
