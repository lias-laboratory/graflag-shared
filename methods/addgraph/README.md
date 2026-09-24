# addgraph

AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN
(IJCAI 2019). Scores **edges** in a sequence of graph snapshots.

## This is a reimplementation, not upstream's code

`SOURCE_CODE` in the `.env` points at https://github.com/Ljiajie/Addgraph, and
that repository is the reference this integration was written from -- but none
of its code runs. `train_graflag.py` defines `SpGraphAttentionLayer`, `SpGAT`,
`HCA`, `GRUCell` and `ScoreNetwork` itself, and imports nothing from a clone.

The Dockerfile used to `git clone` that repository anyway, and the script
opened with `sys.path.insert(0, '/app/src/UCI_D_Addgraph/framwork')` -- a path
no import ever resolved against. Both are gone. What was downloaded was never
read, so removing it changes nothing about what runs, and leaving it in place
implied the results came from upstream's implementation when they do not.

Read any number this method produces as "our reading of AddGraph", not as a
reproduction of the paper's.

## Parameters

Declared as a `Config` dataclass in `train_graflag.py`; the `.env` supplies the
defaults and `--params NAME=value` overrides one. `results.json` records the
effective values under `metadata.method_parameters`.

`_GPU` is the GPU index and **-1 means CPU**, which is what `graflag run
--no-gpu` sets. It is read by `graflag_runner.device()`, not by the model.

`_TRAINING_RATIO` is only consulted when the dataset's `split.npz` carries no
`train_pos_id`/`test_pos_id` arrays; with them, the split's own snapshot
assignment is used and the ratio has no effect. Every `*_snapshot` dataset in
this repository carries them.

`_ANOMALY_RATE` used to be declared here. It was parsed, copied into the config
and never read -- anomalies are injected when the dataset is built, so nothing
the method does at run time could depend on it. A sweep over it produced
identical runs. It was removed rather than left to look meaningful.

## Datasets

`SUPPORTED_DATASETS=*_snapshot`. The loader wants a snapshot series
(`acc_graph.npy`) plus a train/test split (`split.npz`); `sta_graph.npy`, the
static graph, is deliberately not used. Verified end to end on `uci_snapshot`
(AUC 0.7514).

## Scores

`result_type` is `EDGE_STREAM_ANOMALY_SCORES`. The score network returns a
plausibility in [0, 1], so the published score is `1 - plausibility`: higher
means more anomalous, which is the direction `graflag_evaluator` assumes.
Ground truth is the split's own labelling -- `test_pos` are normal, `test_neg`
are the injected anomalies.

Only the test edges are scored (`scored_split` is `test`): the published vector
is the final model's score for every edge of the split's test snapshots.
