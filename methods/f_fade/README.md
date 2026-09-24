# F-FADE

Frequency Factorization for Anomaly Detection in Edge Streams (WSDM 2021).

Upstream: https://github.com/snap-stanford/F-FADE

## What runs

Upstream's `main.py`, unmodified, at the commit `SOURCE_REF` pins. There is no
patch: `main.py` already writes its score vector to `score.txt` before
computing its own AUC, so the scores GraFlag publishes are the ones the method
measured itself on.

The method is **unsupervised**. It models the time-evolving distribution of
interaction frequencies between node pairs and scores each edge by the
likelihood of its observed frequency; no label is read during training.

## The dataset is rendered, not redirected

F-FADE's `Dataset` parses whitespace-separated `timestamp source destination
label` from a single file. GraFlag stores `source,destination,timestamp` in
`Data.csv` with labels alongside in `Label.csv` — same information, different
column order and a different split across files.

`render_dataset()` writes the four columns in F-FADE's order. Reordering
columns in this script is a smaller and more honest change than patching the
authors' loader to read ours. It streams line by line rather than loading:
`anograph_darpa` is 4.5M records and three parallel arrays of it is memory the
container has no reason to spend.

It checks that the number of records written equals both input line counts.
`zip()` stops at the shorter file, so a `Data.csv` and `Label.csv` that
disagree would otherwise silently truncate the stream and publish scores for a
prefix while calling it the whole.

## The published scores cover a suffix of the stream

This is the thing to know before comparing F-FADE's number with anything else.

Nothing before `--t_setup` edges is scored at all — that prefix is used to set
the model up. Upstream aligns with `dataset.label[-len(F_FADE):]`, and this
integration publishes the matching suffix of the labels rather than the full
column. Pairing the whole label column with a shorter score vector would
misalign every score with a different edge's label and still produce a
plausible AUC.

`metadata.summary.dataset_info` records `scored_split: "stream_suffix"` with
`scored_samples`, `skipped_setup_edges` and `total_edges`, so the property
travels with the result.

## NaN scores are published as computed

`main.py` writes `score.txt` first and *then* replaces NaN with 0 for its own
AUC. This integration publishes the file as written, because zero-filling here
would publish a vector the method did not produce. The evaluator excludes
non-finite scores and reports the count under `filtering`, and
`training_info.nan_scores` records how many there were.

The consequence is worth stating: where NaNs occur, upstream's reported AUC and
GraFlag's differ, because upstream scores those edges 0 and GraFlag excludes
them. Neither is wrong; they are different questions.

## Dependency substitution

Upstream pins `torch==1.4.0`, `numpy==1.19.1`, `scipy==1.1.0` and
`scikit_learn==0.22.2`.

**numpy is pinned below 1.24**, not left current: `model.py:246` calls
`np.int`, which numpy 1.24 removed after deprecating it in 1.20. The first run
of this integration died there with `module 'numpy' has no attribute 'int'`
after building and rendering the whole 1.1M-edge stream. Pinning restores the
era the code was written against, which is a smaller change than rewriting the
authors' array construction.

**torch is not pinned**: 1.4.0 has no wheel for a current Python, so the image
installs a current CPU build. The operations F-FADE uses are stable across that
gap, but this is a substitution rather than the authors' declared environment,
and a number from it should be read with that in mind.

## CPU only

`_GPU=-1`. F-FADE trains a small embedding online rather than a GPU model and
its own `run.sh` sizes the work for a CPU, so the image carries the CPU torch
wheel and declares no device.

## Verification

| Dataset | Params | Gate 2 | Gate 3 (`auc_roc`) | Gate 4 |
|---|---|---|---|---|
| `anograph_darpa` | upstream defaults | completed, 173 s, 2,389 MB | **0.9186** | 0 failed, 1 warned, 3 passed |
| `anograph_iscx` | `EPOCHS=1 ONLINE_TRAIN_STEPS=2` | completed, 25 s, 1,146 MB | 0.5023 | 0 failed, 2 warned, 2 passed |

On DARPA — F-FADE's own paper dataset — `main.py` prints `AUC:
0.9186416535125347` and the evaluator computes **0.9186** independently from
the published vector. That agreement is the check that matters: the scores in
`results.json` are the ones the method measured itself on.

4,214,099 edges scored of 4,554,344; the first **340,245** went to `--t_setup`
and carry no score, and the published ground truth is the matching suffix.

The ISCX row is a deliberately reduced smoke run — one epoch against the
default five, on a dataset F-FADE does not publish on. It is recorded to show
the pipeline end to end, not as the method's performance, and 0.5023 should
not be read as one.
