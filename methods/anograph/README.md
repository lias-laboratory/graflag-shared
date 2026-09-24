# anograph

AnoGraph: Sketch-Based Anomaly Detection in Streaming Graphs (KDD 2023).
Depending on `_ALGORITHM`, scores either **time windows** or **edges** — see
"One method, two granularities" below.

## The published scores are upstream's own

The Dockerfile clones https://github.com/Stream-AD/AnoGraph at the commit
`SOURCE_REF` pins and builds its C++; `train_graflag.py` runs the resulting
binary and publishes the scores **the binary wrote**. Upstream emits them to
`../results/<algorithm>_<dataset>[_<tw>_<et>]_score.csv` as one `score label`
pair per line (`utils.cpp:8-15`), and that is the same file upstream's own
`metrics.py` reads to produce the AUCs in the paper.

This was not always true, and the reason it was not is worth keeping:

- `setup_anograph_data` used to write `../data/<name>.csv` and
  `<name>_label.csv`. Upstream's `ReadUtils` opens `../data/<name>/Data.csv`
  and `../data/<name>/Label.csv` (`utils.cpp:26,43,57`) — different paths, so
  the binary never saw an input.
- It did not say so. Every reader answers a NULL from `fopen` with `exit(0)`,
  so a run that read nothing at all exited **successfully** with an empty
  results directory.
- The integration filled the gap with a local density heuristic and published
  that. The premise written down at the time — "the binary reports a single
  AUC and no per-edge score" — was simply wrong; it writes a full score file.

So the fix was to hand upstream its input where it looks for it, and three
checks now stand between a silent failure and a published number:
`run_anograph` deletes the score file before the run so a stale one cannot be
read back as this run's result, treats a nonzero exit as fatal, and treats *no
score file after a zero exit* as fatal too — that last one being exactly what
`exit(0)` looks like from the outside.

## Verified against upstream, score for score

`anograph_iscx` is byte-identical to the ISCX that ships in upstream's `data/`
(`cmp` reports no difference; 1,097,070 edges, 46,456 anomalous either side).
That makes a direct reproduction possible, and it was run: upstream's own
`demo.sh` path — `process_data.py ISCX 60 100`, `./main anograph ISCX 60 100 2
32`, `metrics.py` — against this method's `results.json` from
`graflag run -m anograph -d anograph_iscx`.

| | upstream | GraFlag |
|---|---|---|
| windows scored | 2751 | 2751 |
| AUC | 0.9480 | 0.9480 |
| labels | — | identical |
| scores | — | identical, `max |difference| = 0` |

Not "close": the same 2751 floats in the same order. `graflag evaluate` then
reports `auc_roc` 0.948, the same number the binary computes for itself, which
is the cross-check that the scores reaching `evaluation.json` are still the
ones the method produced.

Worth noting for anyone reproducing it by hand: upstream's `main` writes into
`../results/`, a directory its repository does not contain, and `fopen` failing
there is as quiet as everywhere else in this codebase — the binary runs, prints
its timings, and leaves nothing behind. `mkdir -p ../results` first. The
container has always created it; only a manual run is exposed.

## The sketch is 32 buckets, not 1024

`_NUM_BUCKETS` was `1024` in this `.env`, a number with no upstream provenance:
`demo.sh` passes `2 32` on all four of its datasets, and those are the settings
its published results are measured at.

It is not an off-spec parameter, it is an unrunnable one. `anograph` and
`anograph_k` score a window by greedily peeling the sketch matrix —
`getAnographDensity` (`anograph.cpp:118-150`) runs `2 * buckets` rounds, each
calling `pickMinRow`, `pickMinCol` and `getMatrixDensity` at O(buckets^2), once
per sketch row per graph. The cost is cubic in the bucket count: about 10^9
operations for ISCX at 32, roughly 3.5 * 10^13 at 1024. A run at 1024 was left
going for thirteen minutes on a worker, at a flat 268 MB, without scoring a
single one of its 2751 windows. At 32 the whole method finishes in 4.2 s.

This is the second defect the first one was hiding. While the binary was being
handed its input at a path it does not read, it returned instantly and
successfully every time, so nothing ever executed the scoring path and nobody
could discover that the declared parameters would not finish. Restoring the
input is what exposed the parameter.

## One method, two granularities

`_ALGORITHM` selects one of upstream's four, and they do not all score the same
thing. This decides the `result_type`, so it changes what `graflag evaluate`
computes over:

| `_ALGORITHM` | Subcommand | Scores | `result_type` |
|---|---|---|---|
| `anograph` | `anograph` | time windows | `GRAPH_ANOMALY_SCORES` |
| `anographk` | `anograph_k` | time windows | `GRAPH_ANOMALY_SCORES` |
| `anoedgeg` | `anoedge_g` | edges | `EDGE_STREAM_ANOMALY_SCORES` |
| `anoedgel` | `anoedge_l` | edges | `EDGE_STREAM_ANOMALY_SCORES` |

The `ALGORITHMS` table in `train_graflag.py` holds the subcommand, the
granularity and the positional argument order as upstream's `demo.sh` spells
them, so adding an algorithm is a row rather than a branch.

Every parameter now reaches the published scores, because the binary is the
only thing producing them. (Under the heuristic, `_NUM_ROWS`, `_NUM_BUCKETS`,
`_K` and `_THRESHOLD` reached the binary alone and changed nothing that was
published.)

## The graph algorithms need labels generated per run

`Label_<tw>_<et>.csv` ships with no dataset: `demo.sh` regenerates it before
every run, because a window counts as anomalous only once it holds
`_EDGE_THRESHOLD` anomalous edges — so the same stream has different ground
truth at different settings. `prepare_graph_labels` runs upstream's
`process_data.py` to produce it.

Two things are checked before the binary is allowed to run, both of which
otherwise surface as SIGABRT with no message (upstream's `assert(graphs.size()
== labels.size())` at the end of `loadGraphData`):

- **The stream must be sorted by timestamp and start in time bin 0.**
  `loadGraphData` closes the current graph whenever `t/time_window` changes,
  counting from `cur_time = 0`, while `process_data.py` groups by the distinct
  values of the same quotient. They agree on the window count only under those
  two conditions.
- **The window labels must have both classes.** A stream that is short, finely
  windowed, or cleaner than the paper's yields a single class, and every metric
  in `RESULTS_STANDARD.md` is undefined over it. Failing here beats publishing
  a result nothing can be computed from.

The edge algorithms skip both: they score every edge and never window.

## Parameters

Declared as a `Config` dataclass; the `.env` supplies the defaults and
`--params NAME=value` overrides one. `_ALGORITHM` is validated in
`__post_init__` — without the check a typo reaches the binary and returns as an
unexplained non-zero exit. There is no `_GPU`: the binary is CPU-only.

**`_TIME_WINDOW=60` and `_EDGE_THRESHOLD=100` are ISCX's settings.** Upstream's
`demo.sh` uses `30` and `50` for DARPA, so:

```bash
graflag run -m anograph -d anograph_darpa --params TIME_WINDOW=30 EDGE_THRESHOLD=50
```

At the ISCX defaults DARPA still produces both classes (777 windows, 232
anomalous, against 1,553 and 412 at the paper's settings), so nothing fails —
the numbers are just not the ones the paper reports.

## Datasets

`SUPPORTED_DATASETS=anograph_*`. Two input shapes are accepted:

- **Native AnoGraph format** — `Data.csv` (`src,dst,timestamp`) plus
  `Label.csv`, one label per row. `anograph_darpa` and `anograph_iscx` ship
  this and are already sorted and bin-0 aligned.
- **GraFlag snapshot format** — `acc_*.npy` plus `split.npz`, converted on the
  fly. Edges are read out of the accumulated adjacency snapshots, the split's
  `test_neg` (the injected anomalies — see `split_test_edges`) are labelled 1,
  and any anomaly not already present as an edge is appended.

Either way the input is written to `/app/src/data/graflag_data/`, because the
binary addresses datasets by name relative to its own code directory.

**On snapshot datasets the published stream is the whole graph, not a test
split.** Converting `uci_snapshot` yields 115,078 edges of which 37 are
anomalies; the other 115,041 include every training edge. Read the resulting
AUC accordingly.

## Scores

Higher means more anomalous, the direction `graflag_evaluator` assumes, and
ground truth is 1 for an anomaly — both as upstream writes them. The scores are
**not** normalised here; they are the binary's raw output, so they compare
directly against the paper.

For an edge algorithm, `edges` and `timestamps` accompany the scores in stream
order. For a graph algorithm, `graph_ids` is the window index — there is no
edge to attach a window score to. `metadata.summary.results.auc_published_scores`
is the AUC of exactly what was published, and
`metadata.summary.scores_are` names the file it was read from.
