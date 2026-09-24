# streamspot

StreamSpot: Graph-Based Anomaly Detection in System Provenance Data (KDD 2016).

Each graph is one system provenance trace. The upstream C++ binary consumes an
edge stream, clusters graphs by their shingle sketches, and prints one anomaly
score per graph per iteration. `train_graflag.py` feeds it the paper's
pre-computed bootstrap clusters, publishes the final iteration's scores as
`GRAPH_ANOMALY_SCORES`, and does no modelling of its own.

## Only the paper's dataset

This integration is wired to the StreamSpot dataset and to nothing else:

- `NUM_GRAPHS = 600`. The scenario a graph belongs to is `graph_id // 100`.
- The attack graphs **are** the ground truth, and they come from the paper, not
  from a label column. `all.tsv` has none: its six fields are source id, source
  type, destination id, destination type, edge type and graph id.

So on any other edge stream the published `ground_truth` would be fabricated
rather than wrong-in-a-detectable-way, and every metric computed from it would
be meaningless. The labels therefore have to be an assumption about the data's
layout -- and the fix was to **check** the assumption rather than assert it.

`scan_edge_stream()` walks `all.tsv` once, collecting the graph ids present
while it counts edges (the pass was already there; this rides along on it), and
raises on a line with fewer than six tab-separated fields or a non-numeric graph
id. `verify_paper_layout()` then requires the id set to be exactly
`{0, ..., 599}`, naming which ids are missing and which are unexpected if not.
A stream that does not have the paper's layout stops there instead of being
scored against the paper's labels.

The labels themselves are now derived rather than written down twice:

```python
ATTACK_SCENARIOS = frozenset(s for s, name in SCENARIOS.items() if "ATTACK" in name)
ATTACK_GRAPH_IDS = frozenset(
    gid for gid in range(NUM_GRAPHS) if gid // 100 in ATTACK_SCENARIOS)
```

`SCENARIOS` already says which scenario is the attack. The previous literal
`set(range(300, 400))` beside it was a second copy of that fact, free to
disagree with the table it was supposed to follow.

Two further checks were already in place and remain:

- `parse_streamspot_output()` raises when the binary prints a number of scores
  other than 600. This used to be a printed warning; a short score line then
  travelled on and died much later, inside sklearn (`inconsistent numbers of
  samples`) or on an `IndexError` in the scenario filter — after the mismatched
  scores had already reached `save_scores()`, which does not check that
  `scores` and `ground_truth` are the same length.
- `Config.__post_init__` rejects an unknown `_DATASET` before the run starts.

Giving this method a dataset of its own would mean reading labels from the
dataset directory instead of from the paper. That is a change to what the
method measures, so it is not done here.

The layout was checked against the downloaded stream rather than taken on
trust -- one pass over `all.tsv` on the manager:

    distinct graph ids: 600
    max graph id: 599
    total edges: 89,770,902

600 graphs numbered 0-599 is what `graph_id // 100` needs to be a scenario
index, so the hundred ids in scenario 3 are the hundred the paper labels as
the attack. That check now runs on every run rather than once, by hand, here.

| Scenario | Graph ids | Content |
|---|---|---|
| 0 | 0-99 | YouTube (benign) |
| 1 | 100-199 | GMail (benign) |
| 2 | 200-299 | VGame (benign) |
| 3 | 300-399 | **Drive-by-download (attack)** |
| 4 | 400-499 | Download (benign) |
| 5 | 500-599 | CNN (benign) |

`_DATASET` selects which scenarios are scored: `all` (all six, 600 graphs),
`ydc` (0, 3, 4, 5 — 400 graphs) or `gfc` (1, 2, 3, 5 — 400 graphs). All three
keep the 100 attack graphs, so the anomaly ratio is 1/6 or 1/4.

## Parameters that do nothing

Three parameters are accepted, recorded in `service_config.json` and in the
result metadata, and never read. Changing one logs a `[WARN]` and produces an
identical result. They are listed in `INERT_PARAMS` in `train_graflag.py`.

| Parameter | Why it has no effect |
|---|---|
| `_TRAINING_RATIO` | The bootstrap clusters are the paper's pre-computed ones (`BOOTSTRAP_CLUSTERS_DATA`), so no train/test split is drawn here. |
| `_GLOBAL_THRESHOLD` | The bootstrap file carries the paper's own global threshold — 0.4823 for `all`, 0.9742 for `ydc`, 1.0288 for `gfc` — and that is the value the binary reads. The `.env` default of 0.6 is not any of them. |
| `_SEED` | Nothing in this integration draws a random number, and the binary takes no seed. `seed_all()` is deliberately not called: calling it would suggest the run's repeatability depends on it. |

The parameters that *are* read are `_CHUNK_LENGTH`, `_NUM_PARALLEL_GRAPHS`,
`_MAX_NUM_EDGES` and `_DATASET`.

`_MAX_NUM_EDGES=-1` means "all edges", and the edge count is passed explicitly
in that case: upstream's docopt default is the string `inf`, which the binary
cannot parse as a `long`.

## Scores

`result_type` is `GRAPH_ANOMALY_SCORES`, one score per graph, from the **final
iteration** of the binary's output. Higher is more anomalous, and
`ground_truth` is 1 for the attack graphs — no inversion is applied or needed.

`metadata.summary.detection_info.auc_roc` is computed here for convenience and
is `null` when only one class is present. It is not the number to quote:
`graflag evaluate` recomputes AUC from the published scores, and that is the
one in `evaluation.json`.

## The binary

The image builds upstream's C++ in a first stage on `ubuntu:16.04` (it does not
compile with a modern g++) and copies the single binary into an `ubuntu:22.04`
runtime. Two things in that stage are guarded:

- The Makefile's `-march=native -mtune=native` is rewritten to `-O3`, bracketed
  by a `grep` before and a negated `grep` after. `sed -i` exits 0 when it
  matches nothing, so without the guard an upstream respelling would silently
  ship a binary tuned for whichever machine built it — which SIGILLs on a swarm
  worker with a different CPU.
- `test -x streamspot` after `make optimized`, and `BINARY.exists()` at run
  time, so a missing binary is one clear message rather than a bare
  `FileNotFoundError` out of `subprocess`.

Nothing imports the clone; only the compiled binary crosses into the runtime
image.

## Datasets

`SUPPORTED_DATASETS=streamspot_*`. The dataset directory is expected to hold
one edge stream, looked up as `all.tsv`, `edges.tsv`, `all.txt` or `edges.txt`,
then as the first `*.tsv` or `*.txt` in sorted order.

`datasets/streamspot_all/` in this repo is a **fetch stub**: it carries only
`metadata.json` and `README.md`, and the edge stream is downloaded on demand.
`graflag run` calls `_ensure_dataset()`, which runs `graflag_data ... fetch
streamspot_all` on the manager so the download lands directly on the NFS share;
`graflag-data fetch streamspot_all` does the same by hand.

Budget for it: `all.tar.gz` is ~84 MB compressed and ~2.2 GB extracted, against
7.0 GB free on `/shared`. Nothing else large can be hydrated alongside it.
