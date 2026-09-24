# dynwalk

NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic
Networks (KDD 2018). Edge-level anomaly scores over a dynamic graph.

Read the first two sections before quoting a number from this method.

## This is not NetWalk's code

`.env` declares `SOURCE_CODE=https://github.com/chengw07/NetWalk`, but the
Dockerfile clones nothing and no line of that repository runs here.
`train_graflag.py` is a reimplementation written from the paper's description,
so every result is attributable to this file and not to the authors'
implementation. The URL is a citation, not a dependency.

What actually runs, in order:

| Step | Where | What it does |
|---|---|---|
| 1 | `main()` :183 | Takes the first `_INIT_PERCENT` of the edge stream as the initial graph |
| 2 | `generate_walks` :96 | `_NUMBER_WALKS` uniform random walks of length `_WALK_LENGTH` from every non-isolated node |
| 3 | `create_node_features` :111 | Builds a per-node vector: log degree, neighbour-degree mean and max, then the top walk co-occurrence counts |
| 4 | `Autoencoder` :53 | Reconstructs that vector; the bottleneck of width `_REPRESENTATION_SIZE` is the node embedding |
| 5 | `compute_edge_embedding` :151 | Hadamard product of the two endpoint embeddings |
| 6 | `main()` :251 | K-means with `_N_CLUSTERS` over the initial edges; an edge scores as its distance to the nearest centroid |

Three deviations from the paper are worth naming. There is no walk reservoir
and no incremental update: the walks are drawn once and the stream is scored
in a single pass. The paper's clique embedding is replaced by the hand-built
structural descriptor in step 3. And the encoder is trained on node features
only, so nothing in the objective is edge-aware.

## What it publishes covers the half it was fitted on

Steps 1 to 6 all use the first `_INIT_PERCENT` (default 0.5) of the stream,
but `save_scores` publishes a score for **every** edge, including that half.
GraFlag's results standard asks for scores from the test split; this method
does not draw one.

Measured on the two smallest supported datasets:

| Dataset | Edges | Fitted-on half | Held-out half |
|---|---|---|---|
| `email_snapshot` | 37,674 | 18,837 edges, **0 anomalous** | 18,837 edges, 10 anomalous |
| `uci_snapshot` | 115,078 | 57,539 edges, **0 anomalous** | 57,539 edges, 37 anomalous |

Every anomaly is in the held-out half, so the fitted-on half contributes
nothing but negatives -- negatives the K-means centroids were placed on, which
therefore sit close to a centroid by construction. They inflate the AUC
without the model having generalised to anything. A previous run reported
0.8526 on `uci_snapshot`; that number is over both halves and is not
comparable with a method that scores only held-out edges.

Scoring only `data_df.iloc[init_size:]` would fix it, and would discard no
anomalies at all. That changes what the method reports, so it is recorded here
rather than applied.

A second consequence of the single fit: a node that first appears in the
second half has degree 0 in the initial graph and no walks, so its feature row
is all zeros. Checked, not assumed -- the encoder maps every zero row to the
same vector (its bias), so:

- every **cold-cold** edge receives one identical score. That is 118 of 37,674
  edges on `email_snapshot` and 53 of 115,078 on `uci_snapshot`, none of them
  anomalous, so the tie costs nothing on these two datasets.
- a **cold-warm** edge gets an embedding that varies only with its warm
  endpoint, so whatever made the cold endpoint unusual is not in the score.
  That covers 6 of the 10 anomalies on `email_snapshot` and 4 of the 37 on
  `uci_snapshot`.

## This method runs on the CPU

The Dockerfile installs the CPU-only wheel
(`torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu`), so
`torch.cuda.is_available()` is false inside the image no matter what the host
offers. `_GPU` is therefore `-1`, and `graflag_runner.device()` returns
`cpu` directly instead of asking for a GPU and falling back.

`.env` does not control the reservation. `graflag run` requests a GPU by
default (`core.py:138`, `gpu: bool = True`), and `docker_ops.py:443-455` turns
that into a `DiscreteResourceSpec` for one `NVIDIA-GPU`. On this cluster all
five nodes advertise the same single card, so a default run of this method
holds a GPU it cannot use and blocks a method that could:

```bash
graflag run -m dynwalk -d uci_snapshot --no-gpu
```

To make it a real GPU method, drop the `--index-url` from the torch install,
move to a CUDA base image, and set `_GPU=0`. Nothing in `train_graflag.py`
needs to change: `device()` already returns `cuda:0` when one is visible.

## Parameters

| Key | Default | Effect |
|---|---|---|
| `_REPRESENTATION_SIZE` | 32 | Autoencoder bottleneck, i.e. the node embedding width |
| `_WALK_LENGTH` | 5 | Nodes per random walk |
| `_NUMBER_WALKS` | 20 | Walks started from each node |
| `_INIT_PERCENT` | 0.5 | Fraction of the stream used to fit walks, autoencoder and K-means |
| `_LEARNING_RATE` | 0.001 | Adam step size |
| `_EPOCHS` | 50 | Autoencoder epochs; loss is recorded to `training.csv` every 10 |
| `_HIDDEN_SIZE` | 64 | Width of the encoder's hidden layer |
| `_N_CLUSTERS` | 5 | K-means clusters over the initial edge embeddings |
| `_SEED` | 42 | Passed to `seed_all()` and to K-means |
| `_GPU` | -1 | CPU. See above -- the image has no CUDA |

Every key is read. `Config` in `train_graflag.py` is the single declaration of
what this method accepts; `params(Config)` coerces each value to the
annotated type and drops a stale key instead of failing.

## Datasets

`SUPPORTED_DATASETS=*_snapshot`, which is documentation only -- GraFlag never
enforces it, and any dataset `graflag_runner.load_dataset()` can read will be
attempted. Present in this repository:

| Dataset | Size | Edges | Anomalies |
|---|---|---|---|
| `email_snapshot` | 3.9 M | 37,674 | 10 |
| `uci_snapshot` | 152 M | 115,078 | 37 |
| `btc_alpha_snapshot` | 570 M | -- | -- |
| `btc_otc_snapshot` | 1.3 G | -- | -- |

`email_snapshot` is the one to smoke-test with. Nothing here is CUDA-bound, so
runtime scales with the edge count.

## Scores

`result_type` is `EDGE_STREAM_ANOMALY_SCORES`. A score is the distance from
the edge's Hadamard embedding to the nearest K-means centroid, min-max
normalised across the whole stream, so it lies in [0, 1] and **higher means
more anomalous**. No inversion is applied anywhere; the evaluator reads the
scores as written.

No threshold is reported. A 95th-percentile cut over the training distances
used to be computed and then discarded unused; because the published scores
are min-max normalised afterwards, a raw-distance cut is not on their scale,
so `threshold` is null rather than misleading.

`summary.results.auc` in `results.json` is this method's own AUC over every
edge it published, and it is null -- not 0.0 -- when the labels make it
unmeasurable, because 0.0 is a real AUC and reporting it hid the difference.
`graflag evaluate` recomputes from `scores` and `ground_truth`, so it sees the
same pool; read it with the caveat in the second section above.
