# AD-GCL

Revisiting Graph Contrastive Learning on Anomaly Detection from a Structural
Imbalance Perspective (AAAI 2025).

Upstream: https://github.com/yimingxu24/AD-GCL

## What runs

Upstream's `AD-GCL/run.py` at the commit `SOURCE_REF` pins. The method is
**unsupervised**: contrastive learning over RWR subgraphs, with no label read
during training.

## The dataset is used as stored, not converted

AD-GCL's `load_mat()` does `sio.loadmat("./Data/<name>.mat")`. GraFlag stores
these six datasets as exactly that `.mat` — `Network`, `Attributes`, `Label` —
so `stage_dataset()` copies the mounted file to where upstream looks and
nothing is converted, parsed or re-encoded. That is why they are registered in
`.mat` form rather than as PyG `.pt`.

The six ship with the repository and are pinned by commit SHA plus `sha256` in
each dataset's `metadata.json`, so the data is pinned exactly as the code is.

## The authors' hyperparameters are required, not optional

`--lr` and `--num_epoch` are declared as `type=float` / `type=int` with **no
`default=`**. Run without them and they arrive as `None`, and the failure lands
inside the optimiser rather than at the argument. There is nothing to fall
back on.

`train_graflag.py` therefore carries the table upstream's README publishes, one
command per dataset:

| dataset | `--lr` | `--num_epoch` | `--threshold` |
|---|---|---|---|
| `gad_cora` | 5e-3 | 200 | 7 |
| `gad_citeseer` | 3e-3 | 200 | 6 |
| `gad_pubmed` | 4e-3 | 100 | 8 |
| `gad_bitcoinotc` | 4e-4 | 100 | 8 |
| `gad_bitotc` | 5e-4 | 100 | 7 |
| `gad_bitalpha` | 5e-3 | 100 | 8 |

`--params LR=...` overrides, and `training_info.hyperparameters_from` records
which of the two the run used. A dataset outside the table raises rather than
guessing.

## The patches

The clone is upstream's tree at `SOURCE_REF` with two changes, both in
`run.py`. `patches/cpu-device.patch` is the device line described under
"It runs on the CPU". `patches/export-scores.patch` adds four lines that write `ano_score_final` —
the per-node mean score over `--auc_test_rounds` rounds — and its labels, when
`GRAFLAG_SCORE_OUT` is set. run.py otherwise computes the vector, measures an
AUC on it, prints the AUC and keeps the vector only for a printed degree-split
breakdown.

Guarded by the environment variable so the patched tree behaves exactly as
upstream when it is unset: running the repository's own README commands gives
byte-identical output to the unpatched clone.

The labels are written beside the scores rather than read from the dataset
separately, because `ano_label` comes from upstream's own `load_mat` and
pairing against a separately-read label column would depend on both agreeing
about node order.

## The stack is upstream's

Upstream's README states **Python 3.8.13, PyTorch 1.12.1, dgl 0.4.3.post1**,
and AD-GCL is written against that DGL: it samples its subgraphs with
`dgl.contrib.sampling.random_walk_with_restart` (`utils.py:84` and `:92`),
which DGL removed in 0.5, and it treats the graph as a host object -- in DGL
0.4, `DGLGraph.to(device)` moves only the graph's feature dictionaries, and
`adj_to_dgl_graph()` attaches none, so the adjacency, the edges and the
degrees stay on the host while the features and the model go to `device`.
The image is built on that stack rather than a current one (see
"Verification" for what a current one did):

| | upstream's README | this image |
|---|---|---|
| Python | 3.8.13 | 3.8.10 (Ubuntu 20.04's) |
| PyTorch | 1.12.1 | 1.12.1, CPU build |
| DGL | 0.4.3.post1 | 0.4.3.post1, CPU build (PyPI's `dgl`) |
| torch_scatter | not stated | 2.1.0, built for torch 1.12.1 (CPU) |
| networkx | not stated | 2.8.8 |

`networkx` is held below 3.0 because `adj_to_dgl_graph()` calls
`nx.from_scipy_sparse_matrix()`, which 3.0 removed; `numpy` below 1.24, which
removed the `np.int`-style aliases code of DGL 0.4's age uses. Both restore the
API the authors wrote against rather than rewriting their calls. DGL 0.4 also
asks on stdin which framework to use unless `DGLBACKEND` is set, so the image
sets `DGLBACKEND=pytorch`.

## It runs on the CPU, and why

Upstream runs on a GPU (`--gpu_id 0` in all six of its README commands). On
this cluster it cannot, with the DGL it was written for. DGL 0.4.3.post1's
newest GPU build, `dgl-cu102`, carries machine code for sm_30, 35, 50, 60 and
70 and **no PTX** (read from the fatbins in its `libdgl.so`), so on a later GPU
-- this cluster's NVIDIA A16 is sm_86 -- there is no kernel to launch. DGL 0.4
does not check the launch, so nothing raises: one copy_src/sum
message-passing step, the operation upstream's `GraphConv` runs, returned all
zeros where torch gives the real sums
(`exp__ad_gcl__gad_cora__20261118_115058`). Run that way, AD-GCL would train
with a graph convolution that outputs nothing and still publish an AUC.

So the image installs the CPU builds of torch, torch_scatter and DGL, and
`patches/cpu-device.patch` changes one line of `run.py`, because upstream
cannot express the CPU: `device = args.gpu_id` is an int that every tensor is
moved to. The patched line is `device = args.gpu_id if args.gpu_id >= 0 else
'cpu'` -- identical to upstream for any GPU index -- and `train_graflag.py`
passes `--gpu_id=-1`. Nothing else in upstream names a device.

`train_graflag.py` checks the library before upstream starts
(`require_dgl()`): the version, the `dgl.contrib` sampler, and that same
message-passing step against the sum written in torch. `_GPU` is `-1` and a
GPU index is refused; run with `--no-gpu`.

Running on the GPU would need DGL 0.4.3.post1 rebuilt from source with PTX
(CUDA 10.2's compiler, which stops at sm_75), which has not been done. A CPU
run is not expected to reproduce the paper's numbers digit for digit: random
streams differ between devices.

## Scored split

`all_nodes`. AD-GCL trains on the graph it scores, unsupervised and
transductive — the same protocol as the `bond_*` methods and ADA-GAD, and
comparable with them. It is not an estimate of performance on unseen nodes.

## Verification

**Verified on a three-node GPU cluster, on the CPU (2026-11-18).** Upstream's
code, on upstream's stack, with the authors' cora configuration (`--lr 5e-3
--num_epoch 200 --threshold 7`, 256 test rounds, seed 1):

| Gate | Result |
|---|---|
| 1. Contract | `tests/test_methods.py` passes |
| 2. Build and run | `exp__ad_gcl__gad_cora__20261118_120340`: completed in 478.7 s, peak RAM 1,357 MB; `require_dgl()` passed; upstream loaded its 94th epoch, chosen on training loss |
| 3. `graflag evaluate` | AUC-ROC **0.9147**, AUC-PR 0.5559, F1@k 0.58 (k = 150) |
| 4. `graflag verify` | 0 failed, 1 warned, 3 passed: the evaluator's AUC equals the 0.9147 run.py printed. The warning is `scored_split: all_nodes`, which is this method's protocol (see "Scored split") |

Run again with the same configuration
(`exp__ad_gcl__gad_cora__20261118_121539`), it chose the same epoch and
published the same 2,708 scores bit for bit (`validation/results/
repeat-ad_gcl-summary.json`). Upstream's README publishes no per-dataset AUC
to compare against, and a CPU run would not match a GPU one digit for digit
anyway.

How it got there. Nine earlier attempts, on a current stack (torch 2.4, DGL
2.4), failed or were interrupted, each one step further than the last. The
files their rows name -- `patches/gpu-compat.patch`, `install_dgl.py`,
`require_dgl_sparse` -- belonged to that image and are gone.

On the development cluster (2026-09-22), five attempts on `gad_cora`, each
one failing and each failure leading to a change in the definition:

| Run | Failure | Changed since |
|---|---|---|
| `exp__ad_gcl__gad_cora__20260922_094340` | `ModuleNotFoundError: No module named 'torch'` while converting the dataset on the manager | datasets are used as stored, and nothing is converted |
| `exp__ad_gcl__gad_cora__20260922_095231` | `ModuleNotFoundError: No module named 'torch_scatter'` inside the container | the image's dependencies |
| `exp__ad_gcl__gad_cora__20260922_095947` | `ImportError: Cannot load DGL C++ sparse library` | torch 2.4.0 and DGL 2.4.0 pinned together, checked by an import at build time |
| `exp__ad_gcl__gad_cora__20260922_100322` | `TypeError: can't convert cuda:0 device type tensor to numpy` | `patches/gpu-compat.patch` |
| `exp__ad_gcl__gad_cora__20260922_100645` | interrupted: the development cluster lost its images during the run | -- |

On a three-node GPU cluster (2026-11-17) four more attempts, each one step further:

| Run | Failure | Changed since |
|---|---|---|
| `exp__ad_gcl__gad_cora__20261117_115829` | build: `HTTP error 403` fetching the DGL 2.4.0 wheel from the S3 bucket behind data.dgl.ai (which was down) | DGL installed from the DGL team's conda channel, SHA-256 pinned (`install_dgl.py`) |
| `exp__ad_gcl__gad_cora__20261117_124957` | build: `import dgl.sparse` -- `libcuda.so.1` missing | that build's sparse library links the driver, which only a running GPU container has: checked with `ldd` at build time and imported at run time (`require_dgl_sparse`) |
| `exp__ad_gcl__gad_cora__20261117_130736` | `run.py:78`: `Expected all tensors to be on the same device ... cuda:0 and cpu` | `patches/gpu-compat.patch`, second hunk: the identity is built on the adjacency's device |
| `exp__ad_gcl__gad_cora__20261117_135032` | `aug.py:47`: `can't convert cuda:0 device type tensor to numpy` | -- (see below) |

The last two are one cause: upstream treats the DGL graph as a host object --
it converts the graph's edges and degrees with `.numpy()`, joins them with
NumPy arrays, and calls `.cpu()` only on the half derived from GPU tensors --
which is what DGL 0.4 did, and DGL 2.4 does not. Behind the device errors
was a hard stop: `utils.generate_rwr_subgraph` samples with
`dgl.contrib.sampling.random_walk_with_restart`, and `dgl.contrib` does not
exist in DGL 2.4.0. Patching symptoms one call at a time was the wrong
approach; upstream's README had stated the versions it needs all along.

On upstream's stack (2026-11-18):

| Run | Failure | Changed since |
|---|---|---|
| `exp__ad_gcl__gad_cora__20261118_114249` | `import dgl`: DGL 0.4 asked on stdin which backend to use, and read EOF | `DGLBACKEND=pytorch` in the image |
| `exp__ad_gcl__gad_cora__20261118_115058` | `require_dgl()`: on the A16, DGL's GPU message passing returned all zeros, with no error | CPU builds and `patches/cpu-device.patch` (see "It runs on the CPU") |
| `exp__ad_gcl__gad_cora__20261118_120340` | completed; verified as above | -- |

The second is the one that matters: nothing raised, and without the check
the run would have trained and published an AUC.

**The other five datasets** (three-node GPU cluster, CPU, 2026-11-18, each
with the authors' configuration for it, 256 test rounds, seed 1; record
`validation/results/ad_gcl-datasets.json`). Every run completed on its first
attempt and passed `graflag verify` with 0 failed, 1 warned (`all_nodes`),
3 passed: the evaluator's AUC equals the one run.py printed.

| Dataset | Configuration | Nodes (anomalous) | Epoch loaded | AUC-ROC | AUC-PR | Time (s) | Peak RAM (MB) |
|---|---|---|---|---|---|---|---|
| `gad_citeseer` | `--lr 3e-3 --num_epoch 200 --threshold 6` | 3,327 (150) | 88 | **0.9407** | 0.5740 | 1,036.5 | 2,390 |
| `gad_pubmed` | `--lr 4e-3 --num_epoch 100 --threshold 8` | 19,717 (600) | 47 | **0.9570** | 0.5709 | 4,439.3 | 26,801 |
| `gad_bitalpha` | `--lr 5e-3 --num_epoch 100 --threshold 8` | 3,219 (300) | 99 | **0.7950** | 0.3540 | 321.5 | 1,307 |
| `gad_bitotc` | `--lr 5e-4 --num_epoch 100 --threshold 7` | 4,863 (300) | 99 | **0.8139** | 0.3092 | 534.3 | 2,205 |
| `gad_bitcoinotc` | `--lr 4e-4 --num_epoch 100 --threshold 8` | 5,881 (300) | 85 | **0.8175** | 0.2871 | 673.6 | 3,138 |

The experiments are `exp__ad_gcl__gad_citeseer__20261118_124303`,
`..._gad_pubmed__20261118_132819`, `..._gad_bitalpha__20261118_130051`,
`..._gad_bitotc__20261118_130645` and `..._gad_bitcoinotc__20261118_131614`.
The epoch loaded is the one upstream's checkpoint selection chose, on training
loss. Pubmed's peak RAM, 26.8 GB, was measured on the CPU, where nothing
moves to a GPU; why it is that large has not been looked into.
