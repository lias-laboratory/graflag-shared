# GeneralDyG

A Generalizable Anomaly Detection Method in Dynamic Graphs (AAAI 2025).

Upstream: https://github.com/YXNTU/GeneralDyG

## How this integration works

GeneralDyG is the one method that does not declare its own `Config`. It
adopts upstream's argparse namespace (`from option import args`), so the
accepted names and their defaults are upstream's, and the `.env` only lists
the subset GraFlag overrides. `apply_params(config, ignore={"gpu"})` writes
those onto the namespace, coercing each value to the type of the default
already sitting there.

It used to go through `--pass-env-args` instead, which is unsafe for exactly
this method: upstream declares `--gpus` and no `--gpu`, so argparse's
abbreviation matching turned `_GPU=0` into `gpus=0` and the device index was
never read. `_GPU` is on the `ignore` list because `device()` reads it
directly.

`results.json` records the whole effective configuration under
`metadata.method_parameters`, with `metadata.injected` naming the subset that
came from the `.env` or `--params`.

## The published scores cover the test split

They did not always. A local `dataset_all.py` — a copy of upstream's dataset
class with the train/test split taken out — used to run a final pass over the
**entire** edge stream, and those were the scores `results.json` carried. Two
things were wrong with that at once:

- Most of what it scored was **training** data. `RESULTS_STANDARD.md` requires
  the published scores to come from the test split, and an AUC computed partly
  over edges the model fitted is not comparable with one that is not.
- It drew a **third** independent `np.random.uniform` feature matrix, so the
  scored inputs matched neither the training features nor the test features.

Upstream has no `DygDatasetAll`; it was ours, so removing it was an
integration fix rather than a change to what GeneralDyG computes. Scoring is
now `eval_epoch(loader_test, ...)` — the same loader, the same features and
the same code path the reported AUC already came from — using the checkpoint
the best-AUC epoch selected. The edge list and timestamps are sliced to the
tail of the CSV matching the test split, and the run fails rather than
publishes if the two do not line up.

`metadata.summary` records `scored_split: "test"` with `scored_samples`, and
the AUC of what was published is `test_auc`.

**There is no validation split, and the checkpoint is chosen on the test set.**
Upstream has no third split: `train.py:181-192` evaluates `loader_test` every
epoch, prints it as `val loss` / `val auc`, keeps the running maximum in
`max_test_auc` and reports it as `best auc`. This integration follows the
protocol but not the naming -- the per-epoch columns in `training.csv` are
`test_loss` and `test_auc`, and the summary key is `best_test_auc`, because a
column called `val_auc` in a file GraFlag publishes is the test AUC under
another name, and the legend of `training_curves.png` would repeat it.

The consequence of the protocol is worth stating plainly: the epoch whose
weights get published is picked using the same edges the published score is
then measured on, which biases `test_auc` upward by an unknown amount. It is
GeneralDyG's own protocol and changing it would change what the method does,
so it stands -- but a number from this method is not a clean held-out
measurement, and comparing it against a method that does hold out a validation
split is not comparing like with like. `slade`, by contrast, reports its best
epoch and publishes its last, so it carries no selection bias.

**The selection used to be inert, which is how it came to be checked.** The
snapshot was `model.state_dict().copy()`, and `state_dict()` returns the live
parameter tensors -- `copy()` duplicates the dict, not the tensors the
optimizer then updates in place. So the "best checkpoint" tracked training and
`load_state_dict` restored the current weights onto themselves. The first run
after the test-split fix made it visible: best AUC at epoch 1 (0.7806), last
epoch 0.6829, and the published score was 0.6829 while the log said the best
checkpoint had been loaded. It could not have been seen before, because the
final pass scored the whole stream and its AUC was not comparable with any
per-epoch number. The snapshot now clones each tensor.

## Upstream's features are random, and still are

Upstream draws node and edge features from `np.random.uniform` in its dataset
class, and `train.py:106-107` builds the train and test datasets back to back,
so each draws its own matrix independently. The model is therefore evaluated
on features it did not train on — by upstream's design, at the commit
`SOURCE_REF` pins.

That is a scientific-validity property of the method, not of this integration,
and changing it would change what GeneralDyG computes, so it is reported here
rather than patched. Treat the absolute numbers accordingly.

`set_seed()` makes a run repeatable (`torch.use_deterministic_algorithms`,
cuDNN determinism, a fixed `CUBLAS_WORKSPACE_CONFIG`), so two runs with the
same `_SEED` draw the same features. That makes the result stable; it does not
make it meaningful.

## Memory: the stream is not resident twice

`btc_alpha.pkl` is 305 MB on disk, and every dataset object built from it is
a *dense* padding of the whole edge stream in float64 -- several times the
file. There used to be three of them: `DygDataset` for train, one for test,
and `DygDatasetAll` for the whole-stream pass.

Holding all three at once is what killed the first run here: exit code 137,
SIGKILL from the OOM killer, on a 15 GB host with a `tmpfs` `/shared` already
taking 3 GB. The run had finished training and printed "Generating
predictions for entire dataset..." -- `resources.csv` recorded 6.3 GB in the
last sample before the kill, with nothing left to do but one forward pass.

The training dataset and its loader are released before the scoring pass; the
test loader is the one being scored, so it stays. Nothing after that point
reads them, and `dataset` at the call site is the *module*, not one of the
objects freed. This changes no number the method produces.

The re-run, still with `DygDatasetAll`, completed on the same host with the
same 3 GB on the `tmpfs` and a peak of **5,340 MB** -- below the 6.3 GB the
killed attempt had reached one forward pass from the end. Dropping the third
dataset removes that pass entirely, so the ceiling is lower again. The
`auc_roc` 0.6702 recorded at `_N_EPOCHS=2` came from the whole-stream scores
and is **not comparable** with what this method now publishes; it is kept here
as the record of an earlier run, not as a baseline.

## The training loader does not pin memory

Upstream's `train.py` builds its training loader with `pin_memory=True`, and
`train_graflag.py` followed it. On two of the three nodes that
corrupts the batches: a host-to-GPU copy from pinned memory there delivers
other data than was pinned (`validation/platform/README.md`, reproduced with
no method code). The first run that reached training
(`exp__generaldyg__generaldyg_btc_alpha__20261117_151358`, on node-1) got
896,452 non-finite values in the second batch's `eadjs`, had every weight NaN
after the first optimizer step, and died in the first evaluation with
`ValueError: Input contains NaN`. The same code without pinning trains with
every value finite through a whole epoch, checked batch by batch. Pinning only
speeds the transfer up, so the loader no
longer pins; nothing the model computes changes.

A corruption that left the values finite would not have crashed: it would have
trained on garbage and published a number. That one did not is luck.

## Datasets

`SUPPORTED_DATASETS=generaldyg_*`. The dataset directory is named for the
method (`generaldyg_btc_alpha`), and `train_graflag.py` maps it back to the
bare name upstream's loaders expect.

### The pickle, built once per CSV and seed

Upstream trains from `<data_set>.pkl`, which its `src/generate_datasets.py`
builds from `<data_set>_0.5_0.<neg>.csv`. On `btc_alpha` that takes about
2 h 30 min, on one core, and it happens on the first run only:
`train_graflag.py` keeps the result in
`<dataset dir>/.generaldyg/<data_set>-neg<neg>-seed<seed>/<data_set>.pkl` and
reuses it. The path is in `results.json` as `summary.dataset_info.pickle`.

Three things went into that:

- **Where it lands.** Upstream's `DygDataset` reads `{dir_data}/<data_set>.pkl`,
  but the generator writes `<data_set>.pkl` into the current directory. The
  first run on a three-node GPU cluster
  (`exp__generaldyg__generaldyg_btc_alpha__20261117_123604`) ran the generator
  from `/app`, spent 2 h 26 min building a pickle inside the container, and
  then failed with `FileNotFoundError` on the dataset directory's copy, which
  had never been written -- and the next run would have built it again. The
  generator now runs in a scratch directory beside the cache, and the pickle
  is `os.replace`d into place, so a build that is killed, or two at once,
  cannot leave a truncated pickle a later run would take as finished. An
  exit 0 that writes no pickle raises.
- **What it depends on.** The generator reads `data_set`, `dir_data` and
  `neg` and no other option, so those, with the seed, name the cache. Named
  after `data_set` alone, as upstream does, a run with another `NEG` would
  train on the first run's pickle.
- **The seed.** The generator draws from the stdlib `random` (node masking,
  edge weights) and never seeds it, so two builds from the same CSV differed.
  `train_graflag.py` runs upstream's file unchanged through `runpy`, with
  `random.seed(SEED)` and `PYTHONHASHSEED=SEED` set first.

That script does not parse at the pinned commit. Upstream's last commit,
`86a014a` ("dense tensor"), made `sparse_mx_to_torch_sparse_tensor` return a
dense tensor and in the same edit dedented its `def` to column 0, which ends
`BatchGraphSample` early: importing the file raises `IndentationError`. The
devcluster never noticed, because a `btc_alpha.pkl` built before that commit
was already on its share; the first run on a fresh share (2026-11-17) was the
first to reach it. `patches/generate-datasets-indent.patch` restores the
indentation and nothing else, and the Dockerfile checks that the file parses
after applying it.

## GPU only

`graflag run --no-gpu` cannot work for GeneralDyG. Upstream moves tensors to
`'cuda'` unconditionally on every forward pass -- both branches of
`GraphConvolution.forward` (`model/layers.py:50-51`, `64-65`) and
`CombinedModel.forward` (`model/Combine.py:18`) -- whatever device the model
is on. `train_graflag.py` therefore refuses a CPU device at start, before the
pickle build, instead of failing at the first forward pass hours later. A
CPU run would need those lines patched to the model's device, which this
integration does not do.

## Verification on a three-node GPU cluster

`exp__generaldyg__generaldyg_btc_alpha__20261117_180354` (2026-11-17, defaults
above, seed 95540, GPU, on node-3): completed in 25.4 min, 10,459 MB peak memory,
5,901 MB GPU; **test AUC-ROC 0.7787** from the selected checkpoint over the
7,000 test edges (70 anomalous); `graflag verify` 0 failed, 0 warned, 4
passed. The pickle it trained on had been built by the previous attempt and
was reused, as intended.

As on the devcluster, the best test AUC is at epoch 1 (0.7787 here, 0.7806
there; the pickles differ -- that one predates upstream's last commit and was
built unseeded) and early stopping ends the run ten epochs later, at epoch 11.
The selection is on the test split, which GeneralDyG has no alternative to
(see above), so 0.7787 is the best of 11 test-set measurements, not a
held-out one.

The attempts before it, each one step further:

| Run | Failure | Changed since |
|---|---|---|
| `..._20261117_123604` | after 2 h 26 min building the pickle in the container's working directory: `FileNotFoundError` for the dataset directory's copy | the pickle is built in, and read from, `.generaldyg/<data_set>-neg<neg>-seed<seed>/`, seeded, and moved into place atomically |
| `..._20261117_151358` (node-1) | the pickle built (2 h 10 min); then `ValueError: Input contains NaN` at the first evaluation -- pinned-memory batches arrived corrupted | the training loader does not pin memory |
