# The bond_* methods

Seventeen wrappers around [PyGOD](https://docs.pygod.org/) detectors. They
share one trainer, one Dockerfile and one contract; only the `.env` differs.
Each method directory has a README with its own parameters. This file is
everything they have in common.

## How a method name becomes a detector

A bond method directory holds `.env` and `Dockerfile` and no Python at all.
`COMMAND=python3 -m graflag_bond.train` runs the shared trainer, which reads
`METHOD_NAME` from the environment -- GraFlag sets it per service
(`docker_ops.py`, `_build_service_env`) -- strips the `bond_` prefix and looks
the remainder up among the classes in `pygod.detector`, case-insensitively
(`graflag_bond/detectors.py`, `from_method_name`). So `bond_dominant` runs
`pygod.detector.DOMINANT` and nothing else about the two differs.

That is also why all 17 Dockerfiles are byte-identical: nothing in the image
depends on which detector will run. They are separate files today, and one
shared image is what the `IMAGE=` key in the plan is for.

`SOURCE_CODE` in each `.env` points at that detector's PyGOD documentation
page, not at a repository to clone. Nothing is cloned; PyGOD is installed from
PyPI.

## Parameters

`.env` declares `_FOO=bar`; the trainer turns it into the keyword argument
`foo=bar` for the detector's constructor. The work is done by
`graflag_runner.method.params()`, called through
`graflag_bond.utils.get_all_parameters()`, and it does three things worth
knowing:

- **Types come from the signature.** `inspect.signature` of the detector's
  `__init__` says what each argument should be, so `_EPOCH=100` arrives as an
  `int` and `_LR=0.004` as a `float`.
- **Unknown keys are dropped, not rejected.** A parameter the detector does
  not accept is discarded with a debug line. PyGOD's API moves between
  releases, so a key that worked before can become inert without anything
  failing -- if a parameter seems to have no effect, that is the first thing
  to check.
- **Two values are resolved from their names.** `_ACT=torch.nn.functional.relu`
  and `_BACKBONE=torch_geometric.nn.GCN` are looked up as attributes
  (`graflag_bond/utils.py`), so they are spelled as import paths rather than
  passed as strings. `_BACKBONE=None` means the detector uses its own default,
  which for the MLP-based detectors is not a GNN at all.

Override any of them per run without editing the `.env`:

```bash
graflag run -m bond_dominant -d bond_gen_100 --params EPOCH=2 HID_DIM=32
```

### `_CONTAMINATION` does nothing to the results

Every bond `.env` declares `_CONTAMINATION=0.1`, and it is passed to the
detector, where PyGOD uses it to derive `threshold_` and the binary
`label_`. GraFlag publishes neither. What it publishes is
`model.decision_score_`, the continuous score, which contamination does not
touch -- so a sweep over `_CONTAMINATION` produces identical `results.json`
files. It is accepted and recorded, and it changes nothing.

`_VERBOSE`, `_SAVE_EMB` and `_COMPILE_MODEL` are passed straight through to
PyGOD and affect logging, an optional embedding dump and `torch.compile`
respectively; none of them changes a score either.

## GPUs

`_GPU` is an index and `-1` means CPU, which is what `graflag run --no-gpu`
sets. `graflag_runner.device()` is not involved here -- the index goes to
PyGOD, which owns the placement.

`bond_scan` is the exception: it declares no `_GPU`, no `_LR` and no `_EPOCH`,
because SCAN is a structural clustering algorithm rather than a trained model.
Run it with `--no-gpu` so Swarm does not reserve a card it cannot use.

## Datasets

`SUPPORTED_DATASETS=bond_*` in every `.env`. That is documentation only --
GraFlag never enforces it -- but it is a real constraint here, because the
trainer loads data with `pygod.utils.load_data()`, which expects PyGOD's own
format. A dataset that is not in that format will fail at load, not at score.

Fourteen are present in this repository, from `bond_disney` (32 KB) and
`bond_gen_100` (48 KB) up to `bond_inj_flickr` (186 MB). `bond_gen_100` is
the one to smoke-test with.

## Scores

`result_type` is `NODE_ANOMALY_SCORES`: one score per node, `node_ids` running
`0..N-1` in the graph's own node order. The score is PyGOD's
`decision_score_`, so **higher means more anomalous**, and no inversion is
applied anywhere.

Ground truth is `data.y` binarised -- any non-zero label counts as anomalous.
That is what lets the injected-anomaly datasets (`bond_inj_*`), whose `y`
distinguishes kinds of injected anomaly rather than just flagging them, be
evaluated without a per-dataset special case.

`threshold` in the metadata is null; see `_CONTAMINATION` above for why there
is no meaningful one to report.

## What is not measured here

The trainer records the duration of `model.fit()` to `training.csv` as a spot
metric, and nothing else. Execution time, peak memory and peak GPU memory in
`results.json` come from `graflag_runner`, which samples the whole process
tree from outside the method; the four point samples this file used to take of
its own process were overwritten on every run and understated the real figures.
