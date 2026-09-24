# `graflag_runner` for integration scripts

Everything below is importable from the top level: `from graflag_runner import
ResultWriter, params, device, paths, upstream, seed_all`. The source is
`libs/graflag_runner/method.py` and its docstrings say why each function
exists, usually by naming the method that got it wrong first. Read them before
writing a variant.

Nothing here is optional style. Each helper replaced between three and ten
hand-written copies that disagreed with each other, and the disagreement is
what produced the bugs.

| Need | Call | Do not |
|---|---|---|
| Read `.env` parameters | `params(Config)` | parse `os.environ` yourself, or keep an `env_mappings` dict |
| Parameters onto upstream's own config | `apply_params(ns, ignore={"gpu"})` | `--pass-env-args` into upstream argparse |
| Pick a device | `device()` | `f"cuda:{gpu}"`, `.cuda()`, or setting `CUDA_VISIBLE_DEVICES` after importing torch |
| `DATA` / `EXP` | `paths()` | `os.environ.get("DATA", ".")` |
| Import the clone | `upstream("src")` | `sys.path.insert(0, "/app/src")` |
| Load a dataset | `load_dataset()`, `load_snapshots()`, `split_test_edges()` | copy another method's loader |
| Reproducibility | `seed_all(seed)` | seed only `torch` |
| Publish | `ResultWriter` | write `results.json` yourself |

## The skeleton

```python
from dataclasses import dataclass
from graflag_runner import (ResultWriter, params, device, paths, upstream,
                            seed_all, info)

upstream("src")                      # before importing anything from the clone
from your_module import YourModel    # noqa: E402

@dataclass
class Config:
    learning_rate: float = 0.001
    epochs: int = 100
    seed: int = 42

def main():
    config = Config(**params(Config))   # _LEARNING_RATE=0.001 -> 0.001, a float
    seed_all(config.seed)
    dev = device()                      # honours _GPU, and -1 means CPU
    p = paths()                         # p.data, p.exp -- raises if unset

    writer = ResultWriter()             # defaults to $EXP; see note below
    for epoch in range(config.epochs):
        loss = train_one_epoch(...)
        writer.spot("training", epoch=epoch, loss=loss)   # -> training.csv

    scores, truth = evaluate_on_the_test_split(...)
    writer.save_scores("EDGE_STREAM_ANOMALY_SCORES", scores, ground_truth=truth)
    writer.add_metadata(method_name="yourmethod", summary={
        "dataset_info": {"scored_split": "test", "scored_samples": len(scores)},
        "training_info": {"test_auc": float(auc)},
    })
    writer.finalize()                   # writes results.json atomically

if __name__ == "__main__":
    main()
```

## Notes that matter

**`ResultWriter()` defaults to `$EXP`; do not hand it `p.experiment`.**
`ExperimentPaths.experiment` is the experiment directory's *name*
(`exp__method__dataset__stamp`), not its path -- the path is `p.exp`. Passing
the name writes `results.json` into a relative directory of that name inside
the container, the method exits 0, and the runner then fails the run with
"Method exited 0 but wrote no results.json". This reference used to show
`ResultWriter(str(p.experiment))` and that is exactly how it happened; every
method that works calls `ResultWriter()` with no argument.

**`params(signature)` drops what the signature does not accept**, with a
warning naming the key. That is the point: it is what makes a `.env` key
that the model no longer takes visible instead of a `TypeError`. It also
coerces to the annotated type, so `_EPOCHS=100` arrives as `int`, and
`_USE_MEMORY=false` as `False` and not the truthy string `"false"`.

**`device()` reads `_GPU`, and `-1` means CPU** — PyGOD's convention, adopted
by GraFlag for every method. `graflag run --no-gpu` sets `_GPU=-1` for any
method that declares `_GPU` (an explicit `--params GPU=...` wins), so make
`device()` the only place your script decides: a device string built by hand
turns `-1` into `cuda:-1`.

**`upstream()` anchors on the calling script**, not on a WORKDIR, and raises
with the path it looked for. Call it before the imports that need it.

**`split_test_edges(split)` labels `test_neg` as 1.** The anomalies in a
snapshot split are the sampled negatives. Getting this backwards yields a
coherent-looking AUC below 0.5.

**Record the split you scored.** `metadata.summary` should carry
`scored_split` and `scored_samples`, and a `*_auc` for whatever you published.
`graflag verify` (gate 4) compares that AUC against the evaluator's; without
it there is nothing to compare, and the run passes with a warning instead of a
check.

**Do not sample resources yourself.** The runner monitors the whole process
tree and merges its numbers into `metadata` afterwards; its values win and
yours are preserved as `method_reported_*`. A method that sampled its own peak
memory reported 409 MB against the monitor's 907 MB.

**`writer.spot(key, **metrics)` locks its schema on the first call per key.**
Pass the same field names every epoch.

**`finalize()` writes to a temp file and `os.replace()`s it.** Never write
`results.json` in place: `open(..., "w")` truncates before `json.dump`
streams, so one unserialisable value leaves a truncated file that still
counts as a result.

## Result types

`{NODE,EDGE,GRAPH}_ANOMALY_SCORES`, `TEMPORAL_{NODE,EDGE,GRAPH}_ANOMALY_SCORES`,
`{NODE,EDGE,GRAPH}_STREAM_ANOMALY_SCORES`. Defined in
`libs/graflag_runner/results.py`; anything else raises. Evaluation needs
`ground_truth` with both classes present, which is the mechanical reason the
scores must come from the test split.
