# GraFlag Shared Storage

NFS-mounted directory shared across all cluster nodes. Contains methods, datasets, experiment results, and shared libraries.

## Structure

```
methods/          GAD method implementations
datasets/         Benchmark datasets
experiments/      Experiment outputs
libs/             Shared Python libraries
    graflag_runner/       Method execution wrapper with resource monitoring
    graflag_evaluator/    Metrics computation and plot generation
    graflag_bond/         PyGOD method integration layer
    graflag_data/         Dataset metadata + on-demand downloader/builder
```

## Methods

Each method directory requires at minimum `.env` and `Dockerfile`. Pattern A methods
(custom training scripts) also include a `train_graflag.py` integration script:

```
methods/method_name/
    .env               Method configuration and parameters (required)
    Dockerfile         Container definition (required)
    train_graflag.py   Custom integration script (Pattern A only, optional)
```

See `graflag-docs/METHOD_INTEGRATION_GUIDE.md` for adding new methods.

## Datasets

Each dataset lives under `datasets/<name>/` and ships only its `metadata.json`
(plus a `README.md`). The actual data files are **not** committed — they are
fetched on demand from the original source by `libs/graflag_data` whenever
`graflag run` is invoked (or explicitly via `graflag-data fetch <name>`).

`metadata.json` describes:
- `files[]` — direct-download URLs (with optional `extract` for archives)
- `build` — an optional command that regenerates files from a base dataset
  (e.g. `datasets/convert_to_strgnn.py` for the `*_snapshot` variants)
- `derived_from` — upstream dataset name for preprocessed variants

Google Drive sources are supported when `graflag_data[gdrive]` is installed
(used by `generaldyg_*`). Naming convention for dataset folders is
`methodprefix_datasetname/` or just `datasetname/`.

## Experiments

Each experiment produces:

```
experiments/exp__method__dataset__timestamp/
    status.json           Experiment lifecycle state
    results.json          Scores and ground truth
    service_config.json   Reproducible configuration
    training.csv          Training metrics log
    build.log             Docker build output
    method_output.txt     Method stdout/stderr
    eval/
        evaluation.json       Computed metrics
        roc_curve.png         ROC curve plot
        pr_curve.png          Precision-recall plot
        score_distribution.png
```
