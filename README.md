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

Copy datasets to `datasets/` with the naming convention `methodprefix_datasetname/` or just `datasetname/`.

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
