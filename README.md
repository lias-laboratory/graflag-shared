# GraFlag Shared Storage

NFS-mounted directory shared across all cluster nodes. Contains methods, datasets, experiment results, and shared libraries.

## Structure

```
methods/          GAD method integrations (33, plus the annotated example/)
images/           Dockerfiles shared by several methods (bond_base: the 17 PyGOD detectors)
datasets/         Benchmark dataset descriptors (43); files are fetched on demand
experiments/      Experiment outputs, created at run time
libs/             Shared Python libraries
    graflag_runner/       Method execution wrapper with resource monitoring
    graflag_evaluator/    Metrics computation and plot generation
    graflag_bond/         PyGOD method integration layer
    graflag_data/         Dataset metadata + on-demand downloader/builder
tests/            The method contract: checks every method definition
.claude/skills/method-integration/   Agent skill for integrating methods (see below)
```

## Methods

Each method directory holds a `.env`, a `README.md`, and either its own
`Dockerfile` or an `IMAGE=` key naming a shared image under `images/`. Methods
with an integration script also include `train_graflag.py`:

```
methods/method_name/
    .env               Method configuration and parameters (required)
    README.md          What upstream does and what the integration changes (required)
    Dockerfile         Container definition (unless .env sets IMAGE=)
    train_graflag.py   Integration script, when COMMAND names one
```

See the [method integration guide](https://lias-laboratory.github.io/graflag/METHOD_INTEGRATION_GUIDE.html)
(`docs/METHOD_INTEGRATION_GUIDE.md` in the graflag repository) for adding new methods,
and check every definition with the contract tests:

```bash
python3 -m unittest discover -s tests
```

### Integrating a method with an AI agent

`.claude/skills/method-integration/` is an agent skill that walks a coding agent
through an integration and its four gates, in order: the contract tests, a
build and run on the cluster, `graflag evaluate`, and `graflag verify`, which
checks that the published scores reproduce the AUC the method reported. Each
method's README records what the gates gave in its `## Verification` section
(or `VERIFICATION.md` does), and the contract tests refuse a method with
neither. Claude Code picks the skill up when started in this repository
(`/method-integration <repository URL>`); other agents can be pointed at
`SKILL.md`. The check also runs on its own, on any finished experiment:

```bash
graflag verify -e exp__method__dataset__timestamp      # graflag 1.2.0 or later
```

With the GraFlag MCP server configured (`graflag mcp`), an agent can drive
gates 2 to 4 as tools instead. The [skill's
page](https://lias-laboratory.github.io/graflag/AGENT_SKILL.html) in the
documentation has the details.

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
