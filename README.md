# GraFlag Shared Storage

The shared directory of [GraFlag](https://github.com/lias-laboratory/graflag), a
platform for reproducible benchmarking of graph anomaly detection (GAD) methods.
It holds the method definitions, the dataset descriptors and the libraries that
run inside the method containers. On a GraFlag cluster this repository is the
NFS-exported directory that every node mounts, and each experiment writes its
outputs into it. Documentation: https://lias-laboratory.github.io/graflag/

## Setup

Clone it into the NFS export on the cluster's manager (`SHARED_DIR` in the
GraFlag configuration, `/shared` by default):

```bash
cd /shared
git clone https://github.com/lias-laboratory/graflag-shared.git .
```

The [quickstart](https://lias-laboratory.github.io/graflag/quickstart.html)
covers the cluster and the `graflag` client that drives it.

## Structure

```
methods/          GAD method integrations (33, plus the annotated example/)
images/           Dockerfiles shared by several methods (bond_base: the 17 PyGOD detectors)
datasets/         Benchmark dataset descriptors (43); files are fetched on demand
experiments/      Experiment outputs, created at run time (not in the repository)
libs/             Shared Python libraries, also published on PyPI
    graflag_runner/       Method execution wrapper with resource monitoring (graflag-runner)
    graflag_evaluator/    Metrics computation and plot generation (graflag-evaluator)
    graflag_bond/         PyGOD method integration layer (graflag-bond)
    graflag_data/         Dataset metadata + on-demand downloader/builder (graflag-data)
tests/            The method contract: checks every method definition
VERIFICATION.md   End-to-end runs of the first methods; later ones record theirs in their README
.claude/skills/method-integration/   Agent skill for integrating methods (see below)
.dockerignore     Keeps datasets/ and experiments/ out of image builds; keep it at the root
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
neither. AI agents that read `.claude/skills/` pick the skill up when started
in this repository (`/method-integration <repository URL>`); other agents can be
pointed at `SKILL.md`. The check also runs on its own, on any finished experiment:

```bash
graflag verify -e exp__method__dataset__timestamp      # graflag 1.2.0 or later
```

With the GraFlag MCP server configured (`graflag mcp`), an agent can drive
gates 2 to 4 as tools instead. The [skill's
page](https://lias-laboratory.github.io/graflag/AGENT_SKILL.html) in the
documentation has the details.

## Datasets

Each dataset lives under `datasets/<name>/` and ships only its `metadata.json`,
with a `README.md` for some. The data files are **not** committed: they are
fetched on demand from the original source by `libs/graflag_data` whenever
`graflag run` needs them (or explicitly via `graflag-data fetch <name>`).

`metadata.json` describes:
- `source`, `source_repo`, `license` — where the data comes from
- `files[]` — direct-download URLs, with optional `extract` and `members` for
  archives and a `sha256` checksum, which `graflag-data verify` re-checks
- `build` — an optional command that produces files from the downloaded ones
  (e.g. `datasets/convert_to_strgnn.py` for the `*_snapshot` variants)
- `derived` / `derived_from` — marks a preprocessed variant and, when it is
  built from another dataset here, names it

Google Drive sources are supported when `graflag_data[gdrive]` is installed
(used by `generaldyg_*`). Naming convention for dataset folders is
`methodprefix_datasetname/` or just `datasetname/`.

## Experiments

Each experiment produces:

```
experiments/exp__method__dataset__timestamp/
    status.json           Experiment lifecycle state
    build.log             Pinned commit and Docker build output (with --build)
    results.json          Scores and ground truth
    method_output.txt     Method stdout/stderr
    service_config.json   The configuration that `graflag run --from-config` replays
    service_details.json  Docker service metadata
    resources.csv         Memory and GPU samples from the runner's monitor
    training.csv          Per-epoch metrics, when the method records them
    eval/
        evaluation.json       Computed metrics
        roc_curve.png         ROC curve plot
        pr_curve.png          Precision-recall plot
        score_distribution.png
        resources_curves.png  One <name>_curves.png per CSV above
```

## License

MIT; see `LICENSE`. Each dataset's `metadata.json` names its source and, where
known, its license.
