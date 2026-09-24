# GraFlag Evaluator

Docker-based evaluation system for graph anomaly detection experiments.

## Features

- Automatic metric computation based on result type
- Plot generation: ROC curves, PR curves, score distributions, spot curves
- Spot file integration: detects and plots training/validation metrics
- Standardized output: evaluation.json with all metrics and metadata
- Docker-based: isolated environment with all dependencies

## Usage

### From CLI

```bash
# Evaluate an experiment (builds Docker image on first run)
graflag evaluate -e exp__generaldyg__btc_alpha__20251211_120000

# Copy results locally
graflag copy --from-remote -s experiments/<exp_name>/eval --dest ./eval_results
```

### Manual Docker Usage

```bash
# Build image (done automatically by CLI)
cd graflag-shared/libs/graflag_evaluator
docker build -t graflag-evaluator:latest .

# Run evaluation
docker run --rm -v /shared:/shared graflag-evaluator:latest /shared/experiments/<exp_name>
```

### From Python

```python
from graflag_evaluator import Evaluator
from pathlib import Path

evaluator = Evaluator(Path("experiments/exp__generaldyg__btc_alpha__20251211_120000"))
eval_path = evaluator.evaluate()
```

## Supported Metrics

All result types get:
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Average precision (`average_precision_score`). Not
  `auc(recall, precision)`: the trapezoidal rule interpolates linearly between
  PR operating points, which is not valid because precision does not vary
  linearly with recall.
- **Precision@K / Recall@K / F1@K** at cut-off `K`, which is reported alongside
  them as `k`. `K` defaults to the number of positives, and **at that default
  precision@K and recall@K are equal by construction** -- both are TP/K when
  K = |positives|. Pass an explicit `k` for distinct operating-point numbers.
  Ties spanning the K-th boundary are resolved by expectation over random
  tie-breaking, so a detector emitting a constant score scores exactly its base
  rate rather than whatever the array order happened to give.
- **Best F1**: Best F1 across all thresholds, with the threshold that achieves
  it (`best_f1_threshold`, which may legitimately be `0.0`).
- **filtering**: how many samples were excluded and why. Scores equal to the
  `-1` (unknown) and `-2` (inactive) sentinels from `RESULTS_STANDARD.md`, and
  any non-finite score, are dropped from the evaluation.

Additional metrics are computed based on result type (edge counts, temporal
span, etc.). Note that "early detection rate" and "temporal consistency" are
declared in `compute_temporal_metrics`' docstring but are not implemented.

If a metric is missing from `evaluation.json`, look for an `errors` list in the
same file: metric and plot failures are recorded there and reflected in the
exit code rather than passing silently.

## Output Structure

```
experiments/exp_name/
+-- results.json        (input)
+-- training.csv        (optional spot file)
+-- validation.csv      (optional spot file)
+-- eval/
    +-- evaluation.json (computed metrics)
    +-- roc_curve.png
    +-- pr_curve.png
    +-- score_distribution.png
    +-- training_curves.png    (one <key>_curves.png per spot CSV)
    +-- resources_curves.png
```

### evaluation.json Format

```json
{
  "experiment_name": "exp__generaldyg__btc_alpha__20251211_120000",
  "result_type": "EDGE_STREAM_ANOMALY_SCORES",
  "metrics": {
    "auc_roc": 0.9234,
    "auc_pr": 0.8765,
    "precision_at_k": 0.8500,
    "recall_at_k": 0.8500,
    "f1_at_k": 0.8500,
    "k": 345,
    "best_f1": 0.8723,
    "best_f1_threshold": 0.5432,
    "num_anomalies": 345,
    "num_samples": 3783,
    "anomaly_ratio": 0.0912,
    "filtering": {
      "total": 3783, "kept": 3783,
      "dropped_unknown": 0, "dropped_inactive": 0, "dropped_non_finite": 0
    }
  },
  "plots": {
    "roc_curve": "roc_curve.png",
    "pr_curve": "pr_curve.png",
    "score_distribution": "score_distribution.png",
    "training_curves": "training_curves.png",
    "resources_curves": "resources_curves.png"
  },
  "spot_files": ["training", "resources"]
}
```

## Adding Custom Metrics

```python
from graflag_evaluator.metrics import MetricCalculator

def compute_custom_metric(scores, ground_truth, **kwargs):
    return {"custom_metric": 0.123}

MetricCalculator.register_metric(
    "EDGE_STREAM_ANOMALY_SCORES",
    compute_custom_metric
)
```

## Architecture

```
graflag_evaluator/
+-- __init__.py          Package exports
+-- evaluator.py         Main orchestrator
+-- metrics.py           Metric calculators with registry
+-- plots.py             Plot generation utilities
+-- run_evaluation.py    Docker container entry point
```

## Troubleshooting

**"results.json not found"** -- Experiment hasn't completed or failed before writing results.

**"No ground_truth found"** -- The results.json must include ground_truth for evaluation.

**"Only one class present"** -- Dataset has no anomalies or only anomalies. Check data preparation.
