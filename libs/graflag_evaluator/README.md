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
graflag copy --from-remote -s experiments/<exp_name>/eval -d ./eval_results
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
- **AUC-PR**: Area under Precision-Recall curve
- **Precision@K**: Precision at top K predictions
- **Recall@K**: Recall at top K predictions
- **F1@K**: F1 score at top K
- **Best F1**: Best F1 across all thresholds

Additional metrics are computed based on result type (edge counts, temporal span, etc.).

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
    +-- spot_curves.png (if spot files exist)
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
    "best_f1": 0.8723,
    "best_f1_threshold": 0.5432,
    "num_anomalies": 345,
    "num_samples": 3783,
    "anomaly_ratio": 0.0912
  },
  "plots": {
    "roc_curve": "roc_curve.png",
    "pr_curve": "pr_curve.png",
    "score_distribution": "score_distribution.png",
    "spot_curves": "spot_curves.png"
  },
  "spot_files": ["training", "validation"]
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
