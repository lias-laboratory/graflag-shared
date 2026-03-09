# GraFlag Evaluator

Docker-based modular evaluation system for graph anomaly detection experiments.

## Features

- **Automatic Metric Computation**: Calculates standard metrics based on result type
- **Plugin Architecture**: Easy to add custom metrics for new result types
- **Plot Generation**: ROC curves, PR curves, score distributions, spot curves
- **Spot File Integration**: Automatically detects and plots spot metrics
- **Standardized Output**: evaluation.json with all metrics and metadata
- **Docker-based**: Isolated environment with all dependencies (numpy, sklearn, matplotlib, pandas)

## Usage

### From CLI (Recommended)

```bash
# Evaluate an experiment (builds Docker image on first run)
graflag.py evaluate -e exp__generaldyg__btc_alpha__20251211_120000

# Copy results locally
graflag.py copy --from-remote -s experiments/<exp_name>/eval -d ./eval_results
```

### Manual Docker Usage

```bash
# Build image (done automatically by CLI)
cd shared/libs/graflag_evaluator
docker build -t graflag-evaluator:latest .

# Run evaluation
docker run --rm -v /shared:/shared graflag-evaluator:latest /shared/experiments/<exp_name>
```

### From Python (Direct)

```python
from graflag_evaluator import Evaluator
from pathlib import Path

# Evaluate experiment
evaluator = Evaluator(Path("experiments/exp__generaldyg__btc_alpha__20251211_120000"))
eval_path = evaluator.evaluate()

# Evaluation results saved to: experiments/.../eval/evaluation.json
# Plots saved to: experiments/.../eval/*.png
```

## Supported Result Types

The evaluator automatically detects the result type from `results.json` and computes appropriate metrics:

### All Types
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve
- **Precision@K**: Precision at top K predictions
- **Recall@K**: Recall at top K predictions
- **F1@K**: F1 score at top K
- **Best F1**: Best F1 across all thresholds

### Temporal Types
- **NODE_ANOMALY_SCORES** + temporal stats
- **TEMPORAL_EDGE_ANOMALY_SCORES** + edge stats
- **EDGE_STREAM_ANOMALY_SCORES** + edge & temporal stats
- **TEMPORAL_GRAPH_ANOMALY_SCORES** + graph stats

### Edge Types
- **EDGE_ANOMALY_SCORES** + edge statistics
- **TEMPORAL_EDGE_ANOMALY_SCORES** + edge analysis
- **EDGE_STREAM_ANOMALY_SCORES** + streaming stats

## Output Structure

```
experiments/exp_name/
├── results.json (input)
├── training.csv (optional spot file)
├── validation.csv (optional spot file)
└── eval/
    ├── evaluation.json (computed metrics)
    ├── roc_curve.png
    ├── pr_curve.png
    ├── score_distribution.png
    └── spot_curves.png (if spot files exist)
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
    "anomaly_ratio": 0.0912,
    "num_unique_edges": 3783,
    "num_unique_nodes": 1234,
    "temporal_span": 2000,
    "num_timestamps": 3783
  },
  "metadata": {
    "method_name": "generaldyg",
    "dataset": "btc_alpha",
    "exec_time": 45.67,
    "memory": 512.34,
    "gpu_memory": 2048.56
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

The evaluator uses a plugin architecture for metrics:

```python
from graflag_evaluator.metrics import MetricCalculator
import numpy as np

# Define your metric function
def compute_custom_metric(scores, ground_truth, **kwargs):
    """Compute custom metric."""
    # Your logic here
    return {
        "custom_metric": 0.123,
        "another_metric": 0.456
    }

# Register for a result type
MetricCalculator.register_metric(
    "EDGE_STREAM_ANOMALY_SCORES",
    compute_custom_metric
)

# Now evaluate() will automatically include your metrics
```

## Dependencies

```python
# Core dependencies
numpy
scikit-learn
matplotlib
pandas

# Already available in GraFlag environment
```

## Architecture

```
graflag_evaluator/
├── __init__.py          # Package exports
├── metrics.py           # Metric calculators with registry
├── plots.py             # Plot generation utilities
└── evaluator.py         # Main orchestrator
```

### Key Components

1. **MetricCalculator**: Registry-based metric computation
   - `register_metric()`: Add new metrics
   - `calculate_metrics()`: Compute all registered metrics

2. **PlotGenerator**: Visualization utilities
   - `plot_roc_curve()`: ROC curve
   - `plot_pr_curve()`: Precision-Recall curve
   - `plot_score_distribution()`: Score histograms
   - `plot_training_curves()`: Training metrics from spot files

3. **Evaluator**: Main orchestrator
   - `evaluate()`: Full evaluation pipeline
   - Auto-detects result type
   - Auto-discovers spot files
   - Generates all outputs

## Examples

### Evaluate Single Experiment

```bash
graflag.py evaluate -e exp__generaldyg__btc_alpha__20251211_120000
```

Output:
```
📊 Evaluating experiment: exp__generaldyg__btc_alpha__20251211_120000
   Result type: EDGE_STREAM_ANOMALY_SCORES
   Scores shape: (3783,)
   Ground truth shape: (3783,)
   Found 2 spot files: ['training', 'validation']
🔢 Computing metrics...
✅ Computed 12 metrics
📈 Generating plots...
✅ ROC curve saved to ...
✅ PR curve saved to ...
✅ Score distribution saved to ...
✅ Spot curves saved to ...
✅ Evaluation complete!

============================================================
📊 Evaluation Summary: exp__generaldyg__btc_alpha__20251211_120000
============================================================
Result Type: EDGE_STREAM_ANOMALY_SCORES

Key Metrics:
  auc_roc: 0.9234
  auc_pr: 0.8765
  precision_at_k: 0.85
  recall_at_k: 0.85
  f1_at_k: 0.85
  best_f1: 0.8723

Plots saved to: experiments/.../eval
============================================================
```

### Copy Results to Local

```bash
# Copy evaluation results to local machine
graflag.py copy --from-remote \
  -s experiments/exp__generaldyg__btc_alpha__20251211_120000/eval \
  -d ./eval_results
```

## Notes

- Evaluation runs on the remote cluster (where Python dependencies are available)
- Results are saved to experiment directory under `eval/`
- Use `copy --from-remote` to download results to local machine
- Plots are PNG format at 150 DPI resolution
- All metrics are rounded to 4 decimal places

## Troubleshooting

### "results.json not found"
Make sure the experiment completed successfully and wrote results.

### "No ground_truth found"
The result file must include ground truth labels for evaluation.

### "Only one class present"
The dataset has no anomalies or only anomalies - check data preparation.

### matplotlib errors
Matplotlib dependencies are required on remote cluster. Already included in GraFlag environment.

## Future Enhancements

Potential additions (plugin-based architecture makes this easy):

- **Time-series metrics**: Early detection rate, temporal precision
- **Graph structure metrics**: Community-based evaluation
- **Multi-class support**: Confusion matrices, per-class metrics
- **Comparison mode**: Compare multiple experiments side-by-side
- **Statistical tests**: Significance testing between methods
- **Export formats**: LaTeX tables, CSV reports

## See Also

- [Result Types Documentation](../../../docs/result_types.md)
- [GraFlag CLI](../../graflag/cli.py)
- [GraFlag Core](../../graflag/core.py)
