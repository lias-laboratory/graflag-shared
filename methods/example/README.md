# GraFlag Method Template

Starting point for integrating new Graph Anomaly Detection (GAD) methods into GraFlag.

## Quick Start

1. **Copy this template:**
   ```bash
   cp -r methods/example methods/your_method_name
   ```

2. **Update the `.env` file:**
   - Set `METHOD_NAME=your_method_name`
   - Update `DESCRIPTION` and `SOURCE_CODE`
   - Set `SUPPORTED_DATASETS` if applicable
   - Add your method's parameters (prefix with `_`)

3. **Update the `Dockerfile`:**
   - Add your dependencies
   - Clone source code if wrapping existing implementation
   - Update the COPY path for your method

4. **(Pattern A only) Implement `train_graflag.py`:**
   - Only needed if writing a custom training script (Pattern A)
   - Pattern B methods (PyGOD via graflag_bond) do not need this file
   - Replace `YourModel` class with your actual implementation
   - Update argument parsing for your parameters
   - Implement training and prediction logic

5. **Test your method:**
   ```bash
   graflag run -m your_method_name -d your_dataset --build
   ```

## File Structure

Required files (all methods):
```
methods/your_method_name/
+-- .env              Method configuration and parameters
+-- Dockerfile        Container definition
```

Pattern A methods (custom training script) also include:
```
+-- train_graflag.py  Custom integration script (optional, Pattern A only)
```

The `COMMAND` field in `.env` defines what runs inside the container. Pattern A methods
typically set `COMMAND=python3 train_graflag.py`, while Pattern B methods (PyGOD via
graflag_bond) set `COMMAND=python3 -m graflag_bond.train` and need no additional files.

## Parameter Passing

Parameters in `.env` prefixed with `_` are passed as CLI arguments when using `--pass-env-args`:

```bash
# In .env
_LEARNING_RATE=0.001
_EPOCHS=100

# Becomes (via --pass-env-args)
python3 train_graflag.py --learning_rate 0.001 --epochs 100
```

Users can override parameters:
```bash
graflag run -m method -d dataset --params LEARNING_RATE=0.01 EPOCHS=50
```

## Result Types

Choose the appropriate result type for `writer.save_scores()`:

| Type | Description |
|------|-------------|
| `NODE_ANOMALY_SCORES` | Static node-level anomaly scores |
| `EDGE_ANOMALY_SCORES` | Static edge-level anomaly scores |
| `GRAPH_ANOMALY_SCORES` | Graph-level anomaly scores |
| `TEMPORAL_*` | Time-indexed scores (2D arrays) |
| `*_STREAM_*` | Streaming scores (1D with timestamps) |

## ResultWriter API

```python
from graflag_runner import ResultWriter

writer = ResultWriter()

# Log training metrics (saved to training.csv)
writer.spot("training", epoch=1, loss=0.5, auc=0.8)

# Save anomaly scores (required)
writer.save_scores(
    result_type="NODE_ANOMALY_SCORES",
    scores=[0.1, 0.9, 0.3],
    ground_truth=[0, 1, 0],
)

# Add metadata
writer.add_metadata(method_name="your_method", dataset="cora")

# Add resource metrics (optional, also set automatically by graflag_runner)
writer.add_resource_metrics(exec_time_ms=1234.5, peak_memory_mb=512.3, peak_gpu_mb=2048.0)

# Finalize and write results.json
writer.finalize()
```

## Environment Variables

Automatically set by GraFlag:

| Variable | Description |
|----------|-------------|
| `DATA` | Path to dataset directory |
| `EXP` | Path to experiment output directory |
| `METHOD_NAME` | Method name from .env |
| `COMMAND` | Command from .env |

## Example Methods

Study these for reference:

- `generaldyg` -- Dynamic GNN (Pattern A, --pass-env-args)
- `taddy` -- Temporal anomaly detection (Pattern A, --pass-env-args)
- `bond_dominant` -- Deep matrix factorization (Pattern B, graflag_bond)
- `bond_cola` -- Contrastive learning (Pattern B, graflag_bond)
