# GraFlag Method Template

This template provides a starting point for integrating new Graph Anomaly Detection (GAD) methods into GraFlag.

## Quick Start

1. **Copy this template:**
   ```bash
   cp -r methods/example methods/your_method_name
   ```

2. **Update the `.env` file:**
   - Set `METHOD_NAME=your_method_name`
   - Update `DESCRIPTION` and `SOURCE_CODE`
   - Add your method's parameters (prefix with `_`)

3. **Update the `Dockerfile`:**
   - Add your dependencies
   - Clone source code if wrapping existing implementation
   - Update the COPY path for your method

4. **Implement `train_graflag.py`:**
   - Replace `YourModel` class with your actual implementation
   - Update argument parsing for your parameters
   - Implement training and prediction logic

5. **Test your method:**
   ```bash
   ./graflag_cli.py benchmark -m your_method_name -d email_snapshot --build
   ```

## File Structure

```
methods/your_method_name/
├── .env              # Method configuration and parameters
├── Dockerfile        # Container definition
├── train_graflag.py  # Main integration script
└── README.md         # Method documentation (optional)
```

## Key Concepts

### Parameter Passing

Parameters in `.env` prefixed with `_` are passed as CLI arguments:

```bash
# In .env
_LEARNING_RATE=0.001
_EPOCHS=100

# Becomes (via --pass-env-args)
python3 train_graflag.py --learning_rate 0.001 --epochs 100
```

Users can override parameters:
```bash
./graflag_cli.py benchmark -m method -d dataset --params LEARNING_RATE=0.01 EPOCHS=50
```

### Data Formats

GraFlag supports multiple dataset formats:

| Format | Files | Description |
|--------|-------|-------------|
| Edge Stream | `Data.csv`, `Label.csv` | Timestamped edges with labels |
| Snapshot | `acc_*.npy`, `split.npz` | Graph snapshots with train/test splits |
| Edge List | `edges.txt` | Simple edge list |

### Result Types

Choose the appropriate result type for `writer.save_scores()`:

| Type | Description |
|------|-------------|
| `NODE_ANOMALY_SCORES` | Static node-level anomaly scores |
| `EDGE_ANOMALY_SCORES` | Static edge-level anomaly scores |
| `GRAPH_ANOMALY_SCORES` | Graph-level anomaly scores |
| `TEMPORAL_*` | Time-indexed scores |
| `*_STREAM_*` | Streaming/dynamic scores |

### ResultWriter API

```python
from graflag_runner import ResultWriter

writer = ResultWriter()

# Log training metrics (saved to training.csv)
writer.spot("training", epoch=1, loss=0.5, auc=0.8)

# Log validation metrics (saved to validation.csv)
writer.spot("validation", epoch=1, auc=0.85)

# Save anomaly scores (required)
writer.save_scores(
    result_type="EDGE_STREAM_ANOMALY_SCORES",
    scores=[0.1, 0.9, ...],
    edges=[[0, 1], [1, 2], ...],
    timestamps=[0, 0, 1, ...],
    ground_truth=[0, 1, 0, ...],
)

# Add metadata
writer.add_metadata(
    method_name="your_method",
    dataset="dataset_name",
    method_parameters={...},
    exec_time=123.45,
    memory=512.0,
)

# Finalize and write results.json
writer.finalize()
```

## Environment Variables

These are automatically set by GraFlag:

| Variable | Description |
|----------|-------------|
| `DATA` | Path to dataset directory |
| `EXP` | Path to experiment output directory |
| `METHOD_NAME` | Method name from .env |

## Output Structure

Your method should produce files in the `EXP` directory:

```
experiments/exp__method__dataset__timestamp/
├── results.json       # Main output (created by ResultWriter)
├── training.csv       # Training metrics (from writer.spot())
├── validation.csv     # Validation metrics (from writer.spot())
├── method_output.txt  # Stdout/stderr capture
└── service_config.json # Experiment configuration
```

## Tips

1. **Reproducibility:** Always set random seeds
2. **Memory:** Track and report peak memory usage
3. **Logging:** Use `writer.spot()` for training curves
4. **Normalization:** Normalize scores to [0, 1] range
5. **Error Handling:** Gracefully handle missing labels

## Example Methods

Study these existing methods for reference:

- `generaldyg` - General dynamic GNN (PyTorch + GPU)
- `taddy` - Temporal anomaly detection
- `strgnn` - Structural temporal GNN
- `anograph` - C++ based streaming method
- `dynwalk` - Random walk + autoencoder

## Need Help?

- Check the main `CLAUDE.md` in the repository root
- Review existing method implementations
- Open an issue on GitHub
