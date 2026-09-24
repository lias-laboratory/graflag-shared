# GraFlag Method Template

Starting point for integrating new Graph Anomaly Detection (GAD) methods into GraFlag.

## Quick Start

1. **Copy this template:**
   ```bash
   cp -r methods/example methods/your_method_name
   ```

2. **Update the `.env` file:**
   - Set `METHOD_NAME=your_method_name` (it must equal the directory name)
   - Update `DESCRIPTION` and `SOURCE_CODE`
   - Set `INTEGRATION=upstream` if the image runs the code `SOURCE_CODE`
     points at, or `reimplementation` if you wrote the model yourself. Two
     methods here used to cite a repository they never ran.
   - Set `SOURCE_REF` to the 40-char commit if you clone anything
     (`git ls-remote <url> HEAD`)
   - Set `SUPPORTED_DATASETS` if applicable
   - Add your method's parameters (prefix with `_`)

3. **Update the `Dockerfile`:**
   - Add your dependencies
   - Clone source code if wrapping an existing implementation -- pinned to
     `SOURCE_REF`, and patch it with `git apply`, never `sed -i`
   - Update the COPY path for your method
   - Leave the `ARG GRAFLAG_LIBS` stanza alone. Which GraFlag libraries a
     build installs is the cluster's choice, not the method's.

   Or delete the Dockerfile entirely and put `IMAGE=<name>` in `.env` if an
   image under `images/` already fits -- that is how the seventeen `bond_*`
   methods share one build instead of seventeen copies of the same 13.5 GB.

4. **(Pattern A only) Implement `train_graflag.py`:**
   - Only needed if writing a custom training script (Pattern A)
   - Pattern B methods (PyGOD via graflag_bond) do not need this file
   - Replace `YourModel` class with your actual implementation
   - Let `params(YourModel)` do the argument handling -- it reads the `_FOO`
     variables, lowercases them, coerces them to the constructor's annotated
     types and drops what the constructor does not accept
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
+-- Dockerfile        Container definition (omit if .env sets IMAGE=)
```

Pattern A methods (custom training script) also include:
```
+-- train_graflag.py  Custom integration script (optional, Pattern A only)
+-- patches/*.patch   Fixes to the cloned upstream, applied with `git apply`
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

# Resource metrics -- rarely needed. graflag_runner measures execution time,
# peak memory and peak GPU from outside the method, and its numbers win;
# anything you pass here is kept beside them as method_reported_<key>.
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
- `taddy` -- Temporal anomaly detection (Pattern A, pinned clone + patches/)
- `bond_dominant` -- Pattern B: a `.env` and nothing else; the image comes
  from `images/bond_base/` via `IMAGE=bond_base`

## Also worth knowing

- **`SUPPORTED_DATASETS` is documentation.** It is printed by
  `graflag list methods` and never enforced; any pair will be attempted.
- **Images are reused unless `--build` is passed.** After editing a method,
  re-run with `--build` or the previous registry image runs again -- and that
  includes changes to `/shared/libs` made with `graflag sync --lib`.
- **`graflag-shared/tests/test_methods.py` checks this contract**: unknown
  `.env` keys, a missing `SOURCE_REF` behind a clone, an `INTEGRATION=upstream`
  with nothing cloned, a duplicated Dockerfile. Run it before you run the
  cluster; it is faster to read the failure than to discover it on a worker.
