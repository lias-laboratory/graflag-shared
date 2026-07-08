# SLADE Wikipedia Dataset

Wikipedia edit network dataset preprocessed for SLADE method.

## Source
- Original: [JODIE Dataset](http://snap.stanford.edu/jodie/)
- Preprocessed by: [SLADE Repository](https://github.com/jhsk777/SLADE)

## Format
CSV file with columns:
- `u`: Source node ID (user)
- `i`: Destination node ID (page)
- `ts`: Timestamp
- `label`: Anomaly label (0=normal, 1=anomaly - banned users)
- `idx`: Edge index

## Statistics
- ~157,474 edges
- Bipartite temporal graph (users editing Wikipedia pages)
- Anomalies: edits by users who were later banned

## Usage
```bash
graflag benchmark -m slade -d slade_wikipedia --build
```

## SLADE-HP Parameters (Hyperparameter-tuned)
For better performance on this dataset:
```bash
graflag benchmark -m slade -d slade_wikipedia --build --params BS=300 SRF=10 DRF=10
```
