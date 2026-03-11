# SLADE Bitcoin OTC Dataset

Bitcoin OTC trust network dataset preprocessed for SLADE method.

## Source
- Original: [Stanford SNAP Bitcoin OTC Dataset](http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- Preprocessed by: [SLADE Repository](https://github.com/jhsk777/SLADE)

## Format
CSV file with columns:
- `u`: Source node ID
- `i`: Destination node ID
- `ts`: Timestamp
- `label`: Anomaly label (0=normal, 1=anomaly)
- `idx`: Edge index

## Statistics
- ~35,592 edges
- Temporal trust network between Bitcoin users on OTC platform

## Usage
```bash
graflag benchmark -m slade -d slade_bitcoinotc --build
```
