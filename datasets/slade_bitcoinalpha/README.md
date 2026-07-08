# SLADE Bitcoin Alpha Dataset

Bitcoin Alpha trust network dataset preprocessed for SLADE method.

## Source
- Original: [Stanford SNAP Bitcoin Alpha Dataset](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
- Preprocessed by: [SLADE Repository](https://github.com/jhsk777/SLADE)

## Format
CSV file with columns:
- `u`: Source node ID
- `i`: Destination node ID
- `ts`: Timestamp
- `label`: Anomaly label (0=normal, 1=anomaly)
- `idx`: Edge index

## Statistics
- ~24,186 edges
- Temporal trust network between Bitcoin users

## Usage
```bash
graflag benchmark -m slade -d slade_bitcoinalpha --build
```
