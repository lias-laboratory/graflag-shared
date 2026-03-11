# StrGNN Email Dataset (5% Anomaly Ratio)

This is a variant of the StrGNN email dataset with 5% anomaly ratio instead of 1%.

## Source

- **Paper**: StrGNN: Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic Graphs (CIKM 2021)
- **Repository**: https://github.com/KnowledgeDiscovery/StrGNN

## Files

- `acc_email.npy`: Accumulated email graph snapshots (T × N × N tensor)
- `sta_email.npy`: Static email graph
- `split.npz`: Train/test split with 5% anomaly ratio (copied from email0.05.npz)

## Task

Edge anomaly detection in dynamic graphs with higher anomaly ratio (5% vs 1% in the original).
