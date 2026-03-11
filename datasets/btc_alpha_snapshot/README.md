# Bitcoin Alpha Dataset (Snapshot Format)

Who-trusts-whom network converted to temporal snapshot format.

## Format
- `acc_graph.npy`: Accumulated adjacency matrices (T, N, N)
- `split.npz`: Train/test split with edge indices and snapshot IDs

## Statistics
- Nodes: 3683
- Snapshots: 10
- Train/Test edges with anomaly injection

## Compatible Methods
- strgnn
- addgraph
