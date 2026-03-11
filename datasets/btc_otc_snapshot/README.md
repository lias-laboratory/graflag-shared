# Bitcoin OTC Dataset (Snapshot Format)

Who-trusts-whom network converted to temporal snapshot format.

## Format
- `acc_graph.npy`: Accumulated adjacency matrices (T, N, N)
- `split.npz`: Train/test split with edge indices and snapshot IDs

## Compatible Methods
- strgnn
- addgraph
