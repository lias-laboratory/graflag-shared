# Pinned-memory copies on the cluster nodes

Found 2026-11-17 while GeneralDyG's first epoch went NaN. A host-to-GPU copy
from **pinned** host memory delivers corrupted data on node-1 and
node-2. Copies from pageable memory are intact on every node, and
node-3 is intact either way. No method code is needed to see it:

    docker run --rm -v $PWD:/work --entrypoint '' <any torch image> \
        python3 /work/pinned_batch.py 20

`pinned_batch.py` builds batches shaped like GeneralDyG's (one 128 x 343 x 128
float64 tensor, about 45 MB, plus lists of small matrices), pins them with the
DataLoader's own `pin_memory()`, moves them with a plain synchronous
`.to("cuda")`, and compares the result with the original:

| Node | torch 2.1.2 (generaldyg image) | torch 2.5.1 (bond_base image) | pageable |
|---|---|---|---|
| node-1 | 19 of 20 batches corrupted | 18 of 20 | 0 of 20 |
| node-2 | 19 of 20 | 18 of 20 | 0 of 20 |
| node-3 | 0 of 20 | 0 of 20 | 0 of 20 |

Without the 45 MB tensor (`pinned_batch.py 20 small`), and with many small
pinned tensors only (`pinned_h2d.py`), nothing is corrupted on any node.

All three are KVM guests with the GPU (an NVIDIA A16, driver 580.178.04) in
PCI pass-through, on AMD EPYC 7713 hosts. What differs:

| | node-1, node-2 | node-3 |
|---|---|---|
| guest IOMMU | QEMU's emulated Intel VT-d (`DMAR ... BOCHS BXPC`, "IOMMU batching disallowed due to virtualization") | `virtio-iommu` |
| kernel | 6.8.0-149-generic | 6.8.0-124-generic |

Which of the two is responsible has not been isolated: that needs a reboot with
a different IOMMU setting or kernel, which is the cluster administrators' call.

**Effect on the benchmark.** `pin_scan.sh` searched every method image for
pinned host memory: only GeneralDyG pins (`pin_memory=True`, from upstream's
`train.py`), and it never completed a run on this cluster before the fix.
RARE copies with `non_blocking=True`, but from pageable tensors, which the
driver stages itself. GeneralDyG's loader no longer pins.
