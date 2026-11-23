#!/bin/bash
# For each method image on this node: every use of pinned host memory in its code.
#   REGISTRY=MANAGER_IP:5000 bash pin_scan.sh IMAGE...
REGISTRY=${REGISTRY:?set REGISTRY to the cluster registry, MANAGER_IP:5000}
for img in "$@"; do
  hits=$(docker run --rm --entrypoint '' -e NVIDIA_VISIBLE_DEVICES=void ${REGISTRY}/${img}:latest sh -c \
    'grep -rn -E "pin_memory|pin_memory_|non_blocking *= *True|cudaHostAlloc|cudaMallocHost|cudaHostRegister|\.pin\(\)" /app --include=*.py --include=*.cpp --include=*.cu --include=*.c --include=*.h 2>/dev/null | grep -v "/\.git/" | head -8' 2>/dev/null)
  if [ -n "$hits" ]; then echo "--- $img"; echo "$hits" | cut -c1-170; else echo "--- $img: none"; fi
done
