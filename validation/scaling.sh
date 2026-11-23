#!/bin/bash
# Multi-node scaling: the same six GPU jobs on 1, 2 and 3 active nodes.
#
#   validation/scaling.sh CONFIG.env
#
# Each node has one GPU and every job reserves it (NVIDIA-GPU=1), so a node
# runs one job at a time and the swarm queues the rest. The manager stays
# active throughout -- the registry is a swarm service on it -- and workers are
# drained to shrink the cluster. All nodes are set back to active on exit,
# whatever happens. bond_base is pulled on every node first so a node that
# joins late does not pay a pull the others did not.
set -euo pipefail
CONFIG=${1:?config.env}
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=${SSH_KEY:-$HOME/.ssh/id_vps}
# The cluster's addresses and worker names live in validation/cluster.env,
# which is not tracked (template: cluster.env.example).
[ -f "$HERE/cluster.env" ] || { echo "[ERROR] $HERE/cluster.env is missing (see cluster.env.example)" >&2; exit 1; }
. "$HERE/cluster.env"
MANAGER=${MANAGER:?MANAGER is not set in cluster.env}
read -r -a NODES <<< "${NODES:?NODES is not set in cluster.env}"
read -r -a WORKERS <<< "${WORKERS:?WORKERS is not set in cluster.env}"
OUT=$HERE/results/scaling.jsonl
SUMMARY=$HERE/results/scaling-summary.jsonl
mgr() { ssh -i "$KEY" "root@$MANAGER" "$@"; }
restore() { for w in "${WORKERS[@]}"; do mgr docker node update --availability active "$w" >/dev/null || true; done; }
trap restore EXIT

if mgr 'docker service ls --format "{{.Name}}" | grep -q "^exp__"'; then
  echo "[ERROR] experiment services are still running; scaling needs an idle cluster" >&2
  exit 1
fi
for n in "${NODES[@]}"; do
  ssh -i "$KEY" "root@$n" "docker pull -q ${REGISTRY:-$MANAGER:5000}/bond_base:latest" >/dev/null
done
echo "[OK] bond_base present on all nodes"

for active in 1 2 3; do
  restore
  for ((i = active - 1; i < ${#WORKERS[@]}; i++)); do
    mgr docker node update --availability drain "${WORKERS[$i]}" >/dev/null
  done
  mgr 'docker node ls --format "{{.Hostname}} {{.Availability}}"' | tr '\n' ' '; echo
  t0=$(date +%s.%N)
  python3 -u "$HERE/campaign.py" "$HERE/jobs-scaling.json" --config "$CONFIG" \
      --out "$OUT" --workers 6 --label-suffix="-n$active" \
      > "$HERE/results/scaling-n$active.console.log" 2>&1
  t1=$(date +%s.%N)
  makespan=$(python3 -c "print(round($t1 - $t0, 1))")
  echo "{\"active_nodes\": $active, \"jobs\": 6, \"makespan_s\": $makespan}" | tee -a "$SUMMARY"
done
