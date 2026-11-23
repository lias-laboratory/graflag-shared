#!/bin/bash
# Record where each experiment task runs, every 20 s, on the manager.
#
#   nohup bash placements.sh > /root/placements.log 2>&1 &
#
# service_details.json records the task's node at creation time, which for a
# run queued behind busy GPUs is empty: the task is still pending. This
# samples `docker service ps` instead, one line per (service, node, state).
while true; do
  for s in $(docker service ls --format '{{.Name}}' | grep -E '^(exp|eval)__'); do
    docker service ps "$s" --no-trunc --format "$(date -u +%FT%TZ) $s {{.Node}} {{.CurrentState}}" 2>/dev/null | head -1
  done
  sleep 20
done
