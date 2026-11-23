#!/bin/bash
# Keep the manager's disk from filling during a long campaign.
#
#   nohup bash janitor.sh > /root/janitor.log 2>&1 &
#
# The manager builds every method image, pushes it to the cluster registry,
# and keeps a local copy it rarely needs again: tasks pull from the registry,
# so a later run on the manager re-pulls. With ~16 GB per image (and BuildKit
# cache) the manager's 244 GB filled before the catalogue was built. Every two
# minutes, when free space is under THRESHOLD_GB, this removes the local copy
# of a method image only when all of these hold:
#   - the registry serves a manifest for it (so nothing is lost);
#   - no container on the manager uses it;
#   - no `docker build` or `docker push` naming it is running;
# and never the evaluator image, which GraFlag looks for locally to decide
# whether to rebuild it. It then drops unreferenced build cache.
# REGISTRY (MANAGER_IP:5000) comes from the environment or from a cluster.env
# beside this script; the repository holds no address (cluster.env.example).
[ -f "$(dirname "$0")/cluster.env" ] && . "$(dirname "$0")/cluster.env"
REGISTRY=${REGISTRY:?set REGISTRY, e.g. in a cluster.env beside this script}
THRESHOLD_GB=${THRESHOLD_GB:-70}
while true; do
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt "$THRESHOLD_GB" ]; then
    echo "[INFO] $(date -u +%FT%TZ) ${free_gb} GB free, below ${THRESHOLD_GB}"
    busy=$(ps -eo args | grep -E "docker (build|push)" | grep -v grep)
    used=$(docker ps --format '{{.Image}}')
    for repo in $(docker images --format '{{.Repository}}' | grep -v -e '^<none>' -e '/' | sort -u); do
      case "$repo" in graflag-evaluator|registry|ubuntu) continue ;; esac
      if echo "$busy" | grep -q -e "-t $repo:" -e "$REGISTRY/$repo:"; then continue; fi
      if echo "$used" | grep -q -e "^$repo:" -e "^$REGISTRY/$repo:"; then continue; fi
      code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' \
        -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json' \
        "http://$REGISTRY/v2/$repo/manifests/latest")
      [ "$code" = 200 ] || continue
      echo "[INFO] removing local copies of $repo (registry has it)"
      docker rmi "$repo:latest" "$REGISTRY/$repo:latest" >/dev/null 2>&1 || true
    done
    docker builder prune -f >/dev/null 2>&1 || true
    echo "[INFO] $(date -u +%FT%TZ) now $(df --output=avail -BG / | tail -1 | tr -dc 0-9) GB free"
  fi
  sleep 120
done
