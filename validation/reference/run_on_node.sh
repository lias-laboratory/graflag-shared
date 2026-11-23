#!/bin/bash
# Run a reference script in a method's image on one node, outside GraFlag.
#
#   validation/reference/run_on_node.sh NODE_IP IMAGE COMMAND...
#
# Copies this directory to NODE:/root/validation, then `docker run`s IMAGE
# (pulled from the cluster registry) with /root/validation mounted at /work
# and the share mounted read-only at /shared -- so a reference reads the same
# dataset files a GraFlag run reads and cannot write into them. No GraFlag
# component is involved: no service, no runner, no ResultWriter, no evaluator.
# Outputs land in NODE:/root/validation/out/. DOCKER_OPTS adds docker run
# options (e.g. --cpus 8).
set -euo pipefail
NODE=${1:?node ip}; IMAGE=${2:?image}; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
# The cluster's addresses live in validation/cluster.env, which is not
# tracked (template: cluster.env.example).
[ -f "$HERE/../cluster.env" ] && . "$HERE/../cluster.env"
KEY=${SSH_KEY:-$HOME/.ssh/id_vps}
REGISTRY=${REGISTRY:?set REGISTRY in validation/cluster.env (see cluster.env.example)}
PROXY_ENV=""
if [ -n "${HTTPS_PROXY_URL:-}" ]; then
  PROXY_ENV="-e http_proxy=$HTTPS_PROXY_URL -e https_proxy=$HTTPS_PROXY_URL -e no_proxy=localhost,127.0.0.1${CLUSTER_SUBNET:+,$CLUSTER_SUBNET}"
fi
rsync -a --exclude out -e "ssh -i $KEY" "$HERE/" "root@$NODE:/root/validation/"
ssh -i "$KEY" "root@$NODE" \
  "docker pull -q $REGISTRY/$IMAGE >/dev/null && \
   docker run --rm --network host $PROXY_ENV ${DOCKER_OPTS:-} \
     -v /root/validation:/work -v /shared:/shared:ro \
     --entrypoint '' $REGISTRY/$IMAGE $(printf '%q ' "$@")"
