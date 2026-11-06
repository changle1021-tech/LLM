#!/usr/bin/env bash
# Run inside the same vLLM environment on each participating node.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    cat <<'HELP'
Usage: ray_multinode.sh head|worker|status [native Ray arguments...]
  head:   NODE_IP=<head_ip> GPUS_PER_NODE=4 ray_multinode.sh head
  worker: NODE_IP=<worker_ip> RAY_HEAD_ADDRESS=<head_ip>:6379 GPUS_PER_NODE=4 ray_multinode.sh worker
  status: RAY_HEAD_ADDRESS=<head_ip>:6379 ray_multinode.sh status
RAY_PORT defaults to 6379. GPUS_PER_NODE is optional (Ray auto-detection).
Head and worker commands stay in the foreground. GPU placement is user-managed.
HELP
}
mode="${1:-}"
[[ $# -eq 0 ]] || shift
case "$mode" in
    -h|--help) usage; exit 0 ;;
    status)
        cmd=(ray status)
        [[ -z "${RAY_HEAD_ADDRESS:-}" ]] || cmd+=(--address "$RAY_HEAD_ADDRESS")
        exec "${cmd[@]}" "$@"
        ;;
    head|worker)
        : "${NODE_IP:?Set NODE_IP to a reachable cluster IP for this node}"
        export VLLM_HOST_IP="$NODE_IP"
        cmd=(ray start --node-ip-address "$NODE_IP" --disable-usage-stats --block)
        if [[ "$mode" == head ]]; then
            cmd+=(--head --port "${RAY_PORT:-6379}")
        else
            : "${RAY_HEAD_ADDRESS:?Set RAY_HEAD_ADDRESS to the head IP:port}"
            cmd+=(--address "$RAY_HEAD_ADDRESS")
        fi
        if [[ -n "${GPUS_PER_NODE:-}" ]]; then
            [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || { echo 'GPUS_PER_NODE must be positive' >&2; exit 2; }
            cmd+=(--num-gpus "$GPUS_PER_NODE")
        fi
        exec "${cmd[@]}" "$@"
        ;;
    *) usage >&2; exit 2 ;;
esac
