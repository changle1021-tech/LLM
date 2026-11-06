#!/usr/bin/env bash
# Independent copy of the vLLM container launcher with multi-node networking.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
detached_mode="${VLLM_DETACHED:-0}"
case "$detached_mode" in
    0|1) ;;
    *) echo 'VLLM_DETACHED must be 1 (detached container) or 0 (interactive)' >&2; exit 2 ;;
esac

mount_args=()
for spec in ${VLLM_EXTRA_MOUNTS:-}; do
    mount_args+=(-v "$spec")
done
env_args=()
for name in HF_TOKEN NODE_IP RAY_HEAD_ADDRESS RAY_PORT GPUS_PER_NODE RAY_PACKAGE \
    VLLM_HOST_IP NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_HCA \
    NCCL_IB_GID_INDEX NCCL_IB_DISABLE UCX_NET_DEVICES \
    VLLM_RAY_DP_PACK_STRATEGY VLLM_RAY_EXTRA_ENV_VARS_TO_COPY \
    VLLM_ALLREDUCE_USE_SYMM_MEM VLLM_ALLREDUCE_USE_FLASHINFER \
    VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC; do
    [[ ! -v "$name" ]] || env_args+=(-e "$name")
done
rdma_args=()
rdma_mode="${VLLM_RDMA:-1}"
case "$rdma_mode" in
    1)
        # Capture status before reading lines; process substitution would hide
        # discovery failures and could start a container without RDMA devices.
        rdma_output="$("${PYTHON_BIN:-python3}" "$SCRIPT_DIR/rdma_multinode.py" \
            --gpus "${VLLM_GPUS:-all}" --format docker-args)"
        mapfile -t rdma_args <<< "$rdma_output"
        ;;
    0) ;;
    *) echo 'VLLM_RDMA must be 1 (automatic GPU/RDMA affinity) or 0 (disabled)' >&2; exit 2 ;;
esac
tty_args=(-it)
remove_args=(--rm)
if [[ "$detached_mode" == 1 ]]; then
    tty_args=(-d)
    # nerdctl rejects --rm together with detached mode.
    remove_args=()
fi
cmd=(docker run "${remove_args[@]}" --name "${VLLM_CONTAINER_NAME:-vllm_multinode}"
    --gpus "${VLLM_GPUS:-all}" --network host --ipc host "${tty_args[@]}"
    -v "$REPO_ROOT:/workspace"
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
    -v "$HOME/.cache/vllm:/root/.cache/vllm"
    -e "VLLM_RDMA=$rdma_mode"
    -e "VLLM_DETACHED=$detached_mode"
    -w /workspace --entrypoint /bin/bash
)
exec "${cmd[@]}" "${mount_args[@]}" "${env_args[@]}" "${rdma_args[@]}" \
    "${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}" -c \
    'set -e; python3 -m pip install datasets matplotlib pandas "nvidia-nccl-cu13==2.29.7" "${RAY_PACKAGE:-ray[cgraph]}"; for p in scripts/patches/*.py; do python3 "$p"; done; export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"; if [[ "${VLLM_RDMA:-1}" == 1 ]]; then python3 bench_multinode/rdma_multinode.py --configure-worker-env; fi; if [[ "${VLLM_DETACHED:-0}" == 1 ]]; then touch /tmp/llmsim-container-ready; exec sleep infinity; fi; exec bash'
