#!/usr/bin/env bash
# Independent replay wrapper. Run once on the Ray launch node.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == validate ]]; then
    shift
    exec "$PYTHON_BIN" -m bench_multinode validate "$@"
fi
[[ "${1:-}" != run ]] || shift

MODEL="${MODEL:-Qwen/Qwen3-32B}"
DATASET="${DATASET:-workloads/sharegpt-qwen3-32b-300-sps10.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
OUTPUT_DIR="${OUTPUT_DIR:-bench_multinode/results/$RUN_ID}"
cmd=("$PYTHON_BIN" -m bench_multinode run
    --model "$MODEL" --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
    --tensor-parallel-size "${TP:-1}" --pipeline-parallel-size "${PP:-1}"
    --data-parallel-size "${DP:-1}"
    --distributed-executor-backend ray
    --ray-address "${RAY_ADDRESS:-auto}"
    --max-num-seqs "${MAX_NUM_SEQS:-128}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --dtype "${DTYPE:-bfloat16}" --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}"
    --seed "${SEED:-42}" --load-format "${LOAD_FORMAT:-auto}"
    --tick-seconds "${TICK_SECONDS:-1}" --num-reqs "${NUM_REQS:-0}"
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS:-0}"
    --log-level "${LOG_LEVEL:-INFO}"
)
[[ -z "${DATA_PARALLEL_BACKEND:-}" ]] || cmd+=(--data-parallel-backend "$DATA_PARALLEL_BACKEND")
[[ -z "${DP_LOCAL:-}" ]] || cmd+=(--data-parallel-size-local "$DP_LOCAL")
[[ -z "${RAY_DP_PACK_STRATEGY:-}" ]] || cmd+=(--ray-dp-pack-strategy "$RAY_DP_PACK_STRATEGY")
[[ -z "${MAX_MODEL_LEN:-}" ]] || cmd+=(--max-model-len "$MAX_MODEL_LEN")
[[ -z "${KV_CACHE_MEMORY_BYTES:-}" ]] || cmd+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
[[ "${EXPERT_PARALLEL:-0}" != 1 ]] || cmd+=(--enable-expert-parallel)
[[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" != 1 ]] || cmd+=(--disable-custom-all-reduce)
[[ "${RESOLVE_ONLY:-0}" != 1 ]] || cmd+=(--resolve-only)
[[ "${SKIP_TOKENIZER_INIT:-0}" != 1 ]] || cmd+=(--skip-tokenizer-init)
[[ "${ENFORCE_EAGER:-0}" != 1 ]] || cmd+=(--enforce-eager)
# Explicit CLI arguments override environment-derived scalar defaults.
exec "${cmd[@]}" "$@"
