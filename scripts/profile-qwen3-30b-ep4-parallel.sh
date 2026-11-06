#!/usr/bin/env bash
# Profile TP1/DP4/EP4 and TP2/DP2/EP4 on two or four identical GPUs.
# Run inside the vLLM environment. GPU IDs refer to devices visible there.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${HW:?Set HW to the hardware label, for example H100_SXM_80GB}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/profiler/perf_qwen_ep4_parallel}"
COMMON=(
  --hardware "$HW"
  --dtype bfloat16
  --gpu-memory-utilization 0.9
  --max-model-len 8192
  --max-num-batched-tokens 2048
  --max-num-seqs 256
  --attention-max-kv 4096
  --attention-decode-q-lens 1
  --attention-chunk-factor 2.0
  --attention-kv-factor 2.0
  --attention-n-factor 1.4142135623730951
  --grid-factor 2.0
  --measurement-iterations 3
  --moe-rounds 3
  --moe-grid-factor 2.0
  --skew-n-factor 2.0
  --skew-pc-factor 2.0
  --skew-kp-factor 2.0
  --skew-kvs-factor 2.0
  --skew-samples-per-cell 32
  --skew-rounds 3
  --skew-seed 0
  --log-level INFO
)

run() {
  local gpu="$1" command="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m profiler "$command" "$MODEL" \
    "${COMMON[@]}" --out-root "$ACTIVE_ROOT" "$@"
}

step() {
  local name="$1"
  shift
  mkdir -p "$ACTIVE_ROOT/.pipeline"
  if [[ ! -f "$ACTIVE_ROOT/.pipeline/$name.done" ]]; then
    run "$@"
    touch "$ACTIVE_ROOT/.pipeline/$name.done"
  fi
}

ordinary() {
  local gpu="$1" tp="$2" group
  ACTIVE_ROOT="$OUT_ROOT/.parallel-workers/ordinary_tp$tp"
  for group in dense per_sequence attention; do
    step "$group" "$gpu" slice --tp 1,2 --tp-refresh "$tp" --group "$group"
  done
  # Local skew sees only this worker's attention table. Pipeline checkpoints
  # resume an interrupted attention/skew step before any earlier metadata write.
  step skew "$gpu" profile --tp "$tp" --only-skew
}

ordinary_tp1() { ordinary "$1" 1; }
ordinary_tp2() { ordinary "$1" 2; }

moe_tp1() {
  ACTIVE_ROOT="$OUT_ROOT/.parallel-workers/moe_tp1"
  step moe "$1" slice --tp 1,2 --tp-refresh 1 --group moe --dp 4 --ep 4
}

moe_tp2() {
  ACTIVE_ROOT="$OUT_ROOT/.parallel-workers/moe_tp2"
  step moe "$1" slice --tp 1,2 --tp-refresh 2 --group moe --dp 2 --ep 4
}

if [[ "${1:-}" == --worker ]]; then
  case "$2" in
    ordinary_tp1|ordinary_tp2|moe_tp1|moe_tp2) "$2" "$3" ;;
    combined_tp1) ordinary_tp1 "$3"; moe_tp1 "$3" ;;
    combined_tp2) ordinary_tp2 "$3"; moe_tp2 "$3" ;;
    *) exit 2 ;;
  esac
  exit 0
fi

IFS=',' read -r -a GPUS <<< "${GPU_IDS:-0,1,2,3}"
if [[ ${#GPUS[@]} != 2 && ${#GPUS[@]} != 4 ]]; then
  echo 'GPU_IDS must contain two or four distinct GPU IDs, separated by commas.' >&2
  exit 2
fi
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  if [[ -z "$gpu" || "$gpu" == *[[:space:]]* || -n "${SEEN[$gpu]:-}" ]]; then
    echo 'GPU_IDS must contain distinct, nonempty GPU IDs without spaces.' >&2
    exit 2
  fi
  SEEN[$gpu]=1
done

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
export HW OUT_ROOT PYTHON_BIN
exec 9>"$OUT_ROOT/.qwen-ep4-parallel.lock"
flock -n 9 || { echo 'Another parallel acquisition owns this output root.' >&2; exit 1; }
LOG_DIR="$(mktemp -d "$OUT_ROOT/logs-XXXXXXXX")"
echo "Logs: $LOG_DIR"

PIDS=()
cleanup() {
  local pid
  for pid in $(jobs -pr); do
    # Each worker and its GPU subprocesses have a dedicated process group.
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait || true
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

start_worker() {
  local role="$1" gpu="$2"
  echo "GPU $gpu: $role"
  setsid bash "$REPO_ROOT/scripts/profile-qwen3-30b-ep4-parallel.sh" --worker "$role" "$gpu" \
    >"$LOG_DIR/$role.log" 2>&1 &
  PIDS+=("$!")
}

if [[ ${#GPUS[@]} == 4 ]]; then
  start_worker ordinary_tp1 "${GPUS[0]}"
  start_worker ordinary_tp2 "${GPUS[1]}"
  start_worker moe_tp1 "${GPUS[2]}"
  start_worker moe_tp2 "${GPUS[3]}"
else
  start_worker combined_tp1 "${GPUS[0]}"
  start_worker combined_tp2 "${GPUS[1]}"
fi

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAILED=1
done
PIDS=()
trap - INT TERM
if [[ "$FAILED" == 1 ]]; then
  echo "Acquisition failed; inspect $LOG_DIR and rerun the same command to resume." >&2
  exit 1
fi

# Merge only finished private bundles. Metadata locks alone cannot protect
# cross-TP attention validation and skew fitting during live acquisition.
"$PYTHON_BIN" - "$OUT_ROOT" "$HW" <<'PY'
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

from profiler.core.bundle_metadata import read_bundle, update_bundle
from profiler.core.config import load_architecture
from profiler.core.skew_calibration import rebuild_bundle
from profiler.core.writer import replicate_tp_stable

output = Path(sys.argv[1])
relative = Path(sys.argv[2]) / 'Qwen/Qwen3-30B-A3B-Instruct-2507/bf16'
destination = output / relative
sources = {role: output / '.parallel-workers' / role / relative
           for role in ('ordinary_tp1', 'ordinary_tp2', 'moe_tp1', 'moe_tp2')}
metadata = {role: read_bundle(path) for role, path in sources.items()}
if destination.exists():
    raise FileExistsError(f'Final bundle already exists: {destination}. '
                          'Keep it, or choose a new OUT_ROOT for a new acquisition.')

def tp_record(meta, tp):
    records = meta.get('per_tp', {})
    return records.get(str(tp), records.get(tp, {}))

ordinary1, ordinary2 = metadata['ordinary_tp1'], metadata['ordinary_tp2']
for key in ('model', 'hardware', 'variant'):
    if any(meta.get(key) != ordinary1.get(key) for meta in metadata.values()):
        raise ValueError(f'Worker bundles disagree on {key}')
for key in ('architecture', 'architecture_sha256', 'attention_grid',
            'gpu', 'vllm_version', 'cuda_version', 'measurement_iterations'):
    if ordinary1.get(key) != ordinary2.get(key):
        raise ValueError(f'Ordinary workers disagree on {key}')

destination.parent.mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='.bf16-parallel-', dir=destination.parent))
# TP2 supplies the shared summary, as in the serial TP1-then-TP2 pipeline.
# Preserve actual observations and identities; do not reconstruct engine facts.
shutil.copytree(sources['ordinary_tp2'], root, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.meta.lock'))
shutil.copytree(sources['ordinary_tp1'] / 'tp1', root / 'tp1')
meta = deepcopy(ordinary2)
meta['per_tp'] = {str(tp): deepcopy(tp_record(metadata[f'ordinary_tp{tp}'], tp))
                  for tp in (1, 2)}
meta['tp_degrees'] = [1, 2]
meta['engine_resolved']['per_tp'] = {
    str(tp): deepcopy(metadata[f'ordinary_tp{tp}']['engine_resolved']['per_tp'].get(
        str(tp), metadata[f'ordinary_tp{tp}']['engine_resolved']['per_tp'].get(tp)))
    for tp in (1, 2)}
for tp in (1, 2):
    source = sources[f'moe_tp{tp}'] / f'tp{tp}'
    shutil.copytree(source / 'moe_components', root / f'tp{tp}' / 'moe_components')
    meta['per_tp'][str(tp)]['moe_components'] = deepcopy(
        tp_record(metadata[f'moe_tp{tp}'], tp)['moe_components'])
update_bundle(root, lambda current: (current.clear(), current.update(meta)))
replicate_tp_stable(root, load_architecture(Path('profiler/models/qwen3.yaml')), [1, 2])
config = json.loads(Path('configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json').read_text())
rebuild_bundle(root, config, [1, 2])
meta = read_bundle(root)
for tp, dp in ((1, 4), (2, 2)):
    directory = root / f'tp{tp}'
    for name in ('dense', 'per_sequence', 'attention', 'skew', 'skew_fit'):
        if not (directory / f'{name}.csv').is_file():
            raise RuntimeError(f'Missing {directory / name}.csv')
    record = meta['per_tp'].get(str(tp), meta['per_tp'].get(tp, {}))
    entries = (record.get('moe_components') or {}).get('entries', [])
    for rank in range(4):
        matches = [entry for entry in entries if entry['target'] == dict(
            tp=tp, dp=dp, ep=4, rank=rank, sequence_parallel=(tp == 2))]
        if len(matches) != 1 or not matches[0]['coverage'].get('complete'):
            raise RuntimeError(f'Incomplete MoE TP{tp}/DP{dp}/EP4 rank {rank}')
        if not (directory / matches[0]['path']).is_file():
            raise RuntimeError(f'Missing MoE CSV for TP{tp} rank {rank}')
manifest = {role: str(path) for role, path in sources.items()}
(root / 'parallel_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
root.rename(destination)
print(f'Completed bundle: {destination}')
PY
