#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/compile.sh [--backend analytical|ns3|all] [--jobs N] [--skip-chakra-install]

Options:
  --backend  Backend to compile (default: analytical)
  --jobs     Positive build parallelism, exported as ASTRA_BUILD_JOBS (default: 4)
  --skip-chakra-install  Use an already installed converter (for unprivileged builds)
  -h, --help Show this help
EOF
}

backend=analytical
jobs=4
skip_chakra=0
while (($#)); do
  case "$1" in
    --backend)
      (($# >= 2)) || { echo '--backend requires a value' >&2; usage >&2; exit 2; }
      backend="$2"
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || { echo '--jobs requires a value' >&2; usage >&2; exit 2; }
      jobs="$2"
      shift 2
      ;;
    --skip-chakra-install)
      skip_chakra=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$backend" in
  analytical|ns3|all) ;;
  *) echo "Invalid backend: $backend (choose analytical, ns3, or all)" >&2; exit 2 ;;
esac
if [[ ! "$jobs" =~ ^[0-9]+$ ]] || ((10#$jobs < 1)); then
  echo "--jobs must be a positive integer: $jobs" >&2
  exit 2
fi

# A container can install Python packages as root, then compile as its owner.
if [[ "$skip_chakra" == 1 ]]; then
  python3 -c 'from chakra.src.converter.llm_converter import LLMConverter'
else
  (
    cd "$REPO_ROOT/astra-sim/extern/graph_frontend/chakra"
    pip3 install .
  )
fi

if [[ "$backend" == analytical || "$backend" == all ]]; then
  (
    cd "$REPO_ROOT/astra-sim"
    ASTRA_BUILD_JOBS="$jobs" bash ./build/astra_analytical/build.sh
  )
fi

if [[ "$backend" == ns3 || "$backend" == all ]]; then
  (
    cd "$REPO_ROOT/astra-sim"
    ASTRA_BUILD_JOBS="$jobs" bash ./build/astra_ns3/build.sh --compile
  )
fi
