#!/bin/bash

# Launch the simulator Docker container (ASTRA-Sim + sim Python deps).
#
# Mounts the repo root regardless of where this script is invoked from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

usage() {
  cat <<'EOF'
Usage: scripts/docker-sim.sh [--help]

Environment:
  SIM_CONTAINER_NAME  Docker container name (default: servingsim_docker)
  SIM_DETACHED        0 for interactive mode (default), 1 for detached mode
  SIM_MEMORY          Optional Docker memory limit, also used as memory-swap
  SIM_CPUS            Optional Docker CPU limit
  SIM_IMAGE           Docker image (default: astrasim/tutorial-micro2024)
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

SIM_CONTAINER_NAME="${SIM_CONTAINER_NAME:-servingsim_docker}"
SIM_DETACHED="${SIM_DETACHED:-0}"
SIM_IMAGE="${SIM_IMAGE:-astrasim/tutorial-micro2024}"
if [[ "$SIM_DETACHED" != 0 && "$SIM_DETACHED" != 1 ]]; then
  echo 'SIM_DETACHED must be 0 or 1' >&2
  exit 2
fi
if [[ -z "$SIM_CONTAINER_NAME" || -z "$SIM_IMAGE" ]]; then
  echo 'SIM_CONTAINER_NAME and SIM_IMAGE must be non-empty' >&2
  exit 2
fi

# The image ships Python 3.10 and the ASTRA-Sim build deps, but none of the
# packages the Python side needs. Every one below is imported by code that runs
# in this container:
#   pyyaml        profiler meta.yaml and the architecture catalogs
#   pyinstrument  serving/__main__.py
#   rich          serving/ and bench/ loggers
#   pandas        scheduler, trace_generator, pim_model
#   numpy         scheduler
#   scipy         new native MoE physical-coordinate interpolation
#   matplotlib    bench/core/plots.py, reached by `python -m bench validate`
docker_args=(run --name "$SIM_CONTAINER_NAME")
if [[ "$SIM_DETACHED" == 1 ]]; then
  docker_args+=(-d)
else
  docker_args+=(-it)
fi
if [[ -n "${SIM_MEMORY:-}" ]]; then
  docker_args+=(--memory "$SIM_MEMORY" --memory-swap "$SIM_MEMORY")
fi
if [[ -n "${SIM_CPUS:-}" ]]; then
  docker_args+=(--cpus "$SIM_CPUS")
fi
docker_args+=(-v "$REPO_ROOT:/app/LLMServingSim" -w /app/LLMServingSim
  "$SIM_IMAGE" bash -c)
if [[ "$SIM_DETACHED" == 1 ]]; then
  docker_args+=('pip3 install pyyaml pyinstrument rich pandas==1.5.3 numpy==1.23.5 scipy==1.10.1 matplotlib==3.5.3 && touch /tmp/llmsim-sim-ready && exec sleep infinity')
else
  docker_args+=('pip3 install pyyaml pyinstrument rich pandas==1.5.3 numpy==1.23.5 scipy==1.10.1 matplotlib==3.5.3 && exec bash')
fi
exec docker "${docker_args[@]}"
