---
sidebar_position: 6
title: Multi-node Bench CLI
---

# `python -m bench_multinode` CLI

`bench_multinode` runs the flat-request vLLM replay on a Ray cluster and validates recorded results against simulator output. It is a separate entry point from `python -m bench`. Run it on the Ray head node with `./bench_multinode/bench_multinode.sh`; an optional `./bench_multinode/docker_multinode.sh` starts the same vLLM 0.31 environment with host networking. `VLLM_GPUS` selects the visible GPUs in that container.

Before starting a run, place identical repository code at the same container path (`/workspace`) on every node, and make model weights available at the same path on all nodes. The dataset and output directory are read and written on the head node only. Start the head and workers with `./bench_multinode/ray_multinode.sh`; then manually place the TP, PP, and DP ranks across the nodes. The harness does not verify physical placement or automatically assign ranks to hosts.

Ray is optional in upstream vLLM. In a bare-metal environment, install `ray[cgraph]` alongside vLLM and Rich; validation additionally needs Matplotlib. Use the same Python, vLLM, Ray and communication-library versions on every node. The optional container launcher installs `ray[cgraph]`; `RAY_PACKAGE` can select an identical pinned requirement on every host. It also retains the original launcher's dependencies and installed-vLLM patches, and mounts this repository at `/workspace`. Use `VLLM_EXTRA_MOUNTS` for space-separated Docker volume specifications such as `/models:/models`, and `VLLM_CONTAINER_NAME` for the container name. Host networking targets Linux Docker hosts.

For automation, `VLLM_DETACHED=1` runs Docker detached instead of opening an interactive shell. After dependencies, patches, and RDMA setup succeed, the container touches `/tmp/llmsim-container-ready` and stays alive; the launcher itself does not wait for readiness. Follow initialization output with `docker logs -f "${VLLM_CONTAINER_NAME:-vllm_multinode}"`. The default `VLLM_DETACHED=0` preserves interactive behavior; other values are rejected.

The container launcher defaults to `VLLM_RDMA=1` and selects RDMA devices automatically for the GPUs named by `VLLM_GPUS`. It reads GPU PCI addresses and `nvidia-smi topo -m`, then uses active physical RDMA ports and their PCI/NUMA paths. When the GPU-to-HCA distances are complete in `nvidia-smi topo -m`, it uses those; otherwise it falls back to sysfs PCI ancestry and NUMA locality. It chooses the closest active HCA port for each selected GPU, includes all ties, and mounts only the union of the needed `/dev/infiniband/uverbsN` devices plus `rdma_cm` when present. It sets exact `NCCL_IB_HCA` and `UCX_NET_DEVICES` values and adds `IPC_LOCK` with unlimited memlock. `VLLM_RDMA=0` disables RDMA device mapping and container device mounts.

Automatic selection accepts `VLLM_GPUS=all` or an explicit Docker device list using physical GPU indices or unique GPU UUIDs, for example `VLLM_GPUS='"device=0,1"'`. A numeric GPU count and MIG identifiers are refused because their physical GPU affinity is ambiguous. No active accessible RDMA port stops the launcher before Docker starts. A GPU without measurable PCI/NUMA affinity is left unbound and NCCL chooses among all active ports. An explicit host `NCCL_IB_HCA` or `UCX_NET_DEVICES` takes precedence over the automatic value, with a notice. `NODE_IP`, socket interface settings, and GID selection remain operator-controlled. Each container also configures vLLM's Ray environment exclusion list to keep node-local HCA, UCX, socket, and GID values from being overwritten by the head node's driver settings. The GPUs selected in each container must match the GPUs intended for Ray on that node; rank placement remains user-managed.

Preview the mapping without starting Docker:

```bash
python3 bench_multinode/rdma_multinode.py --gpus '"device=0,1"'
```

The JSON output lists the selected GPUs, nearest active ports, mapping method, mounted devices, and exact NCCL/UCX environment values. The helper can also emit Docker arguments with `--format docker-args`.

## H100 TP4 startup troubleshooting

In the observed H100, vLLM 0.31, two-node TP4 startup, rank 0 stalled during symmetric-memory rendezvous while another rank progressed to a later FlashInfer broadcast. For this experiment, explicitly set `VLLM_ALLREDUCE_USE_SYMM_MEM=0` on both nodes before starting the containers; the multi-node Docker launcher forwards it when set. `--disable-custom-all-reduce` alone does not disable this symmetric-memory path. This is an experiment-specific workaround, not a launcher default or a general recommendation. The launcher also passes through `VLLM_ALLREDUCE_USE_FLASHINFER` and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` only when provided, and does not select or force a collective transport.

```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  VLLM_GPUS='"device=2,3"' VLLM_EXTRA_MOUNTS='/models:/models' \
  ./bench_multinode/docker_multinode.sh
```

## Two-node setup example

The host GPU indices are resolved independently on each node. For example, the head can expose GPUs `0,1` while the worker exposes GPUs `2,3`; make the same model directory available at `/models` on both hosts. Use the matching `VLLM_GPUS` value both when starting each container and when reviewing the RDMA preview.

On the head host:

```bash
VLLM_GPUS='"device=0,1"' VLLM_EXTRA_MOUNTS='/models:/models' \
  ./bench_multinode/docker_multinode.sh
```

Inside the head container, start Ray:

```bash
NODE_IP=10.0.0.1 GPUS_PER_NODE=2 ./bench_multinode/ray_multinode.sh head
```

On the worker host, start a container with its own GPU selection and the same model mount:

```bash
VLLM_GPUS='"device=2,3"' VLLM_EXTRA_MOUNTS='/models:/models' \
  ./bench_multinode/docker_multinode.sh
```

Inside that worker container:

```bash
NODE_IP=10.0.0.2 RAY_HEAD_ADDRESS=10.0.0.1:6379 GPUS_PER_NODE=2 \
  ./bench_multinode/ray_multinode.sh worker
```

The host's `NODE_IP` must be reachable by the other node. GPU numbering in the worker's list is local to that host; automatic HCA selection uses the selected GPUs' local PCI topology.

## Ray helper

The helper accepts `head`, `worker`, and `status`:

```bash
NODE_IP=10.0.0.1 ./bench_multinode/ray_multinode.sh head
NODE_IP=10.0.0.2 RAY_HEAD_ADDRESS=10.0.0.1:6379 ./bench_multinode/ray_multinode.sh worker
RAY_HEAD_ADDRESS=10.0.0.1:6379 ./bench_multinode/ray_multinode.sh status
```

`NODE_IP` is required for `head` and `worker`. `RAY_HEAD_ADDRESS` selects the head for `worker` or a remote `status` command. `GPUS_PER_NODE` is optional; Ray otherwise detects GPUs. `RAY_PORT` defaults to `6379`. Both head and worker use `ray start --block`, so keep those terminals open and run the benchmark in another terminal in the head environment. Native Ray arguments can follow the helper command. GPU selection uses the container's `VLLM_GPUS`, or `CUDA_VISIBLE_DEVICES` before starting Ray in a bare-metal environment.

## Parallel placement examples

These examples launch a vLLM workload with four GPUs per node. They assume the Ray cluster is already up and `NODE_IP` / `RAY_HEAD_ADDRESS` resolve between nodes.

TP=4, PP=2, DP=1 across two 4-GPU nodes:

```bash
TP=4 PP=2 DP=1 ./bench_multinode/bench_multinode.sh run \
  --model /models/my-model --dataset /data/requests.jsonl \
  --output-dir /results/tp4-pp2
```

DP=2, TP=4, PP=1 across two 4-GPU nodes. `DP_LOCAL=1` means one DP rank is local to the launch/control node; it is not the global DP count or the worker count:

```bash
TP=4 PP=1 DP=2 DP_LOCAL=1 ./bench_multinode/bench_multinode.sh run \
  --model /models/my-model --dataset /data/requests.jsonl \
  --output-dir /results/dp2-tp4
```

DP=2, TP=4, PP=2 across four 4-GPU nodes, using the span packing strategy. Leave `DP_LOCAL` unset so the frontend uses its default:

```bash
TP=4 PP=2 DP=2 RAY_DP_PACK_STRATEGY=span \
  ./bench_multinode/bench_multinode.sh run \
  --model /models/my-model --dataset /data/requests.jsonl \
  --output-dir /results/dp2-tp4-pp2
```

The shell wrapper defaults `TP`, `PP`, and `DP` to `1`; `DP_LOCAL` and `RAY_DP_PACK_STRATEGY` are optional. `RAY_ADDRESS` defaults to `auto`. `RAY_DP_PACK_STRATEGY` can select `strict`, `fill`, or `span`; it passes the chosen native strategy through and does not choose a strategy automatically. The CLI has matching parallelism options: `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, optional `--data-parallel-size-local`, `--distributed-executor-backend ray`, `--data-parallel-backend ray`, `--ray-address`, and optional `--ray-dp-pack-strategy`.

## Run options

Use `python -m bench_multinode run --help` as the canonical, current option list. The run command accepts the standard replay, scheduler, precision, workload, and output options from the single-node bench, except `--record-gate-stats` is not supported. Multi-node options include:

| Option | Meaning |
| --- | --- |
| `--tensor-parallel-size` | Tensor-parallel size (default `1`) |
| `--pipeline-parallel-size` | Pipeline-parallel size (default `1`) |
| `--data-parallel-size` | Data-parallel size (default `1`) |
| `--data-parallel-size-local` | Optional count of DP ranks local to the launch/control node |
| `--distributed-executor-backend` | Ray executor backend |
| `--data-parallel-backend` | Defaults to `mp` for DP=1 and `ray` for DP>1; DP engines on other nodes require `ray` |
| `--ray-address` | Ray address; defaults to `RAY_ADDRESS`, or `auto` when unset |
| `--ray-dp-pack-strategy` | Optional native `strict`, `fill`, or `span` packing strategy |
| `--run-timeout-seconds` | Replay timeout after engine startup; default `0` disables it |
| `--kernel-config` | Native vLLM kernel settings as a JSON object; explicit settings are recorded in `engine_kwargs` |
| `--all2all-backend` | Explicit native expert transport, such as `allgather_reducescatter`; omission retains the vLLM default |

The shell wrapper also exposes `RAY_ADDRESS` (default `auto`) and `RAY_DP_PACK_STRATEGY` (unset by default). For the regular replay flags and their exact defaults, consult `python -m bench_multinode run --help` rather than assuming the single-node runner's interface is identical.

`DATA_PARALLEL_BACKEND` optionally overrides engine launch mode. A single DP replica uses a local engine core and Ray GPU workers by default; this allows TP/PP to span nodes without Ray DP packing constraints. DP>1 uses Ray for engine cores as well. When using this Ray DP backend and a single replica spans nodes, explicitly choose `span`, including when opting into Ray DP at DP=1. The harness does not infer this from the GPU count. `DP_LOCAL` must be unset or `1` with `span` in the reviewed upstream path. Engine startup remains governed by vLLM's native `VLLM_ENGINE_READY_TIMEOUT_S` environment setting; the replay timeout does not interrupt its synchronous constructor.

The new entry point uses `spawn` for vLLM engine subprocesses because it connects to Ray before constructing the engine. The effective process method is recorded in `meta.json.distributed`; this does not change GPU placement or collective backend selection. Exiting the benchmark disconnects its Ray client and shuts down its engines, while leaving the independently started Ray cluster running.

## Validate

`python -m bench_multinode validate --help` lists the validator options. Validation reads the recorded run, simulator request CSV, and simulator log on CPU; it does not start vLLM. The wrapper provides the same command as `./bench_multinode/bench_multinode.sh validate ...`.

## Recorded output and limits

Choose a fresh output path: an existing output directory is refused. The run writes `run_start.json` before connecting, `engine_start.json` when the engine starts, per-request `requests.jsonl`, per-tick `timeseries.csv`, and `meta.json` last as the completion marker. If a run fails, it writes `ABORTED.json`. Requests record the `data_parallel_rank` mapped from native frontend finished-request statistics. The validator estimates the epoch-to-monotonic clock offset separately for each DP engine; request pickup residuals can bias that estimate, so timestamps do not imply exact cross-node synchronization. Use this entry point's copied validator for multi-node DP output; the original validator fits one global offset.

Frontend-observed timestamps and metrics are recorded separately from engine-native metrics; one is not silently substituted for the other. The run may also capture read-only native graph execution observations. Those observations do not audit physical GPU placement. Gate statistics are not supported.

The independent entry point copies the original runner, recorder, logger, stat logger, startup capacity observer, graph observer, validator and plotting helpers under `bench_multinode/core/*_multinode.py`. It does not import `bench.*`. The graph observer continues to use shared CPU-only helpers from `profiler.core.execution_metadata`. Existing benchmark files and reference runs are unchanged.

Current verification uses CPU, recorded-reference validation and mocked engine/launcher checks. No real multi-node GPU run has validated this entry point yet.
