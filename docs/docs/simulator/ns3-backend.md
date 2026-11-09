---
sidebar_position: 9
title: ns-3 packet-network backend
---

# ns-3 packet-network backend

The `ns3` network backend runs ASTRA-Sim's packet-level ns-3 network model. It is selected with `--network-backend ns3`; the default `analytical` backend remains the fast option. The ns-3 executable is built separately:

```bash
scripts/compile.sh --backend ns3 --jobs 4
```

The command builds `astra-sim/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-optimized`. `--jobs` must be a positive integer. The compile script installs the checked-out Chakra converter unless `--skip-chakra-install` is supplied. One validated Docker setup installed Chakra beforehand as root with `pip install --no-deps`, then compiled as UID 1001 using `--skip-chakra-install`; protobuf 7.36.2 was present in that environment. This is a tested privilege arrangement, not a requirement that every environment use that protobuf version or install Chakra as root.

Run a case with a unique `--run-id` and output path:

```bash
python3 -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --network-backend ns3 --run-id ns3_smoke \
  --keep-inputs --output outputs/ns3_smoke.csv
```

The logical ns-3 dimensions are taken dynamically from the generated ASTRA-Sim `network.yml` (`npus_count`) and must multiply to the cluster's total NPU count. The backend does not assume a fixed rank count. With no `ns3` object in the cluster config, it creates a full physical mesh using uniform common link properties.

For explicit physical topologies, add a top-level `ns3` object. NPU endpoint IDs are `0` through `total_npu - 1`; optional switch IDs must be consecutive, starting at `total_npu`. Explicit links use endpoint IDs and decimal Gbit/s:

```json
{
  "ns3": {
    "switches": [4],
    "links": [
      {"src": 0, "dst": 4, "bandwidth_gbps": 100, "latency_ns": 500},
      {"src": 1, "dst": 4, "bandwidth_gbps": 100, "latency_ns": 500},
      {"src": 2, "dst": 4, "bandwidth_gbps": 100, "latency_ns": 500},
      {"src": 3, "dst": 4, "bandwidth_gbps": 100, "latency_ns": 500}
    ],
    "packet_payload_size": 1000
  }
}
```

Common cluster `link_bw` is interpreted as GiB/s and converted for ns-3; common `link_latency` is in ns. Common list-valued bandwidth or latency must be uniform. Use explicit `ns3.links` for heterogeneous physical links. Explicit link rates (`bandwidth_gbps`) use decimal Gbit/s, and link latency uses ns. NPU endpoints are not transit switches, so an explicit switched topology must connect every NPU pair through the switch nodes. `packet_payload_size` must be an integer from 64 through 65500. Optional `ns3.config` overrides keys supported by the checked-in ns-3 template or the frontend monitor settings in `serving/core/ns3_config.py`; topology and generated output paths cannot be overridden. Set packet payload through `ns3.packet_payload_size`, rather than through `ns3.config`.

`ns3.host_dispatch_latency_ns` optionally adds a measured host handoff delay before each model forward (nonnegative integer nanoseconds; default `0`). It represents time between GPU iterations spent on host scheduling, dispatch and result handling; it does not change GPU component latencies or network propagation. TP start/completion controllers sharing the same forward use one dispatch deadline. Use an independent native runtime measurement for the intended executor and workload regime; a delay measured for synchronous Ray TP4 is not a general hardware constant. This option is separate from `async_scheduling`.

Each run writes generated topology, configuration, manifest, and backend output files below that run's `astra-sim/inputs/runs/<run-id>/ns3/` directory. Use distinct run IDs to keep these artifacts isolated. `--keep-inputs` retains the generated ASTRA-Sim inputs and workloads after a successful simulation; without it, normal cleanup applies.

## Instance completion

The ns-3 serving driver uses the same completion barrier as the analytical
backends. An instance's tail rank reports iteration completion only after every
rank in that instance has completed that iteration. This includes rank-local
work after the final collective. A managed rank already executing a later
iteration counts as complete for the target, preserving PP lookahead. Other
instances continue independently; the barrier does not require global idleness.
Rebuild with `scripts/compile.sh --backend ns3` after updating the driver.
See [parallelism mechanics](parallelism-mechanics.md#instance-completion-across-ranks)
for the shared frontend/backend contract.

## Network monitoring

Network monitoring records port throughput and configured RDMA QP rates during
ordinary ns-3 serving runs. It is enabled independently of `ENABLE_TRACE` and
does not need PyViz, packet-log conversion or `--save-trace-text`. Rebuild the
ns-3 backend after updating this feature, then retain a run with `--keep-inputs`:

```bash
python3 -m serving \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --dataset workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --network-backend ns3 --run-id network_inspect --keep-inputs \
  --output outputs/network_inspect.csv

python3 scripts/ns3-network-plots.py \
  astra-sim/inputs/runs/network_inspect/ns3/output
```

Open the generated `plots/index.html` in a browser. The plotting command uses
the Python standard library and produces self-contained interactive HTML;
network access and additional plotting packages are unnecessary. Node pages
show port transmit/receive rates and utilization relative to each physical
link's configured capacity. Node-pair pages show the sum of active QPs'
configured rates and their active count. Display samples are bounded while
the raw CSV records remain unchanged. Use `--help` for output-directory and
display-resolution options.

An example cluster fragment explicitly sets the sampling intervals:

```json
{
  "ns3": {
    "config": {
      "ENABLE_NETWORK_MONITOR": 1,
      "BW_MON_INTERVAL_NS": 1000000,
      "QP_MON_INTERVAL_NS": 10000
    }
  }
}
```

Intervals are positive integer nanoseconds. Set `ENABLE_NETWORK_MONITOR` to
`0` to disable this output. The generated `ns3/config.txt` records the effective
settings; `serving/core/ns3_config.py` defines their defaults and owns output
paths, which cannot be supplied through `ns3.config`.

### Output schema

The monitor writes these records under the retained run's `ns3/output/`:

| File | Contents |
| --- | --- |
| `port_bandwidth.csv` | `time_ns`, `node_id`, `port_id`, `peer_node_id`, `link_bandwidth_bps`, `interval_ns`, `tx_bytes`, `rx_bytes`, `tx_gbps`, `rx_gbps` |
| `qp_rate.csv` | `time_ns`, `src_node_id`, `dst_node_id`, `src_port`, `dst_port`, `size_bytes`, `rate_bps` |
| `network_stats.json` | Sampling settings, output row counts, QP lifecycle counts and cumulative port-byte totals |

A port row describes the interval ending at `time_ns`; its byte counts are
interval deltas. Every physical port has an initial zero baseline, followed
by fixed windows, including idle windows, and a final partial window when
needed. `peer_node_id` maps the port to its adjacent node; parallel physical
links retain separate port identities. Bytes include packet headers and
control traffic. Transmission is counted at PHY start and reception at
successful MAC delivery. Charging a whole packet at the boundary can make a
short window's rate exceed physical capacity; these counters measure packet
throughput, not application goodput or a reconstructed bit-level waveform.

QP rows capture the first observed transmission, sampled changes in the
configured sending rate, and an actual-completion marker with `rate_bps=0`.
This rate is congestion-control state, not achieved throughput. Summing QP
rates for a node pair does not measure the capacity of its multi-hop path.

Sampling observes existing network events and serving steps rather than
adding simulator events. The monitor flushes a final partial window at
shutdown. It does not extend the legacy `qlen.txt` observation window or
manufacture PFC events: queue output can be empty when no qualifying queue
sample occurs, and `pfc.txt` can be empty when no PFC event occurs.

The periodic byte-counter sampling design follows
[SimAI/ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud), with a
frontend implementation that uses the current serving event loop.

## Collective probes

Analytical `collective_links` timing overrides do not have a packet-level equivalent and are rejected for ns-3 runs. To sample primitive collective behavior independently, use the probe script:

```bash
python3 scripts/ns3-collective-probe.py \
  --cluster-config configs/cluster/single_node_single_instance.json \
  --output-dir outputs/ns3_collective_probe \
  --sizes 10240 40960 1048576 \
  --payload 4096
```

`--cluster-config` and `--output-dir` are required; the output directory must not already exist. `--sizes` accepts one or more positive per-operation sample sizes in bytes (default: `10240 40960 139264 327680 1048576 4194304`). `--bandwidth` and `--latency` optionally override the cluster's common link bandwidth (GiB/s) and latency (ns); `--payload` selects the packet payload size (default 4096). The probe generates all-reduce, all-gather, and reduce-scatter traces and records samples and backend logs in its output directory.

The probe preserves explicit `ns3.links`, switches and backend overrides. For an explicit topology, edit its link entries instead of passing the common `--bandwidth` or `--latency` flags. Primitive measurements exclude serving host handoff time: the probe sets `host_dispatch_latency_ns` to zero even if the serving configuration supplies a nonzero value.

The integration addresses build and protocol issues found during bring-up: warning macros no longer override the optimized build with `-O0`, `PreRun` clears marked times, queue monitoring stops at the end condition, and the serving protocol handles start/end messages and idle deadlines. Real runtime checks cover a switched topology, logical dimensions `[2, 2]`, unchanged default-zero dispatch behavior, and charging a measured dispatch delay once per TP4 forward.

On the tested two-host H100/vLLM 0.31 setup, full 300-request comparisons used synchronous scheduling, 128 maximum sequences, a 2048-token batch budget, and 16 GiB KV reservation per GPU. Llama-3.1-8B TP4 had all 15 reported latency statistics within 10% after fresh profile acquisition and an independently GPU-traced 2.349 ms host dispatch measurement. Qwen3-30B-A3B TP1 with DP=EP2 and DP=EP4 had 14/15 and 12/15 statistics within 10%, respectively, using native collective calibration and zero added host delay. EP4 TTFT P99 remained 31.1% high (24.1% against a repeated native reference); these MoE tails remain a limitation. Counts use the project's rounded validation summary.

Communication parameters describe an effective full-mesh collective model fitted from independent primitive measurements, rather than physical single-NIC bandwidth. They were not fitted to serving latency. The measured host delay is specific to the synchronous Ray TP4 executor and profiling regime. These results do not establish accuracy for other topologies, executors, or asynchronous scheduling. Recorded configurations, complete summaries, parameter controls, and reproduction commands are in `outputs/ns3_integration_20261024/REPORT.md` in the validation checkout.
