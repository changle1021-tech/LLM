---
title: Multi-node Hardware Profiler
---

# `profiler.hardware_multinode`

Run the hardware profiler across two nodes with `torchrun`. Both nodes must use
the same launch arguments. `--output-dir` may name a shared directory or
separate local directories, but it must be absent on every node before launch.
Replace `fresh_dir` and the RDMA reference path below with the paths for this
run.

On node 0, launch:

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 \
  --master-addr=172.16.30.101 --master-port=19596 \
  -m profiler.hardware_multinode \
  --hardware H100_2x2 --npus 4 --repeats 3 \
  --rdma-reference /path/to/rdma_mapping.json --output-dir fresh_dir
```

On node 1, use the same command and change `--node-rank=0` to
`--node-rank=1`.

All ranks first use Gloo to confirm the run and that `--output-dir` is absent
on every node; only after that preflight do they create it. Each rank writes
`rank_NNN_start.json`, and each node-local rank 0 writes its before/after NIC
snapshots using that rank’s global ID. Global rank 0 alone writes the measured
repeat files, aggregate samples, provenance, final hardware fits, and completion
marker. `--rdma-reference` is optional; when supplied, it checks the current
`NCCL_IB_HCA`, `UCX_NET_DEVICES`, and mounted uverbs devices against the RDMA
mapping recorded by the vLLM launcher, then records the matched mapping. For
the matched experiment described here, pass that run’s `rdma_mapping.json` on
both nodes. A mismatch
refuses the measurement. For example, a reference mapping that selects
`mlx5_2:1` and `mlx5_3:1` measures those selected resources; a reference
containing all NICs likewise measures that full set. The tool does not expand a
reference mapping to all machine NICs.

The global-rank-0 output directory contains the aggregate samples and run
provenance, followed by the per-repeat results:

```text
fresh_dir/
├── hardware.yaml
├── raw_samples.json
├── provenance.json
├── repeat_001/
│   └── hardware.yaml
└── ...
```

`COMPLETED.json` is written last when the run completes. Failures after output
creation leave `ABORTED_rank_NNN.json`; global rank 0 also writes `ABORTED.json`.
A rejected preflight leaves the output directory uncreated.

The profiler measures hardware operations; it does not run a model or report
end-to-end serving timings. Its fitter is the same as the local profiler in
`profiler/core/hardware.py`: it jointly fits common bandwidth and latency, then
fits each operation's bandwidth with that common latency fixed. Effective group parameters describe
the measured group behavior and are not physical NIC rates. The original
rank-zero CUDA-graph and isolated timing protocol is preserved. The run does
not split results into intra-node and inter-node groups.

## Optional operation-specific effective links

The acquisition keeps the original shared-latency rule. An explicit offline
alternative fits both bandwidth and latency independently for each collective:

```bash
python -m profiler.hardware_multinode_links \
  --hardware-yaml fresh_dir/hardware.yaml --output-dir fresh_links
```

This reads only the completed primitive measurements, uses the original Ring
relative-least-squares fitter on every recorded CUDA-graph sample, and saves
the input/source hashes and residuals in `fresh_links/fit.json`.
`fresh_links/cluster_links.json` contains the simulator cluster fields
`link_bw`, `link_latency`, and `collective_links`. Merge those fields into a
copy of the deployment config; retain the original model, memory, and scheduler
settings. This alternative has one latency per operation and is a different
model from the acquisition's shared-latency default. Neither model establishes
end-to-end accuracy until separately validated against a completed benchmark.
