# Multi-node bench

`bench_multinode` is an independent multi-node vLLM replay entry point. Run `python -m bench_multinode run` on the Ray head node and compare a completed run with `python -m bench_multinode validate`. See the [multi-node CLI guide](../docs/docs/reference/bench-multinode-cli.md) for setup, placement examples, options, and output details.

## Layout

```text
bench_multinode/
├── __main__.py                 # python -m bench_multinode {run,validate}
├── bench_multinode.sh          # host wrapper for run / validate
├── docker_multinode.sh         # optional vLLM 0.31 container with host networking
├── ray_multinode.sh            # head / worker / status Ray helper
├── rdma_multinode.py           # GPU-to-RDMA affinity and local NIC setup
└── core/                       # independent _multinode.py runner and helpers
```


Set `VLLM_DETACHED=1` to start the Docker container in the background. The launcher performs setup, then creates `/tmp/llmsim-container-ready` inside the container and keeps it alive; view setup output with `docker logs -f ${VLLM_CONTAINER_NAME:-vllm_multinode}`. The default remains an interactive container.
