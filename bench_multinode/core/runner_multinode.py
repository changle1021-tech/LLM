"""vLLM benchmark runner — strict replay of an existing dataset.

The runner reads a LLMServingSim-format JSONL workload (the same format
``python -m workloads.generators sharegpt`` produces and ``python -m serving
--dataset`` consumes) and replays every request through vLLM with its
``input_tok_ids`` and ``output_toks`` pinned to the simulator workload.
Generated content and execution timing need not be identical.

Independent copy of the benchmark runner for the vLLM 0.31 native interfaces:
  * ``vllm.v1.engine.async_llm.AsyncLLM`` — async engine, ``generate()``
    yields ``RequestOutput`` per chunk, ``RequestOutput.metrics`` carries
    per-request ``RequestStateStats`` (arrival_time / queued_ts /
    scheduled_ts / first_token_ts / last_token_ts).
  * ``vllm.v1.metrics.loggers.StatLoggerBase`` — pluggable per-engine stat
    logger; we hook it via ``BenchStatLogger`` to capture per-iteration
    scheduler/iteration stats for ``timeseries.csv``.

Output: ``<output-dir>/{meta.json, requests.jsonl, timeseries.csv}``.
The dataset itself is not modified — generation lives in
``workloads/generators``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

from bench_multinode.core import recorder_multinode as recorder
from bench_multinode.core.kv_cache_metadata_multinode import ReadyResponseCapture
from bench_multinode.core.execution_metadata_multinode import capture_execution


# Observe caller overrides without changing vLLM's communication selection.
_ALL_REDUCE_ENV = (
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def _json_object(value: str) -> dict:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError('must be a JSON object')
    return result


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id passed verbatim to vllm.AsyncLLM.")
    p.add_argument("--dataset", required=True,
                   help="Path to flat LLMServingSim-format JSONL requests with "
                        "nonempty input_tok_ids. Session rows containing "
                        "sub_requests are skipped; use a flat-only workload "
                        "for a matched simulator comparison.")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory for this run "
                        "(meta.json/requests.jsonl/timeseries.csv).")
    p.add_argument("--tensor-parallel-size", type=_positive_int, default=1,
                   dest="tensor_parallel_size",
                   help="vLLM tensor_parallel_size.")
    p.add_argument("--pipeline-parallel-size", type=_positive_int, default=1,
                   help="vLLM pipeline_parallel_size across the selected GPUs.")
    p.add_argument("--distributed-executor-backend", choices=["ray"], default="ray",
                   help="Worker executor for this independent Ray entry point.")
    p.add_argument("--data-parallel-backend", choices=["ray", "mp"], default=None,
                   help="Default: mp for DP=1 (local engine, Ray GPU workers), "
                        "ray for DP>1 (Ray launches every DP engine). "
                        "DP engines on other nodes require ray.")
    p.add_argument("--data-parallel-size-local", type=_positive_int, default=None,
                   help="DP engines assigned to the launch node; leave unset for "
                        "vLLM's strategy default. This is not the global DP size.")
    p.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"),
                   help="Existing native Ray cluster address, e.g. 10.0.0.1:6379. "
                        "Default: RAY_ADDRESS or auto. Never starts a local cluster.")
    p.add_argument("--ray-dp-pack-strategy", choices=["strict", "fill", "span"],
                   default=None,
                   help="Optional VLLM_RAY_DP_PACK_STRATEGY override. Use span when "
                        "one DP replica spans nodes. No placement is inferred.")
    p.add_argument("--run-timeout-seconds", type=_nonnegative_float, default=0.0,
                   help="Replay timeout after engine startup; 0 disables it. "
                        "Use vLLM's native startup timeout for engine initialization.")
    p.add_argument("--data-parallel-size", type=_positive_int, default=1,
                   dest="data_parallel_size",
                   help="vLLM data_parallel_size (DP across engines).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   dest="enable_expert_parallel", default=False,
                   help="vLLM enable_expert_parallel for MoE models.")
    p.add_argument("--all2all-backend", default=None,
                   help="Explicit native expert transport; omitted uses vLLM's default.")
    p.add_argument("--kernel-config", type=_json_object, default=None,
                   help="Native vLLM KernelConfig as JSON, recorded with the run.")
    p.add_argument("--disable-custom-all-reduce", action="store_true",
                   default=False,
                   help="Explicitly disable vLLM custom all-reduce. Off by default; "
                        "other communication backends retain their vLLM defaults. "
                        "The benchmark keeps its two communication fusion passes "
                        "disabled. This flag alone does not enforce NCCL-only.")
    p.add_argument("--max-num-seqs", type=_positive_int, default=128,
                   dest="max_num_seqs",
                   help="vLLM scheduler max_num_seqs (per-engine running cap).")
    p.add_argument("--max-num-batched-tokens", type=_positive_int, default=2048,
                   dest="max_num_batched_tokens",
                   help="vLLM scheduler max_num_batched_tokens.")
    p.add_argument("--max-model-len", type=_positive_int, default=None,
                   dest="max_model_len",
                   help="vLLM max_model_len (None = model's max).")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   dest="kv_cache_dtype",
                   help="vLLM kv_cache_dtype.")
    p.add_argument("--kv-cache-memory-bytes", type=_positive_int, default=None,
                   help="Explicit per-GPU KV cache budget. Use for capacity-matched "
                        "controls; verify kv_cache.capacity.per_engine in meta.json. "
                        "None keeps vLLM's automatic memory profiling.")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for vLLM.")
    p.add_argument("--resolve-only", action="store_true", dest="resolve_only",
                   default=False,
                   help="Boot the engine, write meta.json, and exit without "
                        "replaying the dataset. Compare each engine's observed "
                        "kv_cache.capacity.per_engine block count; the raw "
                        "kv_cache.num_gpu_blocks can sum multiple DP engines. "
                        "vLLM settles capacity at boot after subtracting "
                        "the activation peak and CUDA context. Cheap with "
                        "--load-format dummy (one boot, no weights, no "
                        "replay), so a config can be checked against the "
                        "simulator's block count. Unsupported observations are "
                        "marked unknown. requests.jsonl "
                        "and timeseries.csv are not written -- there is no "
                        "run to record.")
    p.add_argument("--skip-tokenizer-init", action="store_true",
                   dest="skip_tokenizer_init", default=False,
                   help="Boot vLLM without loading a tokenizer. The replay "
                        "never needs one -- it feeds token ids directly "
                        "(TokensPrompt) and pins the output length "
                        "(min_tokens == max_tokens, ignore_eos), so nothing "
                        "here tokenises input or reads generated text. What it "
                        "buys is the ability to bench a checkpoint whose "
                        "tokenizer is not on disk. For a config-only diagnostic, "
                        "create a separate directory containing config.json, "
                        "pass it as --model and use --load-format dummy. Bench "
                        "does not materialize the repository's named JSON files "
                        "into that layout. Real-weight runs still need weights.")
    p.add_argument("--enforce-eager", action="store_true",
                   dest="enforce_eager", default=False,
                   help="Run vLLM eager, with torch.compile and cudagraphs "
                        "off. Leave it off for a compiled serving reference. "
                        "An eager reference helps isolate execution-mode effects; "
                        "ordinary layerwise acquisition preserves module "
                        "boundaries. Record and match the target mode: simulated "
                        "graph padding does not reproduce graph kernel fusion.")
    p.add_argument("--load-format", default="auto", dest="load_format",
                   help="vLLM load_format. 'dummy' skips reading weights and "
                        "initializes them randomly. Useful for controlled "
                        "dense-kernel diagnostics, but not automatically "
                        "end-to-end ground truth: verify resolved KV capacity "
                        "and execution paths, and use real weights or explicit "
                        "routing controls for MoE. Token IDs and output counts "
                        "are fixed by replay; routing and scheduling need not "
                        "be invariant. Recorded in meta.json.")
    p.add_argument("--tick-seconds", type=_positive_float, default=1.0,
                   dest="tick_seconds",
                   help="Stat logger downsample interval (timeseries.csv row spacing).")
    p.add_argument("--num-reqs", type=_nonnegative_int, default=0,
                   dest="num_reqs",
                   help="Cap on number of requests from the dataset (0 = all).")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")


def run(args: argparse.Namespace) -> int:
    global log
    from bench_multinode.core import logger_multinode as log

    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Multinode Bench",
        f"vLLM end-to-end run -> {args.output_dir}",
    )

    if args.data_parallel_size_local is not None and args.data_parallel_size_local > args.data_parallel_size:
        raise ValueError("--data-parallel-size-local cannot exceed --data-parallel-size")
    if args.data_parallel_backend is None:
        args.data_parallel_backend = 'ray' if args.data_parallel_size > 1 else 'mp'
    local = args.data_parallel_size_local
    if (args.data_parallel_backend != 'ray' and local is not None
            and local < args.data_parallel_size):
        raise ValueError('DP engines on other nodes require --data-parallel-backend ray')
    if args.ray_address and args.ray_address.startswith('ray://'):
        raise ValueError('Ray client addresses (ray://) cannot host vLLM workers')

    requests = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests:
        raise ValueError(f"No requests loaded from {args.dataset}")
    log.info("Loaded %d requests from %s", len(requests), args.dataset)

    # Exclusive run directories prevent stale complete artifacts masking failures.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    recorder.write_run_start(output_dir, model=args.model, dataset_path=args.dataset)
    try:
        from bench_multinode.core.stat_logger_multinode import BenchStatLogger

        BenchStatLogger.reset()
        with _ray_session(args):
            asyncio.run(_drive(args, requests, output_dir))
    except BaseException as exc:
        recorder.write_aborted(output_dir, exc)
        raise
    return 0


@contextmanager
def _ray_session(args):
    """Connect only to an existing cluster; leave its services running on exit."""
    import ray

    repo_root = str(Path(__file__).resolve().parents[2])
    names = ('RAY_ADDRESS', 'PYTHONPATH', 'VLLM_RAY_EXTRA_ENV_VARS_TO_COPY',
             'VLLM_RAY_DP_PACK_STRATEGY', 'VLLM_WORKER_MULTIPROC_METHOD')
    original = {name: os.environ.get(name) for name in names}
    connected = False
    try:
        os.environ['RAY_ADDRESS'] = args.ray_address
        # Ray has live threads before the local engine core is constructed.
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        os.environ['PYTHONPATH'] = os.pathsep.join(
            filter(None, (repo_root, original['PYTHONPATH'])))
        extra = set(filter(None, (original['VLLM_RAY_EXTRA_ENV_VARS_TO_COPY'] or '').split(',')))
        extra.update(('PYTHONPATH', 'RAY_ADDRESS'))
        os.environ['VLLM_RAY_EXTRA_ENV_VARS_TO_COPY'] = ','.join(sorted(extra))
        if args.ray_dp_pack_strategy is not None:
            os.environ['VLLM_RAY_DP_PACK_STRATEGY'] = args.ray_dp_pack_strategy
        if ray.is_initialized():
            raise RuntimeError('Run bench_multinode in a fresh Python process without an existing Ray context')
        with log.stage(f"Connecting to Ray at {args.ray_address}"):
            ray.init(address=args.ray_address,
                     runtime_env={'env_vars': {'PYTHONPATH': os.environ['PYTHONPATH']}},
                     log_to_driver=True)
        connected = True
        yield
    finally:
        try:
            if connected:
                ray.shutdown()
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> list[dict]:
    """Read a LLMServingSim-format JSONL workload.

    Skips agentic-session rows (with ``sub_requests``) — bench currently
    handles only flat requests. Each row must carry ``input_tok_ids``;
    bench cannot tokenize on the fly because the dataset's tokenizer may
    differ from ``args.model``.
    """
    requests: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sub_requests" in row:
                continue  # agentic sessions: not supported in bench yet
            if "input_tok_ids" not in row or not row["input_tok_ids"]:
                raise ValueError(
                    f"Row missing input_tok_ids in {path}; regenerate the "
                    f"dataset with `python -m workloads.generators`."
                )
            if (not isinstance(row['input_tok_ids'], list)
                    or any(type(token) is not int or token < 0 for token in row['input_tok_ids'])
                    or type(row.get('output_toks')) is not int or row['output_toks'] < 1
                    or type(row.get('input_toks')) is not int
                    or row['input_toks'] != len(row['input_tok_ids'])
                    or type(row.get('arrival_time_ns')) is not int or row['arrival_time_ns'] < 0):
                raise ValueError(f'Invalid token IDs, token counts or arrival_time_ns in {path}')
            requests.append(row)
            if cap and len(requests) >= cap:
                break
    return requests


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def _drive(args: argparse.Namespace, requests: list[dict], output_dir: Path) -> None:
    # Imports stay deferred so help and validation need no GPU packages.
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.config import CompilationConfig, KernelConfig, PassConfig
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM
    from bench_multinode.core.stat_logger_multinode import BenchStatLogger

    compilation_config = CompilationConfig(pass_config=PassConfig(
        fuse_allreduce_rms=False, fuse_gemm_comms=False,
    ))
    native_overrides = {}
    if args.all2all_backend is not None:
        native_overrides['all2all_backend'] = args.all2all_backend
    if args.kernel_config is not None:
        native_overrides['kernel_config'] = KernelConfig(**args.kernel_config)
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        data_parallel_size_local=args.data_parallel_size_local,
        distributed_executor_backend=args.distributed_executor_backend,
        data_parallel_backend=args.data_parallel_backend,
        enable_expert_parallel=args.enable_expert_parallel,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        compilation_config=compilation_config,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        seed=args.seed,
        load_format=args.load_format,
        skip_tokenizer_init=args.skip_tokenizer_init,
        enforce_eager=args.enforce_eager,
        worker_extension_cls='bench_multinode.core.execution_metadata_multinode.ExecutionObserver',
        disable_log_stats=False,
        **native_overrides,
    )
    engine_kwargs = _engine_kwargs_for_meta(engine_args)
    if args.kernel_config is not None:
        engine_kwargs['kernel_config'] = _config_fields(engine_args.kernel_config)
    engine_kwargs['compilation_config'] = {'pass_config': {
        name: getattr(compilation_config.pass_config, name)
        for name in ('fuse_allreduce_rms', 'fuse_gemm_comms')
    }}
    common = dict(
        model=args.model, vllm_version=_vllm_version(), engine_kwargs=engine_kwargs,
        dataset_path=str(args.dataset), dataset_hash=_hash_file(Path(args.dataset)),
        tick_seconds=args.tick_seconds,
        distributed={
            'ray_address': args.ray_address,
            'ray_dp_pack_strategy': os.environ.get('VLLM_RAY_DP_PACK_STRATEGY'),
            'worker_multiproc_method': os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'),
            'placement': 'user_managed',
        },
    )
    ready_capture = ReadyResponseCapture()
    engine = None
    records = []
    with log.stage('Booting AsyncLLM'):
        with log.capture_stdio():
            with ready_capture:
                engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[BenchStatLogger])
    try:
        common['started_at'] = _utc_now()
        common['kv_cache'] = _kv_cache_facts(engine, ready_capture)
        common['resolved_config'] = _resolved_config(engine)
        common['hardware'] = _hardware_facts()
        common['hardware']['scope'] = 'frontend_host_only'
        # Keep the existing graph observation, without physical placement checks.
        common['execution'] = await capture_execution(engine)
        recorder.write_engine_start(output_dir, num_requests=len(requests), **common)
        if not args.resolve_only:
            BenchStatLogger.reset(start_time=time.monotonic())
            with log.stage(f'Submitting {len(requests)} requests'):
                replay = _submit_all(engine, requests, SamplingParams, TokensPrompt)
                if args.run_timeout_seconds:
                    records = await asyncio.wait_for(replay, args.run_timeout_seconds)
                else:
                    records = await replay
            complete = await BenchStatLogger.wait_for_request_engines(
                [r['request_id'] for r in records])
            if not complete:
                # Without every engine identity, all rows share one clock domain.
                log.warning('Native statistics omitted some DP engine identities; '
                            'arrival alignment falls back to one clock domain')
            for record in records if complete else ():
                rank = BenchStatLogger.request_engines[record['request_id']]
                if not 0 <= rank < args.data_parallel_size:
                    raise RuntimeError(f'Unexpected DP engine index: {rank}')
                record['data_parallel_rank'] = rank
    finally:
        with log.stage('Shutting AsyncLLM down'):
            engine.shutdown()

    common['finished_at'] = _utc_now()
    if args.resolve_only:
        recorder.write_meta(output_dir, num_requests=0, resolve_only=True, **common)
        log.success('Resolved engine configuration -> %s', output_dir)
        return

    # The completion marker is last, after every data artifact succeeds.
    recorder.write_requests(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)
    from bench_multinode.core.validate_multinode import engine_clock_offsets
    offsets = engine_clock_offsets(records)
    recorder.write_meta(
        output_dir, status='completed', num_requests=len(records),
        timing={
            'engine_clock_domain': 'per_data_parallel_rank',
            'arrival_alignment': 'per_engine_minimum_queued_minus_arrival',
            'epoch_to_monotonic_offsets': {str(k): v for k, v in offsets.items()},
            'alignment_limit': 'Residual bias is each engine minimum request pickup delay; '
                               'this is not exact clock synchronization.',
            'frontend_observations': 'Separate frontend monotonic timestamps; '
                                     'first and last observations may contain multiple tokens.',
        },
        **common,
    )
    log.success('%d requests, %d timeseries rows -> %s', len(records), len(rows), output_dir)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


async def _submit_all(engine, requests: list[dict], SamplingParams, TokensPrompt) -> list[dict]:
    """Schedule each request at its arrival offset, gather metrics."""
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            target = t0_loop + req["arrival_time_ns"] / 1e9
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            # Strict replay: pin output length to whatever the dataset
            # recorded. ``ignore_eos`` blocks early termination; ``min_tokens``
            # blocks vLLM's async-scheduling early-exit (see vllm/v1/engine/
            # async_llm.py:async-scheduling block) so n_out is exactly fixed.
            n_out = int(req["output_toks"])
            sp = SamplingParams(
                min_tokens=n_out,
                max_tokens=n_out,
                ignore_eos=True,
                temperature=0.0,
            )
            prompt = TokensPrompt(prompt_token_ids=list(req["input_tok_ids"]))
            request_id = f"bench-{idx}"

            last_metrics = None
            generated_tokens = 0
            first_observed = last_observed = None
            dispatched = time.monotonic()
            finished = False
            async for output in engine.generate(prompt, sp, request_id):
                if output.metrics is not None:
                    last_metrics = output.metrics
                # SamplingParams defaults to cumulative outputs, with one sequence.
                count = len(output.outputs[0].token_ids) if output.outputs else 0
                if count > generated_tokens:
                    last_observed = time.monotonic()
                    if first_observed is None:
                        first_observed = last_observed
                generated_tokens = count
                finished = output.finished
            if not finished or generated_tokens != n_out or last_metrics is None:
                raise RuntimeError(f'{request_id}: incomplete output or missing native metrics '
                                   f'(expected {n_out} tokens, observed {generated_tokens})')
            bar.advance()
            record = _record_from_metrics(idx, req, last_metrics)
            lifecycle = ('arrival_time', 'queued_ts', 'scheduled_ts', 'first_token_ts', 'last_token_ts')
            if any(record[key] is None or not math.isfinite(record[key]) for key in lifecycle):
                raise RuntimeError(f'{request_id}: incomplete native lifecycle timestamps')
            if not 0 < record['queued_ts'] <= record['scheduled_ts'] <= record['first_token_ts'] <= record['last_token_ts']:
                raise RuntimeError(f'{request_id}: invalid native lifecycle timestamp order')
            record.update(
                actual_output_toks=generated_tokens,
                frontend_dispatch_ts=dispatched,
                frontend_first_output_ts=first_observed,
                frontend_last_output_ts=last_observed,
                frontend_completed_ts=time.monotonic(),
                frontend_ttft_ms=(first_observed - dispatched) * 1000.0,
                frontend_last_output_latency_ms=(last_observed - dispatched) * 1000.0,
            )
            return record

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def _record_from_metrics(idx: int, req: dict, metrics) -> dict:
    """Project ``RequestStateStats`` onto our flat per-request schema."""
    if metrics is None:
        return {
            "request_id": f"bench-{idx}",
            "input_toks": int(req["input_toks"]),
            "output_toks": int(req["output_toks"]),
            "arrival_time": None,
            "queued_ts": None,
            "scheduled_ts": None,
            "first_token_ts": None,
            "last_token_ts": None,
        }
    return {
        "request_id": f"bench-{idx}",
        "input_toks": int(req["input_toks"]),
        "output_toks": int(req["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "queued_ts": getattr(metrics, "queued_ts", None),
        "scheduled_ts": getattr(metrics, "scheduled_ts", None),
        "first_token_ts": getattr(metrics, "first_token_ts", None),
        "last_token_ts": getattr(metrics, "last_token_ts", None),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_kwargs_for_meta(engine_args) -> dict:
    fields = (
        "model", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "data_parallel_size_local", "distributed_executor_backend", "data_parallel_backend",
        "enable_expert_parallel", "max_num_seqs", "max_num_batched_tokens",
        "disable_custom_all_reduce",
        "all2all_backend",
        "max_model_len", "dtype", "kv_cache_dtype", "kv_cache_memory_bytes", "seed", "load_format",
        # Recorded for the same reason load_format is: a run booted from a
        # tokenizer-less config directory should never be mistaken for one
        # booted from the real checkpoint.
        "skip_tokenizer_init",
        # Recorded because it changes what the run *is*: a compiled engine and
        # an eager one are two different ground truths, and only the eager one
        # is comparable with a profiled latency.
        "enforce_eager",
    )
    return {k: getattr(engine_args, k, None) for k in fields}


# Values longer than this are recorded as a type tag instead, so one HF config
# object cannot bury the rest of meta.json.
_MAX_REPR = 200


def _normalize(value, depth: int = 0):
    """Reduce a config value to something JSON can hold.

    Scalars pass through. Enums become their value. Short flat containers are
    kept element-wise. Anything else -- an HF config object, a torch dtype, a
    callable -- becomes a ``"<type>"`` tag: the field is still recorded as
    present and resolved, without dragging its whole object graph into the file.
    """
    import enum

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return _normalize(value.value, depth + 1)
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        if len(seq) > 64:
            return f"<{type(value).__name__} len={len(seq)}>"
        return [_normalize(v, depth + 1) for v in seq]
    if isinstance(value, dict):
        if len(value) > 64 or depth > 2:
            return f"<dict len={len(value)}>"
        return {str(k): _normalize(v, depth + 1) for k, v in value.items()}
    text = repr(value)
    if len(text) <= _MAX_REPR:
        return text
    return f"<{type(value).__name__}>"


def _config_fields(obj) -> dict:
    """Field name -> normalized value for one config object.

    vLLM's config classes are pydantic-decorated dataclasses, so try the
    dataclass field list first, then pydantic's, then fall back to __dict__.
    Order does not matter: the dict is sorted on the way out.
    """
    import dataclasses

    names: list[str] = []
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    elif hasattr(type(obj), "model_fields"):
        names = list(type(obj).model_fields)
    elif hasattr(obj, "__dict__"):
        names = list(vars(obj))
    out = {}
    for name in sorted(names):
        if name.startswith("_"):
            continue
        try:
            out[name] = _normalize(getattr(obj, name))
        except Exception as exc:                     # a property may raise
            out[name] = f"<unreadable: {type(exc).__name__}>"
    return out


def _resolved_config(engine) -> dict:
    """Snapshot vLLM's *resolved* configuration, one section per sub-config.

    Reads the engine's own ``VllmConfig`` after boot, so these are the values
    vLLM actually ran with rather than the arguments we asked for -- defaults
    filled in, ``max_model_len`` inferred, ``num_gpu_blocks`` profiled. Walks
    ``VllmConfig``'s field list rather than a hand-kept list of interesting
    knobs, so a vLLM upgrade that adds one shows up here on its own.
    """
    cfg = getattr(engine, "vllm_config", None)
    if cfg is None:
        return {}
    out = {}
    for name, sub in _config_fields(cfg).items():
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            out[name] = sub                      # a scalar on VllmConfig itself
        else:
            fields = _config_fields(value)
            if name == "compilation_config":
                # This is an execution axis, not a verbose diagnostic list.
                # Preserve it even when there are more than 64 capture points.
                sizes = getattr(value, 'cudagraph_capture_sizes', None)
                if sizes is not None:
                    fields['cudagraph_capture_sizes'] = list(sizes)
                passes = getattr(value, "pass_config", None)
                if passes is not None:
                    fields["pass_config"] = _config_fields(passes)
            out[name] = fields if fields else sub
    return out


def _kv_cache_facts(engine, ready_capture=None) -> dict:
    """Raw client cache fields, with separately qualified startup capacities.

    DP clients can sum engine counts. The legacy ``num_kv_tokens`` product is
    raw arithmetic, not verified aggregate capacity when block sizes differ.
    Only complete ``capacity.per_engine`` observations support matching an
    individual engine's block pool; unknown observations are not inferred.
    """
    capacity = (ready_capture or ReadyResponseCapture()).snapshot(engine)
    cfg = getattr(engine, "vllm_config", None)
    if cfg is None:
        return {"capacity": capacity}
    cache = getattr(cfg, "cache_config", None)
    if cache is None:
        return {"capacity": capacity}
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    facts = {
        "num_gpu_blocks": blocks,
        "num_cpu_blocks": getattr(cache, "num_cpu_blocks", None),
        "block_size": block_size,
        "gpu_memory_utilization": getattr(cache, "gpu_memory_utilization", None),
        "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
        "kv_cache_memory_bytes": getattr(cache, "kv_cache_memory_bytes", None),
        "capacity": capacity,
    }
    if isinstance(blocks, int) and isinstance(block_size, int):
        facts["num_kv_tokens"] = blocks * block_size
    return facts


def _hardware_facts() -> dict:
    """Which accelerator this ran on, for matching against profiler/perf/<hw>/."""
    facts = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    facts["all_reduce_environment"] = {
        name: os.environ.get(name) for name in _ALL_REDUCE_ENV
    }
    try:
        if hasattr(os, "sched_getaffinity"):
            facts["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("Mems_allowed_list:"):
                    facts["numa_mems_allowed"] = line.split(":", 1)[1].strip()
    except OSError as exc:
        facts["placement_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch
        facts["torch_version"] = torch.__version__
        facts["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            facts["device_count"] = torch.cuda.device_count()
            facts["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            facts["device_total_memory_bytes"] = props.total_memory
            facts["device_capability"] = f"{props.major}.{props.minor}"
            facts["device_uuid"] = str(getattr(props, "uuid", "unknown"))
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
