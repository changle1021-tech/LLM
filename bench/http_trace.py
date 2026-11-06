"""Sample project workloads and export server-timed HTTP replay traces."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import aiohttp


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def checked_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def read_receipt(headers, request_id):
    if headers.get("x-llmsim-receipt-version") != "1":
        raise ValueError(
            "Missing middleware headers. Restart vLLM with "
            "--middleware bench.arrival_middleware.ArrivalMiddleware"
        )
    if headers.get("x-llmsim-request-id") != request_id:
        raise ValueError("Receipt request ID mismatch")

    clock = headers.get("x-llmsim-clock-id")
    if not clock:
        raise ValueError("Missing server clock ID")

    result = {"server_clock_id": clock}
    fields = {
        "x-llmsim-arrival-monotonic-ns": "server_arrival_monotonic_ns",
        "x-llmsim-arrival-unix-ns": "server_arrival_unix_ns",
    }
    for header, field in fields.items():
        text = headers.get(header, "")
        if not text.isdigit():
            raise ValueError(f"Invalid or missing header: {header}")
        result[field] = int(text)
    return result


def sample_requests(args):
    candidates = []
    with args.dataset.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected JSON object")
            if "sub_requests" in row:
                raise ValueError("Only flat requests are supported")

            tokens = row.get("input_tok_ids")
            if not isinstance(tokens, list) or not tokens:
                raise ValueError(
                    f"Line {line_number}: nonempty input_tok_ids required"
                )
            for token in tokens:
                checked_int(token, "token ID")

            count = checked_int(row.get("input_toks"), "input_toks", 1)
            if count != len(tokens):
                raise ValueError(
                    f"Line {line_number}: input_toks differs from token IDs"
                )

            output_len = args.output_len
            if output_len is None:
                output_len = checked_int(
                    row.get("output_toks"), "output_toks", 1
                )

            if args.input_len is not None:
                if len(tokens) < args.input_len:
                    continue
                tokens = tokens[:args.input_len]

            candidates.append({
                "input_tok_ids": tokens,
                "requested_output_toks": output_len,
                "dataset_line": line_number,
            })

    if len(candidates) < args.num_requests:
        raise ValueError(
            f"Only {len(candidates)} eligible requests; "
            f"need {args.num_requests}. "
            "Reduce --num-requests or --input-len."
        )
    return random.Random(args.seed).sample(candidates, args.num_requests)


async def send_one(session, args, record):
    record["client_dispatch_monotonic_ns"] = time.monotonic_ns()
    record["client_dispatch_unix_ns"] = time.time_ns()
    record["dispatch_lateness_ns"] = max(
        0,
        record["client_dispatch_monotonic_ns"]
        - record["scheduled_client_monotonic_ns"],
    )

    payload = {
        "model": args.model,
        "prompt": record["input_tok_ids"],
        "max_tokens": record["requested_output_toks"],
        "min_tokens": record["requested_output_toks"],
        "ignore_eos": True,
        "temperature": 0,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async with session.post(
            args.base_url + "/v1/completions",
            json=payload,
            headers={"x-llmsim-request-id": record["request_id"]},
        ) as response:
            record["http_status"] = response.status
            record["client_headers_monotonic_ns"] = time.monotonic_ns()

            receipt_error = None
            try:
                record.update(
                    read_receipt(response.headers, record["request_id"])
                )
            except ValueError as exc:
                receipt_error = str(exc)
                record["receipt_error"] = receipt_error

            if response.status != 200:
                body = (await response.text())[:8192]
                raise ValueError(f"HTTP {response.status}: {body}")

            usage = None
            done = False
            async for line in response.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue

                item = json.loads(data)
                if item.get("error"):
                    raise ValueError(f"Stream error: {item['error']}")

                if any(c.get("text") for c in item.get("choices", [])):
                    stamp = time.monotonic_ns()
                    record.setdefault("client_first_chunk_monotonic_ns", stamp)
                    record["client_last_chunk_monotonic_ns"] = stamp
                    record["observed_text_chunks"] = record.get("observed_text_chunks", 0) + 1
                if item.get("usage") is not None:
                    usage = item["usage"]

            if not done:
                raise ValueError("Stream ended without [DONE]")
            if not isinstance(usage, dict):
                raise ValueError("Response omitted token usage")
            if receipt_error:
                raise ValueError(receipt_error)

            record["input_toks"] = checked_int(
                usage.get("prompt_tokens"), "prompt_tokens", 1
            )
            record["output_toks"] = checked_int(
                usage.get("completion_tokens"), "completion_tokens", 1
            )
            if record["input_toks"] != len(record["input_tok_ids"]):
                raise ValueError("Prompt usage differs from sent token IDs")

            record["status"] = "success"

    except asyncio.CancelledError:
        record.update(status="failed", error="Request cancelled")
    except Exception as exc:
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        record["client_completed_monotonic_ns"] = time.monotonic_ns()


async def drive(args, records):
    headers = {}
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(
        headers=headers,
        connector=connector,
        timeout=timeout,
    ) as session:
        probe_id = uuid.uuid4().hex
        async with session.get(
            args.base_url + "/v1/models",
            headers={"x-llmsim-request-id": probe_id},
        ) as response:
            if response.status != 200:
                raise ValueError(
                    f"Model probe HTTP {response.status}: "
                    f"{(await response.text())[:8192]}"
                )
            read_receipt(response.headers, probe_id)
            models = {
                item["id"]
                for item in (await response.json()).get("data", [])
            }
            if args.model not in models:
                raise ValueError(
                    f"Model ID not found. Available IDs: {sorted(models)}"
                )

        start = time.monotonic_ns()
        for record in records:
            record["scheduled_client_monotonic_ns"] = (
                start + record["planned_arrival_time_ns"]
            )

        tasks = []
        pending = set()
        try:
            for record in records:
                delay = (
                    record["scheduled_client_monotonic_ns"]
                    - time.monotonic_ns()
                ) / 1e9
                await asyncio.sleep(max(0, delay))

                if (
                    args.max_inflight is not None
                    and len(pending) >= args.max_inflight
                ):
                    record.update(
                        status="failed",
                        error="Client max-inflight reached; request not sent",
                    )
                    continue

                task = asyncio.create_task(send_one(session, args, record))
                tasks.append(task)
                pending.add(task)
                task.add_done_callback(pending.discard)

            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def export(args, records, error):
    out = args.output_dir
    with (out / "requests.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    received = [
        r for r in records if "server_arrival_monotonic_ns" in r
    ]
    success = [r for r in records if r["status"] == "success"]
    clocks = {r["server_clock_id"] for r in received}
    complete = (
        error is None
        and len(success) == len(records)
        and len(clocks) == 1
    )

    trace_name = None
    actual_qps = None
    if len(clocks) == 1:
        first = min(r["server_arrival_monotonic_ns"] for r in received)
        last = max(r["server_arrival_monotonic_ns"] for r in received)
        if len(received) > 1 and last > first:
            actual_qps = (len(received) - 1) * 1e9 / (last - first)

        trace_name = "trace.jsonl" if complete else "trace.partial.jsonl"
        with (out / trace_name).open("x") as stream:
            ordered = sorted(
                success,
                key=lambda r: (r["server_arrival_monotonic_ns"], r["index"]),
            )
            for record in ordered:
                row = {
                    name: record[name]
                    for name in (
                        "request_id",
                        "input_tok_ids",
                        "input_toks",
                        "output_toks",
                        "server_arrival_monotonic_ns",
                        "server_arrival_unix_ns",
                        "server_clock_id",
                    )
                }
                row["arrival_time_ns"] = (
                    record["server_arrival_monotonic_ns"] - first
                )
                stream.write(json.dumps(row) + "\n")

    elif len(clocks) > 1:
        error = "Multiple server clock domains; no replay trace written"

    summary = {
        "schema": "llmsim-http-trace-v1",
        "status": "complete" if complete else "incomplete",
        "arrival_boundary": "server ASGI entry before request body read",
        "model": args.model,
        "dataset": str(args.dataset),
        "seed": args.seed,
        "arrival_pattern": args.arrival_pattern,
        "planned_qps": args.qps,
        "server_arrival_qps": actual_qps,
        "requested_requests": len(records),
        "successful_requests": len(success),
        "failed_or_unsent_requests": len(records) - len(success),
        "requests_with_server_receipt": len(received),
        "server_clock_ids": sorted(clocks),
        "trace_file": trace_name,
        "error": error,
        "output_note": "Generated output token IDs are not recorded",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))

    for record in records:
        if record.get("error"):
            print("First request error:", record["error"])
            break
    from .http_metrics import report_metrics
    report_metrics(records, args.output_dir)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-requests", type=positive_int, default=100)
    parser.add_argument("--qps", type=positive_float, required=True)
    parser.add_argument(
        "--arrival-pattern", choices=["fixed", "poisson"], default="fixed"
    )
    parser.add_argument("--input-len", type=positive_int)
    parser.add_argument("--output-len", type=positive_int)
    parser.add_argument("--max-inflight", type=positive_int)
    parser.add_argument("--timeout", type=positive_float, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")

    if args.output_dir.exists():
        parser.error("--output-dir already exists; use a new directory")
    try:
        sampled = sample_requests(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    run_id = uuid.uuid4().hex
    rng = random.Random(args.seed)
    elapsed = 0.0
    records = []
    for index, item in enumerate(sampled):
        if index:
            if args.arrival_pattern == "poisson":
                elapsed += rng.expovariate(args.qps)
            else:
                elapsed = index / args.qps
        records.append({
            **item,
            "index": index,
            "request_id": f"{run_id}-{index}",
            "status": "not_sent",
            "planned_arrival_time_ns": round(elapsed * 1e9),
        })

    args.output_dir.mkdir(parents=True)
    error = None
    try:
        asyncio.run(drive(args, records))
    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return export(args, records, error)


if __name__ == "__main__":
    raise SystemExit(main())
