"""Report client-observed HTTP benchmark performance."""

import argparse
import json
from pathlib import Path


def distribution(values):
    values = sorted(values)

    def percentile(q):
        if not values:
            return None
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(.50),
        "p90": percentile(.90),
        "p95": percentile(.95),
        "p99": percentile(.99),
    }


def report_metrics(records, output_dir):
    sent = [r for r in records if "client_dispatch_monotonic_ns" in r]
    success = [r for r in sent if r["status"] == "success"]
    ends = [
        r["client_completed_monotonic_ns"]
        for r in sent if "client_completed_monotonic_ns" in r
    ]
    duration = (
        (max(ends) - min(r["client_dispatch_monotonic_ns"] for r in sent))
        / 1e9 if sent and ends else 0
    )

    e2e, ttft, tpot = [], [], []
    for r in success:
        start = r["client_dispatch_monotonic_ns"]
        end = r.get("client_completed_monotonic_ns")
        first = r.get("client_first_chunk_monotonic_ns")
        last = r.get("client_last_chunk_monotonic_ns")
        if end is not None:
            e2e.append((end - start) / 1e6)
        if first is not None:
            ttft.append((first - start) / 1e6)
        if (
            first is not None and last is not None and last > first
            and r["output_toks"] > 1
            and r.get("observed_text_chunks", 0) > 1
        ):
            tpot.append((last - first) / 1e6 / (r["output_toks"] - 1))

    input_tokens = sum(r["input_toks"] for r in success)
    output_tokens = sum(r["output_toks"] for r in success)
    rate = lambda count: count / duration if duration > 0 else None

    result = {
        "requested": len(records),
        "sent": len(sent),
        "succeeded": len(success),
        "failed_sent": sum(r["status"] == "failed" for r in sent),
        "unsent": len(records) - len(sent),
        "duration_s": duration,
        "successful_request_throughput_rps": rate(len(success)),
        "successful_input_tokens": input_tokens,
        "successful_output_tokens": output_tokens,
        "input_token_throughput_tps": rate(input_tokens),
        "output_token_throughput_tps": rate(output_tokens),
        "e2e_ms": distribution(e2e),
        "ttft_obs_ms": distribution(ttft),
        "tpot_est_ms": distribution(tpot),
        "note": (
            "Client-observed latency; successful requests only. "
            "Duration spans first dispatch to last recorded completion. "
            "Historical TTFT may include an empty first chunk. "
            "TPOT_est requires multiple observed nonempty text chunks."
        ),
    }

    output = Path(output_dir) / "metrics.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    print("\n===== HTTP benchmark results (client observed) =====")
    print(
        f"Requests: total={len(records)} sent={len(sent)} "
        f"success={len(success)} failed_sent={result['failed_sent']} "
        f"unsent={result['unsent']}"
    )
    print(f"Duration: {duration:.3f} s")
    print(f"Request throughput: {fmt(rate(len(success)))} req/s")
    print(f"Input / output tokens: {input_tokens} / {output_tokens}")
    print(
        f"Input / output throughput: {fmt(rate(input_tokens))} / "
        f"{fmt(rate(output_tokens))} tokens/s"
    )
    print(
        f"{'Latency (ms)':<15} {'n':>6} "
        f"{'mean':>12} {'p50':>12} {'p90':>12} {'p95':>12} {'p99':>12}"
    )
    for label, key in (
        ("E2E", "e2e_ms"),
        ("TTFT_obs", "ttft_obs_ms"),
        ("TPOT_est", "tpot_est_ms"),
    ):
        stats = result[key]
        columns = " ".join(
            f"{fmt(stats[name]):>12}"
            for name in ("mean", "p50", "p90", "p95", "p99")
        )
        print(f"{label:<15} {stats['n']:>6} {columns}")
    print("TPOT_est is estimated from stream chunks; unavailable values are N/A.")
    print(f"Saved: {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    with (args.run_dir / "requests.jsonl").open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    report_metrics(records, args.run_dir)


if __name__ == "__main__":
    main()
