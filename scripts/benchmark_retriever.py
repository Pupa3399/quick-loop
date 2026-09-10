from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq


def percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 100 HTTP retrieval queries")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--data", type=Path, default=Path("/data2/wuguanting/quick_loop/data/searchr1/test.parquet")
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.queries < 1 or args.batch_size < 1:
        raise ValueError("queries and batch-size must be positive")
    questions = pq.read_table(args.data, columns=["question"])["question"].to_pylist()
    questions = questions[: args.queries]
    request_latencies: list[float] = []
    per_query_latencies: list[float] = []
    started = time.perf_counter()
    with httpx.Client(base_url=args.url, timeout=120.0) as client:
        for offset in range(0, len(questions), args.batch_size):
            batch = questions[offset : offset + args.batch_size]
            request_started = time.perf_counter()
            response = client.post(
                "/search/batch", json={"queries": batch, "top_k": args.top_k}
            )
            response.raise_for_status()
            latency = time.perf_counter() - request_started
            results = response.json()["results"]
            if len(results) != len(batch) or any(
                len(result["documents"]) != args.top_k for result in results
            ):
                raise RuntimeError("Retriever returned an incomplete benchmark batch")
            request_latencies.append(latency)
            per_query_latencies.extend([latency / len(batch)] * len(batch))
    elapsed = time.perf_counter() - started
    report = {
        "queries": len(per_query_latencies),
        "batch_size": args.batch_size,
        "http_requests": len(request_latencies),
        "mean_latency_ms": statistics.fmean(per_query_latencies) * 1000,
        "p50_latency_ms": statistics.median(per_query_latencies) * 1000,
        "p95_latency_ms": percentile(per_query_latencies, 0.95) * 1000,
        "mean_request_latency_ms": statistics.fmean(request_latencies) * 1000,
        "qps": len(per_query_latencies) / elapsed,
        "elapsed_seconds": elapsed,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
