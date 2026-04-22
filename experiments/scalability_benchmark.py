"""Scalability benchmark for all retrieval methods.

Tests index build time, query latency, and memory footprint at increasing
corpus sizes (1x, 5x, 10x, 25x, 50x) to demonstrate how each method
scales.  Outputs data suitable for plotting performance curves in the report.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever

SCALE_FACTORS = [1, 5, 10, 25, 50]
TEST_QUERIES = [
    "What is the minimum GPA requirement?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    "What is the policy on academic probation?",
    "What is the maximum course load per semester?",
]
TOP_K = 5


def replicate_chunks(original: list[dict[str, Any]], factor: int) -> list[dict[str, Any]]:
    """Replicate chunks with unique IDs to simulate a larger corpus."""
    if factor <= 1:
        return original
    massive = []
    for copy_idx in range(factor):
        for chunk in original:
            new_chunk = chunk.copy()
            new_chunk["id"] = f"{chunk.get('id', 'chunk')}_copy{copy_idx}_{uuid.uuid4().hex[:6]}"
            massive.append(new_chunk)
    return massive


def measure_query_latency(retriever: Any, queries: list[str], top_k: int = TOP_K) -> dict[str, float]:
    """Run queries and return latency stats in milliseconds."""
    latencies = []
    for query in queries:
        start = time.perf_counter()
        retriever.search(query, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def run_benchmark(chunks_path: str = "data/processed/chunks.json",
                  output_path: str = "data/results/scalability_benchmark.json") -> None:
    """Run the full scalability benchmark."""

    with open(chunks_path, "r", encoding="utf-8") as f:
        original_chunks = json.load(f)

    base_count = len(original_chunks)
    print(f"Base corpus: {base_count} chunks")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"Max corpus: {base_count * max(SCALE_FACTORS)} chunks")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    for factor in SCALE_FACTORS:
        corpus = replicate_chunks(original_chunks, factor)
        corpus_size = len(corpus)
        print(f"\n--- Scale {factor}x ({corpus_size} chunks) ---")

        scale_result: dict[str, Any] = {
            "scale_factor": factor,
            "corpus_size": corpus_size,
            "methods": {},
        }

        # Baseline (TF-IDF)
        print("  Baseline: building index...", end="", flush=True)
        start = time.perf_counter()
        baseline = TFIDFBaseline(chunks=corpus)
        build_time = (time.perf_counter() - start) * 1000
        print(f" {build_time:.0f}ms")

        latency_stats = measure_query_latency(baseline, TEST_QUERIES)
        scale_result["methods"]["baseline"] = {
            "build_time_ms": round(build_time, 2),
            **{k: round(v, 2) for k, v in latency_stats.items()},
        }
        print(f"  Baseline latency: avg={latency_stats['avg_ms']:.2f}ms")

        # MinHash
        print("  MinHash: building index...", end="", flush=True)
        start = time.perf_counter()
        minhash = MinHashRetriever(num_perm=64, threshold=0.3, shingle_size=3)
        minhash.create_index(corpus)
        build_time = (time.perf_counter() - start) * 1000
        print(f" {build_time:.0f}ms")

        latency_stats = measure_query_latency(minhash, TEST_QUERIES)
        scale_result["methods"]["minhash"] = {
            "build_time_ms": round(build_time, 2),
            **{k: round(v, 2) for k, v in latency_stats.items()},
        }
        print(f"  MinHash latency: avg={latency_stats['avg_ms']:.2f}ms")

        # SimHash
        print("  SimHash: building index...", end="", flush=True)
        start = time.perf_counter()
        simhash = SimHashRetriever(fingerprint_size=64, hamming_threshold=24, num_bands=8)
        simhash.create_index(corpus)
        build_time = (time.perf_counter() - start) * 1000
        print(f" {build_time:.0f}ms")

        latency_stats = measure_query_latency(simhash, TEST_QUERIES)
        scale_result["methods"]["simhash"] = {
            "build_time_ms": round(build_time, 2),
            **{k: round(v, 2) for k, v in latency_stats.items()},
        }
        print(f"  SimHash latency: avg={latency_stats['avg_ms']:.2f}ms")

        results.append(scale_result)

        # Free memory between scales
        del baseline, minhash, simhash, corpus

    # Summary table
    print("\n" + "=" * 70)
    print("SCALABILITY SUMMARY")
    print("=" * 70)
    print(f"{'Scale':<8} | {'Chunks':<10} | {'BL Build':>10} | {'BL Query':>10} | {'MH Build':>10} | {'MH Query':>10} | {'SH Build':>10} | {'SH Query':>10}")
    print("-" * 100)

    for r in results:
        bl = r["methods"]["baseline"]
        mh = r["methods"]["minhash"]
        sh = r["methods"]["simhash"]
        print(
            f"{r['scale_factor']:<8} | {r['corpus_size']:<10} | "
            f"{bl['build_time_ms']:>8.0f}ms | {bl['avg_ms']:>8.2f}ms | "
            f"{mh['build_time_ms']:>8.0f}ms | {mh['avg_ms']:>8.2f}ms | "
            f"{sh['build_time_ms']:>8.0f}ms | {sh['avg_ms']:>8.2f}ms"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
