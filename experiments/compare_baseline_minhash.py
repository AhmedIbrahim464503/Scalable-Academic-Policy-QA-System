"""Compare TF-IDF baseline and MinHash retrieval on the test query set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever


def _result_ids(results: list[dict]) -> set[str]:
    return {str(result.get("id") or result.get("chunk_id")) for result in results}


chunks_path = Path("data/processed/chunks.json")
queries_path = Path("data/processed/test_queries.json")
output_path = Path("data/results/baseline_vs_minhash.json")

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

with queries_path.open("r", encoding="utf-8") as file:
    test_queries = json.load(file)

print("Initializing retrievers...")
baseline = TFIDFBaseline(chunks_path)
minhash = MinHashRetriever(num_perm=128, threshold=0.5)
minhash.create_index(chunks)

results = []
top_k = 5

for query_obj in test_queries:
    query = query_obj["query"]

    start = time.perf_counter()
    baseline_results = baseline.search(query, top_k=top_k)
    baseline_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    minhash_results = minhash.search(query, top_k=top_k)
    minhash_time = (time.perf_counter() - start) * 1000

    baseline_ids = _result_ids(baseline_results)
    minhash_ids = _result_ids(minhash_results)
    overlap = len(baseline_ids & minhash_ids) / float(top_k)

    results.append(
        {
            "query": query,
            "baseline_time_ms": baseline_time,
            "minhash_time_ms": minhash_time,
            "speedup": baseline_time / minhash_time if minhash_time > 0 else 0.0,
            "overlap": overlap,
            "baseline_count": len(baseline_results),
            "minhash_count": len(minhash_results),
        }
    )

    speedup = baseline_time / minhash_time if minhash_time > 0 else 0.0
    print(f"Query: {query[:50]}")
    print(
        f"  Baseline: {baseline_time:.1f}ms | "
        f"MinHash: {minhash_time:.1f}ms | "
        f"Speedup: {speedup:.1f}x | "
        f"Overlap: {overlap:.0%}"
    )

with output_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Avg Baseline Time: {np.mean([item['baseline_time_ms'] for item in results]):.1f}ms")
print(f"Avg MinHash Time: {np.mean([item['minhash_time_ms'] for item in results]):.1f}ms")
print(f"Avg Speedup: {np.mean([item['speedup'] for item in results]):.1f}x")
print(f"Avg Overlap: {np.mean([item['overlap'] for item in results]):.0%}")
