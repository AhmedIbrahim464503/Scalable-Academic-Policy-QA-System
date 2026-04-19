"""Compare baseline, MinHash, and SimHash retrieval on the test query set."""

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
from src.lsh_simhash import SimHashRetriever

MINHASH_DEFAULTS = {"num_perm": 64, "threshold": 0.3, "shingle_size": 3}
SIMHASH_DEFAULTS = {"fingerprint_size": 64, "hamming_threshold": 24}


def result_ids(results: list[dict]) -> list[str]:
    return [str(item.get("id") or item.get("chunk_id")) for item in results]


with open("data/processed/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

with open("data/processed/test_queries.json", "r", encoding="utf-8") as file:
    test_queries = json.load(file)

print("Initializing all methods...")
baseline = TFIDFBaseline("data/processed/chunks.json")
minhash = MinHashRetriever(**MINHASH_DEFAULTS)
minhash.create_index(chunks)
simhash = SimHashRetriever(**SIMHASH_DEFAULTS)
simhash.create_index(chunks)

results = []

for query_obj in test_queries:
    query = query_obj["query"]
    print(f"\nQuery: {query}")

    query_result = {"query": query, "methods": {}}

    for name, retriever in [("baseline", baseline), ("minhash", minhash), ("simhash", simhash)]:
        start = time.perf_counter()
        method_results = retriever.search(query, top_k=5)
        latency = (time.perf_counter() - start) * 1000

        query_result["methods"][name] = {
            "latency_ms": latency,
            "num_results": len(method_results),
            "chunk_ids": result_ids(method_results),
        }
        print(f"  {name:10s}: {latency:6.1f}ms | {len(method_results)} results")

    results.append(query_result)

with open("data/results/all_methods_comparison.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for method in ["baseline", "minhash", "simhash"]:
    latencies = [item["methods"][method]["latency_ms"] for item in results]
    print(f"\n{method.upper()}:")
    print(f"  Avg Latency: {np.mean(latencies):.1f}ms")
    print(f"  Min Latency: {np.min(latencies):.1f}ms")
    print(f"  Max Latency: {np.max(latencies):.1f}ms")

print("\nComparison complete.")

