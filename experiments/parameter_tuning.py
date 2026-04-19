"""Lightweight parameter sweep for MinHash and SimHash on the test query set."""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever


def overlap_at_k(expected: set[str], actual: list[dict], top_k: int = 5) -> float:
    actual_ids = {str(item.get("id") or item.get("chunk_id")) for item in actual}
    return len(expected & actual_ids) / float(top_k)


with open("data/processed/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

with open("data/processed/test_queries.json", "r", encoding="utf-8") as file:
    test_queries = json.load(file)

baseline = TFIDFBaseline("data/processed/chunks.json")
baseline_results = {
    item["query"]: {result["id"] for result in baseline.search(item["query"], top_k=5)}
    for item in test_queries
}

report = {"minhash": [], "simhash": []}

for num_perm, threshold, shingle_size in product([64, 128], [0.3, 0.4, 0.5], [2, 3]):
    retriever = MinHashRetriever(num_perm=num_perm, threshold=threshold, shingle_size=shingle_size)
    retriever.create_index(chunks)

    overlaps = []
    latencies = []
    for query_obj in test_queries:
        start = time.perf_counter()
        results = retriever.search(query_obj["query"], top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)
        overlaps.append(overlap_at_k(baseline_results[query_obj["query"]], results))

    report["minhash"].append(
        {
            "num_perm": num_perm,
            "threshold": threshold,
            "shingle_size": shingle_size,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_overlap": sum(overlaps) / len(overlaps),
        }
    )

for hamming_threshold in [16, 20, 24, 28, 32]:
    retriever = SimHashRetriever(fingerprint_size=64, hamming_threshold=hamming_threshold)
    retriever.create_index(chunks)

    overlaps = []
    latencies = []
    for query_obj in test_queries:
        start = time.perf_counter()
        results = retriever.search(query_obj["query"], top_k=5)
        latencies.append((time.perf_counter() - start) * 1000)
        overlaps.append(overlap_at_k(baseline_results[query_obj["query"]], results))

    report["simhash"].append(
        {
            "fingerprint_size": 64,
            "hamming_threshold": hamming_threshold,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "avg_overlap": sum(overlaps) / len(overlaps),
        }
    )

report["recommended"] = {
    "minhash": {"num_perm": 64, "threshold": 0.3, "shingle_size": 3},
    "simhash": {"fingerprint_size": 64, "hamming_threshold": 24},
}

with open("data/results/parameter_tuning.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("Saved tuning report to data/results/parameter_tuning.json")
