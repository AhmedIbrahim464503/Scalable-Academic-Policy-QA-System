"""Evaluate retrieval accuracy using ground-truth page annotations.

Computes Precision@k, Recall@k, and query latency for all three methods
(baseline TF-IDF, MinHash LSH, SimHash) against the annotated test queries.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import TFIDFBaseline
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever

MINHASH_DEFAULTS = {"num_perm": 64, "threshold": 0.3, "shingle_size": 3}
SIMHASH_DEFAULTS = {"fingerprint_size": 64, "hamming_threshold": 24, "num_bands": 8}
TOP_K = 5


def get_page(chunk: dict[str, Any]) -> int | None:
    """Extract page number from a chunk dict."""
    page = chunk.get("page") or chunk.get("page_number") or chunk.get("source_page")
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return None


def precision_at_k(retrieved: list[dict], expected_pages: list[int], k: int = TOP_K) -> float:
    """Fraction of top-k retrieved chunks whose page is in the expected set."""
    if not retrieved or not expected_pages:
        return 0.0
    expected_set = set(expected_pages)
    hits = sum(1 for chunk in retrieved[:k] if get_page(chunk) in expected_set)
    return hits / min(k, len(retrieved))


def recall_at_k(retrieved: list[dict], expected_pages: list[int], k: int = TOP_K) -> float:
    """Fraction of expected pages covered by any of the top-k results."""
    if not retrieved or not expected_pages:
        return 0.0
    expected_set = set(expected_pages)
    found_pages = {get_page(chunk) for chunk in retrieved[:k]}
    hits = len(expected_set & found_pages)
    return hits / len(expected_set)


def run_evaluation(chunks_path: str = "data/processed/chunks.json",
                   queries_path: str = "data/processed/test_queries.json",
                   output_path: str = "data/results/accuracy_evaluation.json") -> None:
    """Run full evaluation and output structured results."""

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(queries_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"Loaded {len(chunks)} chunks and {len(test_queries)} queries")
    print("=" * 70)

    # Initialize retrievers
    print("Initializing retrievers...")
    baseline = TFIDFBaseline(chunks_path)
    minhash = MinHashRetriever(**MINHASH_DEFAULTS)
    minhash.create_index(chunks)
    simhash = SimHashRetriever(**SIMHASH_DEFAULTS)
    simhash.create_index(chunks)

    retrievers = {"baseline": baseline, "minhash": minhash, "simhash": simhash}

    results: list[dict[str, Any]] = []
    method_metrics: dict[str, dict[str, list[float]]] = {
        name: {"precision": [], "recall": [], "latency_ms": []}
        for name in retrievers
    }

    for query_obj in test_queries:
        query = query_obj["query"]
        expected = query_obj.get("expected_pages", [])
        query_result: dict[str, Any] = {"query": query, "expected_pages": expected, "methods": {}}

        for name, retriever in retrievers.items():
            start = time.perf_counter()
            search_results = retriever.search(query, top_k=TOP_K)
            latency_ms = (time.perf_counter() - start) * 1000

            prec = precision_at_k(search_results, expected, TOP_K)
            rec = recall_at_k(search_results, expected, TOP_K)

            method_metrics[name]["precision"].append(prec)
            method_metrics[name]["recall"].append(rec)
            method_metrics[name]["latency_ms"].append(latency_ms)

            retrieved_pages = [get_page(c) for c in search_results[:TOP_K]]
            query_result["methods"][name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "latency_ms": round(latency_ms, 2),
                "retrieved_pages": retrieved_pages,
                "top_score": round(float(search_results[0].get("score", 0)) if search_results else 0, 4),
            }

        results.append(query_result)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Method':<12} | {'Avg Prec@5':>10} | {'Avg Rec@5':>10} | {'Avg Latency':>12} | {'Med Latency':>12}")
    print("-" * 70)

    summary: dict[str, Any] = {}
    for name, metrics in method_metrics.items():
        avg_prec = float(np.mean(metrics["precision"]))
        avg_rec = float(np.mean(metrics["recall"]))
        avg_lat = float(np.mean(metrics["latency_ms"]))
        med_lat = float(np.median(metrics["latency_ms"]))
        print(f"{name:<12} | {avg_prec:>10.4f} | {avg_rec:>10.4f} | {avg_lat:>10.2f}ms | {med_lat:>10.2f}ms")
        summary[name] = {
            "avg_precision_at_5": round(avg_prec, 4),
            "avg_recall_at_5": round(avg_rec, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "median_latency_ms": round(med_lat, 2),
        }

    print("=" * 70)

    output = {
        "config": {
            "top_k": TOP_K,
            "chunks_path": chunks_path,
            "num_chunks": len(chunks),
            "num_queries": len(test_queries),
            "minhash_params": MINHASH_DEFAULTS,
            "simhash_params": SIMHASH_DEFAULTS,
        },
        "summary": summary,
        "per_query": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
