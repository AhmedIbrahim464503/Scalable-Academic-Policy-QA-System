"""SimHash retrieval using the simhash library."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from simhash import Simhash


class SimHashRetriever:
    """Approximate similarity search using SimHash fingerprints."""

    def __init__(self, fingerprint_size: int = 64, hamming_threshold: int = 20) -> None:
        self.fingerprint_size = fingerprint_size
        self.hamming_threshold = hamming_threshold

        self.chunks: dict[str, dict[str, Any]] = {}
        self.fingerprints: dict[str, Simhash] = {}

        self.index_time = 0.0
        self.total_queries = 0

    def _normalize_chunk(self, chunk: dict[str, Any], fallback_index: int) -> dict[str, Any]:
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{fallback_index}")
        page = chunk.get("page") or chunk.get("page_number") or chunk.get("source_page") or "n/a"
        return {
            **chunk,
            "id": chunk_id,
            "page": page,
            "text": str(chunk.get("text", "")),
        }

    def _preprocess_text(self, text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _token_features(self, text: str) -> list[str]:
        """Use token bigrams when possible so the fingerprint carries more structure."""
        words = self._preprocess_text(text).split()
        if len(words) < 2:
            return words or [""]
        return [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]

    def _create_simhash(self, text: str) -> Simhash:
        return Simhash(self._token_features(text), f=self.fingerprint_size)

    def create_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build the SimHash fingerprint store."""
        print("\nBuilding SimHash index...")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Parameters: bits={self.fingerprint_size}, hamming_threshold={self.hamming_threshold}")

        start = time.time()
        self.chunks.clear()
        self.fingerprints.clear()

        for index, raw_chunk in enumerate(chunks, start=1):
            if index % 100 == 0 or index == len(chunks):
                print(f"  Progress: {index}/{len(chunks)}", end="\r")

            chunk = self._normalize_chunk(raw_chunk, fallback_index=index - 1)
            chunk_id = chunk["id"]
            self.chunks[chunk_id] = chunk
            self.fingerprints[chunk_id] = self._create_simhash(chunk["text"])

        self.index_time = time.time() - start
        print(f"\nIndex built in {self.index_time:.2f}s")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar chunks by Hamming distance and return top results."""
        self.total_queries += 1
        query_hash = self._create_simhash(query)

        results: list[dict[str, Any]] = []
        for chunk_id, chunk_hash in self.fingerprints.items():
            distance = query_hash.distance(chunk_hash)
            if distance > self.hamming_threshold:
                continue

            similarity = 1.0 - (distance / self.fingerprint_size)
            chunk = self.chunks[chunk_id].copy()
            chunk["score"] = float(similarity)
            chunk["hamming_distance"] = int(distance)
            results.append(chunk)

        if not results:
            for chunk_id, chunk_hash in self.fingerprints.items():
                distance = query_hash.distance(chunk_hash)
                similarity = 1.0 - (distance / self.fingerprint_size)
                chunk = self.chunks[chunk_id].copy()
                chunk["score"] = float(similarity)
                chunk["hamming_distance"] = int(distance)
                results.append(chunk)

        results.sort(key=lambda item: (item["score"], -item["hamming_distance"]), reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        return {
            "method": "SimHash",
            "num_chunks": len(self.chunks),
            "fingerprint_size": self.fingerprint_size,
            "hamming_threshold": self.hamming_threshold,
            "index_time_s": self.index_time,
            "total_queries": self.total_queries,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMHASH")
    print("=" * 60)

    chunks_path = Path("data/processed/chunks.json")
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    print(f"\nLoaded {len(chunks)} chunks")

    retriever = SimHashRetriever(fingerprint_size=64, hamming_threshold=20)
    retriever.create_index(chunks)

    query = "What is the minimum GPA requirement?"
    print(f"\nTest Query: {query}")

    start = time.time()
    results = retriever.search(query, top_k=5)
    latency = (time.time() - start) * 1000

    print(f"\nFound {len(results)} results in {latency:.2f}ms")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. Score: {result['score']:.3f} | "
            f"Hamming: {result['hamming_distance']} | Page: {result['page']}"
        )
        print(f"   {result['text'][:100]}...")

    print("\n" + "=" * 60)
    print("Stats:", retriever.get_stats())
    print("=" * 60)
