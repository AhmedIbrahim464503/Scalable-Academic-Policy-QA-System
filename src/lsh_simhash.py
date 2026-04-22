"""SimHash retrieval with bit-band partitioning for sub-linear search.

Instead of scanning every fingerprint (O(N)), the index partitions each
fingerprint into ``num_bands`` bands and hashes each band into a bucket
table.  At query time we only compute full Hamming distance for the union
of candidates across all band tables, giving near-O(1) approximate lookup.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from simhash import Simhash


class SimHashRetriever:
    """Approximate similarity search using SimHash fingerprints with band indexing."""

    def __init__(self, fingerprint_size: int = 64, hamming_threshold: int = 20, num_bands: int = 8) -> None:
        self.fingerprint_size = fingerprint_size
        self.hamming_threshold = hamming_threshold
        self.num_bands = num_bands
        self.band_size = fingerprint_size // num_bands

        self.chunks: dict[str, dict[str, Any]] = {}
        self.fingerprints: dict[str, Simhash] = {}
        # Band tables: list of dicts mapping band_hash -> set of chunk_ids
        self.band_tables: list[dict[int, set[str]]] = [defaultdict(set) for _ in range(num_bands)]

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
        cleaned = re.sub(r"[^\w\s.]", "", text.lower())
        raw_words = cleaned.split()
        stops = {"nust", "university", "student", "handbook", "policy", "chapter", "page", "section", 
                 "the", "in", "is", "of", "and", "to", "for", "a", "an", "on", "with", "as", "or", "be", "it", "are"}
        words = [w for w in raw_words if w not in stops]
        return " ".join(words)

    def _token_features(self, text: str) -> list[str]:
        """Use unigrams and bigrams so both short queries and long chunks match well."""
        words = self._preprocess_text(text).split()
        if not words:
            return [""]

        features: list[str] = list(words)  # unigrams
        # bigrams
        for index in range(len(words) - 1):
            features.append(f"{words[index]} {words[index + 1]}")
        return features

    def _create_simhash(self, text: str) -> Simhash:
        return Simhash(self._token_features(text), f=self.fingerprint_size)

    def _get_bands(self, fingerprint: Simhash) -> list[int]:
        """Partition the fingerprint value into num_bands sub-hashes."""
        value = fingerprint.value
        bands: list[int] = []
        mask = (1 << self.band_size) - 1
        for _ in range(self.num_bands):
            bands.append(value & mask)
            value >>= self.band_size
        return bands

    def create_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build the SimHash fingerprint store with band-indexed lookup tables."""
        print("\nBuilding SimHash index with bit-band partitioning...")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Parameters: bits={self.fingerprint_size}, hamming_threshold={self.hamming_threshold}, bands={self.num_bands}")

        start = time.time()
        self.chunks.clear()
        self.fingerprints.clear()
        self.band_tables = [defaultdict(set) for _ in range(self.num_bands)]

        for index, raw_chunk in enumerate(chunks, start=1):
            if index % 100 == 0 or index == len(chunks):
                print(f"  Progress: {index}/{len(chunks)}", end="\r")

            chunk = self._normalize_chunk(raw_chunk, fallback_index=index - 1)
            chunk_id = chunk["id"]
            self.chunks[chunk_id] = chunk

            fp = self._create_simhash(chunk["text"])
            self.fingerprints[chunk_id] = fp

            # Insert into band tables for sub-linear lookup
            bands = self._get_bands(fp)
            for band_idx, band_hash in enumerate(bands):
                self.band_tables[band_idx][band_hash].add(chunk_id)

        self.index_time = time.time() - start
        print(f"\nIndex built in {self.index_time:.2f}s")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search using band-indexed approximate lookup, then rank by Hamming distance."""
        self.total_queries += 1
        query_hash = self._create_simhash(query)

        # Phase 1: Collect candidates from band tables (sub-linear)
        candidate_ids: set[str] = set()
        query_bands = self._get_bands(query_hash)
        for band_idx, band_hash in enumerate(query_bands):
            candidate_ids.update(self.band_tables[band_idx].get(band_hash, set()))

        # Phase 2: Score candidates by full Hamming distance
        results: list[dict[str, Any]] = []
        for chunk_id in candidate_ids:
            distance = query_hash.distance(self.fingerprints[chunk_id])
            if distance > self.hamming_threshold:
                continue
            similarity = 1.0 - (distance / self.fingerprint_size)
            chunk = self.chunks[chunk_id].copy()
            chunk["score"] = float(similarity)
            chunk["hamming_distance"] = int(distance)
            results.append(chunk)

        # Fallback: if band lookup returned nothing useful, scan all (graceful degradation)
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
            "method": "SimHash (bit-band indexed)",
            "num_chunks": len(self.chunks),
            "fingerprint_size": self.fingerprint_size,
            "hamming_threshold": self.hamming_threshold,
            "num_bands": self.num_bands,
            "band_size": self.band_size,
            "index_time_s": self.index_time,
            "total_queries": self.total_queries,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMHASH (BIT-BAND INDEXED)")
    print("=" * 60)

    chunks_path = Path("data/processed/chunks.json")
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    print(f"\nLoaded {len(chunks)} chunks")

    retriever = SimHashRetriever(fingerprint_size=64, hamming_threshold=20, num_bands=8)
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
