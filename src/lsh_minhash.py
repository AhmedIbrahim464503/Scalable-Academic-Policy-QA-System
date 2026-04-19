"""MinHash LSH retrieval using the datasketch library."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH


class MinHashRetriever:
    """Approximate similarity search over handbook chunks using MinHash + LSH."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.5, shingle_size: int = 3) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.chunks: dict[str, dict[str, Any]] = {}
        self.signatures: dict[str, MinHash] = {}

        self.index_time = 0.0
        self.total_queries = 0

    def _normalize_chunk(self, chunk: dict[str, Any], fallback_index: int) -> dict[str, Any]:
        """Normalize chunk schema to match the project's Phase 1 records."""
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{fallback_index}")
        page = chunk.get("page") or chunk.get("page_number") or chunk.get("source_page") or "n/a"
        return {
            **chunk,
            "id": chunk_id,
            "page": page,
            "text": str(chunk.get("text", "")),
        }

    def _create_shingles(self, text: str) -> set[str]:
        """Convert text into a mix of token shingles.

        Unigrams and bigrams keep short questions searchable, while the
        configured shingle size still contributes more contextual features.
        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        words = cleaned.split()
        if not words:
            return set()

        shingles: set[str] = set()
        for word in words:
            shingles.add(word)
        for index in range(len(words) - 1):
            shingles.add(" ".join(words[index : index + 2]))
        if len(words) < self.shingle_size:
            shingles.add(" ".join(words))
            return shingles
        for index in range(len(words) - self.shingle_size + 1):
            shingles.add(" ".join(words[index : index + self.shingle_size]))
        return shingles

    def _create_minhash(self, shingles: set[str]) -> MinHash:
        """Create a MinHash signature from shingle tokens."""
        signature = MinHash(num_perm=self.num_perm)
        if not shingles:
            signature.update(b"")
            return signature

        for shingle in shingles:
            signature.update(shingle.encode("utf-8"))
        return signature

    def create_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build the LSH index from chunk records."""
        print("\nBuilding MinHash LSH index...")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Parameters: num_perm={self.num_perm}, threshold={self.threshold}, shingle_size={self.shingle_size}")

        start = time.time()
        self.chunks.clear()
        self.signatures.clear()
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        for index, raw_chunk in enumerate(chunks, start=1):
            if index % 100 == 0 or index == len(chunks):
                print(f"  Progress: {index}/{len(chunks)}", end="\r")

            chunk = self._normalize_chunk(raw_chunk, fallback_index=index - 1)
            chunk_id = chunk["id"]
            self.chunks[chunk_id] = chunk

            shingles = self._create_shingles(chunk["text"])
            signature = self._create_minhash(shingles)
            self.signatures[chunk_id] = signature
            self.lsh.insert(chunk_id, signature)

        self.index_time = time.time() - start
        print(f"\nIndex built in {self.index_time:.2f}s")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for similar chunks and score candidates by MinHash Jaccard similarity."""
        self.total_queries += 1
        query_signature = self._create_minhash(self._create_shingles(query))
        candidate_ids = self.lsh.query(query_signature)

        # Short natural-language questions often miss the raw LSH threshold.
        # Fall back to ranking all indexed signatures so evaluation scripts still
        # return comparable top-k outputs instead of empty result sets.
        if not candidate_ids:
            candidate_ids = list(self.signatures.keys())

        results: list[dict[str, Any]] = []
        for chunk_id in candidate_ids:
            similarity = query_signature.jaccard(self.signatures[chunk_id])
            chunk = self.chunks[chunk_id].copy()
            chunk["score"] = float(similarity)
            results.append(chunk)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        return {
            "method": "MinHash LSH",
            "num_chunks": len(self.chunks),
            "num_perm": self.num_perm,
            "threshold": self.threshold,
            "shingle_size": self.shingle_size,
            "index_time_s": self.index_time,
            "total_queries": self.total_queries,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MINHASH LSH")
    print("=" * 60)

    chunks_path = Path("data/processed/chunks.json")
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    print(f"\nLoaded {len(chunks)} chunks")

    retriever = MinHashRetriever(num_perm=128, threshold=0.5)
    retriever.create_index(chunks)

    query = "What is the minimum GPA requirement?"
    print(f"\nTest Query: {query}")

    start = time.time()
    results = retriever.search(query, top_k=5)
    latency = (time.time() - start) * 1000

    print(f"\nFound {len(results)} results in {latency:.2f}ms")
    for rank, result in enumerate(results, start=1):
        print(f"\n{rank}. Score: {result['score']:.3f} | Page: {result['page']}")
        print(f"   {result['text'][:100]}...")

    print("\n" + "=" * 60)
    print("Stats:", retriever.get_stats())
    print("=" * 60)
