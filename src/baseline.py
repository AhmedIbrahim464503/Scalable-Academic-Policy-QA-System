"""Compatibility wrapper for the terminal app's baseline interface.

The terminal CLI expects a ``TFIDFBaseline`` class exposing
``search(query, top_k)``. Phase 1 already implements weighted TF-IDF retrieval
in ``BaselineRetrievalAgent``, so this wrapper delegates to that implementation
when possible and keeps a small fallback for ad-hoc in-memory testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .baseline_retrieval import BaselineRetrievalAgent
except ImportError:
    BaselineRetrievalAgent = None


class TFIDFBaseline:
    """Baseline retriever with a stable CLI-facing API."""

    def __init__(
        self,
        chunks_path: str | Path | None = None,
        chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        self.chunks_path = Path(chunks_path) if chunks_path else None
        self.chunks = chunks or []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_matrix = None
        self.agent = None

        if BaselineRetrievalAgent is not None and self.chunks_path and not self.chunks:
            self.agent = BaselineRetrievalAgent(data_path=str(self.chunks_path))
            self.chunks = self.agent.data
        elif not self.chunks and self.chunks_path:
            self.load()
        elif self.chunks:
            self.build(self.chunks)

    def load(self) -> None:
        """Load chunks from disk and build the TF-IDF matrix."""
        if not self.chunks_path or not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {self.chunks_path}")

        data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("chunks.json must contain a list of chunk objects.")
        self.build(data)

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build internal retrieval state from chunk records."""
        self.chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
        texts = [str(chunk.get("text", "")).strip() for chunk in self.chunks]
        if not texts:
            raise ValueError("No chunk text found to index.")
        self.chunk_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the most similar chunks for a query."""
        if not query.strip():
            return []
        if self.agent is not None:
            results, _latency = self.agent.retrieve(query, top_k=top_k)
            return results
        if self.chunk_matrix is None:
            if self.chunks:
                self.build(self.chunks)
            else:
                self.load()

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            score = float(similarities[index])
            chunk = dict(self.chunks[index])
            chunk["score"] = score
            results.append(chunk)
        return results
