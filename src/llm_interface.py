"""Groq LLM integration for answer generation."""

from __future__ import annotations

import os
import re
from typing import Any, Generator

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GroqAnswerGenerator:
    """Generate cited handbook answers using Groq chat completions."""

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile") -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.total_tokens = 0
        self.query_count = 0

    def format_chunks(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a compact context block."""
        formatted: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            page = chunk.get("page_number") or chunk.get("page") or chunk.get("source_page") or "n/a"
            formatted.append(f"[Chunk {index}] (Page {page})")
            formatted.append(str(chunk.get("text", "")))
            formatted.append("---")
        return "\n".join(formatted)

    def generate_answer_stream(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> Generator[str | dict[str, Any], None, None]:
        """
        Generate an answer with streaming tokens.

        The generator yields string tokens during generation, then a final
        metadata dictionary when the stream completes.
        """
        self.query_count += 1
        if not chunks:
            yield "No relevant information found in the handbook."
            return

        formatted_chunks = self.format_chunks(chunks)
        system_prompt = """You are an academic policy assistant for NUST (National University of Sciences and Technology).

CRITICAL RULES:
1. Answer ONLY using the provided handbook chunks below
2. ALWAYS cite page numbers using format [Page X]
3. If information is in multiple chunks, cite all: [Pages 12, 15, 18]
4. If the answer is NOT in the chunks, say: "I don't have that information in the handbook."
5. Be concise and professional
6. Do not make assumptions or add information not in chunks"""

        user_message = f"""HANDBOOK CHUNKS:
{formatted_chunks}

USER QUESTION: {query}

Answer concisely using ONLY the information above, with page citations:"""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=500,
                stream=True,
            )

            full_answer = ""
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                token = choice.delta.content if choice and choice.delta else None
                if not token:
                    continue
                full_answer += token
                yield token

            citations = self._extract_citations(full_answer)
            validation = self._validate_citations(citations, chunks)
            yield {
                "complete": True,
                "citations": citations,
                "validation": validation,
                "full_answer": full_answer,
            }
        except Exception as exc:
            yield f"\n\nError: {exc}\n\nShowing retrieved chunks instead."

    def _extract_citations(self, answer: str) -> list[int]:
        """Extract page numbers from inline citations."""
        citations: list[int] = []
        citation_blocks = re.findall(r"\[(Pages? [^\]]+)\]", answer)
        for block in citation_blocks:
            citations.extend(int(page) for page in re.findall(r"\d+", block))
        return sorted(set(citations))

    def _validate_citations(self, citations: list[int], chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Check whether cited pages are present in retrieved chunks."""
        available_pages = {
            int(page)
            for chunk in chunks
            for page in [chunk.get("page_number") or chunk.get("page") or chunk.get("source_page")]
            if isinstance(page, int) or (isinstance(page, str) and page.isdigit())
        }
        valid = [page for page in citations if page in available_pages]
        hallucinated = [page for page in citations if page not in available_pages]
        return {
            "valid": len(hallucinated) == 0,
            "valid_citations": valid,
            "hallucinated_pages": hallucinated,
            "warning": (
                f"Cited pages {hallucinated} not in retrieved chunks" if hallucinated else None
            ),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return lightweight generator stats."""
        return {
            "query_count": self.query_count,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


if __name__ == "__main__":
    print("Testing Groq LLM Integration...")

    test_chunks = [
        {
            "chunk_id": "TEST_001",
            "text": "75% attendance is mandatory for students to appear in end semester exams.",
            "page_number": 16,
            "score": 0.95,
        },
        {
            "chunk_id": "TEST_002",
            "text": "Students with less than 75% attendance will not be allowed to sit for exams.",
            "page_number": 16,
            "score": 0.87,
        },
    ]

    llm = GroqAnswerGenerator()
    query = "What is the attendance policy?"

    print(f"\nQuery: {query}\n")
    print("Streaming answer:")
    print("-" * 60)

    for token in llm.generate_answer_stream(query, test_chunks):
        if isinstance(token, dict) and token.get("complete"):
            print("\n" + "-" * 60)
            print(f"Citations: {token['citations']}")
            if token["validation"]["warning"]:
                print(token["validation"]["warning"])
        else:
            print(token, end="", flush=True)
