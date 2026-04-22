"""LLM-powered query enhancement for improving retrieval accuracy.

Before the raw user query hits the retriever, this module optionally asks
the Groq LLM to expand it into richer search terms.  Example:

    Input:  "max credit hours semester"
    Output: "maximum number of credit hours allowed per semester course load limit"

The expansion adds synonyms and related academic terms so the TF-IDF
vectorizer and MinHash/SimHash shingles have more surface area to match
against handbook chunks.

This does NOT bypass retrieval — it only improves the *input* to retrieval.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class QueryEnhancer:
    """Expand user queries using a lightweight LLM call."""

    SYSTEM_PROMPT = """You are a query expansion assistant for a university handbook search system.

Your ONLY job: take the user's search query and expand it into a richer search string.

Rules:
1. Output ONLY the expanded search terms, nothing else
2. Add synonyms and related academic terms
3. Keep the original intent intact
4. Do NOT answer the question
5. Do NOT add explanations or formatting
6. Keep output under 40 words
7. Focus on terms that would appear in a university handbook

Example:
Input: "max credit hours semester"
Output: "maximum number of credit hours allowed per semester course load limit registration"

Example:
Input: "gpa requirement"
Output: "minimum GPA cumulative grade point average CGPA academic standing requirement threshold"

Example:
Input: "attendance policy"
Output: "attendance policy requirement minimum percentage classes lectures absence leave"
"""

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile") -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = model
        self.enabled = False
        self.enhancement_count = 0

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                self.enabled = True
            except Exception:
                pass

    def enhance(self, query: str) -> str:
        """Expand a query into richer search terms.

        Returns the enhanced query on success, or the original query if
        enhancement is disabled or the LLM call fails.
        """
        if not self.enabled or not self.client:
            return query

        if not query.strip():
            return query

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
                stream=False,
            )
            enhanced = response.choices[0].message.content.strip()
            if enhanced:
                self.enhancement_count += 1
                return enhanced
        except Exception:
            pass

        return query

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "enhancement_count": self.enhancement_count,
        }
