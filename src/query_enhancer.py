"""LLM-powered query enhancement using Multi-Query Retrieval.

This module uses the Groq LLM to generate multiple alternative versions of a 
user's question to improve the recall of the retrieval step.
"""

from __future__ import annotations

import os
import json
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class QueryEnhancer:
    """Generates alternative search queries to improve retrieval coverage."""

    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile") -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = model
        self.enabled = False
        self.enhancement_count = 0

        self.system_prompt = """You are a search expert for a university policy database.
Your job is to generate 2-3 alternative versions of a user's question to help find different parts of the handbook.

RULES:
1. Output ONLY a JSON list of strings. 
2. Each string should be a complete, distinct search query.
3. Focus on different aspects (e.g., if asking about 'attendance', one query should be about 'exam eligibility', another about 'shortage of attendance').
4. Do NOT include the original query in your list.
5. NO explanation, NO conversational text.

Example:
Input: "What is the minimum attendance?"
Output: ["mandatory attendance percentage for exams", "shortage of attendance consequences", "attendance policy requirements"]
"""

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                self.enabled = True
            except Exception:
                pass

    def generate_queries(self, query: str) -> list[str]:
        """Generate a list of alternative queries (including the original)."""
        queries = [query]
        if not self.enabled or not self.client:
            return queries

        self.enhancement_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f'Input: "{query}"'},
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"} if "70b" in self.model else None
            )
            content = response.choices[0].message.content.strip()
            
            # Extract JSON list (handle potential LLM formatting variations)
            if "[" in content and "]" in content:
                json_str = content[content.find("["):content.rfind("]")+1]
                alternatives = json.loads(json_str)
                if isinstance(alternatives, list):
                    queries.extend(alternatives[:3]) # Take top 3
            
            return queries
        except Exception:
            return queries
