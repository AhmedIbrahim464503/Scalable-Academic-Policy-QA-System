"""Session state helpers for the terminal QA interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.terminal.config import HISTORY_FILE, MAX_HISTORY_SIZE


class SessionState:
    """Tracks user session metadata across a terminal run."""

    def __init__(
        self,
        history_file: str | Path = HISTORY_FILE,
        current_method: str = "baseline",
    ) -> None:
        self.history_file = Path(history_file)
        self.query_history: list[dict[str, Any]] = []
        self.current_method = current_method
        self.total_queries = 0
        self.avg_latency = 0.0
        self.last_results: list[dict[str, Any]] = []
        self.last_query = ""

    def add_query(self, query: str, latency: float, result_count: int = 0) -> None:
        """Append a query record and update aggregate latency."""
        self.total_queries += 1
        if self.total_queries == 1:
            self.avg_latency = latency
        else:
            cumulative = self.avg_latency * (self.total_queries - 1)
            self.avg_latency = (cumulative + latency) / self.total_queries

        record = {
            "query": query,
            "latency": latency,
            "result_count": result_count,
        }
        self.query_history.append(record)
        if len(self.query_history) > MAX_HISTORY_SIZE:
            self.query_history = self.query_history[-MAX_HISTORY_SIZE:]
        self.last_query = query

    def get_stats(self) -> dict[str, Any]:
        """Return serializable session statistics."""
        return {
            "current_method": self.current_method,
            "total_queries": self.total_queries,
            "avg_latency": self.avg_latency,
            "history_size": len(self.query_history),
            "last_query": self.last_query,
        }

    def save_history(self) -> None:
        """Persist history to disk for prompt-toolkit reuse and commands."""
        self.history_file.write_text(
            json.dumps(self.query_history, indent=2),
            encoding="utf-8",
        )

    def load_history(self) -> None:
        """Load persisted history if present; ignore malformed files safely."""
        if not self.history_file.exists():
            return

        try:
            data = json.loads(self.history_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, list):
            return

        self.query_history = [item for item in data if isinstance(item, dict)][
            -MAX_HISTORY_SIZE:
        ]
        self.total_queries = len(self.query_history)
        if self.query_history:
            latencies = [
                float(item.get("latency", 0.0))
                for item in self.query_history
                if isinstance(item.get("latency", 0.0), (int, float))
            ]
            if latencies:
                self.avg_latency = sum(latencies) / len(latencies)
            self.last_query = str(self.query_history[-1].get("query", ""))

