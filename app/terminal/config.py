"""Configuration values for the terminal QA interface."""

from __future__ import annotations

DEFAULT_TOP_K = 5
MAX_PREVIEW_LENGTH = 150
HISTORY_FILE = ".nust_qa_history"
MAX_HISTORY_SIZE = 100

COLOR_SCHEME = {
    "header": "bold green",
    "accent": "bold yellow",
    "score": "yellow",
    "page": "cyan",
    "user_input": "white",
    "system": "dim white",
    "success": "green",
    "error": "bold red",
    "muted": "grey70",
    "border": "green",
}

