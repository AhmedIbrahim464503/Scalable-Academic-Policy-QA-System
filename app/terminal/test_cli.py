from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("rich")
pytest.importorskip("prompt_toolkit")

from rich.console import Console

from app.terminal.cli import RetrievalEngine, TerminalQAApp, main
from app.terminal.state import SessionState
from app.terminal.ui_components import highlight_text, render_help


class DummyEngine:
    def __init__(self) -> None:
        self.chunk_count = 3
        self.loaded_methods: list[str] = []

    def load_method(self, method: str) -> object:
        self.loaded_methods.append(method)
        if method != "baseline":
            raise NotImplementedError("not implemented")
        return object()

    def search(self, method: str, query: str, top_k: int = 5):  # noqa: ARG002
        from app.terminal.cli import SearchResponse

        return SearchResponse(
            chunks=[
                {"text": "Attendance below 75 percent triggers a warning.", "page": 12, "score": 0.92, "metadata": {}},
                {"text": "Students may withdraw within the published deadline.", "page": 18, "score": 0.81, "metadata": {}},
            ],
            latency=0.012,
            method=method,
            answer="Retrieval-only answer placeholder.",
        )


def make_app(tmp_path: Path) -> TerminalQAApp:
    app = TerminalQAApp(
        console=Console(record=True, width=120),
        project_root=tmp_path,
        chunks_path=tmp_path / "chunks.json",
        history_file=tmp_path / ".history.json",
    )
    app.engine = DummyEngine()
    return app


def test_session_state_round_trip(tmp_path: Path) -> None:
    state = SessionState(history_file=tmp_path / ".history.json")
    state.add_query("attendance", 0.1, 3)
    state.save_history()

    restored = SessionState(history_file=tmp_path / ".history.json")
    restored.load_history()
    assert restored.total_queries == 1
    assert restored.query_history[0]["query"] == "attendance"


def test_highlight_text_marks_query_terms() -> None:
    rendered = highlight_text("Attendance policy applies.", ["attendance"])
    assert "Attendance" in rendered.plain
    assert rendered.spans


def test_render_help_contains_commands() -> None:
    help_panel = render_help()
    console = Console(record=True, width=120)
    console.print(help_panel)
    assert "/help" in console.export_text()


def test_process_query_updates_state(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    response = app.process_query("attendance policy")
    assert response.method == "baseline"
    assert app.state.total_queries == 1
    assert app.state.last_results


def test_history_and_expand_commands_do_not_crash(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.process_query("attendance")
    assert app.handle_command("/history") is True
    assert app.handle_command("/expand 1") is True


def test_invalid_method_command_is_handled(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    assert app.handle_command("/method unsupported") is True


def test_export_creates_json(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.process_query("withdrawal")
    exported = app._export_results()
    assert exported is not None
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["query"] == "withdrawal"


def test_retrieval_engine_missing_chunks(tmp_path: Path) -> None:
    engine = RetrievalEngine(project_root=tmp_path, chunks_path=tmp_path / "missing.json")
    try:
        engine.load_method("baseline")
    except FileNotFoundError as exc:
        assert "chunks.json" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_click_single_query_failure_for_missing_backend(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--chunks-path",
            str(tmp_path / "missing.json"),
            "--query",
            "attendance policy",
            "--history-file",
            str(tmp_path / ".history.json"),
        ],
    )
    assert result.exit_code == 1
