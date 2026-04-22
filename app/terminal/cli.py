"""Production-style terminal QA interface.

This module uses:
- Rich for structured terminal rendering
- prompt_toolkit for the input experience
- Click for the command entrypoint
"""

from __future__ import annotations

import importlib
import inspect
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from prompt_toolkit import HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style
from pygments.lexers.markup import MarkdownLexer
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.terminal.config import COLOR_SCHEME, DEFAULT_TOP_K, HISTORY_FILE
from app.terminal.state import SessionState
from app.terminal.ui_components import (
    create_progress_spinner,
    render_chunk_preview,
    render_header,
    render_help,
    render_query_result,
)
from src.llm_interface import GroqAnswerGenerator
from src.lsh_minhash import MinHashRetriever
from src.lsh_simhash import SimHashRetriever

APP_VERSION = "0.1.0"
SUPPORTED_METHODS = ("baseline", "minhash", "simhash")
MINHASH_DEFAULTS = {"num_perm": 64, "threshold": 0.3, "shingle_size": 3}
SIMHASH_DEFAULTS = {"fingerprint_size": 64, "hamming_threshold": 24}

DEBUG_HELP_TEXT = """[bold cyan]Debug Mode Commands[/bold cyan]

[yellow]/inspect <rank>[/yellow]
  Show detailed analysis of a retrieved chunk

[yellow]/multicompare[/yellow]
  Compare current query across all methods with overlap analysis

[yellow]/params [method][/yellow]
  Show parameter analysis and tuning guidance

[yellow]/export[/yellow]
  Export current debug session to JSON

All regular commands still work in debug mode.
"""

try:
    from app.terminal.debug_ui import (
        compare_retrieval_results,
        export_debug_session,
        show_chunk_details,
        show_parameter_analysis,
    )
except Exception:
    compare_retrieval_results = None
    export_debug_session = None
    show_chunk_details = None
    show_parameter_analysis = None


@dataclass
class SearchResponse:
    chunks: list[dict[str, Any]]
    latency: float
    method: str
    answer: str
    citations: list[int] | None = None
    llm_latency: float = 0.0
    llm_validation: dict[str, Any] | None = None


class RetrievalEngine:
    """Loads retrieval models with caching and normalizes search output."""

    def __init__(self, project_root: Path, chunks_path: Path) -> None:
        self.project_root = project_root
        self.chunks_path = chunks_path
        self._cache: dict[str, Any] = {}
        self._chunks_cache: list[dict[str, Any]] | None = None
        self.chunk_count = 0

    def _load_chunks(self) -> list[dict[str, Any]]:
        if self._chunks_cache is not None:
            return self._chunks_cache
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found at {self.chunks_path}. Add data/processed/chunks.json first."
            )
        data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("chunks.json must contain a top-level list.")
        self.chunk_count = len(data)
        self._chunks_cache = [item for item in data if isinstance(item, dict)]
        return self._chunks_cache

    def _load_baseline_class(self) -> type[Any]:
        try:
            module = importlib.import_module("src.baseline")
        except ModuleNotFoundError as exc:
            if exc.name == "src" or exc.name == "src.baseline":
                raise FileNotFoundError(
                    "src/baseline.py is missing or importable package metadata is incomplete."
                ) from exc
            raise ModuleNotFoundError(
                f"Missing dependency while importing src.baseline: {exc.name}"
            ) from exc

        baseline_cls = getattr(module, "TFIDFBaseline", None)
        if baseline_cls is None:
            raise AttributeError("TFIDFBaseline class not found in src.baseline.")
        return baseline_cls

    def _instantiate_baseline(self, baseline_cls: type[Any], chunks: list[dict[str, Any]]) -> Any:
        """Try common constructor shapes because the baseline signature may evolve."""
        candidate_kwargs = [
            {"chunks_path": str(self.chunks_path)},
            {"data_path": str(self.chunks_path)},
            {"chunks": chunks},
            {},
        ]
        candidate_args = [
            (str(self.chunks_path),),
            (chunks,),
            tuple(),
        ]

        for kwargs in candidate_kwargs:
            try:
                return baseline_cls(**kwargs)
            except TypeError:
                continue

        for args in candidate_args:
            try:
                return baseline_cls(*args)
            except TypeError:
                continue

        raise TypeError("Unable to initialize TFIDFBaseline with supported constructor patterns.")

    def load_method(self, method: str) -> Any:
        if method in self._cache:
            return self._cache[method]

        chunks = self._load_chunks()

        if method == "baseline":
            baseline_cls = self._load_baseline_class()
            model = self._instantiate_baseline(baseline_cls, chunks)

            # If the baseline exposes a load/build hook, run it once during startup.
            for hook_name in ("load", "build", "fit", "initialize"):
                hook = getattr(model, hook_name, None)
                if callable(hook):
                    signature = inspect.signature(hook)
                    try:
                        if len(signature.parameters) == 0:
                            hook()
                        elif len(signature.parameters) == 1:
                            hook(chunks)
                    except Exception:
                        pass
                    break
        elif method == "minhash":
            model = MinHashRetriever(**MINHASH_DEFAULTS)
            model.create_index(chunks)
        elif method == "simhash":
            model = SimHashRetriever(**SIMHASH_DEFAULTS)
            model.create_index(chunks)
        else:
            raise NotImplementedError(f"Method '{method}' is not supported.")

        self._cache[method] = model
        return model

    def search(self, method: str, query: str, top_k: int = DEFAULT_TOP_K) -> SearchResponse:
        model = self.load_method(method)
        start = time.perf_counter()
        raw_results = model.search(query, top_k=top_k)
        latency = time.perf_counter() - start
        chunks = self._normalize_results(raw_results)
        answer = self._build_placeholder_answer(query, chunks)
        return SearchResponse(chunks=chunks, latency=latency, method=method, answer=answer)

    def _normalize_results(self, raw_results: Any) -> list[dict[str, Any]]:
        if raw_results is None:
            return []

        normalized: list[dict[str, Any]] = []
        for item in list(raw_results):
            chunk = self._normalize_single_result(item)
            if chunk:
                normalized.append(chunk)
        return normalized

    def _normalize_single_result(self, item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            text = item.get("text") or item.get("chunk") or item.get("content") or item.get("preview")
            page = item.get("page") or item.get("page_number") or item.get("source_page") or "n/a"
            score = item.get("score") or item.get("similarity") or item.get("distance") or 0.0
            metadata = item.get("metadata", {})
            if page == "n/a" and isinstance(metadata, dict):
                page = metadata.get("page", "n/a")
            return {
                "text": str(text or ""),
                "page": page,
                "score": float(score),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }

        if isinstance(item, (list, tuple)) and item:
            if len(item) >= 2 and isinstance(item[0], (int, float)):
                return {"text": str(item[1]), "page": "n/a", "score": float(item[0]), "metadata": {}}
            if len(item) >= 2 and isinstance(item[1], (int, float)):
                return {"text": str(item[0]), "page": "n/a", "score": float(item[1]), "metadata": {}}
            return {"text": str(item[0]), "page": "n/a", "score": 0.0, "metadata": {}}

        if hasattr(item, "__dict__"):
            payload = vars(item)
            return self._normalize_single_result(payload)

        return None

    def _build_placeholder_answer(self, query: str, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return f"No supporting chunks were retrieved for '{query}'."

        top_chunk = chunks[0]
        preview = str(top_chunk.get("text", "")).strip().replace("\n", " ")
        preview = preview[:220]
        return (
            "Answer generation is currently in retrieval-only mode. "
            f"The most relevant handbook evidence appears to be from page {top_chunk.get('page', 'n/a')}: "
            f"{preview}"
        )


class TerminalQAApp:
    """Interactive terminal application with Claude Code-inspired polish."""

    def __init__(
        self,
        console: Console | None = None,
        project_root: Path | None = None,
        chunks_path: Path | None = None,
        top_k: int = DEFAULT_TOP_K,
        history_file: Path | None = None,
        debug: bool = False,
    ) -> None:
        self.console = console or Console()
        self.project_root = project_root or Path.cwd()
        self.chunks_path = chunks_path or self.project_root / "data" / "processed" / "chunks.json"
        self.top_k = top_k
        self.debug = debug
        self.state = SessionState(history_file=history_file or self.project_root / HISTORY_FILE)
        self.state.load_history()
        self.engine = RetrievalEngine(project_root=self.project_root, chunks_path=self.chunks_path)
        self.llm: GroqAnswerGenerator | None = None
        self.llm_enabled = False
        self.running = True
        self.multiline_mode = False

        self.common_queries = [
            "/help",
            "/history",
            "/stats",
            "/methods",
            "/compare",
            "/export",
            "/clear",
            "/quit",
            "What is the attendance policy?",
            "How many credit hours are required for graduation?",
            "What are the probation rules?",
            "How does course withdrawal work?",
        ]
        self.session: PromptSession[str] | None = None

    def _create_prompt_session(self) -> PromptSession[str]:
        return PromptSession(
            history=FileHistory(str(self.state.history_file.with_suffix(".prompt"))),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(self.common_queries, ignore_case=True, sentence=True),
            multiline=False,
            lexer=PygmentsLexer(MarkdownLexer),
            style=Style.from_dict(
                {
                    "prompt": "ansigreen bold",
                    "bottom-toolbar": "bg:#1b1f23 #d7d7af",
                }
            ),
            key_bindings=self._build_key_bindings(),
            bottom_toolbar=self._bottom_toolbar,
        )

    def _build_key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        # Submit on Enter so the shell behaves like a normal command interface.
        @bindings.add("enter", filter=~completion_is_selected)
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        # Keep multiline support available on demand without making the default UX awkward.
        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-l")
        def _clear(_: Any) -> None:
            self.console.clear()
            self.console.print(self.header_panel)

        @bindings.add("f1")
        def _help(_: Any) -> None:
            self.console.print(render_help())

        @bindings.add("c-d")
        def _exit(event: Any) -> None:
            self.running = False
            event.app.exit(result="/exit")

        return bindings

    @property
    def header_panel(self) -> Panel:
        return render_header(
            {
                "chunks": self.engine.chunk_count or "not loaded",
                "model": self.state.current_method,
                "version": APP_VERSION,
            }
        )

    def _bottom_toolbar(self) -> HTML:
        clock = datetime.now().strftime("%H:%M:%S")
        return HTML(
            f"<b> model </b>{self.state.current_method}"
            f"    <b> chunks </b>{self.engine.chunk_count or 'n/a'}"
            f"    <b> time </b>{clock}"
        )

    def initialize(self) -> None:
        self.console.print(self.header_panel)
        self.console.print(
            Panel(
                Text("Loading retrieval methods...", style=COLOR_SCHEME["accent"]),
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )

        load_plan = [
            ("baseline", "Loading TF-IDF baseline..."),
            ("minhash", "Building MinHash index..."),
            ("simhash", "Building SimHash index..."),
        ]
        for method_name, message in load_plan:
            spinner = create_progress_spinner(message)
            with spinner:
                task = spinner.add_task(method_name, total=None)
                self.engine.load_method(method_name)
                spinner.update(task, description=f"{method_name} ready")
            self.console.print(
                Panel(
                    Text(f"{method_name} ready", style=COLOR_SCHEME["success"]),
                    border_style=COLOR_SCHEME["border"],
                    box=ROUNDED,
                )
            )

        self.console.print(
            Panel(
                Text("Initializing LLM...", style=COLOR_SCHEME["accent"]),
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )
        try:
            self.llm = GroqAnswerGenerator()
            self.llm_enabled = True
            self.console.print(
                Panel(
                    Text("LLM ready (Groq: llama-3.3-70b)", style=COLOR_SCHEME["success"]),
                    border_style=COLOR_SCHEME["border"],
                    box=ROUNDED,
                )
            )
        except Exception as exc:
            self.llm = None
            self.llm_enabled = False
            self.console.print(
                Panel(
                    Text(f"LLM initialization failed: {exc}", style="red"),
                    border_style="red",
                    box=ROUNDED,
                )
            )
            self.console.print(
                Panel(
                    Text("Running in retrieval-only mode", style="yellow"),
                    border_style="yellow",
                    box=ROUNDED,
                )
            )

        self.console.print(
            Panel(
                Text("System ready. Type /help for commands.", style=COLOR_SCHEME["system"]),
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )
        self.console.print(self.header_panel)

    def run(self) -> None:
        try:
            self.initialize()
            self.session = self._create_prompt_session()
        except Exception as exc:
            self.render_error(exc, "Startup failed", "Verify src/baseline.py and data/processed/chunks.json.")
            return

        while self.running:
            try:
                with patch_stdout():
                    user_input = self.session.prompt(
                        HTML(f"<prompt>{'nust-qa [DEBUG]' if self.debug else 'nust-qa'}</prompt> ❯ "),
                        prompt_continuation="... ",
                    )
            except KeyboardInterrupt:
                self.console.print(
                    Panel(
                        "Input cancelled. Press Ctrl+D or use /exit to quit.",
                        border_style="red",
                        box=ROUNDED,
                    )
                )
                continue
            except EOFError:
                break

            command = (user_input or "").strip()
            if not command:
                continue

            try:
                if command.startswith("/"):
                    self.running = self.handle_command(command)
                else:
                    self.process_query(command)
            except KeyboardInterrupt:
                self.console.print(
                    Panel(
                        "Query interrupted safely.",
                        border_style="red",
                        box=ROUNDED,
                    )
                )
            except Exception as exc:
                self.render_error(exc, "Runtime error", "Adjust the query or inspect the retrieval backend.")

        self.shutdown()

    def process_query(self, query: str) -> SearchResponse:
        with self.console.status("[dim]Searching handbook index...[/dim]", spinner="dots"):
            response = self.engine.search(self.state.current_method, query, self.top_k)

        if self.llm_enabled and self.llm is not None and response.chunks:
            answer_text = ""
            citations: list[int] = []
            validation: dict[str, Any] | None = None
            completed = False
            llm_start = time.perf_counter()
            with Live(
                Panel(
                    Markdown(""),
                    title="[bold green]Generating[/bold green]",
                    border_style=COLOR_SCHEME["border"],
                    box=ROUNDED,
                ),
                console=self.console,
                refresh_per_second=20,
            ) as live:
                for token in self.llm.generate_answer_stream(query, response.chunks):
                    if isinstance(token, dict) and token.get("complete"):
                        citations = token.get("citations", [])
                        validation = token.get("validation")
                        answer_text = token.get("full_answer", answer_text)
                        completed = True
                        break
                    answer_text += str(token)
                    live.update(
                        Panel(
                            Markdown(answer_text or " "),
                            title="[bold green]Generating[/bold green]",
                            border_style=COLOR_SCHEME["border"],
                            box=ROUNDED,
                        )
                    )
            response.answer = (
                answer_text
                if completed
                else f"{self.engine._build_placeholder_answer(query, response.chunks)}\n\n{answer_text.strip()}"
            )
            response.citations = citations
            response.llm_validation = validation
            response.llm_latency = time.perf_counter() - llm_start

        self.state.add_query(query, response.latency, len(response.chunks))
        self.state.last_results = response.chunks
        self.state.last_query = query
        self.state.save_history()

        self.console.print(
            render_query_result(
                query=query,
                chunks=response.chunks,
                metrics={
                    "latency": response.latency,
                    "llm_latency": response.llm_latency,
                    "method": response.method,
                    "answer": response.answer,
                    "citations": response.citations or [],
                },
            )
        )
        if response.llm_validation and response.llm_validation.get("warning"):
            self.console.print(
                Panel(
                    str(response.llm_validation["warning"]),
                    border_style="red",
                    box=ROUNDED,
                )
            )
        return response

    def handle_command(self, raw_command: str) -> bool:
        name, *args = raw_command.split()
        command = name.lower()

        if command == "/help":
            self.console.print(render_help())
            return True

        if command == "/method":
            return self._handle_method_command(args)

        if command == "/methods":
            self._render_methods()
            return True

        if command == "/compare":
            self._compare_methods()
            return True

        if self.debug and command == "/inspect":
            self._inspect_result(args)
            return True

        if self.debug and command == "/multicompare":
            self._multi_compare_debug()
            return True

        if self.debug and command == "/params":
            self._show_params(args)
            return True

        if command == "/debug-help":
            self._render_debug_help()
            return True

        if command == "/history":
            self._render_history()
            return True

        if command == "/export":
            self._export_results()
            return True

        if command == "/stats":
            self._render_stats()
            return True

        if command == "/clear":
            self.console.clear()
            self.console.print(self.header_panel)
            return True

        if command in {"/quit", "/exit"}:
            return False

        if command == "/expand":
            self._expand_result(args)
            return True

        self.console.print(
            Panel(
                f"Unknown command: {raw_command}",
                subtitle="Use /help to see supported commands.",
                border_style="red",
                box=ROUNDED,
            )
        )
        return True

    def _handle_method_command(self, args: list[str]) -> bool:
        if not args:
            self.console.print(
                Panel(
                    f"Current method: {self.state.current_method}",
                    subtitle=f"Usage: /method {'|'.join(SUPPORTED_METHODS)}",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return True

        requested = args[0].lower()
        if requested not in SUPPORTED_METHODS:
            self.console.print(
                Panel(
                    f"Unsupported method '{requested}'.",
                    subtitle=f"Choose from: {', '.join(SUPPORTED_METHODS)}",
                    border_style="red",
                    box=ROUNDED,
                )
            )
            return True

        try:
            self.engine.load_method(requested)
        except NotImplementedError as exc:
            self.render_error(exc, "Method unavailable", "Only baseline is implemented at the moment.")
            return True
        except Exception as exc:
            self.render_error(exc, "Method switch failed", "Verify the retrieval backend is installed correctly.")
            return True

        self.state.current_method = requested
        self.console.print(
            Panel(
                f"Retrieval method set to {requested}.",
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )
        return True

    def _render_methods(self) -> None:
        table = Table(box=ROUNDED, title="Retrieval Methods")
        table.add_column("Active", style=COLOR_SCHEME["accent"], width=8)
        table.add_column("Method", style=COLOR_SCHEME["header"])
        table.add_column("Status", style=COLOR_SCHEME["system"])

        for method in SUPPORTED_METHODS:
            loaded = "loaded" if method in self.engine._cache else "ready"
            marker = "->" if method == self.state.current_method else ""
            table.add_row(marker, method, loaded)
        self.console.print(table)

    def _inspect_result(self, args: list[str]) -> None:
        if not self.state.last_results:
            self.console.print(
                Panel(
                    "No results to inspect. Run a query first.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return
        if not args:
            self.console.print(Panel("Usage: /inspect <rank>", border_style="yellow", box=ROUNDED))
            return
        try:
            rank = int(args[0])
        except ValueError:
            self.console.print(Panel("Usage: /inspect <rank>", border_style="red", box=ROUNDED))
            return
        if rank < 1 or rank > len(self.state.last_results):
            self.console.print(
                Panel(
                    f"Rank must be 1-{len(self.state.last_results)}.",
                    border_style="red",
                    box=ROUNDED,
                )
            )
            return
        if show_chunk_details is None:
            self.console.print(Panel("Debug UI is not available.", border_style="red", box=ROUNDED))
            return
        show_chunk_details(self.state.last_results[rank - 1], rank)

    def _multi_compare_debug(self) -> None:
        if not self.state.last_query:
            self.console.print(
                Panel(
                    "No query to compare. Ask something first.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return
        if compare_retrieval_results is None:
            self.console.print(Panel("Debug UI is not available.", border_style="red", box=ROUNDED))
            return
        all_results: dict[str, list[dict[str, Any]]] = {}
        with self.console.status("[dim]Running query on all methods...[/dim]", spinner="dots"):
            for method in SUPPORTED_METHODS:
                all_results[method] = self.engine.search(method, self.state.last_query, self.top_k).chunks
        compare_retrieval_results(self.state.last_query, all_results)

    def _show_params(self, args: list[str]) -> None:
        method = args[0].lower() if args else self.state.current_method
        if show_parameter_analysis is None:
            self.console.print(Panel("Debug UI is not available.", border_style="red", box=ROUNDED))
            return
        if method == "minhash":
            show_parameter_analysis("minhash", MINHASH_DEFAULTS)
            return
        if method == "simhash":
            show_parameter_analysis("simhash", SIMHASH_DEFAULTS)
            return
        if method == "baseline":
            show_parameter_analysis("baseline", {})
            return
        self.console.print(
            Panel(
                f"Unknown method: {method}",
                subtitle=f"Choose from: {', '.join(SUPPORTED_METHODS)}",
                border_style="red",
                box=ROUNDED,
            )
        )

    def _render_debug_help(self) -> None:
        if not self.debug:
            self.console.print(
                Panel(
                    "Debug mode is off. Start with --debug to enable extra analysis commands.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return
        self.console.print(Panel(DEBUG_HELP_TEXT, border_style="cyan", box=ROUNDED, title="Debug Commands"))

    def _compare_methods(self) -> None:
        if not self.state.last_query:
            self.console.print(
                Panel(
                    "No previous query to compare.",
                    subtitle="Run a query first, then use /compare.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return

        comparison_rows: list[tuple[str, SearchResponse]] = []
        with self.console.status("[dim]Comparing all retrieval methods...[/dim]", spinner="dots"):
            for method in SUPPORTED_METHODS:
                response = self.engine.search(method, self.state.last_query, self.top_k)
                if self.llm_enabled and self.llm is not None and response.chunks:
                    llm_start = time.perf_counter()
                    answer_text = ""
                    for token in self.llm.generate_answer_stream(self.state.last_query, response.chunks):
                        if isinstance(token, dict) and token.get("complete"):
                            answer_text = token.get("full_answer", answer_text)
                            break
                        answer_text += str(token)
                    response.answer = answer_text
                    response.llm_latency = time.perf_counter() - llm_start
                comparison_rows.append((method, response))

        table = Table(box=ROUNDED, title=f"Comparison: {self.state.last_query}")
        table.add_column("Method", style=COLOR_SCHEME["header"])
        table.add_column("Retrieval", style=COLOR_SCHEME["score"])
        if self.llm_enabled:
            table.add_column("LLM", style="magenta")
            table.add_column("Total", style=COLOR_SCHEME["success"])
        else:
            table.add_column("Total", style=COLOR_SCHEME["success"])
        table.add_column("Results", style=COLOR_SCHEME["success"])
        table.add_column("Top Page", style=COLOR_SCHEME["page"])
        table.add_column("Top Score", style=COLOR_SCHEME["accent"])

        for method, response in comparison_rows:
            top_page = response.chunks[0]["page"] if response.chunks else "n/a"
            top_score = f"{float(response.chunks[0]['score']):.4f}" if response.chunks else "n/a"
            if self.llm_enabled:
                table.add_row(
                    method,
                    f"{response.latency * 1000:.1f} ms",
                    f"{response.llm_latency * 1000:.0f} ms",
                    f"{(response.latency + response.llm_latency) * 1000:.0f} ms",
                    str(len(response.chunks)),
                    str(top_page),
                    top_score,
                )
            else:
                table.add_row(
                    method,
                    f"{response.latency * 1000:.1f} ms",
                    f"{response.latency * 1000:.0f} ms",
                    str(len(response.chunks)),
                    str(top_page),
                    top_score,
                )

        self.console.print(table)

    def _render_history(self) -> None:
        table = Table(box=ROUNDED, title="Query History")
        table.add_column("#", style=COLOR_SCHEME["header"], width=4)
        table.add_column("Query", style=COLOR_SCHEME["user_input"])
        table.add_column("Latency", style=COLOR_SCHEME["score"], width=12)
        table.add_column("Results", style=COLOR_SCHEME["page"], width=8)

        for index, item in enumerate(reversed(self.state.query_history[-10:]), start=1):
            table.add_row(
                str(index),
                str(item.get("query", "")),
                f"{float(item.get('latency', 0.0)) * 1000:.1f} ms",
                str(item.get("result_count", 0)),
            )
        self.console.print(table)

    def _render_stats(self) -> None:
        stats = self.state.get_stats()
        body = Text.assemble(
            ("current method: ", COLOR_SCHEME["system"]),
            (str(stats["current_method"]), COLOR_SCHEME["page"]),
            ("\ntotal queries: ", COLOR_SCHEME["system"]),
            (str(stats["total_queries"]), COLOR_SCHEME["accent"]),
            ("\navg latency: ", COLOR_SCHEME["system"]),
            (f"{float(stats['avg_latency']) * 1000:.1f} ms", COLOR_SCHEME["score"]),
            ("\nchunks loaded: ", COLOR_SCHEME["system"]),
            (str(self.engine.chunk_count or "n/a"), COLOR_SCHEME["success"]),
        )
        self.console.print(Panel(body, title="System Stats", border_style=COLOR_SCHEME["border"], box=ROUNDED))

    def _expand_result(self, args: list[str]) -> None:
        if not self.state.last_results:
            self.console.print(
                Panel(
                    "No search results available to expand.",
                    subtitle="Run a query first.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return

        if not args:
            self.console.print(
                Panel(
                    "Usage: /expand <rank>",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return

        try:
            rank = int(args[0])
        except ValueError:
            self.console.print(Panel("Rank must be a number.", border_style="red", box=ROUNDED))
            return

        if rank < 1 or rank > len(self.state.last_results):
            self.console.print(
                Panel(
                    f"Rank out of range. Choose 1-{len(self.state.last_results)}.",
                    border_style="red",
                    box=ROUNDED,
                )
            )
            return

        self.console.print(
            render_chunk_preview(
                self.state.last_results[rank - 1],
                rank=rank,
                query_terms=self.state.last_query.split(),
            )
        )

    def _export_results(self) -> Path | None:
        if not self.state.last_results:
            self.console.print(
                Panel(
                    "No results to export.",
                    subtitle="Run a query before exporting.",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            return None

        export_dir = self.project_root / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = export_dir / f"nust_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "query": self.state.last_query,
            "method": self.state.current_method,
            "results": self.state.last_results,
            "exported_at": datetime.now().isoformat(),
        }
        filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.console.print(
            Panel(
                f"Exported current results to {filename}",
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )
        if self.debug and export_debug_session is not None:
            export_debug_session(
                self.state.last_query,
                {
                    "method": self.state.current_method,
                    "results": self.state.last_results,
                },
                filename=f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )
        return filename

    def render_error(self, exc: Exception, title: str, suggestion: str) -> None:
        self.console.print(
            Panel(
                f"{type(exc).__name__}: {exc}\n\nSuggestion: {suggestion}",
                title=f"[bold red]{title}[/bold red]",
                border_style="red",
                box=ROUNDED,
            )
        )

    def shutdown(self) -> None:
        self.state.save_history()
        self.console.print(
            Panel(
                "Session closed gracefully.",
                border_style=COLOR_SCHEME["border"],
                box=ROUNDED,
            )
        )


@click.command()
@click.option("--chunks-path", default="data/processed/chunks.json", show_default=True, type=click.Path(path_type=Path))
@click.option("--top-k", default=DEFAULT_TOP_K, show_default=True, type=int)
@click.option("--query", default=None, help="Run a single query and exit.")
@click.option("--history-file", default=HISTORY_FILE, show_default=True, type=click.Path(path_type=Path))
@click.option("--debug", is_flag=True, help="Enable debug mode with extra analysis commands.")
def main(chunks_path: Path, top_k: int, query: str | None, history_file: Path, debug: bool) -> None:
    """Launch the NUST QA terminal interface."""
    app = TerminalQAApp(
        project_root=Path.cwd(),
        chunks_path=Path.cwd() / chunks_path,
        top_k=top_k,
        history_file=Path.cwd() / history_file,
        debug=debug,
    )

    if query:
        try:
            app.initialize()
            app.process_query(query)
            app.shutdown()
        except Exception as exc:
            app.render_error(exc, "Query failed", "Verify the baseline and chunk data are available.")
            raise SystemExit(1) from exc
        return

    app.run()


if __name__ == "__main__":
    main()
