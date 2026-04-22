"""Reusable Rich UI components for the terminal QA interface."""

from __future__ import annotations

import re
from typing import Iterable

from rich import box
from rich.align import Align
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from app.terminal.config import COLOR_SCHEME, MAX_PREVIEW_LENGTH


def render_header(system_info: dict[str, str | int] | None = None) -> Panel:
    """Render the startup header.

    Rich usage:
    - ``Text`` builds styled ASCII art line by line.
    - ``Group`` stacks multiple renderables inside one panel.
    - ``Panel`` gives the Claude Code style boxed shell chrome.
    """

    logo = Text(
        "\n".join(
            [
                " _   _ _   _ ____ _____   ___   _    ",
                "| \\ | | | | / ___|_   _| / _ \\ / \\   ",
                "|  \\| | | | \\___ \\ | |  | | | / _ \\  ",
                "| |\\  | |_| |___) || |  | |_| / ___ \\ ",
                "|_| \\_|\\___/|____/ |_|   \\__\\/_/   \\_\\",
            ]
        ),
        style=COLOR_SCHEME["header"],
    )
    subtitle = Text("Academic Policy QA Terminal", style=COLOR_SCHEME["system"])

    renderables: list[object] = [Align.center(logo), Align.center(subtitle)]
    if system_info:
        info = Text.assemble(
            ("chunks ", COLOR_SCHEME["system"]),
            (str(system_info.get("chunks", "n/a")), COLOR_SCHEME["accent"]),
            ("   model ", COLOR_SCHEME["system"]),
            (str(system_info.get("model", "not loaded")), COLOR_SCHEME["success"]),
            ("   version ", COLOR_SCHEME["system"]),
            (str(system_info.get("version", "0.1.0")), COLOR_SCHEME["page"]),
        )
        renderables.append(Align.center(info))

    return Panel(
        Group(*renderables),
        border_style=COLOR_SCHEME["border"],
        box=box.ROUNDED,
        padding=(1, 2),
        title="[bold yellow]NUST QA[/bold yellow]",
    )


def highlight_text(text: str, query_terms: Iterable[str]) -> Text:
    """Highlight matching terms inside a chunk preview with Rich spans."""
    rich_text = Text(text, style=COLOR_SCHEME["user_input"])
    for term in sorted({t.strip().lower() for t in query_terms if t.strip()}, key=len, reverse=True):
        for match in re.finditer(re.escape(term), text.lower()):
            rich_text.stylize("bold black on yellow", match.start(), match.end())
    return rich_text


def render_chunk_preview(chunk: dict[str, object], rank: int, query_terms: Iterable[str] | None = None) -> Panel:
    """Render a single source chunk in an expanded form."""
    preview_text = str(chunk.get("text", ""))
    highlighted = highlight_text(preview_text, query_terms or [])
    title = Text.assemble(
        (f"Source #{rank}", COLOR_SCHEME["header"]),
        ("  page ", COLOR_SCHEME["system"]),
        (str(chunk.get("page", "n/a")), COLOR_SCHEME["page"]),
        ("  score ", COLOR_SCHEME["system"]),
        (f"{float(chunk.get('score', 0.0)):.4f}", COLOR_SCHEME["score"]),
    )
    return Panel(
        highlighted,
        border_style=COLOR_SCHEME["border"],
        title=title,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_metrics(latency: float, method: str, count: int) -> Panel:
    """Render the metrics footer.

    Rich usage:
    - ``Text.assemble`` avoids manual ANSI formatting.
    - ``Panel`` keeps the footer visually separated from content.
    """

    metrics = Text.assemble(
        ("query time ", COLOR_SCHEME["system"]),
        (f"{latency * 1000:.1f} ms", COLOR_SCHEME["accent"]),
        ("   results ", COLOR_SCHEME["system"]),
        (str(count), COLOR_SCHEME["success"]),
        ("   method ", COLOR_SCHEME["system"]),
        (method, COLOR_SCHEME["page"]),
    )
    return Panel(metrics, border_style=COLOR_SCHEME["border"], box=box.MINIMAL)


def render_help() -> Panel:
    """Render the help screen with commands and keyboard shortcuts."""
    command_table = Table(box=box.SIMPLE_HEAVY, expand=True, show_header=True)
    command_table.add_column("Command", style=COLOR_SCHEME["header"], no_wrap=True)
    command_table.add_column("Description", style=COLOR_SCHEME["system"])
    command_table.add_row("/help", "Show this help screen.")
    command_table.add_row("/method [baseline|minhash|simhash]", "Switch retrieval mode.")
    command_table.add_row("/methods", "List loaded retrieval methods.")
    command_table.add_row("/compare", "Compare all methods on the last query.")
    command_table.add_row("/debug-help", "Show extra commands available in debug mode.")
    command_table.add_row("/history", "Show recent query history.")
    command_table.add_row("/export", "Export current results to JSON.")
    command_table.add_row("/stats", "Show session statistics.")
    command_table.add_row("/clear", "Clear the terminal and redraw header.")
    command_table.add_row("/quit or /exit", "Exit the application.")
    command_table.add_row("/expand [rank]", "Expand a result chunk after a search.")

    shortcuts = Markdown(
        "\n".join(
            [
                "### Keyboard Shortcuts",
                "- `Ctrl+L`: clear screen",
                "- `Ctrl+D`: exit",
                "- `F1`: show help",
                "- `Up/Down`: command history",
                "- `Esc` then `Enter`: submit multi-line input",
            ]
        )
    )
    return Panel(
        Group(command_table, shortcuts),
        title="[bold green]Help[/bold green]",
        border_style=COLOR_SCHEME["border"],
        box=box.ROUNDED,
        padding=(1, 2),
    )


def create_progress_spinner(message: str) -> Progress:
    """Create a Rich spinner for loading and query processing states.

    Rich usage:
    - ``Progress`` with ``SpinnerColumn`` gives a non-blocking animated status.
    - ``TextColumn`` lets us keep brand-consistent styling without raw escapes.
    """

    return Progress(
        SpinnerColumn(style=COLOR_SCHEME["accent"]),
        TextColumn(f"[{COLOR_SCHEME['system']}]{message}"),
        transient=True,
    )


def render_query_result(query: str, chunks: list[dict[str, object]], metrics: dict[str, object]) -> Group:
    """Render the full search result view with answer, sources, and footer."""
    answer_body = Markdown(
        str(
            metrics.get(
                "answer",
                "LLM answer generation is not enabled yet. Showing retrieved handbook evidence only.",
            )
        )
    )
    answer_panel = Panel(
        answer_body,
        title=f"[bold green]Answer[/bold green]  [dim]Query:[/dim] {query}",
        border_style=COLOR_SCHEME["border"],
        box=box.ROUNDED,
        padding=(1, 2),
        subtitle=(
            f"[cyan]Cited Pages:[/cyan] {', '.join(map(str, metrics.get('citations', [])))}"
            if metrics.get("citations")
            else None
        ),
    )

    table = Table(box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
    table.add_column("Rank", style=COLOR_SCHEME["header"], width=6)
    table.add_column("Score", style=COLOR_SCHEME["score"], width=10)
    table.add_column("Page", style=COLOR_SCHEME["page"], width=8)
    table.add_column("Preview", style=COLOR_SCHEME["user_input"], overflow="fold")

    query_terms = query.split()
    for index, chunk in enumerate(chunks, start=1):
        preview = str(chunk.get("text", "")).strip().replace("\n", " ")
        preview = preview[:MAX_PREVIEW_LENGTH]
        preview_text = highlight_text(preview, query_terms)
        if len(str(chunk.get("text", ""))) > MAX_PREVIEW_LENGTH:
            preview_text.append("...")
        table.add_row(
            str(index),
            f"{float(chunk.get('score', 0.0)):.4f}",
            str(chunk.get("page", "n/a")),
            preview_text,
        )

    sources_panel = Panel(
        table,
        title="[bold green]Sources[/bold green]",
        subtitle="[dim]Use /expand <rank> to inspect full chunk text[/dim]",
        border_style=COLOR_SCHEME["border"],
        box=box.ROUNDED,
        padding=(1, 1),
    )

    metric_text = Text.assemble(
        ("query time ", COLOR_SCHEME["system"]),
        (f"{float(metrics.get('latency', 0.0)) * 1000:.1f} ms", COLOR_SCHEME["accent"]),
        ("   ", COLOR_SCHEME["system"]),
    )
    llm_latency = float(metrics.get("llm_latency", 0.0))
    if llm_latency > 0:
        metric_text.append("llm time ", style=COLOR_SCHEME["system"])
        metric_text.append(f"{llm_latency * 1000:.0f} ms", style="magenta")
        metric_text.append("   ", style=COLOR_SCHEME["system"])
    metric_text.append("results ", style=COLOR_SCHEME["system"])
    metric_text.append(str(len(chunks)), style=COLOR_SCHEME["success"])
    metric_text.append("   method ", style=COLOR_SCHEME["system"])
    metric_text.append(str(metrics.get("method", "baseline")), style=COLOR_SCHEME["page"])
    footer = Panel(
        metric_text,
        border_style=COLOR_SCHEME["border"],
        box=box.MINIMAL,
    )
    return Group(answer_panel, sources_panel, footer)
