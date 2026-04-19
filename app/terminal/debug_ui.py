"""Debug UI helpers for analyzing retrieval behavior in the terminal."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "n/a")


def _page_number(chunk: dict[str, Any]) -> str:
    return str(chunk.get("page") or chunk.get("page_number") or "n/a")


def show_chunk_details(chunk: dict[str, Any], rank: int) -> None:
    """Show full details for one retrieved chunk."""
    console.print(
        Panel(
            f"""[bold]Chunk {rank} Details[/bold]

[cyan]Chunk ID:[/cyan] {_chunk_id(chunk)}
[cyan]Page:[/cyan] {_page_number(chunk)}
[cyan]Score:[/cyan] {float(chunk.get('score', 0.0)):.6f}
[cyan]Method:[/cyan] {chunk.get('method', 'N/A')}

[cyan]Full Text:[/cyan]
{chunk.get('text', '')}

[cyan]Text Length:[/cyan] {len(str(chunk.get('text', '')))} characters
[cyan]Word Count:[/cyan] {len(str(chunk.get('text', '')).split())} words
            """,
            border_style="cyan",
            title=f"Chunk Analysis - Rank {rank}",
        )
    )


def compare_retrieval_results(query: str, results: dict[str, list[dict[str, Any]]]) -> None:
    """Compare all retrieval methods side by side for a single query."""
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="Multi-Method Comparison"))

    stats_table = Table(title="Retrieval Statistics")
    stats_table.add_column("Method", style="cyan")
    stats_table.add_column("Results", style="yellow")
    stats_table.add_column("Avg Score", style="green")
    stats_table.add_column("Top Score", style="magenta")
    stats_table.add_column("Top Page", style="blue")

    for method_name, chunks in results.items():
        if chunks:
            avg_score = sum(float(chunk.get("score", 0.0)) for chunk in chunks) / len(chunks)
            top_score = float(chunks[0].get("score", 0.0))
            top_page = _page_number(chunks[0])
        else:
            avg_score = 0.0
            top_score = 0.0
            top_page = "N/A"

        stats_table.add_row(
            method_name,
            str(len(chunks)),
            f"{avg_score:.4f}",
            f"{top_score:.4f}",
            top_page,
        )

    console.print(stats_table)

    baseline_ids = {_chunk_id(chunk) for chunk in results.get("baseline", [])}
    minhash_ids = {_chunk_id(chunk) for chunk in results.get("minhash", [])}
    simhash_ids = {_chunk_id(chunk) for chunk in results.get("simhash", [])}

    overlap_table = Table(title="Chunk ID Overlap")
    overlap_table.add_column("Comparison")
    overlap_table.add_column("Common Chunks")
    overlap_table.add_column("Overlap %")

    if baseline_ids and minhash_ids:
        common = baseline_ids & minhash_ids
        overlap_table.add_row("Baseline ∩ MinHash", str(len(common)), f"{(len(common) / len(baseline_ids)) * 100:.0f}%")
    if baseline_ids and simhash_ids:
        common = baseline_ids & simhash_ids
        overlap_table.add_row("Baseline ∩ SimHash", str(len(common)), f"{(len(common) / len(baseline_ids)) * 100:.0f}%")
    if minhash_ids and simhash_ids:
        common = minhash_ids & simhash_ids
        overlap_table.add_row("MinHash ∩ SimHash", str(len(common)), f"{(len(common) / max(len(minhash_ids), 1)) * 100:.0f}%")

    console.print(overlap_table)

    console.print("\n[bold]Side-by-Side Top 3 Results:[/bold]\n")
    for rank in range(3):
        console.print(f"[bold yellow]═══ Rank {rank + 1} ═══[/bold yellow]")
        columns = []
        for method_name in ["baseline", "minhash", "simhash"]:
            chunks = results.get(method_name, [])
            if rank < len(chunks):
                chunk = chunks[rank]
                columns.append(
                    Panel(
                        f"""[cyan]Page:[/cyan] {_page_number(chunk)}
[cyan]Score:[/cyan] {float(chunk.get('score', 0.0)):.4f}
[cyan]ID:[/cyan] {_chunk_id(chunk)}

[dim]{str(chunk.get('text', ''))[:150]}...[/dim]
                        """,
                        title=f"[bold]{method_name}[/bold]",
                        border_style="green" if method_name == "baseline" else "yellow",
                    )
                )
            else:
                columns.append(Panel("[dim]No result[/dim]", title=method_name))
        console.print(Columns(columns))
        console.print()


def show_parameter_analysis(method: str, params: dict[str, Any]) -> None:
    """Show tuning guidance for the selected retrieval method."""
    if method == "minhash":
        console.print(
            Panel(
                f"""[bold cyan]MinHash LSH Parameters[/bold cyan]

[yellow]Current Settings:[/yellow]
- num_perm = {params.get('num_perm', 'N/A')}
- threshold = {params.get('threshold', 'N/A')}
- shingle_size = {params.get('shingle_size', 'N/A')}

[yellow]Parameter Impact:[/yellow]

[cyan]num_perm (hash functions):[/cyan]
  Low (32-64):   Fast indexing, less accurate
  Medium (128):  Balanced
  High (256+):   Slower, more stable

[cyan]threshold (similarity cutoff):[/cyan]
  Low (0.1-0.3):  More results, more false positives
  Medium (0.4-0.5): Balanced
  High (0.6+):    Stricter, fewer results

[cyan]shingle_size (n-gram size):[/cyan]
  Low (1-2):   Flexible but noisy
  Medium (3):  Better sentence context
  High (4+):   Strict phrase matching
                """,
                border_style="cyan",
            )
        )
    elif method == "simhash":
        console.print(
            Panel(
                f"""[bold cyan]SimHash Parameters[/bold cyan]

[yellow]Current Settings:[/yellow]
- fingerprint_size = {params.get('fingerprint_size', 'N/A')} bits
- hamming_threshold = {params.get('hamming_threshold', 'N/A')} bits

[yellow]Parameter Impact:[/yellow]

[cyan]hamming_threshold:[/cyan]
  0-3 bits:   Near-duplicates only
  4-10 bits:  Strict similarity
  11-20 bits: Moderate similarity
  20+ bits:   Loose matching

[cyan]fingerprint_size:[/cyan]
  64 bits:  Standard footprint
  128 bits: More precision, more memory
                """,
                border_style="magenta",
            )
        )
    else:
        console.print(Panel("Baseline has no tunable parameters.", border_style="yellow"))


def export_debug_session(query: str, results: dict[str, Any], filename: str = "debug_session.json") -> Path:
    """Export a debug session snapshot to the results directory."""
    output_path = Path("data/results") / filename
    debug_data = {
        "query": query,
        "timestamp": time.time(),
        "results": results,
    }
    output_path.write_text(json.dumps(debug_data, indent=2), encoding="utf-8")
    console.print(f"[green]✓ Debug session exported to {output_path}[/green]")
    return output_path


def show_accuracy_matrix(test_results: list[dict[str, Any]]) -> None:
    """Show summary overlap metrics against baseline."""
    console.print(Panel("[bold]Accuracy Analysis[/bold]", border_style="cyan"))

    minhash_overlaps: list[float] = []
    simhash_overlaps: list[float] = []

    for result in test_results:
        baseline_ids = {_chunk_id(chunk) for chunk in result.get("baseline_results", [])}
        minhash_ids = {_chunk_id(chunk) for chunk in result.get("minhash_results", [])}
        simhash_ids = {_chunk_id(chunk) for chunk in result.get("simhash_results", [])}
        if baseline_ids:
            minhash_overlaps.append(len(baseline_ids & minhash_ids) / len(baseline_ids))
            simhash_overlaps.append(len(baseline_ids & simhash_ids) / len(baseline_ids))

    metrics_table = Table(title="Accuracy Metrics (vs Baseline)")
    metrics_table.add_column("Method")
    metrics_table.add_column("Avg Overlap")
    metrics_table.add_column("Min Overlap")
    metrics_table.add_column("Max Overlap")
    metrics_table.add_column("Std Dev")

    if minhash_overlaps:
        metrics_table.add_row(
            "MinHash",
            f"{np.mean(minhash_overlaps):.1%}",
            f"{np.min(minhash_overlaps):.1%}",
            f"{np.max(minhash_overlaps):.1%}",
            f"{np.std(minhash_overlaps):.3f}",
        )
    if simhash_overlaps:
        metrics_table.add_row(
            "SimHash",
            f"{np.mean(simhash_overlaps):.1%}",
            f"{np.min(simhash_overlaps):.1%}",
            f"{np.max(simhash_overlaps):.1%}",
            f"{np.std(simhash_overlaps):.3f}",
        )

    console.print(metrics_table)

