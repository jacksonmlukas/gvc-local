#!/usr/bin/env python3
"""CLI script to build the RAG index from puzzle data and solve traces.

Usage
-----
    # Puzzles only (minimum viable index):
    python scripts/build_rag_index.py \\
        --puzzles data/puzzles.json \\
        --output  data/rag_index

    # Puzzles + traces:
    python scripts/build_rag_index.py \\
        --puzzles data/puzzles.json \\
        --traces  data/traces.jsonl \\
        --output  data/rag_index

    # With non-default options:
    python scripts/build_rag_index.py \\
        --puzzles data/puzzles.json \\
        --output  data/rag_index \\
        --no-ivf \\
        --batch-size 128

The puzzles file should be a JSON array matching the Eyefyre/NYT-Connections-Answers
schema (each element has an ``"answers"`` list of ``{level, group, members}``).

The traces file is optional and can be either:
- A JSON array of trace objects, or
- A JSONL file (one JSON object per line), detected by ``.jsonl`` extension.
"""

from __future__ import annotations

import logging
import sys

# Ensure the project root is importable when running as a standalone script.
# When installed via pip/setuptools this is unnecessary, but during development
# you may run ``python scripts/build_rag_index.py`` directly.
from pathlib import Path

import click

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gvc_local.rag.indexer import IndexConfig, build_index  # noqa: E402


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--puzzles",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to puzzles JSON file (Eyefyre schema).",
)
@click.option(
    "--traces",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional path to solve-traces file (JSON or JSONL).",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write the FAISS index, metadata, and config.",
)
@click.option(
    "--batch-size",
    default=64,
    type=int,
    show_default=True,
    help="Embedding batch size (tune for your GPU/RAM).",
)
@click.option(
    "--no-ivf",
    is_flag=True,
    default=False,
    help="Force a flat index instead of IVF (slower search, exact results).",
)
@click.option(
    "--nprobe",
    default=8,
    type=int,
    show_default=True,
    help="Number of IVF cells to visit at query time.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def main(
    puzzles: str,
    traces: str | None,
    output: str,
    batch_size: int,
    no_ivf: bool,
    nprobe: int,
    verbose: bool,
) -> None:
    """Build the RAG index for gvc-local puzzle-solving agents."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config = IndexConfig(
        batch_size=batch_size,
        use_ivf=not no_ivf,
        nprobe=nprobe,
    )

    click.echo("Building RAG index ...")
    click.echo(f"  Puzzles : {puzzles}")
    click.echo(f"  Traces  : {traces or '(none)'}")
    click.echo(f"  Output  : {output}")
    click.echo(f"  IVF     : {'yes' if config.use_ivf else 'no (flat)'}")
    click.echo(f"  Batch   : {config.batch_size}")

    out_path = build_index(
        puzzles_path=puzzles,
        output_dir=output,
        traces_path=traces,
        config=config,
    )

    click.echo(f"Done. Index written to {out_path}")


if __name__ == "__main__":
    main()
