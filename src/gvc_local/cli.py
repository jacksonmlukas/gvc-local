"""CLI entrypoint. Mirrors upstream's run.py interface so we can diff results cleanly.

Usage:
    gvc-local <solver> <model> --start 0 --end 10 \
        --base-url http://localhost:8000/v1

Solvers:  basic | cot | gvc | snap_gvc
Models:   llama-3.1-8b | llama-3.3-70b | qwen-2.5-7b
"""

from __future__ import annotations

import json
import sys

import click

from .endpoint import EndpointConfig

MODEL_MAP = {
    "llama-3.1-8b": EndpointConfig.llama31_8b,
    "llama-3.3-70b": EndpointConfig.llama33_70b,
    "qwen-2.5-7b": EndpointConfig.qwen25_7b,
}


@click.command()
@click.argument("solver", type=click.Choice(["basic", "cot", "gvc", "snap_gvc"]))
@click.argument("model", type=click.Choice(list(MODEL_MAP)))
@click.option("--start", type=int, default=0, help="First puzzle index (inclusive).")
@click.option("--end", type=int, default=10, help="Last puzzle index (exclusive).")
@click.option("--base-url", type=str, default="http://localhost:8000/v1", help="vLLM OpenAI-compatible endpoint.")
@click.option("--temperature", type=float, default=None)
@click.option("--out", type=click.Path(), default="results/run.jsonl", help="Write per-puzzle results here.")
@click.option("--dry-run", is_flag=True, help="Just print the plan and exit (use before spending GPU time).")
def main(
    solver: str,
    model: str,
    start: int,
    end: int,
    base_url: str,
    temperature: float | None,
    out: str,
    dry_run: bool,
) -> None:
    cfg = MODEL_MAP[model](base_url=base_url)
    if temperature is not None:
        cfg.default_temperature = temperature

    plan = {
        "solver": solver,
        "model": cfg.model,
        "endpoint": cfg.base_url,
        "temperature": cfg.default_temperature,
        "puzzles": f"[{start},{end})",
        "out": out,
    }
    click.echo(json.dumps(plan, indent=2))
    if dry_run:
        return

    # TODO (M1–M2): wire the solver to the endpoint.
    # - Import the upstream rsallms solver for `solver`
    # - Patch its completion function to use cfg.client().chat()
    # - Run over puzzles[start:end], write jsonl to `out`
    # Placeholder so imports resolve and smoke tests pass:
    click.echo("Solver dispatch not wired yet — see M1 in ROADMAP.md.", err=True)
    sys.exit(2)


if __name__ == "__main__":
    main()
