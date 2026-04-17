"""Command-line interface for running GVC/Snap-GVC solvers on Connections puzzles.

Usage::

    # Solve puzzles 0–9 with Snap-GVC on Llama 3.1 8B (local vLLM)
    gvc-local snap_gvc llama-3.1-8b --start 0 --end 10

    # Use Groq free-tier inference (no GPU needed)
    gvc-local snap_gvc llama-3.1-8b --provider groq

    # Use Together AI
    gvc-local gvc llama-3.1-8b --provider together

    # Save traces for fine-tuning
    gvc-local gvc llama-3.1-8b --provider groq --traces data/traces/

    # Point at a custom vLLM endpoint
    gvc-local snap_gvc qwen-2.5-7b --base-url http://gpu-box:8000/v1

    # Dry-run to verify config
    gvc-local snap_gvc llama-3.1-8b --dry-run
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from gvc_local.endpoint import Client, EndpointConfig
from gvc_local.game import load_games
from gvc_local.solvers.base import BaseSolver

logger = logging.getLogger("gvc_local.cli")

# ---------------------------------------------------------------------------
# Model / provider registry
# ---------------------------------------------------------------------------

PROVIDERS = ("local", "groq", "together")

# Maps (provider, friendly-model-name) → EndpointConfig factory method name.
# The "local" provider uses the base factory + a user-supplied --base-url.
MODEL_MAP: dict[str, dict[str, str]] = {
    "local": {
        "llama-3.1-8b": "llama31_8b",
        "llama-3.3-70b": "llama33_70b",
        "qwen-2.5-7b": "qwen25_7b",
    },
    "groq": {
        "llama-3.1-8b": "groq_llama8b",
        "llama-3.3-70b": "groq_llama70b",
        "qwen-2.5-7b": "groq_qwen7b",
    },
    "together": {
        "llama-3.1-8b": "together_llama8b",
        "llama-3.3-70b": "together_llama70b",
        "qwen-2.5-7b": "together_qwen7b",
    },
}

MODEL_CHOICES = list(MODEL_MAP["local"])


def _resolve_endpoint(model_key: str, provider: str, base_url: str | None) -> EndpointConfig:
    """Map a friendly model name + provider to an ``EndpointConfig``."""
    try:
        factory_name = MODEL_MAP[provider][model_key]
    except KeyError as exc:
        raise click.ClickException(
            f"Unknown provider/model combo: provider={provider!r}, model={model_key!r}"
        ) from exc
    factory = getattr(EndpointConfig, factory_name)
    # Only pass base_url for local provider (cloud providers set their own).
    if provider == "local" and base_url:
        return factory(base_url=base_url)
    return factory()


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def _build_solver(
    solver_name: str, client: Client, *, temperature: float | None = None
) -> BaseSolver:
    """Instantiate the requested solver backed by *client*."""
    from gvc_local.solvers.gvc import GVCSolver
    from gvc_local.solvers.snap_gvc import SnapGVCSolver

    kwargs = {}
    if temperature is not None:
        kwargs["guesser_temperature"] = temperature
        kwargs["validator_temperature"] = temperature

    if solver_name in ("basic", "cot"):
        # basic/cot use GVC with a single internal retry — no consensus loop.
        return GVCSolver(client, max_internal_retries=1, **kwargs)
    elif solver_name == "gvc":
        return GVCSolver(client, **kwargs)
    elif solver_name == "snap_gvc":
        return SnapGVCSolver(client, **kwargs)
    else:
        raise click.ClickException(f"Unknown solver: {solver_name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("solver", type=click.Choice(["basic", "cot", "gvc", "snap_gvc"]))
@click.argument("model", type=click.Choice(MODEL_CHOICES))
@click.option("--start", type=int, default=0, help="First puzzle index (inclusive).")
@click.option("--end", type=int, default=10, help="Last puzzle index (exclusive).")
@click.option(
    "--provider",
    type=click.Choice(PROVIDERS),
    default="local",
    help="Inference provider (local vLLM, groq, or together).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="Override endpoint URL (mainly for local vLLM). Cloud providers set this automatically.",
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Override guesser/validator temperature.",
)
@click.option(
    "--traces",
    type=click.Path(),
    default=None,
    help="Directory to write JSONL interaction traces (for fine-tuning).",
)
@click.option(
    "--out",
    type=click.Path(),
    default=None,
    help="Output JSONL path for results summary.",
)
@click.option(
    "--delay",
    type=float,
    default=0.0,
    help="Seconds to wait between puzzles (helps with cloud rate limits).",
)
@click.option("--dry-run", is_flag=True, help="Print config and exit.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(
    solver: str,
    model: str,
    start: int,
    end: int,
    provider: str,
    base_url: str | None,
    temperature: float | None,
    traces: str | None,
    out: str | None,
    delay: float,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run GVC or Snap-GVC solvers on NYT Connections puzzles."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    endpoint = _resolve_endpoint(model, provider, base_url)

    plan = {
        "solver": solver,
        "provider": provider,
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "puzzle_range": f"[{start}, {end})",
        "temperature": temperature or endpoint.default_temperature,
        "traces": traces,
        "output": out,
    }
    click.echo(json.dumps(plan, indent=2))

    if dry_run:
        return

    # Load puzzle data
    click.echo("Loading puzzles from GitHub data source ...")
    try:
        all_games = load_games()
    except Exception as exc:
        raise click.ClickException(f"Failed to load puzzles: {exc}") from exc

    if end > len(all_games):
        click.echo(
            f"Warning: requested end={end} but only {len(all_games)} puzzles available. "
            f"Clamping to {len(all_games)}.",
            err=True,
        )
        end = len(all_games)

    games = all_games[start:end]
    click.echo(f"Loaded {len(games)} puzzles (indices {start}–{end - 1}).\n")

    # Build solver
    client = endpoint.client()
    solver_instance = _build_solver(solver, client, temperature=temperature)

    # Traces directory
    trace_dir = Path(traces) if traces else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    # Run solver over puzzles
    out_path = Path(out) if out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_solved = 0
    wall_start = time.time()

    for idx, game in enumerate(games):
        puzzle_num = start + idx
        click.echo(f"--- Puzzle {puzzle_num} ({idx + 1}/{len(games)}) ---")

        trace_path = trace_dir / f"puzzle_{puzzle_num:04d}.jsonl" if trace_dir else None

        try:
            solved_cats = solver_instance.play(game, trace_path=trace_path)
        except Exception as exc:
            logger.error("Solver crashed on puzzle %d: %s", puzzle_num, exc)
            solved_cats = [False, False, False, False]

        is_solved = all(solved_cats)
        if is_solved:
            total_solved += 1

        n_correct = sum(solved_cats)
        click.echo(
            f"  Result: {'SOLVED' if is_solved else 'FAILED'}  "
            f"({n_correct}/4 categories)  "
            f"strikes={game.current_strikes}"
        )

        row = {
            "puzzle_id": puzzle_num,
            "solved": is_solved,
            "categories_solved": sum(solved_cats),
            "strikes": game.current_strikes,
        }
        results.append(row)

        # Append incrementally so partial results survive crashes
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")

        # Reset game state for next run (games are mutated in place)
        game.reset()

        # Rate-limit delay (cloud providers throttle aggressively)
        if delay > 0 and idx < len(games) - 1:
            time.sleep(delay)

    wall_elapsed = time.time() - wall_start

    # Summary
    n = len(results)
    solve_rate = total_solved / n if n else 0.0
    click.echo(f"\n{'=' * 50}")
    click.echo(f"  Solver:     {solver}")
    click.echo(f"  Model:      {endpoint.model}")
    click.echo(f"  Puzzles:    {n}")
    click.echo(f"  Solved:     {total_solved}/{n}  ({solve_rate:.1%})")
    click.echo(f"  Wall time:  {wall_elapsed:.1f}s")
    click.echo(f"{'=' * 50}")

    if out_path:
        click.echo(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
