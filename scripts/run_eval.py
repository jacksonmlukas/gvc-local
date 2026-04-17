#!/usr/bin/env python3
"""CLI entry point for running evaluations.

Supports both *Connections* puzzles (the default) and *GAIA* benchmark tasks
through a unified interface.

Usage examples::

    # Connections evaluation (basic solver, Llama 3.1 8B, puzzles 0–50)
    python scripts/run_eval.py --solver basic --model llama-3.1-8b \\
        --start 0 --end 50 --out results/basic_llama8b.jsonl

    # GAIA Level-1 evaluation
    python scripts/run_eval.py --benchmark gaia --solver basic \\
        --model llama-3.1-8b --out results/gaia_l1.jsonl

    # Disable W&B tracking
    python scripts/run_eval.py --solver cot --model qwen-2.5-7b --no-wandb

    # Load config from YAML
    python scripts/run_eval.py --config configs/eval.yaml
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml

# Ensure the project root is on sys.path when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from gvc_local.endpoint import EndpointConfig
from gvc_local.eval.harness import EvalConfig, run_evaluation, RunResult
from gvc_local.eval.gaia import GaiaEvaluator, load_gaia_tasks
from gvc_local.eval.tracking import ExperimentTracker
from gvc_local.eval_harness import summarize

logger = logging.getLogger("gvc_local.eval")

# ---------------------------------------------------------------------------
# Model registry (mirrors cli.py)
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "llama-3.1-8b": EndpointConfig.llama31_8b,
    "llama-3.3-70b": EndpointConfig.llama33_70b,
    "qwen-2.5-7b": EndpointConfig.qwen25_7b,
}


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def _make_connections_solver(
    solver_name: str,
    endpoint: EndpointConfig,
) -> callable:
    """Create a Connections puzzle solver callable.

    Returns a function ``game -> RunResult`` suitable for
    :func:`run_evaluation`.

    .. note::

       Actual solver dispatch is wired in M1/M2.  This stub exercises the
       full pipeline so the harness, tracking, and reporting layers can be
       validated end-to-end.
    """

    def _stub_solver(game: dict) -> RunResult:
        """Placeholder solver that always returns an unsolved result.

        Replace with real solver dispatch once the upstream rsallms solvers
        are integrated (see ROADMAP.md M1).
        """
        return RunResult(
            solved=False,
            guesses=0,
            out_of_board_guesses=0,
            puzzle_id=game.get("puzzle_id", 0),
            strata=game.get("category", "uncategorised"),
        )

    return _stub_solver


def _make_gaia_solver(
    solver_name: str,
    endpoint: EndpointConfig,
) -> callable:
    """Create a GAIA Q&A solver callable.

    Returns a function ``question -> answer`` suitable for
    :class:`GaiaEvaluator`.
    """
    client = endpoint.client()

    def _llm_solver(question: str) -> str:
        """Send the question to the LLM and return the response."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise AI assistant. Answer the question with "
                    "the most concise, accurate answer possible. If the answer "
                    "is a number, return just the number. If it is a name, "
                    "return just the name."
                ),
            },
            {"role": "user", "content": question},
        ]
        return client.chat(messages=messages, temperature=0.0)

    return _llm_solver


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str) -> dict:
    """Load evaluation config from a YAML file."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _build_eval_config(
    solver: str,
    model_key: str,
    start: int,
    end: int,
    output: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    n_bootstrap: int,
    per_stratum: int,
) -> EvalConfig:
    """Build an EvalConfig from CLI arguments."""
    model_id = MODEL_MAP[model_key]().model  # resolve to full HF model string
    return EvalConfig(
        solver=solver,
        model=model_id,
        puzzle_range=(start, end),
        n_bootstrap=n_bootstrap,
        per_stratum=per_stratum,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        output_path=Path(output),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--benchmark",
    type=click.Choice(["connections", "gaia"]),
    default="connections",
    help="Which benchmark to run.",
)
@click.option(
    "--solver",
    type=click.Choice(["basic", "cot", "gvc", "snap_gvc"]),
    default="basic",
    help="Solver strategy.",
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_MAP)),
    default="llama-3.1-8b",
    help="Model to evaluate.",
)
@click.option("--start", type=int, default=0, help="First puzzle index (inclusive).")
@click.option("--end", type=int, default=10, help="Last puzzle index (exclusive).")
@click.option(
    "--gaia-level",
    type=click.IntRange(1, 3),
    default=1,
    help="GAIA difficulty level (1-3).",
)
@click.option(
    "--gaia-max-tasks",
    type=int,
    default=None,
    help="Maximum number of GAIA tasks to evaluate.",
)
@click.option(
    "--base-url",
    type=str,
    default="http://localhost:8000/v1",
    help="vLLM OpenAI-compatible endpoint URL.",
)
@click.option(
    "--out",
    type=click.Path(),
    default="results/eval_results.jsonl",
    help="Output JSONL path for results.",
)
@click.option("--wandb-project", type=str, default="gvc-local-eval", help="W&B project name.")
@click.option("--wandb-run-name", type=str, default=None, help="W&B run name.")
@click.option("--no-wandb", is_flag=True, help="Disable W&B tracking.")
@click.option("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
@click.option("--per-stratum", type=int, default=5, help="Max samples per stratum.")
@click.option("--config", type=click.Path(exists=True), default=None, help="YAML config file (overrides CLI args).")
@click.option("--games-path", type=click.Path(exists=True), default=None, help="Path to games JSONL file (Connections).")
@click.option("--dry-run", is_flag=True, help="Print config and exit.")
@click.option("-v", "--verbose", is_flag=True, help="Enable DEBUG logging.")
def main(
    benchmark: str,
    solver: str,
    model: str,
    start: int,
    end: int,
    gaia_level: int,
    gaia_max_tasks: int | None,
    base_url: str,
    out: str,
    wandb_project: str,
    wandb_run_name: str | None,
    no_wandb: bool,
    n_bootstrap: int,
    per_stratum: int,
    config: str | None,
    games_path: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run evaluation on Connections puzzles or GAIA benchmark tasks."""
    # Logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # YAML overrides
    if config:
        cfg_data = _load_yaml_config(config)
        benchmark = cfg_data.get("benchmark", benchmark)
        solver = cfg_data.get("solver", solver)
        model = cfg_data.get("model", model)
        start = cfg_data.get("puzzle_start", start)
        end = cfg_data.get("puzzle_end", end)
        out = cfg_data.get("output_path", out)
        wandb_project = cfg_data.get("wandb", {}).get("project", wandb_project)
        wandb_run_name = cfg_data.get("wandb", {}).get("run_name", wandb_run_name)
        no_wandb = cfg_data.get("wandb", {}).get("disabled", no_wandb)
        n_bootstrap = cfg_data.get("n_bootstrap", n_bootstrap)
        per_stratum = cfg_data.get("per_stratum", per_stratum)
        gaia_level = cfg_data.get("gaia", {}).get("level", gaia_level)
        gaia_max_tasks = cfg_data.get("gaia", {}).get("max_tasks", gaia_max_tasks)
        base_url = cfg_data.get("base_url", base_url)

    effective_wandb = None if no_wandb else wandb_project

    # Print plan
    plan = {
        "benchmark": benchmark,
        "solver": solver,
        "model": model,
        "base_url": base_url,
        "wandb_project": effective_wandb,
        "output": out,
    }
    if benchmark == "connections":
        plan["puzzle_range"] = f"[{start}, {end})"
    else:
        plan["gaia_level"] = gaia_level
        plan["gaia_max_tasks"] = gaia_max_tasks

    click.echo(json.dumps(plan, indent=2))
    if dry_run:
        return

    # Endpoint
    endpoint_factory = MODEL_MAP[model]
    endpoint = endpoint_factory(base_url=base_url)

    # Dispatch by benchmark
    if benchmark == "connections":
        _run_connections(
            solver=solver,
            model=model,
            endpoint=endpoint,
            start=start,
            end=end,
            out=out,
            wandb_project=effective_wandb,
            wandb_run_name=wandb_run_name,
            n_bootstrap=n_bootstrap,
            per_stratum=per_stratum,
            games_path=games_path,
        )
    elif benchmark == "gaia":
        _run_gaia(
            solver=solver,
            model=model,
            endpoint=endpoint,
            level=gaia_level,
            max_tasks=gaia_max_tasks,
            out=out,
            wandb_project=effective_wandb,
            wandb_run_name=wandb_run_name,
            n_bootstrap=n_bootstrap,
        )


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _run_connections(
    solver: str,
    model: str,
    endpoint: EndpointConfig,
    start: int,
    end: int,
    out: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    n_bootstrap: int,
    per_stratum: int,
    games_path: str | None,
) -> None:
    """Run evaluation on NYT Connections puzzles."""
    # Load games
    games: list[dict] = []
    if games_path:
        with open(games_path) as fh:
            for line in fh:
                games.append(json.loads(line))
        click.echo(f"Loaded {len(games)} games from {games_path}")
    else:
        # Generate placeholder games for pipeline testing
        click.echo(
            "No --games-path provided. Using placeholder games for pipeline validation."
        )
        games = [
            {"puzzle_id": i, "category": "placeholder", "board": []}
            for i in range(max(end, 100))
        ]

    model_id = MODEL_MAP[model]().model
    eval_config = EvalConfig(
        solver=solver,
        model=model_id,
        puzzle_range=(start, end),
        n_bootstrap=n_bootstrap,
        per_stratum=per_stratum,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        output_path=Path(out),
    )

    solver_fn = _make_connections_solver(solver, endpoint)
    results = run_evaluation(eval_config, solver_fn, games)
    click.echo(f"\nEvaluation complete. {len(results)} puzzles evaluated.")


def _run_gaia(
    solver: str,
    model: str,
    endpoint: EndpointConfig,
    level: int,
    max_tasks: int | None,
    out: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    n_bootstrap: int,
) -> None:
    """Run evaluation on GAIA benchmark tasks."""
    tasks = load_gaia_tasks(level=level, max_tasks=max_tasks)
    if not tasks:
        click.echo("No GAIA tasks loaded. Check dataset access.", err=True)
        sys.exit(1)

    solver_fn = _make_gaia_solver(solver, endpoint)
    evaluator = GaiaEvaluator(solver=solver_fn)
    results = evaluator.evaluate(tasks)

    # Unified reporting through the same pipeline
    metrics = summarize(results)

    # W&B tracking
    tracker = ExperimentTracker(enabled=wandb_project is not None)
    if wandb_project:
        run_name = wandb_run_name or f"gaia-l{level}-{solver}-{model}"
        tracker.init(
            project=wandb_project,
            run_name=run_name,
            config={
                "benchmark": "gaia",
                "gaia_level": level,
                "solver": solver,
                "model": model,
                "n_tasks": len(tasks),
            },
            tags=["gaia", f"level-{level}", "eval"],
        )
        for r in results:
            tracker.log_puzzle(r.puzzle_id, r)
        tracker.log_summary(metrics)
        tracker.log_table(results)
        tracker.finish()

    # Save results
    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in results:
            fh.write(json.dumps({
                "puzzle_id": r.puzzle_id,
                "solved": r.solved,
                "guesses": r.guesses,
                "out_of_board_guesses": r.out_of_board_guesses,
                "strata": r.strata,
            }) + "\n")

    click.echo(f"\nGAIA evaluation complete. {len(results)} tasks evaluated.")
    click.echo(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
