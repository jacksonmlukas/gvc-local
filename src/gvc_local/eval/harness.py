"""Full evaluation harness orchestrating solver runs, stratified analysis, and reporting.

This module ties together the statistical primitives in :mod:`eval_harness`
(``RunResult``, ``stratified_sample``, ``summarize``) with the W&B tracking
layer, console pretty-printing, and JSONL persistence.  It is the main
programmatic entry point for running evaluations -- the CLI
(``scripts/run_eval.py``) is a thin wrapper around :func:`run_evaluation`.

Design
------
* **EvalConfig** carries all tunables for a single evaluation run.
* **run_evaluation** orchestrates the full pipeline: invoke the solver on each
  game, collect ``RunResult`` objects, compute stratified statistics, log to
  W&B, pretty-print Table-1-style output, and persist raw results.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..eval_harness import RunResult, stratified_sample, summarize
from .tracking import ExperimentTracker

try:
    import wandb as _wandb  # type: ignore[import-untyped]
except ImportError:
    _wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver protocol
# ---------------------------------------------------------------------------


class SolverCallable(Protocol):
    """Protocol describing any callable that can solve a game/puzzle.

    Implementations receive the game object (typically a dict with board state)
    and return a :class:`RunResult`.
    """

    def __call__(self, game: dict[str, Any]) -> RunResult: ...


# ---------------------------------------------------------------------------
# Puzzle categorisation
# ---------------------------------------------------------------------------

# Default strata used for stratified analysis.  These mirror the category
# taxonomy from Pandian et al.  Solvers or datasets may override this via
# game metadata.
DEFAULT_STRATA_FIELD = "category"

_FALLBACK_STRATA = "uncategorised"


def categorise_puzzle(game: dict[str, Any], strata_field: str = DEFAULT_STRATA_FIELD) -> str:
    """Extract the stratum label from a game dict.

    Falls back to ``"uncategorised"`` if the field is absent.
    """
    return str(game.get(strata_field, _FALLBACK_STRATA))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run.

    Attributes
    ----------
    solver : str
        Name of the solver strategy (e.g. ``"basic"``, ``"cot"``, ``"gvc"``,
        ``"snap_gvc"``).
    model : str
        Model identifier matching vLLM model strings.
    puzzle_range : tuple[int, int]
        ``(start, end)`` slice of puzzle indices to evaluate.
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    per_stratum : int
        Max puzzles per stratum in stratified sampling.
    wandb_project : str | None
        W&B project.  ``None`` disables tracking.
    wandb_run_name : str | None
        Optional explicit run name for W&B.
    wandb_tags : list[str]
        Tags applied to the W&B run.
    output_path : Path
        Path to write the JSONL results file.
    strata_field : str
        Key in game dicts holding the stratum label.
    """

    solver: str = "basic"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    puzzle_range: tuple[int, int] = (0, 10)
    n_bootstrap: int = 1000
    per_stratum: int = 5
    wandb_project: str | None = "gvc-local-eval"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["eval", "gvc-local"])
    output_path: Path = Path("results/eval_results.jsonl")
    strata_field: str = DEFAULT_STRATA_FIELD


# ---------------------------------------------------------------------------
# Console table formatting
# ---------------------------------------------------------------------------

_TABLE_HEADER = "| {metric:<25s} | {mean:>8s} | {ci_lower:>8s} | {ci_upper:>8s} |"
_TABLE_ROW = "| {metric:<25s} | {mean:>8.4f} | {ci_lower:>8.4f} | {ci_upper:>8.4f} |"
_TABLE_SEP = "+" + "-" * 27 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+"


def _pretty_print_summary(
    metrics: dict[str, tuple[float, float, float]],
    solver: str,
    model: str,
    n_puzzles: int,
) -> str:
    """Format metrics into a Table-1-style ASCII table and return as string.

    The table is also printed to stdout.
    """
    lines: list[str] = []
    lines.append("")
    lines.append(f"  Evaluation Summary: {solver} / {model}  (n={n_puzzles})")
    lines.append(_TABLE_SEP)
    lines.append(
        _TABLE_HEADER.format(metric="Metric", mean="Mean", ci_lower="CI Low", ci_upper="CI High")
    )
    lines.append(_TABLE_SEP)

    # Friendly display names
    display_names = {
        "solve_rate": "Solve Rate",
        "semantic_grounding": "OOB Guesses (avg)",
        "guesses_per_puzzle": "Guesses / Puzzle",
    }

    for key, (mean, lower, upper) in metrics.items():
        lines.append(
            _TABLE_ROW.format(
                metric=display_names.get(key, key),
                mean=mean,
                ci_lower=lower,
                ci_upper=upper,
            )
        )
    lines.append(_TABLE_SEP)

    table = "\n".join(lines)
    print(table)
    return table


def _pretty_print_per_stratum(
    results: Sequence[RunResult],
) -> str:
    """Print per-stratum breakdown of solve rates."""
    by_stratum: dict[str, list[RunResult]] = {}
    for r in results:
        by_stratum.setdefault(r.strata, []).append(r)

    lines: list[str] = ["\n  Per-Stratum Breakdown:"]
    lines.append(f"  {'Stratum':<25s} {'n':>5s} {'Solve%':>8s}")
    lines.append("  " + "-" * 40)

    for stratum in sorted(by_stratum):
        items = by_stratum[stratum]
        n = len(items)
        pct = sum(1 for r in items if r.solved) / n if n else 0.0
        lines.append(f"  {stratum:<25s} {n:>5d} {pct:>8.1%}")

    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def _save_results_jsonl(results: Sequence[RunResult], path: Path) -> Path:
    """Write results to a JSONL file. Returns the resolved path."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    logger.info("Saved %d results to %s", len(results), path)
    return path


def load_results_jsonl(path: str | Path) -> list[RunResult]:
    """Load RunResults from a JSONL file written by :func:`_save_results_jsonl`.

    Useful for re-analysis without re-running the solver.
    """
    results: list[RunResult] = []
    with open(path) as fh:
        for line in fh:
            data = json.loads(line)
            results.append(RunResult(**data))
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_evaluation(
    config: EvalConfig,
    solver: SolverCallable,
    games: Sequence[dict[str, Any]],
) -> list[RunResult]:
    """Run a complete evaluation pipeline.

    Parameters
    ----------
    config : EvalConfig
        Evaluation parameters.
    solver : SolverCallable
        Callable that accepts a game dict and returns a RunResult.
    games : sequence of dict
        Game / puzzle data.  Only ``config.puzzle_range`` slice is used.

    Returns
    -------
    list[RunResult]
        One result per evaluated puzzle.

    Side Effects
    ------------
    * Prints a Table-1-style summary to stdout.
    * Writes raw results as JSONL to ``config.output_path``.
    * Logs to W&B if ``config.wandb_project`` is set.
    """
    start, end = config.puzzle_range
    subset = list(games[start:end])
    logger.info(
        "Starting evaluation: solver=%s model=%s puzzles=[%d,%d) (%d games)",
        config.solver,
        config.model,
        start,
        end,
        len(subset),
    )

    # ------------------------------------------------------------------
    # Tracking setup
    # ------------------------------------------------------------------
    tracker = ExperimentTracker(enabled=config.wandb_project is not None)
    if config.wandb_project:
        run_name = config.wandb_run_name or f"{config.solver}-{config.model.split('/')[-1]}"
        tracker.init(
            project=config.wandb_project,
            run_name=run_name,
            config={
                "solver": config.solver,
                "model": config.model,
                "puzzle_start": start,
                "puzzle_end": end,
                "n_bootstrap": config.n_bootstrap,
                "per_stratum": config.per_stratum,
            },
            tags=config.wandb_tags,
        )

    # ------------------------------------------------------------------
    # Run solver over games
    # ------------------------------------------------------------------
    results: list[RunResult] = []
    wall_start = time.monotonic()

    for idx, game in enumerate(subset):
        puzzle_id = game.get("puzzle_id", start + idx)
        logger.info("Solving puzzle %s (%d/%d) ...", puzzle_id, idx + 1, len(subset))

        try:
            result = solver(game)
            # Ensure strata is populated
            if not result.strata or result.strata == "":
                result.strata = categorise_puzzle(game, config.strata_field)
            results.append(result)
            tracker.log_puzzle(puzzle_id, result)
        except Exception:
            logger.exception("Solver failed on puzzle %s — recording as unsolved.", puzzle_id)
            fallback = RunResult(
                solved=False,
                guesses=0,
                out_of_board_guesses=0,
                puzzle_id=puzzle_id,
                strata=categorise_puzzle(game, config.strata_field),
            )
            results.append(fallback)
            tracker.log_puzzle(puzzle_id, fallback)

    wall_elapsed = time.monotonic() - wall_start
    logger.info("Solver finished in %.1f s.", wall_elapsed)

    # ------------------------------------------------------------------
    # Stratified sampling & statistics
    # ------------------------------------------------------------------
    if config.per_stratum > 0 and len(results) > 0:
        sampled = stratified_sample(results, per_stratum=config.per_stratum)
    else:
        sampled = list(results)

    metrics = summarize(sampled)

    # ------------------------------------------------------------------
    # Console reporting
    # ------------------------------------------------------------------
    _pretty_print_summary(metrics, config.solver, config.model, len(sampled))
    _pretty_print_per_stratum(results)

    # ------------------------------------------------------------------
    # Persistence & tracking
    # ------------------------------------------------------------------
    saved_path = _save_results_jsonl(results, config.output_path)
    logger.info("Results written to %s", saved_path)

    tracker.log_summary(metrics)
    tracker.log_table(results)

    if config.wandb_project:
        # Log wall time and output artefact
        try:
            tracker._run.summary["wall_time_s"] = wall_elapsed  # type: ignore[union-attr]
            artifact = _wandb.Artifact(  # type: ignore[union-attr]
                name=f"eval-results-{config.solver}",
                type="eval-results",
            )
            artifact.add_file(str(saved_path))
            tracker._run.log_artifact(artifact)  # type: ignore[union-attr]
        except Exception:
            logger.warning("Could not log W&B artifact — continuing.", exc_info=True)

    tracker.finish()

    return results
