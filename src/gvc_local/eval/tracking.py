"""Weights & Biases experiment tracking for evaluation runs.

Wraps the wandb SDK behind a thin ``ExperimentTracker`` facade so the rest of
the eval harness never imports wandb directly.  If wandb is not installed the
tracker silently falls back to console logging -- useful for CI or quick local
runs where you don't want the dependency.

Typical usage::

    tracker = ExperimentTracker()
    tracker.init(project="gvc-local-eval", run_name="snap_gvc-llama31-8b",
                 config={"solver": "snap_gvc", "model": "llama-3.1-8b"})
    for result in results:
        tracker.log_puzzle(result.puzzle_id, result)
    tracker.log_summary(summarize(results))
    tracker.log_table(results)
    tracker.finish()
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ..eval_harness import RunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# wandb availability
# ---------------------------------------------------------------------------
try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


def wandb_available() -> bool:
    """Return ``True`` if the wandb package is importable."""
    return _WANDB_AVAILABLE


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Unified interface for tracking evaluation experiments.

    When wandb is available every call proxies into the wandb SDK.  Otherwise,
    metrics are logged to the Python ``logging`` module at INFO level, making
    the harness runnable without any external services.

    Parameters
    ----------
    enabled : bool, optional
        Force-disable tracking even if wandb is installed.  Defaults to
        ``True`` (use wandb when available).
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._use_wandb = _WANDB_AVAILABLE and enabled
        self._run: Any = None  # wandb.Run once initialized
        self._puzzle_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        """Initialize a tracking run.

        Parameters
        ----------
        project : str
            W&B project name (e.g. ``"gvc-local-eval"``).
        run_name : str, optional
            Human-readable run name shown in the dashboard.
        config : dict, optional
            Flat dict of hyperparameters / eval settings.
        tags : list[str], optional
            Tags attached to the run for filtering.
        notes : str, optional
            Markdown description of the run.
        """
        if self._use_wandb:
            assert wandb is not None
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or ["eval", "gvc-local"],
                notes=notes,
                reinit=True,
            )
            logger.info("W&B run initialized: %s/%s", project, run_name)
        else:
            logger.info(
                "[console] Tracking run (wandb unavailable): project=%s  name=%s",
                project,
                run_name,
            )
            if config:
                for k, v in config.items():
                    logger.info("[console]   config.%s = %s", k, v)

    # ------------------------------------------------------------------
    # Per-puzzle logging
    # ------------------------------------------------------------------

    def log_puzzle(self, puzzle_id: int | str, result: RunResult) -> None:
        """Log a single puzzle outcome.

        Each call records one step in W&B (for live loss-curve-style charts)
        and appends a row for the eventual results table.

        Parameters
        ----------
        puzzle_id : int or str
            Unique puzzle / task identifier.
        result : RunResult
            The outcome for this puzzle.
        """
        row = {
            "puzzle_id": puzzle_id,
            "solved": result.solved,
            "guesses": result.guesses,
            "out_of_board_guesses": result.out_of_board_guesses,
            "strata": result.strata,
        }
        self._puzzle_rows.append(row)

        if self._use_wandb and self._run is not None:
            wandb.log(
                {
                    "puzzle/solved": int(result.solved),
                    "puzzle/guesses": result.guesses,
                    "puzzle/oob_guesses": result.out_of_board_guesses,
                },
            )
        else:
            logger.info(
                "[console] puzzle=%s  solved=%s  guesses=%d  oob=%d  strata=%s",
                puzzle_id,
                result.solved,
                result.guesses,
                result.out_of_board_guesses,
                result.strata,
            )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def log_summary(self, metrics: dict[str, tuple[float, float, float]]) -> None:
        """Log headline metrics with confidence intervals as run summary values.

        Parameters
        ----------
        metrics : dict
            Mapping of metric name to ``(mean, ci_lower, ci_upper)`` triples,
            as returned by :func:`eval_harness.summarize`.
        """
        flat: dict[str, float] = {}
        for name, (mean, lower, upper) in metrics.items():
            flat[f"{name}/mean"] = mean
            flat[f"{name}/ci_lower"] = lower
            flat[f"{name}/ci_upper"] = upper

        if self._use_wandb and self._run is not None:
            for k, v in flat.items():
                self._run.summary[k] = v
            wandb.log(flat)
            logger.info("W&B summary updated with %d metrics.", len(flat))
        else:
            for k, v in flat.items():
                logger.info("[console] summary  %s = %.4f", k, v)

    # ------------------------------------------------------------------
    # Full results table
    # ------------------------------------------------------------------

    def log_table(self, results: Sequence[RunResult]) -> None:
        """Create a W&B Table for interactive exploration of all results.

        Parameters
        ----------
        results : sequence of RunResult
            Complete list of evaluation outcomes.
        """
        rows = [asdict(r) for r in results]

        if self._use_wandb and self._run is not None:
            columns = list(rows[0].keys()) if rows else []
            table = wandb.Table(columns=columns, data=[list(r.values()) for r in rows])
            wandb.log({"results_table": table})
            logger.info("W&B table logged with %d rows.", len(rows))
        else:
            logger.info("[console] Results table (%d rows) — skipped (no wandb).", len(rows))

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Finalise and close the tracking run."""
        if self._use_wandb and self._run is not None:
            self._run.finish()
            logger.info("W&B run finished.")
        else:
            logger.info("[console] Tracking run finished.")
        self._run = None
        self._puzzle_rows.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ExperimentTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()
