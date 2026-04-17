"""Base solver -- abstract interface and shared game-loop logic.

All solvers (GVC, Snap-GVC, etc.) inherit from ``BaseSolver`` and implement
:meth:`guess`. The base class provides:

* The ``play()`` game loop that matches the upstream interface.
* Metrics tracking (failed guesses, hallucinated words, solve order).
* Optional trace recording to JSONL for fine-tuning data collection.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gvc_local.game import Connections, GameOverError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight metrics tracker (no upstream Metrics dependency required)
# ---------------------------------------------------------------------------


@dataclass
class SolverMetrics:
    """Accumulates per-game metrics during a solve run."""

    failed_guesses: int = 0
    hallucinated_words: list[list[str]] = field(default_factory=list)
    solve_order: list[int] = field(default_factory=list)  # category indices in order solved
    total_llm_calls: int = 0
    wall_time_s: float = 0.0

    def increment_failed(self) -> None:
        self.failed_guesses += 1

    def record_hallucinations(self, guessed: list[str], remaining: list[str]) -> None:
        """Track words in the guess that are not on the board."""
        remaining_upper = {w.upper() for w in remaining}
        bad = [w for w in guessed if w.upper() not in remaining_upper]
        if bad:
            self.hallucinated_words.append(bad)

    def record_solve(self, category_index: int) -> None:
        self.solve_order.append(category_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_guesses": self.failed_guesses,
            "hallucinated_words": self.hallucinated_words,
            "solve_order": self.solve_order,
            "total_llm_calls": self.total_llm_calls,
            "wall_time_s": round(self.wall_time_s, 2),
        }


# ---------------------------------------------------------------------------
# Trace recorder (optional)
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Append-only JSONL writer for agent interaction traces.

    Each line is a JSON object with ``{"event": ..., "data": ..., "ts": ...}``.
    Useful for collecting fine-tuning data from successful solves.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def record(self, event: str, data: dict[str, Any]) -> None:
        line = json.dumps({"event": event, "data": data, "ts": time.time()})
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Abstract base solver
# ---------------------------------------------------------------------------


class BaseSolver(abc.ABC):
    """Abstract base for all puzzle solvers.

    Subclasses must implement :meth:`guess`.
    """

    @abc.abstractmethod
    def guess(
        self,
        remaining_words: list[str],
        group_size: int,
        failed_guesses: list[list[str]],
        metrics: SolverMetrics,
    ) -> tuple[list[str], str]:
        """Produce a guess of ``group_size`` words and a category label.

        Parameters
        ----------
        remaining_words:
            The words still on the board.
        group_size:
            Number of words per group (always 4 for standard Connections).
        failed_guesses:
            List of previously failed word groups (sorted, uppercased).
        metrics:
            Metrics accumulator for the current game.

        Returns
        -------
        A 2-tuple of (word_group, category_label).
        """
        ...

    def play(
        self,
        game: Connections,
        *,
        trace_path: str | Path | None = None,
    ) -> list[bool]:
        """Play a full game of Connections.

        Parameters
        ----------
        game:
            A ``Connections`` game instance (mutated in place).
        trace_path:
            If provided, agent interactions are written to this JSONL file.

        Returns
        -------
        A list of booleans indicating which of the original categories were
        solved, aligned with ``game._og_groups``.
        """
        metrics = SolverMetrics()
        recorder = TraceRecorder(trace_path) if trace_path else None
        failed_guesses: list[list[str]] = []
        t0 = time.time()

        if recorder:
            recorder.record(
                "game_start",
                {
                    "words": game.all_words,
                    "categories": [
                        {"group": c.group, "level": c.level, "members": c.members}
                        for c in game._og_groups
                    ],
                },
            )

        while not game.is_over:
            remaining = game.all_words
            try:
                guess_words, category = self.guess(
                    remaining_words=remaining,
                    group_size=game.group_size,
                    failed_guesses=failed_guesses,
                    metrics=metrics,
                )
            except Exception as exc:
                logger.error("[Solver] guess() raised %s: %s", type(exc).__name__, exc)
                if recorder:
                    recorder.record("guess_error", {"error": str(exc)})
                break

            # Validate the guess against the game engine
            try:
                result = game.category_guess_check(guess_words)
            except GameOverError:
                logger.warning("[Solver] game over (max strikes reached)")
                if recorder:
                    recorder.record("game_over", {"reason": "max_strikes"})
                break

            if result is None:
                # Wrong guess
                metrics.increment_failed()
                metrics.record_hallucinations(guess_words, remaining)
                sorted_guess = sorted(w.upper() for w in guess_words)
                failed_guesses.append(sorted_guess)
                logger.info(
                    "[Solver] WRONG: %s (category=%s)  strikes=%d",
                    guess_words,
                    category,
                    game.current_strikes,
                )
                if recorder:
                    recorder.record(
                        "wrong_guess",
                        {
                            "guess": guess_words,
                            "category": category,
                            "strikes": game.current_strikes,
                        },
                    )
            else:
                # Correct guess
                cat_idx = game._og_groups.index(result)
                metrics.record_solve(cat_idx)
                logger.info(
                    "[Solver] CORRECT: %s -> %s (level %d)",
                    guess_words,
                    result.group,
                    result.level,
                )
                if recorder:
                    recorder.record(
                        "correct_guess",
                        {
                            "guess": guess_words,
                            "proposed_category": category,
                            "actual_category": result.group,
                            "level": result.level,
                        },
                    )

        metrics.wall_time_s = time.time() - t0

        if recorder:
            recorder.record(
                "game_end",
                {
                    "solved": game.solved_categories,
                    "metrics": metrics.to_dict(),
                },
            )
            recorder.close()

        logger.info(
            "[Solver] game finished: solved=%s  failed=%d  time=%.1fs",
            game.solved_categories,
            metrics.failed_guesses,
            metrics.wall_time_s,
        )

        return game.solved_categories
