"""Snap-GVC solver -- dual-process (System-1 / System-2) puzzle solver.

This is the paper's key contribution: a dual-process architecture that
switches between conservative deliberation (GVC loop) and fast intuitive
guessing (Snap) based on performance:

    1. **Conservative phase (System-2):** Run the full GVC guesser-validator
       loop. If the solver accumulates too many wrong guesses or the internal
       loop fails to reach consensus, switch to snap.
    2. **Snap phase (System-1):** High-temperature, minimal-prompt guessing.
       Submit guesses directly (no validator). When snap gets one correct,
       switch back to the conservative phase.

This mirrors the cognitive science notion that when careful analysis fails,
a fast heuristic "snap" can break the impasse, and success restores
confidence for deliberative reasoning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from gvc_local.agents.snap_guesser import SnapGuesserAgent
from gvc_local.endpoint import Client
from gvc_local.game import Connections, GameOverError
from gvc_local.rag.retriever import PuzzleRetriever
from gvc_local.solvers.base import BaseSolver, SolverMetrics, TraceRecorder
from gvc_local.solvers.gvc import GVCSolver, _grounding_check

logger = logging.getLogger(__name__)


class SnapGVCSolver(BaseSolver):
    """Dual-process solver combining conservative GVC with snap guessing.

    Parameters
    ----------
    client:
        vLLM endpoint client.
    retriever:
        Optional RAG retriever.
    max_conservative_wrong:
        Number of wrong game-engine guesses in the conservative phase before
        switching to snap.  (Default: 2, more aggressive than upstream's 3
        to conserve API budget on rate-limited tiers.)
    max_conservative_errors:
        Number of internal errors (parse failures, etc.) in conservative
        phase before switching to snap.  (Default: 3, raised from 2 to
        tolerate more rate-limit retries.)
    max_internal_retries:
        Max guesser-validator rounds per conservative guess.
    snap_temperature:
        Temperature for the snap guesser.
    guesser_temperature:
        Temperature for the conservative guesser.
    validator_temperature:
        Temperature for the validator.
    rag_k:
        Number of RAG results to retrieve.
    """

    def __init__(
        self,
        client: Client,
        *,
        retriever: PuzzleRetriever | None = None,
        max_conservative_wrong: int = 2,
        max_conservative_errors: int = 3,
        max_internal_retries: int = 8,
        snap_temperature: float = 0.9,
        guesser_temperature: float = 0.7,
        validator_temperature: float = 0.7,
        rag_k: int = 3,
    ) -> None:
        self.client = client
        self.max_conservative_wrong = max_conservative_wrong
        self.max_conservative_errors = max_conservative_errors

        # Sub-solvers / agents
        self.gvc = GVCSolver(
            client,
            retriever=retriever,
            max_internal_retries=max_internal_retries,
            guesser_temperature=guesser_temperature,
            validator_temperature=validator_temperature,
            rag_k=rag_k,
        )
        self.snap = SnapGuesserAgent(client, temperature=snap_temperature)

        # Shared state across phases
        self._failed_guesses: list[list[str]] = []
        self._sorted_failed: list[list[str]] = []

    # ------------------------------------------------------------------
    # BaseSolver.guess -- not the main entry point for Snap-GVC, but
    # provided for interface compatibility.
    # ------------------------------------------------------------------

    def guess(
        self,
        remaining_words: list[str],
        group_size: int = 4,
        failed_guesses: list[list[str]] | None = None,
        metrics: SolverMetrics | None = None,
    ) -> tuple[list[str], str]:
        """Single-guess entry point (delegates to conservative GVC)."""
        return self.gvc.guess(remaining_words, group_size, failed_guesses, metrics)

    # ------------------------------------------------------------------
    # Deterministic swap engine — zero LLM calls
    # ------------------------------------------------------------------

    @staticmethod
    def _try_all_swaps(
        near_miss: list[str],
        remaining_words: list[str],
        game: Connections,
        sorted_failed: list[list[str]],
        metrics: SolverMetrics,
        recorder: TraceRecorder | None,
    ) -> bool:
        """Try every single-word swap on a near-miss guess.

        When a guess is "one away" (3/4 correct), there are at most
        ``4 * (len(remaining) - 4)`` candidate swaps.  We submit each
        to the game engine without any LLM call.

        Returns True if one of the swaps is correct (category solved).
        """
        near_set = set(w.upper() for w in near_miss)
        pool = [w for w in remaining_words if w.upper() not in near_set]
        upper_miss = [w.upper() for w in near_miss]

        for swap_out_idx in range(len(upper_miss)):
            for swap_in in pool:
                candidate = list(upper_miss)
                candidate[swap_out_idx] = swap_in.upper()
                sorted_cand = sorted(candidate)

                # Skip if this exact combo was already tried
                if sorted_cand in sorted_failed:
                    continue

                try:
                    result = game.category_guess_check(candidate)
                except GameOverError:
                    logger.warning("[SwapEngine] game over during swap search")
                    return False

                if result is not None:
                    cat_idx = game._og_groups.index(result)
                    metrics.record_solve(cat_idx)
                    logger.info(
                        "[SwapEngine] SOLVED via swap: %s -> %s "
                        "(swapped %s for %s)",
                        candidate, result.group,
                        upper_miss[swap_out_idx], swap_in,
                    )
                    if recorder:
                        recorder.record(
                            "correct_guess",
                            {
                                "phase": "swap_engine",
                                "guess": candidate,
                                "actual_category": result.group,
                                "swapped_out": upper_miss[swap_out_idx],
                                "swapped_in": swap_in,
                            },
                        )
                    return True
                else:
                    # Wrong swap — track it
                    metrics.increment_failed()
                    sorted_failed.append(sorted_cand)
                    if recorder:
                        recorder.record(
                            "wrong_guess",
                            {
                                "phase": "swap_engine",
                                "guess": candidate,
                                "category": "swap_attempt",
                            },
                        )

                    if game.is_over:
                        return False

        logger.info("[SwapEngine] exhausted all swaps, none correct")
        return False

    # ------------------------------------------------------------------
    # Main game loop -- overrides BaseSolver.play()
    # ------------------------------------------------------------------

    def play(
        self,
        game: Connections,
        *,
        trace_path: str | Path | None = None,
    ) -> list[bool]:
        """Play a full game with dual-process switching.

        The outer loop alternates between conservative and snap phases.
        """
        metrics = SolverMetrics()
        recorder = TraceRecorder(trace_path) if trace_path else None
        self._failed_guesses = []
        self._sorted_failed = []
        self.gvc.reset()

        t0 = time.time()

        if recorder:
            recorder.record(
                "game_start",
                {
                    "words": game.all_words,
                    "solver": "snap_gvc",
                },
            )

        while not game.is_over:
            # ---- Conservative phase (System-2) ----
            wrong_count = 0
            error_count = 0

            logger.info("[SnapGVC] entering CONSERVATIVE phase")
            if recorder:
                recorder.record("phase", {"phase": "conservative"})

            while not game.is_over:
                remaining = game.all_words

                try:
                    guess_words, category = self.gvc.guess(
                        remaining_words=remaining,
                        group_size=game.group_size,
                        failed_guesses=self._failed_guesses,
                        metrics=metrics,
                    )
                except Exception as exc:
                    logger.warning("[SnapGVC] conservative guess error: %s", exc)
                    error_count += 1
                    self.gvc._reset_guess_state()
                    if error_count >= self.max_conservative_errors:
                        logger.info("[SnapGVC] too many errors, switching to snap")
                        break
                    continue

                # Check for sentinel "None"/"Error" returns (matching upstream)
                if category in ("None", "Error"):
                    logger.info("[SnapGVC] conservative returned sentinel, switching to snap")
                    self.gvc._reset_guess_state()
                    break

                # Submit to game engine
                try:
                    result = game.category_guess_check(guess_words)
                except GameOverError:
                    logger.warning("[SnapGVC] game over during conservative phase")
                    break

                if result is None:
                    # Wrong guess
                    metrics.increment_failed()
                    metrics.record_hallucinations(guess_words, remaining)
                    sorted_g = sorted(w.upper() for w in guess_words)
                    self._failed_guesses.append(sorted_g)
                    self._sorted_failed.append(sorted_g)
                    wrong_count += 1

                    # Track near-misses and elimination pairs
                    self.gvc.record_failed_pairs(guess_words)
                    if game.last_one_away:
                        self.gvc.record_near_miss(guess_words)
                        logger.info(
                            "[SnapGVC] CONSERVATIVE ONE AWAY: %s — "
                            "activating swap engine", guess_words
                        )
                        # Deterministic swap: try all single-word swaps
                        swap_solved = self._try_all_swaps(
                            guess_words, remaining, game,
                            self._sorted_failed, metrics, recorder,
                        )
                        if swap_solved:
                            wrong_count = 0
                            self.gvc._reset_guess_state()
                            continue  # back to conservative with one fewer group

                    logger.info(
                        "[SnapGVC] CONSERVATIVE WRONG (%d/%d): %s",
                        wrong_count,
                        self.max_conservative_wrong,
                        guess_words,
                    )
                    if recorder:
                        recorder.record(
                            "wrong_guess",
                            {
                                "phase": "conservative",
                                "guess": guess_words,
                                "category": category,
                                "one_away": game.last_one_away,
                            },
                        )

                    # Reset agent state for fresh attempt
                    self.gvc._reset_guess_state()

                    if wrong_count >= self.max_conservative_wrong:
                        logger.info("[SnapGVC] max conservative wrongs, switching to snap")
                        break
                else:
                    # Correct guess
                    cat_idx = game._og_groups.index(result)
                    metrics.record_solve(cat_idx)
                    wrong_count = 0  # reset on success
                    logger.info(
                        "[SnapGVC] CONSERVATIVE CORRECT: %s -> %s",
                        guess_words,
                        result.group,
                    )
                    if recorder:
                        recorder.record(
                            "correct_guess",
                            {
                                "phase": "conservative",
                                "guess": guess_words,
                                "actual_category": result.group,
                            },
                        )
                    self.gvc._reset_guess_state()

            if game.is_over:
                break

            # ---- Snap phase (System-1) ----
            logger.info("[SnapGVC] entering SNAP phase")
            if recorder:
                recorder.record("phase", {"phase": "snap"})

            snap_correct = False
            snap_attempts = 0
            max_snap_attempts = 10  # cap snap retries to avoid infinite loops
            while not game.is_over and not snap_correct and snap_attempts < max_snap_attempts:
                snap_attempts += 1
                remaining = game.all_words
                feedback = self._format_snap_feedback()

                try:
                    guess_words, reason = self.snap.snap_guess(
                        remaining_words=remaining,
                        feedback=feedback,
                    )
                    metrics.total_llm_calls += 1
                except ValueError as exc:
                    logger.warning("[SnapGVC] snap parse error: %s", exc)
                    continue

                # Grounding check
                grounded, error = _grounding_check(
                    guess_words, remaining, game.group_size, self._sorted_failed
                )
                if not grounded:
                    logger.info("[SnapGVC] snap grounding failed: %s", error)
                    continue

                # Submit to game engine
                try:
                    result = game.category_guess_check(guess_words)
                except GameOverError:
                    logger.warning("[SnapGVC] game over during snap phase")
                    break

                if result is None:
                    metrics.increment_failed()
                    metrics.record_hallucinations(guess_words, remaining)
                    sorted_g = sorted(w.upper() for w in guess_words)
                    self._failed_guesses.append(sorted_g)
                    self._sorted_failed.append(sorted_g)

                    # Track near-misses and elimination pairs
                    self.gvc.record_failed_pairs(guess_words)
                    if game.last_one_away:
                        self.gvc.record_near_miss(guess_words)
                        logger.info(
                            "[SnapGVC] SNAP ONE AWAY: %s — "
                            "activating swap engine", guess_words
                        )
                        swap_solved = self._try_all_swaps(
                            guess_words, remaining, game,
                            self._sorted_failed, metrics, recorder,
                        )
                        if swap_solved:
                            snap_correct = True  # switch back to conservative
                            break

                    logger.info("[SnapGVC] SNAP WRONG: %s", guess_words)
                    if recorder:
                        recorder.record(
                            "wrong_guess",
                            {
                                "phase": "snap",
                                "guess": guess_words,
                                "reason": reason,
                                "one_away": game.last_one_away,
                            },
                        )
                else:
                    cat_idx = game._og_groups.index(result)
                    metrics.record_solve(cat_idx)
                    snap_correct = True
                    logger.info(
                        "[SnapGVC] SNAP CORRECT: %s -> %s  (switching back to conservative)",
                        guess_words,
                        result.group,
                    )
                    if recorder:
                        recorder.record(
                            "correct_guess",
                            {
                                "phase": "snap",
                                "guess": guess_words,
                                "actual_category": result.group,
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
            "[SnapGVC] game finished: solved=%s  failed=%d  time=%.1fs",
            game.solved_categories,
            metrics.failed_guesses,
            metrics.wall_time_s,
        )

        self.gvc.reset()
        return game.solved_categories

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_snap_feedback(self) -> str:
        """Build feedback string for the snap agent about failed guesses and near-misses."""
        if not self._sorted_failed:
            return ""

        sections: list[str] = []

        # Failed guesses
        lines = [
            "Note: You must not return any 4-word groupings from the "
            "following groups as they're not part of the solution:"
        ]
        for fg in self._sorted_failed:
            lines.append(f"  - {', '.join(fg)}")
        sections.append("\n".join(lines))

        # Near-miss hints
        if self.gvc._near_misses:
            nm_lines = [
                "IMPORTANT: These guesses were ONE AWAY (3/4 correct). "
                "Try swapping exactly one word:"
            ]
            for nm in self.gvc._near_misses:
                nm_lines.append(f"  - {', '.join(nm)}")
            sections.append("\n".join(nm_lines))

        return "\n\n".join(sections)
