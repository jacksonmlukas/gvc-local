"""GVC (Guesser-Validator-Consensus) solver.

Implements the conservative, deliberative puzzle-solving loop:

1. **Guesser** proposes a group of 4 words + category label.
2. **Grounding check** ensures the guess is valid (words on board, not a
   repeat, correct count).
3. **Validator** reviews the guesser's proposal and either agrees or rejects
   with feedback.
4. If consensus (guesser + validator agree on the same group), submit to the
   game engine. Otherwise, loop with accumulated feedback.

This is the *System-2* component used by both the standalone GVC solver and
as the conservative phase of the Snap-GVC solver.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from gvc_local.agents.guesser import GuesserAgent
from gvc_local.agents.validator import ValidatorAgent
from gvc_local.endpoint import Client
from gvc_local.rag.retriever import PuzzleRetriever, RetrievalResult
from gvc_local.solvers.base import BaseSolver, SolverMetrics

logger = logging.getLogger(__name__)


def _grounding_check(
    guess: list[str],
    remaining_words: list[str],
    group_size: int,
    sorted_failed_guesses: list[list[str]],
) -> tuple[bool, str]:
    """Deterministic validation of a guess before submitting to the game engine.

    Returns (is_valid, error_message).
    """
    proc_guess = [w.strip().upper().replace(",", "") for w in guess]
    proc_remaining = [w.strip().upper().replace(",", "") for w in remaining_words]
    proc_failed = [
        [w.strip().upper().replace(",", "") for w in g] for g in sorted_failed_guesses
    ]

    errors: list[str] = []

    # Rule 1: all words must be on the board
    missing = [w for w in proc_guess if w not in proc_remaining]
    if missing:
        errors.append(f"Word(s) {missing} not in remaining words.")
        return False, " ".join(errors)

    # Rule 2: correct group size
    if len(proc_guess) != group_size:
        errors.append(f"Expected {group_size} words, got {len(proc_guess)}.")
        return False, " ".join(errors)

    # Rule 3: not a repeat of a failed guess
    sorted_guess = sorted(proc_guess)
    for fg in proc_failed:
        if sorted_guess == sorted(fg):
            errors.append(f"Guess {sorted_guess} repeats a previously failed grouping.")
            return False, " ".join(errors)

    return True, ""


class GVCSolver(BaseSolver):
    """Conservative Guesser-Validator-Consensus solver.

    Parameters
    ----------
    client:
        vLLM endpoint client (shared by guesser and validator).
    retriever:
        Optional RAG retriever for historical puzzle context.
    max_internal_retries:
        Maximum guesser-validator rounds before giving up on a single guess
        and returning the best attempt.
    guesser_temperature:
        Temperature for the guesser agent.
    validator_temperature:
        Temperature for the validator agent.
    rag_k:
        Number of RAG results to retrieve per guess.
    """

    def __init__(
        self,
        client: Client,
        *,
        retriever: PuzzleRetriever | None = None,
        max_internal_retries: int = 15,
        guesser_temperature: float = 0.7,
        validator_temperature: float = 0.7,
        rag_k: int = 3,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.max_internal_retries = max_internal_retries
        self.rag_k = rag_k

        self.guesser = GuesserAgent(client, temperature=guesser_temperature)
        self.validator = ValidatorAgent(client, temperature=validator_temperature)

        # Per-game mutable state (reset between games via play())
        self._failed_guesses: list[list[str]] = []
        self._sorted_failed: list[list[str]] = []
        self._guesser_understanding: list[list[str]] | None = None
        self._rejected_buffer: deque[list[str]] = deque(maxlen=max_internal_retries)
        self._validator_feedback: str | None = None

    # ------------------------------------------------------------------
    # BaseSolver interface
    # ------------------------------------------------------------------

    def guess(
        self,
        remaining_words: list[str],
        group_size: int = 4,
        failed_guesses: list[list[str]] | None = None,
        metrics: SolverMetrics | None = None,
    ) -> tuple[list[str], str]:
        """Run the GVC loop and return (group, category).

        This method performs multiple internal guesser-validator rounds until
        consensus is reached or retries are exhausted.
        """
        if failed_guesses is not None:
            self._failed_guesses = failed_guesses
            self._sorted_failed = [sorted(g) for g in failed_guesses]

        if metrics is None:
            metrics = SolverMetrics()

        # Build feedback string from failed guesses
        feedback = self._format_failed_feedback()

        # RAG context
        rag_context = ""
        if self.retriever is not None:
            try:
                result = self.retriever.retrieve(remaining_words, k=self.rag_k)
                rag_context = result.format_for_prompt()
            except Exception as exc:
                logger.warning("[GVC] RAG retrieval failed: %s", exc)

        # Reset per-guess state
        self._rejected_buffer.clear()
        self._validator_feedback = None

        for attempt in range(1, self.max_internal_retries + 1):
            # Step 1: Guesser
            try:
                group, category, understanding = self.guesser.guess(
                    remaining_words=remaining_words,
                    feedback=feedback,
                    rag_context=rag_context,
                    validator_feedback=self._validator_feedback or "",
                    previous_understanding=self._guesser_understanding,
                )
                self._guesser_understanding = understanding if understanding else self._guesser_understanding
                metrics.total_llm_calls += 1
            except ValueError as exc:
                logger.warning("[GVC] guesser parse error on attempt %d: %s", attempt, exc)
                continue

            # Step 2: Grounding check
            grounded, error = _grounding_check(
                group, remaining_words, group_size, self._sorted_failed
            )
            if not grounded:
                logger.info("[GVC] grounding failed: %s", error)
                self._rejected_buffer.append(group)
                self._validator_feedback = error
                continue

            # If only one group left, skip validation
            if len(remaining_words) <= group_size:
                logger.info("[GVC] last group, skipping validation")
                self._reset_guess_state()
                return group, category

            # Step 3: Validator
            try:
                # Build the raw guesser reply for the validator (reconstruct)
                guesser_reply_text = (
                    f"Group: {', '.join(group)}\n"
                    f"Category: {category}"
                )
                agreement, val_feedback, val_group = self.validator.validate(
                    guesser_reply=guesser_reply_text,
                    remaining_words=remaining_words,
                    feedback=feedback,
                )
                metrics.total_llm_calls += 1
            except ValueError as exc:
                logger.warning("[GVC] validator parse error on attempt %d: %s", attempt, exc)
                continue

            if agreement:
                logger.info("[GVC] consensus reached on attempt %d", attempt)
                self._reset_guess_state()
                return group, category

            # No consensus -- accumulate feedback and retry
            self._rejected_buffer.append(group)
            self._validator_feedback = val_feedback or error
            logger.info(
                "[GVC] no consensus on attempt %d/%d for category '%s'",
                attempt,
                self.max_internal_retries,
                category,
            )

        # Exhausted retries -- submit the last guesser guess as a hail-mary
        logger.warning("[GVC] retries exhausted, submitting last guess")
        self._reset_guess_state()
        return group, category  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_failed_feedback(self) -> str:
        if not self._sorted_failed:
            return ""
        lines = [
            "- The following groups have been guessed but are NOT part of the solution:"
        ]
        for fg in self._sorted_failed:
            lines.append(f"  - {', '.join(fg)}")
        return "\n".join(lines)

    def _reset_guess_state(self) -> None:
        """Clear per-guess (not per-game) transient state."""
        self._rejected_buffer.clear()
        self._validator_feedback = None
        # Keep _guesser_understanding across guesses within a game

    def reset(self) -> None:
        """Full reset for a new game."""
        self._failed_guesses = []
        self._sorted_failed = []
        self._guesser_understanding = None
        self._rejected_buffer.clear()
        self._validator_feedback = None
