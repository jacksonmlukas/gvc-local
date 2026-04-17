"""GAIA benchmark adapter for unified evaluation.

`GAIA <https://huggingface.co/datasets/gaia-benchmark/GAIA>`_ is a benchmark
for general AI assistants.  The project road-map mentions Level-1 GAIA tasks as
a current candidate for cross-benchmark evaluation of the open-weight solvers.

This module provides:

* **GaiaTask** — typed container for a single GAIA task.
* **load_gaia_tasks** — loader that pulls tasks from HuggingFace datasets.
* **GaiaEvaluator** — orchestrates running a solver/agent over GAIA tasks and
  produces :class:`~gvc_local.eval_harness.RunResult` objects for unified
  reporting through the same pipeline used for Connections puzzles.

Usage::

    from gvc_local.eval.gaia import GaiaEvaluator, load_gaia_tasks

    tasks = load_gaia_tasks(level=1)
    evaluator = GaiaEvaluator(solver=my_agent)
    results = evaluator.evaluate(tasks)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence

from ..eval_harness import RunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GaiaTask:
    """A single GAIA benchmark task.

    Attributes
    ----------
    task_id : str
        Unique identifier from the dataset.
    question : str
        The question text presented to the agent.
    expected_answer : str
        Gold-standard answer for scoring.
    level : int
        GAIA difficulty level (1, 2, or 3).
    metadata : dict
        Any additional metadata from the dataset (file names, annotator
        notes, task type, etc.).
    """

    task_id: str
    question: str
    expected_answer: str
    level: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# HuggingFace dataset identifiers
_HF_DATASET = "gaia-benchmark/GAIA"
_HF_SPLIT_MAP = {
    "validation": "validation",
    "test": "test",
}


def load_gaia_tasks(
    level: int = 1,
    split: str = "validation",
    max_tasks: int | None = None,
) -> list[GaiaTask]:
    """Load GAIA tasks from HuggingFace datasets.

    Parameters
    ----------
    level : int
        Difficulty level to filter (1, 2, or 3).  Level 1 is the paper's
        current candidate.
    split : str
        Dataset split — ``"validation"`` (has answers) or ``"test"``.
    max_tasks : int, optional
        Cap the number of returned tasks (useful for debugging).

    Returns
    -------
    list[GaiaTask]
        Loaded and filtered tasks.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to load GAIA tasks.  "
            "Install it with:  pip install datasets"
        ) from exc

    logger.info(
        "Loading GAIA dataset (level=%d, split=%s) from HuggingFace ...",
        level,
        split,
    )
    ds = load_dataset(
        _HF_DATASET,
        "2023_all",  # GAIA config containing all levels
        split=_HF_SPLIT_MAP.get(split, split),
        trust_remote_code=True,
    )

    tasks: list[GaiaTask] = []
    for row in ds:
        row_level = int(row.get("Level", row.get("level", 0)))
        if row_level != level:
            continue

        task = GaiaTask(
            task_id=str(row.get("task_id", row.get("id", ""))),
            question=str(row.get("Question", row.get("question", ""))),
            expected_answer=str(row.get("Final answer", row.get("final_answer", ""))),
            level=row_level,
            metadata={
                k: v
                for k, v in row.items()
                if k not in {"task_id", "id", "Question", "question", "Final answer",
                             "final_answer", "Level", "level"}
            },
        )
        tasks.append(task)
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    logger.info("Loaded %d GAIA level-%d tasks.", len(tasks), level)
    return tasks


# ---------------------------------------------------------------------------
# Answer comparison / scoring
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase, strip whitespace and punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(predicted: str, expected: str) -> bool:
    """Case/punctuation-insensitive exact match."""
    return _normalise(predicted) == _normalise(expected)


def fuzzy_match(predicted: str, expected: str, threshold: float = 0.85) -> bool:
    """Fuzzy string similarity via ``SequenceMatcher``.

    Returns ``True`` if the normalised similarity ratio exceeds *threshold*.
    """
    ratio = SequenceMatcher(None, _normalise(predicted), _normalise(expected)).ratio()
    return ratio >= threshold


def numeric_match(predicted: str, expected: str, rel_tol: float = 1e-3) -> bool:
    """Try to parse both strings as numbers and compare with tolerance."""
    try:
        p = float(re.sub(r"[^\d.\-eE]", "", predicted))
        e = float(re.sub(r"[^\d.\-eE]", "", expected))
        if e == 0:
            return abs(p) < rel_tol
        return abs(p - e) / abs(e) < rel_tol
    except (ValueError, ZeroDivisionError):
        return False


def score_answer(predicted: str, expected: str) -> dict[str, bool]:
    """Score a predicted answer against the expected answer.

    Returns a dict with boolean flags for each matching strategy.
    """
    return {
        "exact_match": exact_match(predicted, expected),
        "fuzzy_match": fuzzy_match(predicted, expected),
        "numeric_match": numeric_match(predicted, expected),
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Type alias: a GAIA solver takes a question string and returns an answer string.
GaiaSolverCallable = Callable[[str], str]


class GaiaEvaluator:
    """Run a solver/agent over GAIA tasks and produce unified results.

    The evaluator bridges GAIA's Q&A format with the Connections-oriented
    :class:`RunResult` dataclass so that both benchmarks flow through the
    same reporting and tracking pipeline.

    Parameters
    ----------
    solver : GaiaSolverCallable
        Callable that maps ``question -> answer``.
    fuzzy_threshold : float
        Similarity threshold for fuzzy matching (default 0.85).

    Example
    -------
    ::

        evaluator = GaiaEvaluator(solver=lambda q: my_agent.answer(q))
        tasks = load_gaia_tasks(level=1)
        results = evaluator.evaluate(tasks)
    """

    def __init__(
        self,
        solver: GaiaSolverCallable,
        *,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        self.solver = solver
        self.fuzzy_threshold = fuzzy_threshold

    def evaluate_single(self, task: GaiaTask) -> tuple[RunResult, dict[str, Any]]:
        """Evaluate a single GAIA task.

        Returns
        -------
        result : RunResult
            Unified result object.  ``solved`` is True if the answer scores
            an exact *or* fuzzy *or* numeric match.  ``guesses`` is always 1
            (single-shot Q&A).  ``out_of_board_guesses`` is 0 for a correct
            answer, 1 otherwise (mapping the concept of "semantic grounding
            failures" to incorrect answers).
        detail : dict
            Per-scoring-method breakdown for logging.
        """
        try:
            predicted = self.solver(task.question)
        except Exception:
            logger.exception("Solver raised on task %s", task.task_id)
            predicted = ""

        scores = score_answer(predicted, task.expected_answer)
        solved = scores["exact_match"] or scores["fuzzy_match"] or scores["numeric_match"]

        result = RunResult(
            solved=solved,
            guesses=1,  # single-shot
            out_of_board_guesses=0 if solved else 1,
            puzzle_id=task.task_id,
            strata=f"gaia-level-{task.level}",
        )

        detail = {
            "task_id": task.task_id,
            "question": task.question[:200],
            "expected": task.expected_answer,
            "predicted": predicted[:500] if predicted else "",
            **scores,
            "solved": solved,
        }

        return result, detail

    def evaluate(
        self,
        tasks: Sequence[GaiaTask],
        *,
        verbose: bool = True,
    ) -> list[RunResult]:
        """Evaluate all tasks and return unified results.

        Parameters
        ----------
        tasks : sequence of GaiaTask
            Tasks to evaluate.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        list[RunResult]
            One result per task.
        """
        results: list[RunResult] = []
        details: list[dict[str, Any]] = []

        for i, task in enumerate(tasks):
            if verbose:
                status = f"[{i + 1}/{len(tasks)}] GAIA {task.task_id}"
                print(status, end=" ... ", flush=True)

            result, detail = self.evaluate_single(task)
            results.append(result)
            details.append(detail)

            if verbose:
                verdict = "PASS" if result.solved else "FAIL"
                print(verdict)

        # Summary
        n_correct = sum(1 for r in results if r.solved)
        total = len(results)
        if verbose and total > 0:
            print(f"\nGAIA Level Summary: {n_correct}/{total} ({n_correct / total:.1%})")

        return results
