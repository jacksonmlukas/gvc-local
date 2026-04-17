"""Stratified-sample eval harness with bootstrapped CIs.

This is the piece that plants your Bloomberg sampling experience directly in the repo.
Used in M4 once baseline numbers are in.

Key ideas:
- Stratified sampling over puzzle category types so reported numbers reflect balanced
  coverage rather than whichever puzzles happened to be in [start, end).
- Bootstrapped 95% CIs on the three headline metrics (solve rate, grounding, guesses
  per puzzle) so results tables show uncertainty, not just point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RunResult:
    solved: bool
    guesses: int
    out_of_board_guesses: int
    puzzle_id: int
    strata: str  # e.g., "wordplay", "cultural", "silent-letter", "tag-fillin"


def stratified_sample(
    pool: Sequence[RunResult],
    per_stratum: int,
    seed: int = 0,
) -> list[RunResult]:
    """Pick `per_stratum` puzzles from each stratum. Pool must be pre-labelled."""
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[RunResult]] = {}
    for r in pool:
        by_stratum.setdefault(r.strata, []).append(r)

    picked: list[RunResult] = []
    for stratum, items in by_stratum.items():
        k = min(per_stratum, len(items))
        idx = rng.choice(len(items), size=k, replace=False)
        picked.extend(items[i] for i in idx)
    return picked


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for a 1-alpha CI."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boots = arr[idx].mean(axis=1)
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return float(arr.mean()), lower, upper


def summarize(results: Sequence[RunResult]) -> dict[str, tuple[float, float, float]]:
    """Headline metrics with CIs. Mirrors the paper's Table 1 columns."""
    solved = [1.0 if r.solved else 0.0 for r in results]
    grounding = [float(r.out_of_board_guesses) for r in results]
    efficiency = [float(r.guesses) for r in results]
    return {
        "solve_rate": bootstrap_ci(solved),
        "semantic_grounding": bootstrap_ci(grounding),
        "guesses_per_puzzle": bootstrap_ci(efficiency),
    }
