"""Tests for stratified sampling and bootstrap CI primitives.

These cover the eval-craft surface area of GVC-Local:
- stratified_sample: reproducibility, per-stratum count semantics, edge cases.
- bootstrap_ci: CI bounds, mean correctness, reproducibility, empty input.
- summarize: end-to-end metric extraction on RunResult lists.

The implementation lives in :mod:`gvc_local.eval_harness`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gvc_local.eval_harness import (
    RunResult,
    bootstrap_ci,
    stratified_sample,
    summarize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_result(
    puzzle_id: int,
    strata: str,
    solved: bool = True,
    guesses: int = 4,
    oob: int = 0,
) -> RunResult:
    return RunResult(
        solved=solved,
        guesses=guesses,
        out_of_board_guesses=oob,
        puzzle_id=puzzle_id,
        strata=strata,
    )


@pytest.fixture()
def heterogeneous_pool() -> list[RunResult]:
    """20 puzzles across 4 strata with realistic heterogeneity."""
    return (
        [_mk_result(i, "wordplay", solved=(i % 2 == 0), guesses=4 + (i % 3)) for i in range(8)]
        + [_mk_result(8 + i, "cultural", solved=(i < 2), guesses=5) for i in range(5)]
        + [_mk_result(13 + i, "silent-letter", solved=True, guesses=4, oob=i) for i in range(4)]
        + [_mk_result(17 + i, "tag-fillin", solved=False, guesses=7) for i in range(3)]
    )


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_each_stratum_capped(self, heterogeneous_pool):
        picked = stratified_sample(heterogeneous_pool, per_stratum=3, seed=0)
        by_stratum: dict[str, int] = {}
        for r in picked:
            by_stratum[r.strata] = by_stratum.get(r.strata, 0) + 1
        # Each stratum gets min(per_stratum, available)
        assert by_stratum["wordplay"] == 3
        assert by_stratum["cultural"] == 3
        assert by_stratum["silent-letter"] == 3
        assert by_stratum["tag-fillin"] == 3  # exactly 3 available, all taken

    def test_per_stratum_exceeds_pool(self, heterogeneous_pool):
        """If per_stratum > stratum size, take all available without error."""
        picked = stratified_sample(heterogeneous_pool, per_stratum=100, seed=0)
        # Should return everything: 8 + 5 + 4 + 3 == 20
        assert len(picked) == 20

    def test_reproducible_with_seed(self, heterogeneous_pool):
        first = stratified_sample(heterogeneous_pool, per_stratum=3, seed=42)
        second = stratified_sample(heterogeneous_pool, per_stratum=3, seed=42)
        assert [r.puzzle_id for r in first] == [r.puzzle_id for r in second]

    def test_different_seeds_differ(self, heterogeneous_pool):
        first = stratified_sample(heterogeneous_pool, per_stratum=3, seed=0)
        second = stratified_sample(heterogeneous_pool, per_stratum=3, seed=1)
        # Different seed should generally give different draws (test is probabilistic
        # but with 4 strata of 3+ items each, P(identical draws) is vanishingly small)
        assert [r.puzzle_id for r in first] != [r.puzzle_id for r in second]

    def test_no_duplicates_within_stratum(self, heterogeneous_pool):
        picked = stratified_sample(heterogeneous_pool, per_stratum=3, seed=0)
        ids = [r.puzzle_id for r in picked]
        assert len(ids) == len(set(ids))

    def test_empty_pool(self):
        assert stratified_sample([], per_stratum=5) == []

    def test_single_stratum(self):
        pool = [_mk_result(i, "wordplay") for i in range(5)]
        picked = stratified_sample(pool, per_stratum=3, seed=0)
        assert len(picked) == 3
        assert all(r.strata == "wordplay" for r in picked)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_mean_matches_arithmetic_mean(self):
        values = [0.0, 1.0, 0.5, 0.25, 0.75]
        mean, _lo, _hi = bootstrap_ci(values, n_resamples=500, seed=0)
        assert mean == pytest.approx(np.mean(values), rel=1e-9)

    def test_ci_brackets_mean(self):
        values = [0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        mean, lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
        assert lo <= mean <= hi

    def test_ci_bounds_are_within_data_range(self):
        """Percentile-bootstrap CI should not exceed data min/max for bounded metrics."""
        values = [0.0, 0.0, 1.0, 1.0, 1.0]
        _mean, lo, hi = bootstrap_ci(values, n_resamples=2000, seed=0)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_reproducible_with_seed(self):
        values = list(range(20))
        a = bootstrap_ci(values, n_resamples=500, seed=7)
        b = bootstrap_ci(values, n_resamples=500, seed=7)
        assert a == b

    def test_different_seeds_differ(self):
        values = list(range(20))
        a = bootstrap_ci(values, n_resamples=500, seed=0)
        b = bootstrap_ci(values, n_resamples=500, seed=1)
        # CI bounds should differ with different seeds (probabilistic, but n=500
        # resamples × 20 values gives effectively zero chance of identical bounds)
        assert (a[1], a[2]) != (b[1], b[2])

    def test_empty_returns_nan(self):
        mean, lo, hi = bootstrap_ci([])
        assert np.isnan(mean)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_constant_input_has_zero_width_ci(self):
        """A constant sequence should produce a degenerate (zero-width) CI."""
        values = [0.5] * 10
        mean, lo, hi = bootstrap_ci(values, n_resamples=500, seed=0)
        assert mean == pytest.approx(0.5)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_ci_narrows_with_more_data(self):
        """A larger pool should produce a narrower CI for the same underlying distribution."""
        rng = np.random.default_rng(0)
        small = rng.uniform(0.0, 1.0, size=20).tolist()
        large = rng.uniform(0.0, 1.0, size=500).tolist()
        _m1, lo1, hi1 = bootstrap_ci(small, n_resamples=1000, seed=0)
        _m2, lo2, hi2 = bootstrap_ci(large, n_resamples=1000, seed=0)
        assert (hi2 - lo2) < (hi1 - lo1)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_returns_three_headline_metrics(self, heterogeneous_pool):
        metrics = summarize(heterogeneous_pool)
        assert set(metrics.keys()) == {"solve_rate", "semantic_grounding", "guesses_per_puzzle"}

    def test_solve_rate_matches_proportion(self):
        results = [
            _mk_result(0, "a", solved=True),
            _mk_result(1, "a", solved=True),
            _mk_result(2, "a", solved=False),
            _mk_result(3, "a", solved=True),
        ]
        metrics = summarize(results)
        mean, _lo, _hi = metrics["solve_rate"]
        assert mean == pytest.approx(0.75)

    def test_each_metric_returns_triple(self, heterogeneous_pool):
        metrics = summarize(heterogeneous_pool)
        for _name, triple in metrics.items():
            assert len(triple) == 3
            mean, lo, hi = triple
            assert lo <= mean <= hi

    def test_empty_pool_returns_nan_triples(self):
        metrics = summarize([])
        for _name, (mean, lo, hi) in metrics.items():
            assert np.isnan(mean)
            assert np.isnan(lo)
            assert np.isnan(hi)
