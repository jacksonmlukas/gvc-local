"""Tests for the results-aggregation module.

Covers join correctness, stratified summarisation, OVERALL row ordering, and
markdown rendering of bootstrap-CI triples.
"""

from __future__ import annotations

from gvc_local.eval.aggregate import (
    AggregatedResult,
    _fmt_ci,
    join_results,
    render_markdown,
    summarise_stratified,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_row(puzzle_id: int, solved: bool = True, strikes: int = 4) -> dict:
    return {
        "puzzle_id": puzzle_id,
        "solved": solved,
        "categories_solved": 4 if solved else 1,
        "strikes": strikes,
    }


# ---------------------------------------------------------------------------
# join_results
# ---------------------------------------------------------------------------


class TestJoinResults:
    def test_joins_on_puzzle_id(self):
        results = [_result_row(0), _result_row(1)]
        strata = {0: "wordplay", 1: "cultural"}
        joined = join_results(results, strata)
        assert joined[0].strata == "wordplay"
        assert joined[1].strata == "cultural"

    def test_missing_strata_falls_back_to_default(self):
        results = [_result_row(99)]
        joined = join_results(results, {})
        assert joined[0].strata == "category"  # DEFAULT_STRATUM

    def test_strikes_field_carried_through(self):
        results = [_result_row(0, solved=False, strikes=18)]
        joined = join_results(results, {0: "wordplay"})
        assert joined[0].strikes == 18
        assert joined[0].solved is False

    def test_handles_guesses_field_as_alias_for_strikes(self):
        """Old result files sometimes use 'guesses' instead of 'strikes'."""
        results = [{"puzzle_id": 0, "solved": True, "guesses": 7, "categories_solved": 4}]
        joined = join_results(results, {0: "wordplay"})
        assert joined[0].strikes == 7


# ---------------------------------------------------------------------------
# summarise_stratified
# ---------------------------------------------------------------------------


class TestSummariseStratified:
    def test_overall_row_first(self):
        rows = [
            AggregatedResult(0, True, 4, 4, "wordplay"),
            AggregatedResult(1, False, 20, 1, "cultural"),
        ]
        summary = summarise_stratified(rows, n_resamples=100)
        assert summary[0].stratum == "OVERALL"
        assert summary[0].n == 2

    def test_one_row_per_stratum_plus_overall(self):
        rows = [
            AggregatedResult(0, True, 4, 4, "wordplay"),
            AggregatedResult(1, True, 4, 4, "wordplay"),
            AggregatedResult(2, False, 20, 1, "cultural"),
        ]
        summary = summarise_stratified(rows, n_resamples=100)
        strata = [r.stratum for r in summary]
        assert strata.count("OVERALL") == 1
        assert "wordplay" in strata
        assert "cultural" in strata
        # Wordplay row should have n=2, cultural n=1
        for r in summary:
            if r.stratum == "wordplay":
                assert r.n == 2
            if r.stratum == "cultural":
                assert r.n == 1

    def test_solve_rate_matches_proportion(self):
        rows = [
            AggregatedResult(0, True, 4, 4, "wordplay"),
            AggregatedResult(1, True, 4, 4, "wordplay"),
            AggregatedResult(2, False, 20, 1, "wordplay"),
            AggregatedResult(3, False, 20, 1, "wordplay"),
        ]
        summary = summarise_stratified(rows, n_resamples=200)
        # OVERALL row, mean of solve_rate should be 0.5
        assert summary[0].solve_rate[0] == 0.5


# ---------------------------------------------------------------------------
# render_markdown + _fmt_ci
# ---------------------------------------------------------------------------


class TestFormat:
    def test_fmt_ci_percent(self):
        out = _fmt_ci((0.6, 0.55, 0.65), pct=True)
        assert "60.0%" in out
        assert "55.0" in out

    def test_fmt_ci_numeric(self):
        out = _fmt_ci((4.5, 4.0, 5.0), pct=False)
        assert "4.50" in out
        assert "4.00" in out

    def test_fmt_ci_nan(self):
        out = _fmt_ci((float("nan"), float("nan"), float("nan")))
        assert out == "n/a"


class TestRenderMarkdown:
    def test_basic_table_structure(self):
        rows = [
            AggregatedResult(0, True, 4, 4, "wordplay"),
            AggregatedResult(1, False, 20, 1, "cultural"),
        ]
        summary = summarise_stratified(rows, n_resamples=100)
        md = render_markdown(summary, "snap_gvc_v3", n_bootstrap=100)
        assert "snap_gvc_v3" in md
        assert "Solve rate" in md
        assert "OVERALL" in md
        assert "wordplay" in md
