"""Tests for GVC solver — grounding check, near-miss memory, fuzzy dedup, elimination logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from gvc_local.solvers.base import SolverMetrics, TraceRecorder
from gvc_local.solvers.gvc import (
    GVCSolver,
    _grounding_check,
    _normalize_words,
    _overlap_count,
)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class TestNormalizeWords:
    def test_basic(self):
        assert _normalize_words(["apple", " banana ", "Cherry"]) == ["APPLE", "BANANA", "CHERRY"]

    def test_strips_commas(self):
        assert _normalize_words(["word,"]) == ["WORD"]

    def test_empty(self):
        assert _normalize_words([]) == []


class TestOverlapCount:
    def test_full_overlap(self):
        assert _overlap_count(["A", "B", "C", "D"], ["A", "B", "C", "D"]) == 4

    def test_partial_overlap(self):
        assert _overlap_count(["A", "B", "C", "D"], ["A", "B", "X", "Y"]) == 2

    def test_no_overlap(self):
        assert _overlap_count(["A", "B", "C", "D"], ["W", "X", "Y", "Z"]) == 0


# ---------------------------------------------------------------------------
# Grounding check tests
# ---------------------------------------------------------------------------


class TestGroundingCheck:
    BOARD = ["APPLE", "BANANA", "CHERRY", "DATE", "RED", "BLUE", "GREEN", "YELLOW"]

    def test_valid_guess(self):
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY", "DATE"],
            self.BOARD, 4, [],
        )
        assert ok is True
        assert err == ""

    def test_missing_words(self):
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY", "GRAPE"],
            self.BOARD, 4, [],
        )
        assert ok is False
        assert "GRAPE" in err

    def test_wrong_group_size(self):
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY"],
            self.BOARD, 4, [],
        )
        assert ok is False
        assert "Expected 4" in err

    def test_exact_repeat_rejected(self):
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY", "DATE"],
            self.BOARD, 4, failed,
        )
        assert ok is False
        assert "repeats" in err

    def test_exact_repeat_different_order(self):
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["DATE", "CHERRY", "BANANA", "APPLE"],
            self.BOARD, 4, failed,
        )
        assert ok is False
        assert "repeats" in err

    # --- Fuzzy dedup tests ---

    def test_fuzzy_dedup_3_of_4_rejected(self):
        """3/4 overlap with a failed guess should be rejected."""
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY", "RED"],  # 3/4 overlap
            self.BOARD, 4, failed,
            fuzzy_dedup=True, max_overlap=3,
        )
        assert ok is False
        assert "shares 3/4" in err

    def test_fuzzy_dedup_2_of_4_allowed(self):
        """2/4 overlap should be allowed."""
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "RED", "BLUE"],  # 2/4 overlap
            self.BOARD, 4, failed,
            fuzzy_dedup=True, max_overlap=3,
        )
        assert ok is True

    def test_fuzzy_dedup_disabled(self):
        """With fuzzy_dedup=False, 3/4 overlap should be allowed."""
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "CHERRY", "RED"],
            self.BOARD, 4, failed,
            fuzzy_dedup=False,
        )
        assert ok is True

    def test_fuzzy_dedup_custom_threshold(self):
        """Custom max_overlap=2 should reject 2/4 overlap."""
        failed = [["APPLE", "BANANA", "CHERRY", "DATE"]]
        ok, err = _grounding_check(
            ["APPLE", "BANANA", "RED", "BLUE"],  # 2/4 overlap
            self.BOARD, 4, failed,
            fuzzy_dedup=True, max_overlap=2,
        )
        assert ok is False
        assert "shares 2/4" in err

    def test_case_insensitive(self):
        ok, err = _grounding_check(
            ["apple", "banana", "cherry", "date"],
            ["APPLE", "BANANA", "CHERRY", "DATE"], 4, [],
        )
        assert ok is True


# ---------------------------------------------------------------------------
# SolverMetrics tests
# ---------------------------------------------------------------------------


class TestSolverMetrics:
    def test_initial_state(self):
        m = SolverMetrics()
        assert m.failed_guesses == 0
        assert m.total_llm_calls == 0
        assert m.solve_order == []
        assert m.hallucinated_words == []

    def test_increment_failed(self):
        m = SolverMetrics()
        m.increment_failed()
        m.increment_failed()
        assert m.failed_guesses == 2

    def test_record_solve(self):
        m = SolverMetrics()
        m.record_solve(2)
        m.record_solve(0)
        assert m.solve_order == [2, 0]

    def test_record_hallucinations(self):
        m = SolverMetrics()
        m.record_hallucinations(["APPLE", "GHOST"], ["APPLE", "BANANA"])
        assert m.hallucinated_words == [["GHOST"]]

    def test_record_hallucinations_none(self):
        m = SolverMetrics()
        m.record_hallucinations(["APPLE", "BANANA"], ["APPLE", "BANANA"])
        assert m.hallucinated_words == []

    def test_to_dict(self):
        m = SolverMetrics()
        m.failed_guesses = 3
        m.total_llm_calls = 10
        m.wall_time_s = 1.23456
        d = m.to_dict()
        assert d["failed_guesses"] == 3
        assert d["total_llm_calls"] == 10
        assert d["wall_time_s"] == 1.23


# ---------------------------------------------------------------------------
# TraceRecorder tests
# ---------------------------------------------------------------------------


class TestTraceRecorder:
    def test_write_and_close(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        rec = TraceRecorder(path)
        rec.record("test_event", {"key": "value"})
        rec.close()

        import json
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["event"] == "test_event"
        assert obj["data"]["key"] == "value"
        assert "ts" in obj

    def test_context_manager(self, tmp_path):
        path = tmp_path / "trace2.jsonl"
        with TraceRecorder(path) as rec:
            rec.record("evt1", {})
            rec.record("evt2", {})
        # File should be closed
        assert rec._fh.closed

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_close_idempotent(self, tmp_path):
        path = tmp_path / "trace3.jsonl"
        rec = TraceRecorder(path)
        rec.close()
        rec.close()  # should not raise

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "subdir" / "deep" / "trace.jsonl"
        rec = TraceRecorder(path)
        rec.record("test", {})
        rec.close()
        assert path.exists()


# ---------------------------------------------------------------------------
# GVCSolver unit tests (mocked LLM)
# ---------------------------------------------------------------------------


def _mock_client():
    """Create a mock Client that returns predictable responses."""
    client = MagicMock()
    return client


class TestGVCSolverState:
    def test_reset_clears_all(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver._failed_guesses = [["A", "B", "C", "D"]]
        solver._sorted_failed = [["A", "B", "C", "D"]]
        solver._near_misses = [["A", "B", "C", "D"]]
        solver._failed_pair_counts = {("A", "B"): 3}
        solver._guesser_understanding = [["X"]]
        solver._validator_feedback = "test"

        solver.reset()

        assert solver._failed_guesses == []
        assert solver._sorted_failed == []
        assert solver._near_misses == []
        assert solver._failed_pair_counts == {}
        assert solver._guesser_understanding is None
        assert solver._validator_feedback is None

    def test_record_near_miss(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver.record_near_miss(["apple", "banana", "cherry", "date"])
        assert len(solver._near_misses) == 1
        assert solver._near_misses[0] == ["APPLE", "BANANA", "CHERRY", "DATE"]

    def test_record_near_miss_no_duplicates(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver.record_near_miss(["apple", "banana", "cherry", "date"])
        solver.record_near_miss(["date", "cherry", "banana", "apple"])  # same sorted
        assert len(solver._near_misses) == 1

    def test_record_failed_pairs(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver.record_failed_pairs(["A", "B", "C", "D"])
        # 4C2 = 6 pairs
        assert len(solver._failed_pair_counts) == 6
        assert solver._failed_pair_counts[("A", "B")] == 1
        assert solver._failed_pair_counts[("C", "D")] == 1

    def test_record_failed_pairs_accumulates(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver.record_failed_pairs(["A", "B", "C", "D"])
        solver.record_failed_pairs(["A", "B", "X", "Y"])
        # (A, B) appeared in both
        assert solver._failed_pair_counts[("A", "B")] == 2


class TestGVCSolverFeedback:
    def test_format_failed_feedback_empty(self):
        client = _mock_client()
        solver = GVCSolver(client)
        assert solver._format_failed_feedback() == ""

    def test_format_failed_feedback_basic(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver._sorted_failed = [["A", "B", "C", "D"]]
        feedback = solver._format_failed_feedback()
        assert "NOT part of the solution" in feedback
        assert "A, B, C, D" in feedback

    def test_format_failed_feedback_with_near_miss(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver._sorted_failed = [["A", "B", "C", "D"]]
        solver._near_misses = [["A", "B", "C", "D"]]
        feedback = solver._format_failed_feedback()
        assert "ONE AWAY" in feedback
        assert "ONE word here is wrong" in feedback

    def test_format_failed_feedback_with_elimination(self):
        client = _mock_client()
        solver = GVCSolver(client)
        solver._sorted_failed = [["A", "B", "C", "D"]]
        solver._failed_pair_counts = {("A", "B"): 3, ("C", "D"): 1}
        feedback = solver._format_failed_feedback()
        assert "ELIMINATION HINT" in feedback
        assert "A + B" in feedback
        # C + D should not appear (count < 3)
        assert "C + D" not in feedback

    def test_format_failed_feedback_elimination_cap(self):
        """Elimination hints should be capped at 5 pairs."""
        client = _mock_client()
        solver = GVCSolver(client)
        solver._sorted_failed = [["A"]]
        # Create 10 pairs with count >= 3
        for i in range(10):
            solver._failed_pair_counts[(f"W{i}", f"X{i}")] = 3
        feedback = solver._format_failed_feedback()
        # Count how many pair lines appear
        pair_lines = [line for line in feedback.split("\n") if " + " in line]
        assert len(pair_lines) <= 5
