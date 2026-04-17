"""Unit tests for the Pydantic request/response models."""

import pytest
from pydantic import ValidationError

from gvc_local.serving.models import (
    GuessResult,
    HealthResponse,
    SolveRequest,
    SolveResponse,
    SolverType,
)

# A valid 16-word board (taken from a public NYT Connections example).
VALID_WORDS = [
    "BASS",
    "TROUT",
    "SALMON",
    "COD",
    "JAZZ",
    "BLUES",
    "ROCK",
    "POP",
    "MARS",
    "VENUS",
    "SATURN",
    "JUPITER",
    "RUBY",
    "PEARL",
    "JADE",
    "AMBER",
]


class TestSolveRequest:
    def test_valid_request(self) -> None:
        req = SolveRequest(words=VALID_WORDS, solver=SolverType.GVC)
        assert len(req.words) == 16
        assert req.solver == SolverType.GVC
        assert req.model is None
        assert req.temperature is None

    def test_solver_defaults_to_gvc(self) -> None:
        req = SolveRequest(words=VALID_WORDS)
        assert req.solver == SolverType.GVC

    def test_optional_overrides(self) -> None:
        req = SolveRequest(
            words=VALID_WORDS,
            solver="snap_gvc",
            model="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.7,
        )
        assert req.solver == SolverType.SNAP_GVC
        assert req.model == "Qwen/Qwen2.5-7B-Instruct"
        assert req.temperature == 0.7

    def test_rejects_fewer_than_16_words(self) -> None:
        with pytest.raises(ValidationError, match="too_short"):
            SolveRequest(words=VALID_WORDS[:15])

    def test_rejects_more_than_16_words(self) -> None:
        with pytest.raises(ValidationError, match="too_long"):
            SolveRequest(words=VALID_WORDS + ["EXTRA"])

    def test_rejects_duplicate_words(self) -> None:
        duped = VALID_WORDS[:15] + [VALID_WORDS[0]]
        with pytest.raises(ValidationError, match="unique"):
            SolveRequest(words=duped)

    def test_rejects_case_insensitive_duplicates(self) -> None:
        duped = VALID_WORDS[:15] + [VALID_WORDS[0].lower()]
        with pytest.raises(ValidationError, match="unique"):
            SolveRequest(words=duped)

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SolveRequest(words=VALID_WORDS, temperature=-0.1)
        with pytest.raises(ValidationError):
            SolveRequest(words=VALID_WORDS, temperature=2.5)

    def test_invalid_solver(self) -> None:
        with pytest.raises(ValidationError):
            SolveRequest(words=VALID_WORDS, solver="invalid")


class TestGuessResult:
    def test_correct_guess(self) -> None:
        g = GuessResult(words=["A", "B", "C", "D"], category="Fish", correct=True)
        assert g.correct is True
        assert g.category == "Fish"

    def test_wrong_word_count(self) -> None:
        with pytest.raises(ValidationError):
            GuessResult(words=["A", "B", "C"], category="Fish", correct=False)


class TestSolveResponse:
    def test_round_trip(self) -> None:
        resp = SolveResponse(
            puzzle_id="abc123",
            guesses=[
                GuessResult(words=["A", "B", "C", "D"], category="Fish", correct=True),
            ],
            solved=False,
            total_guesses=1,
            latency_ms=1234.5,
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        data = resp.model_dump()
        assert data["puzzle_id"] == "abc123"
        assert data["total_guesses"] == 1
        assert data["token_usage"]["total_tokens"] == 150


class TestHealthResponse:
    def test_construction(self) -> None:
        h = HealthResponse(status="ok", model="llama-8b", uptime_s=42.0)
        assert h.status == "ok"
        assert h.uptime_s == 42.0
