"""Integration tests for the FastAPI app.

These tests exercise the HTTP layer without a live vLLM backend.  The
solver dispatch is mocked so we can verify routing, validation,
monitoring wiring, and error handling in isolation.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gvc_local.serving.app import create_app
from gvc_local.serving.models import GuessResult


@pytest.fixture()
def client() -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert body["uptime_s"] >= 0


# ---------------------------------------------------------------------------
# GET /metrics  (empty)
# ---------------------------------------------------------------------------


def test_metrics_empty(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] == 0
    assert body["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# POST /solve  — happy path with mocked solver dispatch
# ---------------------------------------------------------------------------

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

# Fake result dict matching the shape returned by _run_solver.
_FAKE_SOLVER_RESULT = {
    "puzzle_id": "test123",
    "guesses": [
        GuessResult(words=["BASS", "TROUT", "SALMON", "COD"], category="Fish", correct=True),
        GuessResult(words=["JAZZ", "BLUES", "ROCK", "POP"], category="Music", correct=True),
        GuessResult(words=["MARS", "VENUS", "SATURN", "JUPITER"], category="Planets", correct=True),
        GuessResult(words=["RUBY", "PEARL", "JADE", "AMBER"], category="Names", correct=True),
    ],
    "solved": True,
    "total_guesses": 4,
    "latency_ms": 1234.56,
    "token_usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
}


@patch("gvc_local.serving.app._run_solver")
def test_solve_happy_path(mock_run_solver: MagicMock, client: TestClient) -> None:
    mock_run_solver.return_value = _FAKE_SOLVER_RESULT

    resp = client.post("/solve", json={"words": VALID_WORDS, "solver": "gvc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_guesses"] == 4
    assert len(body["guesses"]) == 4
    assert body["latency_ms"] >= 0
    assert body["token_usage"]["total_tokens"] == 200

    # Verify that the monitor was updated.
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 1
    assert metrics["error_count"] == 0


# ---------------------------------------------------------------------------
# POST /solve  — validation errors
# ---------------------------------------------------------------------------


def test_solve_rejects_wrong_word_count(client: TestClient) -> None:
    resp = client.post("/solve", json={"words": VALID_WORDS[:10], "solver": "gvc"})
    assert resp.status_code == 422


def test_solve_rejects_invalid_solver(client: TestClient) -> None:
    resp = client.post("/solve", json={"words": VALID_WORDS, "solver": "not_a_solver"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /solve  — backend failure
# ---------------------------------------------------------------------------


@patch("gvc_local.serving.app._run_solver")
def test_solve_backend_error(mock_run_solver: MagicMock, client: TestClient) -> None:
    mock_run_solver.side_effect = ConnectionError("vLLM is down")

    resp = client.post("/solve", json={"words": VALID_WORDS, "solver": "gvc"})
    assert resp.status_code == 502
    assert "Solver error" in resp.json()["detail"]

    # Monitor should record the failure.
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 1
    assert metrics["error_count"] == 1


# ---------------------------------------------------------------------------
# POST /solve  — solver returns unsolved
# ---------------------------------------------------------------------------


@patch("gvc_local.serving.app._run_solver")
def test_solve_unsolved(mock_run_solver: MagicMock, client: TestClient) -> None:
    mock_run_solver.return_value = {
        "puzzle_id": "test456",
        "guesses": [],
        "solved": False,
        "total_guesses": 0,
        "latency_ms": 500.0,
        "token_usage": {},
    }

    resp = client.post("/solve", json={"words": VALID_WORDS, "solver": "snap_gvc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_guesses"] == 0
    assert body["solved"] is False


# ---------------------------------------------------------------------------
# CORS headers present
# ---------------------------------------------------------------------------


def test_cors_headers(client: TestClient) -> None:
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" in resp.headers
