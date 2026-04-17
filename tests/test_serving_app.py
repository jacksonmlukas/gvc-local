"""Integration tests for the FastAPI app.

These tests exercise the HTTP layer without a live vLLM backend.  The
OpenAI client call inside the solver is mocked so we can verify routing,
validation, monitoring wiring, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gvc_local.serving.app import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


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
# POST /solve  — happy path with mocked vLLM
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

# Fake OpenAI-style response object returned by client.chat_raw().
_FAKE_RESPONSE = MagicMock()
_FAKE_RESPONSE.choices = [MagicMock()]
_FAKE_RESPONSE.choices[0].message.content = (
    '[{"words": ["BASS","TROUT","SALMON","COD"], "category": "Fish"},'
    ' {"words": ["JAZZ","BLUES","ROCK","POP"], "category": "Music"},'
    ' {"words": ["MARS","VENUS","SATURN","JUPITER"], "category": "Planets"},'
    ' {"words": ["RUBY","PEARL","JADE","AMBER"], "category": "Names"}]'
)
_FAKE_RESPONSE.usage = MagicMock()
_FAKE_RESPONSE.usage.prompt_tokens = 120
_FAKE_RESPONSE.usage.completion_tokens = 80
_FAKE_RESPONSE.usage.total_tokens = 200


@patch("gvc_local.serving.app.EndpointConfig.client")
def test_solve_happy_path(mock_client_factory: MagicMock, client: TestClient) -> None:
    mock_client = MagicMock()
    mock_client.chat_raw.return_value = _FAKE_RESPONSE
    mock_client_factory.return_value = mock_client

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


@patch("gvc_local.serving.app.EndpointConfig.client")
def test_solve_backend_error(mock_client_factory: MagicMock, client: TestClient) -> None:
    mock_client = MagicMock()
    mock_client.chat_raw.side_effect = ConnectionError("vLLM is down")
    mock_client_factory.return_value = mock_client

    resp = client.post("/solve", json={"words": VALID_WORDS, "solver": "gvc"})
    assert resp.status_code == 502
    assert "Solver error" in resp.json()["detail"]

    # Monitor should record the failure.
    metrics = client.get("/metrics").json()
    assert metrics["total_requests"] == 1
    assert metrics["error_count"] == 1


# ---------------------------------------------------------------------------
# POST /solve  — unparseable model output
# ---------------------------------------------------------------------------


@patch("gvc_local.serving.app.EndpointConfig.client")
def test_solve_unparseable_output(mock_client_factory: MagicMock, client: TestClient) -> None:
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock()]
    fake_resp.choices[0].message.content = "I don't know how to solve this puzzle."
    fake_resp.usage = MagicMock()
    fake_resp.usage.prompt_tokens = 50
    fake_resp.usage.completion_tokens = 20
    fake_resp.usage.total_tokens = 70

    mock_client = MagicMock()
    mock_client.chat_raw.return_value = fake_resp
    mock_client_factory.return_value = mock_client

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
