"""FastAPI application for the GVC-Local puzzle-solving API.

This module wires together the endpoint config, solver dispatch, and
request monitoring into a production-ready ASGI app.  vLLM runs as a
separate service; this API talks to it via the OpenAI-compatible client
defined in ``gvc_local.endpoint``.

Run locally::

    uvicorn gvc_local.serving.app:create_app --factory --host 0.0.0.0 --port 8080

Environment variables
---------------------
VLLM_BASE_URL   Base URL for the vLLM OpenAI-compatible endpoint.
                Default: ``http://localhost:8000/v1``
VLLM_MODEL      Model identifier served by vLLM.
                Default: ``meta-llama/Meta-Llama-3.1-8B-Instruct``
API_LOG_LEVEL   Logging level (DEBUG, INFO, WARNING, ...).
                Default: ``INFO``
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gvc_local.endpoint import EndpointConfig

from .models import (
    GuessResult,
    HealthResponse,
    SolveRequest,
    SolveResponse,
)
from .monitoring import RequestMonitor

logger = logging.getLogger("gvc_local.serving")

# ---------------------------------------------------------------------------
# Application state kept in app.state so it's accessible via dep injection.
# ---------------------------------------------------------------------------

_BOOT_TIME: float = 0.0
"""Module-level start timestamp, set during the lifespan startup."""


def _build_endpoint_config(model_override: str | None = None) -> EndpointConfig:
    """Construct an ``EndpointConfig`` from env vars with optional model override."""
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = model_override or os.getenv(
        "VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    return EndpointConfig(model=model, base_url=base_url)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log configuration on startup; clean up on shutdown."""
    global _BOOT_TIME  # noqa: PLW0603
    _BOOT_TIME = time.monotonic()

    cfg = _build_endpoint_config()
    app.state.endpoint_config = cfg
    app.state.monitor = RequestMonitor(buffer_size=5000)

    logger.info(
        "GVC-Local API starting  model=%s  endpoint=%s",
        cfg.model,
        cfg.base_url,
    )
    yield
    logger.info("GVC-Local API shutting down")


# ---------------------------------------------------------------------------
# Solver dispatch
# ---------------------------------------------------------------------------


def _run_solver(
    request: SolveRequest,
    cfg: EndpointConfig,
) -> dict[str, Any]:
    """Execute the chosen solver against the vLLM backend.

    Returns a dict with keys consumed by ``SolveResponse``.

    Currently this performs a *single-round demonstration call* to prove
    end-to-end connectivity.  Full GVC/Snap-GVC orchestration will land
    once the agent modules (M2 on the roadmap) are wired in.  The response
    shape is stable so downstream clients can integrate now.
    """
    client = cfg.client()
    puzzle_id = uuid.uuid4().hex[:12]

    board_str = ", ".join(request.words)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an expert solver for the NYT Connections puzzle. "
                "You are given 16 words and must find four groups of four "
                "related words.  Return ONLY a JSON array of four objects, "
                'each with "words" (list of 4 strings) and "category" (string).'
            ),
        },
        {
            "role": "user",
            "content": f"Find the four groups in: [{board_str}]",
        },
    ]

    temperature = request.temperature if request.temperature is not None else cfg.default_temperature

    t0 = time.perf_counter()
    raw = client.chat_raw(
        messages=messages,
        temperature=temperature,
        max_tokens=cfg.default_max_tokens,
    )
    latency_ms = (time.perf_counter() - t0) * 1_000

    # Extract token usage from the raw OpenAI-style response.
    usage: dict[str, int] = {}
    if raw.usage:
        usage = {
            "prompt_tokens": raw.usage.prompt_tokens or 0,
            "completion_tokens": raw.usage.completion_tokens or 0,
            "total_tokens": raw.usage.total_tokens or 0,
        }

    # Parse model output into guesses.  The model may not return valid JSON
    # on every attempt -- we handle that gracefully.
    guesses = _parse_model_guesses(raw.choices[0].message.content or "")

    solved = len(guesses) >= 4 and all(g.correct for g in guesses[:4])

    return {
        "puzzle_id": puzzle_id,
        "guesses": guesses,
        "solved": solved,
        "total_guesses": len(guesses),
        "latency_ms": round(latency_ms, 2),
        "token_usage": usage,
    }


def _parse_model_guesses(content: str) -> list[GuessResult]:
    """Best-effort parse of the model's JSON output into ``GuessResult`` items.

    If parsing fails we return an empty list rather than crashing -- the
    caller will report ``solved=False`` and the monitoring layer captures
    the failure.
    """
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON array from markdown fences.
        import re

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(data, list):
        return []

    results: list[GuessResult] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        words = item.get("words", [])
        if not isinstance(words, list) or len(words) != 4:
            continue
        results.append(
            GuessResult(
                words=[str(w) for w in words],
                category=str(item.get("category", "")),
                # Without the answer key we can't verify correctness here.
                # The eval harness checks this; the API just reports what
                # the model returned.  Mark as correct=False by default.
                correct=False,
            )
        )
    return results


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Application factory.  Use with ``uvicorn --factory``."""
    log_level = os.getenv("API_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    app = FastAPI(
        title="GVC-Local API",
        version="0.1.0",
        description=(
            "Serving layer for GVC/Snap-GVC puzzle solvers backed by "
            "open-weight models via vLLM."
        ),
        lifespan=_lifespan,
    )

    # -- Middleware ---------------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handler -------------------------------------------------

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check API logs for details."},
        )

    # -- Routes ------------------------------------------------------------

    @app.post("/solve", response_model=SolveResponse)
    async def solve(req: SolveRequest) -> SolveResponse:
        """Run a solver against 16 Connections words and return the results."""
        cfg: EndpointConfig = app.state.endpoint_config
        monitor: RequestMonitor = app.state.monitor

        # Allow per-request model override.
        if req.model:
            cfg = _build_endpoint_config(model_override=req.model)

        record = monitor.start(solver=req.solver.value, model=cfg.model)

        try:
            result = _run_solver(req, cfg)
            monitor.finish(
                record,
                tokens_in=result.get("token_usage", {}).get("prompt_tokens", 0),
                tokens_out=result.get("token_usage", {}).get("completion_tokens", 0),
                success=True,
            )
            return SolveResponse(**result)

        except Exception as exc:
            monitor.finish(record, success=False, error=str(exc))
            logger.exception("Solver failed for request: %s", req.solver.value)
            raise HTTPException(
                status_code=502,
                detail=f"Solver error: {exc}",
            ) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe.  Returns the configured model and uptime."""
        cfg: EndpointConfig = app.state.endpoint_config
        uptime = time.monotonic() - _BOOT_TIME
        return HealthResponse(
            status="ok",
            model=cfg.model,
            uptime_s=round(uptime, 2),
        )

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """Aggregate monitoring stats over the in-memory request buffer."""
        monitor: RequestMonitor = app.state.monitor
        return monitor.summary()

    return app
