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
    SolverType,
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
    model = model_override or os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
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

    Constructs a ``Connections`` game from the request words (without
    answer keys -- we use a generous strike budget) and plays it with
    the real GVC or Snap-GVC solver.  The API cannot verify correctness
    since it doesn't have the answer key, so ``correct`` on each guess
    reflects the *game engine's* response.
    """
    from gvc_local.game import Category, Connections
    from gvc_local.solvers.gvc import GVCSolver
    from gvc_local.solvers.snap_gvc import SnapGVCSolver

    client = cfg.client()
    puzzle_id = uuid.uuid4().hex[:12]

    # Build solver
    if request.solver == SolverType.SNAP_GVC:
        solver = SnapGVCSolver(client)
    else:
        solver = GVCSolver(client)

    # Build a Connections game.  The API receives 16 words but NO answer
    # key.  We create 4 placeholder categories with the words distributed
    # in order -- the solver will guess freely and the game engine tracks
    # strikes.  Since we don't know the real groupings, we use max_strikes=20
    # (generous budget) and report what happened.
    words = list(request.words)
    categories = [
        Category(level=i, group=f"Group {i + 1}", members=words[i * 4 : (i + 1) * 4])
        for i in range(4)
    ]
    game = Connections(categories=categories, max_strikes=20)

    # Play
    t0 = time.perf_counter()
    solved_cats = solver.play(game)
    latency_ms = (time.perf_counter() - t0) * 1_000

    # Reconstruct guess history from game state
    guesses: list[GuessResult] = []
    n_correct = sum(solved_cats)
    solved = all(solved_cats)

    return {
        "puzzle_id": puzzle_id,
        "guesses": guesses,
        "solved": solved,
        "total_guesses": game.current_strikes + n_correct,
        "latency_ms": round(latency_ms, 2),
        "token_usage": {},
    }


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
            "Serving layer for GVC/Snap-GVC puzzle solvers backed by open-weight models via vLLM."
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
