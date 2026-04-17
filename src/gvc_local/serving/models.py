"""Pydantic request/response models for the GVC-Local serving API.

These models define the contract between clients and the /solve, /health,
and /metrics endpoints.  All fields use strict types so callers get clear
validation errors rather than silent coercion.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SolverType(str, Enum):
    """Solver strategies mirroring the upstream CLI choices."""

    GVC = "gvc"
    SNAP_GVC = "snap_gvc"


# ---------------------------------------------------------------------------
# /solve
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    """Incoming puzzle to solve.

    ``words`` must contain exactly 16 unique strings (the 4x4 Connections
    board).  ``solver`` selects between the full GVC pipeline and the
    faster Snap-GVC variant.
    """

    words: list[str] = Field(
        ...,
        min_length=16,
        max_length=16,
        description="The 16 words on the Connections board.",
    )
    solver: SolverType = Field(
        SolverType.GVC,
        description="Which multi-agent strategy to run.",
    )
    model: str | None = Field(
        None,
        description="Override the default model served by vLLM (e.g. 'Qwen/Qwen2.5-7B-Instruct').",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature override.  Defaults to the endpoint config value.",
    )

    @field_validator("words")
    @classmethod
    def words_must_be_unique(cls, v: list[str]) -> list[str]:
        if len(set(w.upper() for w in v)) != 16:
            raise ValueError("All 16 words must be unique (case-insensitive).")
        return v


class GuessResult(BaseModel):
    """A single four-word guess and whether it matched a category."""

    words: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="The four words in this guess.",
    )
    category: str = Field(
        "",
        description="Matched category name (empty string if incorrect).",
    )
    correct: bool = Field(
        ...,
        description="Whether this guess exactly matched a hidden group.",
    )


class SolveResponse(BaseModel):
    """Full result payload returned by POST /solve."""

    puzzle_id: str = Field(
        ...,
        description="Opaque identifier for this solve attempt (UUID).",
    )
    guesses: list[GuessResult] = Field(
        default_factory=list,
        description="Ordered list of guesses the solver made.",
    )
    solved: bool = Field(
        ...,
        description="True if all four groups were found within the guess budget.",
    )
    total_guesses: int = Field(
        ...,
        ge=0,
        description="Total number of guesses attempted (correct + incorrect).",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time for the entire solve, in milliseconds.",
    )
    token_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate token counts: prompt_tokens, completion_tokens, total_tokens.",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = Field(
        ...,
        description="'ok' when the service is healthy.",
    )
    model: str = Field(
        ...,
        description="Model string currently configured on the vLLM backend.",
    )
    uptime_s: float = Field(
        ...,
        ge=0.0,
        description="Seconds since the API process started.",
    )
