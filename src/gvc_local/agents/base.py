"""Base agent class for the GVC multi-agent puzzle solver.

Every specialised agent (Guesser, Validator, Snap) inherits from ``Agent``.
The base class handles:

* System + user message formatting for single-turn completions.
* Retry logic with exponential backoff on transient endpoint errors.
* Optional token-usage tracking for cost analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from gvc_local.endpoint import Client

logger = logging.getLogger(__name__)

# Transient errors that are safe to retry.
_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)

# Defaults — tuned for Groq free-tier rate limits (429s can last minutes)
_MAX_RETRIES = 6
_BACKOFF_BASE = 3.0  # seconds (3, 9, 27, 81, 243, 729 — capped at 90s)


class Agent:
    """Lightweight wrapper that pairs a ``Client`` with a role and system prompt.

    Parameters
    ----------
    client:
        The vLLM-backed chat client (see ``gvc_local.endpoint.Client``).
    role:
        A human-readable role name used in logging (e.g. ``"Guesser"``).
    system_prompt:
        The system message sent with every completion request.
    """

    def __init__(self, client: Client, role: str, system_prompt: str) -> None:
        self.client = client
        self.role = role
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(
        self,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send *system_prompt* + *user_message* and return the completion text.

        Retries up to ``_MAX_RETRIES`` times on transient network / server
        errors with exponential backoff.
        """
        messages = self._build_messages(user_message)
        return self._call_with_retry(messages, temperature=temperature, max_tokens=max_tokens)

    def respond_with_usage(
        self,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Like :meth:`respond` but also returns a ``usage`` dict.

        The dict has at minimum ``prompt_tokens`` and ``completion_tokens``
        keys (matching the OpenAI response schema).
        """
        messages = self._build_messages(user_message)
        return self._call_raw_with_retry(messages, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call ``client.chat`` with retry on transient failures."""
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                text = self.client.chat(messages, **kwargs)
                logger.debug("[%s] completion (%d chars)", self.role, len(text))
                return text
            except _RETRYABLE as exc:
                last_exc = exc
                wait = min(_BACKOFF_BASE**attempt, 90.0)  # cap at 90s
                logger.warning(
                    "[%s] transient error on attempt %d/%d, retrying in %.1fs: %s",
                    self.role,
                    attempt,
                    _MAX_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
        raise RuntimeError(f"[{self.role}] failed after {_MAX_RETRIES} retries") from last_exc

    def _call_raw_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call ``client.chat_raw`` with retry; return (text, usage_dict)."""
        kwargs: dict[str, Any] = {
            "temperature": temperature
            if temperature is not None
            else self.client.cfg.default_temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.client.cfg.default_max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self.client.chat_raw(messages, **kwargs)
                text = resp.choices[0].message.content or ""
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                logger.debug("[%s] completion with usage: %s", self.role, usage)
                return text, usage
            except _RETRYABLE as exc:
                last_exc = exc
                wait = min(_BACKOFF_BASE**attempt, 90.0)  # cap at 90s
                logger.warning(
                    "[%s] transient error (raw) on attempt %d/%d, retrying in %.1fs: %s",
                    self.role,
                    attempt,
                    _MAX_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"[{self.role}] raw call failed after {_MAX_RETRIES} retries"
        ) from last_exc

    def __repr__(self) -> str:
        return f"<Agent role={self.role!r} model={self.client.cfg.model!r}>"
