"""
Endpoint wrapper conforming to the upstream `rsallms.EndpointConfig` shape.

The upstream paper (Pandian et al., 2025) drives inference through an OpenAI-compatible
API.  vLLM, Groq, and Together AI all expose the same interface, so at the client level
we can swap GPT-4o for Llama / Qwen without touching the agent orchestration code.

Supported providers:

* **local** – vLLM running on your own hardware (default).
* **groq** – Free-tier cloud inference via ``GROQ_API_KEY``.
* **together** – Together AI inference + fine-tuning via ``TOGETHER_API_KEY``.

Usage::

    # Local vLLM
    cfg = EndpointConfig.llama31_8b(base_url="http://localhost:8000/v1")

    # Groq (free tier)
    cfg = EndpointConfig.groq_llama8b()

    # Together AI
    cfg = EndpointConfig.together_llama8b()

    client = cfg.client()
    resp = client.chat(messages=[...], temperature=0.3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Model identifiers by provider
# ---------------------------------------------------------------------------

# vLLM / HuggingFace model strings
LLAMA_31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_33_70B = "meta-llama/Meta-Llama-3.3-70B-Instruct"
QWEN_25_7B = "Qwen/Qwen2.5-7B-Instruct"

# Groq model IDs (free tier)
GROQ_LLAMA_31_8B = "llama-3.1-8b-instant"
GROQ_LLAMA_33_70B = "llama-3.3-70b-versatile"
GROQ_QWEN_25_7B = "qwen-2.5-7b-instruct"  # if unavailable, falls back to 32b

# Together AI model IDs
TOGETHER_LLAMA_31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TOGETHER_LLAMA_33_70B = "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo"
TOGETHER_QWEN_25_7B = "Qwen/Qwen2.5-7B-Instruct-Turbo"

# Provider base URLs
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"


@dataclass
class EndpointConfig:
    """Matches the shape of upstream rsallms.EndpointConfig."""

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM ignores, but the OpenAI SDK requires a non-empty string
    default_temperature: float = 0.3
    default_max_tokens: int = 1024
    # arbitrary extra kwargs passed through to the underlying chat call
    extra_body: dict[str, Any] = field(default_factory=dict)

    # --- Local (vLLM) constructors ----------------------------------------

    @classmethod
    def llama31_8b(cls, **overrides: Any) -> EndpointConfig:
        return cls(model=LLAMA_31_8B, **overrides)

    @classmethod
    def llama33_70b(cls, **overrides: Any) -> EndpointConfig:
        return cls(model=LLAMA_33_70B, **overrides)

    @classmethod
    def qwen25_7b(cls, **overrides: Any) -> EndpointConfig:
        return cls(model=QWEN_25_7B, **overrides)

    # --- Groq constructors ------------------------------------------------

    @classmethod
    def groq_llama8b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=GROQ_LLAMA_31_8B,
            base_url=GROQ_BASE_URL,
            api_key=os.environ.get("GROQ_API_KEY", ""),
            **overrides,
        )

    @classmethod
    def groq_llama70b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=GROQ_LLAMA_33_70B,
            base_url=GROQ_BASE_URL,
            api_key=os.environ.get("GROQ_API_KEY", ""),
            **overrides,
        )

    @classmethod
    def groq_qwen7b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=GROQ_QWEN_25_7B,
            base_url=GROQ_BASE_URL,
            api_key=os.environ.get("GROQ_API_KEY", ""),
            **overrides,
        )

    # --- Together AI constructors -----------------------------------------

    @classmethod
    def together_llama8b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=TOGETHER_LLAMA_31_8B,
            base_url=TOGETHER_BASE_URL,
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            **overrides,
        )

    @classmethod
    def together_llama70b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=TOGETHER_LLAMA_33_70B,
            base_url=TOGETHER_BASE_URL,
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            **overrides,
        )

    @classmethod
    def together_qwen7b(cls, **overrides: Any) -> EndpointConfig:
        return cls(
            model=TOGETHER_QWEN_25_7B,
            base_url=TOGETHER_BASE_URL,
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            **overrides,
        )

    # --- Client factory ---------------------------------------------------

    def client(self) -> Client:
        return Client(self)


class Client:
    """Thin wrapper over the OpenAI SDK with project-specific defaults.

    Kept intentionally small so we can diff against upstream's endpoint code later.
    """

    def __init__(self, cfg: EndpointConfig):
        self.cfg = cfg
        self._openai = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Single-turn chat. Returns the assistant's completion text."""
        resp = self._openai.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.default_temperature if temperature is None else temperature,
            max_tokens=self.cfg.default_max_tokens if max_tokens is None else max_tokens,
            extra_body={**self.cfg.extra_body, **extra.pop("extra_body", {})},
            **extra,
        )
        return resp.choices[0].message.content or ""

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Escape hatch when you need the full response object (logprobs, usage, etc.)."""
        return self._openai.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            **kwargs,
        )
