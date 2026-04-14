# SPDX-License-Identifier: GPL-2.0-or-later
"""Direct LLM provider clients (PRD §4.2.6).

Each provider implements LLMClientProtocol from agentic_runner.py:
    async complete(iteration_id, messages) -> LLMResponse

Providers:
  - AnthropicClient: Direct Anthropic SDK (anthropic package)
  - OpenAIClient: Direct OpenAI SDK (openai package) — also OpenRouter
  - GeminiClient: Google GenAI SDK (google-genai package)
  - OllamaClient: Local Ollama server (ollama package)
  - LiteLLMClient: Multi-provider via litellm (re-exported from llm_client)

All enforce temperature=0 and capture full metadata for agentic_trace/.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from apmode.backends.llm_client import LLMConfig, LLMResponse

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, type] = {}


def register_provider(name: str) -> Any:
    """Decorator to register a provider client class."""

    def decorator(cls: type) -> type:
        _PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


def create_llm_client(config: LLMConfig) -> Any:
    """Factory: create an LLM client from config.

    Resolves provider name to the appropriate client class.
    Falls back to LiteLLMClient if the provider is not directly supported.
    """
    provider = config.provider
    if provider in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[provider](config)

    # Fallback to litellm for unknown providers
    from apmode.backends.llm_client import LLMClient

    logger.info("Provider '%s' not in registry, falling back to litellm", provider)
    return LLMClient(config)


def available_providers() -> list[str]:
    """Return names of all registered providers + litellm fallback."""
    return sorted({*_PROVIDER_REGISTRY, "litellm"})


def _compute_payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Anthropic (direct SDK)
# ---------------------------------------------------------------------------


@register_provider("anthropic")
class AnthropicClient:
    """Direct Anthropic SDK client.

    Uses: ``anthropic.AsyncAnthropic().messages.create()``
    Requires: ``pip install anthropic`` or ``uv add anthropic``
    Auth: ANTHROPIC_API_KEY env var
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=None,  # reads ANTHROPIC_API_KEY from env
        )
        if self._config.api_base:
            client = anthropic.AsyncAnthropic(
                api_key=None,
                base_url=self._config.api_base,
            )

        # Anthropic uses separate system param, not a system message
        system_text = ""
        user_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": user_messages,
            "system": system_text,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await client.messages.create(**payload)
        elapsed = time.monotonic() - start

        raw_text = response.content[0].text if response.content else ""

        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model,
            model_version=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=_estimate_anthropic_cost(
                self._config.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# OpenAI (direct SDK) — also works for OpenRouter via base_url
# ---------------------------------------------------------------------------


@register_provider("openai")
class OpenAIClient:
    """Direct OpenAI SDK client.

    Uses: ``openai.AsyncOpenAI().chat.completions.create()``
    Requires: ``pip install openai`` or ``uv add openai``
    Auth: OPENAI_API_KEY env var

    For OpenRouter: set api_base="https://openrouter.ai/api/v1"
    and OPENROUTER_API_KEY as OPENAI_API_KEY.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {}
        if self._config.api_base:
            kwargs["base_url"] = self._config.api_base

        client = openai.AsyncOpenAI(**kwargs)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await client.chat.completions.create(**payload)
        elapsed = time.monotonic() - start

        raw_text = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage

        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model or self._config.model,
            model_version=getattr(response, "system_fingerprint", "") or response.model or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=_estimate_openai_cost(
                self._config.model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# Register openrouter as an alias that sets the base_url
@register_provider("openrouter")
class OpenRouterClient(OpenAIClient):
    """OpenRouter via OpenAI-compatible API.

    Requires: ``pip install openai``
    Auth: OPENROUTER_API_KEY or OPENAI_API_KEY env var
    """

    def __init__(self, config: LLMConfig) -> None:
        import os

        # Override api_base to OpenRouter endpoint
        patched = LLMConfig(
            model=config.model,
            provider="openrouter",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_base=config.api_base or "https://openrouter.ai/api/v1",
        )
        super().__init__(patched)
        # OpenRouter uses its own key; set OPENAI_API_KEY for the OpenAI SDK
        self._api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {"base_url": self._config.api_base}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        client = openai.AsyncOpenAI(**kwargs)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await client.chat.completions.create(**payload)
        elapsed = time.monotonic() - start

        raw_text = response.choices[0].message.content or "" if response.choices else ""
        usage = response.usage

        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model or self._config.model,
            model_version=getattr(response, "system_fingerprint", "") or response.model or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=_estimate_openai_cost(
                self._config.model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# Google Gemini (new google-genai SDK)
# ---------------------------------------------------------------------------


@register_provider("gemini")
class GeminiClient:
    """Google Gemini via the google-genai SDK.

    Uses: ``genai.Client().aio.models.generate_content()``
    Requires: ``pip install google-genai`` or ``uv add google-genai``
    Auth: GEMINI_API_KEY or GOOGLE_API_KEY env var
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client()

        # Separate system instruction from conversation turns.
        # Gemini supports multi-turn via list of Content objects with
        # role="user" or role="model" (Gemini's name for "assistant").
        system_text = ""
        contents: list[types.Content] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                gemini_role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        config = types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            system_instruction=system_text if system_text else None,
        )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "contents": [
                {
                    "role": c.role,
                    "text": c.parts[0].text if c.parts else "",
                }
                for c in contents
            ],
            "system_instruction": system_text,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await client.aio.models.generate_content(
            model=self._config.model,
            contents=contents,
            config=config,
        )
        elapsed = time.monotonic() - start

        raw_text = response.text or ""

        # Extract usage metadata
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return LLMResponse(
            raw_text=raw_text,
            model_id=self._config.model,
            model_version=self._config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_gemini_cost(self._config.model, input_tokens, output_tokens),
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# Ollama (local inference)
# ---------------------------------------------------------------------------


@register_provider("ollama")
class OllamaClient:
    """Local Ollama server client.

    Uses: ``ollama.AsyncClient().chat()``
    Requires: ``pip install ollama`` or ``uv add ollama``
    Auth: None (local server, default http://localhost:11434)
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        import ollama

        host = self._config.api_base or "http://localhost:11434"
        client = ollama.AsyncClient(host=host)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await client.chat(
            model=self._config.model,
            messages=messages,
            options={"temperature": self._config.temperature},
        )
        elapsed = time.monotonic() - start

        raw_text: str = response["message"]["content"]

        # Ollama provides eval_count and prompt_eval_count
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        return LLMResponse(
            raw_text=raw_text,
            model_id=self._config.model,
            model_version=response.get("model", self._config.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # local inference, no cost
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# Cost estimation helpers (approximate mid-2026 rates per 1M tokens)
# ---------------------------------------------------------------------------


def _estimate_anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates: dict[str, tuple[float, float]] = {
        "claude-opus-4-6": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-haiku-4-5-20251001": (0.80, 4.0),
    }
    in_rate, out_rate = rates.get(model, (3.0, 15.0))
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


def _estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates: dict[str, tuple[float, float]] = {
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "o3-mini": (1.10, 4.40),
    }
    in_rate, out_rate = rates.get(model, (2.50, 10.0))
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


def _estimate_gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates: dict[str, tuple[float, float]] = {
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-2.5-pro": (1.25, 10.0),
        "gemini-2.5-flash": (0.15, 0.60),
    }
    in_rate, out_rate = rates.get(model, (0.15, 0.60))
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)
