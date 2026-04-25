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

import contextlib
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from apmode.backends.llm_client import LLMConfig, LLMResponse, _await_llm

if TYPE_CHECKING:
    from collections.abc import Callable

    from apmode.backends.agentic_runner import LLMClientProtocol

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# Values are provider factories: callables that take an LLMConfig and return
# an LLMClientProtocol. Classes satisfy this via their ``__init__`` signature,
# so ``@register_provider(...)`` on ``class FooClient`` inserts ``FooClient``
# directly. Using Callable (not ``type[LLMClientProtocol]``) avoids mypy's
# Protocol-has-no-__init__ limitation at the call site.
_PROVIDER_REGISTRY: dict[str, Callable[[LLMConfig], LLMClientProtocol]] = {}

_ClientT = TypeVar("_ClientT", bound="LLMClientProtocol")


def register_provider(name: str) -> Callable[[type[_ClientT]], type[_ClientT]]:
    """Decorator to register a provider client class.

    Preserves the concrete subclass type so ``@register_provider("x") class
    Foo(...): ...`` still yields ``Foo`` (not ``type[LLMClientProtocol]``),
    which matters for subclasses like ``OpenRouterClient(OpenAIClient)``.
    """

    def decorator(cls: type[_ClientT]) -> type[_ClientT]:
        _PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


def create_llm_client(config: LLMConfig) -> LLMClientProtocol:
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

    The underlying ``AsyncAnthropic`` (which owns an ``httpx.AsyncClient``
    connection pool) is constructed lazily on first ``complete()`` and
    reused across calls. Per-call construction would leak pool sockets
    when ``asyncio.CancelledError`` or a timeout drops the client before
    its ``__del__`` runs; the agentic runner manages closure via an
    ``contextlib.AsyncExitStack`` — call :meth:`aclose` from your
    teardown path.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {
                "api_key": None,
                "timeout": self._config.timeout_seconds,
            }
            if self._config.api_base:
                kwargs["base_url"] = self._config.api_base
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def aclose(self) -> None:
        """Release the underlying httpx connection pool, if one was opened."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        client = self._get_client()
        timeout_s = self._config.timeout_seconds

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
        response = await _await_llm(client.messages.create(**payload), timeout=timeout_s)
        elapsed = time.monotonic() - start

        raw_text = response.content[0].text if response.content else ""

        # Anthropic's Messages API does not expose a deterministic
        # ``system_fingerprint``, so the escrow falls back to the
        # response's request ID — a unique server-side identifier of
        # THIS call. ICH M15 reproducibility requires a fingerprint the
        # sponsor can re-reference.
        request_id = getattr(response, "_request_id", None) or getattr(response, "id", None)
        model_version = f"{response.model}@{request_id}" if request_id else response.model
        if not request_id:
            logger.warning(
                "anthropic.model_version.best_effort",
                reason="response has no request_id or id field",
                model_id=response.model,
            )

        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model,
            model_version=model_version,
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
    and OPENROUTER_API_KEY as OPENAI_API_KEY. The underlying httpx
    pool is reused across calls; see :class:`AnthropicClient` for the
    rationale and :meth:`aclose` for shutdown.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None
        # Default to no extra kwargs; OpenRouter overrides this in
        # ``__init__`` to inject the API key.
        self._extra_kwargs: dict[str, Any] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {"timeout": self._config.timeout_seconds}
            if self._config.api_base:
                kwargs["base_url"] = self._config.api_base
            kwargs.update(self._extra_kwargs)
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        timeout_s = self._config.timeout_seconds
        client = self._get_client()

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await _await_llm(client.chat.completions.create(**payload), timeout=timeout_s)
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

    Inherits :meth:`OpenAIClient.complete`, :meth:`_get_client`, and
    :meth:`aclose` verbatim — the only difference is the base URL +
    explicit API key plumbed through ``_extra_kwargs`` so the lazy
    client builder picks them up.
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
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self._extra_kwargs["api_key"] = api_key


# ---------------------------------------------------------------------------
# Google Gemini (new google-genai SDK)
# ---------------------------------------------------------------------------


@register_provider("gemini")
class GeminiClient:
    """Google Gemini via the google-genai SDK.

    Uses: ``genai.Client().aio.models.generate_content()``
    Requires: ``pip install google-genai`` or ``uv add google-genai``
    Auth: GEMINI_API_KEY or GOOGLE_API_KEY env var

    The SDK's ``Client`` wraps an httpx pool; reuse one across calls
    and let :meth:`aclose` drop it on shutdown.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    async def aclose(self) -> None:
        # google-genai exposes a synchronous ``close`` on Client (and
        # variants exist on internal sub-clients). Best-effort: call
        # whichever is available; ignore SDK-version skew.
        if self._client is not None:
            for closer_name in ("aclose", "close"):
                closer = getattr(self._client, closer_name, None)
                if closer is None:
                    continue
                with contextlib.suppress(Exception):
                    result = closer()
                    if hasattr(result, "__await__"):
                        await result
                break
            self._client = None

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()

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
        response = await _await_llm(
            client.aio.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=config,
            ),
            timeout=self._config.timeout_seconds,
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

    The async client owns an httpx pool. Reuse it across calls and
    drop it via :meth:`aclose`.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import ollama

            host = self._config.api_base or "http://localhost:11434"
            self._client = ollama.AsyncClient(host=host)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            # ollama-python's AsyncClient exposes ``_client`` (httpx);
            # newer versions add a top-level close coroutine. Try both.
            for attempt in (
                getattr(self._client, "aclose", None),
                getattr(self._client, "close", None),
            ):
                if attempt is None:
                    continue
                with contextlib.suppress(Exception):
                    result = attempt()
                    if hasattr(result, "__await__"):
                        await result
                break
            self._client = None

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        client = self._get_client()

        # Hash the same payload we send — including options — so the recorded
        # request_payload_hash fingerprints the full request (PRD §4.2.6
        # model-version escrow).
        options = {"temperature": self._config.temperature}
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "options": options,
        }
        payload_hash = _compute_payload_hash(payload)

        start = time.monotonic()
        response = await _await_llm(
            client.chat(
                model=self._config.model,
                messages=messages,
                options=options,
            ),
            timeout=self._config.timeout_seconds,
        )
        elapsed = time.monotonic() - start

        # ollama >=0.4 returns a Pydantic ChatResponse; older versions
        # returned a raw dict. Prefer attribute access, fall back to
        # dict-style access with explicit isinstance guards so a
        # malformed response (missing ``message`` key, or a non-dict
        # ``message``) raises a typed error instead of leaking
        # ``KeyError``/``TypeError`` mid-iteration.
        message = getattr(response, "message", None)
        if message is not None:
            raw_text = str(getattr(message, "content", "") or "")
            input_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
            output_tokens = int(getattr(response, "eval_count", 0) or 0)
            model_version = getattr(response, "model", self._config.model)
        elif isinstance(response, dict):
            msg_dict = response.get("message")
            if not isinstance(msg_dict, dict):
                raise ValueError(
                    f"ollama returned non-dict 'message' field: {type(msg_dict).__name__}"
                )
            raw_text = str(msg_dict.get("content", "") or "")
            input_tokens = int(response.get("prompt_eval_count", 0) or 0)
            output_tokens = int(response.get("eval_count", 0) or 0)
            model_version = str(response.get("model", self._config.model))
        else:
            raise ValueError(f"unrecognised ollama response type: {type(response).__name__}")

        return LLMResponse(
            raw_text=raw_text,
            model_id=self._config.model,
            model_version=model_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # local inference, no cost
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


# ---------------------------------------------------------------------------
# Cost estimation helpers (approximate mid-2026 rates per 1M tokens)
# ---------------------------------------------------------------------------


def _compute_cost(
    rates: dict[str, tuple[float, float]],
    fallback: tuple[float, float],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute USD cost from per-1M-token (input, output) rates.

    Single source of truth for token-cost arithmetic; per-provider
    helpers below carry only the rate tables and a default. Keeps the
    refactor cheap to delete later if pricing moves to a remote
    metadata source.
    """
    in_rate, out_rate = rates.get(model, fallback)
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


_ANTHROPIC_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}
_OPENAI_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
}
_GEMINI_RATES: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
}


def _estimate_anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return _compute_cost(_ANTHROPIC_RATES, (3.0, 15.0), model, input_tokens, output_tokens)


def _estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return _compute_cost(_OPENAI_RATES, (2.50, 10.0), model, input_tokens, output_tokens)


def _estimate_gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return _compute_cost(_GEMINI_RATES, (0.15, 0.60), model, input_tokens, output_tokens)
