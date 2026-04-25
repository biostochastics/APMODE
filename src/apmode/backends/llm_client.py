# SPDX-License-Identifier: GPL-2.0-or-later
"""LLM client abstraction with reproducibility tracing (PRD §4.2.6).

Wraps litellm for provider-agnostic LLM access. Enforces temperature=0,
captures all metadata for agentic_trace/, and supports deterministic replay
from cached outputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = structlog.get_logger(__name__)


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM provider call exceeds ``LLMConfig.timeout_seconds``.

    Subclasses the built-in ``TimeoutError`` so existing ``asyncio.wait_for``
    handlers catch it naturally.
    """


async def _await_llm[T](awaitable: Awaitable[T], timeout: float) -> T:
    """Wrap an LLM provider coroutine with an outer deadline.

    Uses :func:`asyncio.timeout` (3.11+) instead of
    :func:`asyncio.wait_for` so the helper composes cleanly inside an
    enclosing :class:`asyncio.TaskGroup` (the timeout deadline is local
    to this scope; an outer cancellation propagates as a plain
    :class:`asyncio.CancelledError` rather than being misclassified as a
    timeout). The native SDK timeout (where supported) still closes the
    HTTP socket; this wrapper additionally guarantees the coroutine
    cannot block the agentic loop past the policy limit, even if the
    SDK retries internally or ignores its own kwarg.
    """
    try:
        async with asyncio.timeout(timeout):
            return await awaitable
    except TimeoutError as exc:
        msg = f"LLM request timed out after {timeout}s"
        raise LLMTimeoutError(msg) from exc


class LLMConfig(BaseModel):
    """Configuration for the LLM client.

    PRD §4.2.6: temperature must be 0 for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    provider: str
    temperature: float = 0.0
    max_tokens: int = 4096
    api_base: str | None = None
    # Hard ceiling on a single LLM API call so a hung provider cannot
    # block the agentic loop indefinitely. Wraps both the native SDK
    # timeout (where supported) and an outer ``asyncio.wait_for`` to
    # cover SDKs whose retry/backoff paths ignore the kwarg.
    timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def enforce_temperature_zero(self) -> LLMConfig:
        if self.temperature != 0.0:
            msg = (
                f"temperature must be 0.0 for agentic reproducibility "
                f"(PRD §4.2.6), got {self.temperature}"
            )
            raise ValueError(msg)
        return self


class LLMResponse(BaseModel):
    """Captures all metadata for agentic_trace/ bundle artifacts."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    model_id: str
    model_version: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    wall_time_seconds: float = Field(ge=0.0)
    request_payload_hash: str


class LLMClient:
    """Provider-agnostic LLM client via litellm.

    All requests use temperature=0 and record full metadata for reproducibility.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Send a completion request and return a traced response."""
        import litellm

        # Split the SDK per-attempt timeout from the outer wall-clock
        # cap so litellm's internal retry has a chance to land within
        # budget rather than being silently aborted by the outer
        # deadline. Half the outer cap leaves headroom for one retry.
        sdk_timeout = max(1.0, self._config.timeout_seconds / 2)
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": sdk_timeout,
            # Keep the retry budget deterministic; the outer
            # ``asyncio.timeout`` is the wall-clock backstop.
            "num_retries": 1,
        }
        if self._config.api_base:
            payload["api_base"] = self._config.api_base

        payload_hash = hashlib.sha256(
            json.dumps(
                {k: v for k, v in payload.items() if k not in {"timeout", "num_retries"}},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        start = time.monotonic()
        response = await _await_llm(
            litellm.acompletion(**payload),
            timeout=self._config.timeout_seconds,
        )
        elapsed = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        # Cost estimation was previously routed through a litellm-side
        # rate table that duplicated the per-provider tables in
        # ``llm_providers.py``. The litellm fallback path is only hit
        # when no direct provider is registered; in that situation we
        # have no reliable per-model rate (any number we picked would
        # be misleading), so we return 0.0 and surface the truth in
        # ``input_tokens``/``output_tokens`` for the audit trail.
        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model or self._config.model,
            model_version=getattr(response, "system_fingerprint", "") or response.model or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=0.0,
            wall_time_seconds=round(elapsed, 3),
            request_payload_hash=payload_hash,
        )


class ReplayClient:
    """Deterministic replay from cached agentic_trace/ outputs.

    Used for reproducibility: replays LLM responses from a prior run
    instead of making live API calls.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    async def complete(
        self,
        iteration_id: str,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        """Return cached response for the given iteration."""
        cache_file = self._cache_dir / f"{iteration_id}_cached_response.json"
        if not cache_file.exists():
            msg = f"No cached response for iteration {iteration_id} at {cache_file}"
            raise FileNotFoundError(msg)

        data: dict[str, Any] = json.loads(cache_file.read_text())
        return LLMResponse(**data)
