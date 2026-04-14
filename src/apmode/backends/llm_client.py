# SPDX-License-Identifier: GPL-2.0-or-later
"""LLM client abstraction with reproducibility tracing (PRD §4.2.6).

Wraps litellm for provider-agnostic LLM access. Enforces temperature=0,
captures all metadata for agentic_trace/, and supports deterministic replay
from cached outputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = structlog.get_logger(__name__)


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

        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if self._config.api_base:
            payload["api_base"] = self._config.api_base

        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        start = time.monotonic()
        response = await litellm.acompletion(**payload)
        elapsed = time.monotonic() - start

        raw_text = response.choices[0].message.content or ""
        usage = response.usage

        return LLMResponse(
            raw_text=raw_text,
            model_id=response.model or self._config.model,
            model_version=getattr(response, "system_fingerprint", "") or response.model or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=_estimate_cost(
                self._config.model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            ),
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


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimation per 1M tokens. Conservative defaults."""
    # Rates per 1M tokens (approximate, mid-2026)
    rates: dict[str, tuple[float, float]] = {
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-haiku-4-5-20251001": (0.80, 4.0),
        "gpt-4o": (2.50, 10.0),
    }
    in_rate, out_rate = rates.get(model, (5.0, 15.0))
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)
