# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for LLM client with reproducibility tracing (PRD §4.2.6)."""

import json
from pathlib import Path

import pytest

from apmode.backends.llm_client import LLMConfig, LLMResponse, ReplayClient


def test_llm_config_enforces_temperature_zero() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LLMConfig(model="test", provider="anthropic", temperature=0.5)


def test_llm_config_accepts_temperature_zero() -> None:
    config = LLMConfig(model="test", provider="anthropic", temperature=0.0)
    assert config.temperature == 0.0


def test_llm_response_has_trace_fields() -> None:
    resp = LLMResponse(
        raw_text='{"transforms": []}',
        model_id="claude-sonnet-4-20250514",
        model_version="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.5,
        request_payload_hash="a" * 64,
    )
    assert resp.model_id == "claude-sonnet-4-20250514"
    assert resp.request_payload_hash == "a" * 64


async def test_replay_client_returns_cached_response(tmp_path: Path) -> None:
    cache_dir = tmp_path / "agentic_trace"
    cache_dir.mkdir()
    cached = {
        "raw_text": '{"transforms": ["swap_module"]}',
        "model_id": "test-model",
        "model_version": "v1",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.001,
        "wall_time_seconds": 1.0,
        "request_payload_hash": "b" * 64,
    }
    (cache_dir / "iter_001_cached_response.json").write_text(json.dumps(cached))

    client = ReplayClient(cache_dir)
    resp = await client.complete("iter_001", messages=[])
    assert resp.raw_text == '{"transforms": ["swap_module"]}'


async def test_replay_client_raises_on_missing_cache(tmp_path: Path) -> None:
    client = ReplayClient(tmp_path)
    with pytest.raises(FileNotFoundError):
        await client.complete("iter_999", messages=[])
