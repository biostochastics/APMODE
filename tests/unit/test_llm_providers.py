# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for direct LLM provider clients."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from apmode.backends.llm_client import LLMConfig, LLMResponse
from apmode.backends.llm_providers import (
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
    available_providers,
    create_llm_client,
)


@pytest.fixture
def anthropic_config() -> LLMConfig:
    return LLMConfig(model="claude-sonnet-4-20250514", provider="anthropic")


@pytest.fixture
def openai_config() -> LLMConfig:
    return LLMConfig(model="gpt-4o", provider="openai")


@pytest.fixture
def gemini_config() -> LLMConfig:
    return LLMConfig(model="gemini-2.5-flash", provider="gemini")


@pytest.fixture
def ollama_config() -> LLMConfig:
    return LLMConfig(model="llama3.1:8b", provider="ollama")


# --- Registry tests ---


def test_available_providers() -> None:
    providers = available_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "gemini" in providers
    assert "ollama" in providers
    assert "openrouter" in providers
    assert "litellm" in providers


def test_create_llm_client_anthropic(anthropic_config: LLMConfig) -> None:
    client = create_llm_client(anthropic_config)
    assert isinstance(client, AnthropicClient)


def test_create_llm_client_openai(openai_config: LLMConfig) -> None:
    client = create_llm_client(openai_config)
    assert isinstance(client, OpenAIClient)


def test_create_llm_client_gemini(gemini_config: LLMConfig) -> None:
    client = create_llm_client(gemini_config)
    assert isinstance(client, GeminiClient)


def test_create_llm_client_ollama(ollama_config: LLMConfig) -> None:
    client = create_llm_client(ollama_config)
    assert isinstance(client, OllamaClient)


def test_create_llm_client_unknown_falls_back_to_litellm() -> None:
    from apmode.backends.llm_client import LLMClient

    config = LLMConfig(model="some-model", provider="unknown_provider")
    client = create_llm_client(config)
    assert isinstance(client, LLMClient)


def test_openrouter_client_sets_base_url() -> None:
    config = LLMConfig(model="anthropic/claude-sonnet-4", provider="openrouter")
    client = OpenRouterClient(config)
    assert client._config.api_base == "https://openrouter.ai/api/v1"


# --- Anthropic mock tests ---


@pytest.mark.asyncio
async def test_anthropic_client_complete(anthropic_config: LLMConfig) -> None:
    import sys

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"transforms": [], "stop": true}')]
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

    mock_client_instance = AsyncMock()
    mock_client_instance.messages.create = AsyncMock(return_value=mock_response)

    fake_mod = MagicMock()
    fake_mod.AsyncAnthropic.return_value = mock_client_instance

    old = sys.modules.get("anthropic")
    sys.modules["anthropic"] = fake_mod
    try:
        client = AnthropicClient(anthropic_config)
        resp = await client.complete(
            "iter_001",
            [
                {"role": "system", "content": "You are a PK assistant."},
                {"role": "user", "content": "Test prompt."},
            ],
        )
    finally:
        if old is not None:
            sys.modules["anthropic"] = old
        else:
            sys.modules.pop("anthropic", None)

    assert resp.model_id == "claude-sonnet-4-20250514"
    assert resp.input_tokens == 100


# --- OpenAI mock tests ---


@pytest.mark.asyncio
async def test_openai_client_complete(openai_config: LLMConfig) -> None:
    import sys

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"transforms": [], "stop": true}'
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4o"
    mock_response.system_fingerprint = "fp_abc123"
    mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=80)

    mock_client_instance = AsyncMock()
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

    fake_mod = MagicMock()
    fake_mod.AsyncOpenAI.return_value = mock_client_instance

    old = sys.modules.get("openai")
    sys.modules["openai"] = fake_mod
    try:
        client = OpenAIClient(openai_config)
        resp = await client.complete(
            "iter_001",
            [{"role": "user", "content": "Test"}],
        )
    finally:
        if old is not None:
            sys.modules["openai"] = old
        else:
            sys.modules.pop("openai", None)

    assert resp.model_id == "gpt-4o"
    assert resp.model_version == "fp_abc123"
    assert resp.input_tokens == 200
    assert resp.output_tokens == 80


# --- Ollama mock tests ---


@pytest.mark.asyncio
async def test_ollama_client_complete(ollama_config: LLMConfig) -> None:
    import sys

    mock_response = {
        "message": {"content": '{"transforms": [], "stop": true}'},
        "model": "llama3.1:8b",
        "prompt_eval_count": 50,
        "eval_count": 30,
    }

    mock_client_instance = AsyncMock()
    mock_client_instance.chat = AsyncMock(return_value=mock_response)

    fake_mod = MagicMock()
    fake_mod.AsyncClient.return_value = mock_client_instance

    old = sys.modules.get("ollama")
    sys.modules["ollama"] = fake_mod
    try:
        client = OllamaClient(ollama_config)
        resp = await client.complete(
            "iter_001",
            [{"role": "user", "content": "Test"}],
        )
    finally:
        if old is not None:
            sys.modules["ollama"] = old
        else:
            sys.modules.pop("ollama", None)

    assert resp.model_id == "llama3.1:8b"
    assert resp.input_tokens == 50
    assert resp.output_tokens == 30
    assert resp.cost_usd == 0.0  # local inference
    assert resp.request_payload_hash  # non-empty


# --- Response format consistency ---


def test_all_providers_return_llm_response_type() -> None:
    """All provider clients declare the same return type."""
    import inspect

    for cls in [AnthropicClient, OpenAIClient, GeminiClient, OllamaClient]:
        sig = inspect.signature(cls.complete)
        assert sig.return_annotation in (LLMResponse, "LLMResponse")
