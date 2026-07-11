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


def test_openrouter_client_preserves_custom_config() -> None:
    config = LLMConfig(
        model="anthropic/claude-sonnet-4",
        provider="openrouter",
        api_base="https://custom.example/v1",
        timeout_seconds=7.5,
        max_tokens=512,
    )
    client = OpenRouterClient(config)
    assert client._config.api_base == "https://custom.example/v1"
    assert client._config.timeout_seconds == 7.5
    assert client._config.max_tokens == 512


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


@pytest.mark.asyncio
async def test_all_providers_return_llm_response_type() -> None:
    """Every provider's ``complete()`` actually returns an ``LLMResponse``.

    Behavioral (not signature) check: each client is driven with a mocked
    vendor SDK and the runtime type + populated token counts are asserted,
    so a provider that returned a dict / vendor object would fail here.
    """
    import contextlib
    import sys
    from collections.abc import Iterator

    @contextlib.contextmanager
    def _swap_modules(**mods: object) -> Iterator[None]:
        saved = {name: sys.modules.get(name) for name in mods}
        sys.modules.update(mods)
        try:
            yield
        finally:
            for name, prev in saved.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev

    _JSON = '{"transforms": [], "stop": true}'
    messages = [
        {"role": "system", "content": "You are a PK assistant."},
        {"role": "user", "content": "Test."},
    ]

    # --- Anthropic ---
    anthropic_resp = MagicMock()
    anthropic_resp.content = [MagicMock(text=_JSON)]
    anthropic_resp.model = "claude-sonnet-4-20250514"
    anthropic_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    anthropic_inst = AsyncMock()
    anthropic_inst.messages.create = AsyncMock(return_value=anthropic_resp)
    anthropic_mod = MagicMock()
    anthropic_mod.AsyncAnthropic.return_value = anthropic_inst
    with _swap_modules(anthropic=anthropic_mod):
        resp = await AnthropicClient(
            LLMConfig(model="claude-sonnet-4-20250514", provider="anthropic")
        ).complete("iter_001", messages)
    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == 100

    # --- OpenAI ---
    openai_resp = MagicMock()
    openai_choice = MagicMock()
    openai_choice.message.content = _JSON
    openai_resp.choices = [openai_choice]
    openai_resp.model = "gpt-4o"
    openai_resp.system_fingerprint = "fp_abc123"
    openai_resp.usage = MagicMock(prompt_tokens=200, completion_tokens=80)
    openai_inst = AsyncMock()
    openai_inst.chat.completions.create = AsyncMock(return_value=openai_resp)
    openai_mod = MagicMock()
    openai_mod.AsyncOpenAI.return_value = openai_inst
    with _swap_modules(openai=openai_mod):
        resp = await OpenAIClient(LLMConfig(model="gpt-4o", provider="openai")).complete(
            "iter_001", messages
        )
    assert isinstance(resp, LLMResponse)
    assert resp.output_tokens == 80

    # --- Gemini (google-genai SDK) ---
    gemini_resp = MagicMock()
    gemini_resp.text = _JSON
    gemini_resp.usage_metadata = MagicMock(prompt_token_count=42, candidates_token_count=17)
    gemini_inst = MagicMock()
    gemini_inst.aio.models.generate_content = AsyncMock(return_value=gemini_resp)
    fake_types = MagicMock()
    fake_types.Content.side_effect = lambda role, parts: MagicMock(role=role, parts=parts)
    fake_types.Part.from_text.side_effect = lambda text: MagicMock(text=text)
    fake_genai = MagicMock()
    fake_genai.types = fake_types
    fake_genai.Client.return_value = gemini_inst
    fake_google = MagicMock()
    fake_google.genai = fake_genai
    with _swap_modules(
        google=fake_google,
        **{"google.genai": fake_genai, "google.genai.types": fake_types},
    ):
        resp = await GeminiClient(LLMConfig(model="gemini-2.5-flash", provider="gemini")).complete(
            "iter_001", messages
        )
    assert isinstance(resp, LLMResponse)
    assert resp.input_tokens == 42

    # --- Ollama (local inference) ---
    ollama_inst = AsyncMock()
    ollama_inst.chat = AsyncMock(
        return_value={
            "message": {"content": _JSON},
            "model": "llama3.1:8b",
            "prompt_eval_count": 50,
            "eval_count": 30,
        }
    )
    ollama_mod = MagicMock()
    ollama_mod.AsyncClient.return_value = ollama_inst
    with _swap_modules(ollama=ollama_mod):
        resp = await OllamaClient(LLMConfig(model="llama3.1:8b", provider="ollama")).complete(
            "iter_001", [{"role": "user", "content": "Test"}]
        )
    assert isinstance(resp, LLMResponse)
    assert resp.cost_usd == 0.0  # local inference is free
