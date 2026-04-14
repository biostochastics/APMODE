# SPDX-License-Identifier: GPL-2.0-or-later
"""Live LLM provider integration tests.

These tests call real LLM APIs to verify provider integration works end-to-end.
Run with: uv run pytest tests/integration/test_llm_providers_live.py -m live

Requires environment variables:
  - ANTHROPIC_API_KEY  (for Anthropic tests)
  - OPENAI_API_KEY     (for OpenAI tests)
  - Ollama running     (for Ollama tests)

Skipped automatically when the relevant API key or service is unavailable.
"""

from __future__ import annotations

import json
import os

import pytest

from apmode.backends.llm_client import LLMConfig, LLMResponse
from apmode.backends.llm_providers import create_llm_client
from apmode.backends.prompts.system_v1 import build_system_prompt
from apmode.backends.transform_parser import parse_llm_response

# Billing/quota errors should skip, not fail
_BILLING_SKIP_PHRASES = [
    "credit balance is too low",
    "exceeded your current quota",
    "rate limit",
    "insufficient_quota",
    "billing",
    "missing authentication",
    "invalid api key",
    "unauthorized",
    "user not found",
    "authenticationerror",
]


def _is_billing_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _BILLING_SKIP_PHRASES)


def _skip_on_billing(func):  # type: ignore[no-untyped-def]
    """Decorator: skip test if API returns billing/quota error."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if _is_billing_error(e):
                pytest.skip(f"Billing/quota error: {e}")
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY"))


def _ollama_chat_model() -> str | None:
    """Return the first chat-capable Ollama model, or None."""
    try:
        import ollama

        response = ollama.list()
        models = response.get("models", [])
        # Skip embedding models and models that don't support chat
        skip_keywords = {"embed", "arctic-embed", "mxbai-embed"}
        # deepseek-r1 small models don't support chat endpoint
        skip_families = {"bert"}
        skip_prefixes = {"deepseek-r1:1"}
        for m in models:
            name: str = getattr(m, "model", None) or m.get("name", "")
            details = getattr(m, "details", None)
            family = getattr(details, "family", "").lower() if details else ""
            name_lower = name.lower()
            if any(kw in name_lower for kw in skip_keywords):
                continue
            if family in skip_families:
                continue
            if any(name_lower.startswith(p) for p in skip_prefixes):
                continue
            return name
        return None
    except Exception:
        return None


_ollama_model = _ollama_chat_model()
has_ollama = _ollama_model is not None

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A realistic PK diagnostic prompt that the LLM should respond to with
# valid Formular transforms.
_SYSTEM_PROMPT = build_system_prompt(
    lane="discovery",
    available_transforms=[
        "swap_module",
        "add_covariate_link",
        "adjust_variability",
        "set_transit_n",
        "toggle_lag",
        "replace_with_node",
    ],
)

_DIAGNOSTIC_PROMPT = """\
## Iteration 1/25

Converged (saem). OFV=-450.2, BIC=480.5, AIC=460.2

### Structural Parameters
  ka = 1.45 (RSE=12.3%)
  V = 68.2 (RSE=8.1%)
  CL = 4.85 (RSE=6.2%)

### Residual Diagnostics
  CWRES mean = 0.42, SD = 1.15
  **High CWRES bias (0.42)** — suggests systematic misfit.
  Outlier fraction = 0.035
  **High shrinkage:** {'ka': 45.2}

### Search History (recent)
  - model_001: BIC=480.5, converged=True
"""


def _validate_llm_response(resp: LLMResponse) -> None:
    """Assert that an LLMResponse has the expected structure."""
    assert isinstance(resp.raw_text, str)
    assert len(resp.raw_text) > 0, "LLM returned empty response"
    assert resp.model_id, "model_id is empty"
    assert resp.input_tokens > 0, "input_tokens should be positive"
    assert resp.output_tokens > 0, "output_tokens should be positive"
    assert resp.wall_time_seconds > 0, "wall_time should be positive"
    assert resp.request_payload_hash, "payload hash is empty"


def _validate_pk_response(raw_text: str) -> None:
    """Assert that the LLM response is valid Formular JSON."""
    result = parse_llm_response(raw_text)
    assert result.success or result.stop, (
        f"LLM response did not parse as valid Formular JSON: {result.errors}\n"
        f"Raw text: {raw_text[:500]}"
    )
    # If not stopping, should have proposed at least one transform
    if not result.stop:
        assert len(result.transforms) > 0, "LLM proposed no transforms and didn't stop"
    assert result.reasoning, "LLM provided no reasoning"


# ---------------------------------------------------------------------------
# Anthropic live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
@_skip_on_billing
async def test_anthropic_live_basic() -> None:
    """Anthropic SDK returns a valid LLMResponse."""
    config = LLMConfig(model="claude-haiku-4-5-20251001", provider="anthropic")
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_001",
        [
            {"role": "system", "content": "Respond with: hello"},
            {"role": "user", "content": "Say hello."},
        ],
    )
    _validate_llm_response(resp)
    assert resp.cost_usd >= 0


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
@_skip_on_billing
async def test_anthropic_live_pk_transforms() -> None:
    """Anthropic produces valid Formular transforms from PK diagnostics."""
    config = LLMConfig(model="claude-haiku-4-5-20251001", provider="anthropic")
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_002",
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _DIAGNOSTIC_PROMPT},
        ],
    )
    _validate_llm_response(resp)
    _validate_pk_response(resp.raw_text)


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
@_skip_on_billing
async def test_anthropic_live_multi_turn() -> None:
    """Anthropic handles multi-turn conversation with history."""
    config = LLMConfig(model="claude-haiku-4-5-20251001", provider="anthropic")
    client = create_llm_client(config)

    # Simulate a second iteration where the LLM has conversation context
    resp = await client.complete(
        "live_test_003",
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _DIAGNOSTIC_PROMPT},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "transforms": [
                            {
                                "type": "adjust_variability",
                                "param": "ka",
                                "action": "remove",
                            }
                        ],
                        "reasoning": "Remove IIV on ka due to 45% shrinkage.",
                        "stop": False,
                    }
                ),
            },
            {
                "role": "user",
                "content": (
                    "## Iteration 2/25\n\n"
                    "Converged (saem). OFV=-455.1, BIC=475.2, AIC=457.1\n\n"
                    "### Residual Diagnostics\n"
                    "  CWRES mean = 0.38, SD = 1.10\n"
                    "  **High CWRES bias (0.38)** — suggests systematic misfit.\n"
                    "  Outlier fraction = 0.030\n"
                ),
            },
        ],
    )
    _validate_llm_response(resp)
    _validate_pk_response(resp.raw_text)


# ---------------------------------------------------------------------------
# OpenAI live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key, reason="OPENAI_API_KEY not set")
@_skip_on_billing
async def test_openai_live_basic() -> None:
    """OpenAI SDK returns a valid LLMResponse."""
    config = LLMConfig(model="gpt-4o-mini", provider="openai")
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_010",
        [
            {"role": "system", "content": "Respond with: hello"},
            {"role": "user", "content": "Say hello."},
        ],
    )
    _validate_llm_response(resp)
    assert resp.model_version, "OpenAI should return system_fingerprint"


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_openai_key, reason="OPENAI_API_KEY not set")
@_skip_on_billing
async def test_openai_live_pk_transforms() -> None:
    """OpenAI produces valid Formular transforms from PK diagnostics."""
    config = LLMConfig(model="gpt-4o-mini", provider="openai")
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_011",
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _DIAGNOSTIC_PROMPT},
        ],
    )
    _validate_llm_response(resp)
    _validate_pk_response(resp.raw_text)


# ---------------------------------------------------------------------------
# OpenRouter live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_openrouter_key, reason="OPENROUTER_API_KEY not set")
@_skip_on_billing
async def test_openrouter_live_basic() -> None:
    """OpenRouter returns a valid LLMResponse via OpenAI-compatible API."""
    config = LLMConfig(
        model="anthropic/claude-haiku-4-5-20251001",
        provider="openrouter",
    )
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_020",
        [
            {"role": "system", "content": "Respond with: hello"},
            {"role": "user", "content": "Say hello."},
        ],
    )
    _validate_llm_response(resp)


# ---------------------------------------------------------------------------
# Ollama live tests
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_ollama, reason="Ollama not running or no chat models available")
async def test_ollama_live_basic() -> None:
    """Ollama local inference returns a valid LLMResponse."""
    assert _ollama_model is not None
    config = LLMConfig(model=_ollama_model, provider="ollama")
    client = create_llm_client(config)

    resp = await client.complete(
        "live_test_030",
        [
            {"role": "system", "content": "Respond with exactly: hello"},
            {"role": "user", "content": "Say hello."},
        ],
    )
    _validate_llm_response(resp)
    assert resp.cost_usd == 0.0, "Ollama should have zero cost"


# ---------------------------------------------------------------------------
# Agentic loop integration (live Anthropic)
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(not has_anthropic_key, reason="ANTHROPIC_API_KEY not set")
@_skip_on_billing
async def test_agentic_loop_live_one_iteration(tmp_path: object) -> None:
    """Run a single agentic iteration with a live LLM and mocked inner runner.

    Verifies the full loop: prompt → LLM call → parse → validate → trace.
    """
    from pathlib import Path
    from unittest.mock import AsyncMock

    from apmode.backends.agentic_runner import AgenticConfig, AgenticRunner
    from apmode.bundle.models import (
        BackendResult,
        BLQHandling,
        ColumnMapping,
        ConvergenceMetadata,
        DataManifest,
        DiagnosticBundle,
        GOFMetrics,
        IdentifiabilityFlags,
        ParameterEstimate,
    )
    from apmode.dsl.ast_models import (
        IIV,
        DSLSpec,
        FirstOrder,
        LinearElim,
        OneCmt,
        Proportional,
    )

    # Build a realistic mock result with signals the LLM should react to
    mock_result = BackendResult(
        model_id="live-test-result",
        backend="nlmixr2",
        converged=True,
        ofv=-450.0,
        aic=460.0,
        bic=480.0,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL", estimate=5.0, se=0.3, rse=6.0, category="structural"
            ),
            "V": ParameterEstimate(
                name="V", estimate=70.0, se=5.6, rse=8.0, category="structural"
            ),
            "ka": ParameterEstimate(
                name="ka", estimate=1.5, se=0.5, rse=33.0, category="structural"
            ),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0, "ka": 48.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.45, cwres_sd=1.1, outlier_fraction=0.03),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0,
                profile_likelihood_ci={"CL": True, "V": True, "ka": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "2.1.0"},
        initial_estimate_source="nca",
    )

    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=mock_result)

    # Use real Anthropic client (cheap model)
    config_llm = LLMConfig(model="claude-haiku-4-5-20251001", provider="anthropic")
    llm_client = create_llm_client(config_llm)

    trace_dir = Path(str(tmp_path)) / "agentic_trace"
    agentic_config = AgenticConfig(max_iterations=2, lane="discovery")

    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=llm_client,
        config=agentic_config,
        trace_dir=trace_dir,
    )

    spec = DSLSpec(
        model_id="live-test-base",
        absorption=FirstOrder(ka=1.5),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
    )

    manifest = DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=50,
        n_observations=450,
        n_doses=50,
    )

    result = await runner.run(
        spec=spec,
        data_manifest=manifest,
        initial_estimates={"CL": 5.0, "V": 70.0, "ka": 1.5},
        seed=42,
    )

    # Verify the result
    assert result is not None
    assert result.backend == "agentic_llm"
    assert result.converged

    # Verify trace files were written
    assert trace_dir.exists()
    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) >= 3, f"Expected at least 3 trace files, got {len(trace_files)}"

    # Verify the LLM was actually called
    assert inner_runner.run.call_count >= 1
