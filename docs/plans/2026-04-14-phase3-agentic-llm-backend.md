# Phase 3: Agentic LLM Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the agentic LLM backend (P3.A) that operates exclusively through typed Formular transforms, with full reproducibility tracing, iteration budget enforcement, and integration into the existing SearchEngine/Orchestrator pipeline.

**Architecture:** The agentic backend is a `BackendRunner`-compatible module that wraps an LLM (via `litellm` for provider flexibility) to propose DSLSpec transforms based on diagnostic feedback. Each iteration: LLM proposes transforms → transforms validated via `validate_dsl()` → compiled to backend code → dispatched to nlmixr2/jax_node → diagnostics fed back to LLM. All I/O is cached verbatim in `agentic_trace/`. Capped at 25 iterations per run.

**Tech Stack:** Python 3.12+, Pydantic v2, litellm (multi-provider LLM gateway), structlog, existing Formular compiler + validator + emitters.

---

### Task 0: Define Formular Transform Types

**Files:**
- Create: `src/apmode/dsl/transforms.py`
- Test: `tests/unit/test_dsl_transforms.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_dsl_transforms.py
"""Tests for Formular transform types (PRD §4.2.5)."""
from apmode.dsl.ast_models import (
    DSLSpec, FirstOrder, LinearElim, MichaelisMenten, OneCmt, TwoCmt,
    Proportional, IIV, CovariateLink,
)
from apmode.dsl.transforms import (
    SwapModule, AddCovariateLink, AdjustVariability,
    SetTransitN, ToggleLag, ReplaceWithNODE,
    apply_transform, validate_transform,
)

def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test-base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )

def test_swap_elimination_linear_to_mm():
    spec = _base_spec()
    t = SwapModule(position="elimination", new_module=MichaelisMenten(Vmax=50.0, Km=5.0))
    new_spec = apply_transform(spec, t)
    assert new_spec.elimination.type == "MichaelisMenten"
    assert new_spec.model_id != spec.model_id  # new ID generated

def test_swap_distribution_1cmt_to_2cmt():
    spec = _base_spec()
    t = SwapModule(position="distribution", new_module=TwoCmt(V1=30.0, V2=40.0, Q=5.0))
    new_spec = apply_transform(spec, t)
    assert new_spec.distribution.type == "TwoCmt"

def test_add_covariate_link():
    spec = _base_spec()
    t = AddCovariateLink(param="CL", covariate="WT", form="power")
    new_spec = apply_transform(spec, t)
    cov_links = [v for v in new_spec.variability if v.type == "CovariateLink"]
    assert len(cov_links) == 1
    assert cov_links[0].param == "CL"

def test_adjust_variability_add():
    spec = _base_spec()
    t = AdjustVariability(param="ka", action="add")
    new_spec = apply_transform(spec, t)
    iiv = [v for v in new_spec.variability if v.type == "IIV"][0]
    assert "ka" in iiv.params

def test_adjust_variability_remove():
    spec = _base_spec()
    t = AdjustVariability(param="V", action="remove")
    new_spec = apply_transform(spec, t)
    iiv = [v for v in new_spec.variability if v.type == "IIV"][0]
    assert "V" not in iiv.params

def test_adjust_variability_upgrade_to_block():
    spec = _base_spec()
    t = AdjustVariability(param="CL", action="upgrade_to_block")
    new_spec = apply_transform(spec, t)
    iiv = [v for v in new_spec.variability if v.type == "IIV"][0]
    assert iiv.structure == "block"

def test_validate_transform_rejects_invalid_param():
    spec = _base_spec()
    t = AddCovariateLink(param="NONEXISTENT", covariate="WT", form="power")
    errors = validate_transform(spec, t)
    assert len(errors) > 0

def test_validate_transform_rejects_duplicate_covariate():
    spec = _base_spec()
    t1 = AddCovariateLink(param="CL", covariate="WT", form="power")
    spec2 = apply_transform(spec, t1)
    t2 = AddCovariateLink(param="CL", covariate="WT", form="exponential")
    errors = validate_transform(spec2, t2)
    assert len(errors) > 0

def test_swap_module_rejects_invalid_position():
    spec = _base_spec()
    t = SwapModule(position="nonexistent", new_module=LinearElim(CL=2.0))
    errors = validate_transform(spec, t)
    assert len(errors) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_dsl_transforms.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'apmode.dsl.transforms'"

**Step 3: Write minimal implementation**

Create `src/apmode/dsl/transforms.py` with:
- 6 Pydantic transform types matching PRD §4.2.5 allowed agent transforms:
  - `SwapModule(position: Literal["absorption","distribution","elimination"], new_module)`
  - `AddCovariateLink(param, covariate, form)`
  - `AdjustVariability(param, action: Literal["add","remove","upgrade_to_block"])`
  - `SetTransitN(n: int)`
  - `ToggleLag(on: bool)`
  - `ReplaceWithNODE(position, constraint_template, dim)`
- `FormularTransform` union type
- `apply_transform(spec, transform) -> DSLSpec` — returns a new spec with a fresh model_id
- `validate_transform(spec, transform) -> list[str]` — returns validation errors

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_dsl_transforms.py -v`
Expected: PASS

**Step 5: Run full suite + type check**

Run: `uv run pytest tests/ -q && uv run mypy src/apmode/ --strict`
Expected: All pass, no type errors

**Step 6: Commit**

```bash
git add src/apmode/dsl/transforms.py tests/unit/test_dsl_transforms.py
git commit -m "feat(dsl): add Formular transform types for agentic backend (PRD §4.2.5)"
```

---

### Task 1: Agentic Trace Models + Bundle Emitter Extensions

**Files:**
- Modify: `src/apmode/bundle/emitter.py` — add agentic trace write methods
- Test: `tests/unit/test_agentic_trace_emitter.py`

The Pydantic models already exist in `bundle/models.py`: `AgenticTraceInput`, `AgenticTraceOutput`, `AgenticTraceMeta`, `RunLineage`. We need the emitter methods.

**Step 1: Write the failing test**

```python
# tests/unit/test_agentic_trace_emitter.py
"""Tests for agentic trace bundle emission."""
import json
from pathlib import Path

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    AgenticTraceInput, AgenticTraceOutput, AgenticTraceMeta, RunLineage,
)

def test_write_agentic_trace_input(tmp_path: Path):
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    inp = AgenticTraceInput(
        iteration_id="iter_001",
        run_id="run_abc",
        candidate_id="cand_001",
        prompt_hash="abc123" * 10 + "abcd",  # 64 hex chars
        prompt_template="system_v1",
        dsl_spec_json='{"model_id":"test"}',
        diagnostics_summary={"cwres_mean": 0.1},
    )
    emitter.write_agentic_trace_input(inp)
    trace_dir = emitter._run_dir / "agentic_trace"
    assert trace_dir.exists()
    f = trace_dir / "iter_001_input.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert data["iteration_id"] == "iter_001"

def test_write_agentic_trace_output(tmp_path: Path):
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    out = AgenticTraceOutput(
        iteration_id="iter_001",
        raw_output='{"transforms": []}',
        parsed_transforms=["swap_module(elimination, MichaelisMenten)"],
        validation_passed=True,
    )
    emitter.write_agentic_trace_output(out)
    f = emitter._run_dir / "agentic_trace" / "iter_001_output.json"
    assert f.exists()

def test_write_agentic_trace_meta(tmp_path: Path):
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    meta = AgenticTraceMeta(
        iteration_id="iter_001",
        model_id="claude-sonnet-4-20250514",
        model_version="claude-sonnet-4-20250514",
        prompt_hash="abc123" * 10 + "abcd",
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.005,
        temperature=0.0,
        wall_time_seconds=2.3,
    )
    emitter.write_agentic_trace_meta(meta)
    f = emitter._run_dir / "agentic_trace" / "iter_001_meta.json"
    assert f.exists()

def test_write_run_lineage(tmp_path: Path):
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    lineage = RunLineage(
        current_run_id="run_002",
        parent_run_ids=["run_001"],
        lineage_type="continuation",
    )
    emitter.write_run_lineage(lineage)
    f = emitter._run_dir / "run_lineage.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert data["parent_run_ids"] == ["run_001"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_agentic_trace_emitter.py -v`
Expected: FAIL with "AttributeError: 'BundleEmitter' object has no attribute 'write_agentic_trace_input'"

**Step 3: Add methods to BundleEmitter**

Add 4 methods to `src/apmode/bundle/emitter.py`:
- `write_agentic_trace_input(self, inp: AgenticTraceInput) -> None`
- `write_agentic_trace_output(self, out: AgenticTraceOutput) -> None`
- `write_agentic_trace_meta(self, meta: AgenticTraceMeta) -> None`
- `write_run_lineage(self, lineage: RunLineage) -> None`

Each creates `agentic_trace/` subdir as needed, writes validated Pydantic JSON.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_agentic_trace_emitter.py -v`
Expected: PASS

**Step 5: Run full suite + type check**

Run: `uv run pytest tests/ -q && uv run mypy src/apmode/ --strict`

**Step 6: Commit**

```bash
git add src/apmode/bundle/emitter.py tests/unit/test_agentic_trace_emitter.py
git commit -m "feat(bundle): add agentic trace + run lineage emitter methods"
```

---

### Task 2: LLM Client Abstraction with Reproducibility

**Files:**
- Create: `src/apmode/backends/llm_client.py`
- Test: `tests/unit/test_llm_client.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_llm_client.py
"""Tests for LLM client with reproducibility tracing."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from apmode.backends.llm_client import (
    LLMClient, LLMConfig, LLMResponse, ReplayClient,
)

@pytest.fixture
def config():
    return LLMConfig(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        temperature=0.0,
        max_tokens=4096,
    )

@pytest.mark.asyncio
async def test_llm_response_has_trace_fields(config):
    """LLMResponse must capture all fields for agentic_trace."""
    mock_response = LLMResponse(
        raw_text='{"transforms": []}',
        model_id="claude-sonnet-4-20250514",
        model_version="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.5,
        request_payload_hash="a" * 64,
    )
    assert mock_response.model_id == "claude-sonnet-4-20250514"
    assert mock_response.request_payload_hash == "a" * 64

def test_llm_config_enforces_temperature_zero():
    """PRD §4.2.6: temperature must be 0 for reproducibility."""
    with pytest.raises(ValueError, match="temperature"):
        LLMConfig(model="test", provider="anthropic", temperature=0.5)

def test_llm_config_accepts_temperature_zero():
    config = LLMConfig(model="test", provider="anthropic", temperature=0.0)
    assert config.temperature == 0.0

@pytest.mark.asyncio
async def test_replay_client_returns_cached_response(tmp_path):
    """ReplayClient returns cached outputs for reproducibility."""
    # Write a cached response
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llm_client.py -v`
Expected: FAIL — module not found

**Step 3: Implement LLM client**

Create `src/apmode/backends/llm_client.py`:
- `LLMConfig` (Pydantic): model, provider, temperature (validator: must be 0.0), max_tokens, api_base (optional)
- `LLMResponse` (Pydantic): raw_text, model_id, model_version, input_tokens, output_tokens, cost_usd, wall_time_seconds, request_payload_hash
- `LLMClient` class: `async complete(iteration_id, messages) -> LLMResponse`
  - Uses `litellm.acompletion()` internally (but tested via mock)
  - Computes SHA-256 of request payload
  - Records timing, token counts, cost
- `ReplayClient` class: reads cached responses from `agentic_trace/` for deterministic replay

**Step 4: Run test**

Run: `uv run pytest tests/unit/test_llm_client.py -v`
Expected: PASS

**Step 5: Full suite + type check**

Run: `uv run pytest tests/ -q && uv run mypy src/apmode/ --strict`

**Step 6: Commit**

```bash
git add src/apmode/backends/llm_client.py tests/unit/test_llm_client.py
git commit -m "feat(backends): add LLM client with reproducibility tracing (PRD §4.2.6)"
```

---

### Task 3: Diagnostic Summarizer for LLM Context

**Files:**
- Create: `src/apmode/backends/diagnostic_summarizer.py`
- Test: `tests/unit/test_diagnostic_summarizer.py`

The LLM receives a structured summary of diagnostics — not raw data. This module converts `BackendResult` + `EvidenceManifest` into a concise text/JSON summary suitable for the LLM prompt.

**Step 1: Write the failing test**

```python
# tests/unit/test_diagnostic_summarizer.py
"""Tests for diagnostic summarizer (agentic LLM context builder)."""
from apmode.backends.diagnostic_summarizer import summarize_diagnostics, summarize_for_llm
from apmode.bundle.models import (
    BackendResult, ConvergenceMetadata, DiagnosticBundle, GOFMetrics,
    IdentifiabilityFlags, BLQHandling, ParameterEstimate,
)

def _mock_result(cwres_mean=0.05, outlier_frac=0.02, converged=True) -> BackendResult:
    return BackendResult(
        model_id="test-model",
        backend="nlmixr2",
        converged=converged,
        ofv=-100.0,
        aic=210.0,
        bic=220.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=2.0, se=0.1, rse=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=30.0, se=1.5, rse=5.0, category="structural"),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem", converged=converged, iterations=500,
            minimization_status="successful", wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=cwres_mean, cwres_sd=1.05, outlier_fraction=outlier_frac),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0, profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "2.1.0"},
        initial_estimate_source="nca",
    )

def test_summarize_diagnostics_returns_dict():
    result = _mock_result()
    summary = summarize_diagnostics(result)
    assert "cwres_mean" in summary
    assert "parameters" in summary
    assert summary["converged"] is True

def test_summarize_for_llm_returns_string():
    result = _mock_result()
    text = summarize_for_llm(result, iteration=3, max_iterations=25)
    assert "Iteration 3/25" in text
    assert "CL" in text
    assert "CWRES" in text

def test_summarize_highlights_high_cwres():
    result = _mock_result(cwres_mean=0.8)
    text = summarize_for_llm(result, iteration=1, max_iterations=25)
    assert "bias" in text.lower() or "high" in text.lower()

def test_summarize_highlights_non_convergence():
    result = _mock_result(converged=False)
    text = summarize_for_llm(result, iteration=1, max_iterations=25)
    assert "not converge" in text.lower() or "failed" in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diagnostic_summarizer.py -v`

**Step 3: Implement diagnostic summarizer**

- `summarize_diagnostics(result) -> dict` — structured summary
- `summarize_for_llm(result, iteration, max_iterations, search_history=None) -> str` — formatted text for LLM prompt

**Step 4-6: Verify, full suite, commit**

```bash
git commit -m "feat(backends): add diagnostic summarizer for agentic LLM context"
```

---

### Task 4: LLM Response Parser (Transforms Extraction)

**Files:**
- Create: `src/apmode/backends/transform_parser.py`
- Test: `tests/unit/test_transform_parser.py`

Parses the LLM's raw text response into a list of `FormularTransform` objects. Handles JSON and structured text formats. Returns validation errors for unparseable responses.

**Step 1: Write the failing test**

```python
# tests/unit/test_transform_parser.py
"""Tests for LLM response → Formular transform parser."""
import json
import pytest
from apmode.backends.transform_parser import parse_llm_response, ParseResult
from apmode.dsl.transforms import SwapModule, AddCovariateLink, AdjustVariability

def test_parse_json_single_transform():
    raw = json.dumps({
        "transforms": [
            {"type": "swap_module", "position": "elimination",
             "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0}}
        ],
        "reasoning": "CWRES show time-dependent bias in elimination phase."
    })
    result = parse_llm_response(raw)
    assert result.success
    assert len(result.transforms) == 1
    assert isinstance(result.transforms[0], SwapModule)

def test_parse_json_compound_transforms():
    raw = json.dumps({
        "transforms": [
            {"type": "swap_module", "position": "elimination",
             "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0}},
            {"type": "add_covariate_link", "param": "CL", "covariate": "WT", "form": "power"},
        ],
        "reasoning": "MM elimination + weight effect on CL."
    })
    result = parse_llm_response(raw)
    assert result.success
    assert len(result.transforms) == 2

def test_parse_stop_signal():
    raw = json.dumps({"transforms": [], "reasoning": "Model is adequate.", "stop": True})
    result = parse_llm_response(raw)
    assert result.success
    assert result.stop is True
    assert len(result.transforms) == 0

def test_parse_invalid_json():
    result = parse_llm_response("this is not json at all")
    assert not result.success
    assert len(result.errors) > 0

def test_parse_unknown_transform_type():
    raw = json.dumps({
        "transforms": [{"type": "unknown_transform", "foo": "bar"}],
        "reasoning": "test"
    })
    result = parse_llm_response(raw)
    assert not result.success
    assert any("unknown" in e.lower() for e in result.errors)

def test_parse_extracts_reasoning():
    raw = json.dumps({
        "transforms": [
            {"type": "adjust_variability", "param": "CL", "action": "upgrade_to_block"}
        ],
        "reasoning": "High correlation between CL and V etas."
    })
    result = parse_llm_response(raw)
    assert result.reasoning == "High correlation between CL and V etas."
```

**Step 2-6: Standard TDD cycle**

```bash
git commit -m "feat(backends): add LLM response parser for Formular transforms"
```

---

### Task 5: Agentic Runner Core Loop

**Files:**
- Create: `src/apmode/backends/agentic_runner.py`
- Test: `tests/unit/test_agentic_runner.py`

This is the central piece — the `AgenticRunner` class that implements the `BackendRunner` protocol and orchestrates the propose → validate → compile → fit → evaluate loop.

**Step 1: Write the failing test**

```python
# tests/unit/test_agentic_runner.py
"""Tests for the agentic LLM backend runner (PRD §4.2.6)."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from apmode.backends.agentic_runner import AgenticRunner, AgenticConfig
from apmode.backends.llm_client import LLMConfig, LLMResponse
from apmode.backends.protocol import BackendRunner
from apmode.bundle.models import (
    BackendResult, ConvergenceMetadata, DiagnosticBundle, GOFMetrics,
    IdentifiabilityFlags, BLQHandling, ParameterEstimate, DataManifest,
    ColumnMapping,
)
from apmode.dsl.ast_models import DSLSpec, FirstOrder, OneCmt, LinearElim, IIV, Proportional

def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )

def _mock_backend_result(model_id="test", bic=220.0, converged=True) -> BackendResult:
    return BackendResult(
        model_id=model_id, backend="nlmixr2", converged=converged,
        ofv=-100.0, aic=210.0, bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=2.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=30.0, category="structural"),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem", converged=converged, iterations=500,
            minimization_status="successful", wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.05, cwres_sd=1.0, outlier_fraction=0.02),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0, profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "2.1.0"},
        initial_estimate_source="nca",
    )

def _mock_data_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT",
        ),
        n_subjects=50, n_observations=500, n_doses=100,
    )

def test_agentic_runner_is_backend_runner():
    """AgenticRunner must satisfy the BackendRunner protocol."""
    assert issubclass(AgenticRunner, BackendRunner) or isinstance(
        AgenticRunner.__new__(AgenticRunner), BackendRunner
    )

@pytest.mark.asyncio
async def test_agentic_runner_respects_iteration_budget(tmp_path: Path):
    """Must stop at max_iterations (PRD: 25)."""
    # Mock inner runner that always returns a converged result
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    # Mock LLM that always proposes a swap (never stops)
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        raw_text=json.dumps({
            "transforms": [{"type": "swap_module", "position": "elimination",
                           "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0}}],
            "reasoning": "Try MM",
        }),
        model_id="test", model_version="v1",
        input_tokens=100, output_tokens=50, cost_usd=0.001,
        wall_time_seconds=1.0, request_payload_hash="c" * 64,
    ))

    config = AgenticConfig(max_iterations=3, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=tmp_path / "agentic_trace",
    )

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    assert mock_llm.complete.call_count <= 3

@pytest.mark.asyncio
async def test_agentic_runner_stops_on_stop_signal(tmp_path: Path):
    """LLM can signal stop when model is adequate."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        raw_text=json.dumps({"transforms": [], "reasoning": "Adequate.", "stop": True}),
        model_id="test", model_version="v1",
        input_tokens=100, output_tokens=50, cost_usd=0.001,
        wall_time_seconds=1.0, request_payload_hash="d" * 64,
    ))

    config = AgenticConfig(max_iterations=25, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=tmp_path / "agentic_trace",
    )

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    assert mock_llm.complete.call_count == 1  # stopped immediately

@pytest.mark.asyncio
async def test_agentic_runner_writes_trace(tmp_path: Path):
    """All LLM I/O must be cached in agentic_trace/."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        raw_text=json.dumps({"transforms": [], "stop": True, "reasoning": "Done."}),
        model_id="test", model_version="v1",
        input_tokens=100, output_tokens=50, cost_usd=0.001,
        wall_time_seconds=1.0, request_payload_hash="e" * 64,
    ))

    trace_dir = tmp_path / "agentic_trace"
    config = AgenticConfig(max_iterations=25, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=trace_dir,
    )

    await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert trace_dir.exists()
    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) >= 2  # at least input + output for 1 iteration

@pytest.mark.asyncio
async def test_agentic_runner_returns_best_result(tmp_path: Path):
    """Runner returns the best BackendResult across iterations."""
    results = [
        _mock_backend_result(model_id="iter1", bic=220.0),
        _mock_backend_result(model_id="iter2", bic=200.0),  # better
    ]
    call_count = 0

    async def mock_run(**kwargs):
        nonlocal call_count
        r = results[min(call_count, len(results) - 1)]
        call_count += 1
        return r

    inner_runner = AsyncMock()
    inner_runner.run = mock_run

    # First call: propose a transform; second call: stop
    responses = [
        LLMResponse(
            raw_text=json.dumps({
                "transforms": [{"type": "swap_module", "position": "elimination",
                               "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0}}],
                "reasoning": "Try MM",
            }),
            model_id="test", model_version="v1",
            input_tokens=100, output_tokens=50, cost_usd=0.001,
            wall_time_seconds=1.0, request_payload_hash="f" * 64,
        ),
        LLMResponse(
            raw_text=json.dumps({"transforms": [], "stop": True, "reasoning": "Done."}),
            model_id="test", model_version="v1",
            input_tokens=100, output_tokens=50, cost_usd=0.001,
            wall_time_seconds=1.0, request_payload_hash="f" * 64,
        ),
    ]

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(side_effect=responses)

    config = AgenticConfig(max_iterations=25, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=tmp_path / "agentic_trace",
    )

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    # Should return the best BIC result
    assert result.bic <= 220.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_agentic_runner.py -v`

**Step 3: Implement AgenticRunner**

Create `src/apmode/backends/agentic_runner.py`:
- `AgenticConfig`: max_iterations (default 25), lane, system_prompt_template
- `AgenticRunner`:
  - Constructor: inner_runner (BackendRunner), llm_client, config, trace_dir
  - `async run(spec, data_manifest, initial_estimates, seed, ...) -> BackendResult`
  - Core loop:
    1. Evaluate current spec via inner_runner
    2. Build diagnostic summary
    3. Send to LLM with system prompt + history
    4. Parse transforms
    5. Validate transforms against spec + lane
    6. Apply transforms to get new spec
    7. Write trace (input, output, meta)
    8. Repeat or stop
  - Returns best BackendResult (lowest BIC among converged iterations)
  - Sets `backend="agentic_llm"` on the returned result

**Step 4-6: Verify, full suite, commit**

```bash
git commit -m "feat(backends): add AgenticRunner core loop (PRD §4.2.6)"
```

---

### Task 6: System Prompt Template

**Files:**
- Create: `src/apmode/backends/prompts/system_v1.py`
- Test: `tests/unit/test_agentic_prompts.py`

The system prompt defines the LLM's role and constraints. It is versioned and its hash is stored in the trace.

**Step 1: Write the failing test**

```python
# tests/unit/test_agentic_prompts.py
"""Tests for agentic system prompt template."""
from apmode.backends.prompts.system_v1 import build_system_prompt, SYSTEM_PROMPT_VERSION
from apmode.dsl.ast_models import DSLSpec, FirstOrder, OneCmt, LinearElim, IIV, Proportional

def test_system_prompt_contains_constraints():
    prompt = build_system_prompt(lane="discovery", available_transforms=[
        "swap_module", "add_covariate_link", "adjust_variability",
        "set_transit_n", "toggle_lag", "replace_with_node",
    ])
    assert "temperature=0" in prompt or "temperature 0" in prompt
    assert "Formular" in prompt
    assert "swap_module" in prompt
    assert "JSON" in prompt

def test_system_prompt_excludes_node_for_submission():
    prompt = build_system_prompt(lane="submission", available_transforms=[
        "swap_module", "add_covariate_link", "adjust_variability",
        "set_transit_n", "toggle_lag",
    ])
    assert "replace_with_node" not in prompt.lower()
    assert "NODE" not in prompt or "not eligible" in prompt

def test_system_prompt_has_version():
    assert SYSTEM_PROMPT_VERSION.startswith("v1")

def test_system_prompt_json_schema_example():
    prompt = build_system_prompt(lane="discovery", available_transforms=["swap_module"])
    assert '"transforms"' in prompt
    assert '"reasoning"' in prompt
```

**Step 2-6: Standard TDD cycle**

```bash
git commit -m "feat(backends): add agentic system prompt v1 with PK domain grounding"
```

---

### Task 7: Orchestrator Integration

**Files:**
- Modify: `src/apmode/orchestrator/__init__.py`
- Modify: `src/apmode/routing.py` (add `agentic_llm` to discovery lane)
- Test: `tests/unit/test_orchestrator_agentic.py`

Wire the `AgenticRunner` into the existing pipeline so the Orchestrator can dispatch to it when the lane allows.

**Step 1: Write the failing test**

```python
# tests/unit/test_orchestrator_agentic.py
"""Tests for orchestrator agentic backend integration."""
from apmode.routing import route, _LANE_BACKENDS

def test_discovery_lane_includes_agentic():
    assert "agentic_llm" in _LANE_BACKENDS["discovery"]

def test_optimization_lane_includes_agentic():
    assert "agentic_llm" in _LANE_BACKENDS["optimization"]

def test_submission_lane_excludes_agentic():
    assert "agentic_llm" not in _LANE_BACKENDS["submission"]
```

**Step 2-6: Standard TDD cycle**

Update `_LANE_BACKENDS` in `routing.py` to include `"agentic_llm"` in discovery lane. Update Orchestrator to accept an optional `agentic_runner` and wire it into the `runners` dict.

```bash
git commit -m "feat(orchestrator): integrate agentic backend into lane routing and dispatch"
```

---

### Task 8: Property Tests for Transform Safety

**Files:**
- Modify: `tests/property/test_dsl_property.py`
- Test: Add Hypothesis strategies for transform generation

**Step 1: Write the failing test**

```python
# Add to tests/property/test_dsl_property.py
from hypothesis import given, strategies as st
from apmode.dsl.transforms import (
    SwapModule, AddCovariateLink, AdjustVariability,
    apply_transform, validate_transform, FormularTransform,
)

@given(st.sampled_from(["absorption", "distribution", "elimination"]))
def test_swap_module_preserves_spec_validity(position):
    """Any swap to a valid module should produce a valid spec."""
    # ... generate valid base spec, apply swap, validate result
    pass  # Implement with real strategies

def test_no_transform_produces_submission_inadmissible_node():
    """ReplaceWithNODE must fail validation in submission lane."""
    # ... ensure ReplaceWithNODE always fails for submission
    pass
```

**Step 2-6: Standard TDD cycle**

```bash
git commit -m "test(property): add Hypothesis property tests for Formular transforms"
```

---

### Task 9: Add litellm Dependency + Configuration

**Files:**
- Modify: `pyproject.toml` — add litellm to optional deps group `[agentic]`

**Step 1: Add dependency**

Add `litellm>=1.40` to an `[agentic]` extras group in pyproject.toml.

**Step 2: Verify install**

Run: `uv sync --all-extras`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add litellm dependency for agentic LLM backend"
```

---

### Task 10: Integration Test — End-to-End Agentic Run (Mocked LLM)

**Files:**
- Create: `tests/integration/test_agentic_e2e.py`

A full integration test that runs the agentic pipeline with a mocked LLM (no real API calls), verifying the entire propose→validate→compile→fit→evaluate→trace cycle.

**Step 1: Write the failing test**

```python
# tests/integration/test_agentic_e2e.py
"""End-to-end integration test for agentic pipeline (mocked LLM)."""
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from apmode.backends.agentic_runner import AgenticRunner, AgenticConfig
from apmode.backends.llm_client import LLMResponse
# ... (build a full pipeline from spec → agentic runner → result → trace check)

@pytest.mark.asyncio
async def test_agentic_e2e_produces_valid_bundle(tmp_path: Path):
    """Full agentic run produces a valid reproducibility bundle."""
    # 1. Start with a base spec
    # 2. Mock LLM proposes 2 transforms then stops
    # 3. Mock inner runner returns results
    # 4. Verify: trace files exist, best result returned, all iterations logged
    pass  # Full implementation during execution
```

**Step 2-6: Standard TDD cycle**

```bash
git commit -m "test(integration): add end-to-end agentic pipeline test with mocked LLM"
```

---

### Task 11: Update ARCHITECTURE.md and CHANGELOG.md

**Files:**
- Modify: `ARCHITECTURE.md` — add §4.2.6 Agentic LLM Backend section
- Modify: `CHANGELOG.md` — add Phase 3 entries

**Step 1: Update docs**

Document the new agentic backend architecture, transform types, trace format, and integration points.

**Step 2: Commit**

```bash
git add ARCHITECTURE.md CHANGELOG.md
git commit -m "docs: document agentic LLM backend architecture (Phase 3)"
```

---

## Dependency Graph

```
Task 0 (Transforms)  ──┐
                        ├──→ Task 4 (Parser)  ──┐
Task 1 (Trace Emitter) ─┤                       ├──→ Task 5 (AgenticRunner) ──→ Task 7 (Orchestrator) ──→ Task 10 (E2E)
Task 2 (LLM Client) ────┤                       │
Task 3 (Diagnostics) ───┘                       │
                                                 │
Task 6 (Prompt) ─────────────────────────────────┘
Task 9 (Dependency) ─── (independent, can be early)
Task 8 (Property Tests) ─── (after Task 0)
Task 11 (Docs) ─── (after Task 10)
```

Tasks 0-3 and 9 can run in parallel. Task 4 needs Task 0. Task 5 needs Tasks 0-4+6. Task 7 needs Task 5. Task 10 needs Task 7. Task 11 is last.
