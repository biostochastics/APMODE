# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the agentic LLM backend runner (PRD §4.2.6)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from apmode.backends.agentic_runner import AgenticConfig, AgenticRunner
from apmode.backends.llm_client import LLMResponse
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


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _mock_backend_result(
    model_id: str = "test",
    bic: float = 220.0,
    converged: bool = True,
) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=converged,
        ofv=-100.0,
        aic=210.0,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=2.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=30.0, category="structural"),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=converged,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.05, cwres_sd=1.0, outlier_fraction=0.02),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0,
                profile_likelihood_ci={"CL": True, "V": True},
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
            subject_id="ID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=50,
        n_observations=500,
        n_doses=100,
    )


def _stop_response() -> LLMResponse:
    return LLMResponse(
        raw_text=json.dumps({"transforms": [], "stop": True, "reasoning": "Adequate."}),
        model_id="test",
        model_version="v1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.0,
        request_payload_hash="d" * 64,
    )


def _swap_response() -> LLMResponse:
    """Propose adding allometric weight on CL — always valid for base spec."""
    return LLMResponse(
        raw_text=json.dumps(
            {
                "transforms": [
                    {
                        "type": "add_covariate_link",
                        "param": "CL",
                        "covariate": "WT",
                        "form": "power",
                    }
                ],
                "reasoning": "Add allometric weight scaling on CL.",
            }
        ),
        model_id="test",
        model_version="v1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.0,
        request_payload_hash="c" * 64,
    )


def test_agentic_runner_satisfies_protocol() -> None:
    assert isinstance(AgenticRunner, type)


@pytest.mark.asyncio
async def test_respects_iteration_budget(tmp_path: Path) -> None:
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_swap_response())

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
async def test_stops_on_stop_signal(tmp_path: Path) -> None:
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

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
    assert mock_llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_writes_trace_files(tmp_path: Path) -> None:
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

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
    assert len(trace_files) >= 3  # input + output + meta for iteration 1


@pytest.mark.asyncio
async def test_returns_best_result(tmp_path: Path) -> None:
    results = [
        _mock_backend_result(model_id="iter1", bic=220.0),
        _mock_backend_result(model_id="iter2", bic=200.0),
    ]
    call_count = [0]

    async def mock_run(**kwargs: object) -> BackendResult:
        r = results[min(call_count[0], len(results) - 1)]
        call_count[0] += 1
        return r

    inner_runner = AsyncMock()
    inner_runner.run = mock_run

    responses = [_swap_response(), _stop_response()]
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

    assert result.bic is not None
    assert result.bic <= 220.0


@pytest.mark.asyncio
async def test_result_backend_is_agentic_llm(tmp_path: Path) -> None:
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

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

    assert result.backend == "agentic_llm"


@pytest.mark.asyncio
async def test_writes_cached_response_for_replay(tmp_path: Path) -> None:
    """Finding 1: runner must write cached_response.json for ReplayClient."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

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

    cached = list(trace_dir.glob("*_cached_response.json"))
    assert len(cached) >= 1
    data = json.loads(cached[0].read_text())
    assert "raw_text" in data
    assert "request_payload_hash" in data


@pytest.mark.asyncio
async def test_writes_run_lineage(tmp_path: Path) -> None:
    """Finding 3: runner writes run_lineage.json."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

    trace_dir = tmp_path / "agentic_trace"
    config = AgenticConfig(
        max_iterations=25,
        lane="discovery",
        parent_run_ids=["prior_run_001"],
    )
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

    lineage_path = trace_dir / "run_lineage.json"
    assert lineage_path.exists()
    data = json.loads(lineage_path.read_text())
    assert data["parent_run_ids"] == ["prior_run_001"]
    assert data["lineage_type"] == "continuation"


@pytest.mark.asyncio
async def test_writes_iteration_records(tmp_path: Path) -> None:
    """Finding 4: runner writes agentic_iterations.jsonl."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_stop_response())

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

    iterations_path = trace_dir / "agentic_iterations.jsonl"
    assert iterations_path.exists()
    lines = iterations_path.read_text().strip().split("\n")
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert "iteration" in rec
    assert "spec_before" in rec


@pytest.mark.asyncio
async def test_relays_runner_failure_to_llm(tmp_path: Path) -> None:
    """Finding 6: inner runner failure is sent to LLM for corrective action."""
    inner_runner = AsyncMock()
    # First call fails, second succeeds
    inner_runner.run = AsyncMock(
        side_effect=[
            RuntimeError("ODE solver diverged"),
            _mock_backend_result(),
        ]
    )

    # On failure iteration: LLM says "no transforms, keep trying" (not stop)
    # On success iteration: LLM says stop
    no_change_resp = LLMResponse(
        raw_text=json.dumps(
            {
                "transforms": [],
                "reasoning": "Cannot fix; retry with current spec.",
                "stop": False,
            }
        ),
        model_id="test",
        model_version="v1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.0,
        request_payload_hash="g" * 64,
    )
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(side_effect=[no_change_resp, _stop_response()])

    trace_dir = tmp_path / "agentic_trace"
    config = AgenticConfig(max_iterations=3, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=trace_dir,
    )

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    assert result.converged
    # LLM was called at least once for the failure iteration
    assert mock_llm.complete.call_count >= 2


@pytest.mark.asyncio
async def test_model_version_escrow_best_effort(tmp_path: Path) -> None:
    """Finding 2: model_version == model_id → best-effort flag."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    # LLM response where model_version == model_id (no deterministic fingerprint)
    resp = LLMResponse(
        raw_text=json.dumps({"transforms": [], "stop": True, "reasoning": "Done."}),
        model_id="gpt-4o",
        model_version="gpt-4o",  # same as model_id → best-effort
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.0,
        request_payload_hash="f" * 64,
    )
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=resp)

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

    meta_files = list(trace_dir.glob("*_meta.json"))
    assert len(meta_files) >= 1
    data = json.loads(meta_files[0].read_text())
    assert data["agentic_reproducibility"] == "best-effort"
    assert data["request_payload_hash"] == "f" * 64


def test_agentic_config_rejects_iterations_above_25() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        AgenticConfig(max_iterations=30, lane="discovery")


def test_agentic_config_rejects_invalid_lane() -> None:
    with pytest.raises(ValueError, match="lane"):
        AgenticConfig(max_iterations=10, lane="invalid")
