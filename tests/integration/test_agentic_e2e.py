# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end integration test for agentic pipeline (mocked LLM)."""

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
        model_id="e2e-base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _mock_result(bic: float = 220.0) -> BackendResult:
    return BackendResult(
        model_id="e2e-result",
        backend="nlmixr2",
        converged=True,
        ofv=-100.0,
        aic=210.0,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL",
                estimate=2.0,
                se=0.1,
                rse=5.0,
                category="structural",
            ),
            "V": ParameterEstimate(
                name="V",
                estimate=30.0,
                se=1.5,
                rse=5.0,
                category="structural",
            ),
            "ka": ParameterEstimate(
                name="ka",
                estimate=1.0,
                se=0.05,
                rse=5.0,
                category="structural",
            ),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.05, cwres_sd=1.0, outlier_fraction=0.02),
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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agentic_e2e_produces_valid_bundle(tmp_path: Path) -> None:
    """Full agentic run: base spec → 1 covariate add → stop → verify trace."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(
        side_effect=[
            _mock_result(bic=220.0),  # iteration 1: evaluate base
            _mock_result(bic=215.0),  # iteration 2: evaluate after transform
        ]
    )

    responses = [
        # Iteration 1: propose adding weight covariate on CL
        LLMResponse(
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
                    "reasoning": "Add allometric weight scaling.",
                }
            ),
            model_id="claude-sonnet-4-20250514",
            model_version="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.005,
            wall_time_seconds=2.0,
            request_payload_hash="a" * 64,
        ),
        # Iteration 2: stop — model is adequate
        LLMResponse(
            raw_text=json.dumps(
                {
                    "transforms": [],
                    "reasoning": "Model is adequate after adding WT covariate.",
                    "stop": True,
                }
            ),
            model_id="claude-sonnet-4-20250514",
            model_version="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.003,
            wall_time_seconds=1.5,
            request_payload_hash="b" * 64,
        ),
    ]

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(side_effect=responses)

    trace_dir = tmp_path / "agentic_trace"
    config = AgenticConfig(max_iterations=25, lane="discovery")
    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=mock_llm,
        config=config,
        trace_dir=trace_dir,
    )

    manifest = DataManifest(
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

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=manifest,
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    # Verify result
    assert result is not None
    assert result.backend == "agentic_llm"
    assert result.converged
    assert result.bic is not None
    assert result.bic <= 220.0

    # Verify trace files
    assert trace_dir.exists()
    input_files = list(trace_dir.glob("*_input.json"))
    output_files = list(trace_dir.glob("*_output.json"))
    meta_files = list(trace_dir.glob("*_meta.json"))

    assert len(input_files) >= 2  # 2 iterations
    assert len(output_files) >= 2
    assert len(meta_files) >= 2

    # Verify trace content is valid JSON
    for f in trace_dir.glob("*.json"):
        data = json.loads(f.read_text())
        assert isinstance(data, dict)

    # Verify LLM was called exactly 2 times
    assert mock_llm.complete.call_count == 2
