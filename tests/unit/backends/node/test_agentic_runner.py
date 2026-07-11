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
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    LinearElim,
    OneCmt,
    Proportional,
    SumIG,
    Transit,
)
from tests._helpers.builders import base_spec as _base_spec
from tests._helpers.builders import mock_backend_result as _mock_backend_result
from tests._helpers.builders import mock_data_manifest as _mock_data_manifest
from tests._helpers.builders import stop_response as _stop_response


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
                        "theta": 0.75,
                        "ref": 70.0,
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
async def test_fixed_parameter_bypasses_llm_loop_and_delegates_once(tmp_path: Path) -> None:
    """LORO-CV contract: fixed_parameter=True must bypass the LLM loop and
    delegate to the inner runner exactly once, forwarding fixed_parameter
    and test_data_path verbatim (CLAUDE.md: no-refit contract)."""
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
    test_csv = tmp_path / "held_out.csv"
    test_csv.write_text("ID,TIME,DV\n99,0.0,0\n")

    result = await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
        fixed_parameter=True,
        test_data_path=test_csv,
    )

    assert result is not None
    mock_llm.complete.assert_not_called()
    assert inner_runner.run.call_count == 1
    _, kwargs = inner_runner.run.call_args
    assert kwargs["fixed_parameter"] is True
    assert kwargs["test_data_path"] == test_csv


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


def _base_spec_with_initial() -> DSLSpec:
    """Same as _base_spec but with initial values so AddCovariateLink

    validates cleanly (the plain _base_spec() has no ``initial:`` block, so
    transforms there always fail post-transform DSL validation — fine for
    tests that only check best-result tracking, but this test needs the
    transform to actually apply so lineage gets recorded).
    """
    return _base_spec().model_copy(update={"initial": {"CL": 2.0, "V": 30.0, "ka": 1.0}})


def _swap_response_with_rationale() -> LLMResponse:
    """Same as _swap_response but with rationale/expected_diagnostic_effect (P2.2)."""
    return LLMResponse(
        raw_text=json.dumps(
            {
                "transforms": [
                    {
                        "type": "add_covariate_link",
                        "param": "CL",
                        "covariate": "WT",
                        "form": "power",
                        "theta": 0.75,
                        "ref": 70.0,
                        "rationale": "Wide body-weight range supports allometric scaling.",
                        "expected_diagnostic_effect": ["reduces CL eta shrinkage"],
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
        request_payload_hash="e" * 64,
    )


@pytest.mark.asyncio
async def test_agentic_lineage_records_rationale_and_applied_at(tmp_path: Path) -> None:
    """P2.2: agentic_lineage.json carries rationale/effect/applied_at pulled

    from the FormularTransform object the LLM supplied, not re-invented.
    """
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [_swap_response_with_rationale(), _stop_response()]
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

    await runner.run(
        spec=_base_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    lineage_path = trace_dir / "agentic_lineage.json"
    assert lineage_path.exists()
    data = json.loads(lineage_path.read_text())
    entries = data["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["rationale"] == "Wide body-weight range supports allometric scaling."
    assert entry["expected_diagnostic_effect"] == ["reduces CL eta shrinkage"]
    assert entry["applied_at"] is not None
    # ISO-8601 round-trips via fromisoformat
    from datetime import datetime

    datetime.fromisoformat(entry["applied_at"])


@pytest.mark.asyncio
async def test_agentic_lineage_defaults_when_no_rationale_supplied(tmp_path: Path) -> None:
    """Backward compat: omitting rationale/effect still produces a valid entry."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [_swap_response(), _stop_response()]
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

    await runner.run(
        spec=_base_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    entry = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"][0]
    assert entry["rationale"] is None
    assert entry["expected_diagnostic_effect"] == []
    assert entry["applied_at"] is not None


# ---------------------------------------------------------------------------
# End-to-end coverage for the four v0.7 DSL transforms exposed to the agent
# same-day as the compiler support landed (commit 124978c). Interface
# parity (available_transforms / _TRANSFORM_DESCRIPTIONS / parser registry)
# was already covered by test_agentic_prompts.py and
# test_transform_parser.py; these tests drive each transform through the
# actual AgenticRunner.run() loop to a compiled/fit candidate, closing the
# gap the xen consensus review flagged: the loop, not just the interface,
# must accept these transforms.
# ---------------------------------------------------------------------------


def _transit_spec_with_initial() -> DSLSpec:
    """Transit absorption — precondition for convert_transit_to_erlang."""
    return DSLSpec(
        model_id="transit-base",
        absorption=Transit(n=3),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        initial={"CL": 2.0, "V": 30.0, "ktr": 2.0, "ka": 1.0},
    )


def _sumig_spec_with_initial() -> DSLSpec:
    """SumIG absorption — precondition for set_sumig_components.

    SumIG additionally requires disposition (CL/V/Q) to be fixed
    externally (ADR-0003 D5 — absorption and disposition are not jointly
    identifiable from oral-only data), certified here via
    ``source="fixed_external"`` priors rather than a manifest-level
    ``disposition_fixed`` flag.
    """
    from apmode.dsl.priors import LogNormalPrior, PriorSpec

    return DSLSpec(
        model_id="sumig-base",
        absorption=SumIG(k=2),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        priors=[
            PriorSpec(
                target="CL",
                family=LogNormalPrior(mu=0.7, sigma=0.1),
                source="fixed_external",
            ),
            PriorSpec(
                target="V",
                family=LogNormalPrior(mu=3.4, sigma=0.1),
                source="fixed_external",
            ),
        ],
        initial={
            "CL": 2.0,
            "V": 30.0,
            "MT_1": 0.5,
            "MT_2": 2.0,
            "RD2_1": 0.3,
            "RD2_2": 0.3,
            "weight_1": 0.6,
        },
    )


def _transform_response(transforms: list[dict[str, object]], *, reasoning: str) -> LLMResponse:
    return LLMResponse(
        raw_text=json.dumps({"transforms": transforms, "reasoning": reasoning}),
        model_id="test",
        model_version="v1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        wall_time_seconds=1.0,
        request_payload_hash="1" * 64,
    )


@pytest.mark.asyncio
async def test_convert_transit_to_erlang_applies_end_to_end(tmp_path: Path) -> None:
    """convert_transit_to_erlang flows through the agent loop to a fit candidate."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [
        _transform_response(
            [{"type": "convert_transit_to_erlang", "n": 3}],
            reasoning="Erlang gives a cleaner ODE form than transit(n, mtt).",
        ),
        _stop_response(),
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

    result = await runner.run(
        spec=_transit_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ktr": 2.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    entries = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"]
    assert len(entries) == 1
    assert entries[0]["transform"].startswith("type='convert_transit_to_erlang'") or (
        "convert_transit_to_erlang" in entries[0]["transform"]
    )
    # The fit was actually driven on the post-transform spec, not the base one.
    fitted_spec = inner_runner.run.call_args_list[-1].kwargs["spec"]
    assert fitted_spec.absorption.type == "Erlang"


@pytest.mark.asyncio
async def test_add_parallel_route_applies_end_to_end(tmp_path: Path) -> None:
    """add_parallel_route flows through the agent loop to a fit candidate."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [
        _transform_response(
            [{"type": "add_parallel_route", "ka2": 0.3, "frac": 0.4}],
            reasoning="Data show a fast/slow absorption split.",
        ),
        _stop_response(),
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

    result = await runner.run(
        spec=_base_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    entries = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"]
    assert len(entries) == 1
    fitted_spec = inner_runner.run.call_args_list[-1].kwargs["spec"]
    assert fitted_spec.absorption.type == "ParallelFirstOrder"
    assert fitted_spec.initial["ka2"] == pytest.approx(0.3)
    assert fitted_spec.initial["frac"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_set_sumig_components_applies_end_to_end(tmp_path: Path) -> None:
    """set_sumig_components flows through the agent loop to a fit candidate."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [
        _transform_response(
            [
                {
                    "type": "set_sumig_components",
                    "MT_1": 0.4,
                    "MT_2": 2.5,
                    "RD2_1": 0.25,
                    "RD2_2": 0.35,
                    "weight_1": 0.55,
                }
            ],
            reasoning="Refine SumIG component timing from VPC diagnostics.",
        ),
        _stop_response(),
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

    result = await runner.run(
        spec=_sumig_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0},
        seed=42,
    )

    assert result is not None
    entries = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"]
    assert len(entries) == 1
    fitted_spec = inner_runner.run.call_args_list[-1].kwargs["spec"]
    assert fitted_spec.initial["MT_1"] == pytest.approx(0.4)
    assert fitted_spec.initial["MT_2"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_set_prior_applies_end_to_end(tmp_path: Path) -> None:
    """set_prior flows through the agent loop to a fit candidate."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [
        _transform_response(
            [
                {
                    "type": "set_prior",
                    "target": "CL",
                    "family": {"type": "LogNormal", "mu": 0.7, "sigma": 0.3},
                    "source": "weakly_informative",
                }
            ],
            reasoning="Weakly-informative LogNormal prior on CL for the Bayesian path.",
        ),
        _stop_response(),
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

    result = await runner.run(
        spec=_base_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    entries = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"]
    assert len(entries) == 1
    fitted_spec = inner_runner.run.call_args_list[-1].kwargs["spec"]
    assert any(p.target == "CL" for p in fitted_spec.priors)


@pytest.mark.asyncio
async def test_compound_multi_transform_proposal_applies_atomically(tmp_path: Path) -> None:
    """A single iteration proposing two independent transforms applies both.

    PRD §4.2.6 frames compound proposals (multiple transforms per
    iteration) as the agent's value-add over a one-transform-at-a-time
    search. ``set_prior`` and ``convert_transit_to_erlang`` are mutually
    independent (neither's precondition depends on the other), so both
    should land in the same iteration's lineage and the fitted spec
    should reflect both changes together.
    """
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result())

    responses = [
        _transform_response(
            [
                {
                    "type": "set_prior",
                    "target": "CL",
                    "family": {"type": "LogNormal", "mu": 0.7, "sigma": 0.3},
                },
                {"type": "convert_transit_to_erlang", "n": 3},
            ],
            reasoning="Combine a CL prior with the transit->erlang simplification.",
        ),
        _stop_response(),
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

    result = await runner.run(
        spec=_transit_spec_with_initial(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0, "ktr": 2.0, "ka": 1.0},
        seed=42,
    )

    assert result is not None
    entries = json.loads((trace_dir / "agentic_lineage.json").read_text())["entries"]
    # Both transforms from the one compound proposal were staged and
    # committed together (all-or-nothing gated by the post-transform
    # validate_dsl check — see agentic_runner.py's staged_lineage comment).
    assert len(entries) == 2
    transform_types = [e["transform"] for e in entries]
    assert any("set_prior" in t for t in transform_types)
    assert any("convert_transit_to_erlang" in t for t in transform_types)

    fitted_spec = inner_runner.run.call_args_list[-1].kwargs["spec"]
    assert fitted_spec.absorption.type == "Erlang"
    assert any(p.target == "CL" for p in fitted_spec.priors)


@pytest.mark.asyncio
async def test_iteration_record_captures_shrinkage_and_auc_cmax(tmp_path: Path) -> None:
    """Trajectory-level gaming detection needs per-iteration shrinkage and
    auc_cmax_be_score, not just bic — these must be persisted to
    agentic_iterations.jsonl."""
    inner_runner = AsyncMock()
    result = _mock_backend_result(bic=200.0)
    result.diagnostics.auc_cmax_be_score = 0.75
    inner_runner.run = AsyncMock(return_value=result)
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_stop_response())

    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=llm,
        config=AgenticConfig(max_iterations=1, run_id="test-run"),
        trace_dir=tmp_path,
    )
    await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0},
        seed=1,
    )

    lines = (tmp_path / "agentic_iterations.jsonl").read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["eta_shrinkage_max"] == max(result.eta_shrinkage.values())
    assert entry["auc_cmax_be_score"] == 0.75


@pytest.mark.asyncio
async def test_trajectory_compliance_json_written_on_every_run(tmp_path: Path) -> None:
    """AgenticRunner.run() must write trajectory_compliance.json into the
    trace dir on every exit path, since _finalise_trace runs in the
    ``finally`` block."""
    inner_runner = AsyncMock()
    inner_runner.run = AsyncMock(return_value=_mock_backend_result(bic=200.0))
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=_stop_response())

    runner = AgenticRunner(
        inner_runner=inner_runner,
        llm_client=llm,
        config=AgenticConfig(max_iterations=1, run_id="test-run-compliance"),
        trace_dir=tmp_path,
    )
    await runner.run(
        spec=_base_spec(),
        data_manifest=_mock_data_manifest(),
        initial_estimates={"CL": 2.0, "V": 30.0},
        seed=1,
    )

    path = tmp_path / "trajectory_compliance.json"
    assert path.exists()
    report = json.loads(path.read_text())
    assert "reward_hacking_suspected" in report
