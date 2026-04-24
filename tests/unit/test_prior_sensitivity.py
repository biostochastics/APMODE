# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Gate 2 prior-sensitivity diagnostic (plan Task 21).

Same three-layer pattern as ``test_prior_data_conflict.py``: helper,
schema invariants, Gate 2 wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from apmode.bayes.prior_sensitivity import compute_prior_sensitivity
from apmode.bundle.models import (
    BackendResult,
    ConvergenceMetadata,
    DiagnosticBundle,
    ParameterEstimate,
    PriorSensitivity,
    PriorSensitivityEntry,
    SamplerConfig,
    ScoringContract,
)
from apmode.governance.gates import evaluate_gate2
from apmode.governance.policy import GatePolicy

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _bayesian_result(*, backend: str = "bayesian_stan") -> BackendResult:
    scoring_contract = ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="hmc_nuts" if backend == "bayesian_stan" else "nlmixr2_focei",
        blq_method="none",
        observation_model="additive",
        float_precision="float64",
    )
    return BackendResult(
        model_id="cand001",
        backend=cast("type[str]", backend),  # type: ignore[arg-type]
        converged=True,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=1.0, category="structural")
        },
        eta_shrinkage={},
        convergence_metadata=ConvergenceMetadata(
            method="nuts",
            converged=True,
            iterations=2000,
            minimization_status="successful",
            wall_time_seconds=1.0,
        ),
        diagnostics=DiagnosticBundle(
            gof={"cwres_mean": 0.0, "cwres_sd": 1.0, "outlier_fraction": 0.0},
            identifiability={"profile_likelihood_ci": {}, "ill_conditioned": False},
            blq={"method": "none", "n_blq": 0, "blq_fraction": 0.0},
            scoring_contract=scoring_contract,
        ),
        wall_time_seconds=1.0,
        backend_versions={},
        initial_estimate_source="fallback",
        sampler_config=SamplerConfig() if backend == "bayesian_stan" else None,
    )


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((_POLICY_DIR / f"{lane}.json").read_text()))


def _check(out: object) -> object | None:
    for c in out.checks:  # type: ignore[attr-defined]
        if c.check_id == "prior_sensitivity":
            return c
    return None


# --- Helper-level tests ---------------------------------------------------


def test_compute_prior_sensitivity_passes_when_all_deltas_below_threshold() -> None:
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0, "V": 30.0},
        baseline_sds={"CL": 0.1, "V": 3.0},
        alternative_means={
            "sigma_x0.5": {"CL": 1.005, "V": 30.05},
            "sigma_x2.0": {"CL": 0.995, "V": 29.95},
        },
    )
    assert artifact.status == "computed"
    assert artifact.max_delta is not None
    assert artifact.max_delta < 0.20
    assert artifact.flagged_parameters == []
    assert artifact.n_parameters == 2
    assert artifact.n_alternatives_per_parameter == 2


def test_compute_prior_sensitivity_flags_parameter_with_large_delta() -> None:
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0, "V": 30.0},
        baseline_sds={"CL": 0.1, "V": 3.0},
        alternative_means={
            # CL drifts 0.05 = 0.5 baseline-SDs under the wider prior; flagged.
            "sigma_x0.5": {"CL": 1.05, "V": 30.05},
            "sigma_x2.0": {"CL": 0.95, "V": 29.95},
        },
    )
    assert artifact.status == "computed"
    assert artifact.max_delta == pytest.approx(0.5, abs=1e-9)
    assert artifact.flagged_parameters == ["CL"]


def test_compute_prior_sensitivity_requires_two_alternatives() -> None:
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0},
        baseline_sds={"CL": 0.1},
        alternative_means={"sigma_x0.5": {"CL": 1.05}},
    )
    assert artifact.status == "not_computed"
    assert artifact.reason is not None
    assert ">=2" in artifact.reason


def test_compute_prior_sensitivity_requires_overlap() -> None:
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0},
        baseline_sds={"CL": 0.1},
        alternative_means={
            "sigma_x0.5": {"V": 30.5},
            "sigma_x2.0": {"V": 29.0},
        },
    )
    assert artifact.status == "not_computed"
    assert artifact.reason is not None
    assert "every alternative" in artifact.reason


def test_compute_prior_sensitivity_skips_zero_sd_parameters() -> None:
    # CL has zero baseline SD (cannot normalise) so it's silently
    # dropped; V remains and is scoreable.
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0, "V": 30.0},
        baseline_sds={"CL": 0.0, "V": 3.0},
        alternative_means={
            "sigma_x0.5": {"CL": 1.05, "V": 30.05},
            "sigma_x2.0": {"CL": 0.95, "V": 29.95},
        },
    )
    assert artifact.status == "computed"
    assert artifact.n_parameters == 1
    assert {e.parameter for e in artifact.entries} == {"V"}


def test_compute_prior_sensitivity_returns_not_computed_on_all_zero_sds() -> None:
    artifact = compute_prior_sensitivity(
        candidate_id="cand001",
        threshold=0.20,
        baseline_means={"CL": 1.0},
        baseline_sds={"CL": 0.0},
        alternative_means={
            "sigma_x0.5": {"CL": 1.05},
            "sigma_x2.0": {"CL": 0.95},
        },
    )
    assert artifact.status == "not_computed"
    assert artifact.reason is not None
    assert "positive baseline posterior SD" in artifact.reason


# --- Schema invariants ----------------------------------------------------


def test_prior_sensitivity_status_computed_requires_max_delta() -> None:
    with pytest.raises(ValueError, match="max_delta"):
        PriorSensitivity(candidate_id="cand001", status="computed", threshold=0.20)


def test_prior_sensitivity_status_not_computed_requires_reason() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        PriorSensitivity(candidate_id="cand001", status="not_computed", threshold=0.20)


def test_prior_sensitivity_entry_delta_must_match_arithmetic() -> None:
    with pytest.raises(ValueError, match=r"|Δmean\|/sd_baseline"):
        PriorSensitivityEntry(
            parameter="CL",
            alternative_id="sigma_x0.5",
            posterior_mean_baseline=1.0,
            posterior_mean_alternative=2.0,
            posterior_sd_baseline=0.1,
            delta_normalized=0.5,  # wrong; should be 10
        )


def test_prior_sensitivity_max_delta_must_match_max_entry() -> None:
    entry = PriorSensitivityEntry(
        parameter="CL",
        alternative_id="sigma_x0.5",
        posterior_mean_baseline=1.0,
        posterior_mean_alternative=1.05,
        posterior_sd_baseline=0.1,
        delta_normalized=0.5,
    )
    with pytest.raises(ValueError, match="max_delta"):
        PriorSensitivity(
            candidate_id="cand001",
            status="computed",
            threshold=0.20,
            max_delta=99.0,  # wrong; should be 0.5
            n_parameters=1,
            n_alternatives_per_parameter=1,
            entries=[entry],
        )


# --- Gate 2 wiring --------------------------------------------------------


def _passing_artifact() -> PriorSensitivity:
    entry = PriorSensitivityEntry(
        parameter="CL",
        alternative_id="sigma_x0.5",
        posterior_mean_baseline=1.0,
        posterior_mean_alternative=1.005,
        posterior_sd_baseline=0.1,
        delta_normalized=0.05,
    )
    return PriorSensitivity(
        candidate_id="cand001",
        status="computed",
        threshold=0.20,
        max_delta=0.05,
        n_parameters=1,
        n_alternatives_per_parameter=2,
        entries=[entry],
        flagged_parameters=[],
    )


def _failing_artifact() -> PriorSensitivity:
    entry = PriorSensitivityEntry(
        parameter="CL",
        alternative_id="sigma_x0.5",
        posterior_mean_baseline=1.0,
        posterior_mean_alternative=1.05,
        posterior_sd_baseline=0.1,
        delta_normalized=0.5,
    )
    return PriorSensitivity(
        candidate_id="cand001",
        status="computed",
        threshold=0.20,
        max_delta=0.5,
        n_parameters=1,
        n_alternatives_per_parameter=2,
        entries=[entry],
        flagged_parameters=["CL"],
    )


def test_discovery_lane_skips_check_regardless_of_artifact() -> None:
    res = _bayesian_result()
    policy = _load_policy("discovery")
    out = evaluate_gate2(res, policy, lane="discovery", prior_sensitivity=_failing_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert check.observed == "not_required"  # type: ignore[attr-defined]


def test_submission_lane_passes_with_below_threshold_artifact() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_sensitivity=_passing_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]


def test_submission_lane_fails_above_threshold() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_sensitivity=_failing_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "max_delta=0.500" in check.observed  # type: ignore[attr-defined]
    assert "flagged=[CL]" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_fails_when_artifact_missing() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_sensitivity=None)
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "artifact_missing" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_fails_on_not_computed_status() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    artifact = PriorSensitivity(
        candidate_id="cand001",
        status="not_computed",
        threshold=0.20,
        reason="cmdstanpy unavailable",
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_sensitivity=artifact)
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "cmdstanpy unavailable" in check.observed  # type: ignore[attr-defined]


def test_non_bayesian_backend_passes_even_when_required() -> None:
    res = _bayesian_result(backend="nlmixr2")
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_sensitivity=None)
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert "not_bayesian_backend" in check.observed  # type: ignore[attr-defined]
