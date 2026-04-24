# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :func:`apmode.governance.gates.evaluate_gate1_bayesian` (plan Task 17).

The evaluator maps per-parameter-class R-hat / ESS diagnostics from
``PosteriorDiagnostics`` against the lane policy's
``Gate1BayesianConfig`` thresholds and applies the severity tier to each
axis (``rhat``, ``ess``, ``divergences``, ``pareto_k``). A ``warn``-
severity violation still appears in the output as a non-passing check's
``evidence_ref`` but leaves the overall ``GateResult.passed`` flag True;
``fail``-severity violations drop the gate.

Tests cover:

* Every threshold check fires when the matching per-class field is
  populated (R-hat bulk ESS tail ESS on each of the four classes).
* Non-Bayesian backends pass trivially (``not_applicable``).
* A missing ``posterior_diagnostics`` field fails hard — the harness
  must always populate it.
* Scalar knobs (divergences, Pareto-k) honour their severity tier.
* Severity ``warn`` keeps the gate passing even under violation.
* The ``classify_param_class`` helper routes DSL-emitted parameter
  names to the documented classes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apmode.bundle.models import (
    BackendResult,
    ConvergenceMetadata,
    DiagnosticBundle,
    ParameterEstimate,
    PosteriorDiagnostics,
    SamplerConfig,
    ScoringContract,
)
from apmode.governance.gates import classify_param_class, evaluate_gate1_bayesian
from apmode.governance.policy import (
    Gate1BayesianConfig,
    Gate1BayesianESS,
    Gate1BayesianRhat,
    GatePolicy,
)

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _make_result(
    diag: PosteriorDiagnostics | None,
    *,
    backend: str = "bayesian_stan",
    model_id: str = "cand001",
) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend=backend,  # type: ignore[arg-type]
        converged=True,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL",
                estimate=1.0,
                se=0.1,
                rse=10.0,
                ci95_lower=0.8,
                ci95_upper=1.2,
                category="structural",
            )
        },
        eta_shrinkage={},
        convergence_metadata=ConvergenceMetadata(
            method="nuts",
            converged=True,
            iterations=2000,
            minimization_status="successful",
            wall_time_seconds=10.0,
        ),
        diagnostics=DiagnosticBundle(
            gof={"cwres_mean": 0.0, "cwres_sd": 1.0, "outlier_fraction": 0.0},
            identifiability={"profile_likelihood_ci": {}, "ill_conditioned": False},
            blq={"method": "none", "n_blq": 0, "blq_fraction": 0.0},
            scoring_contract=(
                ScoringContract(
                    nlpd_kind="marginal",
                    re_treatment="integrated",
                    nlpd_integrator="hmc_nuts",
                    blq_method="none",
                    observation_model="additive",
                    float_precision="float64",
                )
                if backend == "bayesian_stan"
                else ScoringContract(
                    nlpd_kind="marginal",
                    re_treatment="integrated",
                    nlpd_integrator="nlmixr2_focei",
                    blq_method="none",
                    observation_model="additive",
                    float_precision="float64",
                )
            ),
        ),
        wall_time_seconds=10.0,
        backend_versions={},
        initial_estimate_source="fallback",
        posterior_diagnostics=diag,
        sampler_config=SamplerConfig() if backend == "bayesian_stan" else None,
    )


def _clean_diag(**overrides: object) -> PosteriorDiagnostics:
    """PosteriorDiagnostics that sails past every Gate 1 Bayesian check."""
    base: dict[str, object] = {
        "rhat_max": 1.00,
        "ess_bulk_min": 2000.0,
        "ess_tail_min": 2000.0,
        "n_divergent": 0,
        "n_max_treedepth": 0,
        "ebfmi_min": 0.5,
        "pareto_k_max": 0.3,
        "rhat_max_by_class": {
            "fixed_effects": 1.00,
            "iiv": 1.00,
            "residual": 1.00,
            "correlations": 1.00,
        },
        "ess_bulk_min_by_class": {
            "fixed_effects": 2000.0,
            "iiv": 2000.0,
            "residual": 2000.0,
            "correlations": 500.0,
        },
        "ess_tail_min_by_class": {
            "fixed_effects": 2000.0,
            "iiv": 2000.0,
            "residual": 2000.0,
            "correlations": 500.0,
        },
    }
    base.update(overrides)
    return PosteriorDiagnostics(**base)  # type: ignore[arg-type]


def _policy(
    *,
    gate1_bayesian: Gate1BayesianConfig | None = None,
    lane: str = "discovery",
) -> GatePolicy:
    """Load the on-disk lane policy and optionally swap in a custom gate1_bayesian.

    Building a ``GatePolicy`` ad-hoc drifts every time someone adds a new
    required ``Gate1Config`` field (Task 1 added ``gate2_5``, a future
    task may add another). Loading from ``policies/<lane>.json`` keeps
    the test fixture in step with the schema without spurious breakage.
    """
    raw = json.loads((_POLICY_DIR / f"{lane}.json").read_text())
    policy = GatePolicy.model_validate(raw)
    if gate1_bayesian is not None:
        policy = policy.model_copy(update={"gate1_bayesian": gate1_bayesian})
    return policy


# --- classify_param_class ------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CL", "fixed_effects"),
        ("V", "fixed_effects"),
        ("ka", "fixed_effects"),
        ("beta_CL_WT", "fixed_effects"),
        ("omega_CL", "iiv"),
        ("omega_iov_V", "iiv"),
        ("sigma_prop", "residual"),
        ("sigma_add", "residual"),
        ("residual_sd", "residual"),
        ("L_corr_CL_V", "correlations"),
        ("L_Omega", "correlations"),
        ("corr_iiv", "correlations"),
    ],
)
def test_classify_param_class(name: str, expected: str) -> None:
    assert classify_param_class(name) == expected


# --- Pass / fail paths ---------------------------------------------------


def test_clean_diagnostics_pass() -> None:
    res = _make_result(_clean_diag())
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is True
    assert "passed" in out.summary_reason.lower()


def test_rhat_over_threshold_fails_with_class_specific_message() -> None:
    """Submission-lane defaults: R-hat 1.03 on fixed effects exceeds 1.01."""
    diag = _clean_diag(
        rhat_max_by_class={
            "fixed_effects": 1.03,
            "iiv": 1.00,
            "residual": 1.00,
            "correlations": 1.00,
        },
    )
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy(lane="submission"))
    assert out.passed is False
    assert "1.03" in out.summary_reason
    assert "fixed_effects" in out.summary_reason


def test_ess_under_threshold_fails() -> None:
    diag = _clean_diag(
        ess_bulk_min_by_class={
            "fixed_effects": 200.0,  # below 400 default
            "iiv": 500.0,
            "residual": 500.0,
            "correlations": 200.0,
        },
    )
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is False
    assert "ESS-bulk" in out.summary_reason
    assert "200" in out.summary_reason
    assert "fixed_effects" in out.summary_reason


def test_divergences_over_tolerance_fails_by_default() -> None:
    diag = _clean_diag(n_divergent=5)
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is False
    assert "divergent" in out.summary_reason.lower()


def test_pareto_k_warn_severity_keeps_gate_passing() -> None:
    """Discovery-lane default: Pareto-k is warn, not fail."""
    diag = _clean_diag(pareto_k_max=0.85)  # > 0.7 default threshold
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is True
    assert "warn" in out.summary_reason.lower() or "pareto" in out.summary_reason.lower()
    # The check itself surfaces the violation via evidence_ref
    pareto_check = next(c for c in out.checks if c.check_id == "bayesian_pareto_k")
    assert pareto_check.evidence_ref is not None
    assert "warn" in pareto_check.evidence_ref


def test_pareto_k_fail_severity_drops_gate() -> None:
    cfg = Gate1BayesianConfig(
        severity={
            "rhat": "fail",
            "ess": "fail",
            "divergences": "fail",
            "pareto_k": "fail",  # Optimization-lane override
        },
    )
    diag = _clean_diag(pareto_k_max=0.85)
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy(gate1_bayesian=cfg))
    assert out.passed is False
    assert "Pareto-k" in out.summary_reason


# --- Threshold overrides -------------------------------------------------


def test_relaxed_fixed_effects_threshold_accepts_wider_rhat() -> None:
    """Discovery lane accepts r-hat up to 1.05 on fixed effects."""
    cfg = Gate1BayesianConfig(
        rhat_max=Gate1BayesianRhat(fixed_effects=1.05),
    )
    diag = _clean_diag(
        rhat_max_by_class={
            "fixed_effects": 1.04,  # <= 1.05 — passes
            "iiv": 1.00,
            "residual": 1.00,
            "correlations": 1.00,
        },
    )
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy(gate1_bayesian=cfg))
    assert out.passed is True


def test_tight_correlations_ess_threshold_still_catches_violation() -> None:
    cfg = Gate1BayesianConfig(
        ess_bulk_min=Gate1BayesianESS(correlations=200),  # tightened
    )
    diag = _clean_diag(
        ess_bulk_min_by_class={
            "fixed_effects": 2000.0,
            "iiv": 2000.0,
            "residual": 2000.0,
            "correlations": 150.0,  # below tightened threshold
        },
    )
    res = _make_result(diag)
    out = evaluate_gate1_bayesian(res, _policy(gate1_bayesian=cfg))
    assert out.passed is False
    assert "correlations" in out.summary_reason


# --- Non-applicable / missing diagnostics --------------------------------


def test_non_bayesian_backend_is_not_applicable() -> None:
    res = _make_result(None, backend="nlmixr2")
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is True
    assert "not_applicable" in out.summary_reason


def test_missing_posterior_diagnostics_fails() -> None:
    res = _make_result(None)  # backend=bayesian_stan but diag=None
    out = evaluate_gate1_bayesian(res, _policy())
    assert out.passed is False
    assert "posterior_diagnostics" in out.summary_reason.lower()


# --- Regression: the check list always includes divergence ---------------


def test_check_list_always_includes_divergences() -> None:
    res = _make_result(_clean_diag())
    out = evaluate_gate1_bayesian(res, _policy())
    assert any(c.check_id == "bayesian_divergences" for c in out.checks)


def test_timestamp_is_iso_utc() -> None:
    res = _make_result(_clean_diag())
    out = evaluate_gate1_bayesian(res, _policy())
    parsed = datetime.fromisoformat(out.timestamp)
    assert parsed.tzinfo is not None
    # Close-to-now: within 10 seconds of evaluation
    now = datetime.now(tz=UTC)
    assert abs((now - parsed).total_seconds()) < 10
