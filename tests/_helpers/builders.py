# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared Pydantic factories/builders for the APMODE test suite.

Single home for the fixture builders that were previously copy-pasted across
test modules (or imported cross-module from sibling test files):

- :func:`make_backend_result` — canonical ``BackendResult`` fixture (was
  ``tests/unit/test_gates.py::_make_backend_result``).
- :func:`make_spec` — canonical baseline ``DSLSpec`` (was
  ``tests/unit/test_semantic_validator.py::_make_spec``).
- :func:`make_data_manifest` — NCA-flavoured ``DataManifest`` (was
  ``tests/unit/test_nca_refinement.py::_make_manifest``).
- :func:`make_evidence_manifest` — ``EvidenceManifest`` (was
  ``tests/unit/test_search_candidates.py::_make_manifest``).
- :func:`base_spec` / :func:`mock_backend_result` / :func:`mock_data_manifest` /
  :func:`stop_response` — agentic-runner builders (were private helpers in
  ``tests/unit/test_agentic_runner.py``).

The builders are behaviour-identical to the canonical originals; the frozen
defaults are pinned by ``tests/unit/test_helpers_contract.py``.
"""

from __future__ import annotations

import json
from typing import cast

from apmode.backends.llm_client import LLMResponse
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    EvidenceManifest,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    PITCalibrationSummary,
    ScoringContract,
    SplitGOFMetrics,
    VPCSummary,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)

_DEFAULT_INITIAL = {"ka": 1.0, "V": 70.0, "CL": 5.0}


def _scoring_contract_for_backend(backend: str) -> ScoringContract:
    """Per-backend classical default for test fixtures (see
    :meth:`BackendResult.validate_backend_scoring_contract_consistency`)."""
    if backend == "bayesian_stan":
        return ScoringContract(
            nlpd_kind="marginal",
            re_treatment="integrated",
            nlpd_integrator="hmc_nuts",
            blq_method="none",
            observation_model="combined",
            float_precision="float64",
        )
    if backend == "jax_node":
        return ScoringContract(
            nlpd_kind="conditional",
            re_treatment="pooled",
            nlpd_integrator="none",
            blq_method="none",
            observation_model="combined",
            float_precision="float32",
        )
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method="none",
        observation_model="combined",
        float_precision="float64",
    )


def make_backend_result(
    *,
    converged: bool = True,
    ofv: float = 150.0,
    aic: float = 160.0,
    bic: float = 170.0,
    cwres_mean: float = 0.01,
    cwres_sd: float = 1.0,
    outlier_fraction: float = 0.02,
    r2: float | None = 0.95,
    vpc_coverage: dict[str, float] | None = None,
    pit_calibration: dict[str, float] | None = None,
    condition_number: float = 15.0,
    ill_conditioned: bool = False,
    profile_ci: dict[str, bool] | None = None,
    shrinkage: dict[str, float] | None = None,
    backend: str = "nlmixr2",
    method: str = "saem",
) -> BackendResult:
    """Build a BackendResult for testing."""
    from apmode.bundle.models import BackendResult as BR

    return BR(
        model_id="test_model",
        backend=backend,  # type: ignore[arg-type]
        converged=converged,
        ofv=ofv,
        aic=aic,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL", estimate=5.0, se=0.5, rse=10.0, category="structural"
            ),
            "V": ParameterEstimate(
                name="V", estimate=70.0, se=7.0, rse=10.0, category="structural"
            ),
            "ka": ParameterEstimate(
                name="ka", estimate=1.5, se=0.2, rse=13.0, category="structural"
            ),
        },
        eta_shrinkage=shrinkage or {"CL": 0.05, "V": 0.08, "ka": 0.12},
        convergence_metadata=ConvergenceMetadata(
            method=method,
            converged=converged,
            iterations=200,
            gradient_norm=0.001,
            minimization_status="successful",
            wall_time_seconds=45.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=cwres_mean,
                cwres_sd=cwres_sd,
                outlier_fraction=outlier_fraction,
                obs_vs_pred_r2=r2,
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage=vpc_coverage or {"p5": 0.92, "p50": 0.97, "p95": 0.93},
                n_bins=10,
                prediction_corrected=False,
            ),
            pit_calibration=PITCalibrationSummary(
                probability_levels=[0.05, 0.50, 0.95],
                # Well-calibrated default: c_p ≈ p so |Δ| = 0 on every
                # band. Tests that exercise PIT failure override via
                # the ``pit_calibration`` kwarg.
                calibration=pit_calibration or {"p5": 0.05, "p50": 0.50, "p95": 0.95},
                n_observations=400,
                n_subjects=50,
                aggregation="subject_robust",
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=condition_number,
                profile_likelihood_ci=profile_ci or {"CL": True, "V": True, "ka": True},
                ill_conditioned=ill_conditioned,
            ),
            blq=BLQHandling(
                method="none",
                n_blq=0,
                blq_fraction=0.0,
            ),
            state_trajectory_valid=True,
            scoring_contract=_scoring_contract_for_backend(backend),
            # Default split_gof so the required-check path doesn't
            # auto-fail non-split-related tests. Individual tests can
            # override.
            split_gof=SplitGOFMetrics(
                train_cwres_mean=0.01,
                train_outlier_fraction=0.02,
                test_cwres_mean=0.02,
                test_outlier_fraction=0.03,
                n_train=40,
                n_test=10,
            ),
        ),
        wall_time_seconds=45.0,
        backend_versions={"nlmixr2": "2.1.2", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


def make_spec(**overrides: object) -> DSLSpec:
    """Build a valid baseline DSLSpec, overriding specific modules.

    Pass ``initial=`` to fully replace the default calibration dict when
    overriding a structural module that needs different calibration names.
    """
    initial_override = overrides.pop("initial", None)
    defaults: dict[str, object] = {
        "model_id": "test_id_000000000000",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
        "initial": dict(_DEFAULT_INITIAL),
    }
    defaults.update(overrides)
    if initial_override is not None:
        defaults["initial"] = initial_override
    return DSLSpec(**defaults)  # type: ignore[arg-type]


def make_data_manifest(**overrides: object) -> DataManifest:
    """Build an NCA-flavoured ``DataManifest`` (was
    ``tests/unit/test_nca_refinement.py::_make_manifest``).

    ``n_subjects`` / ``n_observations`` / ``n_doses`` are overridable; by
    default ``n_doses`` tracks ``n_subjects`` (one dose per subject) unless
    overridden explicitly.
    """
    n_subjects = cast("int", overrides.pop("n_subjects", 30))
    n_observations = cast("int", overrides.pop("n_observations", 270))
    n_doses = cast("int", overrides.pop("n_doses", n_subjects))
    defaults: dict[str, object] = {
        "data_sha256": "0" * 64,
        "ingestion_format": "nonmem_csv",
        "column_mapping": ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        "n_subjects": n_subjects,
        "n_observations": n_observations,
        "n_doses": n_doses,
    }
    defaults.update(overrides)
    return DataManifest(**defaults)  # type: ignore[arg-type]


def make_evidence_manifest(**overrides: object) -> EvidenceManifest:
    """Create a test EvidenceManifest with defaults (was
    ``tests/unit/test_search_candidates.py::_make_manifest``)."""
    defaults: dict[str, object] = {
        "route_certainty": "confirmed",
        "absorption_complexity": "simple",
        "nonlinear_clearance_evidence_strength": "none",
        "richness_category": "moderate",
        "identifiability_ceiling": "medium",
        "covariate_burden": 2,
        "covariate_correlated": False,
        "blq_burden": 0.0,
        "protocol_heterogeneity": "single-study",
        "absorption_phase_coverage": "adequate",
        "elimination_phase_coverage": "adequate",
    }
    defaults.update(overrides)
    return EvidenceManifest(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Agentic-runner builders (were private helpers in test_agentic_runner.py)
# ---------------------------------------------------------------------------


def base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="base",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def mock_backend_result(
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
            state_trajectory_valid=True,
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


def mock_data_manifest() -> DataManifest:
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


def stop_response() -> LLMResponse:
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
