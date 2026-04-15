# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite B: NODE-specific validation scenarios (PRD §5, Phase 2).

Three scenarios validating the hybrid NODE backend and cross-paradigm ranking:

  B1: NODE absorption recovery — can NODE learn a known nonlinear absorption shape?
      Uses A7's saturable absorption ground truth. Tests distillation surrogate
      fidelity (AUC/Cmax within 80-125% GMR per PRD §5).

  B2: NODE elimination under sparse data — does the Lane Router correctly flag
      data_insufficient and withhold NODE dispatch when sampling is too sparse?
      Tests the manifest → dispatch constraint chain (PRD §4.2.1).

  B3: Cross-paradigm ranking correctness — given converged results from both
      nlmixr2 and jax_node, does Gate 3 produce a correct qualified ranking
      using VPC concordance + NPE composite score? (PRD §4.3.1)
"""

from __future__ import annotations

from typing import Literal

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
    VPCSummary,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    NODEElimination,
    OneCmt,
    Proportional,
    TwoCmt,
)

# ---------------------------------------------------------------------------
# B1: NODE absorption recovery spec
# ---------------------------------------------------------------------------


def scenario_b1() -> DSLSpec:
    """B1: 2-cmt with NODE absorption for shape recovery testing.

    Same structural model as A7 but specifically for NODE backend validation.
    The NODE should learn the saturable absorption dynamics from simulated data.
    """
    return DSLSpec(
        model_id="suite_b_scenario_b1",
        absorption=NODEAbsorption(dim=4, constraint_template="bounded_positive"),
        distribution=TwoCmt(V1=50.0, V2=80.0, Q=10.0),
        elimination=LinearElim(CL=4.0),
        variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


# ---------------------------------------------------------------------------
# B2: Sparse data + NODE elimination (should be blocked by dispatch)
# ---------------------------------------------------------------------------


def scenario_b2_spec() -> DSLSpec:
    """B2: NODE elimination spec that should NOT be dispatched under sparse data."""
    return DSLSpec(
        model_id="suite_b_scenario_b2",
        absorption=FirstOrder(ka=1.5),
        distribution=OneCmt(V=70.0),
        elimination=NODEElimination(dim=3, constraint_template="monotone_decreasing"),
        variability=[IIV(params=["V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
    )


def sparse_evidence_manifest() -> EvidenceManifest:
    """EvidenceManifest flagging sparse data + inadequate absorption coverage.

    This combination should cause the Lane Router to remove jax_node from
    the admissible backends list (PRD §4.2.1).
    """
    return EvidenceManifest(
        route_certainty="confirmed",
        absorption_complexity="simple",
        nonlinear_clearance_evidence_strength="none",
        richness_category="sparse",
        identifiability_ceiling="low",
        covariate_burden=0,
        covariate_correlated=False,
        blq_burden=0.0,
        protocol_heterogeneity="single-study",
        absorption_phase_coverage="inadequate",
        elimination_phase_coverage="adequate",
    )


def sparse_data_manifest() -> DataManifest:
    """DataManifest for a sparse dataset (< 4 obs/subject)."""
    return DataManifest(
        data_sha256="b2" * 32,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=20,
        n_observations=60,  # 3 obs/subject — sparse
        n_doses=20,
    )


# ---------------------------------------------------------------------------
# B3: Cross-paradigm ranking mock results
# ---------------------------------------------------------------------------


def make_b3_result(
    model_id: str,
    backend: Literal["nlmixr2", "jax_node"],
    bic: float,
    vpc_coverage: dict[str, float] | None = None,
) -> BackendResult:
    """Build a mock BackendResult for cross-paradigm ranking validation.

    Creates realistic results from different backends to test that
    Gate 3 ranking correctly handles mixed-paradigm candidate sets
    with qualified comparison flags.
    """
    if vpc_coverage is None:
        vpc_coverage = {"p5": 0.92, "p50": 0.96, "p95": 0.93}

    return BackendResult(
        model_id=model_id,
        backend=backend,
        converged=True,
        ofv=bic - 40.0,
        aic=bic - 20.0,
        bic=bic,
        parameter_estimates={
            "ka": ParameterEstimate(
                name="ka",
                estimate=1.5,
                se=0.15,
                rse=10.0,
                ci95_lower=1.2,
                ci95_upper=1.8,
                category="structural",
            ),
            "V": ParameterEstimate(
                name="V",
                estimate=70.0,
                se=7.0,
                rse=10.0,
                ci95_lower=56.0,
                ci95_upper=84.0,
                category="structural",
            ),
            "CL": ParameterEstimate(
                name="CL",
                estimate=5.0,
                se=0.5,
                rse=10.0,
                ci95_lower=4.0,
                ci95_upper=6.0,
                category="structural",
            ),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem" if backend == "nlmixr2" else "adam",
            converged=True,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status="successful",
            wall_time_seconds=60.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=0.02,
                cwres_sd=1.01,
                outlier_fraction=0.01,
                obs_vs_pred_r2=0.95,
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage=vpc_coverage,
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"ka": True, "V": True, "CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=60.0,
        backend_versions=(
            {"nlmixr2": "3.0.0", "R": "4.4.1"}
            if backend == "nlmixr2"
            else {"jax": "0.9.2", "python": "3.12.0"}
        ),
        initial_estimate_source="nca",
    )


# Reference parameters for B scenarios
B1_REFERENCE_PARAMS: dict[str, float] = {
    "V1": 50.0,
    "V2": 80.0,
    "Q": 10.0,
    "CL": 4.0,
}

ALL_SCENARIOS = [
    ("B1", scenario_b1),
    ("B2", scenario_b2_spec),
]
