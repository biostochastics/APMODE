# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite B integration tests (PRD §5, Phase 2).

Tests NODE-specific validation scenarios:
  B1: NODE absorption recovery — spec construction and mock fit
  B2: Lane Router blocks NODE under sparse data
  B3: Cross-paradigm ranking produces correct qualified ordering
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.benchmarks.suite_b import (
    B1_REFERENCE_PARAMS,
    make_b3_result,
    scenario_b1,
    scenario_b2_spec,
    sparse_data_manifest,
    sparse_evidence_manifest,
)
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    PITCalibrationSummary,
    VPCSummary,
)
from apmode.governance.gates import evaluate_gate1, evaluate_gate2, evaluate_gate3
from apmode.governance.policy import GatePolicy
from apmode.routing import route

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((POLICY_DIR / f"{lane}.json").read_text()))


# ---------------------------------------------------------------------------
# B1: NODE absorption recovery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteBAbsorptionRecovery:
    """B1: NODE spec construction and mock fit for absorption recovery."""

    def test_b1_spec_has_node_modules(self) -> None:
        """B1 spec uses NODE absorption."""
        spec = scenario_b1()
        assert spec.has_node_modules()
        assert spec.absorption.type == "NODE_Absorption"

    def test_b1_spec_structural_params(self) -> None:
        """B1 structural params are the mechanistic ones (not NODE weights)."""
        spec = scenario_b1()
        params = spec.structural_param_names()
        assert "V1" in params
        assert "V2" in params
        assert "Q" in params
        assert "CL" in params
        # NODE absorption weights are NOT named structural params
        assert "ka" not in params

    def test_b1_mock_fit_converges(self) -> None:
        """Mock NODE fit passes Gate 1 with seed results."""
        spec = scenario_b1()
        result = _make_node_fit(spec, B1_REFERENCE_PARAMS)
        seed1 = _make_node_fit(spec, B1_REFERENCE_PARAMS, bias=0.025)
        seed2 = _make_node_fit(spec, B1_REFERENCE_PARAMS, bias=0.015)

        policy = _load_policy("discovery")
        g1 = evaluate_gate1(result, policy, seed_results=[seed1, seed2])
        assert g1.passed, f"B1 failed Gate 1: {g1.summary_reason}"

    def test_b1_passes_gate2_discovery(self) -> None:
        """B1 NODE spec passes Gate 2 in discovery lane."""
        spec = scenario_b1()
        result = _make_node_fit(spec, B1_REFERENCE_PARAMS)
        policy = _load_policy("discovery")
        g2 = evaluate_gate2(result, policy, lane="discovery")
        assert g2.passed, f"B1 failed Gate 2 discovery: {g2.summary_reason}"

    def test_b1_fails_gate2_submission(self) -> None:
        """B1 NODE spec is excluded from submission lane."""
        spec = scenario_b1()
        result = _make_node_fit(spec, B1_REFERENCE_PARAMS)
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert not g2.passed, "B1 NODE should not pass submission Gate 2"


# ---------------------------------------------------------------------------
# B2: NODE under sparse data — dispatch constraint chain
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteBSparseData:
    """B2: Lane Router correctly blocks NODE under sparse data."""

    def test_b2_sparse_manifest_flags(self) -> None:
        """Sparse evidence manifest has correct flags."""
        manifest = sparse_evidence_manifest()
        assert manifest.richness_category == "sparse"
        assert manifest.absorption_phase_coverage == "inadequate"
        assert manifest.identifiability_ceiling == "low"

    def test_b2_discovery_lane_blocks_node(self) -> None:
        """Discovery lane removes jax_node when data is sparse + inadequate."""
        manifest = sparse_evidence_manifest()
        dispatch = route("discovery", manifest)
        assert "jax_node" not in dispatch.backends
        assert not dispatch.data_sufficient_for_node

    def test_b2_submission_lane_never_has_node(self) -> None:
        """Submission lane never includes jax_node regardless of data quality."""
        manifest = sparse_evidence_manifest()
        dispatch = route("submission", manifest)
        assert "jax_node" not in dispatch.backends

    def test_b2_adequate_data_allows_node(self) -> None:
        """With adequate data, discovery lane includes jax_node."""
        from apmode.bundle.models import EvidenceManifest

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="moderate",
            identifiability_ceiling="medium",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.0,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
            node_dim_budget=8,
        )
        dispatch = route("discovery", manifest)
        assert "jax_node" in dispatch.backends
        assert dispatch.data_sufficient_for_node

    def test_b2_spec_is_node(self) -> None:
        """B2 spec has NODE elimination modules."""
        spec = scenario_b2_spec()
        assert spec.has_node_modules()
        assert spec.elimination.type == "NODE_Elimination"

    def test_b2_data_manifest_is_sparse(self) -> None:
        """B2 data manifest reflects sparse sampling."""
        dm = sparse_data_manifest()
        # 60 obs / 20 subjects = 3 obs/subject — sparse
        assert dm.n_observations / dm.n_subjects < 4


# ---------------------------------------------------------------------------
# B3: Cross-paradigm ranking
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteBCrossParadigmRanking:
    """B3: Cross-paradigm ranking with mixed backends."""

    def test_b3_ranking_orders_by_composite(self) -> None:
        """Better VPC concordance should rank higher when cross-paradigm.

        Updated in v0.3.0-rc5: Gate 3 cross-paradigm composite defaults now
        have ``bic_weight=0`` (PRD §10 Q2 — likelihoods incomparable across
        observation models), so BIC alone cannot decide ranking. Tests
        exercising cross-paradigm precedence must vary VPC/NPE instead.
        """
        # NODE fits the observed percentile bands more tightly.
        classical = make_b3_result(
            "classical_1",
            "nlmixr2",
            bic=550.0,
            vpc_coverage={"p5": 0.85, "p50": 0.87, "p95": 0.84},
        )
        node = make_b3_result(
            "node_1",
            "jax_node",
            bic=520.0,
            vpc_coverage={"p5": 0.91, "p50": 0.93, "p95": 0.90},
        )

        policy = _load_policy("discovery")
        g3, ranked = evaluate_gate3([classical, node], policy)

        assert g3.passed
        assert len(ranked) == 2
        # Better VPC concordance (NODE) should rank first.
        assert ranked[0].candidate_id == "node_1"
        assert ranked[1].candidate_id == "classical_1"

    def test_b3_mixed_backends_in_ranking(self) -> None:
        """Both backends are represented in the ranking."""
        classical = make_b3_result("c1", "nlmixr2", bic=540.0)
        node = make_b3_result("n1", "jax_node", bic=535.0)

        policy = _load_policy("discovery")
        _, ranked = evaluate_gate3([classical, node], policy)

        backends = {rc.backend for rc in ranked}
        assert backends == {"nlmixr2", "jax_node"}

    def test_b3_single_backend_ranking(self) -> None:
        """Single-backend ranking still works (no cross-paradigm needed)."""
        c1 = make_b3_result("c1", "nlmixr2", bic=540.0)
        c2 = make_b3_result("c2", "nlmixr2", bic=530.0)

        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([c1, c2], policy)

        assert g3.passed
        assert ranked[0].candidate_id == "c2"  # lower BIC

    def test_b3_three_candidate_ranking(self) -> None:
        """Three candidates from mixed backends ranked by VPC/NPE composite.

        Updated in v0.3.0-rc5 for the same reason as
        ``test_b3_ranking_orders_by_composite``. VPC concordance is set to
        monotonically differ across candidates so the ordering is
        deterministic under the default Gate3Config (BIC off).
        """
        c1 = make_b3_result(
            "classical_a",
            "nlmixr2",
            bic=560.0,
            vpc_coverage={"p5": 0.80, "p50": 0.82, "p95": 0.81},
        )
        c2 = make_b3_result(
            "node_a",
            "jax_node",
            bic=525.0,
            vpc_coverage={"p5": 0.91, "p50": 0.93, "p95": 0.90},
        )
        c3 = make_b3_result(
            "classical_b",
            "nlmixr2",
            bic=535.0,
            vpc_coverage={"p5": 0.87, "p50": 0.88, "p95": 0.86},
        )

        policy = _load_policy("discovery")
        g3, ranked = evaluate_gate3([c1, c2, c3], policy)

        assert g3.passed
        assert len(ranked) == 3
        # Ordered by VPC concordance: node_a > classical_b > classical_a.
        assert ranked[0].candidate_id == "node_a"
        assert ranked[1].candidate_id == "classical_b"
        assert ranked[2].candidate_id == "classical_a"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_fit(
    spec: object,
    reference: dict[str, float],
    bias: float = 0.02,
) -> BackendResult:
    """Mock jax_node BackendResult for NODE specs."""
    estimates = {
        n: ParameterEstimate(
            name=n,
            estimate=v * (1 + bias),
            se=v * 0.1,
            rse=10.0,
            ci95_lower=v * 0.8,
            ci95_upper=v * 1.2,
            category="structural",
        )
        for n, v in reference.items()
    }
    model_id = spec.model_id if hasattr(spec, "model_id") else "node_mock"
    return BackendResult(
        model_id=model_id,
        backend="jax_node",
        converged=True,
        ofv=500.0,
        aic=520.0,
        bic=540.0,
        parameter_estimates=estimates,
        eta_shrinkage={"CL": 0.05},
        convergence_metadata=ConvergenceMetadata(
            method="adam",
            converged=True,
            iterations=200,
            gradient_norm=0.001,
            minimization_status="successful",
            wall_time_seconds=45.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=0.03, cwres_sd=1.02, outlier_fraction=0.01, obs_vs_pred_r2=0.94
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.91, "p50": 0.95, "p95": 0.92},
                n_bins=10,
                prediction_corrected=False,
            ),
            pit_calibration=PITCalibrationSummary(
                probability_levels=[0.05, 0.50, 0.95],
                calibration={"p5": 0.05, "p50": 0.50, "p95": 0.95},
                n_observations=96,
                n_subjects=12,
                aggregation="subject_robust",
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=20.0,
                profile_likelihood_ci={n: True for n in reference},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=45.0,
        backend_versions={"jax": "0.9.2", "python": "3.12.0"},
        initial_estimate_source="nca",
    )
