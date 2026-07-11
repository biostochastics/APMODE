# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 2 (Lane Admissibility), Gate 3 (within-paradigm ranking),
and the SearchSpace dispatch constraints.

Verifies the governance funnel: gates are sequential disqualifiers,
failures logged with per-check reasons, thresholds from policy files.
"""

from __future__ import annotations

import pytest

from apmode.governance.gates import (
    evaluate_gate2,
    evaluate_gate3,
)
from tests._helpers.builders import make_backend_result as _make_backend_result
from tests._helpers.policies import load_policy as _load_policy

# ---------------------------------------------------------------------------
# Gate 2 Tests
# ---------------------------------------------------------------------------


class TestGate2:
    """Gate 2: Lane-Specific Admissibility."""

    def test_submission_all_passing(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is True
        assert g2.gate_name == "lane_admissibility"

    def test_submission_rejects_node(self) -> None:
        result = _make_backend_result(backend="jax_node")
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is False
        failed_ids = {c.check_id for c in g2.checks if not c.passed}
        assert "node_eligibility" in failed_ids

    def test_discovery_allows_node(self) -> None:
        result = _make_backend_result(backend="jax_node")
        policy = _load_policy("discovery")
        g2 = evaluate_gate2(result, policy, lane="discovery")
        node_check = next(c for c in g2.checks if c.check_id == "node_eligibility")
        assert node_check.passed is True

    def test_submission_shrinkage_too_high(self) -> None:
        result = _make_backend_result(shrinkage={"CL": 0.05, "V": 0.45, "ka": 0.10})
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is False
        failed_ids = {c.check_id for c in g2.checks if not c.passed}
        assert "shrinkage" in failed_ids

    def test_identifiability_ill_conditioned(self) -> None:
        result = _make_backend_result(ill_conditioned=True, condition_number=5000.0)
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is False
        failed_ids = {c.check_id for c in g2.checks if not c.passed}
        assert "identifiability" in failed_ids

    def test_identifiability_missing_profile_ci(self) -> None:
        result = _make_backend_result(profile_ci={"CL": True, "V": False, "ka": True})
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is False
        ident_check = next(c for c in g2.checks if c.check_id == "identifiability")
        assert ident_check.passed is False

    def test_discovery_no_identifiability_required(self) -> None:
        result = _make_backend_result(ill_conditioned=True)
        policy = _load_policy("discovery")
        g2 = evaluate_gate2(result, policy, lane="discovery")
        ident_check = next(c for c in g2.checks if c.check_id == "identifiability")
        assert ident_check.passed is True  # discovery doesn't require identifiability

    def test_optimization_loro_required(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("optimization")
        g2 = evaluate_gate2(result, policy, lane="optimization")
        loro_check = next(c for c in g2.checks if c.check_id == "loro_required")
        # LORO not yet implemented → fails
        assert loro_check.passed is False

    def test_invalid_lane_raises(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("submission")
        with pytest.raises(ValueError, match="Invalid lane"):
            evaluate_gate2(result, policy, lane="invalid_lane")

    def test_gate2_has_all_checks(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        check_ids = {c.check_id for c in g2.checks}
        expected = {
            "interpretable_parameterization",
            "shrinkage",
            "identifiability",
            "node_eligibility",
            "reproducible_estimation",
            "loro_required",
            "bayesian_prior_justification",
            # plan Tasks 20 + 21
            "prior_data_conflict",
            "prior_sensitivity",
        }
        assert expected == check_ids


# ---------------------------------------------------------------------------
# Gate 3 Tests
# ---------------------------------------------------------------------------


class TestGate3:
    """Gate 3: Within-Paradigm Ranking."""

    def test_ranking_by_bic(self) -> None:
        r1 = _make_backend_result(bic=170.0)
        r2 = _make_backend_result(bic=160.0)
        r3 = _make_backend_result(bic=180.0)
        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([r1, r2, r3], policy)
        assert g3.passed is True
        assert g3.gate_name == "within_paradigm_ranking"
        assert len(ranked) == 3
        assert ranked[0].bic == 160.0
        assert ranked[1].bic == 170.0
        assert ranked[2].bic == 180.0

    def test_empty_survivors(self) -> None:
        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([], policy)
        assert g3.passed is False
        assert ranked == []

    def test_single_survivor(self) -> None:
        r1 = _make_backend_result(bic=150.0)
        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([r1], policy)
        assert g3.passed is True
        assert len(ranked) == 1
        assert ranked[0].rank == 1

    def test_tie_breaking_equal_bic(self) -> None:
        """Equal BIC: ranking should be stable (all get ranked, no crash)."""
        r1 = _make_backend_result(bic=170.0)
        r2 = _make_backend_result(bic=170.0)
        r3 = _make_backend_result(bic=170.0)
        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([r1, r2, r3], policy)
        assert g3.passed is True
        assert len(ranked) == 3
        # All should be ranked 1..3
        assert {rc.rank for rc in ranked} == {1, 2, 3}
        # BIC spread should be 0
        bic_spread = next(c for c in g3.checks if c.check_id == "bic_spread")
        assert bic_spread.observed == 0.0

    def test_none_bic_sorted_last(self) -> None:
        """Candidates with missing BIC are excluded from BIC ranking."""
        r1 = _make_backend_result(bic=170.0)
        r2 = _make_backend_result(bic=None)  # type: ignore[arg-type]
        policy = _load_policy("submission")
        _g3, ranked = evaluate_gate3([r1, r2], policy)
        assert ranked[0].bic == 170.0
        assert len(ranked) == 1


# ---------------------------------------------------------------------------
# Dispatch Constraint Tests
# ---------------------------------------------------------------------------


class TestDispatchConstraints:
    """Verify BLQ and IOV dispatch constraints in SearchSpace."""

    def test_blq_forces_m3(self) -> None:
        from apmode.bundle.models import EvidenceManifest
        from apmode.search.candidates import SearchSpace

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.25,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        space = SearchSpace.from_manifest(manifest)
        assert space.force_blq_method == "m3"

    def test_heterogeneous_forces_iov(self) -> None:
        from apmode.bundle.models import EvidenceManifest
        from apmode.search.candidates import SearchSpace

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.05,
            protocol_heterogeneity="pooled-heterogeneous",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        space = SearchSpace.from_manifest(manifest)
        assert space.force_iov is True

    def test_blq_m3_in_generated_candidates(self) -> None:
        from apmode.dsl.ast_models import BLQM3
        from apmode.search.candidates import SearchSpace, generate_root_candidates

        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
            force_blq_method="m3",
        )
        candidates = generate_root_candidates(space)
        assert len(candidates) == 1
        assert isinstance(candidates[0].observation, BLQM3)

    def test_iov_in_generated_candidates(self) -> None:
        from apmode.dsl.ast_models import IOV
        from apmode.search.candidates import SearchSpace, generate_root_candidates

        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
            force_iov=True,
        )
        candidates = generate_root_candidates(space)
        assert len(candidates) == 1
        iov_items = [v for v in candidates[0].variability if isinstance(v, IOV)]
        assert len(iov_items) == 1
        assert iov_items[0].params == ["CL"]

    def test_compound_blq_and_iov_constraints(self) -> None:
        """BLQ > 0.20 + pooled-heterogeneous: both M3 and IOV in candidates."""
        from apmode.bundle.models import EvidenceManifest
        from apmode.dsl.ast_models import BLQM3, IOV
        from apmode.search.candidates import SearchSpace, generate_root_candidates

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.25,
            protocol_heterogeneity="pooled-heterogeneous",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        space = SearchSpace.from_manifest(manifest)
        assert space.force_blq_method == "m3"
        assert space.force_iov is True

        candidates = generate_root_candidates(space)
        assert len(candidates) >= 1
        for c in candidates:
            assert isinstance(c.observation, BLQM3)
            iov_items = [v for v in c.variability if isinstance(v, IOV)]
            assert len(iov_items) == 1
