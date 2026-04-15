# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Lane Router dispatch decisions."""

from __future__ import annotations

from apmode.bundle.models import EvidenceManifest
from apmode.routing import route


def _make_manifest(**overrides: object) -> EvidenceManifest:
    """Build a default EvidenceManifest with overrides."""
    defaults = {
        "route_certainty": "confirmed",
        "absorption_complexity": "simple",
        "nonlinear_clearance_evidence_strength": "none",
        "richness_category": "rich",
        "identifiability_ceiling": "high",
        "covariate_burden": 0,
        "covariate_correlated": False,
        "blq_burden": 0.05,
        "protocol_heterogeneity": "single-study",
        "absorption_phase_coverage": "adequate",
        "elimination_phase_coverage": "adequate",
    }
    defaults.update(overrides)
    return EvidenceManifest(**defaults)  # type: ignore[arg-type]


class TestLaneRouter:
    """Lane Router dispatch decision tests."""

    def test_submission_excludes_node(self) -> None:
        decision = route("submission", _make_manifest())
        assert "jax_node" not in decision.backends
        assert decision.node_eligible is False

    def test_discovery_includes_node(self) -> None:
        decision = route("discovery", _make_manifest())
        assert "jax_node" in decision.backends
        assert decision.node_eligible is True

    def test_sparse_inadequate_removes_node(self) -> None:
        manifest = _make_manifest(
            richness_category="sparse",
            absorption_phase_coverage="inadequate",
        )
        decision = route("discovery", manifest)
        assert "jax_node" not in decision.backends
        assert decision.data_sufficient_for_node is False

    def test_low_identifiability_removes_node(self) -> None:
        manifest = _make_manifest(identifiability_ceiling="low")
        decision = route("discovery", manifest)
        assert "jax_node" not in decision.backends
        assert decision.data_sufficient_for_node is False

    def test_blq_constraint_noted(self) -> None:
        manifest = _make_manifest(blq_burden=0.25)
        decision = route("submission", manifest)
        assert any("M3/M4" in c for c in decision.constraints)

    def test_heterogeneous_constraint_noted(self) -> None:
        manifest = _make_manifest(protocol_heterogeneity="pooled-heterogeneous")
        decision = route("submission", manifest)
        assert any("IOV" in c for c in decision.constraints)

    def test_submission_always_has_nlmixr2(self) -> None:
        decision = route("submission", _make_manifest())
        assert "nlmixr2" in decision.backends

    def test_optimization_has_all_backends(self) -> None:
        decision = route("optimization", _make_manifest())
        assert "nlmixr2" in decision.backends
        assert decision.node_eligible is True

    def test_blq_burden_at_boundary_no_constraint(self) -> None:
        """blq_burden=0.20 exactly should NOT trigger the M3/M4 constraint (> 0.20)."""
        manifest = _make_manifest(blq_burden=0.20)
        decision = route("submission", manifest)
        assert not any("M3/M4" in c for c in decision.constraints)

    def test_blq_burden_just_above_boundary(self) -> None:
        """blq_burden=0.21 should trigger the M3/M4 constraint."""
        manifest = _make_manifest(blq_burden=0.21)
        decision = route("submission", manifest)
        assert any("M3/M4" in c for c in decision.constraints)

    def test_compound_blq_and_heterogeneous(self) -> None:
        """Both BLQ > 0.20 AND pooled-heterogeneous: both constraints noted."""
        manifest = _make_manifest(
            blq_burden=0.25,
            protocol_heterogeneity="pooled-heterogeneous",
        )
        decision = route("submission", manifest)
        assert any("M3/M4" in c for c in decision.constraints)
        assert any("IOV" in c for c in decision.constraints)

    def test_compound_sparse_and_blq(self) -> None:
        """Sparse + inadequate + high BLQ: NODE removed AND BLQ constraint noted."""
        manifest = _make_manifest(
            richness_category="sparse",
            absorption_phase_coverage="inadequate",
            blq_burden=0.30,
        )
        decision = route("discovery", manifest)
        assert "jax_node" not in decision.backends
        assert decision.data_sufficient_for_node is False
        assert any("M3/M4" in c for c in decision.constraints)

    def test_invalid_lane_raises(self) -> None:
        import pytest

        manifest = _make_manifest()
        with pytest.raises(ValueError, match="Invalid lane"):
            route("invalid_lane", manifest)  # type: ignore[arg-type]

    def test_covariate_missingness_noted(self) -> None:
        from apmode.bundle.models import CovariateSpec

        manifest = _make_manifest(
            covariate_missingness=CovariateSpec(
                pattern="MCAR",
                fraction_incomplete=0.25,
                strategy="impute-median",
            ),
        )
        decision = route("submission", manifest)
        assert any("missingness" in c.lower() for c in decision.constraints)


class TestRouteWithPolicy:
    """Policy-driven ``MissingDataDirective`` is attached to DispatchDecision."""

    def test_no_policy_no_directive(self) -> None:
        manifest = _make_manifest()
        decision = route("discovery", manifest)
        assert decision.missing_data_directive is None

    def test_policy_attaches_directive(self) -> None:
        from apmode.governance.policy import MissingDataPolicy

        manifest = _make_manifest(blq_burden=0.30)
        decision = route("submission", manifest, MissingDataPolicy(blq_m3_threshold=0.10))
        assert decision.missing_data_directive is not None
        assert decision.missing_data_directive.blq_method == "M3"

    def test_directive_m7plus_when_below_threshold(self) -> None:
        from apmode.governance.policy import MissingDataPolicy

        manifest = _make_manifest(blq_burden=0.05)
        decision = route("discovery", manifest, MissingDataPolicy(blq_m3_threshold=0.10))
        assert decision.missing_data_directive is not None
        assert decision.missing_data_directive.blq_method == "M7+"
