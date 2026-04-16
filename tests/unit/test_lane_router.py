# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Lane Router dispatch decisions."""

from __future__ import annotations

from apmode.bundle.models import EvidenceManifest
from apmode.governance.policy import MissingDataPolicy
from apmode.routing import route

# Default policy used when a test needs the BLQ advisory string. All live
# orchestrator calls pass a policy; the pre-v0.3 unversioned 0.20 literal
# fallback has been removed, so tests that exercise BLQ routing must
# explicitly opt in to a policy to match production behavior.
_DEFAULT_BLQ_POLICY = MissingDataPolicy(blq_m3_threshold=0.10)


def _make_manifest(**overrides: object) -> EvidenceManifest:
    """Build a default EvidenceManifest with overrides."""
    # Manifest schema v2 (2026-04-15): `node_dim_budget` gates NODE in the
    # Lane Router. A rich default fixture (richness=rich, absorption=adequate)
    # must explicitly set the budget that the profiler would derive.
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
        "node_dim_budget": 8,
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
        decision = route("submission", manifest, _DEFAULT_BLQ_POLICY)
        # Directive-driven advisory replaces the legacy "M3/M4 required" string.
        assert any("BLQ method M3" in c for c in decision.constraints)

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

    def test_blq_below_policy_threshold_selects_m7plus(self) -> None:
        """blq_burden below the policy's ``blq_m3_threshold`` → M7+."""
        manifest = _make_manifest(blq_burden=0.05)
        decision = route("submission", manifest, _DEFAULT_BLQ_POLICY)
        # Directive resolves to M7+ when burden ≤ threshold (Wijk 2025).
        assert any("BLQ method M7+" in c for c in decision.constraints)

    def test_blq_above_policy_threshold_selects_m3(self) -> None:
        """blq_burden above ``blq_m3_threshold`` → M3 (Beal 2001)."""
        manifest = _make_manifest(blq_burden=0.21)
        decision = route("submission", manifest, _DEFAULT_BLQ_POLICY)
        assert any("BLQ method M3" in c for c in decision.constraints)

    def test_compound_blq_and_heterogeneous(self) -> None:
        """High BLQ + pooled-heterogeneous: both constraints noted."""
        manifest = _make_manifest(
            blq_burden=0.25,
            protocol_heterogeneity="pooled-heterogeneous",
        )
        decision = route("submission", manifest, _DEFAULT_BLQ_POLICY)
        assert any("BLQ method M3" in c for c in decision.constraints)
        assert any("IOV" in c for c in decision.constraints)

    def test_compound_sparse_and_blq(self) -> None:
        """Sparse + inadequate + high BLQ: NODE removed AND M3 advisory present."""
        manifest = _make_manifest(
            richness_category="sparse",
            absorption_phase_coverage="inadequate",
            blq_burden=0.30,
        )
        decision = route("discovery", manifest, _DEFAULT_BLQ_POLICY)
        assert "jax_node" not in decision.backends
        assert decision.data_sufficient_for_node is False
        assert any("BLQ method M3" in c for c in decision.constraints)

    def test_no_policy_suppresses_blq_advisory(self) -> None:
        """Without a policy, no BLQ advisory is emitted (pre-v0.3 0.20 literal removed)."""
        manifest = _make_manifest(blq_burden=0.50)
        decision = route("submission", manifest)
        assert not any(c.startswith("BLQ") for c in decision.constraints)

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
