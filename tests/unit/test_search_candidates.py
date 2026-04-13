# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for automated search candidate generation (PRD §4.2.3)."""

from __future__ import annotations

import pytest

from apmode.bundle.models import EvidenceManifest
from apmode.search.candidates import (
    SearchDAG,
    SearchSpace,
    generate_root_candidates,
)


def _make_manifest(**overrides: object) -> EvidenceManifest:
    """Create a test EvidenceManifest with defaults."""
    defaults = {
        "route_certainty": "confirmed",
        "absorption_complexity": "simple",
        "nonlinear_clearance_signature": False,
        "richness_category": "moderate",
        "identifiability_ceiling": "medium",
        "covariate_burden": 2,
        "covariate_correlated": False,
        "blq_burden": 0.0,
        "protocol_heterogeneity": "single-study",
        "absorption_phase_coverage": "adequate",
        "elimination_phase_coverage": "adequate",
    }
    defaults.update(overrides)  # type: ignore[arg-type]
    return EvidenceManifest(**defaults)  # type: ignore[arg-type]


class TestSearchSpace:
    """SearchSpace creation and manifest constraints."""

    def test_default_search_space(self) -> None:
        space = SearchSpace()
        assert 1 in space.structural_cmt
        assert "first_order" in space.absorption_types
        assert "linear" in space.elimination_types

    def test_sparse_data_constrains_space(self) -> None:
        manifest = _make_manifest(richness_category="sparse")
        space = SearchSpace.from_manifest(manifest)
        assert space.structural_cmt == [1]
        assert space.absorption_types == ["first_order"]

    def test_nonlinear_clearance_includes_mm(self) -> None:
        manifest = _make_manifest(nonlinear_clearance_signature=True)
        space = SearchSpace.from_manifest(manifest)
        assert "michaelis_menten" in space.elimination_types

    def test_linear_clearance_excludes_mm(self) -> None:
        manifest = _make_manifest(nonlinear_clearance_signature=False)
        space = SearchSpace.from_manifest(manifest)
        assert space.elimination_types == ["linear"]

    def test_simple_absorption_deprioritizes_transit(self) -> None:
        manifest = _make_manifest(absorption_complexity="simple")
        space = SearchSpace.from_manifest(manifest)
        assert "transit" not in space.absorption_types

    def test_lag_signature_includes_lagged(self) -> None:
        manifest = _make_manifest(absorption_complexity="lag-signature")
        space = SearchSpace.from_manifest(manifest)
        assert "lagged_first_order" in space.absorption_types

    def test_covariates_added_to_space(self) -> None:
        manifest = _make_manifest()
        space = SearchSpace.from_manifest(manifest, covariate_names=["WT", "AGE"])
        assert len(space.covariates) > 0


class TestGenerateRootCandidates:
    """Root candidate generation from search space."""

    def test_generates_candidates(self) -> None:
        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
        )
        candidates = generate_root_candidates(space)
        assert len(candidates) == 1

    def test_multiple_dimensions(self) -> None:
        space = SearchSpace(
            structural_cmt=[1, 2],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional", "combined"],
        )
        candidates = generate_root_candidates(space)
        assert len(candidates) == 4  # 2 cmt x 1 abs x 1 elim x 2 err

    def test_candidates_have_unique_ids(self) -> None:
        space = SearchSpace(
            structural_cmt=[1, 2],
            absorption_types=["first_order", "lagged_first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
        )
        candidates = generate_root_candidates(space)
        ids = [c.model_id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_candidates_pass_validation(self) -> None:
        from apmode.backends.protocol import Lane
        from apmode.dsl.validator import validate_dsl

        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
        )
        candidates = generate_root_candidates(space)
        for spec in candidates:
            errors = validate_dsl(spec, lane=Lane.SUBMISSION)
            assert len(errors) == 0, f"Validation errors for {spec.model_id}: {errors}"

    def test_custom_base_params(self) -> None:
        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
        )
        params = {"ka": 2.0, "CL": 10.0, "V": 50.0}
        candidates = generate_root_candidates(space, base_params=params)
        assert len(candidates) == 1
        assert candidates[0].absorption.ka == 2.0  # type: ignore[union-attr]


class TestSearchDAG:
    """Search DAG tracking for candidate lineage."""

    def test_add_root(self) -> None:
        dag = SearchDAG()
        from apmode.benchmarks.suite_a import scenario_a1

        spec = scenario_a1()
        node = dag.add_root(spec)
        assert node.parent_id is None
        assert dag.size == 1

    def test_add_child(self) -> None:
        dag = SearchDAG()
        from apmode.benchmarks.suite_a import scenario_a1, scenario_a4

        parent = scenario_a1()
        child = scenario_a4()
        dag.add_root(parent)
        dag.add_child(parent.model_id, child, "swap_elimination:MichaelisMenten")
        assert dag.size == 2

    def test_get_children(self) -> None:
        dag = SearchDAG()
        from apmode.benchmarks.suite_a import scenario_a1, scenario_a4

        parent = scenario_a1()
        child = scenario_a4()
        dag.add_root(parent)
        dag.add_child(parent.model_id, child, "swap_elimination")
        children = dag.get_children(parent.model_id)
        assert len(children) == 1
        assert children[0].candidate_id == child.model_id

    def test_update_score(self) -> None:
        dag = SearchDAG()
        from apmode.benchmarks.suite_a import scenario_a1

        spec = scenario_a1()
        dag.add_root(spec)
        dag.update_score(spec.model_id, score=150.5, converged=True)
        node = dag.get_node(spec.model_id)
        assert node is not None
        assert node.score == pytest.approx(150.5)
        assert node.converged is True

    def test_to_lineage_entries(self) -> None:
        dag = SearchDAG()
        from apmode.benchmarks.suite_a import scenario_a1, scenario_a4

        parent = scenario_a1()
        child = scenario_a4()
        dag.add_root(parent)
        dag.add_child(parent.model_id, child, "swap_elimination")
        entries = dag.to_lineage_entries()
        assert len(entries) == 2
        root_entry = next(e for e in entries if e["parent_id"] is None)
        assert root_entry["candidate_id"] == parent.model_id
