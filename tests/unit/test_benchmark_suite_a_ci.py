# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A CI assertions (PRD §5, §8, ARCHITECTURE.md §6).

Validates structure recovery and parameter bias for all 4 scenarios using
mock BackendResults that simulate successful nlmixr2 fits. These tests run
in CI without R installed.

Assertions:
  - Structure recovery: correct absorption, distribution, elimination types
  - Parameter bias: mock estimates within 20% of ground truth (REFERENCE_PARAMS)
  - Gate passage: correctly-fit models pass Gate 1 + Gate 2 (submission)
  - Candidate lineage: search space generates candidates matching each scenario
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    scenario_a1,
    scenario_a2,
    scenario_a3,
    scenario_a4,
)
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)
from apmode.governance.gates import evaluate_gate1, evaluate_gate2
from apmode.governance.policy import GatePolicy

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _load_policy(lane: str) -> GatePolicy:
    path = POLICY_DIR / f"{lane}.json"
    return GatePolicy.model_validate(json.loads(path.read_text()))


def _make_mock_fit(
    spec: DSLSpec,
    reference: dict[str, float],
    bias_fraction: float = 0.02,
) -> BackendResult:
    """Build a mock BackendResult simulating a successful fit.

    Applies a small bias to reference params to simulate estimation noise.
    """
    param_estimates = {}
    for name, value in reference.items():
        estimated = value * (1 + bias_fraction)
        param_estimates[name] = ParameterEstimate(
            name=name,
            estimate=estimated,
            se=value * 0.1,
            rse=10.0,
            ci95_lower=value * 0.8,
            ci95_upper=value * 1.2,
            fixed=False,
            category="structural",
        )

    return BackendResult(
        model_id=spec.model_id,
        backend="nlmixr2",
        converged=True,
        ofv=500.0,
        aic=520.0,
        bic=540.0,
        parameter_estimates=param_estimates,
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
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
                coverage={"p5": 0.92, "p50": 0.96, "p95": 0.93},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={n: True for n in reference},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=60.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


# ---------------------------------------------------------------------------
# Structure Recovery
# ---------------------------------------------------------------------------


class TestStructureRecovery:
    """Validate that DSLSpec scenarios encode the expected PK structure."""

    def test_a1_structure(self) -> None:
        """A1: 1-cmt oral, first-order absorption, linear elimination."""
        spec = scenario_a1()
        assert spec.absorption.type == "FirstOrder"
        assert spec.distribution.type == "OneCmt"
        assert spec.elimination.type == "Linear"

    def test_a2_structure(self) -> None:
        """A2: 2-cmt IV, parallel linear + MM elimination."""
        spec = scenario_a2()
        assert spec.absorption.type == "FirstOrder"  # large ka approximates IV
        assert spec.distribution.type == "TwoCmt"
        assert spec.elimination.type == "ParallelLinearMM"

    def test_a3_structure(self) -> None:
        """A3: Transit absorption (n=3), 1-cmt, linear elimination."""
        spec = scenario_a3()
        assert spec.absorption.type == "Transit"
        assert spec.absorption.n == 3  # type: ignore[union-attr]
        assert spec.distribution.type == "OneCmt"
        assert spec.elimination.type == "Linear"

    def test_a4_structure(self) -> None:
        """A4: 1-cmt oral, Michaelis-Menten elimination."""
        spec = scenario_a4()
        assert spec.absorption.type == "FirstOrder"
        assert spec.distribution.type == "OneCmt"
        assert spec.elimination.type == "MichaelisMenten"


# ---------------------------------------------------------------------------
# Parameter Bias — within 20% of ground truth
# ---------------------------------------------------------------------------


class TestParameterBias:
    """Mock estimates within 20% of REFERENCE_PARAMS pass bias check."""

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_bias_within_20pct(self, name: str, factory: object) -> None:
        """Estimates with 2% bias are within 20% tolerance."""
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref, bias_fraction=0.02)

        for param_name, ref_value in ref.items():
            est = result.parameter_estimates[param_name].estimate
            rel_error = abs(est - ref_value) / ref_value
            assert rel_error < 0.20, (
                f"{name}.{param_name}: estimate={est:.4g}, "
                f"ref={ref_value:.4g}, relative error={rel_error:.2%}"
            )

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_bias_at_boundary_fails(self, name: str, factory: object) -> None:
        """25% bias exceeds the 20% tolerance on at least one parameter."""
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref, bias_fraction=0.25)

        any_over = False
        for param_name, ref_value in ref.items():
            est = result.parameter_estimates[param_name].estimate
            rel_error = abs(est - ref_value) / ref_value
            if rel_error >= 0.20:
                any_over = True
        assert any_over, f"{name}: 25% bias should exceed 20% tolerance"


# ---------------------------------------------------------------------------
# Gate Passage — correctly-fit models pass governance
# ---------------------------------------------------------------------------


class TestGatePassage:
    """Good fits of all 4 scenarios pass Gate 1 + Gate 2 (submission)."""

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_good_fit_passes_gate1(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref, bias_fraction=0.02)
        policy = _load_policy("submission")
        # Provide seed results for stability check
        seed1 = _make_mock_fit(spec, ref, bias_fraction=0.025)
        seed2 = _make_mock_fit(spec, ref, bias_fraction=0.015)
        g1 = evaluate_gate1(result, policy, seed_results=[seed1, seed2])
        assert g1.passed is True, f"Scenario {name} failed Gate 1: {g1.summary_reason}"

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_good_fit_passes_gate2_submission(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref, bias_fraction=0.02)
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed is True, f"Scenario {name} failed Gate 2: {g2.summary_reason}"


# ---------------------------------------------------------------------------
# Search Space — scenarios produce valid search candidates
# ---------------------------------------------------------------------------


class TestSearchSpaceCoverage:
    """Evidence manifests generate search spaces covering each scenario."""

    def test_a1_in_search_space(self) -> None:
        """Linear elimination should be in default search space."""
        from apmode.search.candidates import SearchSpace, generate_root_candidates

        space = SearchSpace(
            structural_cmt=[1],
            absorption_types=["first_order"],
            elimination_types=["linear"],
            error_types=["proportional"],
        )
        candidates = generate_root_candidates(space)
        assert len(candidates) >= 1
        assert any(c.elimination.type == "Linear" for c in candidates)

    def test_a4_mm_in_search_space(self) -> None:
        """Nonlinear clearance signature should include MM candidates."""
        from apmode.bundle.models import EvidenceManifest
        from apmode.search.candidates import SearchSpace

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_signature=True,
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.05,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        space = SearchSpace.from_manifest(manifest)
        assert "michaelis_menten" in space.elimination_types
