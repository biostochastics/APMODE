# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 1 (Technical Validity) and Gate 2 (Lane Admissibility).

Verifies the governance funnel: gates are sequential disqualifiers,
failures logged with per-check reasons, thresholds from policy files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    PITCalibrationSummary,
    ScoringContract,
    SplitGOFMetrics,
    VPCSummary,
)
from apmode.governance.gates import (
    evaluate_gate1,
    evaluate_gate2,
    evaluate_gate2_5,
    evaluate_gate3,
)
from apmode.governance.policy import GatePolicy

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


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


def _make_backend_result(
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


def _load_policy(lane: str) -> GatePolicy:
    """Load a policy from the policies directory."""
    path = POLICY_DIR / f"{lane}.json"
    return GatePolicy.model_validate(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Gate 1 Tests
# ---------------------------------------------------------------------------


class TestGate1:
    """Gate 1: Technical Validity."""

    def test_all_passing(self) -> None:
        result = _make_backend_result()
        seed_r2 = _make_backend_result(ofv=150.5)
        seed_r3 = _make_backend_result(ofv=149.5)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy, seed_results=[seed_r2, seed_r3])
        assert g1.passed is True
        assert g1.gate_name == "technical_validity"
        assert all(c.passed for c in g1.checks)

    def test_convergence_failure(self) -> None:
        result = _make_backend_result(converged=False)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        assert g1.passed is False
        failed = [c for c in g1.checks if not c.passed]
        assert any(c.check_id == "convergence" for c in failed)

    def test_cwres_mean_too_high(self) -> None:
        result = _make_backend_result(cwres_mean=0.5)  # submission threshold is 0.1
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        assert g1.passed is False
        failed_ids = {c.check_id for c in g1.checks if not c.passed}
        assert "cwres_mean" in failed_ids

    def test_outlier_fraction_too_high(self) -> None:
        result = _make_backend_result(outlier_fraction=0.15)  # threshold 0.05
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        assert g1.passed is False
        failed_ids = {c.check_id for c in g1.checks if not c.passed}
        assert "cwres_outlier_fraction" in failed_ids

    def test_pit_calibration_tail_miscalibrated(self) -> None:
        """Tail miscalibration > submission tol_tail=0.03 should fail Gate 1.

        c_0.05=0.20 means the model's 5th-percentile predictive quantile
        is too high (the observed value falls at or below it 20% of the
        time rather than the expected 5%) — a classic tail-heavy residual
        misspecification symptom. |0.20 - 0.05| = 0.15 > 0.03.
        """
        result = _make_backend_result(pit_calibration={"p5": 0.20, "p50": 0.50, "p95": 0.95})
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        assert g1.passed is False
        failed_ids = {c.check_id for c in g1.checks if not c.passed}
        assert "pit_calibration" in failed_ids

    def test_seed_stability_with_consistent_seeds(self) -> None:
        result1 = _make_backend_result(ofv=150.0)
        result2 = _make_backend_result(ofv=150.5)
        result3 = _make_backend_result(ofv=149.5)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result1, policy, seed_results=[result2, result3])
        seed_check = next(c for c in g1.checks if c.check_id == "seed_stability")
        assert seed_check.passed is True

    def test_seed_stability_with_inconsistent_seeds(self) -> None:
        result1 = _make_backend_result(ofv=150.0)
        result2 = _make_backend_result(ofv=300.0)
        result3 = _make_backend_result(ofv=50.0)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result1, policy, seed_results=[result2, result3])
        seed_check = next(c for c in g1.checks if c.check_id == "seed_stability")
        assert seed_check.passed is False

    def test_seed_stability_not_probed_passes(self) -> None:
        """Missing seed results → "not probed" pass.

        The orchestrator only runs seed replicates for the top-K
        candidates by BIC; absence of evidence for the rest is an
        orchestrator choice, not a candidate defect. Seed stability is a
        positive confirmation when probed, not a disqualifier when
        skipped.
        """
        result = _make_backend_result()
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy, seed_results=None)
        seed_check = next(c for c in g1.checks if c.check_id == "seed_stability")
        assert seed_check.passed is True
        assert "not_probed" in str(seed_check.observed)

    def test_pit_missing_fails_when_required(self) -> None:
        """Missing PIT calibration should fail when policy requires it."""
        from apmode.bundle.models import BackendResult as BR

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["pit_calibration"] = None
        result_no_pit = BR.model_validate(data)
        policy = _load_policy("submission")
        policy.gate1.pit_required = True
        g1 = evaluate_gate1(result_no_pit, policy)
        pit_check = next(c for c in g1.checks if c.check_id == "pit_calibration")
        assert pit_check.passed is False
        assert pit_check.observed == "pit_not_available"

    def test_pit_missing_passes_when_not_required(self) -> None:
        """When pit_required=False, missing PIT passes with explicit marker."""
        from apmode.bundle.models import BackendResult as BR

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["pit_calibration"] = None
        result_no_pit = BR.model_validate(data)
        policy = _load_policy("submission")
        policy.gate1.pit_required = False
        g1 = evaluate_gate1(result_no_pit, policy)
        pit_check = next(c for c in g1.checks if c.check_id == "pit_calibration")
        assert pit_check.passed is True
        assert pit_check.observed == "pit_not_configured"

    def test_discovery_policy_more_lenient(self) -> None:
        # Discovery allows higher CWRES mean (0.15 vs 0.10)
        result = _make_backend_result(cwres_mean=0.12)
        seeds = [_make_backend_result(ofv=150.5), _make_backend_result(ofv=149.5)]
        sub_policy = _load_policy("submission")
        disc_policy = _load_policy("discovery")
        g1_sub = evaluate_gate1(result, sub_policy, seed_results=seeds)
        g1_disc = evaluate_gate1(result, disc_policy, seed_results=seeds)
        assert g1_sub.passed is False  # fails submission
        assert g1_disc.passed is True  # passes discovery

    def test_gate_result_has_all_checks(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        check_ids = {c.check_id for c in g1.checks}
        expected = {
            "convergence",
            "parameter_plausibility",
            "state_trajectory_validity",
            "cwres_mean",
            "cwres_outlier_fraction",
            "pit_calibration",
            "split_integrity",
            "seed_stability",
            "imputation_stability",
        }
        assert expected == check_ids

    def test_parameter_plausibility_negative_volume(self) -> None:
        """Negative structural parameter (e.g. V < 0) should fail plausibility."""
        result = _make_backend_result()
        result.parameter_estimates["V"] = ParameterEstimate(
            name="V", estimate=-10.0, se=1.0, rse=10.0, category="structural"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is False
        assert "non-positive" in str(plaus.observed)

    def test_parameter_plausibility_zero_clearance(self) -> None:
        """Zero CL is pharmacologically implausible — should fail."""
        result = _make_backend_result()
        result.parameter_estimates["CL"] = ParameterEstimate(
            name="CL", estimate=0.0, se=0.5, rse=10.0, category="structural"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is False
        assert "non-positive" in str(plaus.observed)

    def test_parameter_plausibility_extreme_rse(self) -> None:
        """RSE > 200% means effectively unidentifiable — should fail."""
        result = _make_backend_result()
        result.parameter_estimates["CL"] = ParameterEstimate(
            name="CL", estimate=5.0, se=15.0, rse=300.0, category="structural"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is False
        assert "RSE" in str(plaus.observed)

    def test_parameter_plausibility_at_lower_bound(self) -> None:
        """Estimate at lower sanity bound (1e-4) should fail."""
        result = _make_backend_result()
        result.parameter_estimates["ka"] = ParameterEstimate(
            name="ka", estimate=1e-5, se=0.001, rse=10.0, category="structural"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is False
        assert "lower bound" in str(plaus.observed)

    def test_parameter_plausibility_at_upper_bound(self) -> None:
        """Estimate at upper sanity bound (1e5) should fail."""
        result = _make_backend_result()
        result.parameter_estimates["V"] = ParameterEstimate(
            name="V", estimate=200000.0, se=1000.0, rse=0.5, category="structural"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is False
        assert "upper bound" in str(plaus.observed)

    def test_parameter_plausibility_iiv_not_checked(self) -> None:
        """Non-structural parameters (IIV) should not trigger plausibility failure."""
        result = _make_backend_result()
        result.parameter_estimates["eta_CL"] = ParameterEstimate(
            name="eta_CL", estimate=-0.5, se=0.1, rse=20.0, category="iiv"
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        plaus = next(c for c in g1.checks if c.check_id == "parameter_plausibility")
        assert plaus.passed is True

    def test_state_trajectory_negative_r2(self) -> None:
        """R² < 0 (pathological fit) should fail state trajectory check."""
        result = _make_backend_result(r2=-0.5)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False

    def test_state_trajectory_missing_r2_passes(self) -> None:
        """When R² is not available but other signals OK, passes."""
        result = _make_backend_result(r2=None)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is True

    def test_state_trajectory_r2_below_threshold(self) -> None:
        """R² below obs_vs_pred_r2_min (0.30) should fail."""
        result = _make_backend_result(r2=0.1)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False
        assert "R²" in str(traj.observed)

    def test_state_trajectory_cwres_sd_too_high(self) -> None:
        """CWRES SD > cwres_sd_max (2.0) indicates misspecification."""
        result = _make_backend_result(cwres_sd=3.0)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False
        assert "cwres_sd" in str(traj.observed)

    def test_state_trajectory_cwres_sd_too_low(self) -> None:
        """CWRES SD < cwres_sd_min (0.50) indicates collapsed residuals."""
        result = _make_backend_result(cwres_sd=0.1)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False
        assert "cwres_sd" in str(traj.observed)

    def test_split_integrity_no_diagnostics_fails_when_required(self) -> None:
        """When split_gof is None and split_integrity_required=True,
        the check must fail — missing required evidence must never
        silently pass (disqualifying-funnel invariant).
        """
        result = _make_backend_result()
        # Clear the fixture default to exercise the missing-evidence path.
        result.diagnostics.split_gof = None
        policy = _load_policy("submission")
        # Default is False; opt in to exercise the missing-evidence path.
        policy.gate1 = policy.gate1.model_copy(update={"split_integrity_required": True})
        g1 = evaluate_gate1(result, policy)
        si = next(c for c in g1.checks if c.check_id == "split_integrity")
        assert si.passed is False
        assert "no_split_diagnostics" in str(si.observed)

    def test_split_integrity_consistent_passes(self) -> None:
        """When train/test metrics are similar, passes."""
        from apmode.bundle.models import SplitGOFMetrics

        result = _make_backend_result()
        result.diagnostics.split_gof = SplitGOFMetrics(
            train_cwres_mean=0.02,
            train_outlier_fraction=0.02,
            test_cwres_mean=0.05,
            test_outlier_fraction=0.03,
            n_train=40,
            n_test=10,
        )
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        si = next(c for c in g1.checks if c.check_id == "split_integrity")
        assert si.passed is True

    def test_split_integrity_overfitting_fails(self) -> None:
        """When test CWRES drifts far from train, fails (overfitting)."""
        from apmode.bundle.models import SplitGOFMetrics

        result = _make_backend_result()
        result.diagnostics.split_gof = SplitGOFMetrics(
            train_cwres_mean=0.01,
            train_outlier_fraction=0.02,
            test_cwres_mean=0.8,  # big drift
            test_outlier_fraction=0.15,  # much worse
            n_train=40,
            n_test=10,
        )
        policy = _load_policy("submission")
        # Default is False; opt in to exercise the overfitting-detection path.
        policy.gate1 = policy.gate1.model_copy(update={"split_integrity_required": True})
        g1 = evaluate_gate1(result, policy)
        si = next(c for c in g1.checks if c.check_id == "split_integrity")
        assert si.passed is False


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
        """Candidates with None BIC should be ranked last."""
        r1 = _make_backend_result(bic=170.0)
        r2 = _make_backend_result(bic=None)  # type: ignore[arg-type]
        policy = _load_policy("submission")
        _g3, ranked = evaluate_gate3([r1, r2], policy)
        assert ranked[0].bic == 170.0
        assert ranked[1].bic == float("inf")


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


# ---------------------------------------------------------------------------
# Gate 2.5 Tests (Credibility Qualification — ICH M15)
# ---------------------------------------------------------------------------


def _make_policy_with_gate25(
    lane: str = "submission",
    *,
    context_required: bool = True,
    limitation_required: bool = False,
    data_adequacy_required: bool = True,
    data_adequacy_ratio_min: float = 5.0,
    sensitivity_required: bool = False,
    ml_transparency_required: bool = False,
) -> GatePolicy:
    """Build a policy with Gate 2.5 config for testing."""
    base = json.loads(POLICY_DIR.joinpath(f"{lane}.json").read_text())
    base["gate2_5"] = {
        "context_of_use_required": context_required,
        "limitation_to_risk_mapping_required": limitation_required,
        "data_adequacy_required": data_adequacy_required,
        "data_adequacy_ratio_min": data_adequacy_ratio_min,
        "sensitivity_analysis_required": sensitivity_required,
        "ai_ml_transparency_required": ml_transparency_required,
    }
    return GatePolicy.model_validate(base)


class TestGate25:
    """Gate 2.5: Credibility Qualification (ICH M15)."""

    def test_no_gate25_config_passes(self) -> None:
        """When policy has no gate2_5, all candidates pass."""
        result = _make_backend_result()
        # submission.json now ships with a gate2_5 block (policy_version 0.5.1);
        # build an explicit no-gate2_5 policy to exercise the ``g25 is None``
        # branch of ``evaluate_gate2_5``.
        base = json.loads(POLICY_DIR.joinpath("submission.json").read_text())
        base.pop("gate2_5", None)
        policy = GatePolicy.model_validate(base)
        g25 = evaluate_gate2_5(result, policy)
        assert g25.passed is True
        assert g25.gate_name == "credibility_qualification"

    def test_passes_with_adequate_context(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25()
        ctx = CredibilityContext(
            context_of_use="Dose adjustment for renal impairment",
            risk_level="medium",
            n_observations=100,
            n_parameters=8,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is True

    def test_fails_missing_context_of_use(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(context_required=True)
        ctx = CredibilityContext(n_observations=100, n_parameters=8)  # no COU
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "context_of_use" in failed_ids

    def test_fails_insufficient_data_adequacy(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(data_adequacy_ratio_min=10.0)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=20,
            n_parameters=8,  # ratio = 2.5 < 10.0
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "data_adequacy" in failed_ids

    def test_data_adequacy_passes_when_ratio_sufficient(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(data_adequacy_ratio_min=5.0)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,  # ratio = 12.5 >= 5.0
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        da = next(c for c in g25.checks if c.check_id == "data_adequacy")
        assert da.passed is True

    def test_node_requires_ml_transparency(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="jax_node")
        policy = _make_policy_with_gate25(
            lane="discovery",
            ml_transparency_required=True,
        )
        ctx = CredibilityContext(
            context_of_use="Discovery analysis",
            n_observations=200,
            n_parameters=10,
            # No ml_transparency_statement
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "ml_transparency" in failed_ids

    def test_node_with_transparency_passes(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="jax_node")
        policy = _make_policy_with_gate25(
            lane="discovery",
            ml_transparency_required=True,
        )
        ctx = CredibilityContext(
            context_of_use="Discovery analysis",
            n_observations=200,
            n_parameters=10,
            ml_transparency_statement=(
                "NODE used for elimination; bounded_positive constraint; 3-dim"
            ),
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        ml = next(c for c in g25.checks if c.check_id == "ml_transparency")
        assert ml.passed is True

    def test_classical_skips_ml_transparency(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="nlmixr2")
        policy = _make_policy_with_gate25(ml_transparency_required=True)
        ctx = CredibilityContext(
            context_of_use="Submission analysis",
            n_observations=200,
            n_parameters=8,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        ml = next(c for c in g25.checks if c.check_id == "ml_transparency")
        assert ml.passed is True  # not applicable for classical

    def test_sensitivity_required_but_missing(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(sensitivity_required=True)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,
            sensitivity_available=False,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "sensitivity_analysis" in failed_ids

    def test_limitation_to_risk_required_but_missing(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(limitation_required=True)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,
            # No limitations or risk_level
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "limitation_to_risk" in failed_ids

    def test_all_checks_present(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25()
        ctx = CredibilityContext(context_of_use="Test", n_observations=100, n_parameters=8)
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        check_ids = {c.check_id for c in g25.checks}
        expected = {
            "context_of_use",
            "limitation_to_risk",
            "data_adequacy",
            "sensitivity_analysis",
            "ml_transparency",
        }
        assert expected == check_ids
