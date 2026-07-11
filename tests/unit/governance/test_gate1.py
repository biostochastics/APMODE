# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 1 (Technical Validity).

Verifies the governance funnel: gates are sequential disqualifiers,
failures logged with per-check reasons, thresholds from policy files.
"""

from __future__ import annotations

from apmode.bundle.models import (
    ParameterEstimate,
    SplitGOFMetrics,
)
from apmode.governance.gates import (
    evaluate_gate1,
)
from tests._helpers.builders import make_backend_result as _make_backend_result
from tests._helpers.policies import load_policy as _load_policy

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

    def test_npde_not_required_by_default_passes_when_absent(self) -> None:
        result = _make_backend_result()  # no npde on this fixture
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is True
        assert npde_check.observed == "npde_not_configured"

    def test_npde_required_fails_when_absent(self) -> None:
        result = _make_backend_result()
        policy = _load_policy("submission")
        policy.gate1.npde_required = True
        g1 = evaluate_gate1(result, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is False
        assert npde_check.observed == "npde_not_available"

    def test_npde_populated_but_not_required_passes_without_gating(self) -> None:
        """Backend emits NPDE but the lane hasn't opted in yet — the
        evidence is surfaced (audit trail) but does not fail the
        candidate, preserving the "additive, opt-in" contract even
        after Task 6 wires npde onto every posterior-predictive-capable
        backend result."""
        from apmode.bundle.models import BackendResult as BR
        from apmode.bundle.models import NPDESummary

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["npde"] = NPDESummary(
            n_subjects=30,
            n_observations=120,
            npde_mean=1.2,
            npde_variance=2.5,
            wilcoxon_p=0.001,
            shapiro_p=0.02,
            fisher_variance_p=0.001,
            bonferroni_p=0.003,  # would fail if gated
        ).model_dump()
        result_bad_npde = BR.model_validate(data)
        policy = _load_policy("submission")
        assert policy.gate1.npde_required is False
        g1 = evaluate_gate1(result_bad_npde, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is True

    def test_npde_required_fails_below_bonferroni_alpha(self) -> None:
        from apmode.bundle.models import BackendResult as BR
        from apmode.bundle.models import NPDESummary

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["npde"] = NPDESummary(
            n_subjects=30,
            n_observations=120,
            npde_mean=1.2,
            npde_variance=2.5,
            wilcoxon_p=0.001,
            shapiro_p=0.02,
            fisher_variance_p=0.001,
            bonferroni_p=0.003,
        ).model_dump()
        result_bad_npde = BR.model_validate(data)
        policy = _load_policy("submission")
        policy.gate1.npde_required = True
        g1 = evaluate_gate1(result_bad_npde, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is False

    def test_npde_required_passes_when_well_calibrated(self) -> None:
        from apmode.bundle.models import BackendResult as BR
        from apmode.bundle.models import NPDESummary

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["npde"] = NPDESummary(
            n_subjects=30,
            n_observations=120,
            npde_mean=0.0,
            npde_variance=1.0,
            wilcoxon_p=0.9,
            shapiro_p=0.9,
            fisher_variance_p=0.9,
            bonferroni_p=1.0,
        ).model_dump()
        result_good_npde = BR.model_validate(data)
        policy = _load_policy("submission")
        policy.gate1.npde_required = True
        g1 = evaluate_gate1(result_good_npde, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is True

    def test_npde_required_fails_on_degenerate_zero_subject_sentinel(self) -> None:
        from apmode.bundle.models import BackendResult as BR
        from apmode.bundle.models import NPDESummary

        result = _make_backend_result()
        data = result.model_dump()
        data["diagnostics"]["npde"] = NPDESummary(
            n_subjects=0,
            n_observations=0,
            npde_mean=0.0,
            npde_variance=0.0,
            wilcoxon_p=1.0,
            shapiro_p=1.0,
            fisher_variance_p=1.0,
            bonferroni_p=1.0,
        ).model_dump()
        result_degenerate = BR.model_validate(data)
        policy = _load_policy("submission")
        policy.gate1.npde_required = True
        g1 = evaluate_gate1(result_degenerate, policy)
        npde_check = next(c for c in g1.checks if c.check_id == "npde_calibration")
        assert npde_check.passed is False
        assert npde_check.observed == "npde_degenerate_no_finite_sims"

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

    def test_seed_stability_not_probed_fails_required_evidence(self) -> None:
        """A three-seed policy cannot pass with only the primary fit."""
        result = _make_backend_result()
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy, seed_results=None)
        seed_check = next(c for c in g1.checks if c.check_id == "seed_stability")
        assert seed_check.passed is False
        assert "insufficient_seeds" in str(seed_check.observed)

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
            "npde_calibration",
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

    def test_state_trajectory_missing_direct_evidence_fails(self) -> None:
        result = _make_backend_result()
        result.diagnostics.state_trajectory_valid = None
        g1 = evaluate_gate1(result, _load_policy("submission"), seed_results=[])
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False
        assert "evidence=unavailable" in str(traj.observed)

    def test_state_trajectory_negative_or_nonfinite_evidence_fails(self) -> None:
        result = _make_backend_result()
        result.diagnostics.state_trajectory_valid = False
        g1 = evaluate_gate1(result, _load_policy("submission"), seed_results=[])
        traj = next(c for c in g1.checks if c.check_id == "state_trajectory_validity")
        assert traj.passed is False

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
