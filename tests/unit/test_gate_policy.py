# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for gate policy schema and CI validation (ARCHITECTURE.md §4.3, PRD §4.3.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from apmode.dsl.lane import Lane
from apmode.governance.policy import (
    Gate1Config,
    Gate2Config,
    Gate25Config,
    GatePolicy,
    RiskGradingConfig,
)
from apmode.governance.validate_policies import validate_policy_file


def _gate1(**overrides: object) -> Gate1Config:
    defaults = dict(
        convergence_required=True,
        cwres_mean_max=0.1,
        outlier_fraction_max=0.05,
        vpc_coverage_target=0.90,
        vpc_coverage_tolerance=0.15,
        seed_stability_n=3,
    )
    defaults.update(overrides)
    return Gate1Config(**defaults)


def _gate2(**overrides: object) -> Gate2Config:
    defaults = dict(
        interpretable_parameterization="required",
        reproducible_estimation="required",
        shrinkage_max=0.30,
        identifiability_required=True,
        node_eligible=False,
        loro_required=False,
    )
    defaults.update(overrides)
    return Gate2Config(**defaults)


class TestGate1Config:
    def test_valid(self) -> None:
        g1 = _gate1()
        assert g1.cwres_mean_max == pytest.approx(0.1)

    def test_vpc_tolerance_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _gate1(vpc_coverage_tolerance=0.0)

    def test_vpc_tolerance_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            _gate1(vpc_coverage_tolerance=1.5)

    def test_vpc_target_within_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            _gate1(vpc_coverage_target=1.5)

    def test_new_prd_fields_default_true(self) -> None:
        g1 = _gate1()
        assert g1.parameter_plausibility_required is True
        assert g1.state_trajectory_validity_required is True
        # ``split_integrity_required`` defaults to False so benchmark /
        # single-fold policies don't need to opt out; policies that
        # enforce held-out GOF agreement set it to True explicitly.
        assert g1.split_integrity_required is False


class TestGate2Config:
    def test_submission(self) -> None:
        g2 = _gate2()
        assert g2.node_eligible is False

    def test_discovery(self) -> None:
        g2 = _gate2(
            interpretable_parameterization="not_required",
            shrinkage_max=None,
            identifiability_required=False,
            node_eligible=True,
        )
        assert g2.node_eligible is True


class TestGate25Config:
    def test_defaults(self) -> None:
        g25 = Gate25Config()
        assert g25.context_of_use_required is True
        assert g25.sensitivity_analysis_required is False


class TestRiskGradingConfig:
    def test_defaults_disabled(self) -> None:
        rg = RiskGradingConfig()
        assert rg.enabled is False
        assert rg.matrix == {}

    def test_monotonic_matrix_accepted(self) -> None:
        rg = RiskGradingConfig(
            enabled=True,
            matrix={
                "low": {"low": "low", "medium": "low", "high": "medium"},
                "medium": {"low": "low", "medium": "medium", "high": "high"},
                "high": {"low": "medium", "medium": "high", "high": "high"},
            },
        )
        assert rg.tier_for("high", "medium") == "high"
        assert rg.tier_for("low", "low") == "low"

    def test_non_monotonic_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="monotonic"):
            RiskGradingConfig(
                enabled=True,
                matrix={
                    "low": {"low": "high", "medium": "low", "high": "low"},
                    "medium": {"low": "low", "medium": "medium", "high": "high"},
                    "high": {"low": "medium", "medium": "high", "high": "high"},
                },
            )

    def test_gate25_config_has_risk_grading_field(self) -> None:
        g25 = Gate25Config()
        assert g25.risk_grading is None
        assert g25.risk_grading_required is False

    def test_unknown_credibility_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown factor"):
            RiskGradingConfig(
                enabled=True,
                credibility_factors={"high": {"bogus_factor": "a"}},
            )


class TestGatePolicy:
    def test_valid_submission(self) -> None:
        policy = GatePolicy(
            policy_version="1.0.0", lane="submission", gate1=_gate1(), gate2=_gate2()
        )
        assert policy.lane == "submission"

    def test_with_gate25(self) -> None:
        policy = GatePolicy(
            policy_version="1.0.0",
            lane="discovery",
            gate1=_gate1(),
            gate2=_gate2(node_eligible=True),
            gate2_5=Gate25Config(ai_ml_transparency_required=True),
        )
        assert policy.gate2_5 is not None
        assert policy.gate2_5.ai_ml_transparency_required is True

    def test_invalid_lane_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GatePolicy(policy_version="1.0.0", lane="invalid", gate1=_gate1(), gate2=_gate2())

    def test_json_roundtrip(self) -> None:
        policy = GatePolicy(
            policy_version="1.0.0", lane="discovery", gate1=_gate1(), gate2=_gate2()
        )
        json_str = policy.model_dump_json()
        roundtripped = GatePolicy.model_validate_json(json_str)
        assert roundtripped.lane == "discovery"


class TestValidatePolicyFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        policy_data = {
            "policy_version": "1.0.0",
            "lane": "submission",
            "gate1": {
                "convergence_required": True,
                "cwres_mean_max": 0.1,
                "outlier_fraction_max": 0.05,
                "vpc_coverage_target": 0.90,
                "vpc_coverage_tolerance": 0.10,
                "seed_stability_n": 3,
            },
            "gate2": {
                "interpretable_parameterization": "required",
                "reproducible_estimation": "required",
                "shrinkage_max": 0.30,
                "identifiability_required": True,
                "node_eligible": False,
                "loro_required": False,
            },
        }
        f = tmp_path / "submission.json"
        f.write_text(json.dumps(policy_data))
        errors = validate_policy_file(f)
        assert errors == []

    def test_invalid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"lane": "invalid"}))
        errors = validate_policy_file(f)
        assert len(errors) > 0

    def test_malformed_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json at all")
        errors = validate_policy_file(f)
        assert len(errors) > 0


class TestLanePoliciesGate3Contract:
    """Each lane policy under ``policies/`` must parse with the intended Gate3.

    These assertions pin the on-disk shape of the gate3 block so an
    accidental edit does not silently revert a lane to the default config
    (which would, e.g., flip discovery back to weighted_sum from Borda).
    """

    _POLICY_DIR = Path(__file__).parent.parent.parent / "policies"

    def _load(self, lane: str) -> GatePolicy:
        data = json.loads((self._POLICY_DIR / f"{lane}.json").read_text())
        return GatePolicy.model_validate(data)

    def test_submission_keeps_default_gate3(self) -> None:
        # Submission has no explicit gate3 block — the default applies
        # (weighted_sum, BIC off, AUC/Cmax off). BIC off is correct cross-
        # paradigm for submission because likelihood scales are
        # incomparable across observation models (PRD §10 Q2).
        policy = self._load("submission")
        assert policy.gate3.composite_method == "weighted_sum"
        assert policy.gate3.vpc_weight == pytest.approx(0.5)
        assert policy.gate3.npe_weight == pytest.approx(0.5)
        assert policy.gate3.bic_weight == 0.0
        assert policy.gate3.auc_cmax_weight == 0.0

    def test_discovery_uses_borda_with_equal_vpc_npe(self) -> None:
        policy = self._load("discovery")
        assert policy.gate3.composite_method == "borda"
        assert policy.gate3.vpc_weight == pytest.approx(0.5)
        assert policy.gate3.npe_weight == pytest.approx(0.5)
        assert policy.gate3.bic_weight == 0.0
        # AUC/Cmax disabled until backend VPC pipeline produces simulation
        # matrices; enabling it forces simulation-based ranking which today
        # would uniform-drop every ranking and just burn cycles.
        assert policy.gate3.auc_cmax_weight == 0.0

    def test_optimization_uses_borda_with_auc_cmax_active(self) -> None:
        policy = self._load("optimization")
        assert policy.gate3.composite_method == "borda"
        # AUC/Cmax BE is active in Optimization now that the nlmixr2
        # backend can emit posterior-predictive simulations. The uniform-
        # drop rule (governance/ranking.py) removes the component when
        # any survivor lacks the score, so mixed-backend survivor sets
        # degrade gracefully to vpc + npe.
        assert policy.gate3.vpc_weight == pytest.approx(0.35)
        assert policy.gate3.npe_weight == pytest.approx(0.35)
        assert policy.gate3.bic_weight == 0.0
        assert policy.gate3.auc_cmax_weight == pytest.approx(0.30)

    def test_all_lanes_policy_version_bumped(self) -> None:
        # 0.4.2 bump: Gate 1 VPC coverage gate replaced with PIT/NPDE-lite
        # predictive calibration. The prior bin-level VPC metric
        # false-rejected textbook-correct models on sparse real datasets
        # (warfarin: 0/33 pass in rc9) because per-band coverage quantized
        # at 1/n_bins — target±tolerance couldn't bridge the discrete
        # gaps. See CHANGELOG rc9 follow-up "PIT/NPDE-lite Gate 1
        # calibration".
        # 0.5.1 bump: Gate 2.5 block added to submission lane. Discovery
        # and optimization versions track in lockstep (sync_readme.py
        # enforces a single gate policy version across all lanes).
        # 0.6.0 bump: Gate 2 prior-data conflict + prior-sensitivity
        # hard-gates added (plan Tasks 20 + 21). Submission requires
        # both; Discovery / Optimization default the new ``*_required``
        # knobs to False.
        # 0.7.0 bump: V&V40-style risk-grading matrix (model_influence x
        # decision_consequence -> tier -> credibility-factor rigor floors)
        # added to gate2_5 across all three lanes. Submission/Optimization
        # enable it; Discovery ships the matrix disabled by default.
        # 0.7.1 bump: agentic_compliance block (AgenticComplianceConfig)
        # added across all three lanes for trajectory-level reward-hacking
        # / rationale-coherence QA on the agentic-LLM backend. Submission
        # ships the same defaults even though the Submission lane excludes
        # the agentic backend, per this test's own lockstep convention.
        for lane in ("submission", "discovery", "optimization"):
            assert self._load(lane).policy_version == "0.7.1"

    def test_agentic_compliance_defaults_present_on_discovery_policy(self) -> None:
        policy = self._load("discovery")
        assert policy.agentic_compliance.reward_hacking_shrinkage_slope_max > 0
        assert 0.0 <= policy.agentic_compliance.rationale_action_coherence_min <= 1.0

    def test_all_lanes_pit_tolerance_calibration(self) -> None:
        """0.4.2: PIT calibration replaces VPC coverage as the Gate 1 gate.

        N-scaled form: ``tol(p, n) = max(floor, z_alpha · sqrt(p(1-p)/n))``.
        Lane ordering:

        * ``z_alpha``: submission 1.5 (strictest), optimization 2.0,
          discovery 2.5 (widest). Higher z_alpha = more permissive.
        * Floors: submission tail/med 0.03/0.05 (tightest), optimization
          0.04/0.07, discovery 0.05/0.10 (widest). Floors bind only at
          large ``n_subjects`` where the z·SE scaling would go vacuous.

        Tail floors are tighter than median because tail miscalibration
        is the most diagnostic signal of residual-error / IIV
        misspecification — same rationale as the z_alpha ordering.
        """
        expected = {
            "submission": (1.5, 0.03, 0.05),
            "optimization": (2.0, 0.04, 0.07),
            "discovery": (2.5, 0.05, 0.10),
        }
        for lane, (z_alpha, floor_tail, floor_med) in expected.items():
            g1 = self._load(lane).gate1
            assert g1.pit_required is True
            assert g1.pit_z_alpha == pytest.approx(z_alpha)
            assert g1.pit_tol_tail_floor == pytest.approx(floor_tail)
            assert g1.pit_tol_median_floor == pytest.approx(floor_med)


class TestNPDEPolicyKnobs:
    """True decorrelated NPDE (Comets & Brendel 2008/2010) Gate 1/3 knobs.

    Additive to PIT/NPDE-lite (``pit_required`` stays the default Gate 1
    calibration gate); ``npde_required`` is opt-in per lane.
    """

    def test_defaults(self) -> None:
        g1 = _gate1()
        assert g1.npde_required is False
        assert g1.npde_bonferroni_alpha == pytest.approx(0.05)
        assert g1.npde_min_k_sims == 300

    def test_npde_min_k_sims_below_gate3_sims_rejected_when_required(self) -> None:
        from apmode.governance.policy import Gate3Config

        with pytest.raises(ValidationError):
            GatePolicy(
                policy_version="0.7.0",
                lane=Lane.SUBMISSION,
                gate1=_gate1(npde_required=True, npde_min_k_sims=500),
                gate2=_gate2(),
                gate3=Gate3Config(n_posterior_predictive_sims=300),
            )

    def test_npde_min_k_sims_at_or_above_gate3_sims_accepted(self) -> None:
        from apmode.governance.policy import Gate3Config

        policy = GatePolicy(
            policy_version="0.7.0",
            lane=Lane.SUBMISSION,
            gate1=_gate1(npde_required=True, npde_min_k_sims=300),
            gate2=_gate2(),
            gate3=Gate3Config(n_posterior_predictive_sims=300),
        )
        assert policy.gate1.npde_required is True

    def test_npde_covariance_ridge_default(self) -> None:
        from apmode.governance.policy import Gate3Config

        assert Gate3Config().npde_covariance_ridge == pytest.approx(1e-8)

    def test_all_lanes_npde_opt_in_default_false(self) -> None:
        """Lane policies do not flip npde_required on in this pass — it
        stays False across submission/discovery/optimization until a
        separate values decision retires PIT-lite as the sole gate."""
        for lane in ("submission", "discovery", "optimization"):
            data = json.loads((Path("policies") / f"{lane}.json").read_text())
            policy = GatePolicy.model_validate(data)
            assert policy.gate1.npde_required is False
