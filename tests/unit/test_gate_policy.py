# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for gate policy schema and CI validation (ARCHITECTURE.md §4.3, PRD §4.3.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from apmode.governance.policy import Gate1Config, Gate2Config, Gate25Config, GatePolicy
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
        assert g1.split_integrity_required is True


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

    def test_optimization_policy_version_bumped(self) -> None:
        # 0.4.1 bump: Gate 1 ``vpc_coverage_upper`` relaxed 0.995 → 1.0
        # across all three lanes because post-hoc binned VPC produces
        # discrete hit-rates (n/k); with 7-bin VPCs the common
        # "well-fitted" value of 7/7=1.0 previously rejected on the upper
        # ceiling — no discrete value exists in (6/7=0.857, 0.995).
        # See CHANGELOG rc9 follow-up.
        policy = self._load("optimization")
        assert policy.policy_version == "0.4.1"

    def test_submission_discovery_policy_version_bumped(self) -> None:
        for lane in ("submission", "discovery"):
            assert self._load(lane).policy_version == "0.4.1"

    def test_all_lanes_vpc_target_tolerance(self) -> None:
        """0.4.1: VPC coverage check is target-with-tolerance, not bounds.

        Discrete small-bin VPCs produce hit-rates in ``{0, 1/k, …, 1.0}``,
        so a fixed ``[lower, upper]`` range creates unreachable gaps.
        The new check is ``|coverage - target| <= tolerance`` with lane-
        specific tolerances: submission 0.10 (strict), optimization 0.12,
        discovery 0.15 (widest, reflecting NODE/agentic variance).
        """
        expected_tolerance = {"submission": 0.10, "discovery": 0.15, "optimization": 0.12}
        for lane, tol in expected_tolerance.items():
            g1 = self._load(lane).gate1
            assert g1.vpc_coverage_target == pytest.approx(0.90)
            assert g1.vpc_coverage_tolerance == pytest.approx(tol)
