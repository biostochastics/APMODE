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
        vpc_coverage_lower=0.85,
        vpc_coverage_upper=0.995,
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

    def test_vpc_bounds_inverted_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _gate1(vpc_coverage_lower=0.99, vpc_coverage_upper=0.85)

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
                "vpc_coverage_lower": 0.85,
                "vpc_coverage_upper": 0.995,
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
