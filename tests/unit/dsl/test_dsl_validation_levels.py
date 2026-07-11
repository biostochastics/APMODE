# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the seven-level Formular DSL validator API (P1.8).

Covers :func:`apmode.dsl.validation_levels.validate` — level selection
(single / subset / ``ALL``), the new data-bound / backend-bound /
policy-bound checks, and :class:`~apmode.dsl.validation_levels.ValidationReport`'s
``all_errors`` flattening and ``ok`` predicate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apmode.dsl.ast_models import (
    IIV,
    CovariateLink,
    DSLSpec,
    ExperimentalFlags,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    ObservationEndpoint,
    OneCmt,
    Proportional,
    Transit,
)
from apmode.dsl.errors import FrmCode
from apmode.dsl.lane import Lane
from apmode.dsl.validation_levels import ValidationLevel, resolve_levels, validate
from apmode.governance.policy import Gate1Config, Gate2Config, GatePolicy


def _make_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": "test_id_000000000000",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
        "initial": {"ka": 1.0, "V": 70.0, "CL": 5.0},
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


def _gate1(**overrides: object) -> Gate1Config:
    defaults: dict[str, object] = dict(
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
    defaults: dict[str, object] = dict(
        interpretable_parameterization="required",
        reproducible_estimation="required",
        shrinkage_max=0.30,
        identifiability_required=True,
        node_eligible=False,
        loro_required=False,
    )
    defaults.update(overrides)
    return Gate2Config(**defaults)


def _policy(lane: Lane, *, node_eligible: bool = False) -> GatePolicy:
    return GatePolicy(
        policy_version="1.0.0",
        lane=lane,
        gate1=_gate1(),
        gate2=_gate2(node_eligible=node_eligible),
    )


class TestResolveLevels:
    def test_single_level(self) -> None:
        assert resolve_levels(ValidationLevel.SEMANTIC) == frozenset({ValidationLevel.SEMANTIC})

    def test_iterable_of_levels(self) -> None:
        levels = resolve_levels([ValidationLevel.SEMANTIC, ValidationLevel.AST])
        assert levels == frozenset({ValidationLevel.SEMANTIC, ValidationLevel.AST})

    def test_all_expands_to_every_checkable_level(self) -> None:
        levels = resolve_levels(ValidationLevel.ALL)
        assert levels == frozenset(
            {
                ValidationLevel.SYNTAX,
                ValidationLevel.AST,
                ValidationLevel.SEMANTIC,
                ValidationLevel.DATA_BOUND,
                ValidationLevel.LANE_BOUND,
                ValidationLevel.BACKEND_BOUND,
                ValidationLevel.POLICY_BOUND,
            }
        )

    def test_all_inside_iterable_still_expands(self) -> None:
        levels = resolve_levels([ValidationLevel.SEMANTIC, ValidationLevel.ALL])
        assert ValidationLevel.SYNTAX in levels
        assert ValidationLevel.POLICY_BOUND in levels


class TestSingleLevelSelection:
    """Selecting one level returns only that level's errors."""

    def test_semantic_only_excludes_ast_violation(self) -> None:
        # ka<=0 -> SEM_POSITIVE; Transit(n=0) -> SEM_POSITIVE_INT (also
        # semantic); to get a genuine cross-taxon spec we pair a semantic
        # violation with an AST violation (duplicate IIV param) in one spec.
        spec = _make_spec(
            absorption=Transit(n=0),
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["CL"], structure="diagonal"),
            ],
            initial={"ktr": 1.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        report = validate(spec, level=ValidationLevel.SEMANTIC, lane=Lane.DISCOVERY)

        assert report.levels_run == frozenset({ValidationLevel.SEMANTIC})
        assert set(report.by_level.keys()) == {ValidationLevel.SEMANTIC}
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.SEM_POSITIVE_INT.value in codes
        assert FrmCode.AST_IIV_NO_DUPLICATE_PARAMS.value not in codes

    def test_ast_only_excludes_semantic_violation(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=0),
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["CL"], structure="diagonal"),
            ],
            initial={"ktr": 1.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        report = validate(spec, level=ValidationLevel.AST, lane=Lane.DISCOVERY)

        assert report.levels_run == frozenset({ValidationLevel.AST})
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.AST_IIV_NO_DUPLICATE_PARAMS.value in codes
        assert FrmCode.SEM_POSITIVE_INT.value not in codes

    def test_bound_level_without_required_input_is_skipped(self) -> None:
        """Selecting DATA_BOUND without ``data`` is not reported as clean."""
        spec = _make_spec()
        report = validate(spec, level=ValidationLevel.DATA_BOUND, lane=Lane.DISCOVERY)
        assert report.levels_run == frozenset({ValidationLevel.DATA_BOUND})
        assert report.by_level[ValidationLevel.DATA_BOUND] == []
        assert report.skipped_levels == {
            ValidationLevel.DATA_BOUND: "data_bound selected but no data was supplied"
        }
        assert report.ok is False


class TestAllLevelSelection:
    def test_all_returns_every_level_as_a_key(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive"),
            experimental=ExperimentalFlags(node=True),
        )
        df = pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": [1.0]})
        report = validate(
            spec,
            level=ValidationLevel.ALL,
            lane=Lane.SUBMISSION,
            data=df,
            backend="stan",
            policy=_policy(Lane.SUBMISSION, node_eligible=False),
        )
        assert report.levels_run == frozenset(
            {
                ValidationLevel.SYNTAX,
                ValidationLevel.AST,
                ValidationLevel.SEMANTIC,
                ValidationLevel.DATA_BOUND,
                ValidationLevel.LANE_BOUND,
                ValidationLevel.BACKEND_BOUND,
                ValidationLevel.POLICY_BOUND,
            }
        )
        assert set(report.by_level.keys()) == report.levels_run

        # NODE absorption in Submission lane -> lane-bound error.
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.LANE_NODE_ADMISSIBILITY.value in codes
        # stan has no NODE code path -> backend-bound error.
        assert FrmCode.BE_CAPABILITY_UNSUPPORTED.value in codes
        # policy lane matches (Submission) but node_eligible=False + NODE spec.
        assert FrmCode.POLICY_NODE_INELIGIBLE.value in codes
        assert report.ok is False


class TestDataBound:
    def test_missing_dvid_column(self) -> None:
        spec = _make_spec(
            observations={
                "parent": ObservationEndpoint(
                    name="parent",
                    dvid=1,
                    prediction="C_central",
                    error=Proportional(sigma_prop=0.1),
                ),
                "metabolite": ObservationEndpoint(
                    name="metabolite",
                    dvid=2,
                    prediction="C_central",
                    error=Proportional(sigma_prop=0.1),
                ),
            }
        )
        df = pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": [1.0]})
        report = validate(spec, level=ValidationLevel.DATA_BOUND, lane=Lane.DISCOVERY, data=df)
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.DATA_REQUIRED_COLUMN_MISSING.value in codes
        assert report.ok is False

    def test_present_dvid_column_is_clean(self) -> None:
        spec = _make_spec(
            observations={
                "parent": ObservationEndpoint(
                    name="parent",
                    dvid=1,
                    prediction="C_central",
                    error=Proportional(sigma_prop=0.1),
                ),
            }
        )
        df = pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": [1.0], "DVID": [1]})
        report = validate(spec, level=ValidationLevel.DATA_BOUND, lane=Lane.DISCOVERY, data=df)
        assert report.all_errors == []
        assert report.ok is True

    def test_missing_covariate_column(self) -> None:
        spec = _make_spec(
            covariates=[
                CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0),
            ]
        )
        df = pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": [1.0]})
        report = validate(spec, level=ValidationLevel.DATA_BOUND, lane=Lane.DISCOVERY, data=df)
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.DATA_COVARIATE_COLUMN_MISSING.value in codes


class TestBackendBound:
    def test_node_capability_gap_against_stan(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive"),
            experimental=ExperimentalFlags(node=True),
        )
        report = validate(
            spec, level=ValidationLevel.BACKEND_BOUND, lane=Lane.DISCOVERY, backend="stan"
        )
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.BE_CAPABILITY_UNSUPPORTED.value in codes
        assert report.ok is False

    def test_unknown_backend(self) -> None:
        spec = _make_spec()
        report = validate(
            spec,
            level=ValidationLevel.BACKEND_BOUND,
            lane=Lane.DISCOVERY,
            backend="not_a_real_emitter",
        )
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.BE_UNKNOWN_BACKEND.value in codes

    def test_supported_spec_is_clean(self) -> None:
        spec = _make_spec()
        report = validate(
            spec, level=ValidationLevel.BACKEND_BOUND, lane=Lane.DISCOVERY, backend="nlmixr2"
        )
        assert report.all_errors == []
        assert report.ok is True


class TestPolicyBound:
    def test_lane_mismatch(self) -> None:
        spec = _make_spec()
        policy = _policy(Lane.DISCOVERY, node_eligible=False)
        report = validate(
            spec, level=ValidationLevel.POLICY_BOUND, lane=Lane.SUBMISSION, policy=policy
        )
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.POLICY_LANE_MISMATCH.value in codes

    def test_node_ineligible_under_policy(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive"),
            experimental=ExperimentalFlags(node=True),
        )
        policy = _policy(Lane.DISCOVERY, node_eligible=False)
        report = validate(
            spec, level=ValidationLevel.POLICY_BOUND, lane=Lane.DISCOVERY, policy=policy
        )
        codes = {str(e.code) for e in report.all_errors}
        assert FrmCode.POLICY_NODE_INELIGIBLE.value in codes

    def test_matching_lane_and_node_eligible_is_clean(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive"),
            experimental=ExperimentalFlags(node=True),
        )
        policy = _policy(Lane.DISCOVERY, node_eligible=True)
        report = validate(
            spec, level=ValidationLevel.POLICY_BOUND, lane=Lane.DISCOVERY, policy=policy
        )
        assert report.all_errors == []
        assert report.ok is True


class TestValidationReportOk:
    def test_ok_false_when_error_severity_present(self) -> None:
        spec = _make_spec(initial={"ka": -1.0, "V": 70.0, "CL": 5.0})
        report = validate(spec, level=ValidationLevel.SEMANTIC, lane=Lane.SUBMISSION)
        assert any(e.severity == "error" for e in report.all_errors)
        assert report.ok is False

    def test_ok_true_when_no_errors(self) -> None:
        spec = _make_spec()
        report = validate(
            spec,
            level=ValidationLevel.ALL,
            lane=Lane.SUBMISSION,
            data=pd.DataFrame({"ID": [1], "TIME": [0.0], "DV": [1.0]}),
            backend="nlmixr2",
            policy=_policy(Lane.SUBMISSION),
        )
        assert report.ok is True
        assert report.all_errors == []
        assert report.skipped_levels == {}

    def test_all_errors_is_stable_level_order(self) -> None:
        spec = _make_spec(initial={"ka": -1.0, "V": 70.0, "CL": 5.0})
        report = validate(spec, level=ValidationLevel.ALL, lane=Lane.SUBMISSION)
        levels_seen = [
            lvl
            for lvl in (
                ValidationLevel.SYNTAX,
                ValidationLevel.AST,
                ValidationLevel.SEMANTIC,
                ValidationLevel.DATA_BOUND,
                ValidationLevel.LANE_BOUND,
                ValidationLevel.BACKEND_BOUND,
                ValidationLevel.POLICY_BOUND,
            )
            for e in report.by_level.get(lvl, [])
        ]
        assert len(levels_seen) == len(report.all_errors)


@pytest.mark.parametrize("level", list(ValidationLevel))
def test_every_level_value_is_selectable(level: ValidationLevel) -> None:
    """No level (including ALL) raises when passed to validate()."""
    spec = _make_spec()
    report = validate(spec, level=level, lane=Lane.SUBMISSION)
    levels = report.levels_run
    if levels & {
        ValidationLevel.DATA_BOUND,
        ValidationLevel.BACKEND_BOUND,
        ValidationLevel.POLICY_BOUND,
    }:
        assert report.skipped_levels
        assert report.ok is False
    else:
        assert report.skipped_levels == {}
        assert report.ok is True
