# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression coverage for cross-layer DSL review findings."""

from __future__ import annotations

import math

import pandas as pd

from apmode.dsl.ast_models import (
    IIV,
    TMDDQSS,
    Additive,
    CovariateLink,
    DSLSpec,
    ExperimentalFlags,
    FirstOrder,
    LinearElim,
    MixedFirstZero,
    NODEAbsorption,
    ObservationEndpoint,
    OneCmt,
    Proportional,
    SumIG,
    TMDDCore,
    UnitsDeclaration,
    ZeroOrder,
)
from apmode.dsl.canonical import spec_fingerprint
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.lane import Lane
from apmode.dsl.migration import migrate_v06_to_v07
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.priors import NormalPrior, PriorSpec
from apmode.dsl.serializer import serialize_spec
from apmode.dsl.stan_emitter import emit_stan
from apmode.dsl.transforms import ReplaceWithNODE, apply_transform
from apmode.dsl.validator import validate_data_bound, validate_dsl


def _base_spec(**updates: object) -> DSLSpec:
    values: dict[str, object] = {
        "model_id": "review-regression",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [],
        "observation": Proportional(sigma_prop=0.1),
        "initial": {"ka": 1.0, "V": 70.0, "CL": 5.0},
    }
    values.update(updates)
    return DSLSpec(**values)  # type: ignore[arg-type]


def test_parameter_references_are_canonicalized_before_emission() -> None:
    spec = _base_spec(
        variability=[IIV(params=["cl"], structure="diagonal")],
        covariates=[
            CovariateLink(param="cl", covariate="SEX", form="categorical", reference="female")
        ],
    )
    assert spec.variability[0].params == ["CL"]
    assert spec.covariates[0].param == "CL"
    code = emit_nlmixr2(spec)
    assert "CL <- exp(lCL + eta.CL + beta_CL_SEX * SEX)" in code


def test_programmatic_invalid_prior_is_rejected_by_semantic_validation() -> None:
    spec = _base_spec(
        priors=[PriorSpec(target="omega_NOPE", family=NormalPrior(mu=0.0, sigma=1.0))]
    )
    errors = validate_dsl(spec, lane=Lane.DISCOVERY)
    assert any(error.constraint == "prior_invalid_declaration" for error in errors)


def test_nonfinite_initial_and_sigma_are_rejected() -> None:
    spec = _base_spec(
        observation=Proportional(sigma_prop=math.inf),
        initial={"ka": 1.0, "V": 70.0, "CL": math.nan},
    )
    errors = validate_dsl(spec, lane=Lane.DISCOVERY)
    assert {error.param for error in errors if error.constraint == "positive"} >= {
        "elimination.CL",
        "observation.sigma_prop",
    }


def test_single_explicit_endpoint_uses_its_declared_prediction_and_sigma_name() -> None:
    endpoint = ObservationEndpoint(
        name="target",
        dvid=1,
        prediction="C_target_total",
        error=Additive(sigma_add=0.2),
    )
    spec = _base_spec(
        distribution=TMDDQSS(),
        observations={"target": endpoint},
        observation=endpoint.error,
        initial={"ka": 1.0, "V": 3.0, "R0": 2.0, "KD": 0.5, "kint": 0.2, "CL": 1.0},
    )
    code = emit_nlmixr2(spec)
    assert "add.sd.target <- 0.2" in code
    assert "Rtot ~ add(add.sd.target)" in code
    assert "cp ~ add(add.sd.target)" not in code


def test_endpoint_order_is_dvid_order_and_noncontiguous_values_fail() -> None:
    plasma = ObservationEndpoint(
        name="plasma",
        dvid=1,
        prediction="C_central",
        error=Proportional(sigma_prop=0.1),
    )
    target = ObservationEndpoint(
        name="target",
        dvid=2,
        prediction="C_target_total",
        error=Additive(sigma_add=0.2),
    )
    spec = _base_spec(
        distribution=TMDDQSS(),
        observations={"target": target, "plasma": plasma},
        initial={"ka": 1.0, "V": 3.0, "R0": 2.0, "KD": 0.5, "kint": 0.2, "CL": 1.0},
    )
    assert [endpoint.name for endpoint in spec.observation_endpoints()] == ["plasma", "target"]
    code = emit_nlmixr2(spec)
    assert code.index("endpoint 'plasma'") < code.index("endpoint 'target'")

    invalid = spec.model_copy(
        update={
            "observations": {
                "plasma": plasma,
                "target": target.model_copy(update={"dvid": 3}),
            }
        }
    )
    assert any(
        error.constraint == "observations_dvid_sequence"
        for error in validate_dsl(invalid, lane=Lane.DISCOVERY)
    )


def test_sumig_k1_emits_one_density_and_suppresses_direct_bolus() -> None:
    spec = _base_spec(
        absorption=SumIG(k=1),
        initial={"MT_1": 2.0, "RD2_1": 0.5, "V": 70.0, "CL": 5.0},
    )
    assert spec.structural_param_names()[:2] == ["MT_1", "RD2_1"]
    code = emit_nlmixr2(spec)
    assert "ig_1 <-" in code
    assert "ig_2 <-" not in code
    assert "_sumig_t <- t - SUMIG_T0" in code
    assert "f(centr) <- 0" in code


def test_sumig_data_contract_rejects_undosed_subjects_and_addl() -> None:
    spec = _base_spec(
        absorption=SumIG(k=1),
        initial={"MT_1": 2.0, "RD2_1": 0.5, "V": 70.0, "CL": 5.0},
    )
    data = pd.DataFrame(
        {
            "NMID": [1, 1, 2],
            "TIME": [0.0, 1.0, 1.0],
            "EVID": [1, 0, 0],
            "AMT": [100.0, 0.0, 0.0],
            "ADDL": [2, 0, 0],
        }
    )
    errors = validate_data_bound(spec, data)
    assert any(error.constraint == "data_sumig_single_dose" for error in errors)
    assert "ADDL>0" in next(
        error.message for error in errors if error.constraint == "data_sumig_single_dose"
    )


def test_tmdd_absorption_duration_targets_existing_central_state() -> None:
    core = _base_spec(
        absorption=ZeroOrder(),
        distribution=TMDDCore(),
        initial={
            "dur": 2.0,
            "V": 3.0,
            "R0": 2.0,
            "kon": 0.4,
            "koff": 0.1,
            "kint": 0.2,
            "CL": 1.0,
        },
    )
    assert "dur(centr) <- dur" in emit_nlmixr2(core)

    qss = _base_spec(
        absorption=MixedFirstZero(),
        distribution=TMDDQSS(),
        initial={
            "ka": 1.0,
            "dur": 2.0,
            "frac": 0.5,
            "V": 3.0,
            "R0": 2.0,
            "KD": 0.5,
            "kint": 0.2,
            "CL": 1.0,
        },
    )
    qss_code = emit_nlmixr2(qss)
    assert "dur(Atot) <- dur" in qss_code
    assert "f(Atot) <- 1 - frac" in qss_code


def test_fixed_external_is_not_an_estimated_parameter_in_either_emitter() -> None:
    fixed = PriorSpec(
        target="CL",
        family=NormalPrior(mu=0.0, sigma=2.0),
        source="fixed_external",
    )
    spec = _base_spec(priors=[fixed])
    assert "lCL <- fix(log(5.0))" in emit_nlmixr2(spec)
    stan = emit_stan(spec)
    assert "real log_CL = log(5);" in stan
    assert "  real log_CL;" not in stan
    assert "log_CL ~" not in stan


def test_stan_allows_negative_dv_for_additive_likelihoods_only() -> None:
    additive = emit_stan(_base_spec(observation=Additive(sigma_add=0.2)))
    proportional = emit_stan(_base_spec())
    assert "vector[N] dv;" in additive
    assert "vector<lower=0>[N] dv;" in proportional


def test_units_change_spec_fingerprint() -> None:
    first = _base_spec(
        units=UnitsDeclaration(time="h", amount="mg", concentration="mg/L", volume="L")
    )
    second = first.model_copy(
        update={
            "units": UnitsDeclaration(time="min", amount="ng", concentration="ng/mL", volume="mL")
        }
    )
    assert spec_fingerprint(first)["digest"] != spec_fingerprint(second)["digest"]


def test_migration_ignores_model_like_text_in_comments_and_strings() -> None:
    source = """model {
    metadata: { intent = "Try FirstOrder(ka=88) later" }
    // Try FirstOrder(ka=99) later
    absorption: FirstOrder(ka=1)
    distribution: OneCmt(V=70)
    elimination: Linear(CL=5)
    observation: Proportional(sigma_prop=0.1)
}"""
    migrated = migrate_v06_to_v07(source).text
    assert "ka = 1" in migrated
    assert "ka = 99" not in migrated
    assert "ka = 88" not in migrated


def test_node_experimental_opt_in_round_trips_through_serializer() -> None:
    spec = _base_spec(
        absorption=NODEAbsorption(dim=2, constraint_template="monotone_increasing"),
        initial={"V": 70.0, "CL": 5.0},
        experimental=ExperimentalFlags(node=True),
    )
    reparsed = compile_dsl(serialize_spec(spec))
    assert reparsed.experimental.node is True
    assert not any(
        error.constraint == "node_experimental_gate"
        for error in validate_dsl(reparsed, lane=Lane.DISCOVERY)
    )


def test_replace_with_node_transform_sets_explicit_opt_in() -> None:
    candidate = apply_transform(
        _base_spec(),
        ReplaceWithNODE(
            position="absorption",
            dim=2,
            constraint_template="monotone_increasing",
        ),
    )
    assert candidate.experimental.node is True
