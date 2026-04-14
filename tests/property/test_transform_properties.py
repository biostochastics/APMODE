# SPDX-License-Identifier: GPL-2.0-or-later
"""Hypothesis property tests for Formular transform safety."""

from hypothesis import given, settings
from hypothesis import strategies as st

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    IIV,
    Additive,
    Combined,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.transforms import (
    AddCovariateLink,
    AdjustVariability,
    ReplaceWithNODE,
    SwapModule,
    ToggleLag,
    apply_transform,
)
from apmode.dsl.validator import validate_dsl


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="prop-test",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


@given(
    obs_module=st.sampled_from(
        [
            Proportional(sigma_prop=0.1),
            Additive(sigma_add=0.5),
            Combined(sigma_prop=0.1, sigma_add=0.5),
        ]
    )
)
@settings(max_examples=20)
def test_swap_observation_always_valid(obs_module: object) -> None:
    """Swapping observation module should always produce a valid spec."""
    spec = _base_spec()
    t = SwapModule(position="observation", new_module=obs_module)  # type: ignore[arg-type]
    new_spec = apply_transform(spec, t)
    errors = validate_dsl(new_spec, lane=Lane.SUBMISSION)
    assert len(errors) == 0, f"Unexpected errors: {errors}"


@given(form=st.sampled_from(["power", "exponential", "linear", "categorical"]))
@settings(max_examples=20)
def test_add_covariate_to_valid_param(form: str) -> None:
    """Adding a covariate to a valid structural param should succeed."""
    spec = _base_spec()
    t = AddCovariateLink(param="CL", covariate="WT", form=form)  # type: ignore[arg-type]
    new_spec = apply_transform(spec, t)
    assert any(
        hasattr(v, "covariate") and v.covariate == "WT"  # type: ignore[union-attr]
        for v in new_spec.variability
    )


def test_replace_with_node_fails_submission_lane() -> None:
    """ReplaceWithNODE must always fail DSL validation in submission lane."""
    spec = _base_spec()
    t = ReplaceWithNODE(
        position="elimination",
        constraint_template="bounded_positive",
        dim=4,
    )
    new_spec = apply_transform(spec, t)
    errors = validate_dsl(new_spec, lane=Lane.SUBMISSION)
    assert any("NODE" in e.message for e in errors)


@given(action=st.sampled_from(["add", "remove", "upgrade_to_block"]))
@settings(max_examples=20)
def test_adjust_variability_produces_valid_spec(action: str) -> None:
    """Variability adjustments on valid params should not crash."""
    spec = _base_spec()
    # Use CL which is always in the spec
    t = AdjustVariability(param="CL", action=action)  # type: ignore[arg-type]
    new_spec = apply_transform(spec, t)
    assert new_spec.model_id != spec.model_id


def test_toggle_lag_roundtrip() -> None:
    """toggle_lag(on) then toggle_lag(off) should return to FirstOrder."""
    spec = _base_spec()
    on_spec = apply_transform(spec, ToggleLag(on=True))
    assert isinstance(on_spec.absorption, LaggedFirstOrder)
    off_spec = apply_transform(on_spec, ToggleLag(on=False))
    assert isinstance(off_spec.absorption, FirstOrder)
    assert off_spec.absorption.ka == spec.absorption.ka
