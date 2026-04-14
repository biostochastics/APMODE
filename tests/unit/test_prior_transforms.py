# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the SetPrior transform (src/apmode/dsl/prior_transforms.py).

Also exercises the integration with FormularTransform via transforms.py:
validate_transform and apply_transform dispatch on SetPrior correctly.
"""

from __future__ import annotations

import pytest

from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.prior_transforms import SetPrior, apply_set_prior, validate_set_prior
from apmode.dsl.priors import (
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorSpec,
)
from apmode.dsl.transforms import apply_transform, validate_transform


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=20.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.3),
    )


class TestValidateSetPrior:
    def test_valid_structural_prior(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.5))
        assert validate_set_prior(spec, t) == []

    def test_unknown_target(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="BOGUS", family=NormalPrior(mu=0, sigma=1))
        errors = validate_set_prior(spec, t)
        assert len(errors) == 1 and "does not resolve" in errors[0]

    def test_wrong_family_for_target(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="CL", family=HalfCauchyPrior(scale=1.0))
        errors = validate_set_prior(spec, t)
        assert len(errors) == 1 and "HalfCauchy" in errors[0]

    def test_iiv_sd_target(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="omega_CL", family=HalfCauchyPrior(scale=0.5))
        assert validate_set_prior(spec, t) == []

    def test_iiv_sd_rejects_normal(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="omega_CL", family=NormalPrior(mu=0, sigma=1))
        errors = validate_set_prior(spec, t)
        assert len(errors) >= 1

    def test_residual_sd_target(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="sigma_prop", family=HalfNormalPrior(sigma=0.3))
        assert validate_set_prior(spec, t) == []

    def test_historical_source_without_justification_rejected(self) -> None:
        spec = _base_spec()
        t = SetPrior(
            target="CL",
            family=NormalPrior(mu=1.5, sigma=0.3),
            source="historical_data",
            historical_refs=["phase2_trial"],
            justification="",
        )
        errors = validate_set_prior(spec, t)
        assert any("justification" in e for e in errors)


class TestApplySetPrior:
    def test_append_new_prior(self) -> None:
        spec = _base_spec()
        assert spec.priors == []
        t = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.5))
        new_spec = apply_set_prior(spec, t)
        assert len(new_spec.priors) == 1
        assert new_spec.priors[0].target == "CL"
        assert new_spec.model_id != spec.model_id

    def test_idempotent_replace(self) -> None:
        spec = _base_spec().model_copy(
            update={"priors": [PriorSpec(target="CL", family=NormalPrior(mu=0, sigma=2))]}
        )
        t = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))
        new_spec = apply_set_prior(spec, t)
        assert len(new_spec.priors) == 1
        assert new_spec.priors[0].family.mu == 1.5  # type: ignore[union-attr]

    def test_multiple_distinct_priors_preserved(self) -> None:
        spec = _base_spec()
        t1 = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))
        s1 = apply_set_prior(spec, t1)
        t2 = SetPrior(target="V", family=NormalPrior(mu=3.0, sigma=0.3))
        s2 = apply_set_prior(s1, t2)
        assert {p.target for p in s2.priors} == {"CL", "V"}

    def test_invalid_transform_raises(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="CL", family=HalfCauchyPrior(scale=1.0))
        with pytest.raises(ValueError):
            apply_set_prior(spec, t)


class TestFormularTransformIntegration:
    """SetPrior should work through the top-level transforms dispatch."""

    def test_validate_transform_routes_to_set_prior(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))
        errors = validate_transform(spec, t)
        assert errors == []

    def test_apply_transform_routes_to_set_prior(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))
        new_spec = apply_transform(spec, t)
        assert len(new_spec.priors) == 1
        assert new_spec.priors[0].target == "CL"

    def test_validate_transform_surfaces_set_prior_errors(self) -> None:
        spec = _base_spec()
        t = SetPrior(target="BOGUS", family=NormalPrior(mu=0, sigma=1))
        errors = validate_transform(spec, t)
        assert len(errors) >= 1


class TestSetPriorWithExoticFamilies:
    def test_mixture_prior_on_structural(self) -> None:
        spec = _base_spec()
        t = SetPrior(
            target="CL",
            family=MixturePrior(
                components=[
                    LogNormalPrior(mu=1.5, sigma=0.3),
                    NormalPrior(mu=0.0, sigma=2.0),
                ],
                weights=[0.8, 0.2],
            ),
        )
        assert validate_set_prior(spec, t) == []
        new_spec = apply_set_prior(spec, t)
        assert isinstance(new_spec.priors[0].family, MixturePrior)

    def test_historical_borrowing_on_structural(self) -> None:
        spec = _base_spec()
        t = SetPrior(
            target="CL",
            family=HistoricalBorrowingPrior(
                map_mean=1.5,
                map_sd=0.3,
                robust_weight=0.2,
                historical_refs=["phase2_trial"],
            ),
            source="historical_data",
            justification="MAP from phase-2 dose-finding study",
            historical_refs=["phase2_trial"],
        )
        errors = validate_set_prior(spec, t)
        assert errors == []
        new_spec = apply_set_prior(spec, t)
        assert isinstance(new_spec.priors[0].family, HistoricalBorrowingPrior)
