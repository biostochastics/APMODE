# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Formular transform types (PRD §4.2.5)."""

import pytest

from apmode.dsl.ast_models import (
    IIV,
    IOV,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    OccasionByStudy,
    OneCmt,
    Proportional,
    Transit,
    TwoCmt,
)
from apmode.dsl.transforms import (
    AddCovariateLink,
    AdjustVariability,
    ReplaceWithNODE,
    SetTransitN,
    SwapModule,
    ToggleLag,
    apply_transform,
    validate_transform,
)


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test-base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


class TestSwapModule:
    def test_swap_elimination_linear_to_mm(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(Vmax=50.0, Km=5.0),
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.elimination.type == "MichaelisMenten"
        assert new_spec.model_id != spec.model_id

    def test_swap_distribution_1cmt_to_2cmt(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="distribution",
            new_module=TwoCmt(V1=30.0, V2=40.0, Q=5.0),
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.distribution.type == "TwoCmt"

    def test_swap_preserves_other_modules(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(Vmax=50.0, Km=5.0),
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.absorption == spec.absorption
        assert new_spec.distribution == spec.distribution
        assert new_spec.observation == spec.observation


class TestAddCovariateLink:
    def test_add_covariate_link(self) -> None:
        spec = _base_spec()
        t = AddCovariateLink(param="CL", covariate="WT", form="power")
        new_spec = apply_transform(spec, t)
        cov_links = [v for v in new_spec.variability if isinstance(v, CovariateLink)]
        assert len(cov_links) == 1
        assert cov_links[0].param == "CL"

    def test_rejects_invalid_param(self) -> None:
        spec = _base_spec()
        t = AddCovariateLink(param="NONEXISTENT", covariate="WT", form="power")
        errors = validate_transform(spec, t)
        assert len(errors) > 0

    def test_rejects_duplicate_covariate(self) -> None:
        spec = _base_spec()
        t1 = AddCovariateLink(param="CL", covariate="WT", form="power")
        spec2 = apply_transform(spec, t1)
        t2 = AddCovariateLink(param="CL", covariate="WT", form="exponential")
        errors = validate_transform(spec2, t2)
        assert len(errors) > 0


class TestAdjustVariability:
    def test_add_param(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="ka", action="add")
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "ka" in iiv.params

    def test_remove_param(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="V", action="remove")
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "V" not in iiv.params

    def test_upgrade_to_block(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="CL", action="upgrade_to_block")
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert iiv.structure == "block"


class TestSetTransitN:
    def test_set_transit_n(self) -> None:
        spec = DSLSpec(
            model_id="transit-base",
            absorption=Transit(n=3, ktr=2.0, ka=1.0),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=2.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        t = SetTransitN(n=6)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, Transit)
        assert new_spec.absorption.n == 6

    def test_rejects_non_transit(self) -> None:
        spec = _base_spec()
        t = SetTransitN(n=6)
        errors = validate_transform(spec, t)
        assert len(errors) > 0


class TestToggleLag:
    def test_toggle_lag_on(self) -> None:
        spec = _base_spec()
        t = ToggleLag(on=True)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, LaggedFirstOrder)
        assert new_spec.absorption.ka == 1.0

    def test_toggle_lag_off(self) -> None:
        spec = DSLSpec(
            model_id="lagged-base",
            absorption=LaggedFirstOrder(ka=1.0, tlag=0.5),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=2.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        t = ToggleLag(on=False)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, FirstOrder)


class TestReplaceWithNODE:
    def test_replace_elimination_with_node(self) -> None:
        spec = _base_spec()
        t = ReplaceWithNODE(
            position="elimination",
            constraint_template="bounded_positive",
            dim=4,
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.elimination.type == "NODE_Elimination"

    def test_replace_absorption_with_node(self) -> None:
        spec = _base_spec()
        t = ReplaceWithNODE(
            position="absorption",
            constraint_template="monotone_increasing",
            dim=3,
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.absorption.type == "NODE_Absorption"


class TestSwapModuleValidation:
    def test_rejects_invalid_position(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            SwapModule(position="nonexistent", new_module=LinearElim(CL=2.0))  # type: ignore[arg-type]

    def test_rejects_wrong_module_for_position(self) -> None:
        from apmode.dsl.transforms import validate_transform

        spec = _base_spec()
        t = SwapModule(position="absorption", new_module=LinearElim(CL=2.0))
        errors = validate_transform(spec, t)
        assert len(errors) > 0
        assert "not valid for position" in errors[0]


class TestVariabilityPruning:
    def test_swap_elimination_prunes_stale_iiv(self) -> None:
        """SwapModule that changes elim params should prune stale IIV refs."""
        spec = _base_spec()  # IIV on [CL, V]
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(Vmax=50.0, Km=5.0),
        )
        new_spec = apply_transform(spec, t)
        # CL is gone (replaced by Vmax, Km), V is still present
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "CL" not in iiv.params
        assert "V" in iiv.params

    def test_swap_distribution_prunes_stale_covariate(self) -> None:
        """Swapping distribution should remove orphaned CovariateLinks."""
        from apmode.dsl.ast_models import CovariateLink

        spec = DSLSpec(
            model_id="test-cov",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=2.0),
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="V", covariate="WT", form="power"),
            ],
            observation=Proportional(sigma_prop=0.1),
        )
        t = SwapModule(
            position="distribution",
            new_module=TwoCmt(V1=30.0, V2=40.0, Q=5.0),
        )
        new_spec = apply_transform(spec, t)
        # V is gone (now V1, V2, Q), CovariateLink on V should be removed
        cov_links = [v for v in new_spec.variability if isinstance(v, CovariateLink)]
        assert len(cov_links) == 0

    def test_swap_downgrades_block_to_diagonal_if_single_param(self) -> None:
        """Block IIV with 2 params, one pruned → downgrade to diagonal."""
        spec = DSLSpec(
            model_id="test-block",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=2.0),
            variability=[IIV(params=["CL", "V"], structure="block")],
            observation=Proportional(sigma_prop=0.1),
        )
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(Vmax=50.0, Km=5.0),
        )
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        # CL removed, only V remains → block downgraded to diagonal
        assert iiv.structure == "diagonal"

    def test_swap_elimination_prunes_stale_iov(self) -> None:
        """SwapModule that changes elim params should prune stale IOV refs."""
        spec = DSLSpec(
            model_id="test-iov-prune",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=2.0),
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                IOV(params=["CL"], occasions=OccasionByStudy()),
            ],
            observation=Proportional(sigma_prop=0.1),
        )
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(Vmax=50.0, Km=5.0),
        )
        new_spec = apply_transform(spec, t)
        # CL is gone, IOV on CL should be pruned
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 0
