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
    ObservationEndpoint,
    OccasionByStudy,
    OccasionByVisit,
    OneCmt,
    Proportional,
    SumIG,
    Transit,
    TwoCmt,
    UnitsDeclaration,
)
from apmode.dsl.transforms import (
    AddCovariateLink,
    AddParallelRoute,
    AdjustVariability,
    ConvertTransitToErlang,
    ReplaceWithNODE,
    SetSumIGComponents,
    SetTransitN,
    SwapModule,
    ToggleLag,
    apply_transform,
    validate_transform,
)


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test-base",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
    )


class TestSwapModule:
    def test_swap_elimination_linear_to_mm(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.elimination.type == "MichaelisMenten"
        assert new_spec.model_id != spec.model_id

    def test_swap_distribution_1cmt_to_2cmt(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="distribution",
            new_module=TwoCmt(),
            initial_overrides={"V1": 50.0, "V2": 80.0, "Q": 10.0},
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.distribution.type == "TwoCmt"

    def test_swap_preserves_other_modules(self) -> None:
        spec = _base_spec()
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.absorption == spec.absorption
        assert new_spec.distribution == spec.distribution
        assert new_spec.observation == spec.observation

    def test_transform_preserves_sidecar_spec_fields(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.1),
                    )
                },
                "units": UnitsDeclaration(time="h", amount="mg", concentration="mg/L", volume="L"),
                "source_meta": {"absorption": (3, 5)},
                "macros_used": ["pkstd.standard_iiv@v1"],
            }
        )
        new_spec = apply_transform(
            spec,
            SwapModule(
                position="elimination",
                new_module=MichaelisMenten(),
                initial_overrides={"Vmax": 100.0, "Km": 10.0},
            ),
        )
        assert new_spec.observations == spec.observations
        assert new_spec.units == spec.units
        assert new_spec.source_meta == spec.source_meta
        assert new_spec.macros_used == spec.macros_used


class TestAddCovariateLink:
    def test_add_covariate_link(self) -> None:
        spec = _base_spec()
        t = AddCovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
        new_spec = apply_transform(spec, t)
        assert len(new_spec.covariates) == 1
        assert new_spec.covariates[0].param == "CL"

    def test_rejects_invalid_param(self) -> None:
        spec = _base_spec()
        t = AddCovariateLink(
            param="NONEXISTENT", covariate="WT", form="power", theta=0.75, ref=70.0
        )
        errors = validate_transform(spec, t)
        assert len(errors) > 0

    def test_rejects_duplicate_covariate(self) -> None:
        spec = _base_spec()
        t1 = AddCovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
        spec2 = apply_transform(spec, t1)
        t2 = AddCovariateLink(param="CL", covariate="WT", form="exponential", theta=0.5)
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

    def test_add_iov_creates_new_item_with_default_occasions(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="CL", action="add_iov")
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 1
        assert iov_items[0].params == ["CL"]
        assert iov_items[0].occasions == OccasionByStudy()

    def test_add_iov_merges_into_existing_item_same_occasions(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "variability": [
                    *_base_spec().variability,
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                ]
            }
        )
        t = AdjustVariability(param="V", action="add_iov", occasions=OccasionByStudy())
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 1
        assert sorted(iov_items[0].params) == ["CL", "V"]

    def test_add_iov_same_param_same_occasions_is_idempotent(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "variability": [
                    *_base_spec().variability,
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                ]
            }
        )
        t = AdjustVariability(param="CL", action="add_iov", occasions=OccasionByStudy())
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 1
        assert iov_items[0].params == ["CL"]

    def test_add_iov_conflicting_occasions_raises(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "variability": [
                    *_base_spec().variability,
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                ]
            }
        )
        t = AdjustVariability(
            param="CL", action="add_iov", occasions=OccasionByVisit(column="VISIT")
        )
        errors = validate_transform(spec, t)
        assert len(errors) > 0
        with pytest.raises(ValueError, match="add_iov"):
            apply_transform(spec, t)

    def test_remove_iov_drops_only_param_removes_item(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "variability": [
                    *_base_spec().variability,
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                ]
            }
        )
        t = AdjustVariability(param="CL", action="remove_iov")
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert iov_items == []

    def test_remove_iov_drops_one_of_several_params(self) -> None:
        spec = _base_spec().model_copy(
            update={
                "variability": [
                    *_base_spec().variability,
                    IOV(params=["CL", "V"], occasions=OccasionByStudy()),
                ]
            }
        )
        t = AdjustVariability(param="CL", action="remove_iov")
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 1
        assert iov_items[0].params == ["V"]

    def test_remove_iov_no_existing_iov_is_noop(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="CL", action="remove_iov")
        new_spec = apply_transform(spec, t)
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert iov_items == []
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "CL" in iiv.params


class TestSetTransitN:
    def test_set_transit_n(self) -> None:
        spec = DSLSpec(
            model_id="transit-base",
            absorption=Transit(n=3),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ktr": 2.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
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
        assert new_spec.initial["ka"] == 1.0
        assert "tlag" in new_spec.initial

    def test_toggle_lag_off(self) -> None:
        spec = DSLSpec(
            model_id="lagged-base",
            absorption=LaggedFirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "tlag": 0.5, "V": 70.0, "CL": 5.0},
        )
        t = ToggleLag(on=False)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, FirstOrder)
        assert "tlag" not in new_spec.initial


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
            SwapModule(position="nonexistent", new_module=LinearElim())  # type: ignore[arg-type]

    def test_rejects_wrong_module_for_position(self) -> None:
        from apmode.dsl.transforms import validate_transform

        spec = _base_spec()
        t = SwapModule(position="absorption", new_module=LinearElim())
        errors = validate_transform(spec, t)
        assert len(errors) > 0
        assert "not valid for position" in errors[0]


class TestVariabilityPruning:
    def test_swap_elimination_prunes_stale_iiv(self) -> None:
        """SwapModule that changes elim params should prune stale IIV refs."""
        spec = _base_spec()  # IIV on [CL, V]
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        new_spec = apply_transform(spec, t)
        # CL is gone (replaced by Vmax, Km), V is still present
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "CL" not in iiv.params
        assert "V" in iiv.params

    def test_swap_distribution_prunes_stale_covariate(self) -> None:
        """Swapping distribution should remove orphaned CovariateLinks."""
        spec = DSLSpec(
            model_id="test-cov",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            covariates=[
                CovariateLink(param="V", covariate="WT", form="power", theta=1.0, ref=70.0)
            ],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        t = SwapModule(
            position="distribution",
            new_module=TwoCmt(),
            initial_overrides={"V1": 50.0, "V2": 80.0, "Q": 10.0},
        )
        new_spec = apply_transform(spec, t)
        # V is gone (now V1, V2, Q), CovariateLink on V should be removed
        assert new_spec.covariates == []

    def test_swap_downgrades_block_to_diagonal_if_single_param(self) -> None:
        """Block IIV with 2 params, one pruned → downgrade to diagonal."""
        spec = DSLSpec(
            model_id="test-block",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="block")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        # CL removed, only V remains → block downgraded to diagonal
        assert iiv.structure == "diagonal"

    def test_swap_elimination_prunes_stale_iov(self) -> None:
        """SwapModule that changes elim params should prune stale IOV refs."""
        spec = DSLSpec(
            model_id="test-iov-prune",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                IOV(params=["CL"], occasions=OccasionByStudy()),
            ],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        t = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        new_spec = apply_transform(spec, t)
        # CL is gone, IOV on CL should be pruned
        iov_items = [v for v in new_spec.variability if isinstance(v, IOV)]
        assert len(iov_items) == 0


class TestRationaleAndExpectedDiagnosticEffect:
    """P2.2 (Formular sharpening plan): every FormularTransform variant

    accepts optional ``rationale``/``expected_diagnostic_effect`` fields.
    These are pure provenance — they must never affect
    ``validate_transform``/``apply_transform`` semantics, and omitting them
    must keep every pre-P2.2 call site working unchanged.
    """

    def test_swap_module_rationale_does_not_affect_semantics(self) -> None:
        spec = _base_spec()
        t_bare = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
        )
        t_annotated = SwapModule(
            position="elimination",
            new_module=MichaelisMenten(),
            initial_overrides={"Vmax": 100.0, "Km": 10.0},
            rationale="CWRES show saturable elimination.",
            expected_diagnostic_effect=["reduces CWRES trend at high dose"],
        )
        assert t_annotated.rationale == "CWRES show saturable elimination."
        assert t_annotated.expected_diagnostic_effect == ["reduces CWRES trend at high dose"]
        assert t_bare.rationale == ""
        assert t_bare.expected_diagnostic_effect == []
        assert validate_transform(spec, t_bare) == validate_transform(spec, t_annotated)
        bare_result = apply_transform(spec, t_bare)
        annotated_result = apply_transform(spec, t_annotated)
        assert bare_result.elimination == annotated_result.elimination
        assert bare_result.initial == annotated_result.initial

    def test_add_covariate_link_rationale_roundtrip(self) -> None:
        spec = _base_spec()
        t = AddCovariateLink(
            param="CL",
            covariate="WT",
            form="power",
            theta=0.75,
            ref=70.0,
            rationale="Wide body-weight range.",
            expected_diagnostic_effect=["lowers CL eta shrinkage"],
        )
        new_spec = apply_transform(spec, t)
        assert len(new_spec.covariates) == 1
        assert t.rationale == "Wide body-weight range."
        assert t.expected_diagnostic_effect == ["lowers CL eta shrinkage"]

    def test_adjust_variability_rationale_roundtrip(self) -> None:
        spec = _base_spec()
        t = AdjustVariability(param="ka", action="add", rationale="ka varies across subjects.")
        new_spec = apply_transform(spec, t)
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "ka" in iiv.params
        assert t.rationale == "ka varies across subjects."

    def test_set_transit_n_rationale_roundtrip(self) -> None:
        spec = DSLSpec(
            model_id="transit-base",
            absorption=Transit(n=3),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ktr": 2.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        t = SetTransitN(
            n=6,
            rationale="Delayed absorption profile.",
            expected_diagnostic_effect=["reduces early time-point residuals"],
        )
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, Transit)
        assert new_spec.absorption.n == 6
        assert t.rationale == "Delayed absorption profile."

    def test_toggle_lag_rationale_roundtrip(self) -> None:
        spec = _base_spec()
        t = ToggleLag(on=True, rationale="Lag observed pre-Tmax.")
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, LaggedFirstOrder)
        assert t.rationale == "Lag observed pre-Tmax."

    def test_replace_with_node_rationale_roundtrip(self) -> None:
        spec = _base_spec()
        t = ReplaceWithNODE(
            position="elimination",
            constraint_template="bounded_positive",
            dim=4,
            rationale="Complex nonlinear elimination not captured by MM.",
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.elimination.type == "NODE_Elimination"
        assert t.rationale == "Complex nonlinear elimination not captured by MM."

    def test_convert_transit_to_erlang_rationale_roundtrip(self) -> None:
        spec = DSLSpec(
            model_id="transit-base",
            absorption=Transit(n=3),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ktr": 2.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        t = ConvertTransitToErlang(n=3, rationale="Simplify to Erlang absorption.")
        new_spec = apply_transform(spec, t)
        assert new_spec.absorption.type == "Erlang"
        assert t.rationale == "Simplify to Erlang absorption."

    def test_add_parallel_route_rationale_roundtrip(self) -> None:
        spec = _base_spec()  # FirstOrder absorption
        t = AddParallelRoute(ka2=0.3, frac=0.4, rationale="Biphasic absorption suspected.")
        new_spec = apply_transform(spec, t)
        assert new_spec.absorption.type == "ParallelFirstOrder"
        assert t.rationale == "Biphasic absorption suspected."

    def test_set_sumig_components_rationale_roundtrip(self) -> None:
        spec = DSLSpec(
            model_id="sumig-base",
            absorption=SumIG(k=2),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={
                "MT_1": 2.0,
                "MT_2": 6.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 0.6,
                "V": 70.0,
                "CL": 5.0,
            },
        )
        t = SetSumIGComponents(
            MT_1=1.0,
            MT_2=5.0,
            RD2_1=0.4,
            RD2_2=0.9,
            weight_1=0.5,
            rationale="Refine SumIG component timing.",
        )
        new_spec = apply_transform(spec, t)
        assert new_spec.initial["MT_1"] == 1.0
        assert t.rationale == "Refine SumIG component timing."

    def test_all_nine_transforms_default_to_empty_provenance(self) -> None:
        """Backward compat: omitting rationale/effect on every transform variant."""
        assert SwapModule(position="elimination", new_module=MichaelisMenten()).rationale == ""
        assert (
            AddCovariateLink(
                param="CL", covariate="WT", form="exponential", theta=0.5
            ).expected_diagnostic_effect
            == []
        )
        assert AdjustVariability(param="CL", action="add").rationale == ""
        assert SetTransitN(n=3).expected_diagnostic_effect == []
        assert ToggleLag(on=True).rationale == ""
        assert (
            ReplaceWithNODE(
                position="absorption", constraint_template="saturable", dim=2
            ).expected_diagnostic_effect
            == []
        )
        assert ConvertTransitToErlang(n=2).rationale == ""
        assert AddParallelRoute(ka2=0.2, frac=0.3).expected_diagnostic_effect == []
        assert (
            SetSumIGComponents(MT_1=1.0, MT_2=2.0, RD2_1=0.1, RD2_2=0.2, weight_1=0.5).rationale
            == ""
        )
