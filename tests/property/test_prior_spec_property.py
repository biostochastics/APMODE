# SPDX-License-Identifier: GPL-2.0-or-later
"""Property tests for the canonical PriorSpec factory (Formular DSL P0.4).

Pins two guarantees for `apmode.dsl.priors.build_prior_spec`:

1. It is a pure, deterministic constructor — calling it twice with the same
   field values (including a "round trip" through an already-built
   `PriorSpec`'s own fields) reproduces an equal `PriorSpec`.
2. It is the *only* place `PriorSpec` instances are produced — the `SetPrior`
   transform (`apmode.dsl.prior_transforms.apply_set_prior`, dispatched via
   `apmode.dsl.transforms.apply_transform`) must yield a `PriorSpec` that is
   field-for-field equal to calling `build_prior_spec` directly with the same
   arguments. This is the parity guarantee this task exists to enforce: two
   producers of `PriorSpec`, one canonical factory underneath both.
"""

from __future__ import annotations

from hypothesis import given, settings

from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.prior_transforms import SetPrior
from apmode.dsl.priors import (
    PriorFamily,
    TargetKind,
    build_prior_spec,
)
from apmode.dsl.transforms import apply_transform
from tests.property._strategies import valid_target_and_family


def _base_spec() -> DSLSpec:
    """Structural params: CL, V, ka — matches test_transform_properties.py."""
    return DSLSpec(
        model_id="prior-prop-test",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


class TestBuildPriorSpecRoundtrip:
    @given(data=valid_target_and_family())
    @settings(max_examples=50)
    def test_field_roundtrip_is_equal(self, data: tuple[str, PriorFamily, TargetKind]) -> None:
        """Rebuilding from an existing PriorSpec's own fields reproduces an
        equal PriorSpec (build_prior_spec is a pure function of its inputs).
        """
        target, family, _kind = data
        spec = build_prior_spec(target=target, family=family)

        rebuilt = build_prior_spec(
            target=spec.target,
            family=spec.family,
            source=spec.source,
            justification=spec.justification,
            doi=spec.doi,
            historical_refs=spec.historical_refs,
        )

        assert rebuilt == spec

    @given(data=valid_target_and_family())
    @settings(max_examples=50)
    def test_structural_params_kwarg_does_not_change_result(
        self, data: tuple[str, PriorFamily, TargetKind]
    ) -> None:
        """Supplying a consistent structural_params context (which only adds a
        validation check, not a construction-affecting field) must not change
        the resulting PriorSpec.
        """
        target, family, _kind = data
        without_context = build_prior_spec(target=target, family=family)
        with_context = build_prior_spec(
            target=target,
            family=family,
            structural_params={"CL", "V", "ka"},
        )
        assert without_context == with_context


class TestSetPriorParityWithBuildPriorSpec:
    """The parity guarantee: SetPrior must route through build_prior_spec."""

    @given(data=valid_target_and_family())
    @settings(max_examples=50)
    def test_set_prior_matches_direct_factory_call(
        self, data: tuple[str, PriorFamily, TargetKind]
    ) -> None:
        target, family, _kind = data
        spec = _base_spec()

        transform = SetPrior(target=target, family=family)
        new_spec = apply_transform(spec, transform)
        applied_prior = next(p for p in new_spec.priors if p.target == target)

        direct = build_prior_spec(
            target=target,
            family=family,
            structural_params=set(spec.structural_param_names()),
        )

        assert applied_prior == direct

    @given(data=valid_target_and_family())
    @settings(max_examples=50)
    def test_set_prior_matches_direct_factory_call_with_source_and_justification(
        self, data: tuple[str, PriorFamily, TargetKind]
    ) -> None:
        """Same parity guarantee, but exercising the non-default source /
        justification / historical_refs path (meta_analysis is informative,
        so it requires a non-empty justification — PriorSpec's own invariant).
        """
        target, family, _kind = data
        spec = _base_spec()
        justification = (
            "Derived from a published population-PK meta-analysis of similar compounds."
        )

        transform = SetPrior(
            target=target,
            family=family,
            source="meta_analysis",
            justification=justification,
        )
        new_spec = apply_transform(spec, transform)
        applied_prior = next(p for p in new_spec.priors if p.target == target)

        direct = build_prior_spec(
            target=target,
            family=family,
            source="meta_analysis",
            justification=justification,
            structural_params=set(spec.structural_param_names()),
        )

        assert applied_prior == direct
