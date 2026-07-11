# SPDX-License-Identifier: GPL-2.0-or-later
"""Property tests for the ``priors:`` grammar block (Formular sharpening plan
§4 Phase 1, P1.5).

Pins the parity guarantee across a wide sample of (target, family) pairs: a
Formular-text-parsed ``PriorSpec`` is field-for-field identical to calling
:func:`apmode.dsl.priors.build_prior_spec` directly with equivalent
arguments, and to applying the
:class:`~apmode.dsl.prior_transforms.SetPrior` transform to a spec with a
matching structural-parameter universe. Mirrors the strategy shape of
``tests/property/test_prior_spec_property.py`` (which pins the same
guarantee for the ``SetPrior`` vs. ``build_prior_spec`` pair) so the two
tests read as one continuous proof: grammar -> SetPrior -> build_prior_spec
all agree.
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
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.prior_transforms import SetPrior
from apmode.dsl.priors import (
    HalfCauchyPrior,
    HalfNormalPrior,
    LKJPrior,
    LogNormalPrior,
    NormalPrior,
    PriorFamily,
    TargetKind,
    build_prior_spec,
)
from apmode.dsl.transforms import apply_transform
from tests.property._strategies import valid_target_and_family

_MODEL_TEMPLATE = """
model {{
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: {{ ka = 1.0, V = 70.0, CL = 5.0 }}
    priors: {{ {target} ~ {family_dsl} }}
}}
"""


def _base_spec() -> DSLSpec:
    """Structural params CL, V, ka — matches ``_MODEL_TEMPLATE`` exactly."""
    return DSLSpec(
        model_id="priors-grammar-prop-test",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
    )


def _family_to_dsl(family: PriorFamily) -> str:
    """Render a PriorFamily instance to the equivalent Formular text.

    Uses ``str(float)`` (Python's shortest round-trip repr), which the
    grammar's ``NUMBER`` terminal accepts unmodified (including its
    scientific-notation form, e.g. ``1e-05``) — the same convention already
    used by ``tests/property/test_dsl_property.py``.
    """
    if isinstance(family, NormalPrior):
        return f"Normal(mu={family.mu}, sigma={family.sigma})"
    if isinstance(family, LogNormalPrior):
        return f"LogNormal(mu={family.mu}, sigma={family.sigma})"
    if isinstance(family, HalfNormalPrior):
        return f"HalfNormal(sigma={family.sigma})"
    if isinstance(family, HalfCauchyPrior):
        return f"HalfCauchy(scale={family.scale})"
    if isinstance(family, LKJPrior):
        return f"LKJ(eta={family.eta})"
    raise TypeError(
        f"unsupported family for this property test: {family.type}"
    )  # pragma: no cover


class TestGrammarMatchesBuildPriorSpec:
    @given(data=valid_target_and_family())
    @settings(max_examples=75)
    def test_parsed_prior_equals_direct_factory_call(
        self, data: tuple[str, PriorFamily, TargetKind]
    ) -> None:
        target, family, _kind = data
        text = _MODEL_TEMPLATE.format(target=target, family_dsl=_family_to_dsl(family))
        spec = compile_dsl(text)
        parsed = next(p for p in spec.priors if p.target == target)

        direct = build_prior_spec(
            target=target,
            family=family,
            structural_params=set(spec.structural_param_names()),
        )
        assert parsed == direct


class TestGrammarMatchesSetPriorTransform:
    @given(data=valid_target_and_family())
    @settings(max_examples=75)
    def test_parsed_prior_equals_set_prior_applied(
        self, data: tuple[str, PriorFamily, TargetKind]
    ) -> None:
        target, family, _kind = data
        text = _MODEL_TEMPLATE.format(target=target, family_dsl=_family_to_dsl(family))
        spec = compile_dsl(text)
        parsed = next(p for p in spec.priors if p.target == target)

        base = _base_spec()
        transform = SetPrior(target=target, family=family)
        new_spec = apply_transform(base, transform)
        applied = next(p for p in new_spec.priors if p.target == target)

        assert parsed == applied
