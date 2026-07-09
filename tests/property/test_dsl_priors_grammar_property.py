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
from hypothesis import strategies as st

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


def _mu() -> st.SearchStrategy[float]:
    return st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def _pos_scale() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def valid_target_and_family(draw: st.DrawFn) -> tuple[str, PriorFamily, TargetKind]:
    """Draw a (target, family, target_kind) triple valid under
    ``apmode.dsl.priors._VALID_FAMILIES``, restricted to the families
    ``_family_to_dsl`` can render (Normal, LogNormal, HalfNormal,
    HalfCauchy, LKJ) — Mixture/HistoricalBorrowing/Gamma/InvGamma/Beta are
    covered by the deterministic golden tests in
    ``tests/unit/test_dsl_priors_grammar.py`` instead.
    """
    kind: TargetKind = draw(
        st.sampled_from(["structural", "iiv_sd", "residual_sd", "corr_iiv", "covariate"])
    )

    family: PriorFamily
    if kind == "structural":
        target = draw(st.sampled_from(["CL", "V", "ka"]))
        family = draw(
            st.one_of(
                st.builds(NormalPrior, mu=_mu(), sigma=_pos_scale()),
                st.builds(LogNormalPrior, mu=_mu(), sigma=_pos_scale()),
            )
        )
    elif kind == "iiv_sd":
        target = "omega_CL"
        family = draw(
            st.one_of(
                st.builds(HalfNormalPrior, sigma=_pos_scale()),
                st.builds(HalfCauchyPrior, scale=_pos_scale()),
            )
        )
    elif kind == "residual_sd":
        target = draw(st.sampled_from(["sigma_prop", "sigma_add"]))
        family = draw(st.builds(HalfNormalPrior, sigma=_pos_scale()))
    elif kind == "corr_iiv":
        target = "corr_iiv"
        family = draw(st.builds(LKJPrior, eta=_pos_scale()))
    else:  # covariate
        target = "beta_CL_WT"
        family = draw(st.builds(NormalPrior, mu=_mu(), sigma=_pos_scale()))

    return target, family, kind


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
