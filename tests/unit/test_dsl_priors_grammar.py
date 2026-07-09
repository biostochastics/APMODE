# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the ``priors:`` grammar block (Formular sharpening plan §4 Phase 1, P1.5).

Covers: parsing all ten ``PriorFamily`` variants, the ``log(...)`` numeric
function, block cardinality (zero-or-one top-level, zero-or-more entries),
malformed declarations surfacing ``FrmCode.PRIOR_INVALID_DECLARATION``
(FRM-PRIOR-001), and — the whole point of this task — that every parsed
entry is field-for-field identical to calling
:func:`apmode.dsl.priors.build_prior_spec` directly and to applying the
:class:`~apmode.dsl.prior_transforms.SetPrior` transform with equivalent
arguments. See ``tests/property/test_dsl_priors_grammar_property.py`` for
the property-based counterpart.
"""

from __future__ import annotations

import math

import pytest

from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.errors import FormularCompileError, FrmCode
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.prior_transforms import SetPrior
from apmode.dsl.priors import (
    BetaPrior,
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    build_prior_spec,
    validate_prior_justification,
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
    {priors_section}
}}
"""


def _wrap(entries: str) -> str:
    return _MODEL_TEMPLATE.format(priors_section=f"priors: {{ {entries} }}")


def _base_spec() -> DSLSpec:
    """Structural params CL, V, ka — matches ``_MODEL_TEMPLATE`` exactly."""
    return DSLSpec(
        model_id="priors-grammar-test",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
    )


class TestParsePriorFamilies:
    """All ten PriorFamily variants parse to the equivalent Pydantic object.

    InvGamma/Beta are only reachable as Mixture components — matching
    ``apmode.dsl.priors._VALID_FAMILIES``, where neither appears standalone
    against any target kind — so their coverage lives in the Mixture case.
    """

    def test_normal_on_structural(self) -> None:
        spec = compile_dsl(_wrap("CL ~ Normal(mu=1.5, sigma=0.5)"))
        assert spec.priors == [
            build_prior_spec(target="CL", family=NormalPrior(mu=1.5, sigma=0.5))
        ]

    def test_lognormal_on_structural(self) -> None:
        spec = compile_dsl(_wrap("CL ~ LogNormal(mu=log(4.0), sigma=0.25)"))
        assert spec.priors == [
            build_prior_spec(target="CL", family=LogNormalPrior(mu=math.log(4.0), sigma=0.25))
        ]

    def test_halfnormal_on_iiv_sd(self) -> None:
        spec = compile_dsl(_wrap("omega_CL ~ HalfNormal(sigma=0.3)"))
        assert spec.priors == [
            build_prior_spec(target="omega_CL", family=HalfNormalPrior(sigma=0.3))
        ]

    def test_halfcauchy_on_iiv_sd(self) -> None:
        spec = compile_dsl(_wrap("omega_CL ~ HalfCauchy(scale=0.5)"))
        assert spec.priors == [
            build_prior_spec(target="omega_CL", family=HalfCauchyPrior(scale=0.5))
        ]

    def test_gamma_on_iiv_sd(self) -> None:
        spec = compile_dsl(_wrap("omega_CL ~ Gamma(alpha=2.0, beta=1.0)"))
        assert spec.priors == [
            build_prior_spec(target="omega_CL", family=GammaPrior(alpha=2.0, beta=1.0))
        ]

    def test_lkj_on_corr_iiv(self) -> None:
        spec = compile_dsl(_wrap("corr_iiv ~ LKJ(eta=2.0)"))
        assert spec.priors == [build_prior_spec(target="corr_iiv", family=LKJPrior(eta=2.0))]

    def test_mixture_with_invgamma_and_beta_components(self) -> None:
        text = _wrap(
            "CL ~ Mixture("
            "components=[InvGamma(alpha=2.0, beta=1.0), Beta(alpha=2.0, beta=2.0)], "
            "weights=[0.5, 0.5])"
        )
        spec = compile_dsl(text)
        assert len(spec.priors) == 1
        family = spec.priors[0].family
        assert isinstance(family, MixturePrior)
        assert family.components == [
            InvGammaPrior(alpha=2.0, beta=1.0),
            BetaPrior(alpha=2.0, beta=2.0),
        ]
        assert family.weights == [0.5, 0.5]

    def test_mixture_of_lognormal_and_normal_on_structural(self) -> None:
        text = _wrap(
            "CL ~ Mixture("
            "components=[LogNormal(mu=1.5, sigma=0.3), Normal(mu=0.0, sigma=2.0)], "
            "weights=[0.8, 0.2])"
        )
        spec = compile_dsl(text)
        expected = build_prior_spec(
            target="CL",
            family=MixturePrior(
                components=[
                    LogNormalPrior(mu=1.5, sigma=0.3),
                    NormalPrior(mu=0.0, sigma=2.0),
                ],
                weights=[0.8, 0.2],
            ),
        )
        assert spec.priors == [expected]

    def test_historical_borrowing_on_structural(self) -> None:
        text = _wrap(
            "CL ~ HistoricalBorrowing(map_mean=1.5, map_sd=0.3, robust_weight=0.3, "
            'historical_refs=["phase2_trial"]) '
            "source=historical_data "
            'justification="MAP built from a phase-2 dose-finding study in a matched population." '
            'historical_refs=["phase2_trial"]'
        )
        spec = compile_dsl(text)
        assert len(spec.priors) == 1
        prior = spec.priors[0]
        assert isinstance(prior.family, HistoricalBorrowingPrior)
        assert prior.family.map_mean == 1.5
        assert prior.family.map_sd == 0.3
        assert prior.family.robust_weight == 0.3
        assert prior.family.historical_refs == ["phase2_trial"]
        assert prior.source == "historical_data"
        assert prior.historical_refs == ["phase2_trial"]

    def test_historical_borrowing_robust_weight_defaults_when_omitted(self) -> None:
        text = _wrap(
            'CL ~ HistoricalBorrowing(map_mean=1.5, map_sd=0.3, historical_refs=["phase2_trial"]) '
            "source=historical_data "
            'justification="MAP built from a phase-2 dose-finding study in a matched population." '
            'historical_refs=["phase2_trial"]'
        )
        spec = compile_dsl(text)
        family = spec.priors[0].family
        assert isinstance(family, HistoricalBorrowingPrior)
        assert family.robust_weight == 0.2  # Python-side default


class TestNumericExpression:
    def test_log_matches_python_math_log(self) -> None:
        spec = compile_dsl(_wrap("CL ~ LogNormal(mu=log(4.0), sigma=0.25)"))
        family = spec.priors[0].family
        assert isinstance(family, LogNormalPrior)
        assert family.mu == pytest.approx(math.log(4.0))

    def test_bare_number_unaffected(self) -> None:
        spec = compile_dsl(_wrap("CL ~ Normal(mu=1.5, sigma=0.5)"))
        family = spec.priors[0].family
        assert isinstance(family, NormalPrior)
        assert family.mu == 1.5

    def test_negative_number(self) -> None:
        spec = compile_dsl(_wrap("CL ~ Normal(mu=-1.5, sigma=0.5)"))
        family = spec.priors[0].family
        assert isinstance(family, NormalPrior)
        assert family.mu == -1.5


class TestBlockCardinality:
    def test_no_priors_block_yields_empty_list(self) -> None:
        text = _MODEL_TEMPLATE.format(priors_section="")
        spec = compile_dsl(text)
        assert spec.priors == []

    def test_empty_priors_block_yields_empty_list(self) -> None:
        spec = compile_dsl(_wrap(""))
        assert spec.priors == []

    def test_multiple_entries_in_one_block(self) -> None:
        text = _wrap("CL ~ Normal(mu=1.5, sigma=0.5) V ~ LogNormal(mu=log(35.0), sigma=0.3)")
        spec = compile_dsl(text)
        assert {p.target for p in spec.priors} == {"CL", "V"}

    def test_duplicate_priors_block_raises_ast_duplicate_block(self) -> None:
        text = _MODEL_TEMPLATE.format(
            priors_section=(
                "priors: { CL ~ Normal(mu=1.5, sigma=0.5) } priors: { V ~ Normal(mu=35, sigma=5) }"
            )
        )
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(text)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK


class TestMalformedPriorRaisesFrmPrior001:
    def test_unknown_target(self) -> None:
        text = _wrap("BOGUS ~ Normal(mu=1.5, sigma=0.5)")
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(text)
        assert exc_info.value.code == FrmCode.PRIOR_INVALID_DECLARATION
        assert "BOGUS" in str(exc_info.value)

    def test_wrong_family_for_target_kind(self) -> None:
        # HalfCauchy is only valid for iiv_sd/iov_sd/residual_sd, not structural.
        text = _wrap("CL ~ HalfCauchy(scale=1.0)")
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(text)
        assert exc_info.value.code == FrmCode.PRIOR_INVALID_DECLARATION

    def test_missing_justification_for_informative_source(self) -> None:
        text = _wrap("CL ~ Normal(mu=1.5, sigma=0.5) source=historical_data")
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(text)
        assert exc_info.value.code == FrmCode.PRIOR_INVALID_DECLARATION

    def test_missing_historical_refs_for_historical_data(self) -> None:
        text = _wrap(
            "CL ~ Normal(mu=1.5, sigma=0.5) source=historical_data "
            'justification="A sufficiently long justification string for Gate 2 review purposes."'
        )
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(text)
        assert exc_info.value.code == FrmCode.PRIOR_INVALID_DECLARATION


class TestPriorGrammarFactoryParity:
    """The parity guarantee this task exists to enforce.

    A parsed ``priors:`` entry is field-for-field identical both to calling
    ``build_prior_spec`` directly and to applying ``SetPrior`` through
    ``apply_transform`` — three producers of ``PriorSpec``, one canonical
    factory underneath all three.
    """

    def test_lognormal_historical_prior_matches_direct_factory_call(self) -> None:
        justification = (
            "Derived from a published population-PK meta-analysis of a matched "
            "compound class with comparable dosing and demographics."
        )
        doi = "10.1038/s41586-021-03819-2"
        text = _wrap(
            "CL ~ LogNormal(mu=log(4.0), sigma=0.25) "
            "source=historical_data "
            f'doi="{doi}" '
            f'justification="{justification}" '
            'historical_refs=["nct01234567"]'
        )
        spec = compile_dsl(text)
        parsed = spec.priors[0]

        direct = build_prior_spec(
            target="CL",
            family=LogNormalPrior(mu=math.log(4.0), sigma=0.25),
            source="historical_data",
            justification=justification,
            doi=doi,
            historical_refs=["nct01234567"],
            structural_params=set(spec.structural_param_names()),
        )
        assert parsed == direct

    def test_lognormal_historical_prior_matches_set_prior_transform(self) -> None:
        justification = (
            "Derived from a published population-PK meta-analysis of a matched "
            "compound class with comparable dosing and demographics."
        )
        doi = "10.1038/s41586-021-03819-2"
        text = _wrap(
            "CL ~ LogNormal(mu=log(4.0), sigma=0.25) "
            "source=historical_data "
            f'doi="{doi}" '
            f'justification="{justification}" '
            'historical_refs=["nct01234567"]'
        )
        spec = compile_dsl(text)
        parsed = spec.priors[0]

        base = _base_spec()
        transform = SetPrior(
            target="CL",
            family=LogNormalPrior(mu=math.log(4.0), sigma=0.25),
            source="historical_data",
            justification=justification,
            historical_refs=["nct01234567"],
        )
        new_spec = apply_transform(base, transform)
        applied = next(p for p in new_spec.priors if p.target == "CL")
        # doi does not round-trip through SetPrior's field set (it has no
        # doi kwarg), so compare everything except doi explicitly, then
        # assert doi separately against the grammar-only expectation.
        assert applied.model_copy(update={"doi": doi}) == parsed

    def test_weakly_informative_normal_matches_both_producers(self) -> None:
        text = _wrap("V ~ Normal(mu=35, sigma=5)")
        spec = compile_dsl(text)
        parsed = spec.priors[0]

        direct = build_prior_spec(
            target="V",
            family=NormalPrior(mu=35.0, sigma=5.0),
            structural_params=set(spec.structural_param_names()),
        )
        assert parsed == direct

        base = _base_spec()
        transform = SetPrior(target="V", family=NormalPrior(mu=35.0, sigma=5.0))
        new_spec = apply_transform(base, transform)
        applied = next(p for p in new_spec.priors if p.target == "V")
        assert parsed == applied


class TestEvidenceQualityViaGrammarPath:
    """validate_prior_justification behaves identically regardless of how the
    PriorSpec was constructed — grammar-parsed, or built directly/via SetPrior.
    """

    def test_short_justification_fires_identically(self) -> None:
        doi = "10.1038/s41586-021-03819-2"
        text = _wrap(
            "CL ~ Normal(mu=1.5, sigma=0.3) "
            "source=historical_data "
            f'doi="{doi}" '
            'justification="too short" '
            'historical_refs=["ref1"]'
        )
        spec = compile_dsl(text)
        grammar_prior = spec.priors[0]

        direct_prior = build_prior_spec(
            target="CL",
            family=NormalPrior(mu=1.5, sigma=0.3),
            source="historical_data",
            justification="too short",
            doi=doi,
            historical_refs=["ref1"],
            structural_params=set(spec.structural_param_names()),
        )
        assert grammar_prior == direct_prior

        grammar_errors = validate_prior_justification(grammar_prior)
        direct_errors = validate_prior_justification(direct_prior)
        assert grammar_errors == direct_errors
        assert any("50" in e for e in grammar_errors)

    def test_malformed_doi_fires_identically(self) -> None:
        justification = (
            "A sufficiently long justification string for Gate 2 review, well "
            "past the fifty character minimum length threshold."
        )
        text = _wrap(
            "CL ~ Normal(mu=1.5, sigma=0.3) "
            "source=historical_data "
            'doi="not-a-doi" '
            f'justification="{justification}" '
            'historical_refs=["ref1"]'
        )
        spec = compile_dsl(text)
        grammar_prior = spec.priors[0]

        direct_prior = build_prior_spec(
            target="CL",
            family=NormalPrior(mu=1.5, sigma=0.3),
            source="historical_data",
            justification=justification,
            doi="not-a-doi",
            historical_refs=["ref1"],
            structural_params=set(spec.structural_param_names()),
        )
        assert grammar_prior == direct_prior

        grammar_errors = validate_prior_justification(grammar_prior)
        direct_errors = validate_prior_justification(direct_prior)
        assert grammar_errors == direct_errors
        assert any("DOI" in e for e in grammar_errors)

    def test_compliant_evidence_passes_via_grammar(self) -> None:
        justification = (
            "Derived from a published population-PK meta-analysis of a matched "
            "compound class with comparable dosing and demographics."
        )
        doi = "10.1038/s41586-021-03819-2"
        text = _wrap(
            "CL ~ Normal(mu=1.5, sigma=0.3) "
            "source=historical_data "
            f'doi="{doi}" '
            f'justification="{justification}" '
            'historical_refs=["ref1"]'
        )
        spec = compile_dsl(text)
        errors = validate_prior_justification(spec.priors[0])
        assert errors == []
