# SPDX-License-Identifier: GPL-2.0-or-later
"""Property-based tests for DSL grammar using Hypothesis."""

import lark.exceptions
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lark import Lark

from apmode.dsl.grammar import load_grammar
from tests.property._strategies import OBSERVATIONS
from tests.property._strategies import pos_float as _pos_float
from tests.property._strategies import pos_int as _pos_int

ABSORPTIONS = [
    "FirstOrder(ka)",
    "ZeroOrder(dur)",
    "LaggedFirstOrder(ka, tlag)",
    "Transit(n={n}, ktr, ka)",
    "MixedFirstZero(ka, dur, frac)",
]

DISTRIBUTIONS = [
    "OneCmt(V)",
    "TwoCmt(V1, V2, Q)",
    "ThreeCmt(V1, V2, V3, Q2, Q3)",
    "TMDD_Core(V, R0, kon, koff, kint)",
]

ELIMINATIONS = [
    "Linear(CL)",
    "MichaelisMenten(Vmax, Km)",
    "ParallelLinearMM(CL, Vmax, Km)",
]


@st.composite
def valid_dsl_spec(draw: st.DrawFn) -> str:
    """Generate a syntactically valid DSL model spec."""
    v1, v2 = draw(_pos_float()), draw(_pos_float())
    n = draw(_pos_int())

    abs_template = draw(st.sampled_from(ABSORPTIONS))
    absorption = abs_template.format(n=n)

    dist_template = draw(st.sampled_from(DISTRIBUTIONS))
    distribution = dist_template

    elim_template = draw(st.sampled_from(ELIMINATIONS))
    elimination = elim_template

    obs_template = draw(st.sampled_from(OBSERVATIONS))
    observation = obs_template.format(v=v1, v2=v2)

    params = draw(st.sampled_from(["CL", "V", "ka", "CL, V", "CL, V, ka"]))
    structure = draw(st.sampled_from(["diagonal", "block"]))
    variability = f"IIV(params=[{params}], structure={structure})"

    # Optionally add a covariate link (P1.6: top-level covariates: block,
    # arrow syntax, no longer embedded in variability:).
    add_cov = draw(st.booleans())
    covariates_block = ""
    if add_cov:
        cov_form = draw(st.sampled_from(["power", "exponential", "linear"]))
        if cov_form == "power":
            covariates_block = "covariates: { CL <- WT.power(theta=0.75, ref=70) }"
        else:
            covariates_block = f"covariates: {{ CL <- WT.{cov_form}(theta=0.5) }}"

    return f"""
    model {{
        absorption: {absorption}
        distribution: {distribution}
        elimination: {elimination}
        variability: {variability}
        observation: {observation}
        {covariates_block}
    }}
    """


@pytest.fixture(scope="module")
def parser() -> Lark:
    return load_grammar()


class TestDSLPropertyBased:
    @given(spec=valid_dsl_spec())
    @settings(max_examples=50)
    def test_valid_specs_always_parse(self, spec: str, parser: Lark) -> None:
        """Any generated valid spec must parse without error."""
        tree = parser.parse(spec)
        assert tree is not None
        assert tree.data == "start"

    @given(garbage=st.text(min_size=1, max_size=100))
    @settings(max_examples=30)
    def test_random_text_never_parses_as_model(self, garbage: str, parser: Lark) -> None:
        """Random text must be *rejected* by the parser, and specifically via a
        grammar error (``UnexpectedInput``) — never a stray ``AttributeError``/
        ``RecursionError`` masquerading as "expected". Narrowing the ``except``
        to ``lark.exceptions.UnexpectedInput`` means any other exception type
        propagates and fails the test rather than being silently swallowed.
        """
        try:
            parser.parse(garbage)
        except lark.exceptions.UnexpectedInput:
            return  # expected: grammar rejected the garbage
        pytest.fail(f"Random text parsed as valid model: {garbage!r}")

    def test_model_keyword_garbage_is_rejected(self, parser: Lark) -> None:
        """Garbage that *contains* the ``model {`` opener (the case the old test
        skipped) must still be rejected by the grammar — an unterminated /
        empty model body is not a valid spec.
        """
        # NB: ``model {}`` (empty but well-formed body) parses at the grammar
        # level — emptiness is a *semantic* error caught by validation, not a
        # syntax error — so it is deliberately excluded here.
        for garbage in ("model {", "model { absorption: }", "model { zzz }"):
            with pytest.raises(lark.exceptions.UnexpectedInput):
                parser.parse(garbage)
