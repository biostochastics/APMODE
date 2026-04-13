# SPDX-License-Identifier: GPL-2.0-or-later
"""Property-based tests for DSL grammar using Hypothesis."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lark import Lark
from lark.exceptions import UnexpectedInput

from apmode.dsl.grammar import load_grammar

ABSORPTIONS = [
    "FirstOrder(ka={v})",
    "ZeroOrder(dur={v})",
    "LaggedFirstOrder(ka={v}, tlag={v2})",
    "Transit(n={n}, ktr={v}, ka={v2})",
    "MixedFirstZero(ka={v}, dur={v2}, frac={v3})",
]

DISTRIBUTIONS = [
    "OneCmt(V={v})",
    "TwoCmt(V1={v}, V2={v2}, Q={v3})",
    "ThreeCmt(V1={v}, V2={v2}, V3={v3}, Q2={v4}, Q3={v5})",
    "TMDD_Core(V={v}, R0={v2}, kon={v3}, koff={v4}, kint={v5})",
]

ELIMINATIONS = [
    "Linear(CL={v})",
    "MichaelisMenten(Vmax={v}, Km={v2})",
    "ParallelLinearMM(CL={v}, Vmax={v2}, Km={v3})",
]

OBSERVATIONS = [
    "Proportional(sigma_prop={v})",
    "Additive(sigma_add={v})",
    "Combined(sigma_prop={v}, sigma_add={v2})",
    "BLQ_M3(loq_value={v})",
    "BLQ_M4(loq_value={v})",
]


def _pos_float() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _pos_int() -> st.SearchStrategy[int]:
    return st.integers(min_value=1, max_value=20)


@st.composite
def valid_dsl_spec(draw: st.DrawFn) -> str:
    """Generate a syntactically valid DSL model spec."""
    v1, v2, v3, v4, v5 = [draw(_pos_float()) for _ in range(5)]
    n = draw(_pos_int())

    abs_template = draw(st.sampled_from(ABSORPTIONS))
    absorption = abs_template.format(v=v1, v2=v2, v3=v3, n=n)

    dist_template = draw(st.sampled_from(DISTRIBUTIONS))
    distribution = dist_template.format(v=v1, v2=v2, v3=v3, v4=v4, v5=v5)

    elim_template = draw(st.sampled_from(ELIMINATIONS))
    elimination = elim_template.format(v=v1, v2=v2, v3=v3)

    obs_template = draw(st.sampled_from(OBSERVATIONS))
    observation = obs_template.format(v=v1, v2=v2)

    params = draw(st.sampled_from(["CL", "V", "ka", "CL, V", "CL, V, ka"]))
    structure = draw(st.sampled_from(["diagonal", "block"]))
    variability = f"IIV(params=[{params}], structure={structure})"

    # Optionally add covariate links for multi-variability tests
    add_cov = draw(st.booleans())
    if add_cov:
        cov_form = draw(st.sampled_from(["power", "exponential", "linear"]))
        variability = f"""{{
            IIV(params=[{params}], structure={structure})
            CovariateLink(param=CL, covariate=WT, form={cov_form})
        }}"""

    return f"""
    model {{
        absorption: {absorption}
        distribution: {distribution}
        elimination: {elimination}
        variability: {variability}
        observation: {observation}
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
        """Random text should not parse as a valid model (with high probability)."""
        if "model" in garbage and "{" in garbage:
            pytest.skip("Contains model-like structure")
        try:
            parser.parse(garbage)
            pytest.fail(f"Random text parsed as valid model: {garbage!r}")
        except (UnexpectedInput, Exception):
            pass  # expected
