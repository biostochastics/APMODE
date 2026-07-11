# SPDX-License-Identifier: GPL-2.0-or-later
"""Property-based tests for DSL grammar → AST → lowered R roundtrip.

Verifies that any valid DSL string parses, transforms to a typed AST,
validates, and lowers to R code — the full compiler pipeline.

Formular sharpening plan §4 Phase 1 (P1.4): structural declarations name
calibration parameters without values; all calibration values live in the
top-level ``initial: { ... }`` block.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lark import Lark

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    LinearElim,
    NODEAbsorption,
    OneCmt,
    Proportional,
)
from apmode.dsl.grammar import compile_dsl, load_grammar
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.validator import validate_dsl
from tests.property._strategies import OBSERVATIONS
from tests.property._strategies import pos_float as _pos_float
from tests.property._strategies import pos_int as _pos_int

# --- Strategies for generating valid DSL text ---

# (structural declaration template, calibration param names it introduces).
# Templates with a "{n}" placeholder carry a structural (non-calibration)
# integer that stays inline; every other name is a bare keyword resolved
# via the top-level initial: block.
ABSORPTIONS: list[tuple[str, list[str]]] = [
    ("FirstOrder(ka)", ["ka"]),
    ("ZeroOrder(dur)", ["dur"]),
    ("LaggedFirstOrder(ka, tlag)", ["ka", "tlag"]),
    ("Transit(n={n}, ktr, ka)", ["ktr", "ka"]),
    ("MixedFirstZero(ka, dur, frac)", ["ka", "dur", "frac"]),
]

DISTRIBUTIONS: list[tuple[str, list[str]]] = [
    ("OneCmt(V)", ["V"]),
    ("TwoCmt(V1, V2, Q)", ["V1", "V2", "Q"]),
    ("ThreeCmt(V1, V2, V3, Q2, Q3)", ["V1", "V2", "V3", "Q2", "Q3"]),
]

ELIMINATIONS: list[tuple[str, list[str]]] = [
    ("Linear(CL)", ["CL"]),
    ("MichaelisMenten(Vmax, Km)", ["Vmax", "Km"]),
    ("ParallelLinearMM(CL, Vmax, Km)", ["CL", "Vmax", "Km"]),
]

STRUCTURES = ["diagonal", "block"]
COV_FORMS = ["power", "exponential", "linear"]

_FRAC_PARAMS = frozenset({"frac"})
_NON_NEGATIVE_PARAMS = frozenset({"tlag"})


def _frac_float() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


def _non_negative_float() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)


def _draw_value_for_param(draw: st.DrawFn, name: str) -> float:
    if name in _FRAC_PARAMS:
        return draw(_frac_float())
    if name in _NON_NEGATIVE_PARAMS:
        return draw(_non_negative_float())
    return draw(_pos_float())


@st.composite
def valid_classical_dsl(draw: st.DrawFn) -> str:
    """Generate a syntactically valid classical (non-NODE) DSL model spec."""
    v1, v2 = draw(_pos_float()), draw(_pos_float())
    n = draw(_pos_int())

    abs_template, abs_params = draw(st.sampled_from(ABSORPTIONS))
    absorption = abs_template.format(n=n)

    dist_template, dist_params = draw(st.sampled_from(DISTRIBUTIONS))
    distribution = dist_template

    elim_template, elim_params = draw(st.sampled_from(ELIMINATIONS))
    elimination = elim_template

    obs_template = draw(st.sampled_from(OBSERVATIONS))
    observation = obs_template.format(v=v1, v2=v2)

    # Deduplicate while preserving order — the three axes never share a
    # calibration name in this generator's template set.
    struct_params = list(dict.fromkeys([*abs_params, *dist_params, *elim_params]))
    initial_values = {p: _draw_value_for_param(draw, p) for p in struct_params}
    initial_text = ", ".join(f"{p} = {v}" for p, v in initial_values.items())

    # Pick 2+ params for block compatibility (need >= 2 for block structure)
    n_params = draw(st.integers(min_value=2, max_value=min(len(struct_params), 4)))
    iiv_params = struct_params[:n_params]
    params = ", ".join(iiv_params)
    structure = draw(st.sampled_from(STRUCTURES))
    variability = f"IIV(params=[{params}], structure={structure})"

    # Optionally add a covariate link on a valid structural param (P1.6:
    # top-level covariates: block, arrow syntax, not embedded in variability).
    add_cov = draw(st.booleans())
    covariates_block = ""
    if add_cov:
        cov_param = draw(st.sampled_from(iiv_params))
        cov_form = draw(st.sampled_from(COV_FORMS))
        cov_theta = draw(_pos_float())
        if cov_form == "power":
            cov_ref = draw(_pos_float())
            covariates_block = (
                f"covariates: {{ {cov_param} <- WT.power(theta={cov_theta}, ref={cov_ref}) }}"
            )
        else:
            covariates_block = f"covariates: {{ {cov_param} <- WT.{cov_form}(theta={cov_theta}) }}"

    return f"""
    model {{
        absorption: {absorption}
        distribution: {distribution}
        elimination: {elimination}
        variability: {variability}
        observation: {observation}
        initial: {{ {initial_text} }}
        {covariates_block}
    }}
    """


@pytest.fixture(scope="module")
def parser() -> Lark:
    return load_grammar()


class TestFullPipelineRoundtrip:
    """Grammar → Parse Tree → AST → Validation → R code: full roundtrip."""

    @given(spec_text=valid_classical_dsl())
    @settings(max_examples=100)
    def test_parse_transform_validate_emit(self, spec_text: str) -> None:
        """Any generated valid classical spec should compile and lower to R."""
        # 1. Parse + Transform
        spec = compile_dsl(spec_text)
        assert isinstance(spec, DSLSpec)

        # 2. Validate (should pass for valid specs)
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == [], f"Unexpected validation errors: {errors}"

        # 3. Lower to R
        r_code = emit_nlmixr2(spec)
        assert isinstance(r_code, str)
        assert len(r_code) > 0

        # 4. R code structural checks
        assert "function()" in r_code
        assert "ini({" in r_code
        assert "model({" in r_code

        # 5. All structural params appear in emitted R
        for param in spec.structural_param_names():
            assert param in r_code, f"Param '{param}' not in R code"

    @given(spec_text=valid_classical_dsl())
    @settings(max_examples=50)
    def test_ast_json_roundtrip(self, spec_text: str) -> None:
        """DSLSpec should survive JSON serialization roundtrip."""
        spec = compile_dsl(spec_text)
        json_data = spec.model_dump()
        roundtripped = DSLSpec.model_validate(json_data)

        assert roundtripped.absorption == spec.absorption
        assert roundtripped.distribution == spec.distribution
        assert roundtripped.elimination == spec.elimination
        assert roundtripped.observation == spec.observation
        assert len(roundtripped.variability) == len(spec.variability)
        assert roundtripped.covariates == spec.covariates
        assert roundtripped.initial == spec.initial

    @given(spec_text=valid_classical_dsl())
    @settings(max_examples=50)
    def test_no_node_modules_in_classical(self, spec_text: str) -> None:
        """Classical specs should never have NODE modules."""
        spec = compile_dsl(spec_text)
        assert not spec.has_node_modules()
        assert spec.node_max_dim() == 0

    @given(spec_text=valid_classical_dsl())
    @settings(max_examples=50)
    def test_structural_params_non_empty(self, spec_text: str) -> None:
        """Every compiled spec should have at least one structural param."""
        spec = compile_dsl(spec_text)
        assert len(spec.structural_param_names()) > 0


class TestNODEValidationProperties:
    """Property-based tests for NODE constraint enforcement."""

    @given(
        dim=st.integers(min_value=1, max_value=20),
        template=st.sampled_from(
            [
                "monotone_increasing",
                "monotone_decreasing",
                "bounded_positive",
                "saturable",
                "unconstrained_smooth",
            ]
        ),
        lane=st.sampled_from([Lane.DISCOVERY, Lane.OPTIMIZATION]),
    )
    @settings(max_examples=100)
    def test_node_dim_constraints_consistent(self, dim: int, template: str, lane: Lane) -> None:
        """NODE validation should be consistent: either passes all checks or fails."""
        from apmode.dsl.validator import _LANE_DIM_CEILING, _TEMPLATE_MAX_DIM

        spec = DSLSpec(
            model_id="test_node_property",
            absorption=NODEAbsorption(dim=dim, constraint_template=template),  # type: ignore[arg-type]
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"V": 70.0, "CL": 5.0},
        )
        errors = validate_dsl(spec, lane=lane)

        template_max = _TEMPLATE_MAX_DIM[template]
        lane_ceiling = _LANE_DIM_CEILING[lane]

        # Verify error presence matches constraint violations
        template_exceeded = dim > template_max
        lane_exceeded = lane_ceiling is not None and dim > lane_ceiling

        template_errors = [e for e in errors if e.constraint == "node_template_max_dim"]
        lane_errors = [e for e in errors if e.constraint == "node_lane_dim_ceiling"]

        assert bool(template_errors) == template_exceeded
        assert bool(lane_errors) == lane_exceeded

    @given(
        dim=st.integers(min_value=1, max_value=4),
        template=st.sampled_from(["monotone_increasing", "saturable"]),
    )
    @settings(max_examples=20)
    def test_node_always_rejected_in_submission(self, dim: int, template: str) -> None:
        """NODE modules are never admissible in Submission lane."""
        spec = DSLSpec(
            model_id="test_node_submission",
            absorption=NODEAbsorption(dim=dim, constraint_template=template),  # type: ignore[arg-type]
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"V": 70.0, "CL": 5.0},
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "node_lane_admissibility" for e in errors)
