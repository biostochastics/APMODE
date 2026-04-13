# SPDX-License-Identifier: GPL-2.0-or-later
"""Property-based tests for DSL grammar → AST → lowered R roundtrip.

Verifies that any valid DSL string parses, transforms to a typed AST,
validates, and lowers to R code — the full compiler pipeline.
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

# --- Strategies for generating valid DSL text ---

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

STRUCTURES = ["diagonal", "block"]
COV_FORMS = ["power", "exponential", "linear"]


def _pos_float() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


def _frac_float() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


def _pos_int() -> st.SearchStrategy[int]:
    return st.integers(min_value=1, max_value=20)


# Map from template string prefix to the structural params it defines
_ELIM_PARAMS: dict[str, list[str]] = {
    "Linear(CL": ["CL"],
    "MichaelisMenten(Vmax": ["Vmax", "Km"],
    "ParallelLinearMM(CL": ["CL", "Vmax", "Km"],
}

_DIST_PARAMS: dict[str, list[str]] = {
    "OneCmt(V": ["V"],
    "TwoCmt(V1": ["V1", "V2"],
    "ThreeCmt(V1": ["V1", "V2", "V3"],
}

_ABS_PARAMS: dict[str, list[str]] = {
    "FirstOrder(ka": ["ka"],
    "ZeroOrder(dur": ["dur"],
    "LaggedFirstOrder(ka": ["ka"],
    "Transit(n": ["n", "ktr", "ka"],
    "MixedFirstZero(ka": ["ka"],
}


def _get_params_for_template(text: str, mapping: dict[str, list[str]]) -> list[str]:
    """Extract structural param names from a formatted template string."""
    for prefix, params in mapping.items():
        if text.startswith(prefix):
            return params
    return []


@st.composite
def valid_classical_dsl(draw: st.DrawFn) -> str:
    """Generate a syntactically valid classical (non-NODE) DSL model spec."""
    v1, v2, v3, v4, v5 = [draw(_pos_float()) for _ in range(5)]
    n = draw(_pos_int())
    frac = draw(_frac_float())

    abs_template = draw(st.sampled_from(ABSORPTIONS))
    absorption = abs_template.format(v=v1, v2=v2, v3=frac, n=n)

    dist_template = draw(st.sampled_from(DISTRIBUTIONS))
    distribution = dist_template.format(v=v1, v2=v2, v3=v3, v4=v4, v5=v5)

    elim_template = draw(st.sampled_from(ELIMINATIONS))
    elimination = elim_template.format(v=v1, v2=v2, v3=v3)

    obs_template = draw(st.sampled_from(OBSERVATIONS))
    observation = obs_template.format(v=v1, v2=v2)

    # Build IIV params from actual structural params to pass validation
    struct_params = (
        _get_params_for_template(absorption, _ABS_PARAMS)
        + _get_params_for_template(distribution, _DIST_PARAMS)
        + _get_params_for_template(elimination, _ELIM_PARAMS)
    )
    # Pick 2+ params for block compatibility (need >= 2 for block structure)
    n_params = draw(st.integers(min_value=2, max_value=min(len(struct_params), 4)))
    iiv_params = struct_params[:n_params]
    params = ", ".join(iiv_params)
    structure = draw(st.sampled_from(STRUCTURES))
    variability = f"IIV(params=[{params}], structure={structure})"

    # Optionally add CovariateLink on a valid structural param
    add_cov = draw(st.booleans())
    if add_cov:
        cov_param = draw(st.sampled_from(iiv_params))
        cov_form = draw(st.sampled_from(COV_FORMS))
        variability = f"""{{
            IIV(params=[{params}], structure={structure})
            CovariateLink(param={cov_param}, covariate=WT, form={cov_form})
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
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
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
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "node_lane_admissibility" for e in errors)
