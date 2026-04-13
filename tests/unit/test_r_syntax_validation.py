# SPDX-License-Identifier: GPL-2.0-or-later
"""R syntax structural validation tests for nlmixr2 emitter output.

These tests go beyond golden master string matching to verify structural
properties of emitted R code: balanced delimiters, parameter consistency
between ini({}) and model({}) blocks, and basic R syntax validity.
"""

from __future__ import annotations

import re

import pytest

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2


def _make_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": "test_rsyntax_000000",
        "absorption": FirstOrder(ka=1.0),
        "distribution": OneCmt(V=70.0),
        "elimination": LinearElim(CL=5.0),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


# All classical model specs to test structural validity
_ALL_SPECS: list[tuple[str, DSLSpec]] = [
    ("1cmt_fo_linear", _make_spec()),
    (
        "1cmt_fo_mm",
        _make_spec(
            elimination=MichaelisMenten(Vmax=100.0, Km=10.0),
            variability=[IIV(params=["Vmax", "V"], structure="diagonal")],
        ),
    ),
    ("1cmt_fo_parallel_mm", _make_spec(elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0))),
    (
        "1cmt_fo_time_varying",
        _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential")),
    ),
    ("1cmt_zo", _make_spec(absorption=ZeroOrder(dur=0.5))),
    ("1cmt_lagged", _make_spec(absorption=LaggedFirstOrder(ka=1.5, tlag=0.3))),
    ("1cmt_transit", _make_spec(absorption=Transit(n=4, ktr=2.0, ka=1.0))),
    ("1cmt_mixed", _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.6))),
    (
        "2cmt",
        _make_spec(
            distribution=TwoCmt(V1=10.0, V2=20.0, Q=3.0),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        ),
    ),
    (
        "3cmt",
        _make_spec(
            distribution=ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=3.0, Q3=1.0),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_core",
        _make_spec(
            distribution=TMDDCore(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05),
            variability=[IIV(params=["CL", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_qss",
        _make_spec(
            distribution=TMDDQSS(V=50.0, R0=10.0, KD=0.5, kint=0.05),
            variability=[IIV(params=["CL", "R0"], structure="diagonal")],
        ),
    ),
    ("blq_m3", _make_spec(observation=BLQM3(loq_value=0.1))),
    ("blq_m4", _make_spec(observation=BLQM4(loq_value=0.5))),
    ("additive", _make_spec(observation=Additive(sigma_add=1.0))),
    ("combined", _make_spec(observation=Combined(sigma_prop=0.1, sigma_add=0.5))),
    ("block_iiv", _make_spec(variability=[IIV(params=["CL", "V"], structure="block")])),
    (
        "iov",
        _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                IOV(params=["CL"], occasions=OccasionByStudy()),
            ]
        ),
    ),
    (
        "covariate_power",
        _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
            ]
        ),
    ),
    (
        "complex_2cmt_mm_cov",
        _make_spec(
            absorption=LaggedFirstOrder(ka=1.5, tlag=0.3),
            distribution=TwoCmt(V1=30.0, V2=40.0, Q=5.0),
            elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0),
            variability=[
                IIV(params=["CL", "V1", "ka"], structure="block"),
                CovariateLink(param="CL", covariate="WT", form="power"),
                CovariateLink(param="V1", covariate="WT", form="power"),
            ],
            observation=Combined(sigma_prop=0.1, sigma_add=0.5),
        ),
    ),
    (
        "complex_3cmt_transit_blq",
        _make_spec(
            absorption=Transit(n=5, ktr=3.0, ka=1.5),
            distribution=ThreeCmt(V1=15.0, V2=25.0, V3=8.0, Q2=4.0, Q3=1.5),
            elimination=LinearElim(CL=3.5),
            variability=[IIV(params=["CL", "V1", "ktr"], structure="diagonal")],
            observation=BLQM3(loq_value=0.05),
        ),
    ),
]


def _extract_block(r_code: str, block_name: str) -> str:
    """Extract content of ini({...}) or model({...}) block."""
    # Find the block, handling nested braces
    start = r_code.find(block_name + "({")
    if start == -1:
        return ""
    # Find the opening brace after block_name(
    brace_start = r_code.index("{", start)
    depth = 1
    i = brace_start + 1
    while i < len(r_code) and depth > 0:
        if r_code[i] == "{":
            depth += 1
        elif r_code[i] == "}":
            depth -= 1
        i += 1
    return r_code[brace_start + 1 : i - 1]


class TestBalancedDelimiters:
    """All emitted R code must have balanced braces, parens, and brackets."""

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_balanced_braces(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        assert r_code.count("{") == r_code.count("}"), (
            f"Unbalanced braces in {name}: {{ = {r_code.count('{')}, }} = {r_code.count('}')}"
        )

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_balanced_parens(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        assert r_code.count("(") == r_code.count(")"), (
            f"Unbalanced parens in {name}: ( = {r_code.count('(')}, ) = {r_code.count(')')}"
        )


class TestIniModelParamConsistency:
    """Every eta and sigma defined in ini({}) should be used in model({}).
    Every eta used in model({}) should be defined in ini({}).
    """

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_etas_defined_in_ini_used_in_model(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        ini_block = _extract_block(r_code, "ini")
        model_block = _extract_block(r_code, "model")

        # Find eta definitions in ini: "eta.X ~ value" or "eta.X + eta.Y ~ c(...)"
        eta_defs = set(re.findall(r"eta\.(\w+)", ini_block))

        # Find eta usages in model: "+ eta.X" or similar
        eta_uses = set(re.findall(r"eta\.(\w+)", model_block))

        # Every eta defined in ini should appear in the model block
        # (Exception: block IIV etas may appear via combined references)
        for eta in eta_defs:
            assert eta in eta_uses, f"eta.{eta} defined in ini but not used in model for {name}"

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_etas_used_in_model_defined_in_ini(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        ini_block = _extract_block(r_code, "ini")
        model_block = _extract_block(r_code, "model")

        eta_defs = set(re.findall(r"eta\.(\w+)", ini_block))
        eta_uses = set(re.findall(r"eta\.(\w+)", model_block))

        for eta in eta_uses:
            assert eta in eta_defs, f"eta.{eta} used in model but not defined in ini for {name}"


class TestStructuralParamsInIni:
    """Every structural parameter back-transformed in model({}) should have
    an initial estimate in ini({}).
    """

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_log_params_have_ini_definitions(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        ini_block = _extract_block(r_code, "ini")
        model_block = _extract_block(r_code, "model")

        # Find log-domain params defined in ini: "lX <- log(...)"
        ini_log_params = set(re.findall(r"(l\w+)\s*<-\s*log\(", ini_block))

        # Find log-domain params used in model back-transforms: "exp(lX ...)"
        model_log_params = set(re.findall(r"exp\((l\w+)", model_block))

        for lp in model_log_params:
            assert lp in ini_log_params, (
                f"Log-param {lp} used in model but not defined in ini for {name}"
            )


class TestBasicRSyntax:
    """Basic R syntax checks on emitted code."""

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_no_empty_assignments(self, name: str, spec: DSLSpec) -> None:
        """No assignment should have an empty RHS."""
        r_code = emit_nlmixr2(spec)
        for i, line in enumerate(r_code.split("\n"), 1):
            stripped = line.strip()
            if "<-" in stripped and not stripped.startswith("#"):
                rhs = stripped.split("<-", 1)[1].strip()
                assert rhs, f"Empty assignment on line {i}: {stripped!r}"

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_no_double_operators(self, name: str, spec: DSLSpec) -> None:
        """No consecutive arithmetic operators like '+ +' or '- -' or '* /'."""
        r_code = emit_nlmixr2(spec)
        for i, line in enumerate(r_code.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Check for doubled binary ops (excluding ** and <- -X which are valid R)
            assert not re.search(r"(?<!<)[+*/]\s*[+*/]", stripped), (
                f"Double operator on line {i}: {stripped!r}"
            )

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_function_structure(self, name: str, spec: DSLSpec) -> None:
        """Emitted code must have function() { ini({}) model({}) } structure."""
        r_code = emit_nlmixr2(spec)
        assert r_code.strip().startswith("#") or r_code.strip().startswith("function")
        assert "function()" in r_code
        assert "ini({" in r_code
        assert "model({" in r_code
        # ini should come before model
        assert r_code.index("ini({") < r_code.index("model({")
