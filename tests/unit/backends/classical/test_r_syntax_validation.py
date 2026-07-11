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
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
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
            elimination=MichaelisMenten(),
            variability=[IIV(params=["Vmax", "V"], structure="diagonal")],
        ),
    ),
    ("1cmt_fo_parallel_mm", _make_spec(elimination=ParallelLinearMM())),
    (
        "1cmt_fo_time_varying",
        _make_spec(elimination=TimeVaryingElim(decay_fn="exponential")),
    ),
    ("1cmt_zo", _make_spec(absorption=ZeroOrder())),
    (
        "1cmt_lagged",
        _make_spec(
            absorption=LaggedFirstOrder(), initial={"ka": 1.5, "tlag": 0.3, "V": 70.0, "CL": 5.0}
        ),
    ),
    ("1cmt_transit", _make_spec(absorption=Transit(n=4))),
    ("1cmt_mixed", _make_spec(absorption=MixedFirstZero())),
    (
        "2cmt",
        _make_spec(
            distribution=TwoCmt(),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        ),
    ),
    (
        "3cmt",
        _make_spec(
            distribution=ThreeCmt(),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_core",
        _make_spec(
            distribution=TMDDCore(),
            variability=[IIV(params=["CL", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_qss",
        _make_spec(
            distribution=TMDDQSS(),
            variability=[IIV(params=["CL", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_core_mm",
        _make_spec(
            distribution=TMDDCore(),
            elimination=MichaelisMenten(),
            variability=[IIV(params=["Vmax", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_qss_mm",
        _make_spec(
            distribution=TMDDQSS(),
            elimination=MichaelisMenten(),
            variability=[IIV(params=["Vmax", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_core_parallel_mm",
        _make_spec(
            distribution=TMDDCore(),
            elimination=ParallelLinearMM(),
            variability=[IIV(params=["CL", "R0"], structure="diagonal")],
        ),
    ),
    (
        "tmdd_qss_time_varying",
        _make_spec(
            distribution=TMDDQSS(),
            elimination=TimeVaryingElim(decay_fn="exponential"),
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
            ],
            covariates=[
                CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0),
            ],
        ),
    ),
    (
        "complex_2cmt_mm_cov",
        _make_spec(
            absorption=LaggedFirstOrder(),
            distribution=TwoCmt(),
            elimination=ParallelLinearMM(),
            variability=[
                IIV(params=["CL", "V1", "ka"], structure="block"),
            ],
            covariates=[
                CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0),
                CovariateLink(param="V1", covariate="WT", form="power", theta=1.0, ref=70.0),
            ],
            observation=Combined(sigma_prop=0.1, sigma_add=0.5),
            initial={
                "ka": 1.5,
                "tlag": 0.3,
                "V1": 30.0,
                "V2": 40.0,
                "Q": 5.0,
                "CL": 2.0,
                "Vmax": 50.0,
                "Km": 5.0,
            },
        ),
    ),
    (
        "complex_3cmt_transit_blq",
        _make_spec(
            absorption=Transit(n=5),
            distribution=ThreeCmt(),
            elimination=LinearElim(),
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


class TestNoForwardReferencedLocals:
    """Regression test for the TMDD ``kel <- CL / V`` ordering bug.

    rxode2 parses ``model({})`` sequentially: a bare identifier read on
    the RHS of an assignment before any earlier line defines it is
    silently reclassified as an expected *data covariate* rather than a
    model-local (confirmed via ``nlmixr2est:::.nlmixr0preProcessCovariatesPresent``),
    which breaks every fit with an opaque "missing data column" error
    instead of a parse failure. Golden-snapshot text comparisons don't
    catch this since the buggy ordering is just as textually stable as
    the fixed ordering. This test tracks line-by-line definitions
    (``name <- ...``, ``d/dt(name) <- ...``, ``name(0) <- ...``) and
    flags any structural-parameter identifier referenced before its own
    defining line.
    """

    # Identifiers legitimately usable before any local "<-" definition:
    # rxode2/R builtins that are never assigned via "<-" in the model
    # block. Function calls (any identifier immediately followed by
    # "(", e.g. exp(...), sqrt(...), transit(...)) are recognized
    # structurally in ``_assert_rhs_defined`` rather than enumerated here.
    _KNOWN_SAFE = {
        "t",
        "amt",
    }

    @staticmethod
    def _covariate_names(spec: DSLSpec) -> set[str]:
        """Covariate columns are legitimately data-sourced, not locally defined."""
        return {cov.covariate for cov in spec.covariates}

    @staticmethod
    def _ini_defined_names(ini_block: str) -> set[str]:
        """Names ``ini({})`` declares as THETA/OMEGA/SIGMA parameters.

        These are true model parameters (not sequential locals): unlike
        ``model({})``, order inside ``ini({})`` doesn't matter and every
        name declared there (log-domain THETAs like ``lCL``, direct
        covariate coefficients like ``beta_CL_WT``, residual terms like
        ``prop.sd``) is available anywhere in ``model({})`` regardless of
        textual position.
        """
        return set(re.findall(r"^\s*(\w+(?:\.\w+)*)\s*<-", ini_block, re.MULTILINE))

    @pytest.mark.parametrize("name,spec", _ALL_SPECS, ids=[s[0] for s in _ALL_SPECS])
    def test_structural_params_defined_before_use(self, name: str, spec: DSLSpec) -> None:
        r_code = emit_nlmixr2(spec)
        ini_block = _extract_block(r_code, "ini")
        model_block = _extract_block(r_code, "model")
        safe = self._KNOWN_SAFE | self._covariate_names(spec) | {"eta", "cp"}

        # ODE states are mutually visible regardless of textual d/dt()
        # order (they're simultaneous, not sequential locals) — e.g.
        # `d/dt(depot) <- -ka * depot` legitimately self-references.
        # Only bare algebraic locals (kel, L, Ctot, Cfree, ...) assigned
        # via plain `<-` are subject to rxode2's strict sequential rule,
        # which is the category the TMDD `kel` bug fell into.
        state_names = set(re.findall(r"d/dt\((\w+)\)", model_block))
        state_names |= set(re.findall(r"(\w+)\(0\)\s*<-", model_block))

        defined: set[str] = self._ini_defined_names(ini_block) | state_names
        for line in model_block.split("\n"):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue

            state_match = re.match(r"d/dt\((\w+)\)\s*<-\s*(.+)", stripped)
            init_match = re.match(r"(\w+)\(0\)\s*<-", stripped)
            obs_match = re.match(r"\w+\s*~\s*", stripped)
            assign_match = re.match(r"(\w+(?:\.\w+)*)\s*<-\s*(.+)", stripped)

            if state_match:
                _lhs, rhs = state_match.groups()
                self._assert_rhs_defined(rhs, defined, safe, name, stripped)
            elif init_match or obs_match:
                continue
            elif assign_match:
                lhs, rhs = assign_match.groups()
                self._assert_rhs_defined(rhs, defined, safe, name, stripped)
                defined.add(lhs.split(".")[0])
                defined.add(lhs)

    @staticmethod
    def _assert_rhs_defined(
        rhs: str, defined: set[str], safe: set[str], case_name: str, context: str
    ) -> None:
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_.]*\s*(\()?", rhs):
            ident = match.group(0).rstrip("( ").rstrip()
            if match.group(1):
                continue  # function call (exp(...), sqrt(...), transit(...), ...)
            base = ident.split(".")[0]
            if ident in safe or base in safe or ident in defined or base in defined:
                continue
            raise AssertionError(
                f"[{case_name}] identifier {ident!r} used before it is defined "
                f"anywhere earlier in the model({{}}) block: {context!r}"
            )
