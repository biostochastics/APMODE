# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for structured FRM-{TAXON}-NNN validator error codes (P0.2).

Verifies that :func:`apmode.dsl.validator.validate_dsl` attaches the
correct :class:`~apmode.dsl.errors.FrmCode` to every violation, that
``severity``/``remediation`` are populated sensibly, and that every code
the validator (or, for block-cardinality/macro-use codes, the grammar
compiler / ``apmode.dsl.macros``) can emit is documented in
``docs/FORMULAR_ERROR_CODES.md`` (the canonical registry).
"""

from __future__ import annotations

import re
from pathlib import Path

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    IIV,
    IOV,
    DSLSpec,
    Erlang,
    ExperimentalFlags,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    NODEAbsorption,
    NODEElimination,
    OccasionByStudy,
    OneCmt,
    Proportional,
    SumIG,
    TMDDCore,
    Transit,
)
from apmode.dsl.errors import FrmCode
from apmode.dsl.priors import PriorSpec, default_structural_prior
from apmode.dsl.validator import validate_dsl

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATOR_SRC = _REPO_ROOT / "src" / "apmode" / "dsl" / "validator.py"
# FRM-AST-010/011 (block cardinality) are raised by compile_dsl on the raw
# parse tree, before a DSLSpec exists — not by validate_dsl — so the coded-
# error registry scan below covers both source files.
_GRAMMAR_SRC = _REPO_ROOT / "src" / "apmode" / "dsl" / "grammar.py"
# FRM-AST-016/017 (unknown/duplicate `use` macro) are raised by
# apmode.dsl.macros.expand_macros, a third call site for
# FormularCompileError alongside grammar.py's block-cardinality checks.
_MACROS_SRC = _REPO_ROOT / "src" / "apmode" / "dsl" / "macros" / "__init__.py"
_ERROR_CODE_DOC = _REPO_ROOT / "docs" / "FORMULAR_ERROR_CODES.md"


def _make_spec(**overrides: object) -> DSLSpec:
    """Build a valid baseline DSLSpec, overriding specific modules."""
    defaults: dict[str, object] = {
        "model_id": "test_id_000000000000",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
        "initial": {"ka": 1.0, "V": 70.0, "CL": 5.0},
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


def _code_for(spec: DSLSpec, lane: Lane, constraint: str) -> str:
    errors = validate_dsl(spec, lane=lane)
    matches = [e for e in errors if e.constraint == constraint]
    assert len(matches) == 1, (
        f"expected exactly one {constraint!r} violation, got {matches} (all errors: {errors})"
    )
    return str(matches[0].code)


class TestFrmSemCodes:
    """FRM-SEM-*: semantic / numeric constraint violations."""

    def test_positive_violation_carries_sem_001(self) -> None:
        spec = _make_spec(initial={"ka": -1.0, "V": 70.0, "CL": 5.0})
        assert _code_for(spec, Lane.SUBMISSION, "positive") == FrmCode.SEM_POSITIVE

    def test_non_negative_violation_carries_sem_002(self) -> None:
        spec = _make_spec(
            absorption=LaggedFirstOrder(),
            initial={"ka": 1.0, "tlag": -0.5, "V": 70.0, "CL": 5.0},
        )
        assert _code_for(spec, Lane.SUBMISSION, "non_negative") == FrmCode.SEM_NON_NEGATIVE

    def test_unit_interval_violation_carries_sem_003(self) -> None:
        # SumIG.weight_1 exists only for k=2; fixed disposition priors keep
        # this fixture focused on the unit-interval error.
        spec = _make_spec(
            absorption=SumIG(k=2),
            initial={
                "MT_1": 2.0,
                "MT_2": 6.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 1.5,
                "V": 70.0,
                "CL": 5.0,
            },
            priors=[
                PriorSpec(target="CL", family=default_structural_prior(), source="fixed_external"),
                PriorSpec(target="V", family=default_structural_prior(), source="fixed_external"),
            ],
        )
        assert _code_for(spec, Lane.DISCOVERY, "unit_interval") == FrmCode.SEM_UNIT_INTERVAL

    def test_positive_int_violation_carries_sem_004(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=0),
            initial={"ktr": 1.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        assert _code_for(spec, Lane.SUBMISSION, "positive_int") == FrmCode.SEM_POSITIVE_INT

    def test_erlang_max_n_violation_carries_sem_005(self) -> None:
        spec = _make_spec(
            absorption=Erlang(n=8),
            initial={"ktr": 1.0, "V": 70.0, "CL": 5.0},
        )
        assert _code_for(spec, Lane.DISCOVERY, "erlang_max_n") == FrmCode.SEM_ERLANG_MAX_N

    def test_sumig_k_range_violation_carries_sem_006(self) -> None:
        spec = _make_spec(
            absorption=SumIG(k=3),
            initial={
                "MT_1": 2.0,
                "MT_2": 6.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 0.6,
                "V": 70.0,
                "CL": 5.0,
            },
        )
        assert _code_for(spec, Lane.DISCOVERY, "sumig_k_range") == FrmCode.SEM_SUMIG_K_RANGE

    def test_sumig_mt_ordering_violation_carries_sem_007(self) -> None:
        spec = _make_spec(
            absorption=SumIG(k=2),
            initial={
                "MT_1": 6.0,
                "MT_2": 2.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 0.6,
                "V": 70.0,
                "CL": 5.0,
            },
            priors=[
                PriorSpec(target="CL", family=default_structural_prior(), source="fixed_external"),
                PriorSpec(target="V", family=default_structural_prior(), source="fixed_external"),
            ],
        )
        assert (
            _code_for(spec, Lane.DISCOVERY, "sumig_mt_ordering") == FrmCode.SEM_SUMIG_MT_ORDERING
        )

    def test_sumig_disposition_fixed_violation_carries_sem_008(self) -> None:
        spec = _make_spec(
            absorption=SumIG(k=2),
            initial={
                "MT_1": 2.0,
                "MT_2": 6.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 0.6,
                "V": 70.0,
                "CL": 5.0,
            },
        )
        assert (
            _code_for(spec, Lane.DISCOVERY, "sumig_disposition_fixed")
            == FrmCode.SEM_SUMIG_DISPOSITION_FIXED
        )

    def test_node_template_max_dim_violation_carries_sem_009(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=6, constraint_template="monotone_increasing")
        )
        assert (
            _code_for(spec, Lane.DISCOVERY, "node_template_max_dim")
            == FrmCode.SEM_NODE_TEMPLATE_MAX_DIM
        )


class TestFrmAstCodes:
    """FRM-AST-*: AST-shape / structural-integrity errors."""

    def test_iiv_duplicate_params_carries_ast_001(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["CL", "V"], structure="diagonal"),
            ]
        )
        assert (
            _code_for(spec, Lane.SUBMISSION, "iiv_no_duplicate_params")
            == FrmCode.AST_IIV_NO_DUPLICATE_PARAMS
        )

    def test_non_empty_iiv_params_carries_ast_003(self) -> None:
        spec = _make_spec(variability=[IIV(params=[], structure="diagonal")])
        assert _code_for(spec, Lane.SUBMISSION, "non_empty") == FrmCode.AST_NON_EMPTY_PARAMS

    def test_block_min_params_carries_ast_004(self) -> None:
        spec = _make_spec(variability=[IIV(params=["CL"], structure="block")])
        assert _code_for(spec, Lane.SUBMISSION, "block_min_params") == FrmCode.AST_BLOCK_MIN_PARAMS

    def test_iiv_param_exists_carries_ast_005(self) -> None:
        spec = _make_spec(variability=[IIV(params=["nonexistent"], structure="diagonal")])
        assert _code_for(spec, Lane.SUBMISSION, "iiv_param_exists") == FrmCode.AST_IIV_PARAM_EXISTS

    def test_iov_param_exists_carries_ast_006(self) -> None:
        spec = _make_spec(variability=[IOV(params=["nonexistent"], occasions=OccasionByStudy())])
        assert _code_for(spec, Lane.SUBMISSION, "iov_param_exists") == FrmCode.AST_IOV_PARAM_EXISTS

    def test_no_variability_on_param_carries_ast_008(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=3),
            initial={"ktr": 1.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
            variability=[IIV(params=["n"], structure="diagonal")],
        )
        assert (
            _code_for(spec, Lane.SUBMISSION, "no_variability_on_param")
            == FrmCode.AST_NO_VARIABILITY_ON_PARAM
        )

    def test_tmdd_rejects_node_elim_carries_ast_009(self) -> None:
        from apmode.dsl.ast_models import ExperimentalFlags, NODEElimination

        spec = _make_spec(
            distribution=TMDDCore(),
            elimination=NODEElimination(dim=2, constraint_template="bounded_positive"),
            initial={
                "ka": 1.0,
                "V": 10.0,
                "R0": 1.0,
                "kon": 0.1,
                "koff": 0.01,
                "kint": 0.05,
            },
            variability=[IIV(params=["V"], structure="diagonal")],
        )
        spec = spec.model_copy(update={"experimental": ExperimentalFlags(node=True)})
        assert (
            _code_for(spec, Lane.SUBMISSION, "tmdd_rejects_node_elim")
            == FrmCode.AST_TMDD_REQUIRES_LINEAR_ELIM
        )

    def test_initial_value_missing_carries_ast_012(self) -> None:
        spec = _make_spec(initial={"V": 70.0, "CL": 5.0})  # missing "ka"
        assert (
            _code_for(spec, Lane.SUBMISSION, "initial_value_missing")
            == FrmCode.AST_INITIAL_VALUE_MISSING
        )

    def test_initial_value_unused_carries_ast_013(self) -> None:
        spec = _make_spec(initial={"ka": 1.0, "V": 70.0, "CL": 5.0, "extra": 1.0})
        assert (
            _code_for(spec, Lane.SUBMISSION, "initial_value_unused")
            == FrmCode.AST_INITIAL_VALUE_UNUSED
        )


class TestFrmLaneCodes:
    """FRM-LANE-*: lane-admissibility rejections."""

    def test_node_lane_admissibility_carries_lane_001(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="monotone_increasing")
        )
        assert (
            _code_for(spec, Lane.SUBMISSION, "node_lane_admissibility")
            == FrmCode.LANE_NODE_ADMISSIBILITY
        )

    def test_node_lane_dim_ceiling_carries_lane_002(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=6, constraint_template="bounded_positive")
        )
        assert (
            _code_for(spec, Lane.OPTIMIZATION, "node_lane_dim_ceiling")
            == FrmCode.LANE_NODE_DIM_CEILING
        )

    def test_lane_absorption_admissibility_carries_lane_003(self) -> None:
        spec = _make_spec(
            absorption=SumIG(k=2),
            initial={
                "MT_1": 2.0,
                "MT_2": 6.0,
                "RD2_1": 0.5,
                "RD2_2": 1.0,
                "weight_1": 0.6,
                "V": 70.0,
                "CL": 5.0,
            },
            priors=[
                PriorSpec(target="CL", family=default_structural_prior(), source="fixed_external"),
                PriorSpec(target="V", family=default_structural_prior(), source="fixed_external"),
            ],
        )
        assert (
            _code_for(spec, Lane.SUBMISSION, "lane_absorption_admissibility")
            == FrmCode.LANE_ABSORPTION_ADMISSIBILITY
        )

    def test_node_experimental_gate_carries_lane_004(self) -> None:
        """NODE variant without ``experimental.node`` opt-in fails, any lane (P0.8)."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="monotone_increasing")
        )
        assert (
            _code_for(spec, Lane.DISCOVERY, "node_experimental_gate")
            == FrmCode.LANE_NODE_EXPERIMENTAL_GATE
        )

    def test_node_experimental_gate_opt_in_suppresses_check(self) -> None:
        """``experimental.node=True`` opts out of FRM-LANE-004 specifically.

        The spec may still fail other checks (e.g. lane admissibility) --
        this only asserts the experimental-gate check itself no longer
        fires, matching the P0.8 requirement.
        """
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="monotone_increasing"),
            experimental=ExperimentalFlags(node=True),
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert not any(e.constraint == "node_experimental_gate" for e in errors)

    def test_node_experimental_gate_fires_independent_of_lane(self) -> None:
        """Unlike FRM-LANE-001 (Submission-only), the experimental gate fires everywhere."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=2, constraint_template="monotone_increasing")
        )
        for lane in (Lane.SUBMISSION, Lane.DISCOVERY, Lane.OPTIMIZATION):
            errors = validate_dsl(spec, lane=lane)
            assert any(
                e.constraint == "node_experimental_gate"
                and e.code == FrmCode.LANE_NODE_EXPERIMENTAL_GATE
                for e in errors
            ), f"expected node_experimental_gate to fire in {lane}"


class TestSeverityAndRemediation:
    """Every error defaults to severity='error'; clear-fix checks carry remediation."""

    def test_default_severity_is_error(self) -> None:
        spec = _make_spec(initial={"ka": -1.0, "V": 70.0, "CL": 5.0})
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].severity == "error"

    def test_positive_violation_has_actionable_remediation(self) -> None:
        spec = _make_spec(initial={"ka": -1.0, "V": 70.0, "CL": 5.0})
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors[0].remediation is not None
        assert "> 0" in errors[0].remediation

    def test_no_variability_on_param_has_remove_remediation(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=3),
            initial={"ktr": 1.0, "ka": 1.0, "V": 70.0, "CL": 5.0},
            variability=[IIV(params=["n"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        matches = [e for e in errors if e.constraint == "no_variability_on_param"]
        assert len(matches) == 1
        assert matches[0].remediation is not None
        assert "Remove" in matches[0].remediation


class TestErrorCodeDocRegistry:
    """Every FrmCode referenced in validator.py/grammar.py must be documented."""

    def test_doc_file_exists(self) -> None:
        assert _ERROR_CODE_DOC.is_file(), (
            f"canonical FRM code registry missing at {_ERROR_CODE_DOC}"
        )

    def _combined_src(self) -> str:
        return (
            _VALIDATOR_SRC.read_text(encoding="utf-8")
            + _GRAMMAR_SRC.read_text(encoding="utf-8")
            + _MACROS_SRC.read_text(encoding="utf-8")
        )

    def test_every_frm_code_used_in_validator_is_documented(self) -> None:
        combined_src = self._combined_src()
        doc_src = _ERROR_CODE_DOC.read_text(encoding="utf-8")

        used_members = {
            member.name
            for member in FrmCode
            if re.search(rf"\bFrmCode\.{re.escape(member.name)}\b", combined_src)
        }
        assert used_members, "no FrmCode members found referenced in validator.py/grammar.py"

        undocumented = [
            member.value
            for member in FrmCode
            if member.name in used_members and member.value not in doc_src
        ]
        assert not undocumented, f"FRM codes used but undocumented: {undocumented}"

    def test_all_frm_code_members_used_in_validator(self) -> None:
        """Guards against dead/aspirational codes: every defined member fires somewhere."""
        combined_src = self._combined_src()
        unused = [
            member.value
            for member in FrmCode
            if not re.search(rf"\bFrmCode\.{re.escape(member.name)}\b", combined_src)
        ]
        assert not unused, (
            f"FrmCode members defined but never referenced in validator.py/grammar.py: {unused}"
        )
