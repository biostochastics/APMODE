# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for `use <macro>` statements (Formular sharpening plan §4 Phase 2, P2.1).

Covers: grammar parsing + expansion into plain AST nodes, `macros_used`
provenance, unknown/duplicate-macro rejection (FrmCode.AST_MACRO_UNKNOWN /
AST_MACRO_DUPLICATE_USE), the documented no-op path when coverage already
exists, macro + hand-authored declaration coexistence (no double-declare),
and the bundle emitter's `expanded.formular` audit artifact.
"""

from __future__ import annotations

import pytest

from apmode.dsl.ast_models import (
    IIV,
    Additive,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.errors import FormularCompileError, FrmCode
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.macros import MACRO_REGISTRY, expand_macros
from apmode.dsl.priors import LogNormalPrior

_BASE_MODEL = """
model {{
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    observation: Proportional(sigma_prop=0.1)
    initial: {{ ka = 1.0, V = 70.0, CL = 5.0 }}
    {extra}
}}
"""


def _compile(extra: str = "") -> DSLSpec:
    return compile_dsl(_BASE_MODEL.format(extra=extra))


class TestUseStandardIiv:
    def test_expands_diagonal_iiv_on_every_structural_param(self) -> None:
        spec = _compile("use pkstd.standard_iiv")
        assert spec.macros_used == ["pkstd.standard_iiv@v1"]
        assert len(spec.variability) == 1
        entry = spec.variability[0]
        assert isinstance(entry, IIV)
        assert entry.structure == "diagonal"
        assert sorted(entry.params) == sorted(spec.structural_param_names())

    def test_no_op_when_every_param_already_covered(self) -> None:
        """No IIV entry added (and no error) when coverage is already complete."""
        spec = _compile(
            "variability: IIV(params=[ka, V, CL], structure=diagonal)\nuse pkstd.standard_iiv"
        )
        # macros_used still records the use (expansion ran; it just chose
        # to make no change to `variability`).
        assert spec.macros_used == ["pkstd.standard_iiv@v1"]
        assert len(spec.variability) == 1
        assert sorted(spec.variability[0].params) == ["CL", "V", "ka"]

    def test_coexists_with_partial_manual_iiv_without_double_declaring(self) -> None:
        """Macro only covers params NOT already in a hand-authored IIV block."""
        extra = "variability: IIV(params=[CL], structure=diagonal)\nuse pkstd.standard_iiv"
        spec = _compile(extra)
        assert spec.macros_used == ["pkstd.standard_iiv@v1"]
        assert len(spec.variability) == 2
        all_params: list[str] = []
        for item in spec.variability:
            all_params.extend(item.params)
        # CL appears exactly once across both entries -- not double-declared.
        assert all_params.count("CL") == 1
        assert set(all_params) == set(spec.structural_param_names())


class TestUseStandardPriors:
    def test_expands_priors_on_every_structural_param(self) -> None:
        spec = _compile("use pkstd.standard_priors")
        assert spec.macros_used == ["pkstd.standard_priors@v1"]
        targets = {p.target for p in spec.priors}
        assert targets == set(spec.structural_param_names())
        for prior in spec.priors:
            assert isinstance(prior.family, LogNormalPrior)
            assert prior.source == "weakly_informative"

    def test_no_op_when_all_priors_already_declared(self) -> None:
        extra = (
            "priors: { ka ~ LogNormal(mu=0.0, sigma=1.0) "
            "V ~ LogNormal(mu=0.0, sigma=1.0) "
            "CL ~ LogNormal(mu=0.0, sigma=1.0) }\nuse pkstd.standard_priors"
        )
        spec = _compile(extra)
        assert spec.macros_used == ["pkstd.standard_priors@v1"]
        assert len(spec.priors) == 3

    def test_coexists_with_partial_manual_prior_without_double_declaring(self) -> None:
        extra = "priors: { CL ~ LogNormal(mu=1.0, sigma=0.5) }\nuse pkstd.standard_priors"
        spec = _compile(extra)
        targets = [p.target for p in spec.priors]
        assert targets.count("CL") == 1
        cl_prior = next(p for p in spec.priors if p.target == "CL")
        assert isinstance(cl_prior.family, LogNormalPrior)
        assert cl_prior.family.mu == 1.0  # hand-authored value preserved, not overwritten
        assert set(targets) == set(spec.structural_param_names())


class TestUseStandardErrorModel:
    def test_proportional_gets_sigma_prop_prior(self) -> None:
        spec = _compile("use pkstd.standard_error_model")
        assert spec.macros_used == ["pkstd.standard_error_model@v1"]
        targets = {p.target for p in spec.priors}
        assert targets == {"sigma_prop"}

    def test_additive_gets_sigma_add_prior(self) -> None:
        model = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Additive(sigma_add=0.5)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            use pkstd.standard_error_model
        }
        """
        spec = compile_dsl(model)
        targets = {p.target for p in spec.priors}
        assert targets == {"sigma_add"}

    def test_no_op_when_sigma_prior_already_declared(self) -> None:
        extra = "priors: { sigma_prop ~ HalfNormal(sigma=1.0) }\nuse pkstd.standard_error_model"
        spec = _compile(extra)
        assert spec.macros_used == ["pkstd.standard_error_model@v1"]
        assert len(spec.priors) == 1

    def test_multi_analyte_observations_is_a_documented_no_op(self) -> None:
        model = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observations: {
                plasma: { dvid=1, prediction=C_central, error=Proportional(sigma_prop=0.1) }
            }
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            use pkstd.standard_error_model
        }
        """
        spec = compile_dsl(model)
        assert spec.macros_used == ["pkstd.standard_error_model@v1"]
        assert spec.priors == []


class TestMultipleMacrosAndOrdering:
    def test_multiple_distinct_macros_expand_in_source_order(self) -> None:
        spec = _compile(
            "use pkstd.standard_iiv\nuse pkstd.standard_priors\nuse pkstd.standard_error_model"
        )
        assert spec.macros_used == [
            "pkstd.standard_iiv@v1",
            "pkstd.standard_priors@v1",
            "pkstd.standard_error_model@v1",
        ]
        assert len(spec.variability) == 1
        assert {p.target for p in spec.priors} == {"ka", "V", "CL", "sigma_prop"}

    def test_zero_use_statements_leaves_macros_used_empty(self) -> None:
        spec = _compile()
        assert spec.macros_used == []


class TestUnknownAndDuplicateMacroRejection:
    def test_unknown_macro_raises_ast_macro_unknown(self) -> None:
        with pytest.raises(FormularCompileError) as exc_info:
            _compile("use pkstd.nonexistent_macro")
        assert exc_info.value.code == FrmCode.AST_MACRO_UNKNOWN

    def test_duplicate_use_of_same_macro_raises_ast_macro_duplicate_use(self) -> None:
        with pytest.raises(FormularCompileError) as exc_info:
            _compile("use pkstd.standard_iiv\nuse pkstd.standard_iiv")
        assert exc_info.value.code == FrmCode.AST_MACRO_DUPLICATE_USE

    def test_duplicate_use_does_not_double_declare_iiv(self) -> None:
        """Direct expand_macros call: confirms rejection happens before any
        double-application could occur (defence in depth alongside the
        grammar-level test above)."""
        spec = DSLSpec(
            model_id="test_macro_dup",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[],
            observation=Additive(sigma_add=0.2),
            initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        with pytest.raises(FormularCompileError) as exc_info:
            expand_macros(spec, ["pkstd.standard_iiv", "pkstd.standard_iiv"])
        assert exc_info.value.code == FrmCode.AST_MACRO_DUPLICATE_USE


class TestRegistryDirectly:
    def test_registry_contains_exactly_the_three_stdlib_macros(self) -> None:
        assert set(MACRO_REGISTRY) == {
            "pkstd.standard_iiv",
            "pkstd.standard_priors",
            "pkstd.standard_error_model",
        }
        for name, (macro_def, _expander) in MACRO_REGISTRY.items():
            assert macro_def.name == name
            assert macro_def.version == "v1"

    def test_expand_macros_is_a_pure_function_returning_new_spec(self) -> None:
        spec = DSLSpec(
            model_id="test_macro_pure",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "V": 70.0, "CL": 5.0},
        )
        expanded = expand_macros(spec, ["pkstd.standard_iiv"])
        assert expanded is not spec
        assert spec.variability == []  # original untouched (frozen model)
        assert len(expanded.variability) == 1
        assert expanded.macros_used == ["pkstd.standard_iiv@v1"]
