# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Benchmark Suite A scenario definitions (PRD S5)."""

import pytest

from apmode.backends.protocol import Lane
from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    scenario_a1,
    scenario_a2,
    scenario_a3,
    scenario_a4,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.validator import validate_dsl


@pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
class TestSuiteAScenarios:
    def test_spec_is_valid(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == [], f"Scenario {name} validation errors: {errors}"

    def test_spec_compiles_to_r(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        r_code = emit_nlmixr2(spec)
        assert "ini({" in r_code
        assert "model({" in r_code

    def test_reference_params_match_spec(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        struct_params = spec.structural_param_names()
        for param in ref:
            assert param in struct_params, (
                f"Reference param '{param}' not in {name} structural params: {struct_params}"
            )

    def test_reference_params_match_spec_values(self, name: str, factory: object) -> None:
        """Verify that REFERENCE_PARAMS values appear in emitted R code.

        Integer params (like Transit n) are emitted without decimal point.
        """
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        r_code = emit_nlmixr2(spec)
        for param, value in ref.items():
            # Try both float and int representations
            int_val = int(value) if value == int(value) else None
            found = str(value) in r_code or (int_val is not None and str(int_val) in r_code)
            assert found, f"Scenario {name}: param {param}={value} not found in emitted R code"


class TestSuiteASpecific:
    """Scenario-specific property tests."""

    def test_a1_is_simplest_model(self) -> None:
        spec = scenario_a1()
        assert not spec.has_node_modules()
        # Should produce linCmt() dynamics (1-cmt first-order + linear)
        r_code = emit_nlmixr2(spec)
        assert "linCmt()" in r_code

    def test_a2_has_parallel_elimination(self) -> None:
        spec = scenario_a2()
        r_code = emit_nlmixr2(spec)
        assert "Vmax" in r_code
        assert "Km" in r_code
        assert "CL" in r_code

    def test_a3_has_transit(self) -> None:
        spec = scenario_a3()
        r_code = emit_nlmixr2(spec)
        assert "transit" in r_code.lower() or "ktr" in r_code

    def test_a4_has_mm_elimination(self) -> None:
        spec = scenario_a4()
        r_code = emit_nlmixr2(spec)
        assert "Vmax" in r_code
        assert "Km" in r_code

    def test_all_scenarios_have_iiv(self) -> None:
        for name, factory in ALL_SCENARIOS:
            spec = factory()
            has_iiv = any(
                v.type == "IIV"
                for v in spec.variability  # type: ignore[union-attr]
            )
            assert has_iiv, f"Scenario {name} should have IIV"

    def test_reference_params_complete(self) -> None:
        """Every scenario has reference params, every param is covered."""
        assert set(REFERENCE_PARAMS.keys()) == {"A1", "A2", "A3", "A4"}
        for name, factory in ALL_SCENARIOS:
            spec = factory()
            struct = set(spec.structural_param_names())
            ref = set(REFERENCE_PARAMS[name].keys())
            assert ref == struct, (
                f"Scenario {name}: reference params {ref} != structural params {struct}"
            )
