# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Benchmark Suite A simulation scaffolding.

Validates:
- R simulation script exists and has correct structure
- Reference params align with DSLSpec scenarios
- Generated CSV files (when available) can be ingested
"""

from __future__ import annotations

from pathlib import Path

from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    scenario_a1,
    scenario_a2,
    scenario_a3,
    scenario_a4,
)

SUITE_A_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "suite_a"


class TestSimulationScaffolding:
    """Validate the simulation R script and reference structure."""

    def test_simulation_script_exists(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        assert script.exists(), "simulate_all.R should exist in benchmarks/suite_a/"

    def test_simulation_script_has_all_scenarios(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "Scenario A1" in content
        assert "Scenario A2" in content
        assert "Scenario A3" in content
        assert "Scenario A4" in content

    def test_simulation_script_uses_rxode2(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "library(rxode2)" in content

    def test_simulation_script_sets_seed(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "set.seed(" in content
        assert "L'Ecuyer-CMRG" in content

    def test_simulation_output_filenames(self) -> None:
        """Check that the R script writes expected filenames."""
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "a1_1cmt_oral_linear.csv" in content
        assert "a2_2cmt_iv_parallel_mm.csv" in content
        assert "a3_transit_1cmt_linear.csv" in content
        assert "a4_1cmt_oral_mm.csv" in content
        assert "reference_params.json" in content


class TestReferenceParamsAlignWithSpecs:
    """Ensure REFERENCE_PARAMS match DSLSpec structural_param_names()."""

    def test_a1_params_match_spec(self) -> None:
        spec = scenario_a1()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A1"].keys())
        assert spec_params == ref_params

    def test_a2_params_match_spec(self) -> None:
        spec = scenario_a2()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A2"].keys())
        assert spec_params == ref_params

    def test_a3_params_match_spec(self) -> None:
        spec = scenario_a3()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A3"].keys())
        assert spec_params == ref_params

    def test_a4_params_match_spec(self) -> None:
        spec = scenario_a4()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A4"].keys())
        assert spec_params == ref_params

    def test_all_scenarios_have_reference_params(self) -> None:
        for name, _ in ALL_SCENARIOS:
            assert name in REFERENCE_PARAMS, f"Missing reference params for {name}"

    def test_reference_param_values_are_positive(self) -> None:
        for name, params in REFERENCE_PARAMS.items():
            for param_name, value in params.items():
                assert value > 0, f"{name}.{param_name} should be positive, got {value}"
