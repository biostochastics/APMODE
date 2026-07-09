# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Benchmark Suite A scenario definitions (PRD §5)."""

import math
from pathlib import Path

import pytest

from apmode.backends.protocol import Lane
from apmode.benchmarks.scoring import score_eta_recovery
from apmode.benchmarks.suite_a import (
    A8_COVARIATE_MODEL_NOTES,
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    load_reference_eta,
    scenario_a1,
    scenario_a2,
    scenario_a3,
    scenario_a4,
    scenario_a5,
    scenario_a6,
    scenario_a7,
    scenario_a8,
    scenario_a13,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.validator import validate_dsl

SUITE_A_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "suite_a"

# A13 (SumIG) is not admissible in the submission lane (ADR-0003 D6, same
# family of restriction as NODE modules) but is otherwise a normal,
# non-NODE, R-compilable spec — it gets its own bucket rather than being
# folded into either CLASSICAL_SCENARIOS (submission-lane tested) or
# NODE_SCENARIOS (has_node_modules()-asserting).
NODE_SCENARIOS = [(n, f) for n, f in ALL_SCENARIOS if n == "A7"]
DISCOVERY_ONLY_NON_NODE_SCENARIOS = [(n, f) for n, f in ALL_SCENARIOS if n == "A13"]

# Every non-NODE scenario, including A13 — used for compilation and
# reference-param-membership checks that don't depend on lane admissibility.
NON_NODE_SCENARIOS = [(n, f) for n, f in ALL_SCENARIOS if n != "A7"]

# Classical scenarios (no NODE modules, submission-lane admissible) — can be
# validated for submission lane and compiled to R code.
CLASSICAL_SCENARIOS = [(n, f) for n, f in NON_NODE_SCENARIOS if n != "A13"]

# SumIG(k=2)'s MT_2 is a *derived* quantity in the emitted R code
# (``MT_2 <- MT_1 + delta_MT_2``, positive-difference parameterisation to
# prevent label switching — ADR-0003) rather than a literal token, so its
# REFERENCE_PARAMS value does not appear verbatim in emitted R code the way
# every other scenario's calibration values do. Maps (scenario, param) ->
# the literal value that *does* appear instead.
_VALUE_LITERAL_OVERRIDES: dict[tuple[str, str], float] = {
    ("A13", "MT_2"): REFERENCE_PARAMS["A13"]["MT_2"] - REFERENCE_PARAMS["A13"]["MT_1"],
}


@pytest.mark.parametrize("name,factory", NON_NODE_SCENARIOS)
class TestSuiteANonNodeCompilation:
    """Compilation + reference-param checks shared by every non-NODE scenario."""

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
        See ``_VALUE_LITERAL_OVERRIDES`` for the one scenario (A13/MT_2)
        whose reference value is not a literal token in the emitted code.
        """
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        r_code = emit_nlmixr2(spec)
        for param, value in ref.items():
            value = _VALUE_LITERAL_OVERRIDES.get((name, param), value)
            # Try both float and int representations
            int_val = int(value) if value == int(value) else None
            found = str(value) in r_code or (int_val is not None and str(int_val) in r_code)
            assert found, f"Scenario {name}: param {param}={value} not found in emitted R code"


@pytest.mark.parametrize("name,factory", CLASSICAL_SCENARIOS)
class TestSuiteAClassicalScenarios:
    """Submission-lane admissibility for classical (non-NODE, non-SumIG) scenarios."""

    def test_spec_is_valid(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == [], f"Scenario {name} validation errors: {errors}"


@pytest.mark.parametrize("name,factory", DISCOVERY_ONLY_NON_NODE_SCENARIOS)
class TestSuiteADiscoveryOnlyScenarios:
    """Scenarios admissible only in Discovery/Optimization lanes (not Submission)."""

    def test_spec_is_valid_discovery(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == [], f"Scenario {name} validation errors: {errors}"

    def test_spec_not_valid_submission(self, name: str, factory: object) -> None:
        """Confirms the lane restriction this bucket exists for is real,
        not just an artifact of how the test groups happen to be split."""
        spec = factory()  # type: ignore[operator]
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors != [], f"Scenario {name} unexpectedly valid in submission lane"


@pytest.mark.parametrize("name,factory", NODE_SCENARIOS)
class TestSuiteANodeScenarios:
    """NODE scenario validation (discovery lane, no R compilation)."""

    def test_spec_is_valid_discovery(self, name: str, factory: object) -> None:
        """NODE specs validate in discovery lane."""
        spec = factory()  # type: ignore[operator]
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == [], f"Scenario {name} validation errors: {errors}"

    def test_spec_has_node_modules(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        assert spec.has_node_modules(), f"Scenario {name} should have NODE modules"

    def test_reference_params_are_mechanistic_only(self, name: str, factory: object) -> None:
        """NODE scenario reference params cover only mechanistic (non-NODE) structural params.

        ``structural_param_names()`` now exposes NODE input-layer weights
        (``node_abs_w*`` / ``node_elim_w*``) so Variability validation can
        target them (#11). Reference params remain the mechanistic subset
        — assert that via a subset/disjoint-complement check rather than
        set equality.
        """
        spec = factory()  # type: ignore[operator]
        ref = set(REFERENCE_PARAMS[name].keys())
        struct = set(spec.structural_param_names())
        mechanistic = {
            s for s in struct if not s.startswith("node_abs_w") and not s.startswith("node_elim_w")
        }
        assert ref == mechanistic, (
            f"Scenario {name}: reference params {ref} != mechanistic params {mechanistic}"
        )


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

    def test_a5_is_tmdd(self) -> None:
        spec = scenario_a5()
        assert not spec.has_node_modules()
        assert spec.distribution.type == "TMDD_QSS"

    def test_a6_has_covariates(self) -> None:
        spec = scenario_a6()
        assert len(spec.covariates) == 3  # WT on CL, WT on V, RENAL on CL

    def test_a7_is_node_absorption(self) -> None:
        spec = scenario_a7()
        assert spec.has_node_modules()
        assert spec.absorption.type == "NODE_Absorption"

    def test_a8_has_crcl_covariate_and_notes(self) -> None:
        """A8 captures the allometric CRCL effect in the DSL and records the
        inexpressible diurnal damping as covariate-model notes metadata.
        """
        spec = scenario_a8()
        assert not spec.has_node_modules()
        cov_links = list(spec.covariates)
        assert len(cov_links) == 1
        link = cov_links[0]
        assert (link.param, link.covariate, link.form) == ("CL", "CRCL", "power")
        for key in ("theta_crcl", "delta_autoind"):
            assert key in A8_COVARIATE_MODEL_NOTES

    def test_all_scenarios_have_iiv(self) -> None:
        for name, factory in ALL_SCENARIOS:
            spec = factory()
            has_iiv = any(
                v.type == "IIV"
                for v in spec.variability  # type: ignore[union-attr]
            )
            assert has_iiv, f"Scenario {name} should have IIV"

    def test_reference_params_complete(self) -> None:
        """Every scenario has reference params."""
        assert set(REFERENCE_PARAMS.keys()) == {
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
            "A17",
            "A18",
            "A19",
            "A20a",
            "A20b",
            "A21",
        }
        for name, factory in NON_NODE_SCENARIOS:
            spec = factory()
            struct = set(spec.structural_param_names())
            ref = set(REFERENCE_PARAMS[name].keys())
            assert ref == struct, (
                f"Scenario {name}: reference params {ref} != structural params {struct}"
            )

    def test_a20a_a20b_share_filename_stem(self) -> None:
        """A20a/A20b are a paired benchmark unit against identical ground
        truth (BLQ_M3 vs. BLQ_M4), not two independent recovery tests."""
        from apmode.benchmarks.suite_a import SCENARIO_FILENAME_STEMS

        assert SCENARIO_FILENAME_STEMS["A20a"] == SCENARIO_FILENAME_STEMS["A20b"]

    def test_a13_disposition_fixed_via_priors(self) -> None:
        """SumIG(k=2) requires disposition (CL/V) fixed externally
        (ADR-0003 D5); the benchmark spec supplies this via
        source="fixed_external" priors rather than a dispatch-time
        EvidenceManifest flag."""
        spec = scenario_a13()
        fixed_targets = {p.target for p in spec.priors if p.source == "fixed_external"}
        assert {"CL", "V"}.issubset(fixed_targets)


class TestLoadReferenceEta:
    """``load_reference_eta`` parses ``*_eta.csv`` ground truth into the
    ``{subject_id: {param: value}}`` shape ``score_eta_recovery`` expects
    (Task 6: proof-of-chain, since no live Suite A fit-and-score driver
    exists yet to wire this into)."""

    def test_parses_a1_eta_csv(self) -> None:
        eta_path = SUITE_A_DIR / "a1_1cmt_oral_linear_eta.csv"
        assert eta_path.exists(), f"fixture missing: {eta_path}"

        true_eta = load_reference_eta(eta_path)

        # Header is "NMID","eta.ka","eta.V","eta.CL" -> prefix stripped,
        # subject ids stringified.
        assert "1" in true_eta
        assert set(true_eta["1"].keys()) == {"ka", "V", "CL"}
        assert true_eta["1"]["ka"] == pytest.approx(0.126267259035449)
        assert true_eta["1"]["V"] == pytest.approx(0.0831826677346906)
        assert true_eta["1"]["CL"] == pytest.approx(0.213646205307045)

    def test_chains_into_score_eta_recovery(self) -> None:
        """End-to-end: ground-truth CSV -> loader -> score_eta_recovery
        produces a finite, sane per-parameter RMSE dict."""
        eta_path = SUITE_A_DIR / "a1_1cmt_oral_linear_eta.csv"
        true_eta = load_reference_eta(eta_path)

        # Hand-built "fitted" etas: reuse a couple of true subjects with a
        # small perturbation plus one unmatched subject id (exercises the
        # missing-subject-is-excluded-not-penalized path already covered
        # unit-wise in test_benchmark_scoring.py).
        fitted_eta = {
            subject_id: {param: value + 0.01 for param, value in params.items()}
            for subject_id, params in list(true_eta.items())[:5]
        }
        fitted_eta["does-not-exist-in-truth"] = {"ka": 0.0, "V": 0.0, "CL": 0.0}

        result = score_eta_recovery(true_eta, fitted_eta)

        assert set(result.keys()) == {"ka", "V", "CL"}
        for param_name, rmse in result.items():
            assert math.isfinite(rmse), f"{param_name} RMSE should be finite"
            assert rmse == pytest.approx(0.01, abs=1e-6)
