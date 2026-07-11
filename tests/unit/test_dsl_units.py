# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the units: block and dimensional-homogeneity checker (P1.3).

Formular sharpening plan §4 Phase 1 (P1.3): units.py is a dimensional-
homogeneity checker, not a unit-conversion library — see that module's
docstring for the exact design (recognized vocabulary, unresolved-token
handling, sigma_prop SD-vs-variance convention).
"""

from __future__ import annotations

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
    UnitsDeclaration,
)
from apmode.dsl.errors import FrmCode
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.units import UnitCoverageReport, check_units_consistency, unit_coverage_report
from apmode.dsl.validator import validate_dsl

_BASE_MODEL = """
model {{
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: {{ ka = 1.0, V = 70.0, CL = 5.0 }}
    units: {{ time = h, amount = {amount}, concentration = {concentration}, volume = {volume} }}
}}
"""


def _make_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": "test_units_000000000000",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
        "initial": {"ka": 1.0, "V": 70.0, "CL": 5.0},
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


class TestGrammarParsesUnitsBlock:
    def test_consistent_units_block_parses_and_compiles(self) -> None:
        text = _BASE_MODEL.format(amount="mg", concentration="ng/mL", volume="L")
        spec = compile_dsl(text)
        assert spec.units is not None
        assert spec.units.time == "h"
        assert spec.units.amount == "mg"
        assert spec.units.concentration == "ng/mL"
        assert spec.units.volume == "L"

    def test_model_without_units_block_has_none(self) -> None:
        text = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        spec = compile_dsl(text)
        assert spec.units is None

    def test_duplicate_units_block_raises_compile_error(self) -> None:
        from apmode.dsl.errors import FormularCompileError

        text = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            units: { time = h, amount = mg, concentration = ng/mL, volume = L }
            units: { time = h, amount = mg, concentration = ng/mL, volume = L }
        }
        """
        try:
            compile_dsl(text)
        except FormularCompileError as exc:
            assert exc.code == FrmCode.AST_DUPLICATE_BLOCK
        else:  # pragma: no cover
            raise AssertionError("expected FormularCompileError for duplicate units: block")


class TestCheckUnitsConsistency:
    def test_consistent_declaration(self) -> None:
        units = UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L")
        result = check_units_consistency(units)
        assert result.status == "consistent"

    def test_mass_prefix_variants_all_consistent(self) -> None:
        for amount, conc in (("mg", "ng/mL"), ("g", "mg/L"), ("ng", "mcg/mL"), ("mcg", "ug/mL")):
            units = UnitsDeclaration(time="h", amount=amount, concentration=conc, volume="L")
            assert check_units_consistency(units).status == "consistent", (amount, conc)

    def test_volume_declared_as_mass_unit_is_mismatched(self) -> None:
        units = UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="mg")
        result = check_units_consistency(units)
        assert result.status == "mismatched"
        assert result.field_status["volume"] == "mismatch"

    def test_concentration_without_slash_is_mismatched(self) -> None:
        units = UnitsDeclaration(time="h", amount="mg", concentration="mg", volume="L")
        result = check_units_consistency(units)
        assert result.status == "mismatched"
        assert "concentration_num" in result.field_status
        assert result.field_status["concentration_num"] == "mismatch"

    def test_unrecognized_token_is_unresolved_not_mismatched(self) -> None:
        units = UnitsDeclaration(time="fortnight", amount="mg", concentration="ng/mL", volume="L")
        result = check_units_consistency(units)
        assert result.status == "unresolved"
        assert result.field_status["time"] == "unresolved"


class TestValidateDslUnitsCode:
    def test_inconsistent_units_fails_with_frm_sem_010(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="mg")
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        matches = [e for e in errors if e.constraint == "units_dimensional_homogeneity"]
        assert len(matches) == 1
        assert matches[0].code == FrmCode.SEM_UNITS_INCONSISTENT

    def test_consistent_units_does_not_fail(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L")
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "units_dimensional_homogeneity" for e in errors)

    def test_no_units_block_does_not_fail(self) -> None:
        spec = _make_spec()
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "units_dimensional_homogeneity" for e in errors)


class TestUnitCoverageReport:
    def test_not_declared_when_no_units_block(self) -> None:
        spec = _make_spec()
        report = unit_coverage_report(spec)
        assert report.status == "not_declared"
        assert report.checked == []
        assert report.unchecked == []
        assert report.mismatched == []

    def test_checked_and_unchecked_for_consistent_declaration(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L")
        )
        report = unit_coverage_report(spec)
        assert report.status == "checked"
        # ka -> Rate, V -> Volume, CL -> Clearance, sigma_prop -> Unitless: all resolvable.
        assert set(report.checked) == {"ka", "V", "CL", "sigma_prop"}
        assert report.mismatched == []

    def test_mismatched_lists_volume_dependent_params(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="mg")
        )
        report = unit_coverage_report(spec)
        mismatched_params = {m.param for m in report.mismatched}
        assert {"V", "CL"} <= mismatched_params
        # ka (Rate) and sigma_prop (Unitless) don't depend on the volume field.
        assert "ka" in report.checked
        assert "sigma_prop" in report.checked

    def test_tmdd_params_dimensionally_checked(self) -> None:
        # TMDDCore params carry exact dimensions: R0/KD -> Concentration,
        # koff/kint -> Rate (1/Time), kon -> 1/(Concentration*Time). With a
        # consistent units block they are dimensionally checked, not unchecked.
        from apmode.dsl.ast_models import TMDDCore

        spec = _make_spec(
            distribution=TMDDCore(),
            elimination=LinearElim(),
            variability=[IIV(params=["V"], structure="diagonal")],
            initial={"V": 10.0, "R0": 1.0, "kon": 0.1, "koff": 0.01, "kint": 0.05, "CL": 5.0},
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        report = unit_coverage_report(spec)
        for name in ("R0", "kon", "koff", "kint", "V", "CL"):
            assert name in report.checked, (name, report.unchecked)
        assert report.mismatched == []

    def test_intercompartmental_clearance_q_checked(self) -> None:
        # Q/Q2/Q3 are inter-compartmental clearances (Volume/Time) — same
        # dimension as CL, so a consistent units block checks them.
        from apmode.dsl.ast_models import ThreeCmt, TwoCmt

        two = _make_spec(
            distribution=TwoCmt(),
            initial={"ka": 1.0, "V1": 10.0, "V2": 20.0, "Q": 3.0, "CL": 5.0, "sigma_prop": 0.1},
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        assert "Q" in unit_coverage_report(two).checked

        three = _make_spec(
            distribution=ThreeCmt(),
            initial={
                "ka": 1.0,
                "V1": 10.0,
                "V2": 20.0,
                "V3": 30.0,
                "Q2": 3.0,
                "Q3": 2.0,
                "CL": 5.0,
                "sigma_prop": 0.1,
            },
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        checked = unit_coverage_report(three).checked
        assert {"Q2", "Q3"} <= set(checked)

    def test_unresolved_token_marks_dependents_unchecked_not_mismatched(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="stone", concentration="ng/mL", volume="L")
        )
        report = unit_coverage_report(spec)
        # amount unresolved -> Volume/Clearance (which depend on amount) unchecked.
        assert "V" in report.unchecked
        assert "CL" in report.unchecked
        assert report.mismatched == []

    def test_sigma_prop_heuristic_warning_for_large_value(self) -> None:
        spec = _make_spec(
            observation=Proportional(sigma_prop=1.5),
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        report = unit_coverage_report(spec)
        assert report.sigma_prop_warnings
        assert "1.5" in report.sigma_prop_warnings[0]

    def test_no_sigma_prop_warning_for_typical_value(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        report = unit_coverage_report(spec)
        assert report.sigma_prop_warnings == []

    def test_combined_observation_sigma_names(self) -> None:
        spec = _make_spec(
            observation=Combined(sigma_prop=0.2, sigma_add=0.05),
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="L"),
        )
        report = unit_coverage_report(spec)
        assert "sigma_prop" in report.checked
        assert "sigma_add" in report.checked

    def test_report_is_json_roundtrippable(self) -> None:
        spec = _make_spec(
            units=UnitsDeclaration(time="h", amount="mg", concentration="ng/mL", volume="mg")
        )
        report = unit_coverage_report(spec)
        dumped = report.model_dump_json()
        restored = UnitCoverageReport.model_validate_json(dumped)
        assert restored == report
