# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for Suite A-External (Schoemaker 2019 grid).

Tests the BenchmarkCase specs, expected structures, and CI cadence
configuration. Does NOT require nlmixr2data R package — these tests
validate the Python-side case definitions only.
"""

from __future__ import annotations

import pytest

from apmode.benchmarks.suite_a_external import (
    ALL_CASES,
    CI_SMOKE_CASES,
    NIGHTLY_CASES,
)


@pytest.mark.integration
class TestSuiteAExternalCases:
    """Validate Suite A-External case definitions."""

    def test_all_cases_count(self) -> None:
        """12 Schoemaker grid datasets → 12 cases."""
        assert len(ALL_CASES) == 12

    def test_all_cases_have_expected_structure(self) -> None:
        """Every A-External case has an expected structure."""
        for case in ALL_CASES:
            assert case.expected_structure is not None, (
                f"{case.case_id}: missing expected_structure"
            )
            assert case.expected_structure.n_compartments in (1, 2)

    def test_all_cases_are_a_external_suite(self) -> None:
        """All cases belong to the A_external suite."""
        for case in ALL_CASES:
            assert case.suite == "A_external"
            assert case.case_id.startswith("a_ext_")

    def test_all_cases_use_submission_lane(self) -> None:
        """Schoemaker grid tests use submission lane (standard validation)."""
        for case in ALL_CASES:
            assert case.lane == "submission"

    @pytest.mark.parametrize("case", ALL_CASES, ids=[c.case_id for c in ALL_CASES])
    def test_case_has_valid_structure(self, case: object) -> None:
        """Each case specifies distribution and elimination types."""
        from apmode.benchmarks.models import BenchmarkCase

        assert isinstance(case, BenchmarkCase)
        assert case.expected_structure is not None
        assert case.expected_structure.distribution in ("OneCmt", "TwoCmt")
        assert case.expected_structure.elimination in ("Linear", "MichaelisMenten")

    def test_oral_cases_have_absorption(self) -> None:
        """Oral route cases must specify FirstOrder absorption."""
        oral_cases = [c for c in ALL_CASES if "oral" in c.case_id]
        assert len(oral_cases) == 4  # oral_1cpt, oral_1cptmm, oral_2cpt, oral_2cptmm
        for case in oral_cases:
            assert case.expected_structure is not None
            assert case.expected_structure.absorption == "FirstOrder"

    def test_iv_cases_have_no_absorption(self) -> None:
        """IV bolus/infusion cases have no absorption specified."""
        iv_cases = [c for c in ALL_CASES if "bolus" in c.case_id or "infusion" in c.case_id]
        assert len(iv_cases) == 8
        for case in iv_cases:
            assert case.expected_structure is not None
            assert case.expected_structure.absorption is None

    def test_nightly_subset(self) -> None:
        """Nightly subset has 3 cases (one per route)."""
        assert len(NIGHTLY_CASES) == 3
        ids = {c.case_id for c in NIGHTLY_CASES}
        assert "a_ext_bolus_1cpt" in ids
        assert "a_ext_infusion_2cptmm" in ids
        assert "a_ext_oral_2cpt" in ids

    def test_ci_smoke_subset(self) -> None:
        """CI smoke test has 1 case (oral_1cpt)."""
        assert len(CI_SMOKE_CASES) == 1
        assert CI_SMOKE_CASES[0].case_id == "a_ext_oral_1cpt"

    def test_weekly_cadence_default(self) -> None:
        """Default CI cadence for Schoemaker cases is weekly."""
        for case in ALL_CASES:
            assert case.ci_cadence == "weekly"
