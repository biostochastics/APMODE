# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for Suite C (expert head-to-head comparison).

Tests the scoring harness, expert comparison methodology, and case definitions.
Uses synthetic data — does NOT require real expert models or datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from apmode.benchmarks.models import BenchmarkScore
from apmode.benchmarks.scoring import (
    aggregate_suite,
    compute_fraction_beats_expert,
    compute_npe,
    compute_prediction_interval_calibration,
    evaluate_expert_comparison,
)
from apmode.benchmarks.suite_c import (
    ALL_CASES,
    CASE_C1_MAVOGLURANT,
    CASE_C2_GENTAMICIN,
    CASE_C3_PROPOFOL,
    DEFAULT_SPLIT,
    MIN_EXPERT_COUNT,
    PRIORITY_ORDER,
    WIN_MARGIN_DELTA,
)

# ---------------------------------------------------------------------------
# Case definition tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteCCases:
    """Validate Suite C case definitions."""

    def test_all_cases_count(self) -> None:
        """3 initial Suite C datasets."""
        assert len(ALL_CASES) == 3

    def test_all_cases_are_suite_c(self) -> None:
        """All cases belong to Suite C."""
        for case in ALL_CASES:
            assert case.suite == "C"
            assert case.ci_cadence == "quarterly"

    def test_all_cases_have_split_strategy(self) -> None:
        """Every Suite C case uses subject-level 5-fold CV."""
        for case in ALL_CASES:
            assert case.split_strategy is not None
            assert case.split_strategy.method == "subject_level_kfold"
            assert case.split_strategy.n_folds == 5

    def test_all_cases_have_expert_models(self) -> None:
        """Every Suite C case has at least one expert model reference."""
        for case in ALL_CASES:
            assert len(case.expert_models) >= 1

    def test_c1_mavoglurant(self) -> None:
        """C1 targets mavoglurant with submission lane."""
        assert CASE_C1_MAVOGLURANT.dataset_id == "nlmixr2data_mavoglurant"
        assert CASE_C1_MAVOGLURANT.lane == "submission"

    def test_c2_gentamicin(self) -> None:
        """C2 targets gentamicin IOV."""
        assert CASE_C2_GENTAMICIN.dataset_id == "ddmore_gentamicin"
        assert "IOV" in CASE_C2_GENTAMICIN.description

    def test_c3_propofol(self) -> None:
        """C3 targets Eleveld propofol (discovery lane — IV infusion)."""
        assert CASE_C3_PROPOFOL.dataset_id == "opentci_propofol"
        assert CASE_C3_PROPOFOL.lane == "discovery"

    def test_priority_order(self) -> None:
        """Priority order starts with mavoglurant (no access barriers)."""
        assert PRIORITY_ORDER[0] == "c1_mavoglurant"
        assert len(PRIORITY_ORDER) == 3

    def test_win_margin(self) -> None:
        """Win margin delta is 2%."""
        assert WIN_MARGIN_DELTA == 0.02

    def test_default_split(self) -> None:
        """Default split is 5-fold subject-level CV."""
        assert DEFAULT_SPLIT.method == "subject_level_kfold"
        assert DEFAULT_SPLIT.n_folds == 5

    def test_min_expert_count(self) -> None:
        """Minimum expert count for reliable median is 3."""
        assert MIN_EXPERT_COUNT == 3


# ---------------------------------------------------------------------------
# Scoring harness tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNPEComputation:
    """Test nonparametric prediction error computation."""

    def test_perfect_prediction(self) -> None:
        """Perfect predictions yield NPE = 0."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # 100 simulations that perfectly match
        sims = np.tile(observed, (100, 1))
        assert compute_npe(observed, sims) == pytest.approx(0.0)

    def test_constant_offset(self) -> None:
        """Constant offset predictions yield NPE = offset."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sims = np.tile(observed + 0.5, (100, 1))
        assert compute_npe(observed, sims) == pytest.approx(0.5)

    def test_npe_is_positive(self) -> None:
        """NPE is always non-negative."""
        rng = np.random.default_rng(42)
        observed = rng.normal(10, 2, size=50)
        sims = rng.normal(10, 3, size=(200, 50))
        assert compute_npe(observed, sims) >= 0.0


@pytest.mark.integration
class TestPICalibration:
    """Test prediction interval calibration."""

    def test_well_calibrated_normal(self) -> None:
        """Simulations from the true model should be well-calibrated."""
        rng = np.random.default_rng(42)
        n_obs = 1000
        observed = rng.normal(0, 1, size=n_obs)
        sims = rng.normal(0, 1, size=(500, n_obs))

        cal = compute_prediction_interval_calibration(observed, sims)

        # Each level should be close to nominal (within ~5%)
        assert abs(cal["50"] - 0.50) < 0.05
        assert abs(cal["80"] - 0.80) < 0.05
        assert abs(cal["90"] - 0.90) < 0.05
        assert abs(cal["95"] - 0.95) < 0.05

    def test_narrow_pi_undercoverage(self) -> None:
        """Too-narrow PIs should show undercoverage."""
        rng = np.random.default_rng(42)
        n_obs = 500
        observed = rng.normal(0, 2, size=n_obs)  # True SD = 2
        sims = rng.normal(0, 0.5, size=(200, n_obs))  # Model SD = 0.5

        cal = compute_prediction_interval_calibration(observed, sims)
        assert cal["95"] < 0.50  # Massive undercoverage


@pytest.mark.integration
class TestExpertComparison:
    """Test expert comparison scoring."""

    def test_apmode_beats_experts(self) -> None:
        """APMODE wins when NPE is sufficiently lower than expert median."""
        beats, median, gap = evaluate_expert_comparison(
            apmode_npe=5.0,
            expert_npes=[8.0, 10.0, 12.0],
            win_margin=0.02,
        )
        assert beats is True
        assert median == 10.0
        assert gap == -5.0

    def test_apmode_loses_to_experts(self) -> None:
        """APMODE loses when NPE exceeds expert median."""
        beats, median, gap = evaluate_expert_comparison(
            apmode_npe=12.0,
            expert_npes=[8.0, 10.0, 11.0],
            win_margin=0.02,
        )
        assert beats is False
        assert median == 10.0
        assert gap == 2.0

    def test_marginal_win_requires_delta(self) -> None:
        """Barely-equal NPE does not count as a win (delta required)."""
        # Expert median = 10.0, threshold = 10.0 * 0.98 = 9.8
        beats, _, _ = evaluate_expert_comparison(
            apmode_npe=9.9,  # Better than median but not by delta
            expert_npes=[9.0, 10.0, 11.0],
            win_margin=0.02,
        )
        assert beats is False

    def test_fraction_beats_expert(self) -> None:
        """Fraction computation across multiple datasets."""
        results = [
            (5.0, [8.0, 10.0, 12.0]),  # Win
            (12.0, [8.0, 10.0, 11.0]),  # Loss
            (3.0, [6.0, 7.0, 8.0]),  # Win
        ]
        frac = compute_fraction_beats_expert(results, win_margin=0.02)
        assert frac == pytest.approx(2.0 / 3.0)

    def test_fraction_empty(self) -> None:
        """Empty results → 0.0 fraction."""
        assert compute_fraction_beats_expert([], win_margin=0.02) == 0.0


@pytest.mark.integration
class TestSuiteAggregation:
    """Test suite-level report aggregation."""

    def test_aggregate_empty(self) -> None:
        """Empty suite produces zero-count summary."""
        report = aggregate_suite("A", [])
        assert report.summary.n_cases == 0
        assert report.summary.pass_rate == 0.0

    def test_aggregate_with_scores(self) -> None:
        """Aggregation computes correct pass rate."""
        scores = [
            BenchmarkScore(
                case_id="test_1",
                run_id="run_1",
                suite="A",
                overall_passed=True,
                wall_time_seconds=10.0,
                structure_recovered=True,
                param_bias={"ka": 0.05},
            ),
            BenchmarkScore(
                case_id="test_2",
                run_id="run_2",
                suite="A",
                overall_passed=False,
                wall_time_seconds=20.0,
                structure_recovered=False,
                param_bias={"ka": 0.25},
            ),
        ]
        report = aggregate_suite("A", scores)
        assert report.summary.n_cases == 2
        assert report.summary.n_passed == 1
        assert report.summary.pass_rate == 0.5
        assert report.summary.structural_recovery_rate == 0.5
        assert report.summary.mean_param_bias is not None
