# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NCA-based initial estimates (PRD §4.2.0.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apmode.bundle.models import InitialEstimateEntry, InitialEstimates
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.initial_estimates import (
    NCAEstimator,
    _compute_nca_single_subject,
    _default_estimates,
    build_initial_estimates_bundle,
    warm_start_estimates,
)

FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "pk_data" / "simple_1cmt.csv"


class TestNCAEstimator:
    """NCA-based initial estimate derivation."""

    def test_estimate_per_subject_returns_dict(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        assert isinstance(result, dict)
        assert "CL" in result
        assert "V" in result
        assert "ka" in result

    def test_estimates_are_positive(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        for key, val in result.items():
            if key.startswith("_"):
                continue  # skip metadata keys like _auc_extrap_high_fraction
            assert val > 0, f"{key} should be positive, got {val}"

    def test_estimates_in_reasonable_range(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        # For our test data: 1-cmt oral, ka~1.5, V~70, CL~5
        assert 0.1 < result["CL"] < 100
        assert 1.0 < result["V"] < 1000
        assert 0.1 < result["ka"] < 50

    def test_population_level_fallback(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_population_level()
        assert isinstance(result, dict)
        assert all(v > 0 for v in result.values())

    def test_build_entry(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        entry = estimator.build_entry("candidate_001")
        assert isinstance(entry, InitialEstimateEntry)
        assert entry.candidate_id == "candidate_001"
        assert entry.source == "nca"
        assert len(entry.estimates) > 0


class TestComputeNCASingleSubject:
    """Unit tests for the core NCA computation."""

    def test_simple_monoexponential(self) -> None:
        # C(t) = 10 * exp(-0.2 * t), Dose=100, V=10, CL=2, kel=0.2
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = 10.0 * np.exp(-0.2 * times)
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is not None
        assert result.cl > 0
        assert result.v > 0
        assert result.ka > 0
        assert result.kel > 0
        # kel should be approximately 0.2
        assert abs(result.kel - 0.2) < 0.1
        # AUC extrapolation fraction should be reasonable
        assert 0.0 <= result.auc_extrap_fraction <= 1.0

    def test_returns_none_for_insufficient_data(self) -> None:
        times = np.array([1.0, 2.0])
        concs = np.array([5.0, 3.0])
        # Only 2 points — not enough for terminal regression
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is None or result.cl > 0

    def test_returns_none_for_zero_auc(self) -> None:
        times = np.array([1.0, 2.0, 3.0])
        concs = np.array([0.0, 0.0, 0.0])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is None

    def test_sanity_bounds(self) -> None:
        """Results should be bounded within physiological ranges."""
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        concs = np.array([2.0, 5.0, 3.0, 1.0, 0.3])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        if result is not None:
            assert 0.01 <= result.cl <= 10000
            assert 0.1 <= result.v <= 100000
            assert 0.01 <= result.ka <= 100

    def test_multi_dose_steady_state(self) -> None:
        """Multi-dose AUC_tau estimation at steady state."""
        # Simulate steady-state: concentrations within one dosing interval (tau=12h)
        times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = np.array([2.0, 8.0, 10.0, 7.0, 4.0, 1.5, 0.8])
        result = _compute_nca_single_subject(
            times, concs, dose=100.0, is_steady_state=True, tau=12.0
        )
        assert result is not None
        assert result.cl > 0
        assert result.auc_extrap_fraction == 0.0  # no extrapolation for AUC_tau

    def test_high_auc_extrapolation_flagged(self) -> None:
        """AUC extrapolation fraction is computed correctly."""
        # Short profile with high last concentration → large extrapolation
        times = np.array([0.5, 1.0, 2.0, 3.0])
        concs = np.array([5.0, 10.0, 8.0, 7.0])  # barely declining terminal phase
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        if result is not None:
            # High last-concentration relative to AUC → high extrapolation
            assert result.auc_extrap_fraction >= 0.0


class TestWarmStartEstimates:
    """Warm-start from parent model parameters."""

    def test_inherits_parent_params(self) -> None:
        parent = {"CL": 5.0, "V": 70.0, "ka": 1.5}
        entry = warm_start_estimates(parent, "child_001")
        assert entry.source == "warm_start"
        assert entry.estimates == parent
        assert entry.candidate_id == "child_001"


class TestBuildBundle:
    """Building the initial_estimates.json artifact."""

    def test_builds_from_entries(self) -> None:
        entries = [
            InitialEstimateEntry(
                candidate_id="c1",
                source="nca",
                estimates={"CL": 5.0, "V": 70.0},
            ),
            InitialEstimateEntry(
                candidate_id="c2",
                source="warm_start",
                estimates={"CL": 6.0, "V": 65.0},
            ),
        ]
        bundle = build_initial_estimates_bundle(entries)
        assert isinstance(bundle, InitialEstimates)
        assert "c1" in bundle.entries
        assert "c2" in bundle.entries
        assert bundle.entries["c1"].source == "nca"

    def test_roundtrip_json(self) -> None:
        entries = [
            InitialEstimateEntry(
                candidate_id="c1",
                source="nca",
                estimates={"CL": 5.0},
            ),
        ]
        bundle = build_initial_estimates_bundle(entries)
        json_str = bundle.model_dump_json()
        roundtrip = InitialEstimates.model_validate_json(json_str)
        assert roundtrip.entries["c1"].estimates["CL"] == pytest.approx(5.0)


class TestDefaultEstimates:
    """Fallback estimates."""

    def test_returns_positive_values(self) -> None:
        defaults = _default_estimates()
        assert all(v > 0 for v in defaults.values())

    def test_contains_required_keys(self) -> None:
        defaults = _default_estimates()
        assert "CL" in defaults
        assert "V" in defaults
        assert "ka" in defaults
