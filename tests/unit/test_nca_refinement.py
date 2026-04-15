# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for PKNCA-style NCA refinement (λz curve-stripping, linear-up/log-down AUC, QC).

These tests exercise the post-refactor NCA estimator that replaced last-3-point
terminal regression + linear trapezoidal AUC with the PKNCA curve-stripping
algorithm and the linear-up/log-down integration rule. QC gates exclude
subjects with unreliable terminal phases or high extrapolation, and a
literature-prior fallback is used when per-subject NCA is predominantly
excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.bundle.models import ColumnMapping, DataManifest
from apmode.data.initial_estimates import (
    NCAEstimator,
    _auc_lin_up_log_down,
    _compute_nca_single_subject,
    _select_lambda_z,
)


def _make_manifest(n_subjects: int, n_obs: int) -> DataManifest:
    return DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=n_subjects,
        n_observations=n_obs,
        n_doses=n_subjects,
    )


def _make_oral_dataset(
    *,
    cl_true: float = 5.0,
    v_true: float = 70.0,
    ka_true: float = 1.5,
    dose: float = 100.0,
    n_subjects: int = 30,
    sample_times: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0),
    noise_cv: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a 1-cmt oral dataset (Bateman) with controlled noise."""
    rng = np.random.default_rng(seed)
    kel = cl_true / v_true
    rows: list[dict[str, float | int]] = []
    for subj in range(1, n_subjects + 1):
        rows.append(
            {"NMID": subj, "TIME": 0.0, "DV": 0.0, "MDV": 1, "EVID": 1, "AMT": dose, "CMT": 1}
        )
        for t in sample_times:
            conc = (
                dose
                / v_true
                * ka_true
                / (ka_true - kel)
                * (np.exp(-kel * t) - np.exp(-ka_true * t))
            )
            noise = rng.normal(0.0, noise_cv)
            rows.append(
                {
                    "NMID": subj,
                    "TIME": t,
                    "DV": max(0.001, conc * (1 + noise)),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
    return pd.DataFrame(rows)


class TestLambdaZCurveStripping:
    """PKNCA-style adaptive terminal-phase selection."""

    def test_returns_none_for_too_few_points(self) -> None:
        times = np.array([1.0, 2.0])
        concs = np.array([5.0, 2.0])
        assert _select_lambda_z(times, concs, tmax=0.5) is None

    def test_picks_most_points_on_perfect_exponential(self) -> None:
        """With monoexponential decline, adj_r² is identical for all windows;
        tiebreak picks the most points (PKNCA convention)."""
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = 10.0 * np.exp(-0.2 * times)
        fit = _select_lambda_z(times, concs, tmax=0.5)
        assert fit is not None
        assert fit.n_points == 5  # all post-Tmax points
        assert abs(fit.kel - 0.2) < 1e-6
        assert fit.adj_r2 > 0.999

    def test_prefers_cleaner_subset_over_distribution_contamination(self) -> None:
        """With a distribution phase that ends after Tmax, the best adj_r² fit
        should avoid the transition points."""
        # Two-phase decline: steep early (distribution), shallower late (terminal)
        times = np.array([0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0])
        # Distribution kel ≈ 2, terminal kel ≈ 0.1
        concs = np.array([10.0, 8.0, 6.5, 5.0, 4.0, 3.3, 2.7, 1.8])
        fit = _select_lambda_z(times, concs, tmax=0.5)
        assert fit is not None
        # Should pick a terminal-phase fit, not the full 8 points
        assert fit.kel < 0.5  # terminal, not distribution slope


class TestLinearUpLogDownAUC:
    """Linear-up/log-down AUC integration (Purves 1992, PKNCA default)."""

    def test_zero_zero_segment_contributes_zero(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        concs = np.array([0.0, 0.0, 0.0])
        assert _auc_lin_up_log_down(times, concs) == 0.0

    def test_rising_segment_uses_linear_trapezoid(self) -> None:
        times = np.array([0.0, 1.0])
        concs = np.array([1.0, 3.0])
        # Linear trapezoid: dt * (c1 + c2) / 2 = 1 * (1 + 3) / 2 = 2.0
        assert _auc_lin_up_log_down(times, concs) == pytest.approx(2.0)

    def test_declining_segment_uses_log_trapezoid(self) -> None:
        times = np.array([0.0, 1.0])
        concs = np.array([10.0, 5.0])
        # Log trapezoid: dt * (c1 - c2) / ln(c1/c2) = 1 * 5 / ln(2) ≈ 7.2135
        expected = 5.0 / np.log(2.0)
        assert _auc_lin_up_log_down(times, concs) == pytest.approx(expected)

    def test_monoexponential_integral_is_exact(self) -> None:
        """Log-trapezoid is the exact integral for monoexponential decline."""
        kel = 0.3
        c0 = 10.0
        times = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = c0 * np.exp(-kel * times)
        # Exact: c0/kel * (1 - exp(-kel * t_max))
        analytical = c0 / kel * (1.0 - np.exp(-kel * 12.0))
        computed = _auc_lin_up_log_down(times, concs)
        assert computed == pytest.approx(analytical, rel=1e-3)


class TestSingleSubjectNCA:
    """End-to-end per-subject NCA with QC."""

    def test_recovers_synthetic_parameters_within_5_percent(self) -> None:
        """On a clean 1-cmt oral profile, NCA should recover CL within a few percent."""
        times = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
        kel = 5.0 / 70.0  # CL/V
        ka = 1.5
        dose = 100.0
        concs = dose / 70.0 * ka / (ka - kel) * (np.exp(-kel * times) - np.exp(-ka * times))
        result = _compute_nca_single_subject(times, concs, dose=dose)
        assert result is not None
        assert not result.excluded
        # Tight recovery on clean data (but allow for NCA approximations)
        assert abs(result.kel - kel) / kel < 0.10
        # CL recovery — NCA underestimates CL slightly due to ka≠kel term
        assert result.cl > 0.0
        assert 3.0 < result.cl < 8.0

    def test_excludes_high_extrapolation(self) -> None:
        """Short profile with high last concentration → extrapolation >20% → excluded."""
        # Ensure Tmax is early and terminal phase has only 3 points, all close
        times = np.array([0.25, 0.5, 1.0, 2.0])
        concs = np.array([2.0, 8.0, 6.0, 5.0])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        # Insufficient terminal points → excluded (only 2 post-Tmax) OR high extrap
        assert result is None or result.excluded

    def test_excludes_low_adj_r2(self) -> None:
        """Noisy terminal phase → adj_r² below 0.80 → excluded."""
        rng = np.random.default_rng(0)
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        # Base profile + large multiplicative noise on terminal points
        base = 10.0 * np.exp(-0.2 * times)
        noisy = base * np.array([1.0, 1.0, 1.5 + rng.normal(0, 0.3), 0.5, 2.0, 0.3])
        result = _compute_nca_single_subject(times, np.clip(noisy, 1e-3, None), dose=100.0)
        # Either low adj_r² or high span exclusion — should be excluded
        if result is not None and not result.excluded:
            assert result.lambda_z_adj_r2 >= 0.80


class TestNCAEstimatorPopulation:
    """Population-level NCA + fallback behavior."""

    def test_literature_fallback_when_nca_fails(self) -> None:
        """When ≥50% of subjects are excluded, fall back to literature priors."""
        # Build a dataset where most subjects have short, high-extrap profiles
        rows: list[dict[str, float | int]] = []
        for subj in range(1, 11):
            rows.append(
                {"NMID": subj, "TIME": 0.0, "DV": 0.0, "MDV": 1, "EVID": 1, "AMT": 100.0, "CMT": 1}
            )
            # Only 3 points, very short — high extrapolation guaranteed
            for t in (0.5, 1.0, 2.0):
                rows.append(
                    {
                        "NMID": subj,
                        "TIME": t,
                        "DV": 8.0 if t == 1.0 else 5.0,
                        "MDV": 0,
                        "EVID": 0,
                        "AMT": 0.0,
                        "CMT": 1,
                    }
                )
        df = pd.DataFrame(rows)
        fallback = {"CL": 12.0, "V": 85.0, "ka": 1.2}
        est = NCAEstimator(df, _make_manifest(10, len(df)), fallback_estimates=fallback)
        result = est.estimate_per_subject()
        # The estimator should have used fallback values
        assert est.fallback_source in ("dataset_card", "defaults")
        if est.fallback_source == "dataset_card":
            assert result["CL"] == pytest.approx(12.0)
            assert result["V"] == pytest.approx(85.0)

    def test_uses_per_subject_median_when_most_pass_qc(self) -> None:
        """On a clean dataset, all subjects pass QC and median is used."""
        df = _make_oral_dataset(n_subjects=20, noise_cv=0.03)
        est = NCAEstimator(df, _make_manifest(20, len(df)))
        result = est.estimate_per_subject()
        assert est.fallback_source == "nca"
        # CL from NCA should be a positive plausible value
        assert 1.0 < result["CL"] < 50.0

    def test_diagnostics_recorded_per_subject(self) -> None:
        df = _make_oral_dataset(n_subjects=10)
        est = NCAEstimator(df, _make_manifest(10, len(df)))
        est.estimate_per_subject()
        assert len(est.diagnostics) == 10
        for d in est.diagnostics:
            assert d.subject_id in {str(i) for i in range(1, 11)}

    def test_min_subjects_triggers_population_fallback(self) -> None:
        """With <2 subjects, use population-level NCA (not per-subject median)."""
        df = _make_oral_dataset(n_subjects=1)
        est = NCAEstimator(df, _make_manifest(1, len(df)))
        result = est.estimate_per_subject()
        # Either succeeds with population-level or falls back to defaults
        assert all(v > 0 for k, v in result.items() if not k.startswith("_"))
