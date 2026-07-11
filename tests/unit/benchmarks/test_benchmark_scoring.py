# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for :mod:`apmode.benchmarks.scoring` (per-parameter metrics).

Covers ``score_eta_recovery`` — per-parameter RMSE between ground-truth
simulated ETAs (Suite A ``<stem>_eta.csv``) and fitted per-subject ETAs
(``BackendResult.per_subject_eta``). Missingness handling mirrors
``score_parameter_bias``: a subject absent from one side is excluded from
that parameter's RMSE rather than penalized as zero; a parameter with no
overlapping subjects at all scores ``NaN`` (unscorable), not ``0.0``.
"""

from __future__ import annotations

import math

from apmode.benchmarks.scoring import score_eta_recovery


def test_score_eta_recovery_basic() -> None:
    true_eta = {"1": {"CL": 0.2, "V": -0.1}, "2": {"CL": -0.1, "V": 0.05}}
    fitted_eta = {"1": {"CL": 0.18, "V": -0.09}, "2": {"CL": -0.12, "V": 0.06}}
    result = score_eta_recovery(true_eta, fitted_eta)
    assert set(result) == {"CL", "V"}
    assert result["CL"] > 0  # RMSE-like, positive
    assert not math.isnan(result["CL"])
    assert not math.isnan(result["V"])


def test_score_eta_recovery_missing_subject_is_nan_safe() -> None:
    true_eta = {"1": {"CL": 0.2}, "2": {"CL": -0.1}}
    fitted_eta = {"1": {"CL": 0.18}}  # subject 2 missing from fit
    result = score_eta_recovery(true_eta, fitted_eta)
    assert not math.isnan(result["CL"])  # scored only over the overlapping subject


def test_score_eta_recovery_no_overlap_is_nan() -> None:
    true_eta = {"1": {"CL": 0.2}}
    fitted_eta = {"2": {"CL": 0.18}}  # disjoint subject IDs
    result = score_eta_recovery(true_eta, fitted_eta)
    assert math.isnan(result["CL"])


def test_score_eta_recovery_perfect_match_is_zero() -> None:
    true_eta = {"1": {"CL": 0.2}, "2": {"CL": -0.1}}
    fitted_eta = {"1": {"CL": 0.2}, "2": {"CL": -0.1}}
    result = score_eta_recovery(true_eta, fitted_eta)
    assert result["CL"] == 0.0


def test_score_eta_recovery_param_missing_from_fitted_subject() -> None:
    true_eta = {"1": {"CL": 0.2, "V": -0.1}, "2": {"CL": -0.1, "V": 0.05}}
    # subject "1" is present but is missing the "V" key entirely
    fitted_eta = {"1": {"CL": 0.18}, "2": {"CL": -0.12, "V": 0.06}}
    result = score_eta_recovery(true_eta, fitted_eta)
    assert not math.isnan(result["V"])  # scored only from subject "2"
