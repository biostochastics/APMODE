# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Bayesian harness's predictive-diagnostics integration.

Exercises :func:`apmode.bayes.harness.build_predictive_from_draws` —
the Python-side plumbing that reshapes Stan's ``y_pred[n]`` posterior-
predictive draws into the canonical
:func:`apmode.backends.predictive_summary.build_predictive_diagnostics`
call. Runs without Stan installed: the helper is pure-numpy plus the
predictive_summary module.
"""

from __future__ import annotations

import numpy as np
import pytest

from apmode.backends.predictive_summary import PredictiveSummaryBundle
from apmode.bayes.harness import build_predictive_from_draws
from apmode.bundle.models import NCASubjectDiagnostic
from apmode.governance.policy import Gate3Config


def _policy() -> Gate3Config:
    return Gate3Config(
        composite_method="weighted_sum",
        vpc_weight=0.5,
        npe_weight=0.5,
        bic_weight=0.0,
        auc_cmax_weight=0.0,
        vpc_n_bins=3,
        auc_cmax_nca_min_eligible=3,
        auc_cmax_nca_min_eligible_fraction=0.5,
    )


class TestBuildPredictiveFromDraws:
    def test_reshape_per_subject_and_populate_bundle(self) -> None:
        # 4 subjects * 3 obs each = 12 total observations; 10 posterior draws.
        n_subjects = 4
        n_obs_per_subj = 3
        n_sims = 10
        n_total = n_subjects * n_obs_per_subj

        rng = np.random.default_rng(0)
        obs_subject_idx = np.repeat(np.arange(1, n_subjects + 1), n_obs_per_subj)
        obs_times = np.tile(np.array([0.5, 1.0, 2.0]), n_subjects)
        observed_dv = np.full(n_total, 5.0)
        y_pred_draws = rng.normal(loc=5.0, scale=0.1, size=(n_sims, n_total))

        diagnostics = [
            NCASubjectDiagnostic(subject_id=str(i + 1), excluded=False) for i in range(n_subjects)
        ]

        bundle = build_predictive_from_draws(
            y_pred_draws,
            obs_subject_idx,
            obs_times,
            observed_dv,
            diagnostics,
            _policy(),
        )
        assert isinstance(bundle, PredictiveSummaryBundle)
        assert bundle.n_subjects_total == n_subjects
        assert bundle.n_subjects_nca_eligible == n_subjects
        # observed ≈ sim mean → BE should be near 1.0
        assert bundle.auc_cmax_be_score == pytest.approx(1.0)
        assert bundle.auc_cmax_source == "observed_trapezoid"

    def test_shape_mismatch_raises(self) -> None:
        y_pred_draws = np.zeros((5, 10))
        # 8 observation records but y_pred has 10 columns
        obs_subject_idx = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        obs_times = np.arange(8, dtype=float)
        observed_dv = np.full(8, 5.0)
        with pytest.raises(ValueError, match="y_pred_draws shape"):
            build_predictive_from_draws(
                y_pred_draws, obs_subject_idx, obs_times, observed_dv, None, _policy()
            )

    def test_obs_vector_length_mismatch_raises(self) -> None:
        y_pred_draws = np.zeros((5, 8))
        obs_subject_idx = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        obs_times = np.arange(7, dtype=float)  # length 7 not 8
        observed_dv = np.full(8, 5.0)
        with pytest.raises(ValueError, match="obs_times"):
            build_predictive_from_draws(
                y_pred_draws, obs_subject_idx, obs_times, observed_dv, None, _policy()
            )

    def test_no_nca_diagnostics_mask_drops_auc_cmax(self) -> None:
        n_subjects = 4
        n_obs_per_subj = 3
        n_sims = 10
        n_total = n_subjects * n_obs_per_subj
        rng = np.random.default_rng(1)

        obs_subject_idx = np.repeat(np.arange(1, n_subjects + 1), n_obs_per_subj)
        obs_times = np.tile(np.array([0.5, 1.0, 2.0]), n_subjects)
        observed_dv = np.full(n_total, 5.0)
        y_pred_draws = rng.normal(loc=5.0, scale=0.1, size=(n_sims, n_total))

        bundle = build_predictive_from_draws(
            y_pred_draws,
            obs_subject_idx,
            obs_times,
            observed_dv,
            None,  # no per-subject QC → all subjects ineligible
            _policy(),
        )
        assert bundle.auc_cmax_be_score is None
        assert bundle.auc_cmax_source is None
        # VPC + NPE should still populate.
        assert bundle.vpc is not None
        assert bundle.npe_score is not None
