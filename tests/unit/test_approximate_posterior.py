# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :mod:`apmode.governance.approximate_posterior` (plan Task 22).

The Laplace/MVN helper produces approximate posterior-predictive draws
for the MLE path so Gate 3 can compare MLE and Bayesian candidates on
commensurate interval tuples. Tests cover:

* Shape + sample statistics (mean / sd) on a well-conditioned covariance.
* Ill-conditioned and singular covariances fall through to
  ``empirical_bootstrap``.
* ``fallback="raise"`` propagates the ``LinAlgError`` so callers that
  prefer hard-failing on degenerate MLE fits can opt out.
* Non-finite inputs, mismatched shapes, and non-positive ``n_draws``
  raise ``ValueError`` before any RNG work.
* Same ``seed`` reproduces the same draws byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from apmode.governance.approximate_posterior import laplace_draws

# --- Happy path ----------------------------------------------------------


def test_laplace_draws_shape_and_mean() -> None:
    theta = np.array([2.0, 30.0, 0.6])
    cov = np.diag([0.01, 1.0, 0.001])
    draws = laplace_draws(theta, cov, n_draws=2000, seed=42)
    assert draws.shape == (2000, 3)
    # Standard error of the mean is sd/sqrt(n). With n=2000 and sd up to
    # ~1.0, three-sigma tolerance is ~0.067. Use 0.1 for a comfortable
    # margin while still catching a sign error.
    assert np.allclose(draws.mean(axis=0), theta, atol=0.1)


def test_marginal_sd_matches_covariance_diagonal() -> None:
    theta = np.array([2.0, 30.0])
    cov = np.diag([0.04, 0.25])
    draws = laplace_draws(theta, cov, n_draws=5000, seed=1)
    sample_sd = draws.std(axis=0, ddof=1)
    expected_sd = np.sqrt(np.diag(cov))
    # 5% relative tolerance — the sampling distribution of SD is narrower
    # than the mean because it's concentration-of-measure
    assert np.allclose(sample_sd, expected_sd, rtol=0.05)


def test_off_diagonal_correlation_is_preserved() -> None:
    """The MVN path (not the fallback) must carry correlations through."""
    theta = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    draws = laplace_draws(theta, cov, n_draws=5000, seed=2)
    sample_corr = np.corrcoef(draws, rowvar=False)
    assert abs(sample_corr[0, 1] - 0.7) < 0.05


def test_same_seed_is_deterministic() -> None:
    theta = np.array([1.0, 2.0])
    cov = np.eye(2)
    a = laplace_draws(theta, cov, n_draws=50, seed=123)
    b = laplace_draws(theta, cov, n_draws=50, seed=123)
    assert np.array_equal(a, b)


# --- Ill-conditioned covariance -----------------------------------------


def test_singular_covariance_falls_back_to_empirical_bootstrap() -> None:
    theta = np.array([2.0, 30.0])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # exactly singular
    draws = laplace_draws(theta, cov, n_draws=100, seed=42)
    assert draws.shape == (100, 2)
    # Fallback uses diagonal so correlations vanish. Check that we
    # didn't silently sample from a degenerate MVN (which would
    # collapse onto a line through theta).
    sample_corr = np.corrcoef(draws, rowvar=False)[0, 1]
    assert abs(sample_corr) < 0.3  # weak correlation, not near-perfect


def test_ill_conditioned_cond_number_triggers_fallback() -> None:
    theta = np.array([1.0, 1.0])
    # Cond number ~1e13 — past the 1e12 gate
    cov = np.array([[1e-13, 0.0], [0.0, 1.0]])
    draws = laplace_draws(theta, cov, n_draws=100, seed=7)
    assert draws.shape == (100, 2)


def test_nonfinite_cov_triggers_fallback() -> None:
    theta = np.array([1.0, 1.0])
    cov = np.array([[1.0, float("nan")], [float("nan"), 1.0]])
    draws = laplace_draws(theta, cov, n_draws=20, seed=0)
    assert draws.shape == (20, 2)
    # Degenerate fallback should still produce finite draws (with a
    # default variance floor in place)
    assert np.all(np.isfinite(draws))


def test_fallback_raise_propagates_linalg_error() -> None:
    theta = np.array([2.0, 30.0])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        laplace_draws(theta, cov, n_draws=10, seed=0, fallback="raise")


# --- Input validation ----------------------------------------------------


def test_rejects_non_positive_n_draws() -> None:
    theta = np.array([1.0])
    cov = np.array([[1.0]])
    with pytest.raises(ValueError, match=r"n_draws must be >= 1"):
        laplace_draws(theta, cov, n_draws=0, seed=0)


def test_rejects_mismatched_shapes() -> None:
    theta = np.array([1.0, 2.0, 3.0])
    cov = np.eye(2)
    with pytest.raises(ValueError, match=r"does not match theta length"):
        laplace_draws(theta, cov, n_draws=5, seed=0)


def test_rejects_non_finite_theta() -> None:
    theta = np.array([1.0, float("inf")])
    cov = np.eye(2)
    with pytest.raises(ValueError, match=r"non-finite entries"):
        laplace_draws(theta, cov, n_draws=5, seed=0)


def test_rejects_non_square_cov() -> None:
    theta = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2]])
    with pytest.raises(ValueError, match=r"2-D square"):
        laplace_draws(theta, cov, n_draws=5, seed=0)
