# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :class:`apmode.bundle.models.MetricTuple` (plan Task 23).

MetricTuple is the commensurate interval carrier that Gate 3 ranking
consumes for both Bayesian (real posterior draws) and MLE (Laplace /
empirical-bootstrap) candidates. Tests cover:

* The three valid ``method`` tags round-trip through ``model_dump``.
* The ``ci_low <= mean <= ci_high`` invariant is enforced at construction.
* A narrower tuple (Bayesian) and a wider tuple (Laplace-bootstrap)
  dump identical key-sets — Gate 3 can read both the same way.
* ``ci_level`` defaults to 0.95 and rejects values outside ``(0, 1)``.
"""

from __future__ import annotations

import pytest

from apmode.bundle.models import MetricTuple


def test_valid_tuple_round_trip() -> None:
    tup = MetricTuple(
        mean=0.5,
        ci_low=0.45,
        ci_high=0.55,
        method="posterior_draws",
    )
    dump = tup.model_dump()
    assert dump["method"] == "posterior_draws"
    assert dump["ci_level"] == 0.95


def test_mean_out_of_ci_bounds_rejected() -> None:
    with pytest.raises(ValueError, match=r"ci_low <= mean <= ci_high"):
        MetricTuple(
            mean=1.0,
            ci_low=1.5,
            ci_high=2.0,
            method="posterior_draws",
        )


def test_ci_low_above_ci_high_rejected() -> None:
    """Cross-paradigm invariant must hold even under swapped bounds."""
    with pytest.raises(ValueError, match=r"ci_low <= mean <= ci_high"):
        MetricTuple(
            mean=0.5,
            ci_low=0.6,
            ci_high=0.4,
            method="posterior_draws",
        )


@pytest.mark.parametrize(
    "method",
    ["posterior_draws", "laplace_draws", "empirical_bootstrap"],
)
def test_all_methods_round_trip(method: str) -> None:
    tup = MetricTuple(
        mean=0.5,
        ci_low=0.45,
        ci_high=0.55,
        method=method,  # type: ignore[arg-type]
    )
    assert tup.method == method


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match=r"validation error"):
        MetricTuple(
            mean=0.5,
            ci_low=0.45,
            ci_high=0.55,
            method="bogus",  # type: ignore[arg-type]
        )


def test_ci_level_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"greater than 0"):
        MetricTuple(
            mean=0.5,
            ci_low=0.45,
            ci_high=0.55,
            method="posterior_draws",
            ci_level=0.0,
        )
    with pytest.raises(ValueError, match=r"less than 1"):
        MetricTuple(
            mean=0.5,
            ci_low=0.45,
            ci_high=0.55,
            method="posterior_draws",
            ci_level=1.0,
        )


def test_bayesian_and_mle_tuples_have_identical_keys() -> None:
    """The whole point is Gate 3 can iterate keys uniformly."""
    bayes = MetricTuple(
        mean=0.5,
        ci_low=0.48,
        ci_high=0.52,
        method="posterior_draws",
    )
    mle = MetricTuple(
        mean=0.5,
        ci_low=0.30,
        ci_high=0.70,
        method="laplace_draws",
    )
    assert bayes.model_dump().keys() == mle.model_dump().keys()


def test_empirical_bootstrap_allowed_even_when_correlations_lost() -> None:
    """Ill-conditioned MLE fits should still produce a MetricTuple."""
    tup = MetricTuple(
        mean=1.0,
        ci_low=0.1,  # much wider than expected
        ci_high=2.5,
        method="empirical_bootstrap",
    )
    assert tup.method == "empirical_bootstrap"
