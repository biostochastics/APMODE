# SPDX-License-Identifier: GPL-2.0-or-later
"""Convergence-flag helper for the Bayesian harness.

The harness previously hard-coded ``converged=True`` for any non-
catastrophic run, which conflated sub-threshold samplers with
trustworthy fits. ``_is_converged`` now enforces a conservative floor
(R-hat <= 1.05, bulk ESS >= 400, divergences == 0) before the
BackendResult is stamped converged.
"""

from __future__ import annotations

from apmode.bayes.harness import _is_converged


def _diag(
    rhat: float = 1.0,
    ess: float = 500.0,
    divergent: int = 0,
) -> dict[str, float | int]:
    return {"rhat_max": rhat, "ess_bulk_min": ess, "n_divergent": divergent}


def test_converged_when_all_thresholds_met() -> None:
    assert _is_converged(_diag(rhat=1.01, ess=800.0, divergent=0)) is True


def test_not_converged_when_rhat_above_floor() -> None:
    assert _is_converged(_diag(rhat=1.06)) is False


def test_not_converged_when_ess_below_floor() -> None:
    assert _is_converged(_diag(ess=399.0)) is False


def test_not_converged_when_any_divergence() -> None:
    assert _is_converged(_diag(divergent=1)) is False


def test_boundary_values_accept_floor() -> None:
    """Exactly at the floor still counts as converged (<= / >=)."""
    assert _is_converged(_diag(rhat=1.05, ess=400.0, divergent=0)) is True
