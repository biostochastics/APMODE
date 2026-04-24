# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 3 ``metric_stack`` + ``laplace_draws`` (plan Task 24).

Guardrails covered:

* ``Gate3Config.metric_stack`` must be non-empty (a lane must declare at
  least one metric; otherwise Gate 3 would reject every candidate).
* ``Gate3Config.validate_metric`` raises :class:`PolicyError` for any
  metric outside the whitelist — this is the anti-metric-shopping gate.
* All three lane policies load with the expected ``metric_stack`` +
  ``laplace_draws`` values. Submission leads with 2000 draws (tight
  intervals for regulatory review); Discovery leaves the default 500.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.governance.policy import (
    Gate3Config,
    GatePolicy,
    PolicyError,
)

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _load(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((_POLICY_DIR / f"{lane}.json").read_text()))


# --- metric_stack --------------------------------------------------------


def test_default_metric_stack_covers_three_metrics() -> None:
    cfg = Gate3Config()
    assert cfg.metric_stack == ["vpc_concordance", "auc_cmax_gmr", "npe"]


def test_empty_metric_stack_rejected() -> None:
    with pytest.raises(ValueError, match=r"at least one metric"):
        Gate3Config(metric_stack=[])


def test_validate_metric_passes_for_whitelisted() -> None:
    cfg = Gate3Config()
    # No return value, no exception = pass
    cfg.validate_metric("vpc_concordance")
    cfg.validate_metric("auc_cmax_gmr")
    cfg.validate_metric("npe")


def test_validate_metric_raises_policy_error_for_off_list() -> None:
    cfg = Gate3Config()
    with pytest.raises(PolicyError, match=r"not on this lane's metric_stack"):
        cfg.validate_metric("custom_secret_metric")


def test_restricted_metric_stack_blocks_default_metrics() -> None:
    """If a lane excludes a metric from its stack, validate_metric rejects it."""
    cfg = Gate3Config(metric_stack=["vpc_concordance"])
    cfg.validate_metric("vpc_concordance")
    with pytest.raises(PolicyError):
        cfg.validate_metric("npe")


# --- laplace_draws -------------------------------------------------------


def test_default_laplace_draws() -> None:
    cfg = Gate3Config()
    assert cfg.laplace_draws == 500


def test_laplace_draws_lower_bound() -> None:
    with pytest.raises(ValueError, match=r"greater than or equal to 100"):
        Gate3Config(laplace_draws=50)


def test_laplace_draws_upper_bound() -> None:
    with pytest.raises(ValueError, match=r"less than or equal to 10000"):
        Gate3Config(laplace_draws=20000)


# --- Lane-policy integration --------------------------------------------


def test_submission_lane_uses_2000_laplace_draws() -> None:
    policy = _load("submission")
    assert policy.gate3.laplace_draws == 2000
    assert set(policy.gate3.metric_stack) == {"vpc_concordance", "auc_cmax_gmr", "npe"}


def test_discovery_lane_uses_default_500_draws() -> None:
    policy = _load("discovery")
    assert policy.gate3.laplace_draws == 500


def test_optimization_lane_uses_intermediate_draws() -> None:
    policy = _load("optimization")
    assert policy.gate3.laplace_draws == 1000


def test_policy_error_inherits_from_value_error() -> None:
    """Existing except-handlers that catch ValueError should still work."""
    cfg = Gate3Config()
    try:
        cfg.validate_metric("bogus")
    except ValueError:
        pass  # expected — PolicyError inherits from ValueError
    else:  # pragma: no cover
        pytest.fail("expected ValueError via PolicyError subclass")
