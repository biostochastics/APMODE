# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for ``Gate1BayesianConfig`` and per-lane overrides (plan Task 16).

The Gate 1 Bayesian block carries per-parameter-class R-hat / ESS floors
plus scalar knobs for divergence tolerance, tree-depth saturation, E-BFMI,
and Pareto-k. The per-lane severity dict determines whether a violation
fails the gate (``fail``) or only raises a warning (``warn``).

Invariants tested here:

* All three lane policies parse into ``GatePolicy`` and expose a
  ``gate1_bayesian`` block that matches the documented lane overrides.
* Defaults follow Vehtari et al. (2021): R-hat ≤ 1.01 for fixed effects /
  residual, 1.02 for IIV, 1.05 for correlations; bulk/tail ESS ≥ 400 for
  fixed / IIV / residual and ≥ 100 for correlations.
* Severity map must cover the four diagnostic axes (rhat, ess,
  divergences, pareto_k) with values in ``{"warn", "fail"}``.
* On-disk ``policies/*.json`` must contain the new block (non-destructive
  merge): the pre-existing ``gate2_5`` block on submission.json is
  preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.governance.policy import (
    Gate1BayesianConfig,
    Gate1BayesianESS,
    Gate1BayesianRhat,
    GatePolicy,
)

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _load(lane: str) -> GatePolicy:
    path = _POLICY_DIR / f"{lane}.json"
    return GatePolicy.model_validate(json.loads(path.read_text()))


# --- Default values ------------------------------------------------------


def test_default_rhat_max_matches_vehtari_2021() -> None:
    cfg = Gate1BayesianConfig()
    assert cfg.rhat_max.fixed_effects == 1.01
    assert cfg.rhat_max.iiv == 1.02
    assert cfg.rhat_max.residual == 1.01
    assert cfg.rhat_max.correlations == 1.05


def test_default_ess_floors_match_arviz_recommendation() -> None:
    cfg = Gate1BayesianConfig()
    assert cfg.ess_bulk_min.fixed_effects == 400
    assert cfg.ess_bulk_min.iiv == 400
    assert cfg.ess_bulk_min.residual == 400
    assert cfg.ess_bulk_min.correlations == 100
    # Tail ESS mirrors bulk by default; lanes may diverge if needed.
    assert cfg.ess_tail_min.fixed_effects == 400


def test_default_scalar_knobs() -> None:
    cfg = Gate1BayesianConfig()
    assert cfg.divergence_tolerance == 0
    assert cfg.max_treedepth_fraction == 0.01
    assert cfg.e_bfmi_min == 0.3
    assert cfg.pareto_k_max == 0.7


def test_default_severity_map_covers_six_axes() -> None:
    cfg = Gate1BayesianConfig()
    assert set(cfg.severity.keys()) == {
        "rhat",
        "ess",
        "divergences",
        "treedepth",
        "ebfmi",
        "pareto_k",
    }
    assert cfg.severity["rhat"] == "fail"
    assert cfg.severity["ess"] == "fail"
    assert cfg.severity["divergences"] == "fail"
    # Tree-depth saturation is warn-by-default — Stan recommends it as a
    # tuning hint, not a sampling-correctness disqualifier (Betancourt 2017).
    assert cfg.severity["treedepth"] == "warn"
    # E-BFMI fails by default — values < 0.3 indicate the sampler cannot
    # explore the posterior efficiently and the fit's tail estimates are
    # unreliable (Betancourt 2017 §6.1).
    assert cfg.severity["ebfmi"] == "fail"
    # Pareto-k is warn-only by default — LOO is an informative check, not
    # a disqualifier, until a canonical log_lik declaration is mandated.
    assert cfg.severity["pareto_k"] == "warn"


# --- Bounds --------------------------------------------------------------


def test_rhat_rejects_value_at_or_below_one() -> None:
    """R-hat < 1 is impossible; schema must reject to catch typos."""
    with pytest.raises(ValueError, match="greater than 1"):
        Gate1BayesianRhat(fixed_effects=1.0)


def test_ess_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        Gate1BayesianESS(fixed_effects=0)


def test_severity_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="validation error"):
        Gate1BayesianConfig(severity={"rhat": "panic"})  # type: ignore[dict-item]


# --- Policy-file integration --------------------------------------------


def test_submission_policy_uses_strict_defaults() -> None:
    policy = _load("submission")
    cfg = policy.gate1_bayesian
    assert cfg.rhat_max.fixed_effects == 1.01
    assert cfg.pareto_k_max == 0.7
    assert cfg.severity["pareto_k"] == "warn"


def test_discovery_policy_relaxes_fixed_effects_rhat() -> None:
    """Discovery lane allows wider R-hat on fixed effects for early screening."""
    policy = _load("discovery")
    cfg = policy.gate1_bayesian
    assert cfg.rhat_max.fixed_effects == 1.05
    # IIV / residual / correlations remain at defaults
    assert cfg.rhat_max.iiv == 1.02
    assert cfg.rhat_max.residual == 1.01
    assert cfg.rhat_max.correlations == 1.05
    assert cfg.severity["pareto_k"] == "warn"


def test_optimization_policy_tightens_pareto_k_and_fails() -> None:
    """Optimization lane is LOO-gated: tighter Pareto-k + hard fail."""
    policy = _load("optimization")
    cfg = policy.gate1_bayesian
    assert cfg.pareto_k_max == 0.5
    assert cfg.severity["pareto_k"] == "fail"


# --- Non-destructive policy merge ---------------------------------------


def test_submission_gate2_5_still_present() -> None:
    """Task 1's gate2_5 block must survive the Task 16 merge."""
    raw = json.loads((_POLICY_DIR / "submission.json").read_text())
    assert "gate2_5" in raw
    assert "gate1_bayesian" in raw
    # Top-level order preservation is nice-to-have, not load-bearing;
    # the presence check is the real invariant.


def test_all_three_lanes_carry_gate1_bayesian_block() -> None:
    for lane in ("submission", "discovery", "optimization"):
        raw = json.loads((_POLICY_DIR / f"{lane}.json").read_text())
        assert "gate1_bayesian" in raw, f"{lane}.json missing gate1_bayesian"
