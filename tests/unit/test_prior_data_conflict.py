# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Gate 2 prior-data conflict diagnostic (plan Task 20).

Three layers, mirroring the production wiring:

1. :func:`compute_prior_data_conflict` — pure-Python helper that
   converts observed + prior-predictive summaries into a
   :class:`PriorDataConflict`.
2. :class:`PriorDataConflict` schema invariants (status / counts /
   per-entry consistency).
3. ``_check_prior_data_conflict`` Gate 2 wiring (skip / fail-closed /
   pass / fail).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from apmode.bayes.prior_data_conflict import (
    compute_observed_summary,
    compute_prior_data_conflict,
    compute_prior_predictive_summaries,
)
from apmode.bundle.models import (
    BackendResult,
    ConvergenceMetadata,
    DiagnosticBundle,
    ParameterEstimate,
    PriorDataConflict,
    PriorDataConflictEntry,
    SamplerConfig,
    ScoringContract,
)
from apmode.governance.gates import evaluate_gate2
from apmode.governance.policy import GatePolicy

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"


def _bayesian_result(*, backend: str = "bayesian_stan") -> BackendResult:
    scoring_contract = ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="hmc_nuts" if backend == "bayesian_stan" else "nlmixr2_focei",
        blq_method="none",
        observation_model="additive",
        float_precision="float64",
    )
    return BackendResult(
        model_id="cand001",
        backend=cast("type[str]", backend),  # type: ignore[arg-type]
        converged=True,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=1.0, category="structural")
        },
        eta_shrinkage={},
        convergence_metadata=ConvergenceMetadata(
            method="nuts",
            converged=True,
            iterations=2000,
            minimization_status="successful",
            wall_time_seconds=1.0,
        ),
        diagnostics=DiagnosticBundle(
            gof={"cwres_mean": 0.0, "cwres_sd": 1.0, "outlier_fraction": 0.0},
            identifiability={"profile_likelihood_ci": {}, "ill_conditioned": False},
            blq={"method": "none", "n_blq": 0, "blq_fraction": 0.0},
            scoring_contract=scoring_contract,
        ),
        wall_time_seconds=1.0,
        backend_versions={},
        initial_estimate_source="fallback",
        sampler_config=SamplerConfig() if backend == "bayesian_stan" else None,
    )


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((_POLICY_DIR / f"{lane}.json").read_text()))


def _check(out: object) -> object | None:
    for c in out.checks:  # type: ignore[attr-defined]
        if c.check_id == "prior_data_conflict":
            return c
    return None


# --- Helper-level tests ---------------------------------------------------


def test_compute_observed_summary_keys_are_canonical() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    summary = compute_observed_summary(obs)
    assert sorted(summary.keys()) == ["max", "mean", "min", "q05", "q95", "sd"]
    assert summary["mean"] == pytest.approx(4.5)
    assert summary["min"] == 1.0
    assert summary["max"] == 8.0


def test_compute_observed_summary_rejects_too_few_values() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compute_observed_summary(np.array([1.0]))


def test_compute_observed_summary_drops_non_finite() -> None:
    obs = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
    summary = compute_observed_summary(obs)
    # Only the three finite values should contribute.
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_prior_predictive_summaries_shape() -> None:
    rng = np.random.default_rng(42)
    draws = rng.normal(loc=3.0, scale=1.0, size=(50, 6))
    summaries = compute_prior_predictive_summaries(draws)
    assert sorted(summaries.keys()) == ["max", "mean", "min", "q05", "q95", "sd"]
    for arr in summaries.values():
        assert arr.shape == (50,)


def test_compute_prior_data_conflict_flags_when_observed_outside_pi() -> None:
    rng = np.random.default_rng(7)
    # Prior predictive draws centred near 3 with small SD; observed mean
    # of 100 sits well outside the 95% PI for the mean statistic.
    draws = rng.normal(loc=3.0, scale=0.5, size=(500, 8))
    ppred = compute_prior_predictive_summaries(draws)
    observed = compute_observed_summary(
        np.array([100.0, 101.0, 102.0, 99.0, 100.5, 101.5, 100.2, 100.7])
    )
    artifact = compute_prior_data_conflict(
        candidate_id="cand001",
        threshold=0.05,
        observed_summary=observed,
        prior_predictive_summary=ppred,
    )
    assert artifact.status == "computed"
    assert artifact.conflict_fraction is not None
    assert artifact.conflict_fraction > 0.5
    flagged_names = {e.name for e in artifact.entries if not e.in_pi}
    assert "mean" in flagged_names


def test_compute_prior_data_conflict_passes_when_observed_inside_pi() -> None:
    rng = np.random.default_rng(11)
    draws = rng.normal(loc=3.0, scale=1.0, size=(2000, 32))
    ppred = compute_prior_predictive_summaries(draws)
    # Observed sample drawn from the same distribution as the prior
    # predictive — every summary statistic should land inside the 95% PI
    # so the candidate passes the gate cleanly.
    observed = compute_observed_summary(rng.normal(loc=3.0, scale=1.0, size=32))
    artifact = compute_prior_data_conflict(
        candidate_id="cand001",
        threshold=0.05,
        observed_summary=observed,
        prior_predictive_summary=ppred,
    )
    assert artifact.status == "computed"
    assert artifact.n_statistics_flagged == 0
    assert artifact.conflict_fraction == pytest.approx(0.0, abs=1e-9)


def test_compute_prior_data_conflict_returns_not_computed_on_no_overlap() -> None:
    artifact = compute_prior_data_conflict(
        candidate_id="cand001",
        threshold=0.05,
        observed_summary={"foo": 1.0},
        prior_predictive_summary={"bar": np.array([1.0, 2.0, 3.0])},
    )
    assert artifact.status == "not_computed"
    assert artifact.reason is not None
    assert "no overlapping" in artifact.reason


# --- Schema invariants ----------------------------------------------------


def test_prior_data_conflict_status_computed_requires_fraction() -> None:
    with pytest.raises(ValueError, match="conflict_fraction"):
        PriorDataConflict(
            candidate_id="cand001",
            status="computed",
            threshold=0.05,
        )


def test_prior_data_conflict_status_not_computed_requires_reason() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        PriorDataConflict(
            candidate_id="cand001",
            status="not_computed",
            threshold=0.05,
        )


def test_prior_data_conflict_entry_in_pi_must_match_bounds() -> None:
    with pytest.raises(ValueError, match="contradicts observed"):
        PriorDataConflictEntry(
            name="mean",
            observed=10.0,
            prior_pi_low=0.0,
            prior_pi_high=1.0,
            in_pi=True,
        )


def test_prior_data_conflict_n_flagged_must_match_entries() -> None:
    entry_in = PriorDataConflictEntry(
        name="mean", observed=0.5, prior_pi_low=0.0, prior_pi_high=1.0, in_pi=True
    )
    with pytest.raises(ValueError, match="n_statistics_flagged"):
        PriorDataConflict(
            candidate_id="cand001",
            status="computed",
            threshold=0.05,
            conflict_fraction=0.0,
            n_statistics_total=1,
            n_statistics_flagged=99,
            entries=[entry_in],
        )


# --- Gate 2 wiring --------------------------------------------------------


def _good_artifact() -> PriorDataConflict:
    entry = PriorDataConflictEntry(
        name="mean", observed=0.5, prior_pi_low=0.0, prior_pi_high=1.0, in_pi=True
    )
    return PriorDataConflict(
        candidate_id="cand001",
        status="computed",
        threshold=0.05,
        conflict_fraction=0.0,
        n_statistics_total=1,
        n_statistics_flagged=0,
        entries=[entry],
    )


def _bad_artifact() -> PriorDataConflict:
    bad = PriorDataConflictEntry(
        name="mean",
        observed=10.0,
        prior_pi_low=0.0,
        prior_pi_high=1.0,
        in_pi=False,
    )
    return PriorDataConflict(
        candidate_id="cand001",
        status="computed",
        threshold=0.05,
        conflict_fraction=1.0,
        n_statistics_total=1,
        n_statistics_flagged=1,
        entries=[bad],
    )


def test_discovery_lane_skips_check_regardless_of_artifact() -> None:
    res = _bayesian_result()
    policy = _load_policy("discovery")
    out = evaluate_gate2(res, policy, lane="discovery", prior_data_conflict=_bad_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert check.observed == "not_required"  # type: ignore[attr-defined]


def test_submission_lane_passes_with_in_pi_artifact() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_data_conflict=_good_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]


def test_submission_lane_fails_when_observed_outside_pi() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_data_conflict=_bad_artifact())
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "fraction=1.000" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_fails_when_artifact_missing() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_data_conflict=None)
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "artifact_missing" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_fails_on_not_computed_status() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    artifact = PriorDataConflict(
        candidate_id="cand001",
        status="not_computed",
        threshold=0.05,
        reason="cmdstanpy unavailable",
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_data_conflict=artifact)
    check = _check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "not_computed" in check.observed  # type: ignore[attr-defined]
    assert "cmdstanpy unavailable" in check.observed  # type: ignore[attr-defined]


def test_non_bayesian_backend_passes_even_when_required() -> None:
    res = _bayesian_result(backend="nlmixr2")
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_data_conflict=None)
    check = _check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert "not_bayesian_backend" in check.observed  # type: ignore[attr-defined]
