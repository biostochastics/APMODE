# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Gate 2 Bayesian prior-justification hard-gate (plan Task 19).

Integrates the Task 14 validator's rules (``min_length`` + Crossref DOI)
with the Gate 2 admissibility funnel. Covers:

* Lane policy knob ``bayesian_prior_justification_required`` default:
  Submission=True (enforced), Discovery/Optimization=False (skipped).
* Non-informative priors (uninformative / weakly_informative) pass
  regardless of justification / DOI — the check is scoped to priors
  that claim external evidence.
* Informative priors without DOI or with a too-short justification
  fail Gate 2 with per-entry diagnostic messages.
* ``prior_manifest=None`` on a Bayesian candidate fails when the lane
  requires prior justification (fail-closed); passes when the lane
  does not require it.
* Submission lane's ``min_length=500`` override tightens the floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from apmode.bundle.models import (
    BackendResult,
    ConvergenceMetadata,
    DiagnosticBundle,
    ParameterEstimate,
    PriorManifest,
    PriorManifestEntry,
    SamplerConfig,
    ScoringContract,
)
from apmode.governance.gates import evaluate_gate2
from apmode.governance.policy import GatePolicy

_POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"
_VALID_DOI = "10.1007/BF01064740"
# > 50 chars so the Discovery-lane default floor passes; below 500 so the
# Submission-lane tightened floor rejects.
_MID_JUSTIFICATION = (
    "Meta-analysis of 12 studies (n=840) using a random-effects logistic "
    "model; posterior mean 2.3, 95% CI [1.9, 2.7] on the log scale."
)
# > 500 chars so even the Submission-lane tightened floor passes.
_LONG_JUSTIFICATION = _MID_JUSTIFICATION + (
    " Forest plot reproduced in supplementary figure S2; I-squared 42% "
    "(moderate heterogeneity). Included studies span pediatric to geriatric "
    "populations, three continents, and four dosing regimens (IV bolus, IV "
    "infusion, oral immediate-release, oral extended-release). Residual "
    "between-study variance is 0.09 on the log scale; the chosen LogNormal "
    "prior has sigma=0.3 to match. Excluded five outlier studies flagged by "
    "the GRADE committee for selection bias. Conclusion: the structural "
    "parameter has reproducible meta-analytic support across populations."
)


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
            "CL": ParameterEstimate(
                name="CL",
                estimate=1.0,
                se=0.1,
                rse=10.0,
                ci95_lower=0.8,
                ci95_upper=1.2,
                category="structural",
            )
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


def _manifest(entries: list[PriorManifestEntry], policy_version: str = "0.5.1") -> PriorManifest:
    return PriorManifest(
        policy_version=policy_version,
        entries=entries,
        default_prior_policy="weakly_informative",
    )


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((_POLICY_DIR / f"{lane}.json").read_text()))


def _justification_check(out: object) -> object | None:
    """Find the bayesian_prior_justification check from a GateResult."""
    for c in out.checks:  # type: ignore[attr-defined]
        if c.check_id == "bayesian_prior_justification":
            return c
    return None


# --- Discovery lane (requirement off) ------------------------------------


def test_discovery_lane_skips_check_regardless_of_manifest() -> None:
    res = _bayesian_result()
    policy = _load_policy("discovery")
    # Even an obviously deficient prior doesn't trip Gate 2 on Discovery
    bad = PriorManifestEntry(
        target="CL",
        family="LogNormal",
        source="meta_analysis",
        hyperparams={"mu": 2.3, "sigma": 0.3},
        justification="too short",
        doi=None,
    )
    out = evaluate_gate2(res, policy, lane="discovery", prior_manifest=_manifest([bad]))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert check.observed == "not_required"  # type: ignore[attr-defined]


# --- Submission lane (requirement on) ------------------------------------


def test_submission_lane_passes_with_valid_informative_prior() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    good = PriorManifestEntry(
        target="CL",
        family="LogNormal",
        source="meta_analysis",
        hyperparams={"mu": 2.3, "sigma": 0.3},
        justification=_LONG_JUSTIFICATION,
        doi=_VALID_DOI,
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=_manifest([good]))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]


def test_submission_lane_rejects_missing_doi() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    bad = PriorManifestEntry(
        target="CL",
        family="LogNormal",
        source="meta_analysis",
        hyperparams={"mu": 2.3, "sigma": 0.3},
        justification=_LONG_JUSTIFICATION,
        doi=None,
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=_manifest([bad]))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "doi missing" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_rejects_too_short_justification() -> None:
    """Submission tightens min_length to 500 — a 200-char blurb isn't enough."""
    res = _bayesian_result()
    policy = _load_policy("submission")
    bad = PriorManifestEntry(
        target="CL",
        family="LogNormal",
        source="meta_analysis",
        hyperparams={"mu": 2.3, "sigma": 0.3},
        justification=_MID_JUSTIFICATION,  # ~200 chars — under 500
        doi=_VALID_DOI,
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=_manifest([bad]))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "< 500" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_aggregates_per_entry_errors() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    entries = [
        PriorManifestEntry(
            target="CL",
            family="LogNormal",
            source="meta_analysis",
            hyperparams={"mu": 2.3, "sigma": 0.3},
            justification="too short",
            doi=None,
        ),
        PriorManifestEntry(
            target="V",
            family="LogNormal",
            source="expert_elicitation",
            hyperparams={"mu": 3.5, "sigma": 0.5},
            justification=_LONG_JUSTIFICATION,
            doi=None,  # still missing DOI
        ),
    ]
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=_manifest(entries))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    observed = check.observed  # type: ignore[attr-defined]
    assert "entries[0]" in observed
    assert "entries[1]" in observed
    assert "'CL'" in observed and "'V'" in observed


def test_submission_lane_passes_weakly_informative_priors() -> None:
    res = _bayesian_result()
    policy = _load_policy("submission")
    entry = PriorManifestEntry(
        target="CL",
        family="LogNormal",
        source="weakly_informative",
        hyperparams={"mu": 0.0, "sigma": 2.0},
        justification="",  # allowed because source is weakly_informative
        doi=None,
    )
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=_manifest([entry]))
    check = _justification_check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert "no_informative_priors" in check.observed  # type: ignore[attr-defined]


def test_submission_lane_fails_when_manifest_missing() -> None:
    """Fail-closed when the lane demands justification but nothing is attached."""
    res = _bayesian_result()
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=None)
    check = _justification_check(out)
    assert check is not None
    assert check.passed is False  # type: ignore[attr-defined]
    assert "prior_manifest_missing" in check.observed  # type: ignore[attr-defined]


def test_non_bayesian_backend_passes_even_when_required() -> None:
    """MLE candidates have no priors — the check must not block them."""
    res = _bayesian_result(backend="nlmixr2")
    policy = _load_policy("submission")
    out = evaluate_gate2(res, policy, lane="submission", prior_manifest=None)
    check = _justification_check(out)
    assert check is not None
    assert check.passed is True  # type: ignore[attr-defined]
    assert "not_bayesian" in check.observed  # type: ignore[attr-defined]
