# SPDX-License-Identifier: GPL-2.0-or-later
"""Prior-data conflict diagnostic (plan Task 20).

Compares observed dataset summary statistics ``T(y_obs)`` against the
prior-predictive distribution ``T(y_rep)`` where ``y_rep`` is drawn from
``p(y | theta) p(theta)`` with no data conditioning. A statistic is
"flagged" when its observed value falls outside the prior 95%
predictive interval — Gate 2 (plan Task 20) treats the fraction flagged
as the conflict score and fails the candidate when it exceeds
``Gate2Config.prior_data_conflict_threshold``.

This module deliberately stays *backend-agnostic*: it consumes already-
computed prior-predictive samples and observed summaries, so:

* the harness can call it with samples produced by a fresh
  ``generated_quantities`` Stan pass (the production path), and
* unit tests can exercise it with synthetic numpy arrays without
  requiring cmdstanpy on the test runner.

References:
    Box GEP (1980). Sampling and Bayes' inference in scientific
    modelling and robustness. *J R Stat Soc Ser A* 143:383-430.
    Evans M, Moshonov H (2006). Checking for prior-data conflict.
    *Bayesian Analysis* 1(4):893-914.
    Gabry J, Simpson D, Vehtari A, Betancourt M, Gelman A (2019).
    Visualization in Bayesian workflow. *J R Stat Soc Ser A* 182:389.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from apmode.bundle.models import PriorDataConflict, PriorDataConflictEntry

if TYPE_CHECKING:
    from collections.abc import Mapping


def compute_observed_summary(observations: np.ndarray) -> dict[str, float]:
    """Return the canonical observed-data summary used for conflict checks.

    The keys are the statistic names threaded through to
    ``PriorDataConflictEntry.name`` so the artefact records exactly
    which statistics were checked. Currently a fixed minimal set:

    * ``mean`` — overall mean DV
    * ``sd`` — overall sample SD (ddof=1) — caller must guarantee n≥2
    * ``min`` / ``max`` — extremes (sensitive to dosing regime)
    * ``q05`` / ``q95`` — 5th and 95th percentiles (insensitive to extreme
      outliers but informative about the bulk of the prior)

    Designed to be called once on the observed dataset and once per
    prior-predictive draw; the helper returns a plain ``dict[str,
    float]`` rather than a Pydantic model so the per-draw cost stays
    minimal.
    """
    if observations.ndim != 1:
        msg = f"observations must be 1-D, got shape {observations.shape}"
        raise ValueError(msg)
    if observations.size < 2:
        msg = f"observations must have at least 2 elements, got {observations.size}"
        raise ValueError(msg)
    finite = observations[np.isfinite(observations)]
    if finite.size < 2:
        msg = "observations must contain at least 2 finite values"
        raise ValueError(msg)
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def compute_prior_predictive_summaries(
    prior_predictive_draws: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply :func:`compute_observed_summary` to each prior-predictive replicate.

    ``prior_predictive_draws`` has shape ``(n_draws, n_observations)`` —
    each row is one ``y_rep`` drawn from the prior predictive
    distribution. Returns a dict mapping statistic name to a 1-D
    ``np.ndarray`` of length ``n_draws`` so :func:`compute_prior_data_conflict`
    can take quantiles directly.
    """
    if prior_predictive_draws.ndim != 2:
        msg = (
            f"prior_predictive_draws must be 2-D (n_draws, n_obs), "
            f"got shape {prior_predictive_draws.shape}"
        )
        raise ValueError(msg)
    n_draws, n_obs = prior_predictive_draws.shape
    if n_draws < 1:
        msg = "prior_predictive_draws must have at least one draw"
        raise ValueError(msg)
    if n_obs < 2:
        msg = f"each prior-predictive draw must contain at least 2 observations, got {n_obs}"
        raise ValueError(msg)
    keys = ("mean", "sd", "min", "max", "q05", "q95")
    accumulated: dict[str, list[float]] = {k: [] for k in keys}
    for draw in prior_predictive_draws:
        # Use the same helper as the observed branch so the two
        # statistics arrays are guaranteed to have the same definitions
        # (e.g. ddof=1 SD, np.quantile with the linear interpolation
        # default). Drift between observed and replicated definitions
        # would silently bias the conflict count.
        summary = compute_observed_summary(draw)
        for k in keys:
            accumulated[k].append(summary[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in accumulated.items()}


def compute_prior_data_conflict(
    *,
    candidate_id: str,
    threshold: float,
    observed_summary: Mapping[str, float],
    prior_predictive_summary: Mapping[str, np.ndarray],
    pi_alpha: float = 0.05,
) -> PriorDataConflict:
    """Build a :class:`PriorDataConflict` from observed and prior-predictive summaries.

    For each statistic ``T`` shared by the two summaries the helper
    computes the prior central PI at level ``1 - pi_alpha`` (default
    95%) from the prior-predictive samples, flags the statistic if the
    observed value falls outside, and aggregates the flagged fraction.

    Statistics present in only one of the inputs are silently ignored —
    callers that need strict equality must validate ahead of time.
    Threshold checking is not the helper's job: the gate consumes
    ``conflict_fraction`` and compares it to the lane policy.

    Args:
        candidate_id: Carried through onto the artefact for audit-trail
            traceability.
        threshold: Gate threshold at which Gate 2 will fail. Stored on
            the artefact so a reviewer can recompute the pass/fail
            decision without re-resolving the lane policy.
        observed_summary: Output of :func:`compute_observed_summary` on
            the observed dataset.
        prior_predictive_summary: Output of
            :func:`compute_prior_predictive_summaries` on the harness's
            ``y_rep`` matrix.
        pi_alpha: One minus the PI level. Default 0.05 → 95% PI.
    """
    if not 0.0 < pi_alpha < 1.0:
        msg = f"pi_alpha must be in (0, 1), got {pi_alpha}"
        raise ValueError(msg)
    if not 0.0 <= threshold <= 1.0:
        msg = f"threshold must be in [0, 1], got {threshold}"
        raise ValueError(msg)

    common = sorted(set(observed_summary.keys()) & set(prior_predictive_summary.keys()))
    if not common:
        return PriorDataConflict(
            candidate_id=candidate_id,
            status="not_computed",
            threshold=threshold,
            reason=(
                "no overlapping statistic names between observed and prior-predictive summaries"
            ),
        )

    n_draws = 0
    entries: list[PriorDataConflictEntry] = []
    lower_q = pi_alpha / 2.0
    upper_q = 1.0 - lower_q
    for name in common:
        samples = np.asarray(prior_predictive_summary[name], dtype=np.float64)
        if samples.ndim != 1:
            msg = f"prior_predictive_summary[{name!r}] must be 1-D, got shape {samples.shape}"
            raise ValueError(msg)
        finite = samples[np.isfinite(samples)]
        if finite.size < 2:
            # Skip the statistic when the prior-predictive samples don't
            # support a meaningful PI; record nothing rather than flag a
            # spurious in/out decision.
            continue
        n_draws = max(n_draws, int(finite.size))
        pi_low = float(np.quantile(finite, lower_q))
        pi_high = float(np.quantile(finite, upper_q))
        observed = float(observed_summary[name])
        in_pi = pi_low <= observed <= pi_high
        entries.append(
            PriorDataConflictEntry(
                name=name,
                observed=observed,
                prior_pi_low=pi_low,
                prior_pi_high=pi_high,
                in_pi=in_pi,
                prior_pred_mean=float(np.mean(finite)),
                prior_pred_sd=float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
            )
        )

    if not entries:
        return PriorDataConflict(
            candidate_id=candidate_id,
            status="not_computed",
            threshold=threshold,
            reason="no prior-predictive statistic had >=2 finite samples",
        )

    flagged = sum(1 for e in entries if not e.in_pi)
    fraction = flagged / len(entries)
    return PriorDataConflict(
        candidate_id=candidate_id,
        status="computed",
        threshold=threshold,
        conflict_fraction=fraction,
        n_statistics_total=len(entries),
        n_statistics_flagged=flagged,
        entries=entries,
        n_prior_predictive_draws=n_draws,
        reason=None,
    )
