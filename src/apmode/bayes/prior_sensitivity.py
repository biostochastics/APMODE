# SPDX-License-Identifier: GPL-2.0-or-later
"""Prior-sensitivity diagnostic (plan Task 21).

Compares each structural parameter's posterior mean under the
*baseline* prior to its posterior mean under N≥2 *alternative* priors,
expressed in baseline-posterior-SD units. Gate 2 (plan Task 21) treats
the maximum normalised delta as the sensitivity score and fails the
candidate when it exceeds ``Gate2Config.sensitivity_max_delta``.

This module deliberately stays *backend-agnostic*: it consumes already-
computed posterior summaries from N+1 Stan refits, so:

* the harness can call it with summaries produced by alternative-prior
  refits (the production path), and
* unit tests can exercise it with synthetic dicts without requiring
  cmdstanpy.

References:
    Roos M, Held L (2011). Sensitivity analysis in Bayesian generalized
    linear mixed models. *Bayesian Analysis* 6(2):259-278.
    Roos M, Martins TG, Held L, Rue H (2015). Sensitivity analysis for
    Bayesian hierarchical models. *Bayesian Analysis* 10(2):321-349.
    Kallioinen N, Paananen T, Bürkner PC, Vehtari A (2024). Detecting
    and diagnosing prior and likelihood sensitivity with power-scaling.
    *Statistics and Computing* 34:57.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.bundle.models import PriorSensitivity, PriorSensitivityEntry

if TYPE_CHECKING:
    from collections.abc import Mapping


def compute_prior_sensitivity(
    *,
    candidate_id: str,
    threshold: float,
    baseline_means: Mapping[str, float],
    baseline_sds: Mapping[str, float],
    alternative_means: Mapping[str, Mapping[str, float]],
) -> PriorSensitivity:
    """Build a :class:`PriorSensitivity` from baseline + alternative posteriors.

    For every (parameter, alternative) pair where both the baseline and
    alternative means are present the helper computes
    ``|Δmean| / posterior_sd_baseline`` and emits one entry. The maximum
    over all entries is the gate-eligible score.

    Args:
        candidate_id: Carried through onto the artefact for audit-trail
            traceability.
        threshold: Gate threshold at which Gate 2 will fail. Stored on
            the artefact and used to compute ``flagged_parameters``.
        baseline_means: ``{param_name: posterior_mean}`` from the
            baseline (production) prior fit.
        baseline_sds: ``{param_name: posterior_sd}`` from the baseline
            fit; supplies the denominator. Must be strictly positive
            for every parameter that also appears in ``alternative_means``.
        alternative_means: ``{alternative_id: {param_name: posterior_mean}}``
            for each alternative prior. Length must be ≥ 2 for the
            artefact to be emitted as ``status="computed"``; below that
            the helper returns ``status="not_computed"`` with the
            reason recorded.

    Returns:
        :class:`PriorSensitivity` with ``status="computed"`` when at
        least one (parameter, alternative) pair was scoreable, and
        ``status="not_computed"`` otherwise (with an explanatory reason).
    """
    if threshold < 0.0:
        msg = f"threshold must be non-negative, got {threshold}"
        raise ValueError(msg)

    if len(alternative_means) < 2:
        return PriorSensitivity(
            candidate_id=candidate_id,
            status="not_computed",
            threshold=threshold,
            reason=(f"need >=2 alternative priors per parameter, got {len(alternative_means)}"),
        )

    common_params = set(baseline_means.keys()) & set(baseline_sds.keys())
    for alt_means in alternative_means.values():
        common_params &= set(alt_means.keys())
    common_params_sorted = sorted(common_params)

    if not common_params_sorted:
        return PriorSensitivity(
            candidate_id=candidate_id,
            status="not_computed",
            threshold=threshold,
            reason="no parameter appears in baseline + every alternative",
        )

    entries: list[PriorSensitivityEntry] = []
    for param in common_params_sorted:
        sd_base = baseline_sds[param]
        if sd_base <= 0.0:
            # Skip non-finite or zero-SD parameters; they cannot
            # produce a meaningful normalised delta and would otherwise
            # blow the artefact up via the entry validator.
            continue
        mean_base = baseline_means[param]
        for alt_id in sorted(alternative_means.keys()):
            mean_alt = alternative_means[alt_id][param]
            delta = abs(mean_alt - mean_base) / sd_base
            entries.append(
                PriorSensitivityEntry(
                    parameter=param,
                    alternative_id=alt_id,
                    posterior_mean_baseline=mean_base,
                    posterior_mean_alternative=mean_alt,
                    posterior_sd_baseline=sd_base,
                    delta_normalized=delta,
                )
            )

    if not entries:
        return PriorSensitivity(
            candidate_id=candidate_id,
            status="not_computed",
            threshold=threshold,
            reason="no parameter had a positive baseline posterior SD",
        )

    max_delta = max(e.delta_normalized for e in entries)
    flagged_parameters = sorted({e.parameter for e in entries if e.delta_normalized > threshold})
    n_parameters = len({e.parameter for e in entries})
    return PriorSensitivity(
        candidate_id=candidate_id,
        status="computed",
        threshold=threshold,
        max_delta=max_delta,
        n_parameters=n_parameters,
        n_alternatives_per_parameter=len(alternative_means),
        entries=entries,
        flagged_parameters=flagged_parameters,
        reason=None,
    )
