# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared posterior-predictive diagnostics (PRD §4.3.1 Gate 3 cross-paradigm).

Single canonical helper that turns per-subject posterior-predictive
simulations into the three Gate 3 simulation-based diagnostics
*atomically*:

* **VPC (visual predictive check)** — time-binned empirical coverage of
  observed percentiles under the simulated percentile's central CI
  (xpose4/PsN "percentile-based VPC with confidence intervals"
  convention). Binning is post-hoc over observation times; there is no
  pre-declared VPC simulation grid.
* **NPE** — median absolute prediction error from the per-sim median
  trajectory at observed times
  (:func:`apmode.benchmarks.scoring.compute_npe`).
* **AUC/Cmax bioequivalence** — per-subject GMR of candidate-sim AUC and
  Cmax vs an observed-data NCA reference, counted as BE-pass when both
  GMRs land in the Smith 2000 FDA goalposts ``[0.80, 1.25]``. Per-subject
  NCA eligibility (``NCASubjectDiagnostic.excluded``) gates which
  subjects count
  (:func:`apmode.benchmarks.scoring.compute_auc_cmax_be_score`).

**Single-solve design.** Backends forward-solve the structural model at
``n_posterior_predictive_sims`` ETA draws *only* at each subject's
observed times. One solve per subject per sim draw supplies all three
diagnostics; no second pass at a pooled VPC grid. This eliminates the
Cmax interpolation bias of a "simulate on shared grid + interpolate
observations" shortcut and halves the ODE cost of a "simulate on both
shared and native grids" alternative. VPC smoothness is traded for
computational simplicity and correctness — bin math is straightforward
when every simulated point matches an observed point.

**Partial-population ban.** Backends must call this helper once at
``BackendResult`` construction time and populate ``vpc``, ``npe_score``,
``auc_cmax_be_score``, and ``auc_cmax_source`` on the
:class:`~apmode.bundle.models.DiagnosticBundle` from the returned
:class:`PredictiveSummaryBundle`. Cherry-picking (e.g. emitting VPC but
omitting NPE) is prohibited — see the ``DiagnosticBundle`` docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apmode.benchmarks.scoring import (
    compute_auc_cmax_be_score,
    compute_npe,
    is_nca_eligible_per_subject,
)
from apmode.bundle.models import PITCalibrationSummary, VPCSummary, _pit_key

if TYPE_CHECKING:
    from apmode.bundle.models import NCASubjectDiagnostic
    from apmode.governance.policy import Gate3Config


logger = logging.getLogger(__name__)

# Taxonomy of NCA-ineligibility reasons surfaced on PredictiveSummaryBundle.
# Keys are mapped from ``NCASubjectDiagnostic.excluded_reason`` substrings.
# Kept as a fixed vocabulary so the audit log aggregates stably across runs.
_MASK_DROP_KEYS: tuple[str, ...] = (
    "absorption",
    "elimination",
    "blq",
    "span",
    "lambda_z",
    "auc_extrap",
    "missing",
    "other",
)


def _classify_exclusion_reason(reason: str | None) -> str:
    """Bucket a free-text exclusion reason into the fixed taxonomy."""
    if not reason:
        return "other"
    low = reason.lower()
    if "absorption" in low:
        return "absorption"
    if "elimination" in low:
        return "elimination"
    if "blq" in low or "lloq" in low or "censoring" in low:
        return "blq"
    if "span" in low:
        return "span"
    if "lambda" in low or "λz" in low or "λ_z" in low:
        return "lambda_z"
    if "auc" in low and "extrap" in low:
        return "auc_extrap"
    if "missing" in low or "insufficient" in low or "n_points" in low:
        return "missing"
    return "other"


__all__ = [
    "PredictiveSummaryBundle",
    "SubjectSimulation",
    "build_predictive_diagnostics",
]


@dataclass(frozen=True)
class SubjectSimulation:
    """Per-subject posterior-predictive simulation matrix + observed data.

    Backends populate this by (1) sampling ``n_sims`` ETA vectors from
    the fitted Ω, (2) forward-solving the structural model at each
    subject's observed time vector for every draw, and (3) packaging the
    result together with the observed concentrations and the subject's
    NCA QC record.

    Arrays are immutable after construction: ``__post_init__`` flips the
    numpy ``writeable`` flag on each array to ``False``, and the
    dataclass is frozen so references cannot be rebound. ``nca_diagnostic``
    is optional because some backends (or pre-NCA workflows) may not
    carry a per-subject QC record — those subjects still contribute to
    VPC and NPE but are treated as NCA-ineligible (mask-dropped) in the
    AUC/Cmax BE computation.
    """

    subject_id: str
    t_observed: np.ndarray
    """Observation times for this subject, shape ``(n_obs_i,)``."""
    observed_dv: np.ndarray
    """Observed DV at each time, shape ``(n_obs_i,)``."""
    sims_at_observed: np.ndarray
    """Simulated DV at each observed time, shape ``(n_sims, n_obs_i)``."""
    nca_diagnostic: NCASubjectDiagnostic | None = None

    def __post_init__(self) -> None:
        # Defense-in-depth against accidental in-place mutation of the
        # simulation arrays. A freshly allocated ndarray is writeable
        # by default; making it read-only after assignment is the
        # numpy idiom for "treat these as immutable from here on."
        for arr in (self.t_observed, self.observed_dv, self.sims_at_observed):
            arr.flags.writeable = False


class PredictiveSummaryBundle(BaseModel):
    """Return type of :func:`build_predictive_diagnostics`.

    Holds all three Gate 3 simulation-based diagnostics derived from one
    simulation matrix. Backends map the fields one-to-one onto
    :class:`~apmode.bundle.models.DiagnosticBundle` in a single
    construction so partial population cannot happen.

    ``n_subjects_total`` / ``n_subjects_nca_eligible`` are surfaced on
    the bundle (not only inside ``auc_cmax_be_score``) so the audit
    trail can distinguish "score dropped because no subjects qualified"
    from "score dropped because backend did not run NCA".
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    vpc: VPCSummary
    # PIT/NPDE-lite predictive calibration (policy 0.4.2 Gate 1 metric).
    # Populated from the same per-subject simulation matrix as ``vpc`` so
    # the atomic-population invariant still holds: a backend that emits
    # ``vpc`` must also emit ``pit_calibration``.
    pit_calibration: PITCalibrationSummary
    npe_score: float = Field(ge=0.0)
    auc_cmax_be_score: float | None = Field(default=None, ge=0.0, le=1.0)
    auc_cmax_source: Literal["observed_trapezoid"] | None = None
    n_subjects_total: int = Field(ge=1)
    n_subjects_nca_eligible: int = Field(ge=0)
    # Per-reason NCA-ineligibility counts. Keys are a fixed taxonomy
    # ``{"absorption", "elimination", "blq", "span", "lambda_z",
    # "auc_extrap", "missing", "other"}``. Reviewers use this to
    # distinguish "8/12 eligible with 4 λz-failures" from "8/12 eligible
    # with 4 extrapolation-failures" — information loss from mask-drop is
    # less concerning when failures are uncorrelated with structural-
    # model discrimination signal.
    mask_drop_reasons: dict[str, int] = Field(default_factory=dict)
    # Provenance of the aggregation choices active for this summary.
    # Lets the bundle audit trail detect drift without re-reading policy.
    npe_aggregation: Literal["flatten", "per_subject_median"] = "flatten"
    auc_cmax_aggregation: Literal["median_trajectory", "median_of_aucs"] = "median_trajectory"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bin_edges_from_pooled_times(all_times: np.ndarray, n_bins: int) -> np.ndarray:
    """Return equal-count quantile bin edges over pooled observation times.

    Collapses to a minimal 2-point span when the pooled time set has
    fewer than two unique values (single-time-point datasets are
    degenerate for VPC but must not crash the ranking). The last edge
    is nudged by a single ULP so the right-most observation falls into
    the final bin under right-exclusive digitization.
    """
    unique_times = np.unique(all_times)
    if unique_times.size < 2:
        lo = float(unique_times[0])
        hi = np.nextafter(lo, np.inf)
        return np.array([lo, hi])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(all_times, quantiles)
    edges = np.unique(raw_edges)
    if edges.size < 2:
        lo = float(edges[0])
        hi = np.nextafter(lo, np.inf)
        return np.array([lo, hi])
    # ``np.unique`` already returns a fresh array, so no extra copy needed
    # before the in-place nudge of the right-most edge.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _compute_vpc_from_sims(
    per_subject_sims: list[SubjectSimulation],
    *,
    percentiles: tuple[float, ...],
    coverage_target: float,
    n_bins: int,
    collapse_warn_ratio: float = 0.5,
) -> VPCSummary:
    """Percentile-based VPC with confidence intervals (xpose4/PsN).

    For each requested percentile ``p`` and each post-hoc time bin:

    1. Compute the empirical ``p``-th percentile of observed values in
       the bin (``obs_p_bin``).
    2. Compute the ``p``-th percentile of simulated values in the bin
       *per sim replicate* → distribution of ``sim_p`` across ``n_sims``.
    3. Derive the central ``coverage_target``·100% CI on ``sim_p``
       (endpoints at ``(1 - coverage_target)/2`` and
       ``1 - (1 - coverage_target)/2`` quantiles).
    4. A bin "hits" if ``obs_p_bin`` lies inside that CI.

    ``coverage[f"p{int(p)}"]`` is the hit fraction across non-empty bins.
    Well-calibrated models target ``coverage_target`` uniformly across
    percentiles, which matches how
    :func:`apmode.governance.ranking.compute_vpc_concordance` consumes
    the dict (single scalar ``target`` compared against every entry).

    Quantile-tied observation times collapse adjacent bin edges via
    ``np.unique`` in :func:`_bin_edges_from_pooled_times`. When the
    effective bin count drops below ``collapse_warn_ratio * n_bins``, a
    :class:`logging.WARNING` surfaces the audit event so reviewers can
    interpret noisy coverage estimates on sparse-sampling designs.
    """
    all_times = np.concatenate([s.t_observed for s in per_subject_sims])
    all_obs = np.concatenate([s.observed_dv for s in per_subject_sims])
    sims_pooled = np.concatenate([s.sims_at_observed for s in per_subject_sims], axis=1)

    n_pooled = all_times.shape[0]
    effective_n_bins = min(n_bins, max(1, n_pooled))
    requested_n_bins = effective_n_bins
    edges = _bin_edges_from_pooled_times(all_times, effective_n_bins)
    effective_n_bins = max(1, edges.size - 1)

    if requested_n_bins >= 2 and effective_n_bins < collapse_warn_ratio * requested_n_bins:
        logger.warning(
            "VPC bin collapse: requested %d bins but only %d effective after "
            "quantile-tie deduplication (pooled_times=%d, unique=%d). "
            "Coverage estimates are noisier than the bin count suggests.",
            requested_n_bins,
            effective_n_bins,
            n_pooled,
            np.unique(all_times).size,
        )

    # Right-exclusive digitization on internal edges; the nextafter
    # adjustment in _bin_edges_from_pooled_times ensures the right-most
    # observation lands in the final bin.
    bin_idx = np.digitize(all_times, edges[1:-1], right=False)

    alpha = (1.0 - coverage_target) / 2.0
    ci_low_q = 100.0 * alpha
    ci_hi_q = 100.0 * (1.0 - alpha)

    coverage_dict: dict[str, float] = {}
    for p in percentiles:
        hits = 0
        total = 0
        for b in range(effective_n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            obs_in_bin = all_obs[mask]
            sims_in_bin = sims_pooled[:, mask]  # (n_sims, n_in_bin)
            if obs_in_bin.size == 0 or sims_in_bin.size == 0:
                continue
            obs_percentile = float(np.percentile(obs_in_bin, p))
            sim_percentiles_across_sims = np.percentile(sims_in_bin, p, axis=1)
            ci_low = float(np.percentile(sim_percentiles_across_sims, ci_low_q))
            ci_hi = float(np.percentile(sim_percentiles_across_sims, ci_hi_q))
            if ci_low <= obs_percentile <= ci_hi:
                hits += 1
            total += 1
        coverage_dict[f"p{int(p)}"] = (hits / total) if total > 0 else 0.0

    return VPCSummary(
        percentiles=list(percentiles),
        coverage=coverage_dict,
        n_bins=effective_n_bins,
        prediction_corrected=False,
    )


def _compute_pit_calibration(
    per_subject_sims: list[SubjectSimulation],
    *,
    probability_levels: tuple[float, ...],
) -> PITCalibrationSummary:
    """Subject-robust PIT / NPDE-lite calibration.

    Replaces the bin-level VPC containment metric with a direct
    predictive-CDF calibration check at each observation. For each
    probability ``p`` in ``probability_levels`` and each observation
    ``j`` on subject ``i``, evaluate the indicator
    ``I_p(i, j) = 1[y_obs[i, j] <= quantile_p({sim[s, j]})]``, average
    within-subject to ``c_p(i)``, then average across subjects:

        c_p = mean_i( mean_j I_p(i, j) )

    Under a well-calibrated predictive distribution, ``c_p`` converges to
    ``p`` — so the Gate 1 check asks ``|c_p - p| <= tol``.

    The **subject-robust** aggregation (inner per-subject mean before the
    outer cross-subject mean) is the recommended form for governance: it
    downweights subjects with many observations so heavily-sampled
    individuals don't dominate the calibration signal — relevant for
    mixed PK designs where dose/richness varies.

    No bin grid, no 1/n_bins quantization: the denominator is effectively
    the subject count (with observations weighted by inverse per-subject
    count), making the metric invariant to time-binning choice and robust
    on sparse real-data designs where the prior VPC metric false-rejected
    textbook-correct models (see CHANGELOG rc9 follow-up).
    """
    if not per_subject_sims:
        msg = "per_subject_sims must be non-empty to compute PIT calibration"
        raise ValueError(msg)
    for p in probability_levels:
        if not 0.0 < p < 1.0:
            msg = f"probability_levels must be in (0, 1), got {p}"
            raise ValueError(msg)

    # Per-subject hit-rate at each probability level. NaN-safe: a failed
    # ODE solve on one sim draw can leave NaNs in ``sims``; mask per-
    # observation so a single NaN sim doesn't poison the whole hit-rate
    # for that subject. ``np.nanpercentile`` returns NaN when *all* sim
    # draws at a given observation index are NaN — those observations
    # are then dropped from the indicator mean (not counted as hits OR
    # misses), not silently treated as a fail.
    percentile_pcts = tuple(100.0 * p for p in probability_levels)
    per_subject_rates: dict[float, list[float]] = {p: [] for p in probability_levels}
    n_observations_used = 0
    n_subjects_used = 0
    for s in per_subject_sims:
        obs = s.observed_dv
        sims = s.sims_at_observed  # (n_sims, n_obs_i)
        n_obs_i = obs.shape[0]
        if n_obs_i == 0:
            continue
        # Drop observations that are non-finite — backends occasionally
        # emit NaN DV for BLQ records or missing timepoints; the PIT
        # indicator is undefined for a non-finite obs.
        obs_finite_mask = np.isfinite(obs)
        if not obs_finite_mask.any():
            continue
        # Predictive p-quantile at each observation, across sim replicates.
        # axis=0 collapses the sim dimension → shape (n_obs_i,).
        # ``nanpercentile`` ignores NaN sim draws per observation column;
        # when every draw is NaN for one column the result is NaN there.
        q_levels = np.nanpercentile(sims, percentile_pcts, axis=0)
        # Keep only observations where BOTH obs is finite AND the
        # simulated quantile is finite (latter guards all-NaN sim rows).
        for idx, p in enumerate(probability_levels):
            q_row = q_levels[idx]
            usable = obs_finite_mask & np.isfinite(q_row)
            n_usable = int(usable.sum())
            if n_usable == 0:
                continue
            hits = (obs[usable] <= q_row[usable]).astype(float)
            per_subject_rates[p].append(float(hits.mean()))
        n_observations_used += int(obs_finite_mask.sum())
        n_subjects_used += 1

    if n_subjects_used == 0 or all(not per_subject_rates[p] for p in probability_levels):
        msg = (
            "No subjects with usable finite observations/simulations for "
            "PIT calibration — cannot produce a calibration summary"
        )
        raise ValueError(msg)

    calibration = {
        _pit_key(p): (float(np.mean(per_subject_rates[p])) if per_subject_rates[p] else 0.0)
        for p in probability_levels
    }

    return PITCalibrationSummary(
        probability_levels=list(probability_levels),
        calibration=calibration,
        # Denominator of the outer mean is ``n_subjects_used`` (subjects
        # that contributed at least one finite obs/sim pair), not
        # ``len(per_subject_sims)`` — the audit field mirrors the
        # actual averaging denominator.
        n_observations=max(n_observations_used, 1),
        n_subjects=n_subjects_used,
        aggregation="subject_robust",
    )


def _per_subject_auc_cmax(t_observed: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Trapezoidal AUC and Cmax of a 1D concentration vector.

    NaN-safe: non-finite entries in either argument are masked before
    integration. Returns ``(NaN, NaN)`` when fewer than 2 finite points
    remain — AUC is undefined on a single point and the BE check
    treats NaN as non-finite (BE-fail).
    """
    mask = np.isfinite(values) & np.isfinite(t_observed)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    t = t_observed[mask]
    v = values[mask]
    order = np.argsort(t)
    t_sorted = t[order]
    v_sorted = v[order]
    auc = float(np.trapezoid(v_sorted, t_sorted))
    cmax = float(np.max(v_sorted))
    return auc, cmax


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_predictive_diagnostics(
    per_subject_sims: list[SubjectSimulation],
    *,
    policy: Gate3Config,
    vpc_percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    vpc_coverage_target: float = 0.90,
) -> PredictiveSummaryBundle:
    """Assemble VPC + NPE + AUC/Cmax BE from per-subject simulations.

    Single canonical path from posterior-predictive simulations to Gate
    3 diagnostics. Backends call this once at ``BackendResult``
    construction; the returned bundle populates
    :class:`~apmode.bundle.models.DiagnosticBundle` atomically so no
    backend can emit partial diagnostics.

    **Inputs**

    * ``per_subject_sims`` — one :class:`SubjectSimulation` per subject,
      each carrying an ``(n_sims, n_obs_i)`` simulation matrix at that
      subject's observed time vector. ``n_sims`` must be uniform across
      subjects (downstream NPE flattens along the observation axis).
    * ``policy`` — the lane's :class:`~apmode.governance.policy.Gate3Config`.
      Controls ``vpc_n_bins`` and the NCA eligibility floors
      (``auc_cmax_nca_min_eligible`` and
      ``auc_cmax_nca_min_eligible_fraction``).
    * ``vpc_percentiles`` / ``vpc_coverage_target`` — VPC percentile
      points and the nominal CI width (default 90% — matches the
      ranker's default ``vpc_concordance_target=0.90``).

    **Raises**

    ``ValueError`` when ``per_subject_sims`` is empty, when simulation
    matrices are malformed (wrong ndim, mis-aligned with the observed
    time/DV vectors), or when ``n_sims`` differs across subjects.
    """
    if not per_subject_sims:
        msg = "per_subject_sims must be non-empty to compute predictive diagnostics"
        raise ValueError(msg)

    n_sims_shared: int | None = None
    for s in per_subject_sims:
        if s.sims_at_observed.ndim != 2:
            msg = (
                f"{s.subject_id}: sims_at_observed must be 2D "
                f"(n_sims, n_obs); got shape {s.sims_at_observed.shape}"
            )
            raise ValueError(msg)
        n_obs_i = s.t_observed.shape[0]
        if s.observed_dv.shape[0] != n_obs_i:
            msg = (
                f"{s.subject_id}: observed_dv shape {s.observed_dv.shape} "
                f"inconsistent with t_observed shape {s.t_observed.shape}"
            )
            raise ValueError(msg)
        if s.sims_at_observed.shape[1] != n_obs_i:
            msg = (
                f"{s.subject_id}: sims_at_observed shape "
                f"{s.sims_at_observed.shape} inconsistent with n_obs={n_obs_i}"
            )
            raise ValueError(msg)
        if n_sims_shared is None:
            n_sims_shared = s.sims_at_observed.shape[0]
        elif s.sims_at_observed.shape[0] != n_sims_shared:
            msg = (
                f"{s.subject_id}: n_sims={s.sims_at_observed.shape[0]} "
                f"differs from earlier subjects (n_sims={n_sims_shared})"
            )
            raise ValueError(msg)

    # NPE — aggregation controlled by policy.
    if policy.npe_aggregation == "flatten":
        # Legacy rc8 path: pool all (obs, sim-median) pairs, then median
        # absolute error. Well-sampled subjects weight more heavily.
        obs_flat = np.concatenate([s.observed_dv for s in per_subject_sims])
        sims_flat = np.concatenate([s.sims_at_observed for s in per_subject_sims], axis=1)
        npe_score = compute_npe(obs_flat, sims_flat)
    else:
        # "per_subject_median": compute NPE per subject then median across
        # subjects. Equal weight regardless of observation count.
        per_subject_npes = [
            compute_npe(s.observed_dv, s.sims_at_observed) for s in per_subject_sims
        ]
        npe_score = float(np.median(np.asarray(per_subject_npes, dtype=float)))

    vpc = _compute_vpc_from_sims(
        per_subject_sims,
        percentiles=vpc_percentiles,
        coverage_target=vpc_coverage_target,
        n_bins=policy.vpc_n_bins,
        collapse_warn_ratio=policy.vpc_n_bin_collapse_warn_ratio,
    )

    # PIT/NPDE-lite calibration — the 0.4.2 Gate 1 gated metric.
    # Probability levels mirror the VPC percentile points (5 / 50 / 95)
    # but expressed as (0, 1) CDF levels instead of 0-100 percentiles.
    # Fixed vocabulary for now — extending to custom levels would need a
    # matching ``Gate1Config.pit_probability_levels`` knob.
    pit_calibration = _compute_pit_calibration(
        per_subject_sims,
        probability_levels=tuple(p / 100.0 for p in vpc_percentiles),
    )

    # AUC/Cmax BE — per-subject, aggregation controlled by policy.
    n_total = len(per_subject_sims)
    cand_auc = np.zeros(n_total, dtype=float)
    cand_cmax = np.zeros(n_total, dtype=float)
    nca_auc = np.zeros(n_total, dtype=float)
    nca_cmax = np.zeros(n_total, dtype=float)
    eligibility = np.zeros(n_total, dtype=bool)
    mask_drop_counts: dict[str, int] = {k: 0 for k in _MASK_DROP_KEYS}

    for i, s in enumerate(per_subject_sims):
        if policy.auc_cmax_aggregation == "median_trajectory":
            # rc8 default: collapse to per-sim median, then trapezoid/max
            # once. Point-estimate summary; ignores distributional uncertainty.
            median_traj = np.median(s.sims_at_observed, axis=0)
            cand_auc[i], cand_cmax[i] = _per_subject_auc_cmax(s.t_observed, median_traj)
        else:
            # "median_of_aucs": per-sim trapezoid → distribution of (AUC, Cmax)
            # across sims → take the median of each. Preserves distributional
            # uncertainty for nonlinear profiles.
            per_sim_auc = np.full(s.sims_at_observed.shape[0], np.nan, dtype=float)
            per_sim_cmax = np.full(s.sims_at_observed.shape[0], np.nan, dtype=float)
            for k in range(s.sims_at_observed.shape[0]):
                per_sim_auc[k], per_sim_cmax[k] = _per_subject_auc_cmax(
                    s.t_observed, s.sims_at_observed[k]
                )
            cand_auc[i] = (
                float(np.nanmedian(per_sim_auc))
                if np.isfinite(per_sim_auc).any()
                else float("nan")
            )
            cand_cmax[i] = (
                float(np.nanmedian(per_sim_cmax))
                if np.isfinite(per_sim_cmax).any()
                else float("nan")
            )
        nca_auc[i], nca_cmax[i] = _per_subject_auc_cmax(s.t_observed, s.observed_dv)
        if s.nca_diagnostic is not None:
            eligible, reason = is_nca_eligible_per_subject(s.nca_diagnostic)
            eligibility[i] = eligible
            if not eligible:
                mask_drop_counts[_classify_exclusion_reason(reason)] += 1
        else:
            # No diagnostic at all = "other" bucket for audit accounting.
            mask_drop_counts["other"] += 1

    n_eligible = int(eligibility.sum())
    auc_cmax_score = compute_auc_cmax_be_score(
        cand_auc,
        cand_cmax,
        nca_auc,
        nca_cmax,
        eligible_mask=eligibility,
        min_eligible=policy.auc_cmax_nca_min_eligible,
        min_eligible_fraction=policy.auc_cmax_nca_min_eligible_fraction,
    )
    auc_cmax_source: Literal["observed_trapezoid"] | None = (
        "observed_trapezoid" if auc_cmax_score is not None else None
    )

    # Strip zero-count entries so the audit field stays readable.
    mask_drop_nonzero = {k: v for k, v in mask_drop_counts.items() if v > 0}

    return PredictiveSummaryBundle(
        vpc=vpc,
        pit_calibration=pit_calibration,
        npe_score=npe_score,
        auc_cmax_be_score=auc_cmax_score,
        auc_cmax_source=auc_cmax_source,
        n_subjects_total=n_total,
        n_subjects_nca_eligible=n_eligible,
        mask_drop_reasons=mask_drop_nonzero,
        npe_aggregation=policy.npe_aggregation,
        auc_cmax_aggregation=policy.auc_cmax_aggregation,
    )
