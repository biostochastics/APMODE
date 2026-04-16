# SPDX-License-Identifier: GPL-2.0-or-later
"""Data Profiler: analyzes ingested PK data and emits an Evidence Manifest (PRD §4.2.1).

The profiler is not merely descriptive — its output constrains downstream dispatch.
Rule-based analysis produces typed manifest fields that the Lane Router must consume.

Dispatch constraints derived from manifest (PRD §4.2.1):
  - richness=sparse + absorption_coverage=inadequate → NODE not dispatched
  - nonlinear_clearance_evidence_strength="strong" → automated search includes MM candidates
  - blq_burden > 0.20 → all backends must use BLQ-aware likelihood (M3/M4)
  - protocol_heterogeneity=pooled-heterogeneous → IOV must be tested
  - covariate_missingness.fraction_incomplete > 0.15 → full-information likelihood
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd  # noqa: TC002
from scipy.signal import find_peaks

from apmode.bundle.models import (
    CovariateSpec,
    ErrorModelPreference,
    EvidenceManifest,
    NonlinearClearanceSignal,
    SignalId,
)
from apmode.data.policy import get_policy
from apmode.data.types import (
    TerminalPhase,
    auc_linup_logdown,
    bootstrap_median_ci,
    fit_best_lambdaz,
    fit_dose_proportionality,
    is_steady_state,
    positive_unblqd_mask,
    safe_log,
    wagner_nelson_ka,
)

_log = logging.getLogger(__name__)
_POLICY = get_policy()

# Thresholds below are sourced from `policies/profiler.json` (v2.1.0+) via
# the loader in `apmode/data/policy.py`. The JSON is the source of truth;
# these module-level constants are a derived view for readable call sites.
# Drift between constants and JSON is guarded by
# `tests/unit/test_profiler_policy_consistency.py` (added alongside this
# refactor). Do NOT add bare numeric literals inside heuristic functions —
# either add a new policy field or import an existing constant here.
_PK_DVIDS: frozenset[str] = _POLICY.pk_dvid_allowlist
_COVARIATE_CORRELATION_THRESHOLD: float = _POLICY.covariate_correlation_threshold_abs_r
_NONLINEAR_MM_CURVATURE_RATIO: float = _POLICY.mm_curvature_ratio
_COMPARTMENTALITY_CURVATURE_RATIO: float = _POLICY.compartmentality_curvature_ratio
_NONLINEAR_TMDD_CURVATURE_RATIO: float = _POLICY.tmdd_curvature_ratio
_MULTI_PEAK_FRACTION_THRESHOLD: float = _POLICY.multi_peak_fraction_threshold
_LAG_SIGNATURE_FRACTION_THRESHOLD: float = _POLICY.lag_signature_fraction_threshold
_BLQ_COVARIATE_MISSINGNESS_CUTOFF: float = _POLICY.covariate_missingness_full_information_cutoff

# Multi-signal voting thresholds for nonlinear-clearance evidence strength.
# A monoexponential terminal fit with adj R² >= this value is treated as
# evidence FOR linear (multi-compartment) kinetics and AGAINST MM saturation.
# Justification: per-subject log-linear regression on the final ~30% of post-
# Cmax samples; warfarin (2-cmt linear) reaches ~0.95 here while a true MM
# late phase below Km also reaches ~0.95 — so this gates on the *combination*
# of curvature ratio AND failure of monoexp explanation, not on R² alone
# (MM kinetics appear linear when C << Km; do not veto on R² alone).
_TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD: float = _POLICY.terminal_monoexp_r2_linear_threshold
_TERMINAL_FIT_MIN_POINTS: int = _POLICY.lambdaz_min_points
_PEAK_PROMINENCE_RANGE_FRACTION: float = _POLICY.peak_prominence_range_fraction
_PEAK_PROMINENCE_CMAX_FLOOR: float = _POLICY.peak_prominence_cmax_floor
_PEAK_MIN_DISTANCE_INTERVALS: float = _POLICY.peak_min_distance_intervals

# ---------------------------------------------------------------------------
# Advisory vs disqualifying thresholds.
#
# The profiler emits EvidenceManifest fields. Downstream consumers (Lane
# Router, Gate 1/2 evaluators) decide which fields are disqualifying and
# which are advisory. The constants below map policy fields to code paths;
# none of them are themselves disqualifying — a manifest field is never a
# hard fail. Disqualification lives in governance/gates.py against the
# GatePolicy (Gate 1 technical validity, Gate 2 lane admissibility, Gate 2.5
# credibility).
# ---------------------------------------------------------------------------

# Routine λz quality threshold (Huang 2025 autoinit default). Advisory:
# used to drop unreliable terminal fits from pool-level statistics.
_LAMBDAZ_ADJ_R2_THRESHOLD: float = _POLICY.lambdaz_adj_r2_threshold

# Subject-quality thresholds for richness / identifiability classification.
_MIN_OBS_PER_SUBJECT_RICH: int = _POLICY.min_obs_per_subject_rich
_MIN_OBS_PER_SUBJECT_MODERATE: int = _POLICY.min_obs_per_subject_moderate
_ABSORPTION_COVERAGE_MIN_PRE_TMAX: float = _POLICY.absorption_coverage_min_pre_tmax
_ELIMINATION_COVERAGE_MIN_POST_TMAX: float = _POLICY.elimination_coverage_min_post_tmax

# NODE dim-budget thresholds (PRD §4.2.4 R6). Drives the `node_dim_budget`
# field on EvidenceManifest which the Lane Router consumes to gate NODE
# dispatch. 0 = NODE excluded; non-zero = eligible in matching lane only.
_NODE_DISCOVERY_MIN_SUBJECTS: int = _POLICY.node_discovery_min_subjects
_NODE_DISCOVERY_MIN_MEDIAN_SAMPLES: int = _POLICY.node_discovery_min_median_samples
_NODE_DISCOVERY_BUDGET: int = _POLICY.node_discovery_budget
_NODE_OPTIMIZATION_MIN_SUBJECTS: int = _POLICY.node_optimization_min_subjects
_NODE_OPTIMIZATION_MIN_MEDIAN_SAMPLES: int = _POLICY.node_optimization_min_median_samples
_NODE_OPTIMIZATION_BUDGET: int = _POLICY.node_optimization_budget

# Flip-flop detection (Richardson 2025). Quality guard is stricter than
# routine λz (0.85 vs 0.70) because flip-flop classification is structurally
# sensitive to terminal-slope quality — kept as a separate policy key.
_FLIP_FLOP_KA_LAMBDAZ_RATIO_POSSIBLE: float = _POLICY.flip_flop_ka_lambdaz_ratio_possible
_FLIP_FLOP_QUALITY_ADJ_R2_MIN: float = _POLICY.flip_flop_quality_adj_r2_min
_FLIP_FLOP_QUALITY_MIN_NPTS: int = _POLICY.flip_flop_quality_min_npts

# Protocol heterogeneity (pooled-study CV gate).
_PROTOCOL_HETEROGENEITY_CV_THRESHOLD: float = (
    _POLICY.protocol_heterogeneity_obs_per_subject_cv_threshold
)

# Error model: narrow-range + low-CV additive detection.
_LOW_CV_ADDITIVE_CEILING: float = _POLICY.low_cv_additive_ceiling


def _fit_terminal(times: np.ndarray, concs: np.ndarray) -> TerminalPhase | None:
    """Policy-driven wrapper around ``fit_best_lambdaz``.

    Threads the policy JSON's Huang-2025 parameters into the algorithm so
    deployments that tune ``policies/profiler.json`` actually see their
    thresholds honored.
    """
    return fit_best_lambdaz(
        times,
        concs,
        min_points=_POLICY.lambdaz_min_points,
        tolerance=_POLICY.lambdaz_tolerance,
        phoenix_constraint=_POLICY.lambdaz_phoenix_constraint,
    )


def _policy_steady_state(
    dose_times: np.ndarray, doses: np.ndarray, *, half_life: float | None
) -> tuple[bool, str]:
    """Policy-driven wrapper around ``is_steady_state``."""
    return is_steady_state(
        dose_times=dose_times,
        doses=doses,
        half_life=half_life,
        n_half_lives_required=_POLICY.ss_n_half_lives_required,
        n_doses_alt=_POLICY.ss_n_doses_alt,
        interval_tolerance=_POLICY.ss_interval_tolerance,
        dose_tolerance=_POLICY.ss_dose_tolerance,
        min_doses=_POLICY.ss_min_doses,
    )


if TYPE_CHECKING:
    from apmode.bundle.models import DataManifest


def profile_data(
    df: pd.DataFrame,
    manifest: DataManifest,
) -> EvidenceManifest:
    """Analyze ingested PK data and produce an Evidence Manifest.

    Args:
        df: Validated DataFrame (post-CanonicalPKSchema validation).
        manifest: DataManifest from ingest step (provides SHA-256, covariates).

    Returns:
        EvidenceManifest with typed fields constraining downstream dispatch.
    """
    df, n_non_pk_dropped = _filter_pk_observations_with_count(df)
    obs = cast("pd.DataFrame", df[df["EVID"] == 0].copy())
    doses = cast("pd.DataFrame", df[df["EVID"] == 1].copy())
    n_subjects = int(cast("int", df["NMID"].nunique()))

    blq_burden = _compute_blq_burden(df)
    lloq_value = _extract_lloq_value(df)
    cmax_p95_p05_ratio = _compute_cmax_dynamic_range(obs)
    dv_cv_percent = _compute_dv_cv_percent(obs)

    # Detect multi-dose BEFORE any terminal / shape metric so all per-subject
    # helpers can window to the last dose interval. The previous order
    # silently contaminated terminal_log_mad on multi-dose subjects
    # (terminal fit spanned subsequent absorption rises).
    multi_dose = _detect_multi_dose(doses)

    terminal_log_mad = _compute_terminal_log_residual_mad(obs, doses, multi_dose=multi_dose)
    error_pref = recommend_error_model(
        obs,
        blq_burden=blq_burden,
        lloq=lloq_value,
        cmax_p95_p05_ratio=cmax_p95_p05_ratio,
        dv_cv_percent=dv_cv_percent,
        terminal_log_mad=terminal_log_mad,
    )

    # Multi-signal classification of nonlinear-clearance evidence (PRD §10
    # Q2 follow-up). Routing: ``strong`` = full MM cross-product;
    # ``moderate`` = single sentinel MM candidate; ``weak``/``none`` = MM
    # not auto-included.
    nlc_strength, nlc_signals = _classify_nonlinear_clearance_evidence_strength(
        obs, doses, multi_dose=multi_dose
    )

    # PR-F new diagnostic signals.
    lambda_z_meta = _lambda_z_window_meta(obs, doses, multi_dose=multi_dose)
    flip_flop = _assess_flip_flop_risk(obs, doses, multi_dose=multi_dose)
    wn_ka = _wagner_nelson_ka_median(obs, doses, multi_dose=multi_dose)
    node_budget = _compute_node_dim_budget(obs, n_subjects, multi_dose=multi_dose)
    tad_flag = _assess_tad_consistency(obs, doses, multi_dose=multi_dose)
    richness = _classify_richness(obs, n_subjects)
    abs_cov = _assess_absorption_coverage(obs)

    return EvidenceManifest(
        data_sha256=manifest.data_sha256,
        route_certainty=_assess_route_certainty(doses),
        absorption_complexity=_assess_absorption_complexity(obs, doses, multi_dose=multi_dose),
        nonlinear_clearance_confidence=_nonlinear_clearance_confidence(
            obs, doses, multi_dose=multi_dose
        ),
        nonlinear_clearance_evidence_strength=nlc_strength,
        multi_dose_detected=multi_dose,
        compartmentality=_assess_compartmentality(obs, doses, multi_dose=multi_dose),
        auc_extrap_fraction_median=_auc_extrap_fraction_median(obs, doses, multi_dose=multi_dose),
        lambda_z_analyzable_fraction=_lambda_z_analyzable_fraction(
            obs, doses, multi_dose=multi_dose
        ),
        peak_prominence_fraction=_peak_prominence_fraction(obs, doses, multi_dose=multi_dose),
        richness_category=richness,
        identifiability_ceiling=_assess_identifiability(obs, n_subjects),
        covariate_burden=len(manifest.covariates),
        covariate_correlated=_check_covariate_correlation(df, manifest),
        covariate_missingness=_assess_covariate_missingness(df, manifest),
        time_varying_covariates=_detect_time_varying_covariates(df, manifest),
        blq_burden=blq_burden,
        lloq_value=lloq_value,
        protocol_heterogeneity=_assess_protocol_heterogeneity(df),
        absorption_phase_coverage=abs_cov,
        elimination_phase_coverage=_assess_elimination_coverage(obs),
        error_model_preference=error_pref,
        cmax_p95_p05_ratio=cmax_p95_p05_ratio,
        dv_cv_percent=dv_cv_percent,
        terminal_log_residual_mad=terminal_log_mad,
        # ---- v3 structured signal provenance ----
        nonlinear_clearance_signals=nlc_signals,
        lambda_z_used_points_median=lambda_z_meta.get("used_points_median"),
        lambda_z_adj_r2_median=lambda_z_meta.get("adj_r2_median"),
        flip_flop_risk=flip_flop,
        wagner_nelson_ka_median=wn_ka,
        node_dim_budget=node_budget,
        tad_consistency_flag=tad_flag,
        n_non_pk_rows_dropped=n_non_pk_dropped,
    )


def _lambda_z_window_meta(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> dict[str, float | None]:
    """Per-subject median of (n_used, adj_r2) from fit_best_lambdaz.

    R6 audit fields — exposes the Phoenix-constrained Huang 2025
    selector's chosen window size and goodness-of-fit so reviewers can
    confirm tail-selection quality without re-running the algorithm.
    """
    if obs.empty:
        return {"used_points_median": None, "adj_r2_median": None}
    n_used: list[int] = []
    r2: list[float] = []
    for subj in obs["NMID"].unique():
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        terminal = _fit_terminal(times, concs)
        if terminal is None:
            continue
        n_used.append(terminal.n_used)
        r2.append(terminal.adj_r2)
    return {
        "used_points_median": float(np.median(n_used)) if n_used else None,
        "adj_r2_median": float(np.median(r2)) if r2 else None,
    }


def _wagner_nelson_ka_median(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> float | None:
    """Population-median Wagner-Nelson ka across single-dose oral subjects.

    R15 (Huang Z et al. 2025; Wagner & Nelson 1963). Multi-dose subjects
    are skipped because Wagner-Nelson assumes a single dose at t=0.
    Returns None when fewer than 4 subjects yield a valid ka.
    """
    if obs.empty or doses.empty:
        return None
    dose_counts = doses.groupby("NMID").size()
    kas: list[float] = []
    for subj in obs["NMID"].unique():
        n_d = int(dose_counts.get(subj, 0))
        if n_d != 1:
            # Skip multi-dose AND zero-dose subjects.
            continue
        subj_obs_filt = obs[obs["NMID"] == subj]
        subj_obs_filt = subj_obs_filt[positive_unblqd_mask(subj_obs_filt)]
        subj_obs_filt = subj_obs_filt.sort_values("TIME")
        times = subj_obs_filt["TIME"].to_numpy(dtype=float)
        concs = subj_obs_filt["DV"].to_numpy(dtype=float)
        if len(concs) < 4:
            continue
        terminal = _fit_terminal(times, concs)
        if terminal is None or terminal.half_life is None:
            continue
        ka = wagner_nelson_ka(times, concs, half_life=terminal.half_life)
        if ka is not None and ka > 0:
            kas.append(ka)
    if len(kas) < 4:
        return None
    return float(np.median(kas))


def _assess_flip_flop_risk(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> Literal["none", "possible", "likely", "unknown"]:
    """Flip-flop kinetics risk for oral profiles (Richardson 2025).

    Combines Wagner-Nelson ka (Wagner 1963) vs population-median λz with a
    tiered terminal-fit quality guard. Decision is **advisory** (surfaced
    on the EvidenceManifest); disqualification happens downstream in the
    Lane Router or ranking layer.

    Tiered quality model:

      * **Floor** (``lambdaz_adj_r2_threshold``, Huang 2025 default 0.70):
        median terminal adj-R² below this → return ``"unknown"``. The
        terminal fit is too poor to support *any* flip-flop call.
      * **Strict** (``flip_flop_quality_adj_r2_min``, Richardson 2025
        default 0.85): median adj-R² between floor and strict → a
        ``"likely"`` call is downgraded to ``"possible"``. The fit is
        usable for routine λz but not strong enough for the structurally
        sensitive ka/ke-ordering decision.
      * **Strong**: median adj-R² ≥ strict AND median terminal points ≥
        ``flip_flop_quality_min_npts`` (default 4) → a ``"likely"`` call
        is retained.

    The ``1.5x`` population-median-λz ratio used for the ``"possible"``
    boundary comes from ``flip_flop.ka_lambdaz_ratio_possible``.
    """
    if doses.empty:
        return "unknown"
    has_rate = "RATE" in doses.columns and (doses["RATE"] > 0).any()
    has_dur = "DUR" in doses.columns and (doses["DUR"] > 0).any()
    if has_rate or has_dur:
        return "none"  # IV infusion — no absorption phase.
    wn_ka = _wagner_nelson_ka_median(obs, doses, multi_dose=multi_dose)
    if wn_ka is None:
        return "unknown"
    # Population-median λz + terminal-fit-quality guard.
    lambdas: list[float] = []
    n_used_pts: list[int] = []
    adj_r2s: list[float] = []
    for subj in obs["NMID"].unique():
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        terminal = _fit_terminal(times, concs)
        if terminal is None or terminal.lambda_z is None:
            continue
        lambdas.append(terminal.lambda_z)
        n_used_pts.append(terminal.n_used)
        adj_r2s.append(terminal.adj_r2)
    if len(lambdas) < 4:
        return "unknown"
    median_lambdaz = float(np.median(lambdas))
    median_npts = float(np.median(n_used_pts))
    median_r2 = float(np.median(adj_r2s))

    # Quality floor (routine λz threshold): below this we cannot speak
    # meaningfully about ka vs ke ordering at all.
    if median_r2 < _LAMBDAZ_ADJ_R2_THRESHOLD:
        return "unknown"

    strict_quality_ok = (
        median_npts >= _FLIP_FLOP_QUALITY_MIN_NPTS and median_r2 >= _FLIP_FLOP_QUALITY_ADJ_R2_MIN
    )
    if wn_ka < median_lambdaz and strict_quality_ok:
        return "likely"
    if wn_ka < _FLIP_FLOP_KA_LAMBDAZ_RATIO_POSSIBLE * median_lambdaz:
        return "possible"
    return "none"


def _compute_node_dim_budget(obs: pd.DataFrame, n_subjects: int, *, multi_dose: bool) -> int:
    """NODE input-dim budget per PRD §4.2.4 R6 (Pharmpy AMD design feasibility).

    Returns one of {0, ``_NODE_OPTIMIZATION_BUDGET``, ``_NODE_DISCOVERY_BUDGET``}.
      * 0 → NODE not eligible (sparse data or inadequate absorption).
      * Optimization budget → moderate density (policy-driven).
      * Discovery budget → rich, well-powered (policy-driven).

    All thresholds come from ``policies/profiler.json#/node_readiness``.
    ``multi_dose`` is reserved for future multi-dose-sensitive adjustments
    (the current v2.1 budget does not depend on it).
    """
    del multi_dose  # intentionally unused at v2.1; kept for signature stability
    if obs.empty or n_subjects < _NODE_OPTIMIZATION_MIN_SUBJECTS // 2:
        # Fewer subjects than half the optimization-lane minimum cannot
        # power NODE at any policy setting; short-circuit to 0 without
        # invoking the richer per-policy comparisons below.
        return 0
    per_subj_counts = obs.groupby("NMID").size()
    median_samples = float(per_subj_counts.median()) if len(per_subj_counts) > 0 else 0.0
    abs_cov = _assess_absorption_coverage(obs)
    if median_samples < _NODE_OPTIMIZATION_MIN_MEDIAN_SAMPLES or abs_cov == "inadequate":
        return 0
    if (
        median_samples >= _NODE_DISCOVERY_MIN_MEDIAN_SAMPLES
        and n_subjects >= _NODE_DISCOVERY_MIN_SUBJECTS
    ):
        return _NODE_DISCOVERY_BUDGET
    if (
        median_samples >= _NODE_OPTIMIZATION_MIN_MEDIAN_SAMPLES
        and n_subjects >= _NODE_OPTIMIZATION_MIN_SUBJECTS
    ):
        return _NODE_OPTIMIZATION_BUDGET
    return 0


def _assess_tad_consistency(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> Literal["clean", "contaminated", "unknown"]:
    """R19: detect TIME/TAD contamination in multi-dose data.

    Follow-up: ( + ):
    checks per-subject membership in the UNION of per-dose intervals
    ``[dose_i, dose_i + median_tau]`` — stricter than the previous global
    ``[first_dose, last_dose+tau]`` window, which always passed
    regardless of whether observations were actually dose-aligned.

    Threshold: fraction_in < tad_in_window_fraction_clean (policy default
    0.80). Returns ``contaminated`` when TIME is clearly not
    TAD-equivalent (shape heuristics should be down-weighted in
    dispatch); ``clean`` otherwise.
    """
    if not multi_dose:
        return "clean"
    if obs.empty or doses.empty:
        return "unknown"
    n_total = 0
    n_in_window = 0
    threshold = _POLICY.tad_in_window_fraction_clean
    for subj in obs["NMID"].unique():
        subj_doses = doses[doses["NMID"] == subj]
        subj_obs = obs[obs["NMID"] == subj]
        if subj_doses.empty or subj_obs.empty:
            continue
        dt = np.sort(subj_doses["TIME"].to_numpy(dtype=float))
        intervals = np.diff(dt)
        if len(intervals) == 0 or not (intervals > 0).any():
            continue
        tau = float(np.median(intervals[intervals > 0]))
        if tau <= 0:
            continue
        times = subj_obs["TIME"].to_numpy(dtype=float)
        n_total += len(times)
        # Union check: observation is in-window iff ANY dose interval
        # [d, d+tau] contains it.
        in_any = np.zeros(len(times), dtype=bool)
        for d in dt:
            in_any |= (times >= d) & (times <= d + tau)
        n_in_window += int(in_any.sum())
    if n_total == 0:
        return "unknown"
    fraction_in = n_in_window / n_total
    if fraction_in < threshold:
        return "contaminated"
    return "clean"


def _filter_pk_observations_with_count(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Same as ``_filter_pk_observations`` but also returns the count of
    rows removed (for manifest auditability)."""
    n_before = len(df)
    out = _filter_pk_observations(df)
    return out, n_before - len(out)


def _filter_pk_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-PK observation rows when DVID column is present.

    Per  +  review:
    datasets that pool PK and PD observations (warfarin's `cp` + `pca`,
    biomarker studies with multiple analytes) must have non-PK rows
    removed before shape geometry runs. Otherwise PCA values rising in
    the late phase cause Tmax-anchored heuristics to silently flip to
    negative-slope / multi-compartment-likely classifications.

    Fail-open semantics: if no observation row matches the PK allowlist
    (e.g. dataset uses DVID=2 for plasma), the filter ABSTAINS and
    returns df unchanged with a WARNING — better than silently nuking
    every observation row. Same for missing/NaN DVID values, which
    stringify to ``"nan"`` and would otherwise be dropped wholesale.

    Dose rows (EVID==1) are always preserved regardless of DVID. The
    decision is logged at INFO with the count and unique values of the
    dropped rows for auditability (analogous to other recoding
    decisions in the ingest layer).
    """
    if "DVID" not in df.columns:
        return df
    dvid_str = df["DVID"].astype(str).str.lower()
    obs_mask = df["EVID"] == 0
    obs_dvids = dvid_str[obs_mask]
    n_pk_matches = int(obs_dvids.isin(_PK_DVIDS).sum())
    if n_pk_matches == 0:
        unique_obs_dvids = sorted(set(obs_dvids.astype(str)))
        # Policy-controlled fail-open. Deployments that
        # want strict DVID enforcement set dvid_fail_open_when_no_match=false
        # in policies/profiler.json; the default stays permissive to keep
        # unknown-DVID datasets profilable.
        if _POLICY.dvid_fail_open_when_no_match:
            _log.warning(
                "Profiler DVID filter: no observation rows match PK allowlist %s; "
                "observed DVID values=%s. Filter abstains (fail-open per policy) "
                "and keeps all rows.",
                sorted(_PK_DVIDS),
                unique_obs_dvids,
            )
            return df
        msg = (
            f"DVID filter failed closed: no observation rows match PK "
            f"allowlist {sorted(_PK_DVIDS)}; observed DVIDs={unique_obs_dvids}. "
            "Extend policies/profiler.json pk_dvid_allowlist or canonicalise DVID at ingest."
        )
        raise ValueError(msg)
    # Preserve ALL non-observation rows (dose events EVID!=0 including
    # NONMEM reset-dose codes EVID=3/4); filter only observation rows by
    # DVID allowlist.
    obs_mask = df["EVID"] == 0
    keep_mask = (~obs_mask) | dvid_str.isin(_PK_DVIDS)
    dropped = int((~keep_mask).sum())
    if dropped == 0:
        return df
    unique_dropped = sorted(set(df.loc[~keep_mask, "DVID"].astype(str).str.lower()))
    _log.info(
        "Profiler: dropping %d non-PK observation rows (DVID values=%s); "
        "kept DVID values matching %s",
        dropped,
        unique_dropped,
        sorted(_PK_DVIDS),
    )
    return df[keep_mask].copy()


# ---------------------------------------------------------------------------
# Error-model selection heuristic (Beal 2001, Ahn 2008)
# ---------------------------------------------------------------------------

_BLQ_M3_TRIGGER: float = _POLICY.blq_m3_trigger
_DYNAMIC_RANGE_PROPORTIONAL: float = _POLICY.dynamic_range_proportional
_HIGH_CV_CEILING: float = _POLICY.high_cv_ceiling
_LLOQ_CMAX_COMBINED: float = _POLICY.lloq_cmax_combined
_TERMINAL_LOG_MAD_COMBINED: float = _POLICY.terminal_log_mad_combined
_NARROW_RANGE_ADDITIVE: float = _POLICY.narrow_range_additive


def recommend_error_model(
    obs: pd.DataFrame,
    *,
    blq_burden: float,
    lloq: float | None,
    cmax_p95_p05_ratio: float | None,
    dv_cv_percent: float | None,
    terminal_log_mad: float | None,
) -> ErrorModelPreference:
    """Return the profiler's preferred error-model family for this dataset.

    Decision tree (priority order):
      1. ``blq_burden >= 0.10`` → BLQ_M3 with proportional/combined underlying.
         Never include additive-only: Ahn 2008 shows M3 gives biased estimates
         when combined with additive-only residual error under heavy censoring
         (the additive sigma absorbs the censored mass, corrupting CL).
      2. Wide dynamic range (Cmax_p95/Cmax_p05 > 50) with moderate CV
         (<80%) → proportional. Classic PK signal where sigma scales with
         concentration over the observed range.
      3. LLOQ/Cmax_median > 5% or terminal log-residual MAD > 0.35 →
         combined. Additive component matters near the LLOQ or when
         terminal noise is large in log-space.
      4. Narrow range (<5) and low CV → additive (rare; typical of
         narrow-range biomarker PD).
      5. Default → proportional, medium confidence.

    Returns ErrorModelPreference with primary, allowed set, confidence,
    rationale.
    """
    # 1. BLQ dominance — never allow additive-only.
    if blq_burden >= _BLQ_M3_TRIGGER:
        allowed: list[Literal["proportional", "additive", "combined"]] = [
            "proportional",
            "combined",
        ]
        return ErrorModelPreference(
            primary="blq_m3",
            allowed=allowed,
            confidence="high",
            rationale=(
                f"BLQ={blq_burden * 100:.1f}% >= {_BLQ_M3_TRIGGER * 100:.0f}%: "
                "Beal 2001 M3 required; additive-only excluded (Ahn 2008)"
            ),
        )

    cmax_median = _cmax_median(obs)
    # 3. LLOQ/Cmax or terminal noise signal → combined
    lloq_ratio = (lloq / cmax_median) if (lloq is not None and cmax_median > 0) else None
    terminal_noisy = terminal_log_mad is not None and terminal_log_mad > _TERMINAL_LOG_MAD_COMBINED
    lloq_close = lloq_ratio is not None and lloq_ratio > _LLOQ_CMAX_COMBINED
    if lloq_close or terminal_noisy:
        reason = []
        if lloq_close and lloq_ratio is not None:
            reason.append(f"LLOQ/Cmax_median={lloq_ratio * 100:.1f}%>5%")
        if terminal_noisy and terminal_log_mad is not None:
            reason.append(f"terminal log MAD={terminal_log_mad:.2f}>0.35")
        # R13: allowed must include
        # proportional alongside combined. Locking out proportional was
        # too aggressive; Ahn 2008 only forbids additive-only under heavy
        # BLQ, not proportional.
        return ErrorModelPreference(
            primary="combined",
            allowed=["combined", "proportional"],
            confidence="medium",
            rationale="; ".join(reason) + " → combined preferred (proportional allowed)",
        )

    # 2. Wide dynamic range with moderate CV → proportional
    if (
        cmax_p95_p05_ratio is not None
        and cmax_p95_p05_ratio > _DYNAMIC_RANGE_PROPORTIONAL
        and (dv_cv_percent is None or dv_cv_percent < _HIGH_CV_CEILING)
    ):
        return ErrorModelPreference(
            primary="proportional",
            allowed=["proportional", "combined"],
            confidence="high",
            rationale=(
                f"dynamic range={cmax_p95_p05_ratio:.1f}>50 with moderate CV → "
                "proportional error scales with concentration"
            ),
        )

    # 4. Narrow-range + low-CV biomarker → additive plausible
    if (
        cmax_p95_p05_ratio is not None
        and cmax_p95_p05_ratio < _NARROW_RANGE_ADDITIVE
        and dv_cv_percent is not None
        and dv_cv_percent < _LOW_CV_ADDITIVE_CEILING
    ):
        return ErrorModelPreference(
            primary="additive",
            allowed=["additive", "combined"],
            confidence="low",
            rationale=(
                f"narrow range={cmax_p95_p05_ratio:.1f} + CV={dv_cv_percent:.1f}% "
                f"(<{_LOW_CV_ADDITIVE_CEILING}%) → additive error plausible (biomarker-like)"
            ),
        )

    # 5. Default: proportional with combined as escape hatch
    return ErrorModelPreference(
        primary="proportional",
        allowed=["proportional", "combined"],
        confidence="medium",
        rationale="default preference: proportional with combined fallback",
    )


def _compute_cmax_dynamic_range(obs: pd.DataFrame) -> float | None:
    """Ratio of the 95th-percentile per-subject Cmax to the 5th-percentile.

    A stable proxy for "does sigma need to scale with concentration?".
    Computed across per-subject maxima (not across all DV values), so tail
    samples near LLOQ do not inflate the range in narrow-Cmax studies.
    Returns None when fewer than 10 subjects have positive concentrations.
    """
    if obs.empty:
        return None
    cmaxes: list[float] = []
    for _, subj in obs.groupby("NMID"):
        pos = subj[subj["DV"] > 0]["DV"]
        if not pos.empty:
            cmaxes.append(float(pos.max()))
    if len(cmaxes) < 10:
        return None
    arr = np.array(cmaxes, dtype=float)
    p95 = float(np.percentile(arr, 95))
    p05 = float(np.percentile(arr, 5))
    if p05 <= 0:
        return None
    return p95 / p05


def _compute_dv_cv_percent(obs: pd.DataFrame) -> float | None:
    """Population coefficient of variation (%) of positive DV values.

    Uses ``ddof=0`` for a population (not sample) CV; the heuristic thresholds
    in ``recommend_error_model`` are calibrated against population statistics.
    """
    if obs.empty:
        return None
    dv_pos = obs[obs["DV"] > 0]["DV"].to_numpy(dtype=float)
    if len(dv_pos) < 10:
        return None
    mean = float(np.mean(dv_pos))
    if mean <= 0:
        return None
    sd = float(np.std(dv_pos, ddof=0))
    return 100.0 * sd / mean


def _compute_terminal_log_residual_mad(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
    *,
    multi_dose: bool,
) -> float | None:
    """Median absolute deviation of log-concentration residuals from the
    per-subject terminal log-linear trend.

    R5 (BLQ mask) + R22 (multi-dose windowing) —
    . Operates on the BLQ-filtered, last-dose-window per-subject
    profile so censored DV=LLOQ rows do not bias slopes and multi-dose
    "terminal" does not span subsequent absorption rises.

    Larger values (>0.35) imply additive error component is needed near
    the tail (combined preferred).
    """
    if obs.empty:
        return None
    residuals: list[float] = []
    for subj, subj_obs in obs.groupby("NMID"):
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        # BLQ-aware filter applied via the source DataFrame, then windowed.
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 4:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 4:
            continue
        tmax_idx = int(np.argmax(concs))
        post_t = times[tmax_idx + 1 :]
        post_c = concs[tmax_idx + 1 :]
        if len(post_t) < 4:
            continue
        half = len(post_t) // 2
        term_t = post_t[half:]
        term_c = post_c[half:]
        if len(term_t) < 3:
            continue
        y = safe_log(term_c)
        mean_t = float(np.mean(term_t))
        mean_y = float(np.mean(y))
        ss_tt = float(np.sum((term_t - mean_t) ** 2))
        if ss_tt <= 0:
            continue
        slope = float(np.sum((term_t - mean_t) * (y - mean_y))) / ss_tt
        intercept = mean_y - slope * mean_t
        pred = intercept + slope * term_t
        residuals.extend(list(y - pred))
    if not residuals:
        return None
    res_arr = np.array(residuals)
    median = float(np.median(res_arr))
    return float(np.median(np.abs(res_arr - median)))


def _cmax_median(obs: pd.DataFrame) -> float:
    """Median per-subject Cmax (robust to outliers)."""
    if obs.empty:
        return 0.0
    cmax_values: list[float] = []
    for _, subj in obs.groupby("NMID"):
        concs = subj["DV"].to_numpy(dtype=float)
        if concs.size == 0:
            continue
        cmax_values.append(float(np.max(concs)))
    return float(np.median(cmax_values)) if cmax_values else 0.0


def _detect_time_varying_covariates(
    df: pd.DataFrame,
    manifest: DataManifest,
) -> bool:
    """Detect whether any covariate changes value within a subject over time.

    Drives FREM preference in the missing-data directive (Nyberg 2024, Jonsson
    2024): when covariates are time-varying, MI per occasion is awkward and
    FREM handles the missingness within the NLME likelihood.

    A covariate is considered time-varying if its per-subject standard
    deviation (treating NaN as a missing observation, not a new value) is
    non-zero for at least one subject, for at least one numeric covariate.
    Categorical covariates are compared by unique-value count per subject.
    """
    cov_names = [c.name for c in manifest.covariates if c.name in df.columns]
    if not cov_names:
        return False

    for col in cov_names:
        series = df[col]
        if series.isna().all():
            continue
        # Per-subject unique non-NaN value count. >1 ⇒ time-varying for that subject.
        for _, subj_series in df.groupby("NMID")[col]:
            if int(subj_series.dropna().nunique()) > 1:
                return True
    return False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient (no scipy dependency).

    Uses average ranking for ties (standard Spearman's rho).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    # Average-rank both arrays (handles ties correctly)
    x_ranks = _average_rank(x)
    y_ranks = _average_rank(y)
    # Pearson correlation on ranks
    x_mean = float(np.mean(x_ranks))
    y_mean = float(np.mean(y_ranks))
    num = float(np.sum((x_ranks - x_mean) * (y_ranks - y_mean)))
    den_x = float(np.sqrt(np.sum((x_ranks - x_mean) ** 2)))
    den_y = float(np.sqrt(np.sum((y_ranks - y_mean) ** 2)))
    if den_x <= 0 or den_y <= 0 or not (np.isfinite(den_x) and np.isfinite(den_y)):
        return 0.0
    return num / (den_x * den_y)


def _average_rank(arr: np.ndarray) -> np.ndarray:
    """Compute average ranks (ties get the mean of their ordinal positions)."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Average ties
    sorted_arr = arr[order]
    i = 0
    while i < len(sorted_arr):
        j = i + 1
        while j < len(sorted_arr) and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:
            avg_rank = float(np.mean(np.arange(i + 1, j + 1, dtype=float)))
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Individual profiling functions
# ---------------------------------------------------------------------------


def _assess_route_certainty(
    doses: pd.DataFrame,
) -> str:
    """Determine route certainty from dosing records.

    confirmed: all doses have consistent CMT and RATE/DUR patterns
    inferred: CMT is present but RATE/DUR ambiguous
    ambiguous: cannot determine route from data alone
    """
    if doses.empty:
        return "ambiguous"

    cmts = doses["CMT"].unique()

    # If RATE column exists and has non-zero values, likely infusion (IV)
    has_rate = "RATE" in doses.columns and (doses["RATE"] > 0).any()
    has_dur = "DUR" in doses.columns and (doses["DUR"] > 0).any()

    # Single CMT with rate/duration info → confirmed IV
    if len(cmts) == 1 and (has_rate or has_dur):
        return "confirmed"

    # Single CMT=1 with no rate info → inferred oral (CMT=1 can also be
    # central for IV; confirmed requires additional evidence like route column)
    if len(cmts) == 1 and cmts[0] == 1 and not has_rate and not has_dur:
        return "inferred"

    # Multiple CMTs or mixed patterns → inferred
    if len(cmts) > 1:
        return "inferred"

    return "inferred"


def _detect_multi_dose(doses: pd.DataFrame) -> bool:
    """True iff any subject has >=2 dose events.

    Drives the dose-interval slicing path used by
    ``_per_subject_terminal_monoexp_r2`` and the peak counter. Without
    slicing, the "post-Cmax terminal phase" of a multi-dose subject spans
    subsequent dose absorption rises and the monoexponential fit is
    meaningless (root cause of warfarin's terminal R² = -0.49 and the
    100% multi-phase absorption false positive).
    """
    if doses.empty or "NMID" not in doses.columns:
        return False
    counts = doses.groupby("NMID").size()
    return bool((counts >= 2).any())


def _last_dose_window(
    subj_doses: pd.DataFrame, subj_obs: pd.DataFrame
) -> tuple[float, float] | None:
    """Return ``(t_start, t_end)`` of the last dose interval, or None when
    the subject is single-dose / lacks dose records. ``t_end`` is the lesser
    of (last_dose_time + median_tau, last observation time) so we never
    project past observed data."""
    if subj_doses.empty or len(subj_doses) < 2:
        return None
    dt = np.sort(subj_doses["TIME"].to_numpy(dtype=float))
    intervals = np.diff(dt)
    if len(intervals) == 0:
        return None
    tau = float(np.median(intervals[intervals > 0])) if (intervals > 0).any() else 0.0
    if tau <= 0:
        return None
    t_start = float(dt[-1])
    t_end_obs = (
        float(subj_obs["TIME"].to_numpy(dtype=float).max()) if not subj_obs.empty else t_start
    )
    return t_start, min(t_start + tau, t_end_obs)


def _windowed_profile(
    subj_obs: pd.DataFrame,
    subj_doses: pd.DataFrame,
    *,
    multi_dose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, concs)`` restricted to the last dose interval when
    ``multi_dose=True`` AND the subject has a meaningful interval; else
    returns the full profile.

    Used by every per-subject shape detector so multi-dose accumulation
    does not contaminate single-dose-style heuristics.
    """
    s = subj_obs.sort_values("TIME")
    times = s["TIME"].to_numpy(dtype=float)
    concs = s["DV"].to_numpy(dtype=float)
    if not multi_dose:
        return times, concs
    window = _last_dose_window(subj_doses, subj_obs)
    if window is None:
        return times, concs
    t_start, t_end = window
    mask = (times >= t_start) & (times <= t_end)
    if mask.sum() < 3:
        return times, concs  # not enough samples in window — fall back
    return times[mask], concs[mask]


def _count_prominent_peaks(times: np.ndarray, concs: np.ndarray) -> int:
    """Count prominent peaks in a concentration-time profile.

    Uses scipy ``find_peaks`` with prominence and inter-peak distance
    guards. A naive local-maximum detector counts every descending-limb
    noise wiggle as a peak; the prominence threshold (relative to
    Cmax/dynamic-range) and distance constraint suppress these.

    Returns the number of peaks meeting both requirements; the caller
    decides whether >=2 indicates multi-phase absorption.
    """
    if len(concs) < 5 or not (concs > 0).any():
        return 0
    cmax = float(np.max(concs))
    cmin = float(np.min(concs[concs > 0]))
    prominence_floor = max(
        _PEAK_PROMINENCE_RANGE_FRACTION * (cmax - cmin),
        _PEAK_PROMINENCE_CMAX_FLOOR * cmax,
    )
    if prominence_floor <= 0:
        return 0
    # Distance is in samples, not time; convert ">=2 sampling intervals"
    # to a sample count (always >=1). Median sampling interval is robust
    # to outlier gaps in the schedule.
    interval_samples = max(1, round(_PEAK_MIN_DISTANCE_INTERVALS)) if len(times) >= 2 else 1
    peaks, _props = find_peaks(concs, prominence=prominence_floor, distance=interval_samples)
    return len(peaks)


def _assess_absorption_complexity(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> str:
    """Classify absorption complexity from concentration-time profiles.

    simple: monotone rise to Cmax then decline
    multi-phase: multiple PROMINENT peaks within the analysis window
    lag-signature: delayed onset (low concentrations at early times)
    unknown: insufficient data to characterize

    For multi-dose datasets the per-subject analysis window is restricted
    to the last dose interval — without this, every subsequent dose Cmax
    registers as a spurious secondary peak.
    """
    if obs.empty:
        return "unknown"

    subjects = obs["NMID"].unique()
    lag_count = 0
    multi_peak_count = 0

    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        times, concs = _windowed_profile(subj_obs, subj_doses, multi_dose=multi_dose)
        if len(concs) < 3:
            continue

        # Lag detection within the analysis window.
        if (times > 0).any():
            early_mask = times <= np.percentile(times[times > 0], 25)
            if early_mask.any():
                early_concs = concs[early_mask]
                max_conc = np.max(concs[concs > 0]) if (concs > 0).any() else 1.0
                if max_conc > 0 and np.all(early_concs < 0.05 * max_conc):
                    lag_count += 1

        if _count_prominent_peaks(times, concs) >= 2:
            multi_peak_count += 1

    n_analyzable = max(1, len(subjects))
    if multi_peak_count / n_analyzable > _MULTI_PEAK_FRACTION_THRESHOLD:
        return "multi-phase"
    if lag_count / n_analyzable > _LAG_SIGNATURE_FRACTION_THRESHOLD:
        return "lag-signature"
    return "simple"


def _peak_prominence_fraction(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> float | None:
    """Fraction of subjects with >=2 prominent peaks within their analysis
    window (last-dose interval for multi-dose, full profile otherwise)."""
    if obs.empty:
        return None
    subjects = obs["NMID"].unique()
    if len(subjects) == 0:
        return None
    n_analyzable = 0
    multi_peak = 0
    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        times, concs = _windowed_profile(subj_obs, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        n_analyzable += 1
        if _count_prominent_peaks(times, concs) >= 2:
            multi_peak += 1
    if n_analyzable == 0:
        return None
    return float(multi_peak / n_analyzable)


def _per_subject_terminal_monoexp_r2(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> list[float]:
    """Per-subject adjusted R² of monoexponential fit to the terminal
    ~30% of post-Cmax samples within the analysis window. Used to
    distinguish 2-compartment linear kinetics (high R² in the late beta
    phase) from MM saturation (R² sometimes also high once C << Km —
    caller must combine with curvature ratio + dose-proportionality,
    never use alone).

    For multi-dose data the per-subject window is restricted to the last
    dose interval; otherwise the "terminal phase" spans subsequent dose
    rises and the monoexp fit is meaningless (warfarin: R² = -0.49 across
    full profile, ~0.95 within the last dosing interval).
    """
    #  apply BLQ mask, use safe_log
    # via fit_best_lambdaz (Huang 2025) which selects max-adj-R² window with
    # the Phoenix WinNonlin time-anchor constraint.
    out: list[float] = []
    for subj in obs["NMID"].unique():
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        terminal = _fit_terminal(times, concs)
        if terminal is None:
            continue
        out.append(terminal.adj_r2)
    return out


def _filter_triples_to_ss_subjects(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
    triples: list[tuple[int, float, float]],
) -> list[tuple[float, float]]:
    """Drop (dose, AUC) pairs from subjects not at steady state.

    operates on (NMID, dose, AUC) triples to avoid the previous
    zip-mismatch bug where skipped subjects in `_get_dose_auc_pairs_impl`
    caused pairs to be re-attached to wrong subjects.

    Pre-SS AUCτ underestimates AUCss and biases Smith 2000 toward false
    saturation. For single-dose subjects (n_doses<2) SS is moot — pass
    through. For multi-dose subjects, gate via `is_steady_state`. Half-life
    is approximated from `fit_best_lambdaz` on the subject's last-dose-window.
    """
    if not triples or doses.empty:
        return [(d, a) for _, d, a in triples]
    dose_counts = doses.groupby("NMID").size()
    out: list[tuple[float, float]] = []
    for subj_id, dose, auc in triples:
        n_doses = int(dose_counts.get(subj_id, 0))
        if n_doses < 2:
            out.append((dose, auc))
            continue
        subj_doses = doses[doses["NMID"] == subj_id]
        subj_obs = obs[obs["NMID"] == subj_id]
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) >= 5:
            times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=True)
            terminal = _fit_terminal(times, concs) if len(concs) >= 5 else None
            half_life = terminal.half_life if terminal is not None else None
        else:
            half_life = None
        ss, _rationale = _policy_steady_state(
            dose_times=subj_doses["TIME"].to_numpy(dtype=float),
            doses=subj_doses["AMT"].to_numpy(dtype=float),
            half_life=half_life,
        )
        if ss:
            out.append((dose, auc))
    return out


def _get_dose_auc_triples(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
) -> list[tuple[int, float, float]]:
    """Per-subject (NMID, dose, AUC) triples — keyed by NMID so SS-gating
    can re-attach correctly. R1 +"""
    out: list[tuple[int, float, float]] = []
    for subj in obs["NMID"].unique():
        subj_obs_raw = obs[obs["NMID"] == subj].sort_values("TIME")
        subj_obs = subj_obs_raw[positive_unblqd_mask(subj_obs_raw)]
        subj_doses = doses[doses["NMID"] == subj]
        if len(subj_obs) < 3 or subj_doses.empty:
            continue
        total_dose = float(subj_doses["AMT"].sum())
        if total_dose <= 0:
            continue
        concs = subj_obs["DV"].to_numpy(dtype=float)
        times = subj_obs["TIME"].to_numpy(dtype=float)
        n_doses = len(subj_doses)
        if n_doses > 1:
            dose_times = subj_doses["TIME"].to_numpy(dtype=float)
            sorted_dt = np.sort(dose_times)
            intervals = np.diff(sorted_dt)
            if len(intervals) > 0 and np.median(intervals) > 0:
                tau = float(np.median(intervals))
                last_dose_time = float(sorted_dt[-1])
                tau_mask = (times >= last_dose_time) & (times <= last_dose_time + tau)
                if tau_mask.sum() >= 2:
                    auc = auc_linup_logdown(times[tau_mask], concs[tau_mask])
                    single_dose = float(subj_doses["AMT"].median())
                    if single_dose > 0 and auc > 0:
                        out.append((int(subj), single_dose, auc))
                    continue
        auc = auc_linup_logdown(times, concs)
        if auc > 0:
            out.append((int(subj), total_dose, auc))
    return out


# R3: _detect_curvature_nonlinearity
# was deleted (a divergent duplicate of _curvature_ratios_per_subject).
# _detect_nonlinear_clearance retained as a thin shim over the strength
# classifier so existing tests (and any external callers) continue to
# work; new code should consume `nonlinear_clearance_evidence_strength`
# directly from the manifest.


def _detect_nonlinear_clearance(obs: pd.DataFrame, doses: pd.DataFrame) -> bool:
    """Boolean shortcut: True iff the strength classifier returns
    `moderate` or `strong`. Wraps the unified pipeline so the boolean
    and graded outputs cannot drift apart.
    """
    if obs.empty or doses.empty:
        return False
    multi_dose = _detect_multi_dose(doses)
    strength, _signals = _classify_nonlinear_clearance_evidence_strength(
        obs, doses, multi_dose=multi_dose
    )
    return strength in {"moderate", "strong"}


def _classify_nonlinear_clearance_evidence_strength(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
    *,
    multi_dose: bool,
) -> tuple[
    Literal["none", "weak", "moderate", "strong"],
    dict[SignalId, NonlinearClearanceSignal],
]:
    """Multi-signal voting for nonlinear-clearance evidence (PRD §10 Q2 follow-up).

    Returns ``(strength, signals)`` where ``signals`` carries the median
    diagnostic values used to derive ``strength`` so they can be emitted on
    the EvidenceManifest for downstream auditability.

    Three independent signals contribute one vote each. ``strong`` requires
    all three; ``moderate`` requires two; ``weak`` requires one; ``none``
    means no signal triggered.

    Signal 1 — curvature ratio: median early/late post-Cmax log-slope ratio
    > _NONLINEAR_MM_CURVATURE_RATIO. By itself this fires on 2-cmt linear
    drugs (e.g., warfarin → 3.92) so it is necessary but not sufficient.

    Signal 2 — terminal monoexponential failure: median per-subject adj R²
    of a log-linear fit to the final ~30% of post-Cmax samples
    < _TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD. A passing R² is evidence
    against MM (the late phase is well-explained by linear elimination).
    A failing R² alone does not prove MM (could be sparse/noisy data).

    Signal 3 - dose-AUC nonproportionality: Smith 2000 power model
    log(AUC)=alpha+beta·log(dose) with translated beta bounds derived from
    [θ_L, θ_H]=[0.80, 1.25] and observed dose ratio. Eligible only when
    ≥3 distinct dose levels AND dose ratio ≥3-fold AND ≥4 SS-gated
    pairs (R10 steady-state filter via is_steady_state). Saturation
    flag fires when beta CI lower bound exceeds the Smith high bound;
    induction when CI upper bound is below the Smith low bound. The
    strongest single discriminator when present; abstains when design
    cannot support the test or pre-SS contamination is suspected.
    """
    signals: dict[SignalId, NonlinearClearanceSignal] = {}
    votes = 0
    eligible = 0

    # Signal 1: curvature ratio + population bootstrap CI.
    ratios = _curvature_ratios_per_subject(obs, doses, multi_dose=multi_dose)
    n_curv = len(ratios)
    if n_curv >= 4:
        eligible += 1
        ci = bootstrap_median_ci(ratios)
        ci_low: float | None
        ci_high: float | None
        if ci is not None:
            median_ratio, ci_low, ci_high = ci
        else:
            median_ratio = float(np.median(ratios))
            ci_low = ci_high = None
        voted = median_ratio > _NONLINEAR_MM_CURVATURE_RATIO
        if voted:
            votes += 1
        signals[SignalId.CURVATURE_RATIO] = NonlinearClearanceSignal(
            signal_id=SignalId.CURVATURE_RATIO,
            algorithm="early_vs_late_post_cmax_log_slope_ratio",
            citation="Richardson 2025 (Commun Med 5:327)",
            policy_key="policies/profiler.json#/mm_curvature_ratio",
            threshold_value=_NONLINEAR_MM_CURVATURE_RATIO,
            observed_value=median_ratio,
            ci90_low=ci_low,
            ci90_high=ci_high,
            eligible=True,
            voted=voted,
            n_subjects=n_curv,
        )
    else:
        signals[SignalId.CURVATURE_RATIO] = NonlinearClearanceSignal(
            signal_id=SignalId.CURVATURE_RATIO,
            algorithm="early_vs_late_post_cmax_log_slope_ratio",
            citation="Richardson 2025 (Commun Med 5:327)",
            policy_key="policies/profiler.json#/mm_curvature_ratio",
            threshold_value=_NONLINEAR_MM_CURVATURE_RATIO,
            eligible=False,
            eligibility_reason=(
                f"fewer than 4 analyzable subjects (n={n_curv}); "
                "insufficient for population-median voting"
            ),
            voted=False,
            n_subjects=n_curv,
        )

    # Signal 2: terminal monoexponential fit.
    r2s = _per_subject_terminal_monoexp_r2(obs, doses, multi_dose=multi_dose)
    n_r2 = len(r2s)
    if n_r2 >= 4:
        eligible += 1
        median_r2 = float(np.median(r2s))
        voted = median_r2 < _TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD
        if voted:
            votes += 1
        signals[SignalId.TERMINAL_MONOEXP] = NonlinearClearanceSignal(
            signal_id=SignalId.TERMINAL_MONOEXP,
            algorithm="huang_2025_lambda_z_adj_r2",
            citation="Huang 2025 (nlmixr2autoinit::find_best_lambdaz)",
            policy_key="policies/profiler.json#/terminal_monoexp_r2_linear_threshold",
            threshold_value=_TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD,
            observed_value=median_r2,
            eligible=True,
            voted=voted,
            n_subjects=n_r2,
        )
    else:
        signals[SignalId.TERMINAL_MONOEXP] = NonlinearClearanceSignal(
            signal_id=SignalId.TERMINAL_MONOEXP,
            algorithm="huang_2025_lambda_z_adj_r2",
            citation="Huang 2025 (nlmixr2autoinit::find_best_lambdaz)",
            policy_key="policies/profiler.json#/terminal_monoexp_r2_linear_threshold",
            threshold_value=_TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD,
            eligible=False,
            eligibility_reason=(f"fewer than 4 subjects with fittable terminal phase (n={n_r2})"),
            voted=False,
            n_subjects=n_r2,
        )

    # Signal 3: Smith 2000 dose-proportionality with SS gating.
    triples = _get_dose_auc_triples(obs, doses)
    ss_gated_pairs = _filter_triples_to_ss_subjects(obs, doses, triples)
    smith_fit = fit_dose_proportionality(ss_gated_pairs)
    smith_extras: dict[str, float | int | str | bool | None] = {
        "dose_ratio": smith_fit.dose_ratio,
        "beta_smith_low": smith_fit.beta_smith_low,
        "beta_smith_high": smith_fit.beta_smith_high,
        "n_pairs": len(ss_gated_pairs),
    }
    if smith_fit.eligible:
        eligible += 1
        voted = bool(smith_fit.saturation_flag)
        if voted:
            votes += 1
        signals[SignalId.DOSE_PROPORTIONALITY_SMITH] = NonlinearClearanceSignal(
            signal_id=SignalId.DOSE_PROPORTIONALITY_SMITH,
            algorithm="smith_2000_power_model",
            citation="Smith 2000 (Clin Pharmacokinet 38:1-16)",
            policy_key="policies/profiler.json#/smith_theta_bounds",
            threshold_value=smith_fit.beta_smith_high,
            observed_value=smith_fit.beta,
            ci90_low=smith_fit.beta_ci90_low,
            ci90_high=smith_fit.beta_ci90_high,
            eligible=True,
            voted=voted,
            n_subjects=len(ss_gated_pairs),
            extras=smith_extras,
        )
    else:
        signals[SignalId.DOSE_PROPORTIONALITY_SMITH] = NonlinearClearanceSignal(
            signal_id=SignalId.DOSE_PROPORTIONALITY_SMITH,
            algorithm="smith_2000_power_model",
            citation="Smith 2000 (Clin Pharmacokinet 38:1-16)",
            policy_key="policies/profiler.json#/smith_theta_bounds",
            eligible=False,
            eligibility_reason=smith_fit.rationale,
            voted=False,
            n_subjects=len(ss_gated_pairs),
            extras=smith_extras,
        )

    # Strength is the fraction of *eligible* signals that fired, not an
    # absolute count out of three. This avoids misclassifying single-dose
    # true-MM datasets (Suite A2/A4) as merely "weak" when only Signal 1
    # could possibly trigger.
    strength: Literal["none", "weak", "moderate", "strong"]
    if eligible == 0 or votes == 0:
        strength = "none"
    else:
        ratio = votes / eligible
        if ratio >= 1.0:
            strength = "strong"
        elif ratio >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"
    return strength, signals


def _curvature_ratios_per_subject(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> list[float]:
    """Per-subject early/late post-Cmax log-slope ratios.

    Shared helper used by the evidence-strength classifier and
    compartmentality assessor. Profile is restricted to the last dose
    interval when ``multi_dose=True`` to avoid contamination from
    subsequent dose rises.
    """
    # R3 + R5: BLQ mask + safe_log; underflow-safe slope ratio.
    # Note: 3-anchor-point ratio is retained here for back-compat with the
    # strength classifier; R4 (two-segment OLS + bootstrap CI) is a
    # follow-up enhancement.
    ratios: list[float] = []
    for subj in obs["NMID"].unique():
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 4:
            continue
        tmax_idx = int(np.argmax(concs))
        post_c = concs[tmax_idx:]
        post_t = times[tmax_idx:]
        if len(post_c) < 4:
            continue
        log_c = safe_log(post_c)
        mid = len(post_c) // 2
        if mid < 2 or len(post_c) - mid < 2:
            continue
        dt_e = post_t[mid] - post_t[0]
        dt_l = post_t[-1] - post_t[mid]
        if dt_e <= 0 or dt_l <= 0:
            continue
        es = abs((log_c[mid] - log_c[0]) / dt_e)
        ls = abs((log_c[-1] - log_c[mid]) / dt_l)
        if ls > 1e-6:
            ratios.append(es / ls)
    return ratios


def _assess_compartmentality(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
    *,
    multi_dose: bool,
) -> Literal["one_compartment", "two_compartment", "multi_compartment_likely", "insufficient"]:
    """Classify likely compartmentality from terminal-phase shape.

    one_compartment: monoexponential late phase (median terminal adj R² high)
        AND modest early/late curvature (ratio close to 1).
    two_compartment: monoexponential terminal R² high AND elevated curvature
        ratio (steep alpha + shallow beta — classic biexponential).
    multi_compartment_likely: terminal R² fails AND curvature elevated
        (poorly captured beta phase or true sum-of-exponentials).
    insufficient: not enough analyzable subjects.
    """
    # R7: use a SEPARATE, lower
    # curvature threshold for biexponential detection. Re-using the MM
    # threshold (1.8) collapsed 2cmt and MM into the same bucket: a 2cmt
    # drug with steep alpha + shallow beta (warfarin-like) hits >1.8 and
    # a true MM with C<<Km in the tail also hits >1.8. The biexponential
    # signature is *moderate* curvature (alpha/beta ratio typically 1.3-3
    # for clinical drugs); MM saturation produces sharper curvature plus
    # other corroborating signals (dose nonproportionality, terminal R²
    # failure) which are scored independently in nlc_evidence_strength.
    r2s = _per_subject_terminal_monoexp_r2(obs, doses, multi_dose=multi_dose)
    ratios = _curvature_ratios_per_subject(obs, doses, multi_dose=multi_dose)
    if len(r2s) < 4 or len(ratios) < 4:
        return "insufficient"
    median_r2 = float(np.median(r2s))
    median_ratio = float(np.median(ratios))
    high_r2 = median_r2 >= _TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD
    elevated_curvature_2cmt = median_ratio > _COMPARTMENTALITY_CURVATURE_RATIO
    if high_r2 and not elevated_curvature_2cmt:
        return "one_compartment"
    if high_r2 and elevated_curvature_2cmt:
        return "two_compartment"
    return "multi_compartment_likely"


def _lambda_z_analyzable_fraction(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> float | None:
    """Fraction of subjects with >=3 positive non-BLQ post-Cmax samples
    within their analysis window — the minimum required for terminal-slope
    estimation. R5 (BLQ-aware) + R22 (multi-dose windowing) — multi-model
    consensus .
    """
    if obs.empty:
        return None
    subjects = obs["NMID"].unique()
    if len(subjects) == 0:
        return None
    analyzable = 0
    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        _times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        tmax_idx = int(np.argmax(concs))
        post = concs[tmax_idx:]
        if (post > 0).sum() >= _TERMINAL_FIT_MIN_POINTS:
            analyzable += 1
    return float(analyzable / len(subjects))


def _auc_extrap_fraction_median(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> float | None:
    """Median across subjects of AUC_extrap / AUC_inf.

    R6 (Huang 2025 best-lambda_z selector with Phoenix constraint,
    review) + R1 (linear-up/log-down AUC, Wagner & Nelson 1963) + R5 (BLQ
    mask) + R22 (multi-dose windowing). Replaces the previous
    break-on-first lambda_z selector that systematically picked the shortest
    tail.
    """
    if obs.empty or doses.empty:
        return None
    fractions: list[float] = []
    for subj in obs["NMID"].unique():
        subj_obs = obs[obs["NMID"] == subj]
        subj_doses = doses[doses["NMID"] == subj] if not doses.empty else doses
        subj_obs_filt = subj_obs[positive_unblqd_mask(subj_obs)]
        if len(subj_obs_filt) < 5:
            continue
        times, concs = _windowed_profile(subj_obs_filt, subj_doses, multi_dose=multi_dose)
        if len(concs) < 5:
            continue
        terminal = _fit_terminal(times, concs)
        if terminal is None or terminal.lambda_z is None:
            continue
        # Linear-up / log-down trapezoid for AUC_last (R1).
        auc_last = auc_linup_logdown(times, concs)
        if auc_last <= 0:
            continue
        c_last = float(concs[-1])
        auc_inf = auc_last + c_last / terminal.lambda_z
        if auc_inf <= 0:
            continue
        fractions.append((c_last / terminal.lambda_z) / auc_inf)
    if not fractions:
        return None
    return float(np.median(fractions))


def _nonlinear_clearance_confidence(
    obs: pd.DataFrame, doses: pd.DataFrame, *, multi_dose: bool
) -> float | None:
    """Confidence score for nonlinear clearance detection (0.0-1.0).

    R3 +  routes through the same unified
    pipeline as the strength classifier — Smith 2000 power model with
    SS-gated triples (instead of legacy Spearman-on-raw-pairs) and the
    unified `_curvature_ratios_per_subject` helper. Previously this
    function had its own inline curvature loop and used Spearman
    without SS gating, so it could disagree with the strength label.

    Score is the maximum of:
      - Smith beta saturation/induction excess (clipped 0-1, derived from
        how far the beta CI lies outside the translated bounds).
      - Curvature-ratio score: (median_ratio - 1) / 2 clipped to [0, 1].
    """
    if obs.empty or doses.empty:
        return None
    scores: list[float] = []
    triples = _get_dose_auc_triples(obs, doses)
    ss_pairs = _filter_triples_to_ss_subjects(obs, doses, triples)
    smith_fit = fit_dose_proportionality(ss_pairs)
    # Saturation excess: how far CI lower bound exceeds the high Smith
    # bound, normalised by the bound's offset from 1.0.
    if (
        smith_fit.eligible
        and smith_fit.beta is not None
        and smith_fit.beta_smith_high is not None
        and smith_fit.beta_ci90_low is not None
        and smith_fit.beta_smith_high > 1.0
    ):
        excess = (smith_fit.beta_ci90_low - smith_fit.beta_smith_high) / (
            smith_fit.beta_smith_high - 1.0
        )
        scores.append(max(0.0, min(1.0, excess)))
    ratios = _curvature_ratios_per_subject(obs, doses, multi_dose=multi_dose)
    if len(ratios) >= 4:
        median_ratio = float(np.median(ratios))
        curvature_score = max(0.0, min(1.0, (median_ratio - 1.0) / 2.0))
        scores.append(curvature_score)
    if not scores:
        return None
    return float(max(scores))


def _classify_richness(obs: pd.DataFrame, n_subjects: int) -> str:
    """Classify sampling richness per PRD §4.2.1.

    sparse:   < ``_MIN_OBS_PER_SUBJECT_MODERATE`` samples/subject
    moderate: in [``_MIN_OBS_PER_SUBJECT_MODERATE``, ``_MIN_OBS_PER_SUBJECT_RICH``]
    rich:     > ``_MIN_OBS_PER_SUBJECT_RICH`` samples/subject

    Thresholds are sourced from ``policies/profiler.json#/subject_quality``.
    """
    if n_subjects == 0:
        return "sparse"

    # Use median samples per subject (robust to PK-intensive outliers)
    per_subj_counts = obs.groupby("NMID").size()
    median_samples = float(per_subj_counts.median()) if len(per_subj_counts) > 0 else 0.0

    if median_samples < _MIN_OBS_PER_SUBJECT_MODERATE:
        return "sparse"
    if median_samples <= _MIN_OBS_PER_SUBJECT_RICH:
        return "moderate"
    return "rich"


def _assess_identifiability(obs: pd.DataFrame, n_subjects: int) -> str:
    """Assess identifiability ceiling based on design and sampling.

    high:   ≥ ``_NODE_DISCOVERY_MIN_SUBJECTS`` subjects AND median samples
            > ``_MIN_OBS_PER_SUBJECT_RICH``
    medium: ≥ ``_NODE_OPTIMIZATION_MIN_SUBJECTS`` subjects AND median samples
            ≥ ``_MIN_OBS_PER_SUBJECT_MODERATE``
    low:    otherwise

    Thresholds are sourced from ``policies/profiler.json#/subject_quality``
    and ``/node_readiness``. Median samples per subject is used (not mean)
    to stay robust to PK-intensive outliers in heterogeneous studies (R11).
    """
    # Half-threshold short-circuit mirrors the NODE budget floor: too few
    # subjects to power even the optimization lane → low.
    if n_subjects < _NODE_OPTIMIZATION_MIN_SUBJECTS // 2:
        return "low"
    per_subj_counts = obs.groupby("NMID").size()
    median_samples = float(per_subj_counts.median()) if len(per_subj_counts) > 0 else 0.0
    if median_samples > _MIN_OBS_PER_SUBJECT_RICH and n_subjects >= _NODE_DISCOVERY_MIN_SUBJECTS:
        return "high"
    if (
        median_samples >= _MIN_OBS_PER_SUBJECT_MODERATE
        and n_subjects >= _NODE_OPTIMIZATION_MIN_SUBJECTS
    ):
        return "medium"
    return "low"


def _check_covariate_correlation(
    df: pd.DataFrame,
    manifest: DataManifest,
) -> bool:
    """Check if any covariates are correlated (|r| > 0.7)."""
    cov_names = [
        c.name for c in manifest.covariates if c.type == "continuous" and c.name in df.columns
    ]
    if len(cov_names) < 2:
        return False

    # Get unique per-subject covariate values
    subj_covs = df.groupby("NMID")[cov_names].first()
    # Numeric only
    numeric_covs = subj_covs.select_dtypes(include=[np.number])
    if numeric_covs.shape[1] < 2:
        return False

    corr_matrix = numeric_covs.corr().abs()
    corr_arr = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 0)
    return bool((corr_arr > _COVARIATE_CORRELATION_THRESHOLD).any())


def _assess_covariate_missingness(
    df: pd.DataFrame,
    manifest: DataManifest,
) -> CovariateSpec | None:
    """Assess covariate missingness pattern and fraction."""
    cov_names = [c.name for c in manifest.covariates if c.name in df.columns]
    if not cov_names:
        return None

    # Per-subject covariate completeness
    subj_covs = df.groupby("NMID")[cov_names].first()
    total_cells = subj_covs.size
    if total_cells == 0:
        return None

    missing_cells = int(subj_covs.isna().sum().sum())
    frac = missing_cells / total_cells

    if frac == 0:
        return None

    # Simple pattern classification:
    # If missingness is independent of outcome → MCAR
    # If missingness correlates with other observed covariates → MAR
    # Default to MAR as conservative assumption
    pattern: str = "MAR"

    strategy = "impute-median" if frac <= _BLQ_COVARIATE_MISSINGNESS_CUTOFF else "full-information"

    return CovariateSpec(
        pattern=pattern,
        fraction_incomplete=round(frac, 4),
        strategy=strategy,
    )


def _compute_blq_burden(df: pd.DataFrame) -> float:
    """Compute fraction of BLQ observations."""
    obs = df[df["EVID"] == 0]
    if obs.empty:
        return 0.0

    if "BLQ_FLAG" in df.columns:
        n_blq = int((obs["BLQ_FLAG"] == 1).sum())
        return n_blq / len(obs)
    return 0.0


def _extract_lloq_value(df: pd.DataFrame) -> float | None:
    """Extract the LLOQ value from data when BLQ observations exist.

    Priority:
      1. Explicit LLOQ column (takes most common value among BLQ rows)
      2. DV value of BLQ rows (for M3-style where DV=LLOQ on censored rows)
      3. None when no BLQ observations
    """
    obs = df[df["EVID"] == 0]
    if obs.empty or "BLQ_FLAG" not in df.columns:
        return None

    blq_rows = obs[obs["BLQ_FLAG"] == 1]
    if blq_rows.empty:
        return None

    if "LLOQ" in df.columns:
        # Use the most common non-null LLOQ value across censored rows
        lloq_values = blq_rows["LLOQ"].dropna()
        if not lloq_values.empty:
            return float(lloq_values.mode().iloc()[0])

    # Fallback: DV of censored rows (M3 convention sets DV=LLOQ)
    dv_values = blq_rows["DV"].dropna()
    if not dv_values.empty:
        return float(dv_values.mode().iloc()[0])

    return None


def _assess_protocol_heterogeneity(df: pd.DataFrame) -> str:
    """Assess protocol heterogeneity.

    single-study: one study or no STUDY_ID column
    pooled-similar: multiple studies, similar designs
    pooled-heterogeneous: multiple studies, different designs
    """
    if "STUDY_ID" not in df.columns:
        return "single-study"

    studies = df["STUDY_ID"].nunique()
    if studies <= 1:
        return "single-study"

    # Check design similarity: compare sampling schedules across studies
    per_study_n_obs = df[df["EVID"] == 0].groupby("STUDY_ID").size()
    per_study_n_subj = df.groupby("STUDY_ID")["NMID"].nunique()

    if len(per_study_n_obs) < 2:
        return "single-study"

    # Coefficient of variation of observations-per-subject across studies.
    # Threshold from policies/profiler.json#/protocol_heterogeneity.
    obs_per_subj = per_study_n_obs / per_study_n_subj
    cv = obs_per_subj.std() / max(obs_per_subj.mean(), 1e-6)
    if cv > _PROTOCOL_HETEROGENEITY_CV_THRESHOLD:
        return "pooled-heterogeneous"

    return "pooled-similar"


def _assess_absorption_coverage(obs: pd.DataFrame) -> str:
    """Assess whether absorption phase is adequately sampled.

    adequate:   avg pre-Tmax obs/subject ≥ ``_ABSORPTION_COVERAGE_MIN_PRE_TMAX``
    inadequate: otherwise

    Threshold is sourced from
    ``policies/profiler.json#/subject_quality/absorption_coverage_min_pre_tmax``.
    """
    if obs.empty:
        return "inadequate"

    subjects = obs["NMID"].unique()
    pre_tmax_counts: list[int] = []

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        if subj_data.empty:
            continue
        # Restrict to positive DV so BLQ zeros do not masquerade as Tmax.
        pos = subj_data[subj_data["DV"] > 0]
        if pos.empty:
            continue
        tmax = pos.loc[pos["DV"].idxmax(), "TIME"]
        pre_tmax = int((subj_data["TIME"] < tmax).sum())
        pre_tmax_counts.append(pre_tmax)

    if not pre_tmax_counts:
        return "inadequate"

    avg_pre_tmax = sum(pre_tmax_counts) / len(pre_tmax_counts)
    return "adequate" if avg_pre_tmax >= _ABSORPTION_COVERAGE_MIN_PRE_TMAX else "inadequate"


def _assess_elimination_coverage(obs: pd.DataFrame) -> str:
    """Assess whether elimination phase is adequately sampled.

    adequate:   avg post-Tmax obs/subject ≥ ``_ELIMINATION_COVERAGE_MIN_POST_TMAX``
    inadequate: otherwise

    Threshold is sourced from
    ``policies/profiler.json#/subject_quality/elimination_coverage_min_post_tmax``.
    """
    if obs.empty:
        return "inadequate"

    subjects = obs["NMID"].unique()
    post_tmax_counts: list[int] = []

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        if subj_data.empty:
            continue
        # R5 (Beal 2001 / Ahn 2008): Tmax must be derived from positive,
        # non-BLQ DV — otherwise a censored DV=LLOQ row can become argmax
        # and post-Tmax counts collapse. Mirrors the absorption-coverage fix.
        pos = subj_data[positive_unblqd_mask(subj_data)]
        if pos.empty:
            continue
        tmax = pos.loc[pos["DV"].idxmax(), "TIME"]
        post_tmax = int((subj_data["TIME"] > tmax).sum())
        post_tmax_counts.append(post_tmax)

    if not post_tmax_counts:
        return "inadequate"

    avg_post_tmax = sum(post_tmax_counts) / len(post_tmax_counts)
    return "adequate" if avg_post_tmax >= _ELIMINATION_COVERAGE_MIN_POST_TMAX else "inadequate"
