# SPDX-License-Identifier: GPL-2.0-or-later
"""Data Profiler: analyzes ingested PK data and emits an Evidence Manifest (PRD §4.2.1).

The profiler is not merely descriptive — its output constrains downstream dispatch.
Rule-based analysis produces typed manifest fields that the Lane Router must consume.

Dispatch constraints derived from manifest (PRD §4.2.1):
  - richness=sparse + absorption_coverage=inadequate → NODE not dispatched
  - nonlinear_clearance_signature=True → automated search includes MM candidates
  - blq_burden > 0.20 → all backends must use BLQ-aware likelihood (M3/M4)
  - protocol_heterogeneity=pooled-heterogeneous → IOV must be tested
  - covariate_missingness.fraction_incomplete > 0.15 → full-information likelihood
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd  # noqa: TC002

from apmode.bundle.models import CovariateSpec, ErrorModelPreference, EvidenceManifest

# Profiler thresholds. These should migrate to a versioned policy artifact
# (`policies/profiler.json`, analogous to gate policies) so bundle consumers
# can recover exactly which cutoffs produced an Evidence Manifest. Until then,
# named constants document intent and localize future updates.
_COVARIATE_CORRELATION_THRESHOLD: float = 0.7  # |r| > 0.7 flagged as correlated
_NONLINEAR_CLEARANCE_DOSE_AUC_THRESHOLD: float = 0.4  # Spearman rho threshold
_NONLINEAR_MM_CURVATURE_RATIO: float = 1.8  # early/late slope ratio
_NONLINEAR_TMDD_CURVATURE_RATIO: float = 0.3  # inverse pattern
_MULTI_PEAK_FRACTION_THRESHOLD: float = 0.3  # fraction of subjects with ≥2 peaks
_LAG_SIGNATURE_FRACTION_THRESHOLD: float = 0.5  # fraction of subjects with lag
_BLQ_COVARIATE_MISSINGNESS_CUTOFF: float = 0.15  # above → full-information likelihood

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
    obs = cast("pd.DataFrame", df[df["EVID"] == 0].copy())
    doses = cast("pd.DataFrame", df[df["EVID"] == 1].copy())
    n_subjects = int(cast("int", df["NMID"].nunique()))

    blq_burden = _compute_blq_burden(df)
    lloq_value = _extract_lloq_value(df)
    cmax_p95_p05_ratio = _compute_cmax_dynamic_range(obs)
    dv_cv_percent = _compute_dv_cv_percent(obs)
    terminal_log_mad = _compute_terminal_log_residual_mad(obs)
    error_pref = recommend_error_model(
        obs,
        blq_burden=blq_burden,
        lloq=lloq_value,
        cmax_p95_p05_ratio=cmax_p95_p05_ratio,
        dv_cv_percent=dv_cv_percent,
        terminal_log_mad=terminal_log_mad,
    )

    return EvidenceManifest(
        data_sha256=manifest.data_sha256,
        route_certainty=_assess_route_certainty(doses),
        absorption_complexity=_assess_absorption_complexity(obs),
        nonlinear_clearance_signature=_detect_nonlinear_clearance(obs, doses),
        nonlinear_clearance_confidence=_nonlinear_clearance_confidence(obs, doses),
        richness_category=_classify_richness(obs, n_subjects),
        identifiability_ceiling=_assess_identifiability(obs, n_subjects),
        covariate_burden=len(manifest.covariates),
        covariate_correlated=_check_covariate_correlation(df, manifest),
        covariate_missingness=_assess_covariate_missingness(df, manifest),
        time_varying_covariates=_detect_time_varying_covariates(df, manifest),
        blq_burden=blq_burden,
        lloq_value=lloq_value,
        protocol_heterogeneity=_assess_protocol_heterogeneity(df),
        absorption_phase_coverage=_assess_absorption_coverage(obs),
        elimination_phase_coverage=_assess_elimination_coverage(obs),
        error_model_preference=error_pref,
        cmax_p95_p05_ratio=cmax_p95_p05_ratio,
        dv_cv_percent=dv_cv_percent,
        terminal_log_residual_mad=terminal_log_mad,
    )


# ---------------------------------------------------------------------------
# Error-model selection heuristic (Beal 2001, Ahn 2008 + multi-agent consensus)
# ---------------------------------------------------------------------------

# Thresholds informed by the multi-agent research consensus + cited papers.
_BLQ_M3_TRIGGER: float = 0.10  # Beal 2001 / Ahn 2008: M3 preferred for BLQ ≥ 10%.
_DYNAMIC_RANGE_PROPORTIONAL: float = 50.0  # Cmax_p95/Cmax_p05 > 50 → proportional.
_HIGH_CV_CEILING: float = 80.0  # CV > 80 with big range → signal is noise-dominated.
_LLOQ_CMAX_COMBINED: float = 0.05  # LLOQ / Cmax_median > 5% → combined needed.
_TERMINAL_LOG_MAD_COMBINED: float = 0.35  # noisy terminal in log-space → combined.
_NARROW_RANGE_ADDITIVE: float = 5.0  # range_ratio < 5 + low CV → additive plausible.


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
      1. ``blq_burden ≥ 0.10`` → BLQ_M3 with proportional/combined underlying.
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
                f"BLQ={blq_burden * 100:.1f}% ≥ {_BLQ_M3_TRIGGER * 100:.0f}%: "
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
        return ErrorModelPreference(
            primary="combined",
            allowed=["combined"],
            confidence="medium",
            rationale="; ".join(reason) + " → combined error needed",
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
        and dv_cv_percent < 30.0
    ):
        return ErrorModelPreference(
            primary="additive",
            allowed=["additive", "combined"],
            confidence="low",
            rationale=(
                f"narrow range={cmax_p95_p05_ratio:.1f} + CV={dv_cv_percent:.1f}% → "
                "additive error plausible (biomarker-like)"
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
    """Ratio of the 95th-percentile DV to the 5th-percentile DV (positive only).

    A stable proxy for "does sigma need to scale with concentration?". Returns
    None when fewer than 20 positive observations exist.
    """
    if obs.empty:
        return None
    dv_pos = obs[obs["DV"] > 0]["DV"].to_numpy(dtype=float)
    if len(dv_pos) < 20:
        return None
    p95 = float(np.percentile(dv_pos, 95))
    p05 = float(np.percentile(dv_pos, 5))
    if p05 <= 0:
        return None
    return p95 / p05


def _compute_dv_cv_percent(obs: pd.DataFrame) -> float | None:
    """Coefficient of variation (%) of positive DV values."""
    if obs.empty:
        return None
    dv_pos = obs[obs["DV"] > 0]["DV"].to_numpy(dtype=float)
    if len(dv_pos) < 10:
        return None
    mean = float(np.mean(dv_pos))
    if mean <= 0:
        return None
    sd = float(np.std(dv_pos, ddof=1))
    return 100.0 * sd / mean


def _compute_terminal_log_residual_mad(obs: pd.DataFrame) -> float | None:
    """Median absolute deviation of log-concentration residuals from the
    per-subject terminal log-linear trend.

    Computed per-subject using the last half of post-Tmax points; combined
    MAD across subjects is reported. Larger values (>0.35) imply additive
    error component is needed near the tail (combined preferred).
    """
    if obs.empty:
        return None
    residuals: list[float] = []
    for _, subj in obs.groupby("NMID"):
        subj_sorted = subj.sort_values("TIME")
        times = subj_sorted["TIME"].to_numpy(dtype=float)
        concs = subj_sorted["DV"].to_numpy(dtype=float)
        pos = concs > 0
        if pos.sum() < 4:
            continue
        t = times[pos]
        c = concs[pos]
        tmax_idx = int(np.argmax(c))
        post_t = t[tmax_idx + 1 :]
        post_c = c[tmax_idx + 1 :]
        if len(post_t) < 4:
            continue
        half = len(post_t) // 2
        term_t = post_t[half:]
        term_c = post_c[half:]
        if len(term_t) < 3:
            continue
        y = np.log(np.clip(term_c, 1e-100, 1e100))
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


def _assess_absorption_complexity(obs: pd.DataFrame) -> str:
    """Classify absorption complexity from concentration-time profiles.

    simple: monotone rise to Cmax then decline
    multi-phase: multiple peaks or shoulder (secondary absorption)
    lag-signature: delayed onset (low concentrations at early times)
    unknown: insufficient data to characterize
    """
    if obs.empty:
        return "unknown"

    # Per-subject analysis: check for delayed onset and multiple peaks
    subjects = obs["NMID"].unique()
    lag_count = 0
    multi_peak_count = 0

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        times = subj_data["TIME"].values
        concs = subj_data["DV"].values

        if len(concs) < 3:
            continue

        # Lag detection: first 2 observations near zero while later ones rise
        if (times > 0).any():
            early_mask = times <= np.percentile(times[times > 0], 25)
            if early_mask.any():
                early_concs = concs[early_mask]
                max_conc = np.max(concs[concs > 0]) if (concs > 0).any() else 1.0
                if max_conc > 0 and np.all(early_concs < 0.05 * max_conc):
                    lag_count += 1

        # Multi-peak detection: count local maxima
        peaks = 0
        for i in range(1, len(concs) - 1):
            if concs[i] > concs[i - 1] and concs[i] > concs[i + 1]:
                peaks += 1
        if peaks >= 2:
            multi_peak_count += 1

    n_analyzable = max(1, len(subjects))
    if multi_peak_count / n_analyzable > _MULTI_PEAK_FRACTION_THRESHOLD:
        return "multi-phase"
    if lag_count / n_analyzable > _LAG_SIGNATURE_FRACTION_THRESHOLD:
        return "lag-signature"
    return "simple"


def _get_dose_auc_pairs(
    obs: pd.DataFrame,
    doses: pd.DataFrame,
) -> list[tuple[float, float]]:
    """Compute per-subject (dose, AUC) pairs using actual AMT from dosing records.

    For multi-dose subjects, uses total dose and AUC over the observed interval.
    Returns pairs where both dose and AUC are positive.
    """
    subjects = obs["NMID"].unique()
    pairs: list[tuple[float, float]] = []

    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj].sort_values("TIME")
        subj_doses = doses[doses["NMID"] == subj]

        if len(subj_obs) < 3 or subj_doses.empty:
            continue

        # Total dose from AMT column (actual dose, not Cmax proxy)
        total_dose = float(subj_doses["AMT"].sum())
        if total_dose <= 0:
            continue

        concs = subj_obs["DV"].values.astype(float)
        times = subj_obs["TIME"].values.astype(float)

        # For multi-dose: use AUC over the dosing interval (AUC_tau)
        # if repeated dosing is detected, otherwise use full AUC_last
        n_doses = len(subj_doses)
        if n_doses > 1:
            dose_times = subj_doses["TIME"].values.astype(float)
            # Estimate tau as median inter-dose interval
            sorted_dt = np.sort(dose_times)
            intervals = np.diff(sorted_dt)
            if len(intervals) > 0 and np.median(intervals) > 0:
                tau = float(np.median(intervals))
                # Use observations within the last dosing interval
                last_dose_time = float(sorted_dt[-1])
                tau_mask = (times >= last_dose_time) & (times <= last_dose_time + tau)
                if tau_mask.sum() >= 2:
                    auc = float(np.trapezoid(concs[tau_mask], times[tau_mask]))
                    # Normalize to single dose for comparability
                    single_dose = float(subj_doses["AMT"].iloc[-1])
                    if single_dose > 0 and auc > 0:
                        pairs.append((single_dose, auc))
                    continue

        # Single-dose or fallback: AUC_last over entire profile
        auc = float(np.trapezoid(concs, times))
        if auc > 0:
            pairs.append((total_dose, auc))

    return pairs


def _detect_nonlinear_clearance(obs: pd.DataFrame, doses: pd.DataFrame) -> bool:
    """Detect signature of nonlinear (dose-dependent) clearance.

    Two complementary heuristics:
    1. Dose-normalized AUC: if multiple dose levels exist, Spearman correlation
       between dose and AUC/dose > 0.4 indicates saturable clearance.
    2. Elimination curvature: for single dose-level studies, compares early vs.
       late post-Cmax log-concentration slopes. MM kinetics produces faster
       decline at high concentrations (early) than low (late), yielding a
       median |early/late| slope ratio > 1.8.
    """
    if obs.empty or doses.empty:
        return False

    pairs = _get_dose_auc_pairs(obs, doses)
    if len(pairs) >= 4:
        dose_vals = np.array([p[0] for p in pairs])
        auc_vals = np.array([p[1] for p in pairs])

        if len(np.unique(dose_vals)) >= 2:
            # Heuristic 1: dose-normalized AUC comparison
            dn_auc = auc_vals / dose_vals
            corr = _spearman_r(dose_vals, dn_auc)
            if corr > _NONLINEAR_CLEARANCE_DOSE_AUC_THRESHOLD:
                return True

    # Heuristic 2: elimination curvature (works with single dose level)
    return _detect_curvature_nonlinearity(obs)


def _detect_curvature_nonlinearity(obs: pd.DataFrame) -> bool:
    """Detect nonlinear clearance via elimination phase curvature.

    For Michaelis-Menten kinetics, the post-Cmax log-concentration curve is
    concave: the slope is steeper at high concentrations (early, near Vmax)
    and shallower at low concentrations (late, near linear regime).

    For TMDD kinetics, the inverse pattern occurs: target-mediated clearance
    saturates at high concentrations (slow early decline) and dominates at low
    concentrations (faster late decline), yielding ratio < 0.3.

    Computes the median ratio of |early slope| / |late slope| across subjects.
    Linear PK (including multi-compartment) typically yields 0.5 ≤ ratio ≤ 1.5;
    MM kinetics yields ratio > 1.8; TMDD yields ratio < 0.3.
    """
    subjects = obs["NMID"].unique()
    ratios: list[float] = []

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        times = subj_data["TIME"].values.astype(float)
        concs = subj_data["DV"].values.astype(float)

        if len(concs) < 5:
            continue

        tmax_idx = int(np.argmax(concs))
        post_c = concs[tmax_idx:]
        post_t = times[tmax_idx:]

        # Need enough post-Cmax points with positive concentrations
        if len(post_c) < 4 or not np.all(post_c > 0):
            continue

        log_c = np.log(post_c)
        mid = len(post_c) // 2
        if mid < 2 or len(post_c) - mid < 2:
            continue

        dt_early = post_t[mid] - post_t[0]
        dt_late = post_t[-1] - post_t[mid]
        if dt_early <= 0 or dt_late <= 0:
            continue

        early_slope = abs((log_c[mid] - log_c[0]) / dt_early)
        late_slope = abs((log_c[-1] - log_c[mid]) / dt_late)

        if late_slope > 1e-6:
            ratios.append(early_slope / late_slope)

    if len(ratios) < 4:
        return False

    median_ratio = float(np.median(ratios))
    # MM kinetics: fast early decline; TMDD: inverse (slow early, fast late).
    return (
        median_ratio > _NONLINEAR_MM_CURVATURE_RATIO
        or median_ratio < _NONLINEAR_TMDD_CURVATURE_RATIO
    )


def _nonlinear_clearance_confidence(obs: pd.DataFrame, doses: pd.DataFrame) -> float | None:
    """Confidence score for nonlinear clearance detection (0.0-1.0).

    Returns the maximum signal from dose-normalized AUC correlation (if
    multiple dose levels) and elimination curvature ratio (scaled to 0-1).
    """
    if obs.empty or doses.empty:
        return None

    scores: list[float] = []

    # Score from dose-normalized AUC
    pairs = _get_dose_auc_pairs(obs, doses)
    if len(pairs) >= 4:
        dose_vals = np.array([p[0] for p in pairs])
        auc_vals = np.array([p[1] for p in pairs])
        if len(np.unique(dose_vals)) >= 2:
            dn_auc = auc_vals / dose_vals
            corr = _spearman_r(dose_vals, dn_auc)
            scores.append(max(0.0, min(1.0, corr)))

    # Score from curvature ratio
    subjects = obs["NMID"].unique()
    ratios: list[float] = []
    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        times = subj_data["TIME"].values.astype(float)
        concs = subj_data["DV"].values.astype(float)
        if len(concs) < 5:
            continue
        tmax_idx = int(np.argmax(concs))
        post_c = concs[tmax_idx:]
        post_t = times[tmax_idx:]
        if len(post_c) < 4 or not np.all(post_c > 0):
            continue
        log_c = np.log(post_c)
        mid = len(post_c) // 2
        if mid < 2 or len(post_c) - mid < 2:
            continue
        dt_early = post_t[mid] - post_t[0]
        dt_late = post_t[-1] - post_t[mid]
        if dt_early <= 0 or dt_late <= 0:
            continue
        early_slope = abs((log_c[mid] - log_c[0]) / dt_early)
        late_slope = abs((log_c[-1] - log_c[mid]) / dt_late)
        if late_slope > 1e-6:
            ratios.append(early_slope / late_slope)
    if len(ratios) >= 4:
        # Map ratio to 0-1: ratio=1.0 → 0.0, ratio=3.0 → 1.0
        median_ratio = float(np.median(ratios))
        curvature_score = max(0.0, min(1.0, (median_ratio - 1.0) / 2.0))
        scores.append(curvature_score)

    if not scores:
        return None
    return float(max(scores))


def _classify_richness(obs: pd.DataFrame, n_subjects: int) -> str:
    """Classify sampling richness per PRD §4.2.1.

    sparse: < 4 samples/subject
    moderate: 4-8 samples/subject
    rich: > 8 samples/subject
    """
    if n_subjects == 0:
        return "sparse"

    # Use median samples per subject (robust to PK-intensive outliers)
    per_subj_counts = obs.groupby("NMID").size()
    median_samples = float(per_subj_counts.median()) if len(per_subj_counts) > 0 else 0.0

    if median_samples < 4:
        return "sparse"
    elif median_samples <= 8:
        return "moderate"
    return "rich"


def _assess_identifiability(obs: pd.DataFrame, n_subjects: int) -> str:
    """Assess identifiability ceiling based on design and sampling.

    high: rich data, many subjects, good phase coverage
    medium: moderate data or limited subjects
    low: sparse data, few subjects
    """
    if n_subjects < 5:
        return "low"

    avg_samples = len(obs) / max(1, n_subjects) if n_subjects > 0 else 0

    if avg_samples > 8 and n_subjects >= 20:
        return "high"
    elif avg_samples >= 4 and n_subjects >= 10:
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
            return float(lloq_values.mode().iloc[0])

    # Fallback: DV of censored rows (M3 convention sets DV=LLOQ)
    dv_values = blq_rows["DV"].dropna()
    if not dv_values.empty:
        return float(dv_values.mode().iloc[0])

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

    # Coefficient of variation of observations-per-subject across studies
    obs_per_subj = per_study_n_obs / per_study_n_subj
    if obs_per_subj.std() / max(obs_per_subj.mean(), 1e-6) > 0.5:
        return "pooled-heterogeneous"

    return "pooled-similar"


def _assess_absorption_coverage(obs: pd.DataFrame) -> str:
    """Assess whether absorption phase is adequately sampled.

    adequate: ≥ 2 pre-Tmax observations per subject on average
    inadequate: < 2 pre-Tmax observations per subject on average
    """
    if obs.empty:
        return "inadequate"

    subjects = obs["NMID"].unique()
    pre_tmax_counts: list[int] = []

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        if subj_data.empty:
            continue
        tmax = subj_data.loc[subj_data["DV"].idxmax(), "TIME"]
        pre_tmax = int((subj_data["TIME"] < tmax).sum())
        pre_tmax_counts.append(pre_tmax)

    if not pre_tmax_counts:
        return "inadequate"

    avg_pre_tmax = sum(pre_tmax_counts) / len(pre_tmax_counts)
    return "adequate" if avg_pre_tmax >= 2.0 else "inadequate"


def _assess_elimination_coverage(obs: pd.DataFrame) -> str:
    """Assess whether elimination phase is adequately sampled.

    adequate: ≥ 3 post-Tmax observations per subject on average
    inadequate: < 3 post-Tmax observations per subject on average
    """
    if obs.empty:
        return "inadequate"

    subjects = obs["NMID"].unique()
    post_tmax_counts: list[int] = []

    for subj in subjects:
        subj_data = obs[obs["NMID"] == subj].sort_values("TIME")
        if subj_data.empty:
            continue
        tmax = subj_data.loc[subj_data["DV"].idxmax(), "TIME"]
        post_tmax = int((subj_data["TIME"] > tmax).sum())
        post_tmax_counts.append(post_tmax)

    if not post_tmax_counts:
        return "inadequate"

    avg_post_tmax = sum(post_tmax_counts) / len(post_tmax_counts)
    return "adequate" if avg_post_tmax >= 3.0 else "inadequate"
