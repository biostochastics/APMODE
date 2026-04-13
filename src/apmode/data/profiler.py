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

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # noqa: TC002 — runtime use

from apmode.bundle.models import CovariateSpec, EvidenceManifest

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
    obs = df[df["EVID"] == 0].copy()
    doses = df[df["EVID"] == 1].copy()
    n_subjects = int(df["NMID"].nunique())

    return EvidenceManifest(
        data_sha256=manifest.data_sha256,
        route_certainty=_assess_route_certainty(doses),
        absorption_complexity=_assess_absorption_complexity(obs),
        nonlinear_clearance_signature=_detect_nonlinear_clearance(obs),
        nonlinear_clearance_confidence=_nonlinear_clearance_confidence(obs),
        richness_category=_classify_richness(obs, n_subjects),
        identifiability_ceiling=_assess_identifiability(obs, n_subjects),
        covariate_burden=len(manifest.covariates),
        covariate_correlated=_check_covariate_correlation(df, manifest),
        covariate_missingness=_assess_covariate_missingness(df, manifest),
        blq_burden=_compute_blq_burden(df),
        protocol_heterogeneity=_assess_protocol_heterogeneity(df),
        absorption_phase_coverage=_assess_absorption_coverage(obs),
        elimination_phase_coverage=_assess_elimination_coverage(obs),
    )


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
        early_mask = times <= np.percentile(times[times > 0], 25) if (times > 0).any() else []
        if isinstance(early_mask, np.ndarray) and early_mask.any():
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
    if multi_peak_count / n_analyzable > 0.3:
        return "multi-phase"
    if lag_count / n_analyzable > 0.5:
        return "lag-signature"
    return "simple"


def _detect_nonlinear_clearance(obs: pd.DataFrame) -> bool:
    """Detect signature of nonlinear (dose-dependent) clearance.

    Uses dose-normalized AUC comparison: if higher doses show
    disproportionately higher exposure, clearance is saturating.
    """
    if obs.empty:
        return False

    # Need observations linked to dose amounts
    # Simple heuristic: check if AUC/dose increases with dose
    subjects = obs["NMID"].unique()
    if len(subjects) < 4:
        return False

    dose_auc_pairs: list[tuple[float, float]] = []
    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj].sort_values("TIME")
        concs = subj_obs["DV"].values
        times = subj_obs["TIME"].values
        if len(concs) < 3:
            continue
        # Trapezoidal AUC
        auc = float(np.trapezoid(concs, times))
        # Try to find dose for this subject from full dataset context
        # (approximation: use max DV as proxy for dose level)
        cmax = float(np.max(concs)) if len(concs) > 0 else 0.0
        if cmax > 0 and auc > 0:
            dose_auc_pairs.append((cmax, auc))

    if len(dose_auc_pairs) < 4:
        return False

    # Check if AUC/Cmax ratio increases with Cmax (nonlinear signature)
    cmax_vals = np.array([p[0] for p in dose_auc_pairs])
    auc_vals = np.array([p[1] for p in dose_auc_pairs])
    ratios = auc_vals / cmax_vals

    # Spearman rank correlation between Cmax and AUC/Cmax ratio
    corr = _spearman_r(cmax_vals, ratios)
    return bool(corr > 0.4)


def _nonlinear_clearance_confidence(obs: pd.DataFrame) -> float | None:
    """Confidence score for nonlinear clearance detection."""
    if obs.empty:
        return None

    subjects = obs["NMID"].unique()
    if len(subjects) < 4:
        return None

    dose_auc_pairs: list[tuple[float, float]] = []
    for subj in subjects:
        subj_obs = obs[obs["NMID"] == subj].sort_values("TIME")
        concs = subj_obs["DV"].values
        times = subj_obs["TIME"].values
        if len(concs) < 3:
            continue
        auc = float(np.trapezoid(concs, times))
        cmax = float(np.max(concs)) if len(concs) > 0 else 0.0
        if cmax > 0 and auc > 0:
            dose_auc_pairs.append((cmax, auc))

    if len(dose_auc_pairs) < 4:
        return None

    cmax_vals = np.array([p[0] for p in dose_auc_pairs])
    auc_vals = np.array([p[1] for p in dose_auc_pairs])
    ratios = auc_vals / cmax_vals

    corr = _spearman_r(cmax_vals, ratios)
    return float(max(0.0, min(1.0, corr)))


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
    # Check if any off-diagonal element > 0.7
    np.fill_diagonal(corr_matrix.values, 0)
    return bool((corr_matrix > 0.7).any().any())


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

    strategy = "impute-median" if frac <= 0.15 else "full-information"

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
