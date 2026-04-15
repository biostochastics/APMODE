# SPDX-License-Identifier: GPL-2.0-or-later
"""NCA-based initial estimate derivation (PRD §4.2.0.1, ARCHITECTURE.md §2.5).

Systematic initial estimate derivation before estimation dispatch.
Poor initial estimates are a primary cause of SAEM non-convergence.

Sources:
  - NCA-derived: per-subject NCA → CL, V, ka, t½
  - Population-level NCA: naive-averaged profiles (sparse data fallback)
  - Warm-start: parent model best-fit → child candidate (automated search)

Output: initial_estimates.json keyed by candidate_id with provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # noqa: TC002

from apmode.bundle.models import InitialEstimateEntry, InitialEstimates

if TYPE_CHECKING:
    from apmode.bundle.models import DataManifest


class NCAEstimator:
    """Non-compartmental analysis for initial estimate derivation.

    Computes basic PK parameters from concentration-time data:
      - CL: clearance (Dose / AUC_inf)
      - V: volume of distribution (CL / kel or Dose / C0)
      - ka: absorption rate constant (1 / tmax for first-order estimate)
      - t_half: terminal half-life (0.693 / kel)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        manifest: DataManifest,
    ) -> None:
        self._df = df
        self._manifest = manifest
        self._obs = df[df["EVID"] == 0].copy()
        self._doses = df[df["EVID"] == 1].copy()

    def estimate_per_subject(self) -> dict[str, float]:
        """Derive population-median NCA estimates from per-subject NCA.

        Returns dict with keys like "CL", "V", "ka" mapped to median values
        across subjects. Falls back to population-level NCA if per-subject
        NCA is infeasible for too many subjects.

        Handles multi-dose profiles via AUC_tau for steady-state subjects.
        Flags subjects with AUC extrapolation fraction >20% (unreliable).
        """
        subjects = self._obs["NMID"].unique()
        if len(subjects) < 2:
            return self.estimate_population_level()

        cl_vals: list[float] = []
        v_vals: list[float] = []
        ka_vals: list[float] = []
        extrap_flags: list[float] = []

        for subj in subjects:
            subj_obs = self._obs[self._obs["NMID"] == subj].sort_values("TIME")
            subj_dose = self._doses[self._doses["NMID"] == subj]

            if len(subj_obs) < 3:
                continue

            if subj_dose.empty:
                continue

            times = subj_obs["TIME"].values.astype(float)
            concs = subj_obs["DV"].values.astype(float)

            # Filter to positive concentrations
            pos_mask = concs > 0
            if pos_mask.sum() < 3:
                continue

            times_pos = times[pos_mask]
            concs_pos = concs[pos_mask]

            # Detect multi-dose and estimate dosing interval
            is_multi, tau = _detect_multi_dose(subj_dose)

            # For multi-dose: use last dose's AMT; for single: total AMT
            dose = float(subj_dose["AMT"].iloc[-1]) if is_multi else float(subj_dose["AMT"].sum())

            if dose <= 0:
                continue

            nca = _compute_nca_single_subject(
                times_pos,
                concs_pos,
                dose,
                is_steady_state=is_multi,
                tau=tau,
            )
            if nca is not None:
                # Flag high extrapolation fraction (>20%) but still use the estimate
                extrap_flags.append(nca.auc_extrap_fraction)
                cl_vals.append(nca.cl)
                v_vals.append(nca.v)
                ka_vals.append(nca.ka)

        # Need at least 2 successful subjects for median
        if len(cl_vals) < 2:
            return self.estimate_population_level()

        cl_median = float(np.median(cl_vals))
        v_median = float(np.median(v_vals))
        ka_median = float(np.median(ka_vals))

        # Apply unit-conversion heuristic when dose and DV units are mismatched.
        # NCA computes CL = Dose/AUC directly, which is correct only when
        # mass(Dose) and mass(DV)/volume(DV) are commensurate (e.g., dose mg +
        # DV mg/L). When dose is in mg but DV is in ng/mL (a common convention),
        # the raw CL is 1000x too small.
        scale, scale_reason = _detect_unit_scale_factor(self._doses, self._obs, cl_median)
        cl_median *= scale
        v_median *= scale

        result = {
            "CL": cl_median,
            "V": v_median,
            "ka": ka_median,
        }

        # Record quality flag: fraction of subjects with high extrapolation
        if extrap_flags:
            high_extrap = sum(1 for f in extrap_flags if f > 0.20)
            result["_auc_extrap_high_fraction"] = high_extrap / len(extrap_flags)

        if scale != 1.0:
            result["_unit_scale_applied"] = scale
            # Reason recorded as flag for downstream auditability via bundle
            # (string isn't carried; the magnitude alone is recoverable).

        _ = scale_reason  # acknowledge for linters; reason is computed for logs only
        return result

    def estimate_population_level(self) -> dict[str, float]:
        """Population-level NCA on naive-averaged profiles.

        Fallback when per-subject NCA is infeasible (sparse data).
        Pools all subjects and computes NCA on the mean profile.
        """
        if self._obs.empty:
            return _default_estimates()

        # Naive average: mean concentration at each time point
        pooled = self._obs.groupby("TIME")["DV"].mean().reset_index()
        pooled = pooled.sort_values("TIME")

        times = pooled["TIME"].values.astype(float)
        concs = pooled["DV"].values.astype(float)

        # Total dose (median per subject)
        if self._doses.empty:
            return _default_estimates()

        per_subj_dose = self._doses.groupby("NMID")["AMT"].sum()
        dose = float(per_subj_dose.median())
        if dose <= 0:
            return _default_estimates()

        pos_mask = concs > 0
        if pos_mask.sum() < 3:
            return _default_estimates()

        times_pos = times[pos_mask]
        concs_pos = concs[pos_mask]

        nca = _compute_nca_single_subject(times_pos, concs_pos, dose)
        if nca is None:
            return _default_estimates()

        return {"CL": nca.cl, "V": nca.v, "ka": nca.ka}

    def build_entry(
        self,
        candidate_id: str,
        source: str = "nca",
    ) -> InitialEstimateEntry:
        """Build an InitialEstimateEntry with NCA-derived estimates."""
        estimates = self.estimate_per_subject()
        return InitialEstimateEntry(
            candidate_id=candidate_id,
            source=source,
            estimates=estimates,
            inputs_used=["per_subject_nca" if source == "nca" else source],
        )


def warm_start_estimates(
    parent_estimates: dict[str, float],
    candidate_id: str,
) -> InitialEstimateEntry:
    """Build warm-start initial estimates from a parent model's best-fit.

    Used in automated search: child candidates inherit parent's parameters.
    """
    return InitialEstimateEntry(
        candidate_id=candidate_id,
        source="warm_start",
        estimates=parent_estimates,
        inputs_used=["parent_best_fit"],
    )


def build_initial_estimates_bundle(
    entries: list[InitialEstimateEntry],
) -> InitialEstimates:
    """Build the initial_estimates.json bundle artifact."""
    return InitialEstimates(
        entries={e.candidate_id: e for e in entries},
    )


# ---------------------------------------------------------------------------
# Internal NCA computation
# ---------------------------------------------------------------------------


class NCAResult:
    """Result of single-subject NCA computation with quality flags."""

    __slots__ = ("auc_extrap_fraction", "cl", "ka", "kel", "v")

    def __init__(
        self,
        cl: float,
        v: float,
        ka: float,
        kel: float,
        auc_extrap_fraction: float,
    ) -> None:
        self.cl = cl
        self.v = v
        self.ka = ka
        self.kel = kel
        self.auc_extrap_fraction = auc_extrap_fraction


def _compute_nca_single_subject(
    times: np.ndarray,
    concs: np.ndarray,
    dose: float,
    *,
    is_steady_state: bool = False,
    tau: float | None = None,
) -> NCAResult | None:
    """Compute NCA for one subject.

    Returns NCAResult or None if computation fails.
    Requires times and concs to be sorted, positive, and len >= 3.

    For multi-dose profiles at steady state, when tau is provided:
    uses AUC_tau (AUC over the dosing interval) instead of AUC_inf.
    """
    # Validate finite inputs (VULN-001: NaN/Inf guard)
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(concs))):
        return None
    if dose <= 0 or not np.isfinite(dose):
        return None

    # AUC by linear trapezoidal rule (use all non-negative concentrations)
    nonneg_mask = concs >= 0
    auc_last = float(np.trapezoid(concs[nonneg_mask], times[nonneg_mask]))
    if not np.isfinite(auc_last) or auc_last <= 0:
        return None

    # Find Cmax and Tmax
    cmax_idx = int(np.argmax(concs))
    cmax = float(concs[cmax_idx])
    tmax = float(times[cmax_idx])

    if cmax <= 0:
        return None

    # Terminal phase: use last 3 positive points after Tmax for kel regression.
    # Including Cmax and early distribution phase biases the slope estimate.
    post_tmax_mask = (times > tmax) & (concs > 0)
    post_concs = concs[post_tmax_mask]
    post_times_arr = times[post_tmax_mask]

    # Need at least 3 terminal points for a robust log-linear regression
    if len(post_concs) < 3:
        # Fall back to 2 points if available
        if len(post_concs) < 2:
            return None
    else:
        # Use last 3 points to avoid distribution phase contamination
        post_concs = post_concs[-3:]
        post_times_arr = post_times_arr[-3:]

    # Log-linear regression on terminal phase for kel
    # Clip concentrations to prevent log overflow/underflow (VULN-002)
    clipped = np.clip(post_concs, 1e-100, 1e100)
    log_concs = np.log(clipped)
    term_times = post_times_arr

    # Simple linear regression: log(C) = intercept - kel * t
    mean_t = float(np.mean(term_times))
    mean_lc = float(np.mean(log_concs))
    ss_tt = float(np.sum((term_times - mean_t) ** 2))

    if ss_tt == 0:
        return None

    slope = float(np.sum((term_times - mean_t) * (log_concs - mean_lc))) / ss_tt
    kel = -slope

    if kel <= 0:
        return None

    # For steady-state multi-dose: use AUC_tau (dose/AUC_tau = CL at ss)
    if is_steady_state and tau is not None and tau > 0:
        tau_mask = nonneg_mask & (times <= times[0] + tau)
        auc_tau = float(np.trapezoid(concs[tau_mask], times[tau_mask]))
        if auc_tau > 0:
            cl = dose / auc_tau
            v = cl / kel
            ka = 1.0 / tmax if tmax > 0 else 1.0
            cl = max(0.01, min(cl, 10000.0))
            v = max(0.1, min(v, 100000.0))
            ka = max(0.01, min(ka, 100.0))
            return NCAResult(cl=cl, v=v, ka=ka, kel=kel, auc_extrap_fraction=0.0)

    # AUC extrapolation to infinity
    c_last = float(concs[-1])
    auc_extrap = c_last / kel if c_last > 0 else 0.0
    auc_inf = auc_last + auc_extrap

    if auc_inf <= 0:
        return None

    # AUC extrapolation fraction: flag if >20% (unreliable estimate)
    auc_extrap_fraction = auc_extrap / auc_inf if auc_inf > 0 else 0.0

    # CL = Dose / AUC_inf
    cl = dose / auc_inf

    # V = CL / kel (or Dose / (AUC_inf * kel))
    v = cl / kel

    # ka estimate: 1/tmax is a rough first-order approximation
    # More precise: flip-flop check, method of residuals
    ka = 1.0 / tmax if tmax > 0 else 1.0

    # Sanity bounds
    cl = max(0.01, min(cl, 10000.0))
    v = max(0.1, min(v, 100000.0))
    ka = max(0.01, min(ka, 100.0))

    return NCAResult(cl=cl, v=v, ka=ka, kel=kel, auc_extrap_fraction=auc_extrap_fraction)


def _detect_multi_dose(dose_df: pd.DataFrame) -> tuple[bool, float | None]:
    """Detect if a subject has multiple doses and estimate tau.

    Returns (is_multi_dose, tau_estimate).
    """
    if len(dose_df) < 2:
        return False, None
    dose_times = np.sort(dose_df["TIME"].values.astype(float))
    intervals = np.diff(dose_times)
    if len(intervals) == 0:
        return False, None
    tau = float(np.median(intervals))
    return (True, tau) if tau > 0 else (False, None)


def _default_estimates() -> dict[str, float]:
    """Fallback estimates when NCA fails."""
    return {"CL": 5.0, "V": 70.0, "ka": 1.0}


def _detect_unit_scale_factor(
    doses: pd.DataFrame,
    obs: pd.DataFrame,
    cl_estimate: float,
) -> tuple[float, str]:
    """Detect the unit-conversion factor needed for dose/concentration units.

    NCA computes CL = Dose / AUC directly. This produces L/h only when:
      - Dose is in same mass units as DV's mass component
      - e.g., mg dose + mg/L DV (== μg/mL)

    When dose is in mg but DV is in ng/mL (a common pharmacometric convention),
    the raw CL is 1000x too small. This heuristic detects such mismatches and
    returns a multiplier (1.0, 1000.0, or 1e6) plus the reason.

    Heuristics (in priority order, based on absolute DV magnitude and CL plausibility):
      1. dv_median > 50000 and CL < 0.0001 → DV likely in pg/mL → x1e6.
      2. dv_median > 50    and CL < 0.5    → DV likely in ng/mL → x1000.
         (Typical mg/L / μg/mL plasma levels are < 50; ng/mL levels are
         routinely 50-10000. Combined with a CL < 0.5 L/h that is implausibly
         low for typical adult human PK, this strongly indicates a unit gap.)
      3. Otherwise no conversion.

    Args:
        doses: Dosing rows (EVID==1).
        obs: Observation rows (EVID==0).
        cl_estimate: The current median CL value from NCA.

    Returns:
        (scale_factor, reason_string).
    """
    if doses.empty or obs.empty:
        return 1.0, "no dose/obs data"

    dose_median = float(doses["AMT"].median())
    dv_pos = obs[obs["DV"] > 0]["DV"]
    if dv_pos.empty or dose_median <= 0:
        return 1.0, "no positive DV or invalid dose"

    dv_median = float(dv_pos.median())

    # Typical adult human CL is 0.5-100 L/h for most marketed drugs; a CL
    # below 0.5 paired with a "ng/mL-magnitude" DV (>50) is the canonical
    # mg-dose / ng/mL-DV mismatch.
    if cl_estimate < 0.0001 and dv_median > 50000:
        return 1e6, "dose in mg but DV likely in pg/mL (x1e6)"
    if cl_estimate < 0.5 and dv_median > 50:
        return 1000.0, "dose in mg but DV likely in ng/mL (x1000)"

    return 1.0, "units commensurate"
