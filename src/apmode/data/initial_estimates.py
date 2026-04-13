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
import pandas as pd  # type: ignore[import-untyped]  # noqa: TC002 — runtime use

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
        """
        subjects = self._obs["NMID"].unique()
        if len(subjects) < 2:
            return self.estimate_population_level()

        cl_vals: list[float] = []
        v_vals: list[float] = []
        ka_vals: list[float] = []
        kel_vals: list[float] = []

        for subj in subjects:
            subj_obs = self._obs[self._obs["NMID"] == subj].sort_values("TIME")
            subj_dose = self._doses[self._doses["NMID"] == subj]

            if len(subj_obs) < 3:
                continue

            dose = float(subj_dose["AMT"].sum()) if not subj_dose.empty else 0.0
            if dose <= 0:
                continue

            times = subj_obs["TIME"].values.astype(float)
            concs = subj_obs["DV"].values.astype(float)

            # Filter to positive concentrations
            pos_mask = concs > 0
            if pos_mask.sum() < 3:
                continue

            times_pos = times[pos_mask]
            concs_pos = concs[pos_mask]

            nca = _compute_nca_single_subject(times_pos, concs_pos, dose)
            if nca is not None:
                cl, v, ka, kel = nca
                cl_vals.append(cl)
                v_vals.append(v)
                ka_vals.append(ka)
                kel_vals.append(kel)

        # Need at least 2 successful subjects for median
        if len(cl_vals) < 2:
            return self.estimate_population_level()

        return {
            "CL": float(np.median(cl_vals)),
            "V": float(np.median(v_vals)),
            "ka": float(np.median(ka_vals)),
        }

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

        cl, v, ka, _kel = nca
        return {"CL": cl, "V": v, "ka": ka}

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


def _compute_nca_single_subject(
    times: np.ndarray,
    concs: np.ndarray,
    dose: float,
) -> tuple[float, float, float, float] | None:
    """Compute NCA for one subject.

    Returns (CL, V, ka, kel) or None if computation fails.
    Requires times and concs to be sorted, positive, and len >= 3.
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

    # AUC extrapolation to infinity
    c_last = float(concs[-1])
    auc_extrap = c_last / kel if c_last > 0 else 0.0
    auc_inf = auc_last + auc_extrap

    if auc_inf <= 0:
        return None

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

    return (cl, v, ka, kel)


def _default_estimates() -> dict[str, float]:
    """Fallback estimates when NCA fails."""
    return {"CL": 5.0, "V": 70.0, "ka": 1.0}
