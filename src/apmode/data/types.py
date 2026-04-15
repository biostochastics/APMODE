# SPDX-License-Identifier: GPL-2.0-or-later
"""Profiler-internal computation artifacts.

These dataclasses are NOT part of the EvidenceManifest contract — they are
passed between profiler helpers to eliminate duplicated fitting logic across
``_per_subject_terminal_monoexp_r2``, ``_compute_terminal_log_residual_mad``,
and ``_auc_extrap_fraction_median``. If a type later needs to cross the
bundle boundary, promote it to a Pydantic model in ``apmode/bundle/models.py``.

Refactor motivation: multi-model consensus 2026-04-15 (glm-5.1, gpt-5.2-pro)
identified three divergent terminal-phase fitting paths that this module
unifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Numerical guard for log() of small/zero concentrations. Stays as a code
# constant (not policy) because it's a numerical-stability floor, not a
# scientific threshold.
_LOG_FLOOR: float = 1e-100


@dataclass(frozen=True)
class TerminalPhase:
    """Best-fit log-linear terminal phase for one subject.

    All fields are computed on the BLQ-filtered, multi-dose-windowed
    profile. ``slope < 0`` for valid elimination phases. ``adj_r2`` is the
    adjusted R² of the chosen log-linear OLS fit.
    """

    times: np.ndarray  # shape (n_used,)
    concs: np.ndarray  # shape (n_used,) — positive, non-BLQ
    slope: float  # log-conc OLS slope; negative for elimination
    intercept: float  # log-conc OLS intercept
    adj_r2: float  # adjusted R²
    n_used: int  # number of points in the chosen tail window
    n_post_cmax: int  # total positive non-BLQ points after Cmax
    method: str  # one of {"terminal_30pct", "best_lambdaz_huang2025"}

    @property
    def lambda_z(self) -> float | None:
        """Terminal elimination rate constant (positive). None if invalid."""
        if self.slope >= 0 or not np.isfinite(self.slope):
            return None
        return -float(self.slope)

    @property
    def half_life(self) -> float | None:
        """Terminal half-life. None if lambda_z invalid."""
        lz = self.lambda_z
        if lz is None or lz <= 0:
            return None
        return float(np.log(2.0) / lz)


def positive_unblqd_mask(obs: pd.DataFrame) -> np.ndarray:
    """Boolean mask selecting positive, non-BLQ observations.

    Beal 2001 / Ahn 2008: when BLQ_FLAG column exists and rows have
    DV=LLOQ (M3 convention), those censored values must NOT contribute to
    shape geometry (Tmax, slopes, R²). Without this mask ``argmax(DV)``
    can land on a censored value and Tmax-relative metrics silently break.
    """
    dv = obs["DV"].to_numpy(dtype=float)
    mask: np.ndarray = dv > 0
    if "BLQ_FLAG" in obs.columns:
        blq: np.ndarray = obs["BLQ_FLAG"].to_numpy(dtype=float)
        mask = mask & (blq != 1)
    return mask


def safe_log(c: np.ndarray) -> np.ndarray:
    """np.log with underflow clipping to avoid -inf in log-linear fits."""
    result: np.ndarray = np.log(np.clip(c, _LOG_FLOOR, None))
    return result


def _ols_log_linear(t: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, adj_r2) of OLS fit log(c) ~ slope*t + intercept.

    Uses ``np.polyfit`` for the fit. Returns (nan, nan, -inf) if degenerate.
    Adjusted R² formula: 1 - (1 - R²)·(n - 1)/(n - 2); for n == 2 falls back
    to plain R².
    """
    n = len(t)
    if n < 2 or t.max() - t.min() <= 0:
        return float("nan"), float("nan"), float("-inf")
    log_c = safe_log(c)
    slope, intercept = np.polyfit(t, log_c, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((log_c - pred) ** 2))
    ss_tot = float(np.sum((log_c - log_c.mean()) ** 2))
    if ss_tot <= 0:
        return float(slope), float(intercept), float("-inf")
    r2 = 1.0 - ss_res / ss_tot
    if n <= 2:
        return float(slope), float(intercept), float(r2)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
    return float(slope), float(intercept), float(adj_r2)


def fit_best_lambdaz(
    times: np.ndarray,
    concs: np.ndarray,
    *,
    min_points: int = 3,
    tolerance: float = 1e-4,
    phoenix_constraint: bool = True,
) -> TerminalPhase | None:
    """Find the best terminal log-linear fit, Huang 2025 algorithm.

    Replicates ``nlmixr2autoinit::find_best_lambdaz`` (Huang Z, Fidler M,
    Lan M, Cheng IL, Kloprogge F, Standing JF (2025) J Pharmacokinet
    Pharmacodyn 52:60. doi:10.1007/s10928-025-10000-z): enumerate all
    candidate tail windows ``n in [min_points, N]`` starting from the end
    of the profile, fit log-linear OLS, require negative slope, pick the
    window with maximum adjusted R²; tie-break (within ``tolerance``) by
    preferring the larger window.

    APMODE adds the Phoenix WinNonlin / gemini-3 constraint that the
    candidate window must start no earlier than ``Tmax + (Tlast - Tmax) /
    2`` — i.e., be drawn from the second half of the post-Cmax phase.
    Without this constraint, max-adj-R² over-truncates noisy 2-cmt
    β-phases and underestimates t½. Set ``phoenix_constraint=False`` to
    disable for unit tests of the raw Huang 2025 algorithm.

    ``times`` and ``concs`` MUST already be filtered to positive, non-BLQ
    observations and sorted by time. Returns None when the profile cannot
    yield any valid fit.
    """
    n_total = len(concs)
    if n_total < min_points:
        return None
    tmax_idx = int(np.argmax(concs))
    post_t = times[tmax_idx:]
    post_c = concs[tmax_idx:]
    n_post = len(post_t)
    if n_post < min_points:
        return None

    # Phoenix constraint: window cannot extend earlier than the midpoint
    # of the post-Cmax window in TIME (not in samples).
    if phoenix_constraint and n_post >= 4:
        t_anchor_min = post_t[0] + 0.5 * (post_t[-1] - post_t[0])
        earliest_eligible = int(np.searchsorted(post_t, t_anchor_min, side="left"))
        # Always allow at least the last min_points.
        max_window_size = n_post - earliest_eligible
        max_window_size = max(min_points, max_window_size)
    else:
        max_window_size = n_post

    best: TerminalPhase | None = None
    for n in range(min_points, max_window_size + 1):
        t_w = post_t[-n:]
        c_w = post_c[-n:]
        slope, intercept, adj_r2 = _ols_log_linear(t_w, c_w)
        if not np.isfinite(slope) or slope >= 0 or adj_r2 == float("-inf"):
            continue
        if best is None:
            best = TerminalPhase(
                times=t_w,
                concs=c_w,
                slope=slope,
                intercept=intercept,
                adj_r2=adj_r2,
                n_used=n,
                n_post_cmax=n_post,
                method="best_lambdaz_huang2025",
            )
            continue
        # Tie-break: prefer larger n if within tolerance, else higher adj_r2.
        if adj_r2 > best.adj_r2 + tolerance or (
            abs(adj_r2 - best.adj_r2) <= tolerance and n > best.n_used
        ):
            best = TerminalPhase(
                times=t_w,
                concs=c_w,
                slope=slope,
                intercept=intercept,
                adj_r2=adj_r2,
                n_used=n,
                n_post_cmax=n_post,
                method="best_lambdaz_huang2025",
            )
    return best


def fit_fixed_tail_terminal(
    times: np.ndarray,
    concs: np.ndarray,
    *,
    tail_fraction: float = 0.30,
    min_points: int = 3,
) -> TerminalPhase | None:
    """Legacy fixed-tail-fraction terminal fit (used for back-compat in
    ``terminal_fit_adj_r2_median`` reporting that documents tail size).

    Inputs MUST already be positive, non-BLQ, time-sorted. Tail = final
    ``tail_fraction`` of post-Cmax points, with at least ``min_points``.
    """
    if len(concs) < min_points:
        return None
    tmax_idx = int(np.argmax(concs))
    post_t = times[tmax_idx:]
    post_c = concs[tmax_idx:]
    n_post = len(post_c)
    if n_post < min_points:
        return None
    n_tail = max(min_points, round(n_post * tail_fraction))
    n_tail = min(n_tail, n_post)
    t_w = post_t[-n_tail:]
    c_w = post_c[-n_tail:]
    slope, intercept, adj_r2 = _ols_log_linear(t_w, c_w)
    if not np.isfinite(slope) or slope >= 0:
        return None
    return TerminalPhase(
        times=t_w,
        concs=c_w,
        slope=slope,
        intercept=intercept,
        adj_r2=adj_r2,
        n_used=n_tail,
        n_post_cmax=n_post,
        method="terminal_30pct",
    )


def auc_linup_logdown(times: np.ndarray, concs: np.ndarray) -> float:
    """Linear-up / log-down trapezoid (Wagner & Nelson 1963 convention).

    Uses linear trapezoid on ascending segments or where either point is
    non-positive; uses log trapezoid ``Δt · (C₁ - C₂) / ln(C₁ / C₂)`` on
    strictly-decreasing positive segments. Robust to sparse late-phase
    sampling where pure linear trapezoid overestimates AUC.

    Inputs MUST be sorted by time. Caller is responsible for filtering
    BLQ rows out of ``concs``.
    """
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        c1 = float(concs[i - 1])
        c2 = float(concs[i])
        if dt <= 0:
            continue
        # Log trapezoid only on strictly-decreasing positive segments.
        if c1 > 0 and c2 > 0 and c2 < c1:
            auc += dt * (c1 - c2) / np.log(c1 / c2)
        else:
            auc += 0.5 * dt * (c1 + c2)
    return float(auc)
