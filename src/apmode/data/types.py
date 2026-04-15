# SPDX-License-Identifier: GPL-2.0-or-later
"""Profiler-internal computation artifacts.

These dataclasses are NOT part of the EvidenceManifest contract — they are
passed between profiler helpers to eliminate duplicated fitting logic across
``_per_subject_terminal_monoexp_r2``, ``_compute_terminal_log_residual_mad``,
and ``_auc_extrap_fraction_median``. If a type later needs to cross the
bundle boundary, promote it to a Pydantic model in ``apmode/bundle/models.py``.

Motivation: prior review identified duplicate terminal-phase fitting paths that this module
unifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Numerical guard for log of small/zero concentrations. Stays as a code
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

    Replicates ``nlmixr2autoinit:find_best_lambdaz`` (Huang Z, Fidler M,
    Lan M, Cheng IL, Kloprogge F, Standing JF (2025) J Pharmacokinet
    Pharmacodyn 52:60. doi:10.1007/s10928-025-10000-z): enumerate all
    candidate tail windows ``n in [min_points, N]`` starting from the end
    of the profile, fit log-linear OLS, require negative slope, pick the
    window with maximum adjusted R²; tie-break (within ``tolerance``) by
    preferring the larger window.

    APMODE adds the Phoenix WinNonlin /  constraint that the
    candidate window must start no earlier than ``Tmax + (Tlast - Tmax) /
    2`` — i.e., be drawn from the second half of the post-Cmax phase.
    Without this constraint, max-adj-R² over-truncates noisy 2-cmt
    beta-phases and underestimates t½. Set ``phoenix_constraint=False`` to
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

    # Input-contract harden: caller docstring requires sorted times, but
    # silently miscomputed on unsorted input via np.searchsorted below.
    # Co-sort here to make the function robust.
    if len(times) > 1 and not np.all(np.diff(times) >= 0):
        order = np.argsort(times)
        times = times[order]
        concs = concs[order]
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
        eligible_window = n_post - earliest_eligible
        # For sparse profiles where eligible_window < min_points, fall
        # back to min_points so 1-cmt and 2-cmt linear datasets with
        # 5-6 post-Cmax samples still yield a terminal fit.
        max_window_size = max(min_points, eligible_window)
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


# ---------------------------------------------------------------------------
# Smith 2000 dose-proportionality power model + Steady-state check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseProportionalityFit:
    """Smith 2000 power-model dose-proportionality fit.

    Smith BP, Vandenhende FR, DeSante KA, et al. (2000). Confidence
    interval criteria for assessment of dose proportionality. *Pharm Res*
    17(10):1278-1283. doi:10.1023/a:1026451721686

    Compares 90% CI of beta from log(AUC)=alpha+beta·log(dose) against translated
    bounds derived from a bioequivalence-style exposure interval
    [θ_L, θ_H] (default [0.80, 1.25]) and the observed dose ratio r:
        beta_low  = 1 + ln(θ_L)/ln(r)
        beta_high = 1 + ln(θ_H)/ln(r)
    The beta acceptance region tightens as r grows (r=3 → ≈[0.81,1.20];
    r=10 → ≈[0.90,1.10]).

    ``eligible`` is True only when ≥3 distinct dose levels AND dose ratio
    ≥ 3-fold (APMODE policy). ``saturation_flag`` set when CI lower bound
    > beta_smith_high; ``induction_flag`` set when CI upper bound < beta_smith_low.
    """

    eligible: bool
    n_pairs: int
    n_dose_levels: int
    dose_ratio: float | None
    beta: float | None
    beta_se: float | None
    beta_ci90_low: float | None
    beta_ci90_high: float | None
    beta_smith_low: float | None
    beta_smith_high: float | None
    saturation_flag: bool
    induction_flag: bool
    rationale: str


def smith_2000_translated_bounds(
    theta_low: float, theta_high: float, dose_ratio: float
) -> tuple[float, float]:
    """Translate exposure interval to beta acceptance bounds (Smith 2000)."""
    if dose_ratio <= 1.0:
        raise ValueError(f"dose_ratio must be > 1; got {dose_ratio}")
    log_r = float(np.log(dose_ratio))
    return (
        1.0 + float(np.log(theta_low)) / log_r,
        1.0 + float(np.log(theta_high)) / log_r,
    )


def fit_dose_proportionality(
    pairs: list[tuple[float, float]],
    *,
    theta_low: float = 0.80,
    theta_high: float = 1.25,
    min_dose_levels: int = 3,
    min_dose_ratio: float = 3.0,
) -> DoseProportionalityFit:
    """Fit Smith 2000 power model log(AUC)=alpha+beta·log(dose); compute beta 90% CI.

    Returns a DoseProportionalityFit; ``eligible=False`` when the design
    cannot support the test (too few dose levels or insufficient range).
    """
    if not pairs:
        return DoseProportionalityFit(
            eligible=False,
            n_pairs=0,
            n_dose_levels=0,
            dose_ratio=None,
            beta=None,
            beta_se=None,
            beta_ci90_low=None,
            beta_ci90_high=None,
            beta_smith_low=None,
            beta_smith_high=None,
            saturation_flag=False,
            induction_flag=False,
            rationale="no (dose, AUC) pairs",
        )
    doses = np.array([p[0] for p in pairs], dtype=float)
    aucs = np.array([p[1] for p in pairs], dtype=float)
    n_levels = len(np.unique(doses))
    dmin, dmax = float(doses.min()), float(doses.max())
    ratio = dmax / dmin if dmin > 0 else None
    n = len(pairs)
    if n_levels < min_dose_levels or ratio is None or ratio < min_dose_ratio or n < 4:
        return DoseProportionalityFit(
            eligible=False,
            n_pairs=n,
            n_dose_levels=n_levels,
            dose_ratio=ratio,
            beta=None,
            beta_se=None,
            beta_ci90_low=None,
            beta_ci90_high=None,
            beta_smith_low=None,
            beta_smith_high=None,
            saturation_flag=False,
            induction_flag=False,
            rationale=(
                f"design ineligible: n_levels={n_levels} (need >={min_dose_levels}), "
                f"dose_ratio={ratio} (need >={min_dose_ratio}), n_pairs={n} (need >=4)"
            ),
        )
    # Review: Smith 2000 Pharm Res 17:1278-1283 derives its
    # translated-bound criterion on one summary AUC per dose level
    # (geometric mean). Fitting individual subject pairs inflates df and
    # narrows the CI by ~sqrt(n_per_level). Aggregate to geometric-mean
    # AUC per dose level before the OLS.
    unique_doses = np.unique(doses)
    if len(unique_doses) < 2:
        # Shouldn't happen — eligibility gate already requires >=3 levels.
        log_d = np.log(doses)
        log_a = safe_log(aucs)
    else:
        group_doses: list[float] = []
        group_log_aucs: list[float] = []
        for d in unique_doses:
            mask = doses == d
            if mask.sum() == 0:
                continue
            log_aucs_at_d = safe_log(aucs[mask])
            # Geometric mean of AUC at this dose level.
            group_doses.append(float(d))
            group_log_aucs.append(float(np.mean(log_aucs_at_d)))
        log_d = np.log(np.asarray(group_doses, dtype=float))
        log_a = np.asarray(group_log_aucs, dtype=float)
    n_fit = len(log_d)
    slope, intercept = np.polyfit(log_d, log_a, 1)
    pred = slope * log_d + intercept
    rss = float(np.sum((log_a - pred) ** 2))
    if n_fit <= 2:
        return DoseProportionalityFit(
            eligible=False,
            n_pairs=n,
            n_dose_levels=n_levels,
            dose_ratio=ratio,
            beta=float(slope),
            beta_se=None,
            beta_ci90_low=None,
            beta_ci90_high=None,
            beta_smith_low=None,
            beta_smith_high=None,
            saturation_flag=False,
            induction_flag=False,
            rationale=f"n_fit={n_fit}<=2; cannot compute beta CI",
        )
    sigma2 = rss / (n_fit - 2)
    sxx = float(np.sum((log_d - log_d.mean()) ** 2))
    if sxx <= 0:
        return DoseProportionalityFit(
            eligible=False,
            n_pairs=n,
            n_dose_levels=n_levels,
            dose_ratio=ratio,
            beta=float(slope),
            beta_se=None,
            beta_ci90_low=None,
            beta_ci90_high=None,
            beta_smith_low=None,
            beta_smith_high=None,
            saturation_flag=False,
            induction_flag=False,
            rationale="zero variance in log(dose); cannot compute beta CI",
        )
    se = float(np.sqrt(sigma2 / sxx))
    try:
        from scipy.stats import t as _student_t

        t_crit = float(_student_t.ppf(0.95, df=n_fit - 2))
    except ImportError:
        # Fallback table for one-sided 95% (two-sided 90% CI half-width).
        _t_table = {
            1: 6.314,
            2: 2.920,
            3: 2.353,
            4: 2.132,
            5: 2.015,
            6: 1.943,
            7: 1.895,
            8: 1.860,
            9: 1.833,
            10: 1.812,
            15: 1.753,
            20: 1.725,
            30: 1.697,
        }
        df_fit = n_fit - 2
        t_crit = _t_table.get(df_fit, 1.645 if df_fit > 30 else 2.0)
    ci_low = float(slope) - t_crit * se
    ci_high = float(slope) + t_crit * se
    smith_low, smith_high = smith_2000_translated_bounds(theta_low, theta_high, ratio)
    saturation = ci_low > smith_high
    induction = ci_high < smith_low
    return DoseProportionalityFit(
        eligible=True,
        n_pairs=n,
        n_dose_levels=n_levels,
        dose_ratio=ratio,
        beta=float(slope),
        beta_se=se,
        beta_ci90_low=ci_low,
        beta_ci90_high=ci_high,
        beta_smith_low=smith_low,
        beta_smith_high=smith_high,
        saturation_flag=saturation,
        induction_flag=induction,
        rationale=(
            f"Smith 2000: r={ratio:.2f}, n={n}, beta={slope:.3f}, "
            f"90%CI=[{ci_low:.3f},{ci_high:.3f}], "
            f"Smith=[{smith_low:.3f},{smith_high:.3f}]"
        ),
    )


def wagner_nelson_ka(
    times: np.ndarray,
    concs: np.ndarray,
    *,
    half_life: float,
) -> float | None:
    """Wagner-Nelson absorption-rate-constant estimate for one subject.

    Wagner JG, Nelson E (1963). Percent absorbed time plots derived from
    blood level and/or urinary excretion data. *J Pharm Sci* 52(6):610-611.
    doi:10.1002/jps.2600520627

    Computes per-time the fraction unabsorbed:
        F_un(t) = 1 - (C(t) + ke·AUC[0..t]) / (ke·AUC_inf)
    where ke = ln(2)/t½ and AUC_inf = AUC_last + C_last/ke.

    Linear regression of ln(F_un) ~ -ka·t yields ka. Returns None when
    the profile has fewer than 4 positive non-BLQ samples or AUC_inf is
    non-positive.

    Inputs MUST be positive, non-BLQ, sorted by time. ``half_life`` must
    come from ``fit_best_lambdaz(...).half_life`` of the same subject.
    """
    if len(times) < 4 or half_life <= 0:
        return None
    ke = float(np.log(2.0) / half_life)
    # Running-accumulator prefix-AUC: O(n), was O(n**2) prior. Each
    # segment uses linear-up/log-down (Wagner & Nelson 1963).
    auc_cum = np.zeros(len(times), dtype=float)
    for i in range(1, len(times)):
        auc_cum[i] = auc_cum[i - 1] + auc_linup_logdown(times[i - 1 : i + 1], concs[i - 1 : i + 1])
    auc_inf = float(auc_cum[-1] + concs[-1] / ke)
    if auc_inf <= 0:
        return None
    f_un = 1.0 - (concs + ke * auc_cum) / (ke * auc_inf)
    # Keep only strictly-positive, finite f_un values.
    mask = (f_un > 0) & np.isfinite(f_un)
    if mask.sum() < 3:
        return None
    log_f = safe_log(f_un[mask])
    t_fit = times[mask]
    if t_fit.max() - t_fit.min() <= 0:
        return None
    slope, _ = np.polyfit(t_fit, log_f, 1)
    if slope >= 0:
        return None
    return float(-slope)


def bootstrap_median_ci(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.90,
    n_boot: int = 1000,
    seed: int = 20260415,
) -> tuple[float, float, float] | None:
    """Population-level bootstrap CI on the MEDIAN of per-subject statistics.

    Follow-up: per-subject bootstrap on 5-10
    post-Cmax points yields singular OLS fits. Instead, compute one
    deterministic statistic per subject, then bootstrap the POPULATION
    median over subjects. Well-defined for any n >= 4.

    Returns ``(median, ci_low, ci_high)`` or ``None`` when n < 4.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return None
    rng = np.random.default_rng(seed)
    n = len(arr)
    medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        medians[i] = float(np.median(sample))
    alpha = 1.0 - confidence
    lo = float(np.quantile(medians, alpha / 2.0))
    hi = float(np.quantile(medians, 1.0 - alpha / 2.0))
    return float(np.median(arr)), lo, hi


def is_steady_state(
    dose_times: np.ndarray,
    doses: np.ndarray,
    *,
    half_life: float | None,
    n_half_lives_required: int = 3,
    n_doses_alt: int = 5,
    interval_tolerance: float = 0.25,
    dose_tolerance: float = 0.20,
    min_doses: int = 3,
) -> tuple[bool, str]:
    """Pragmatic steady-state check:
    ``(elapsed >= n_half_lives_required * t½ AND n_doses >= min_doses)
    OR n_doses >= n_doses_alt``.

    Adapted from ``nlmixr2autoinit:is_ss`` (Huang Z et al. 2025) with
    APMODE-specific aggregation. nlmixr2's "combined" semantics use the
    smaller of the two rules (more permissive). The disjunction here
    handles both sparse oncology trials with short t½ (4 doses, elapsed
    ≥ 3·t½) and dense rich trials (≥5 doses) without false positives
    from pre-SS AUCτ.

    Tolerances: interval variation ±25%, dose variation ±20% (matching
    nlmixr2autoinit defaults). Returns (is_ss, rationale).
    """
    if len(dose_times) < min_doses:
        return False, f"n_doses={len(dose_times)} < min_doses={min_doses}"
    sorted_t = np.sort(dose_times)
    intervals = np.diff(sorted_t)
    if len(intervals) == 0:
        return False, "single dose"
    pos_int = intervals[intervals > 0]
    if len(pos_int) == 0:
        return False, "no positive inter-dose intervals"
    median_int = float(np.median(pos_int))
    if median_int <= 0:
        return False, "median interval is zero"
    interval_dev = float(np.max(np.abs(intervals - median_int) / median_int))
    if interval_dev > interval_tolerance:
        return False, f"interval variation={interval_dev:.2f} > tol={interval_tolerance}"
    if len(doses) > 0:
        median_d = float(np.median(doses))
        if median_d > 0:
            dose_dev = float(np.max(np.abs(doses - median_d) / median_d))
            if dose_dev > dose_tolerance:
                return False, f"dose variation={dose_dev:.2f} > tol={dose_tolerance}"
    elapsed = float(sorted_t[-1] - sorted_t[0])
    n_doses_n = len(sorted_t)
    if half_life is not None and half_life > 0:
        ss_by_t12 = elapsed >= n_half_lives_required * half_life and n_doses_n >= min_doses
    else:
        ss_by_t12 = False
    # Scientific-methodology review: n-doses-alone cannot justify
    # SS without pharmacokinetic evidence. A q24h regimen with t1/2=72h
    # reaches ~3.3% SS after 5 doses. Require half_life whenever the
    # count-only branch would be the deciding factor.
    if half_life is None or half_life <= 0:
        return (
            False,
            (
                f"n_doses={n_doses_n}, elapsed={elapsed:.2f}, t1/2=None: "
                "SS not declared without half-life evidence"
            ),
        )
    ss_by_count = n_doses_n >= n_doses_alt
    return (
        ss_by_t12 or ss_by_count,
        (
            f"n_doses={n_doses_n}, elapsed={elapsed:.2f}, "
            f"t1/2={half_life:.3f}, ss_by_t12={ss_by_t12}, ss_by_count={ss_by_count}"
        ),
    )
