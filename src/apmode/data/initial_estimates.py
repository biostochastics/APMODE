# SPDX-License-Identifier: GPL-2.0-or-later
"""NCA-based initial estimate derivation (PRD §4.2.0.1, ARCHITECTURE.md §2.5).

Systematic initial estimate derivation before estimation dispatch.
Poor initial estimates are a primary cause of SAEM non-convergence.

Algorithm references:
  - PKNCA ``pk.calc.half.life`` curve-stripping for terminal lambda_z selection:
    https://humanpred.github.io/pknca/articles/v06-half-life-calculation.html
  - PKNCA ``pk.calc.auc`` linear-up/log-down integration (Purves 1992):
    https://humanpred.github.io/pknca/articles/v23-auc-integration-methods.html
  - Gabrielsson & Weiner, PKPD Data Analysis (5th ed., 2017) §2.8.4.
  - Wang et al. (2025), J Pharmacokinet Pharmacodyn, "automated pipeline to
    generate initial estimates for population PK base models."

Sources of initial estimates (in priority order):
  1. Per-subject NCA with QC gates → median across surviving subjects.
  2. Population-level NCA on naive-averaged profile (sparse data fallback).
  3. Published-model key_estimates (via ``fallback_estimates``) when a
     DatasetCard provides them AND per-subject NCA fails QC.
  4. Conservative defaults (CL=5 L/h, V=70 L, ka=1/h).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd  # noqa: TC002

from apmode.bundle.models import (
    InitialEstimateEntry,
    InitialEstimates,
    NCASubjectDiagnostic,
)

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from apmode.bundle.models import DataManifest


# ---------------------------------------------------------------------------
# QC gate thresholds (PKNCA/FDA bioequivalence conventions)
# ---------------------------------------------------------------------------

# Minimum adjusted R² for terminal-phase log-linear regression. PKNCA uses the
# curve-stripping algorithm; values below ~0.80 indicate either distribution
# phase contamination or noise-dominated terminal data.
_NCA_MIN_ADJ_R2: float = 0.80

# FDA bioequivalence guidance and PKNCA recommend AUC_extrap ≤ 20% of AUC_inf;
# beyond this, the terminal extrapolation dominates and CL estimates are
# unreliable.
_NCA_MAX_EXTRAP_FRACTION: float = 0.20

# PKNCA default: span of terminal-phase data must cover at least one estimated
# half-life (span_ratio = Δt / t½ ≥ 1).
_NCA_MIN_SPAN_RATIO: float = 1.0

# PKNCA default: at least 3 points required for a meaningful log-linear
# regression with adj_r2 tiebreak.
_NCA_MIN_LAMBDA_POINTS: int = 3

# Tiebreak tolerance on adj_r2: fits within this window of the best fit are
# considered equivalent and the tiebreak (most points) decides. Matches PKNCA
# ``adj.r.squared.factor`` default.
_NCA_ADJ_R2_FACTOR: float = 1e-4

# If ≥50% of subjects fail QC, abandon per-subject NCA and fall back to
# literature priors or defaults (Wang 2025 convention).
_NCA_FALLBACK_EXCLUSION_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class LambdaZFit:
    """Terminal-phase log-linear regression result (PKNCA-style)."""

    kel: float
    r2: float
    adj_r2: float
    n_points: int
    span_ratio: float
    t_first: float
    t_last: float


@dataclass
class NCAResult:
    """Result of single-subject NCA computation with QC flags."""

    cl: float
    v: float
    ka: float
    kel: float
    tmax: float
    cmax: float
    auc_last: float
    auc_inf: float
    auc_extrap_fraction: float
    lambda_z_adj_r2: float
    lambda_z_n_points: int
    span_ratio: float
    excluded: bool = False
    excluded_reason: str | None = None


class NCAEstimator:
    """Non-compartmental analysis for initial estimate derivation.

    Computes basic PK parameters from concentration-time data using
    PKNCA-compatible algorithms:
      - CL: Dose / AUC_inf (or Dose / AUC_tau at steady state)
      - V: CL / kel
      - ka: 1 / Tmax (first-order approximation)
      - kel: terminal lambda_z via adaptive curve-stripping

    Per-subject results are filtered by QC gates (adj_r²≥0.80, extrapolation
    ≤20%, span ≥1 half-life, ≥3 terminal points). Subjects failing QC are
    excluded from the population median. When ≥50% of subjects are excluded,
    the estimator falls back to literature priors (``fallback_estimates``)
    or conservative defaults.

    The estimator retains a ``diagnostics`` list after estimation, which the
    orchestrator writes to ``nca_diagnostics.jsonl`` in the bundle.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        manifest: DataManifest,
        *,
        fallback_estimates: dict[str, float] | None = None,
    ) -> None:
        """Initialize with a validated DataFrame and optional literature fallback.

        Args:
            df: Validated NONMEM-style PK data (EVID in {0, 1}).
            manifest: DataManifest from ingest (SHA, column mapping, covariates).
            fallback_estimates: Optional literature priors (e.g., from a
                DatasetCard's published_model.key_estimates). When provided
                and per-subject NCA fails QC, these are used as root
                initial estimates instead of conservative defaults.
        """
        self._df = df
        self._manifest = manifest
        self._obs: pd.DataFrame = cast("pd.DataFrame", df[df["EVID"] == 0].copy())
        self._doses: pd.DataFrame = cast("pd.DataFrame", df[df["EVID"] == 1].copy())
        self._fallback_estimates = fallback_estimates
        self.diagnostics: list[NCASubjectDiagnostic] = []
        self.fallback_source: str = "nca"  # nca | dataset_card | defaults

    def estimate_per_subject(self) -> dict[str, float]:
        """Derive population-median NCA estimates from per-subject NCA.

        Returns a dict with keys like ``CL``, ``V``, ``ka`` mapped to median
        values across QC-passing subjects. Underscore-prefixed keys carry
        metadata (unit scale, excluded fraction). Falls back to
        ``fallback_estimates`` or defaults when QC exclusion exceeds 50%.
        """
        self.diagnostics = []  # reset for repeated calls
        subjects: list[object] = list(self._obs["NMID"].unique())
        if len(subjects) < 2:
            self.fallback_source = "defaults"
            return self.estimate_population_level()

        included: list[NCAResult] = []
        for subj in subjects:
            subject_result = self._nca_for_subject(subj)
            self.diagnostics.append(_to_diagnostic(str(subj), subject_result))
            if not subject_result.excluded:
                included.append(subject_result)

        n_subjects = len(subjects)
        n_excluded = n_subjects - len(included)
        excluded_fraction = n_excluded / n_subjects if n_subjects > 0 else 1.0

        if excluded_fraction >= _NCA_FALLBACK_EXCLUSION_THRESHOLD or not included:
            return self._apply_fallback(excluded_fraction)

        cl_median = float(np.median([r.cl for r in included]))
        v_median = float(np.median([r.v for r in included]))
        ka_median = float(np.median([r.ka for r in included]))

        # Unit-scaling heuristic: NCA computes CL=Dose/AUC directly, which is
        # correct only when dose and DV mass units match. A mg dose with ng/mL
        # DV (a routine convention) yields CL 1000x too small; the heuristic
        # detects that case and applies a multiplier.
        scale, _reason = _detect_unit_scale_factor(self._doses, self._obs, cl_median)
        cl_median *= scale
        v_median *= scale

        estimates: dict[str, float] = {
            "CL": cl_median,
            "V": v_median,
            "ka": ka_median,
            "_excluded_fraction": round(excluded_fraction, 4),
        }
        if scale != 1.0:
            estimates["_unit_scale_applied"] = scale
        self.fallback_source = "nca"
        return estimates

    def estimate_population_level(self) -> dict[str, float]:
        """Population-level NCA on naive-averaged profiles (sparse-data fallback)."""
        if self._obs.empty:
            return _default_estimates()

        grouped = cast("pd.Series", self._obs.groupby("TIME")["DV"].mean())
        pooled = grouped.reset_index().sort_values("TIME")
        times = pooled["TIME"].to_numpy(dtype=float)
        concs = pooled["DV"].to_numpy(dtype=float)

        if self._doses.empty:
            return _default_estimates()
        per_subj_dose = cast("pd.Series", self._doses.groupby("NMID")["AMT"].sum())
        dose = float(per_subj_dose.median())
        if dose <= 0:
            return _default_estimates()

        pos_mask = concs > 0
        if pos_mask.sum() < _NCA_MIN_LAMBDA_POINTS:
            return _default_estimates()

        nca = _compute_nca_single_subject(times[pos_mask], concs[pos_mask], dose)
        if nca is None or nca.excluded:
            return _default_estimates()
        # Apply the same unit-scale heuristic as ``estimate_per_subject`` so
        # single-subject / sparse fallbacks don't silently return CL/V that
        # are 1000x off for mg-dose / ng-mL data.
        scale, _reason = _detect_unit_scale_factor(self._doses, self._obs, nca.cl)
        return {"CL": nca.cl * scale, "V": nca.v * scale, "ka": nca.ka}

    def build_entry(
        self,
        candidate_id: str,
        source: Literal["nca", "warm_start", "fallback"] = "nca",
        *,
        estimates: dict[str, float] | None = None,
    ) -> InitialEstimateEntry:
        """Build an InitialEstimateEntry with NCA-derived estimates.

        When ``estimates`` is provided, it is reused verbatim and
        :meth:`estimate_per_subject` is not re-invoked. Pass the
        pre-computed estimates to avoid redundant NCA computation and to
        preserve the diagnostics produced by the original call (the
        method resets ``self.diagnostics`` on every invocation).
        """
        est = estimates if estimates is not None else self.estimate_per_subject()
        inputs = [f"per_subject_nca:{self.fallback_source}" if source == "nca" else source]
        return InitialEstimateEntry(
            candidate_id=candidate_id,
            source=source,
            estimates=est,
            inputs_used=inputs,
        )

    def emit_plots(self, out_dir: Path) -> int:
        """Emit per-subject NCA diagnostic plots (semilog + λz window + AUC).

        Does nothing (and returns 0) if matplotlib is not installed.
        Returns the number of plots written.
        """
        try:
            import matplotlib  # type: ignore[import-not-found]

            matplotlib.use("Agg")  # non-interactive
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        except ImportError:
            return 0

        if not self.diagnostics:
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for diag in self.diagnostics:
            mask = self._obs["NMID"].astype(str) == diag.subject_id
            subj_obs = cast("pd.DataFrame", self._obs[mask].sort_values(by="TIME"))
            if subj_obs.empty:
                continue
            times = subj_obs["TIME"].to_numpy(dtype=float)
            concs = subj_obs["DV"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(6, 4))
            pos = concs > 0
            if pos.any():
                ax.semilogy(times[pos], concs[pos], "o-", label="observed")
            title_parts = [f"Subject {diag.subject_id}"]
            if diag.lambda_z_adj_r2 is not None:
                title_parts.append(f"adj_r²={diag.lambda_z_adj_r2:.3f}")
            if diag.lambda_z_n_points is not None:
                title_parts.append(f"n_λz={diag.lambda_z_n_points}")
            if diag.auc_extrap_fraction is not None:
                title_parts.append(f"extrap={diag.auc_extrap_fraction * 100:.1f}%")
            if diag.excluded:
                title_parts.append("EXCLUDED")
            ax.set_title(" · ".join(title_parts))
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration (log)")
            ax.grid(visible=True, which="both", alpha=0.3)
            if diag.excluded and diag.excluded_reason:
                ax.text(
                    0.02,
                    0.02,
                    diag.excluded_reason,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="red",
                )
            safe_id = _sanitize_filename(diag.subject_id)
            path = out_dir / f"subject_{safe_id}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=90)
            plt.close(fig)
            n_written += 1
        return n_written

    # --- Internals ---

    def _nca_for_subject(self, subj: object) -> NCAResult:
        """Compute NCA for one subject; always returns an NCAResult.

        When computation cannot proceed (too few points, no terminal fit),
        returns an excluded placeholder so the diagnostic emission still
        records the subject in nca_diagnostics.jsonl.
        """
        subj_obs = cast(
            "pd.DataFrame",
            self._obs[self._obs["NMID"] == subj].sort_values(by="TIME"),
        )
        subj_dose = cast("pd.DataFrame", self._doses[self._doses["NMID"] == subj])

        if len(subj_obs) < _NCA_MIN_LAMBDA_POINTS or subj_dose.empty:
            return _excluded_result(str(subj), "insufficient observations or doses")

        times = subj_obs["TIME"].to_numpy(dtype=float)
        concs = subj_obs["DV"].to_numpy(dtype=float)
        pos_mask = concs > 0
        if pos_mask.sum() < _NCA_MIN_LAMBDA_POINTS:
            return _excluded_result(str(subj), "fewer than 3 positive concentrations")

        is_multi, tau = _detect_multi_dose(subj_dose)
        amt_series = subj_dose["AMT"]
        dose = float(amt_series.iloc[-1]) if is_multi else float(amt_series.sum())
        if dose <= 0:
            return _excluded_result(str(subj), "non-positive dose")

        last_dose_time = float(subj_dose["TIME"].max()) if not subj_dose.empty else None
        result = _compute_nca_single_subject(
            times[pos_mask],
            concs[pos_mask],
            dose,
            is_steady_state=is_multi,
            tau=tau,
            last_dose_time=last_dose_time,
        )
        if result is None:
            return _excluded_result(str(subj), "no viable terminal-phase fit or zero AUC")
        return result

    def _apply_fallback(self, excluded_fraction: float) -> dict[str, float]:
        """Populate initial estimates from a literature prior or conservative defaults.

        The textual source (``"dataset_card"`` vs ``"defaults"``) is tracked on
        ``self.fallback_source``. Underscore-prefixed dict keys carry numeric
        metadata only.

        #27: when the dataset card does not carry a ``ka``, the rc8
        path silently defaulted to 1.0 /h - a 10-to-100x warm-start error
        for fast- or slow-absorption drugs. Emit a structured warning
        and set a ``_ka_defaulted=1`` metadata flag so credibility and
        Gate 1 consumers can see the substitution.
        """
        if self._fallback_estimates:
            self.fallback_source = "dataset_card"
            est = dict(self._fallback_estimates)
            if "ka" not in est:
                _logger.warning(
                    "initial_estimates_ka_defaulted",
                    extra={
                        "defaulted_value": 1.0,
                        "reason": "dataset_card fallback did not specify ka",
                        "excluded_fraction": excluded_fraction,
                    },
                )
                est["ka"] = 1.0
                est["_ka_defaulted"] = 1.0
            est["_excluded_fraction"] = round(excluded_fraction, 4)
            return est
        self.fallback_source = "defaults"
        _logger.warning(
            "initial_estimates_using_defaults",
            extra={
                "reason": "no NCA and no dataset_card prior available",
                "excluded_fraction": excluded_fraction,
            },
        )
        est = _default_estimates()
        # _default_estimates carries ka=1.0 — flag it so the audit
        # trail is consistent with the dataset_card branch.
        est["_ka_defaulted"] = 1.0
        est["_excluded_fraction"] = round(excluded_fraction, 4)
        return est


# ---------------------------------------------------------------------------
# Warm-start + bundle construction
# ---------------------------------------------------------------------------


def warm_start_estimates(
    parent_estimates: dict[str, float],
    candidate_id: str,
) -> InitialEstimateEntry:
    """Build warm-start initial estimates from a parent model's best-fit."""
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
    return InitialEstimates(entries={e.candidate_id: e for e in entries})


# ---------------------------------------------------------------------------
# Internal NCA computation
# ---------------------------------------------------------------------------


def _compute_nca_single_subject(
    times: np.ndarray,
    concs: np.ndarray,
    dose: float,
    *,
    is_steady_state: bool = False,
    tau: float | None = None,
    last_dose_time: float | None = None,
) -> NCAResult | None:
    """Compute NCA for one subject with QC gates.

    Algorithm:
      1. AUC_last via linear-up/log-down trapezoidal (Purves 1992, PKNCA default).
      2. Terminal lambda_z via adaptive curve-stripping (PKNCA pk.calc.half.life).
      3. AUC_inf = AUC_last + C_last/lambda_z (or AUC_tau at steady state).
      4. CL = Dose / AUC_inf (or Dose / AUC_tau).
      5. QC gates: adj_r²≥0.80, extrap≤20%, span_ratio≥1, n≥3.

    Returns an NCAResult with ``excluded=True`` and ``excluded_reason`` when
    a QC gate fails; returns None only when computation cannot proceed at all
    (non-finite inputs, non-positive dose, zero AUC).
    """
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(concs))):
        return None
    if dose <= 0 or not np.isfinite(dose):
        return None

    cmax_idx = int(np.argmax(concs))
    cmax = float(concs[cmax_idx])
    tmax = float(times[cmax_idx])
    if cmax <= 0:
        return None

    auc_last = _auc_lin_up_log_down(times, concs)
    if not np.isfinite(auc_last) or auc_last <= 0:
        return None

    lam = _select_lambda_z(times, concs, tmax)
    if lam is None:
        # No viable terminal-phase fit — computation cannot proceed.
        return None

    kel = lam.kel
    adj_r2 = lam.adj_r2
    n_points = lam.n_points
    span_ratio = lam.span_ratio

    # Steady-state multi-dose branch: use AUC_tau instead of AUC_inf.
    # Anchor the tau window on the last dose time when provided, so a long
    # profile spanning multiple dosing intervals collapses to the final
    # interval (true SS). Fallback to times[0]+tau preserves legacy behavior
    # for call sites that haven't been updated yet.
    if is_steady_state and tau is not None and tau > 0:
        t_tau_start = float(last_dose_time) if last_dose_time is not None else float(times[0])
        t_tau_end = t_tau_start + tau
        tau_mask = (times >= t_tau_start) & (times <= t_tau_end)
        if tau_mask.sum() >= 2:
            auc_tau = _auc_lin_up_log_down(times[tau_mask], concs[tau_mask])
            if auc_tau > 0:
                cl = dose / auc_tau
                v = cl / kel
                ka = 1.0 / tmax if tmax > 0 else 1.0
                return _maybe_exclude(
                    NCAResult(
                        cl=cl,
                        v=v,
                        ka=ka,
                        kel=kel,
                        tmax=tmax,
                        cmax=cmax,
                        auc_last=auc_tau,
                        auc_inf=auc_tau,
                        auc_extrap_fraction=0.0,
                        lambda_z_adj_r2=adj_r2,
                        lambda_z_n_points=n_points,
                        span_ratio=span_ratio,
                    )
                )

    c_last = float(concs[-1])
    auc_extrap = c_last / kel if c_last > 0 else 0.0
    auc_inf = auc_last + auc_extrap
    if auc_inf <= 0:
        return None
    auc_extrap_fraction = auc_extrap / auc_inf

    cl = dose / auc_inf
    v = cl / kel
    ka = 1.0 / tmax if tmax > 0 else 1.0

    return _maybe_exclude(
        NCAResult(
            cl=cl,
            v=v,
            ka=ka,
            kel=kel,
            tmax=tmax,
            cmax=cmax,
            auc_last=auc_last,
            auc_inf=auc_inf,
            auc_extrap_fraction=auc_extrap_fraction,
            lambda_z_adj_r2=adj_r2,
            lambda_z_n_points=n_points,
            span_ratio=span_ratio,
        )
    )


def _select_lambda_z(
    times: np.ndarray,
    concs: np.ndarray,
    tmax: float,
    *,
    min_points: int = _NCA_MIN_LAMBDA_POINTS,
    adj_r2_factor: float = _NCA_ADJ_R2_FACTOR,
) -> LambdaZFit | None:
    """PKNCA-style curve-stripping for terminal lambda_z.

    Algorithm (matches PKNCA ``pk.calc.half.life`` with
    ``allow.tmax.in.half.life = FALSE``):

      1. Eligible points are strictly post-Tmax with positive concentrations.
      2. For each window size k from ``min_points`` to n_eligible, fit a
         log-linear regression to the LAST k points: log(C) = a - kel·t.
      3. Discard fits with non-positive kel.
      4. Identify the maximum adjusted R² among surviving fits.
      5. Keep fits within ``adj_r2_factor`` of the maximum (ties).
      6. Tiebreak: choose the fit with the most points.

    Returns dict with kel, adj_r2, r2, n_points, span_ratio, t_first, t_last,
    or None if no eligible fit exists.

    Reference:
      https://humanpred.github.io/pknca/articles/v06-half-life-calculation.html
      Gabrielsson & Weiner §2.8.4.
    """
    post_mask = (times > tmax) & (concs > 0)
    t_post = times[post_mask]
    c_post = concs[post_mask]
    if len(t_post) < min_points:
        return None

    # Ensure sorted by time (the caller sorts, but be defensive)
    order = np.argsort(t_post)
    t_post = t_post[order]
    c_post = c_post[order]

    fits: list[LambdaZFit] = []
    for k in range(min_points, len(t_post) + 1):
        tw = t_post[-k:]
        cw = c_post[-k:]
        slope, _intercept, r2 = _log_linear_fit(tw, cw)
        if slope >= 0 or not np.isfinite(slope):
            continue
        kel = -slope
        # Adjusted R²; avoid division-by-zero for k=2 (guarded by min_points≥3).
        adj_r2 = 1.0 - (1.0 - r2) * (k - 1) / max(k - 2, 1)
        half_life = np.log(2.0) / kel if kel > 0 else float("inf")
        span = float(tw[-1] - tw[0])
        span_ratio = span / half_life if half_life > 0 and np.isfinite(half_life) else 0.0
        fits.append(
            LambdaZFit(
                kel=float(kel),
                r2=float(r2),
                adj_r2=float(adj_r2),
                n_points=int(k),
                span_ratio=float(span_ratio),
                t_first=float(tw[0]),
                t_last=float(tw[-1]),
            )
        )

    if not fits:
        return None

    best_adj_r2 = max(f.adj_r2 for f in fits)
    qualifying = [f for f in fits if f.adj_r2 >= best_adj_r2 - adj_r2_factor]
    return max(qualifying, key=lambda f: f.n_points)


def _log_linear_fit(
    times: np.ndarray,
    concs: np.ndarray,
) -> tuple[float, float, float]:
    """Least-squares fit of log(C) = intercept + slope·t.

    Returns (slope, intercept, r²). Uses explicit sums to avoid external deps.
    """
    clipped = np.clip(concs, 1e-100, 1e100)
    y = np.log(clipped)
    x = times
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    ss_xx = float(np.sum((x - mean_x) ** 2))
    ss_yy = float(np.sum((y - mean_y) ** 2))
    ss_xy = float(np.sum((x - mean_x) * (y - mean_y)))
    if ss_xx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    # Clip r² to [0, 1]; floating-point noise with near-constant y can push
    # the ratio slightly above 1 and corrupt the lambda_z tiebreak.
    r2 = (ss_xy * ss_xy) / (ss_xx * ss_yy) if ss_yy > 0 else 1.0
    return slope, intercept, float(max(0.0, min(1.0, r2)))


def _auc_lin_up_log_down(times: np.ndarray, concs: np.ndarray) -> float:
    """Linear-up/log-down AUC integration (PKNCA default; Purves 1992).

    Per-interval rules:
      - Both endpoints zero: contributes 0.
      - Declining AND both endpoints positive: log-trapezoid
        ``dt * (C1 - C2) / ln(C1/C2)``.
      - Otherwise (rising, plateau, or post-BLQ zero): linear trapezoid
        ``dt * (C1 + C2) / 2``.

    Reference:
      https://humanpred.github.io/pknca/articles/v23-auc-integration-methods.html
      Purves (1992) J Pharmacokin Biopharm 20:211.
    """
    if len(times) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0:
            continue
        c1 = float(concs[i])
        c2 = float(concs[i + 1])
        if c1 == 0.0 and c2 == 0.0:
            continue
        if c2 < c1 and c2 > 0.0 and c1 > 0.0:
            # Log trapezoid — exact for monoexponential decline. Guard
            # against c1≈c2 where ln(c1/c2)→0 would blow up; in that case
            # the log and linear trapezoids converge so use linear.
            log_ratio = np.log(c1 / c2)
            if abs(log_ratio) < 1e-9:
                auc += dt * (c1 + c2) / 2.0
            else:
                auc += dt * (c1 - c2) / log_ratio
        else:
            auc += dt * (c1 + c2) / 2.0
    return float(auc)


def _detect_multi_dose(dose_df: pd.DataFrame) -> tuple[bool, float | None]:
    """Detect multiple doses and estimate the inter-dose interval (tau)."""
    if len(dose_df) < 2:
        return False, None
    dose_times = np.sort(dose_df["TIME"].to_numpy(dtype=float))
    intervals = np.diff(dose_times)
    if len(intervals) == 0:
        return False, None
    tau = float(np.median(intervals))
    return (True, tau) if tau > 0 else (False, None)


def _default_estimates() -> dict[str, float]:
    """Fallback estimates when NCA fails and no literature prior is provided."""
    return {"CL": 5.0, "V": 70.0, "ka": 1.0}


def _maybe_exclude(result: NCAResult) -> NCAResult:
    """Mark a result as excluded when any QC gate fails."""
    reasons: list[str] = []
    if result.lambda_z_adj_r2 < _NCA_MIN_ADJ_R2:
        reasons.append(f"adj_r²={result.lambda_z_adj_r2:.3f}<{_NCA_MIN_ADJ_R2:.2f}")
    if result.auc_extrap_fraction > _NCA_MAX_EXTRAP_FRACTION:
        reasons.append(
            f"AUC_extrap={result.auc_extrap_fraction * 100:.1f}%"
            f">{_NCA_MAX_EXTRAP_FRACTION * 100:.0f}%"
        )
    if result.span_ratio < _NCA_MIN_SPAN_RATIO:
        reasons.append(f"span_ratio={result.span_ratio:.2f}<{_NCA_MIN_SPAN_RATIO:.1f}")
    if result.lambda_z_n_points < _NCA_MIN_LAMBDA_POINTS:
        reasons.append(f"n_λz={result.lambda_z_n_points}<{_NCA_MIN_LAMBDA_POINTS}")
    if reasons:
        result.excluded = True
        result.excluded_reason = "; ".join(reasons)
    return result


def _excluded_result(
    _subject_id: str,
    reason: str,
    *,
    tmax: float = 0.0,
    cmax: float = 0.0,
    auc_last: float = 0.0,
) -> NCAResult:
    """Build a placeholder excluded result (used when computation can't proceed)."""
    return NCAResult(
        cl=0.0,
        v=0.0,
        ka=0.0,
        kel=0.0,
        tmax=tmax,
        cmax=cmax,
        auc_last=auc_last,
        auc_inf=0.0,
        auc_extrap_fraction=0.0,
        lambda_z_adj_r2=0.0,
        lambda_z_n_points=0,
        span_ratio=0.0,
        excluded=True,
        excluded_reason=reason,
    )


def _to_diagnostic(subject_id: str, r: NCAResult) -> NCASubjectDiagnostic:
    return NCASubjectDiagnostic(
        subject_id=subject_id,
        tmax=r.tmax if not r.excluded or r.tmax > 0 else None,
        cmax=r.cmax if not r.excluded or r.cmax > 0 else None,
        cl=r.cl if not r.excluded else None,
        v=r.v if not r.excluded else None,
        ka=r.ka if not r.excluded else None,
        kel=r.kel if not r.excluded else None,
        auc_last=r.auc_last if r.auc_last > 0 else None,
        auc_inf=r.auc_inf if r.auc_inf > 0 else None,
        auc_extrap_fraction=r.auc_extrap_fraction if not r.excluded or r.auc_last > 0 else None,
        lambda_z_adj_r2=r.lambda_z_adj_r2 if r.lambda_z_adj_r2 > 0 else None,
        lambda_z_n_points=r.lambda_z_n_points if r.lambda_z_n_points > 0 else None,
        span_ratio=r.span_ratio if r.span_ratio > 0 else None,
        excluded=r.excluded,
        excluded_reason=r.excluded_reason,
    )


def _sanitize_filename(value: str) -> str:
    """Strip unsafe characters for filesystem paths (subject IDs can be weird)."""
    import re

    return re.sub(r"[^A-Za-z0-9_\-]", "_", value) or "unknown"


def _detect_unit_scale_factor(
    doses: pd.DataFrame,
    obs: pd.DataFrame,
    cl_estimate: float,
) -> tuple[float, str]:
    """Detect the unit-conversion factor needed for dose/concentration units.

    NCA computes CL = Dose / AUC directly, which yields L/h only when dose and
    DV mass units match. When dose is in mg but DV is in ng/mL (a routine
    pharmacometric convention), the raw CL is 1000x too small. This heuristic
    detects such mismatches and returns a multiplier (1.0, 1000.0, or 1e6).

    Heuristics (priority order):
      1. cl_estimate < 1e-4 AND dv_median > 50000 AND dose_median >= 1 → DV
         likely in pg/mL → x1e6.
      2. cl_estimate < 0.5   AND dv_median > 50    AND dose_median >= 1 → DV
         likely in ng/mL → x1000.
      3. Otherwise no conversion.

    Typical adult human CL is 0.5-100 L/h; pairing an implausibly small CL
    with a large DV magnitude (>50) strongly indicates a mg-dose/ng-mL-DV gap.
    The ``dose_median >= 1`` guard prevents over-correction of preclinical
    data (small-animal studies, biologics) where a legitimate low-CL drug
    with ng/mL concentrations may superficially match the heuristic.
    """
    if doses.empty or obs.empty:
        return 1.0, "no dose/obs data"
    # Median over positive doses so placebo / zero-dose records do not
    # collapse the heuristic when the active-treatment arm is obviously
    # mismatched.
    pos_doses = doses[doses["AMT"] > 0]["AMT"]
    dose_median = float(pos_doses.median()) if not pos_doses.empty else 0.0
    dv_pos = obs[obs["DV"] > 0]["DV"]
    if dv_pos.empty or dose_median <= 0:
        return 1.0, "no positive DV or invalid dose"
    dv_median = float(dv_pos.median())
    if cl_estimate < 0.0001 and dv_median > 50000 and dose_median >= 1.0:
        return 1e6, "dose in mg but DV likely in pg/mL (x1e6)"
    if cl_estimate < 0.5 and dv_median > 50 and dose_median >= 1.0:
        return 1000.0, "dose in mg but DV likely in ng/mL (x1000)"
    return 1.0, "units commensurate"
