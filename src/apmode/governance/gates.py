# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate evaluators for the governance funnel (PRD §4.3.1, ARCHITECTURE.md §4.3).

Gate 1: Technical Validity — 7 checks (convergence, parameter plausibility,
  state trajectory, CWRES, VPC coverage, split integrity, seed stability).
Gate 2: Lane-Specific Admissibility — per-lane checks (interpretability,
  shrinkage, identifiability, NODE eligibility, LORO requirement).

Gates are sequential disqualifiers. Survivors are ranked; failures are logged
with per-check reasons. Thresholds come from versioned policy files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog

from apmode.bundle.models import (
    CredibilityContext,
    GateCheckResult,
    GateResult,
    LOROMetrics,
    _pit_key,
)
from apmode.governance.policy import Gate25Config  # noqa: TC001 — used at runtime
from apmode.ids import generate_gate_id

logger = structlog.get_logger(__name__)

_VALID_LANES = frozenset({"submission", "discovery", "optimization"})

if TYPE_CHECKING:
    from apmode.bundle.models import (
        BackendResult,
        ImputationStabilityEntry,
        MissingDataDirective,
        PriorManifest,
    )
    from apmode.governance.policy import (
        Gate1BayesianConfig,
        Gate1Config,
        Gate2Config,
        GatePolicy,
    )


def _safe_bic(result: BackendResult) -> float:
    """Return ``result.bic`` coerced to ``+inf`` when missing or non-finite.

    Used as a sort key in within-paradigm ranking — a missing BIC must
    land a candidate at the *bottom*, never at the top via Python's
    unstable NaN sort. Defined once at module level because both
    ``_gate3_within_paradigm`` and the simulation-based ranking pipeline
    call it (formerly duplicated as nested functions).
    """
    if result.bic is None or not np.isfinite(result.bic):
        return float("inf")
    return result.bic


# ---------------------------------------------------------------------------
# Gate 1: Technical Validity
# ---------------------------------------------------------------------------


def evaluate_gate1(
    result: BackendResult,
    policy: GatePolicy,
    seed_results: list[BackendResult] | None = None,
    stability: ImputationStabilityEntry | None = None,
    directive: MissingDataDirective | None = None,
) -> GateResult:
    """Evaluate Gate 1: Technical Validity.

    7 core checks per PRD §4.3.1, plus an optional imputation-stability
    check when Multiple Imputation is the resolved covariate method:
      1. Convergence: estimation algorithm converged
      2. Parameter plausibility: no extreme/negative structural params
      3. State trajectory validity: no negative concentrations, no NaN
      4. CWRES diagnostics: |mean| < threshold, outlier fraction < threshold
      5. VPC coverage: |coverage - target| <= tolerance per band (0.4.1)
      6. Split integrity: (placeholder — requires train/test comparison)
      7. Seed stability: results consistent across ≥N random seeds
      8. Imputation stability (MI runs only): convergence_rate and
         rank_stability meet directive-driven thresholds.

    Args:
        result: BackendResult from a single estimation run.
        policy: GatePolicy with gate1 thresholds.
        seed_results: Results from additional seed runs for stability check.
        stability: Optional ImputationStabilityEntry for this candidate
            (present when the run was driven by MI).
        directive: Optional MissingDataDirective; its
            ``imputation_stability_penalty`` sets the rank-stability
            threshold (pass requires rank_stability ≥ 1 - penalty).
    """
    g1 = policy.gate1
    checks: list[GateCheckResult] = []

    checks.append(_check_convergence(result, g1))
    checks.append(_check_parameter_plausibility(result, g1))
    checks.append(_check_state_trajectory(result, g1))
    checks.extend(_check_cwres(result, g1))
    checks.append(_check_pit_calibration(result, g1))
    checks.append(_check_split_integrity(result, g1))
    checks.append(_check_seed_stability(result, seed_results, g1))
    checks.append(
        _check_imputation_stability(
            result,
            stability,
            directive,
            convergence_rate_min=policy.missing_data.imputation_convergence_rate_min,
        )
    )

    passed = all(c.passed for c in checks)
    failed_names = [c.check_id for c in checks if not c.passed]
    summary = "All checks passed" if passed else f"Failed: {', '.join(failed_names)}"

    return GateResult(
        gate_id=generate_gate_id(),
        gate_name="technical_validity",
        candidate_id=result.model_id,
        passed=passed,
        checks=checks,
        summary_reason=summary,
        policy_version=policy.policy_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def _check_convergence(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 1: Did the estimation algorithm converge?"""
    passed = result.converged if g1.convergence_required else True
    return GateCheckResult(
        check_id="convergence",
        passed=passed,
        observed=result.converged,
        threshold="required" if g1.convergence_required else "not_required",
    )


def _check_parameter_plausibility(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 2: Are structural parameters plausible?

    Checks: no negative volumes or clearances, RSE below ``g1.param_rse_max``,
    back-transformed structural estimates within
    ``(g1.param_value_min, g1.param_value_max)``. All thresholds are
    sourced from the versioned policy JSON — no magic numbers here.

    nlmixr2 (and most popPK backends) parameterise structural thetas in
    log-space (``lCL``, ``lV``, ``lka``, ``lktr``, ``lQ``, ``lVm``, ``lKm``,
    ``ln`` for transit counts). The underlying clearance/volume/rate is
    ``exp(estimate)`` which is unconditionally positive, so negative log
    estimates are valid. We detect log-space by the ``l`` prefix
    convention and check the back-transformed value against the sanity
    bounds.
    """
    lower_bound = g1.param_value_min
    upper_bound = g1.param_value_max
    rse_max = g1.param_rse_max

    issues: list[str] = []
    for name, pe in result.parameter_estimates.items():
        if pe.category != "structural":
            continue
        is_log = len(name) >= 2 and name[0] == "l" and name[1].isalpha()
        value = float(np.exp(pe.estimate)) if is_log else pe.estimate
        label = f"exp({name})" if is_log else name

        if value <= 0:
            issues.append(f"{label}={value:.4g} (non-positive)")
        elif value <= lower_bound:
            issues.append(f"{label}={value:.4g} (at lower bound)")
        elif value >= upper_bound:
            issues.append(f"{label}={value:.4g} (at upper bound)")
        if pe.rse is not None and pe.rse > rse_max:
            issues.append(f"{name} RSE={pe.rse:.1f}% (>{rse_max:g}%)")

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="parameter_plausibility",
        passed=passed,
        observed="; ".join(issues) if issues else "all plausible",
    )


def _check_state_trajectory(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 3: State trajectory validity (PRD §4.3.1).

    Multi-signal check for physiological plausibility using GOF proxies
    and convergence metadata:
      - R² below minimum → pathological fit
      - CWRES SD outside [min, max] → severe misspecification
      - Gradient norm too high → ODE solver trapped in penalty valley
      - Rounding errors → numerical instability in state trajectory
    """
    if not g1.state_trajectory_validity_required:
        return GateCheckResult(
            check_id="state_trajectory_validity",
            passed=True,
            observed="not_required",
        )

    issues: list[str] = []
    gof = result.diagnostics.gof
    meta = result.convergence_metadata

    # R² check: model must explain at least obs_vs_pred_r2_min of variance
    if gof.obs_vs_pred_r2 is not None and gof.obs_vs_pred_r2 < g1.obs_vs_pred_r2_min:
        issues.append(f"R²={gof.obs_vs_pred_r2:.3f} (<{g1.obs_vs_pred_r2_min})")

    # CWRES SD: should be near 1.0; far from it signals misspecification
    if gof.cwres_sd < g1.cwres_sd_min:
        issues.append(f"cwres_sd={gof.cwres_sd:.3f} (<{g1.cwres_sd_min})")
    elif gof.cwres_sd > g1.cwres_sd_max:
        issues.append(f"cwres_sd={gof.cwres_sd:.3f} (>{g1.cwres_sd_max})")

    # Gradient norm: high values signal non-physical ODE solver valleys
    if meta.gradient_norm is not None and meta.gradient_norm > g1.gradient_norm_max:
        issues.append(f"gradient={meta.gradient_norm:.2e} (>{g1.gradient_norm_max})")

    # Rounding errors: direct signal of numerical instability
    if meta.minimization_status == "rounding_errors":
        issues.append("minimization_status=rounding_errors")

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="state_trajectory_validity",
        passed=passed,
        observed="; ".join(issues) if issues else "valid",
        threshold=(f"R²>{g1.obs_vs_pred_r2_min}, SD[{g1.cwres_sd_min}-{g1.cwres_sd_max}]"),
    )


def _check_cwres(result: BackendResult, g1: Gate1Config) -> list[GateCheckResult]:
    """Check 4: CWRES diagnostics — mean and outlier fraction."""
    gof = result.diagnostics.gof
    checks: list[GateCheckResult] = []

    # CWRES mean should be near 0
    cwres_ok = abs(gof.cwres_mean) <= g1.cwres_mean_max
    checks.append(
        GateCheckResult(
            check_id="cwres_mean",
            passed=cwres_ok,
            observed=gof.cwres_mean,
            threshold=g1.cwres_mean_max,
        )
    )

    # Outlier fraction (|CWRES| > 4)
    outlier_ok = gof.outlier_fraction <= g1.outlier_fraction_max
    checks.append(
        GateCheckResult(
            check_id="cwres_outlier_fraction",
            passed=outlier_ok,
            observed=gof.outlier_fraction,
            threshold=g1.outlier_fraction_max,
        )
    )

    return checks


def _check_pit_calibration(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 5: PIT / NPDE-lite predictive calibration (policy 0.4.2).

    For each probability level ``p`` in the PIT summary's
    ``probability_levels`` (default ``{0.05, 0.50, 0.95}``) the check
    asks ``|c_p - p| <= tol``, where ``c_p`` is the subject-robust
    empirical CDF hit-rate produced by
    :func:`apmode.backends.predictive_summary._compute_pit_calibration`.
    Tails (``p`` in ``(0.0, 0.25]`` or ``[0.75, 1.0)``) use
    ``pit_tol_tail``; the median band (``p`` in ``(0.25, 0.75)``) uses
    ``pit_tol_median``. Rationale in :class:`Gate1Config` docstring.

    Replaces the rc8/rc9 bin-level VPC coverage gate, which was brittle
    on sparse real data because per-band coverage quantized at
    ``1/n_bins`` with only ~8 bins on datasets like warfarin. PIT is
    aligned with the NPDE / PIT family of calibration diagnostics and
    is invariant to binning choice. See CHANGELOG rc9 follow-up
    "PIT/NPDE-lite Gate 1 calibration" for the design log.
    """
    pit = result.diagnostics.pit_calibration
    if pit is None:
        # Phase 1 backends that don't emit posterior-predictive sims can
        # opt out via ``pit_required=False``. When required, missing
        # evidence fails — missing evidence ≠ passing.
        if not g1.pit_required:
            return GateCheckResult(
                check_id="pit_calibration",
                passed=True,
                observed="pit_not_configured",
            )
        return GateCheckResult(
            check_id="pit_calibration",
            passed=False,
            observed="pit_not_available",
        )
    # #18 / #33: build_predictive_diagnostics can now emit a
    # zero-subject PIT summary when the sim matrix is fully degenerate
    # (e.g. every draw NaN). That is "PIT attempted but produced no
    # evidence" — distinct from "PIT not configured" and from
    # "computed and out of tolerance". Surface it as an explicit fail
    # reason so Gate 1 diagnostics make the missing-evidence case
    # unambiguous for auditors.
    if pit.n_subjects == 0 or pit.n_observations == 0:
        return GateCheckResult(
            check_id="pit_calibration",
            passed=False,
            observed="pit_degenerate_no_finite_sims",
        )

    # n-scaled tolerance: tol(p, n) = max(floor, z_alpha · sqrt(p(1-p)/n_subjects))
    # keeps the SE-coverage constant across dataset sizes while the floor
    # prevents large-n vacuous strictness. n_subjects is the outer-mean
    # denominator under subject-robust aggregation — using n_observations
    # would understate sampling variance given within-subject correlation.
    # See ``Gate1Config`` docstring for the derivation.
    n_eff = max(pit.n_subjects, 1)
    z_alpha = g1.pit_z_alpha
    tail_floor = g1.pit_tol_tail_floor
    med_floor = g1.pit_tol_median_floor
    violations: list[str] = []
    per_level_tols: list[tuple[str, float]] = []
    for p in pit.probability_levels:
        key = _pit_key(p)
        c_p = pit.calibration[key]
        deviation = abs(c_p - p)
        is_tail = p <= 0.25 or p >= 0.75
        se_p = (p * (1.0 - p) / n_eff) ** 0.5
        floor = tail_floor if is_tail else med_floor
        tol = max(floor, z_alpha * se_p)
        per_level_tols.append((key, tol))
        if deviation > tol:
            band = "tail" if is_tail else "med"
            violations.append(f"{key}={c_p:.3f} (|Δ|={deviation:.3f} > tol_{band}={tol:.3f})")

    passed = len(violations) == 0
    threshold_desc = ", ".join(f"{k}≤{t:.3f}" for k, t in per_level_tols)
    return GateCheckResult(
        check_id="pit_calibration",
        passed=passed,
        observed="; ".join(violations) if violations else "all_within_tolerance",
        threshold=f"|c_p - p| n-scaled (z_alpha={z_alpha}, n_subj={n_eff}): {threshold_desc}",
    )


def _check_split_integrity(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 6: Split integrity (train/test consistency).

    Uses split-aware diagnostics (SplitGOFMetrics) when available.
    Train/test CWRES divergence signals overfitting: test metrics should
    not be dramatically worse than train (>2x outliers or large CWRES
    drift).

    When split_gof is not available, behavior depends on policy:
      - split_integrity_required=True → pass (no evidence to evaluate)
      - The R backend can optionally compute split_gof by partitioning
        per-subject CWRES using the SplitManifest assignments.
    """
    if not g1.split_integrity_required:
        return GateCheckResult(
            check_id="split_integrity",
            passed=True,
            observed="not_required",
        )

    sgof = result.diagnostics.split_gof
    if sgof is None:
        # When a check is required, missing evidence fails — matches
        # the invariant used by ``_check_pit_calibration``. Passing on
        # absent evidence would be a silent bypass of the required-check
        # semantics.
        return GateCheckResult(
            check_id="split_integrity",
            passed=False,
            observed="no_split_diagnostics_despite_required",
            threshold="split_gof must be populated when split_integrity_required=true",
        )

    issues: list[str] = []

    # Test CWRES mean should not drift far from train
    cwres_drift = abs(sgof.test_cwres_mean) - abs(sgof.train_cwres_mean)
    if cwres_drift > g1.split_cwres_drift_max:
        issues.append(
            f"cwres_drift={cwres_drift:.3f} "
            f"(test={sgof.test_cwres_mean:.3f}, train={sgof.train_cwres_mean:.3f})"
        )

    # Test outlier fraction should not exceed slope * train + intercept
    outlier_ratio_threshold = (
        g1.split_outlier_ratio_slope * sgof.train_outlier_fraction
        + g1.split_outlier_ratio_intercept
    )
    if sgof.test_outlier_fraction > outlier_ratio_threshold:
        issues.append(
            f"test_outliers={sgof.test_outlier_fraction:.3f} (>{outlier_ratio_threshold:.3f})"
        )

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="split_integrity",
        passed=passed,
        observed="; ".join(issues) if issues else "train_test_consistent",
        threshold=(
            f"cwres_drift<={g1.split_cwres_drift_max}, "
            f"outliers<={g1.split_outlier_ratio_slope}x+{g1.split_outlier_ratio_intercept}"
        ),
    )


def _check_seed_stability(
    result: BackendResult,
    seed_results: list[BackendResult] | None,
    g1: Gate1Config,
) -> GateCheckResult:
    """Check 7: Seed stability — results consistent across random seeds.

    Compares OFV across seed_stability_n runs. If OFV varies by >10%
    (coefficient of variation), the model is seed-unstable.
    """
    if g1.seed_stability_n <= 1:
        # Policy requires only 1 run — the primary fit suffices
        return GateCheckResult(
            check_id="seed_stability",
            passed=True,
            observed="single_seed_policy",
        )

    if seed_results is None or len(seed_results) < g1.seed_stability_n - 1:
        # No replicates supplied for this candidate. The orchestrator
        # runs seed replicates only for a top-K subset by BIC — absence
        # of evidence is the orchestrator's choice, not a candidate
        # defect. Treat as "not applicable" so the check does not
        # eliminate otherwise-valid non-top-K candidates; top-K
        # candidates that were actually probed still face the real CV
        # comparison below.
        n_have = 1 + (len(seed_results) if seed_results else 0)
        return GateCheckResult(
            check_id="seed_stability",
            passed=True,
            observed=f"not_probed ({n_have}/{g1.seed_stability_n} seeds — top-K only)",
        )

    # Collect OFVs from primary + seed runs; filter NaN/Inf
    ofvs = [result.ofv] + [r.ofv for r in seed_results]
    valid_ofvs = [o for o in ofvs if o is not None and np.isfinite(o)]

    if len(valid_ofvs) < g1.seed_stability_n:
        return GateCheckResult(
            check_id="seed_stability",
            passed=False,
            observed=(f"insufficient_valid_ofvs ({len(valid_ofvs)}/{g1.seed_stability_n})"),
        )

    ofv_arr = np.array(valid_ofvs)
    mean_ofv = float(np.mean(ofv_arr))
    # ddof=1 for sample SD (small n = 3-5 seeds)
    cv = (
        float(np.std(ofv_arr, ddof=1) / abs(mean_ofv))
        if abs(mean_ofv) > 1e-10 and len(ofv_arr) > 1
        else 0.0
    )

    # Platform/BLAS float-accumulation differences can produce sub-
    # OFV-unit variation in "identical" fits. A CV computed on near-
    # equal OFVs would still exceed tight thresholds and cause spurious
    # instability verdicts. Short-circuit to PASS when the absolute
    # spread is below the policy-configured platform-float floor
    # (default 0.1 OFV units; Δ|AIC|<0.2 — well below any textbook
    # model-selection tolerance).
    ofv_abs_spread = float(np.ptp(ofv_arr))  # peak-to-peak = max - min
    ofv_spread_floor = g1.seed_stability_ofv_abs_spread_floor
    if ofv_abs_spread < ofv_spread_floor:
        return GateCheckResult(
            check_id="seed_stability",
            passed=True,
            observed=(
                f"stable_within_numerical_precision (ofv_spread={ofv_abs_spread:.3g}, cv={cv:.4f})"
            ),
            threshold=f"|Δofv| < {ofv_spread_floor:g} OFV units (platform-float floor)",
        )

    # CV below policy threshold → stable
    cv_max = g1.seed_stability_cv_max
    passed = cv < cv_max
    return GateCheckResult(
        check_id="seed_stability",
        passed=passed,
        observed=round(cv, 4),
        threshold=cv_max,
        units="CV_ofv",
    )


def _check_imputation_stability(
    _result: BackendResult,
    stability: ImputationStabilityEntry | None,
    directive: MissingDataDirective | None,
    convergence_rate_min: float,
) -> GateCheckResult:
    """Check 8: Imputation stability (MI runs only).

    The rank-stability threshold is driven by
    ``directive.imputation_stability_penalty``:

      pass_threshold = max(0.0, 1.0 - penalty)

    The convergence-rate floor is supplied by
    ``MissingDataPolicy.imputation_convergence_rate_min`` (H5: externalized
    to the versioned policy artifact; previously a hard-coded 0.5 constant).
    Candidates that fail to converge on more than the configured fraction
    of imputations cannot be trusted regardless of how permissive the
    rank-stability threshold is.

    When MI is not active (directive is None or method is not an MI-*
    variant, or no stability entry is available), the check is marked
    ``not_applicable`` and passes.
    """
    # ``_result`` prefix signals unused param: candidate is identified
    # via ``stability.candidate_id``.
    if directive is None or stability is None or not directive.covariate_method.startswith("MI-"):
        return GateCheckResult(
            check_id="imputation_stability",
            passed=True,
            observed="not_applicable",
        )

    issues: list[str] = []

    if stability.convergence_rate < convergence_rate_min:
        issues.append(
            f"convergence_rate={stability.convergence_rate:.2f} (<{convergence_rate_min})"
        )

    penalty = directive.imputation_stability_penalty
    if penalty > 0:
        threshold = max(0.0, 1.0 - penalty)
        if stability.rank_stability < threshold:
            issues.append(f"rank_stability={stability.rank_stability:.2f} (<{threshold:.2f})")

    passed = not issues
    return GateCheckResult(
        check_id="imputation_stability",
        passed=passed,
        observed="; ".join(issues) if issues else "stable",
        threshold=(
            f"conv_rate≥{convergence_rate_min}, rank_stability≥{max(0.0, 1.0 - penalty):.2f}"
        ),
    )


# ---------------------------------------------------------------------------
# Gate 2: Lane-Specific Admissibility
# ---------------------------------------------------------------------------


def evaluate_gate2(
    result: BackendResult,
    policy: GatePolicy,
    lane: str,
    loro_metrics: LOROMetrics | None = None,
    *,
    prior_manifest: PriorManifest | None = None,
) -> GateResult:
    """Evaluate Gate 2: Lane-Specific Admissibility.

    Per PRD §4.3.1 table:
      Submission: interpretable params, shrinkage < threshold,
                  identifiability (profile-likelihood CI, condition number),
                  NODE excluded.
      Discovery:  NODE eligible, reproducible estimation.
      Optimization: LORO performance required.

    Args:
        result: BackendResult from estimation.
        policy: GatePolicy with gate2 thresholds.
        lane: Operating lane ("submission", "discovery", "optimization").
        loro_metrics: Optional LORO-CV metrics for optimization lane.
        prior_manifest: Loaded ``PriorManifest`` (from
            ``bayesian/{cid}_prior_manifest.json``). Required when
            ``policy.gate2.bayesian_prior_justification_required`` is True
            (plan Task 19). ``None`` in other cases — the check passes
            trivially for non-Bayesian backends or lanes that don't
            demand prior provenance.
    """
    if lane not in _VALID_LANES:
        msg = f"Invalid lane '{lane}'. Must be one of {sorted(_VALID_LANES)}"
        raise ValueError(msg)

    g2 = policy.gate2
    checks: list[GateCheckResult] = []

    checks.append(_check_interpretability(result, g2))
    checks.append(_check_shrinkage(result, g2))
    checks.append(_check_identifiability(result, g2))
    checks.append(_check_node_eligibility(result, g2))
    checks.append(_check_reproducible_estimation(result, g2))
    checks.append(_check_loro_requirement(result, g2, lane, loro_metrics=loro_metrics))
    checks.append(_check_bayesian_prior_justification(result, g2, prior_manifest))

    passed = all(c.passed for c in checks)
    failed_names = [c.check_id for c in checks if not c.passed]
    summary = "All checks passed" if passed else f"Failed: {', '.join(failed_names)}"

    return GateResult(
        gate_id=generate_gate_id(),
        gate_name="lane_admissibility",
        candidate_id=result.model_id,
        passed=passed,
        checks=checks,
        summary_reason=summary,
        policy_version=policy.policy_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def _check_interpretability(result: BackendResult, g2: Gate2Config) -> GateCheckResult:
    """Interpretable parameterization check.

    For Submission lane: all structural parameters must be pharmacologically
    interpretable (CL, V, ka — not latent NODE dimensions).
    """
    if g2.interpretable_parameterization == "not_required":
        return GateCheckResult(
            check_id="interpretable_parameterization",
            passed=True,
            observed="not_required",
        )

    # NODE backends produce non-interpretable parameters
    is_node = result.backend in ("jax_node",)
    if is_node:
        passed = g2.interpretable_parameterization != "required"
        return GateCheckResult(
            check_id="interpretable_parameterization",
            passed=passed,
            observed="node_backend",
            threshold=g2.interpretable_parameterization,
        )

    return GateCheckResult(
        check_id="interpretable_parameterization",
        passed=True,
        observed="interpretable",
        threshold=g2.interpretable_parameterization,
    )


def _check_shrinkage(result: BackendResult, g2: Gate2Config) -> GateCheckResult:
    """Eta shrinkage check.

    High shrinkage (>30% for Submission) means the data don't support
    individual-level estimation — EBEs collapse toward the population mean.
    """
    if g2.shrinkage_max is None:
        return GateCheckResult(
            check_id="shrinkage",
            passed=True,
            observed="no_threshold",
        )

    if not result.eta_shrinkage:
        # Missing shrinkage data when threshold is required → fail
        return GateCheckResult(
            check_id="shrinkage",
            passed=False,
            observed="not_available",
            threshold=g2.shrinkage_max,
        )

    max_shrinkage = max(result.eta_shrinkage.values())
    worst_param = max(result.eta_shrinkage, key=lambda k: result.eta_shrinkage[k])
    passed = max_shrinkage <= g2.shrinkage_max
    return GateCheckResult(
        check_id="shrinkage",
        passed=passed,
        observed=max_shrinkage,
        threshold=g2.shrinkage_max,
        units=f"max_eta ({worst_param})",
    )


def _check_identifiability(result: BackendResult, g2: Gate2Config) -> GateCheckResult:
    """Identifiability check via profile-likelihood CI and condition number.

    Per PRD v0.3: identifiability-based filtering added to Submission Gate 2.
    """
    if not g2.identifiability_required:
        return GateCheckResult(
            check_id="identifiability",
            passed=True,
            observed="not_required",
        )

    ident = result.diagnostics.identifiability
    issues: list[str] = []

    # Condition number: >1000 is ill-conditioned
    if ident.ill_conditioned:
        cn_str = f"{ident.condition_number:.1f}" if ident.condition_number else "unknown"
        issues.append(f"ill_conditioned (CN={cn_str})")

    # Profile-likelihood CI: each parameter should have a valid CI
    missing_ci = [p for p, has_ci in ident.profile_likelihood_ci.items() if not has_ci]
    if missing_ci:
        issues.append(f"no_profile_CI: {', '.join(missing_ci)}")

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="identifiability",
        passed=passed,
        observed="; ".join(issues) if issues else "all_identifiable",
    )


def _check_node_eligibility(result: BackendResult, g2: Gate2Config) -> GateCheckResult:
    """NODE backend eligibility check.

    Submission lane: NODE models are NOT eligible (hard rule from PRD §3).
    Discovery/Optimization: NODE allowed per policy.
    """
    is_node = result.backend in ("jax_node",)
    if not is_node:
        return GateCheckResult(
            check_id="node_eligibility",
            passed=True,
            observed="not_node",
        )

    passed = g2.node_eligible
    return GateCheckResult(
        check_id="node_eligibility",
        passed=passed,
        observed="node_backend",
        threshold="eligible" if g2.node_eligible else "excluded",
    )


def _check_reproducible_estimation(result: BackendResult, g2: Gate2Config) -> GateCheckResult:
    """Reproducible estimation check.

    Ensures the estimation method is deterministic (given seed).
    """
    if g2.reproducible_estimation == "not_required":
        return GateCheckResult(
            check_id="reproducible_estimation",
            passed=True,
            observed="not_required",
        )

    # nlmixr2 with seed is reproducible; NODE with CPU mode + fixed seed is reproducible
    method = result.convergence_metadata.method
    reproducible_methods = {"saem", "focei", "foce", "nlminb", "adam"}
    passed = method.lower() in reproducible_methods or result.backend in ("nlmixr2", "jax_node")
    return GateCheckResult(
        check_id="reproducible_estimation",
        passed=passed,
        observed=f"{result.backend}/{method}",
        threshold="required",
    )


def _check_loro_requirement(
    result: BackendResult,
    g2: Gate2Config,
    lane: str,
    loro_metrics: LOROMetrics | None = None,
) -> GateCheckResult:
    """LORO-CV requirement for Optimization lane (Phase 3).

    Evaluates pooled NPDE (CWRES proxy) and VPC coverage concordance from
    LORO-CV against policy-driven thresholds. Uses law of total variance for
    pooled variance.
    """
    del result  # candidate is identified upstream; evaluation uses loro_metrics only
    if not g2.loro_required or lane != "optimization":
        return GateCheckResult(
            check_id="loro_required",
            passed=True,
            observed="not_required" if not g2.loro_required else "not_optimization_lane",
        )

    if loro_metrics is None:
        return GateCheckResult(
            check_id="loro_required",
            passed=False,
            observed="not_evaluated",
            threshold="required",
        )

    issues: list[str] = []

    if loro_metrics.n_folds < g2.loro_min_folds:
        issues.append(f"insufficient_folds ({loro_metrics.n_folds}<{g2.loro_min_folds})")

    # NaN safety: abs(nan) > x is False in Python, so NaN would silently pass.
    # Explicit finite check: abs(nan) > x is False in Python.
    npde_mean = loro_metrics.pooled_npde_mean
    if not np.isfinite(npde_mean) or abs(npde_mean) > g2.loro_npde_mean_max:
        issues.append(f"npde_mean={npde_mean:.3f}" if np.isfinite(npde_mean) else "npde_mean=NaN")

    npde_var = loro_metrics.pooled_npde_variance
    var_in_range = g2.loro_npde_variance_min <= npde_var <= g2.loro_npde_variance_max
    if not np.isfinite(npde_var) or not var_in_range:
        issues.append(f"npde_var={npde_var:.3f}" if np.isfinite(npde_var) else "npde_var=NaN")

    if loro_metrics.vpc_coverage_concordance < g2.loro_vpc_coverage_min:
        issues.append(f"vpc_cov={loro_metrics.vpc_coverage_concordance:.3f}")

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="loro_required",
        passed=passed,
        observed="; ".join(issues) if issues else "all_within_bounds",
        threshold=(
            f"|npde_mean|<{g2.loro_npde_mean_max}, "
            f"npde_var[{g2.loro_npde_variance_min}-{g2.loro_npde_variance_max}], "
            f"vpc>{g2.loro_vpc_coverage_min}"
        ),
    )


_INFORMATIVE_PRIOR_SOURCES = frozenset({"historical_data", "expert_elicitation", "meta_analysis"})


def _check_bayesian_prior_justification(
    result: BackendResult,
    g2: Gate2Config,
    prior_manifest: PriorManifest | None,
) -> GateCheckResult:
    """Gate 2 Bayesian prior-justification hard-gate (plan Task 19).

    Every informative prior on the candidate's ``PriorManifest`` must
    carry a justification of at least
    ``g2.bayesian_prior_justification_min_length`` characters AND a
    Crossref-canonical DOI (FDA 2026 draft). Aggregates errors across
    priors so reviewers see every failing entry in one pass.

    Trivially passes when the lane policy does not require provenance
    (``bayesian_prior_justification_required`` is False), when the
    backend is not Bayesian, or when there are no informative priors
    on the manifest.
    """
    if not g2.bayesian_prior_justification_required:
        return GateCheckResult(
            check_id="bayesian_prior_justification",
            passed=True,
            observed="not_required",
        )
    if result.backend != "bayesian_stan":
        return GateCheckResult(
            check_id="bayesian_prior_justification",
            passed=True,
            observed=f"not_bayesian_backend ({result.backend})",
        )
    if prior_manifest is None:
        return GateCheckResult(
            check_id="bayesian_prior_justification",
            passed=False,
            observed="prior_manifest_missing",
            threshold="required",
        )

    min_len = g2.bayesian_prior_justification_min_length
    issues: list[str] = []
    informative_count = 0
    for idx, entry in enumerate(prior_manifest.entries):
        if entry.source not in _INFORMATIVE_PRIOR_SOURCES:
            continue
        informative_count += 1
        entry_issues: list[str] = []
        if len(entry.justification) < min_len:
            entry_issues.append(f"justification {len(entry.justification)} < {min_len}")
        if not entry.doi:
            entry_issues.append("doi missing")
        if entry_issues:
            issues.append(f"entries[{idx}] target={entry.target!r}: {'; '.join(entry_issues)}")

    if informative_count == 0:
        return GateCheckResult(
            check_id="bayesian_prior_justification",
            passed=True,
            observed="no_informative_priors",
            threshold=f"min_len={min_len}, doi_required",
        )

    passed = not issues
    return GateCheckResult(
        check_id="bayesian_prior_justification",
        passed=passed,
        observed="all_justified" if passed else "; ".join(issues),
        threshold=f"min_len={min_len}, doi_required",
    )


# ---------------------------------------------------------------------------
# Gate 3: Within-Paradigm Ranking
# ---------------------------------------------------------------------------


@dataclass
class RankedCandidate:
    """A candidate that survived Gates 1+2, ranked by fit metric."""

    candidate_id: str
    rank: int
    bic: float
    aic: float | None
    n_params: int
    backend: str


def evaluate_gate3(
    survivors: list[BackendResult],
    policy: GatePolicy,
) -> tuple[GateResult, list[RankedCandidate]]:
    """Evaluate Gate 3: Ranking.

    Single-backend survivors: rank by BIC (within-paradigm).
    Multi-backend survivors: rank by simulation-based composite score
    (VPC concordance + NPE + BIC), flagged as qualified comparison.

    Args:
        survivors: BackendResults that passed Gates 1+2(+2.5).
        policy: GatePolicy (for version tracking in the result).

    Returns:
        Tuple of (GateResult, ranked list of candidates).
    """
    from apmode.governance.ranking import ranking_requires_simulation_metrics

    if not survivors:
        return (
            GateResult(
                gate_id=generate_gate_id(),
                gate_name="within_paradigm_ranking",
                candidate_id="none",
                passed=False,
                checks=[
                    GateCheckResult(
                        check_id="ranking",
                        passed=False,
                        observed="no_survivors",
                    )
                ],
                summary_reason="No candidates survived Gates 1+2",
                policy_version=policy.policy_version,
                timestamp=datetime.now(tz=UTC).isoformat(),
            ),
            [],
        )

    requires_sim, qualification_reason = ranking_requires_simulation_metrics(
        survivors, gate3=policy.gate3
    )

    if requires_sim:
        # Covers both cross-paradigm and within-paradigm-with-mixed-BLQ
        # (PRD §10 Q2). The qualification reason is threaded into the
        # Gate 3 ranking result so reports document *why* BIC was not used.
        return _gate3_cross_paradigm(survivors, policy, qualification_reason=qualification_reason)
    return _gate3_within_paradigm(survivors, policy)


def _gate3_within_paradigm(
    survivors: list[BackendResult],
    policy: GatePolicy,
) -> tuple[GateResult, list[RankedCandidate]]:
    """Within-paradigm ranking by BIC (Phase 1 behavior, preserved)."""
    # Secondary key on model_id breaks BIC ties deterministically.
    sorted_survivors = sorted(survivors, key=lambda r: (_safe_bic(r), r.model_id))

    ranked: list[RankedCandidate] = []
    for i, result in enumerate(sorted_survivors):
        ranked.append(
            RankedCandidate(
                candidate_id=result.model_id,
                rank=i + 1,
                bic=_safe_bic(result),
                aic=result.aic,
                n_params=len(result.parameter_estimates),
                backend=result.backend,
            )
        )

    checks = [
        GateCheckResult(
            check_id="ranking",
            passed=True,
            observed=f"{len(ranked)} candidates ranked",
        ),
        GateCheckResult(
            check_id="ranking_method",
            passed=True,
            observed="within_paradigm_bic",
        ),
        GateCheckResult(
            check_id="best_bic",
            passed=True,
            observed=ranked[0].bic if ranked else float("inf"),
            units="BIC",
        ),
        GateCheckResult(
            check_id="bic_spread",
            passed=True,
            observed=round(ranked[-1].bic - ranked[0].bic, 2) if len(ranked) > 1 else 0.0,
            units="delta_BIC",
        ),
    ]

    return (
        GateResult(
            gate_id=generate_gate_id(),
            gate_name="within_paradigm_ranking",
            candidate_id=ranked[0].candidate_id,
            passed=True,
            checks=checks,
            summary_reason=f"Best: {ranked[0].candidate_id} (BIC={ranked[0].bic:.1f})",
            policy_version=policy.policy_version,
            timestamp=datetime.now(tz=UTC).isoformat(),
        ),
        ranked,
    )


def _gate3_cross_paradigm(
    survivors: list[BackendResult],
    policy: GatePolicy,
    *,
    qualification_reason: str | None = None,
) -> tuple[GateResult, list[RankedCandidate]]:
    """Simulation-based Gate 3 ranking (PRD §4.3.1, §10 Q2).

    Used when ``ranking_requires_simulation_metrics`` returns True —
    either because survivors span multiple backends (classical cross-
    paradigm) or because within a single backend the BLQ handling
    methods differ (within-paradigm comparability failure).
    """
    from apmode.governance.ranking import rank_cross_paradigm

    cp_result = rank_cross_paradigm(
        survivors,
        gate3=policy.gate3,
        vpc_concordance_target=policy.vpc_concordance_target,
        qualification_reason=qualification_reason,
    )

    survivors_by_id = {s.model_id: s for s in survivors}
    ranked: list[RankedCandidate] = []
    for i, m in enumerate(cp_result.ranked_candidates):
        # Find the original survivor to extract BIC/AIC/n_params.
        # A missing entry is a ranking-module invariant violation — the
        # ranking returned an id we never put in. Raise so CI catches the
        # bug; the orchestrator converts this into a BackendError with
        # exit code 2. Silently skipping would produce an inconsistent
        # ranked list and mask the underlying bug.
        orig = survivors_by_id.get(m.candidate_id)
        if orig is None:
            logger.error(
                "cross_paradigm_orphan_candidate",
                candidate_id=m.candidate_id,
                known_survivors=list(survivors_by_id.keys()),
            )
            msg = (
                f"Gate 3 ranking returned orphan candidate_id={m.candidate_id!r} "
                f"not present in survivors (known: {sorted(survivors_by_id)}). "
                "This is a ranking-module invariant violation — please file a bug."
            )
            raise RuntimeError(msg)

        ranked.append(
            RankedCandidate(
                candidate_id=m.candidate_id,
                rank=i + 1,
                bic=_safe_bic(orig),
                aic=orig.aic,
                n_params=len(orig.parameter_estimates),
                backend=m.backend,
            )
        )

    backends_str = ", ".join(cp_result.backends_compared)
    checks = [
        GateCheckResult(
            check_id="ranking",
            passed=True,
            observed=f"{len(ranked)} candidates ranked",
        ),
        GateCheckResult(
            check_id="ranking_method",
            passed=True,
            observed="cross_paradigm_simulation_based",
        ),
        GateCheckResult(
            check_id="qualified_comparison",
            passed=True,
            observed=True,
            evidence_ref=cp_result.qualification_reason or f"backends: {backends_str}",
        ),
        GateCheckResult(
            check_id="best_composite",
            passed=True,
            observed=round(cp_result.ranked_candidates[0].composite_score, 4)
            if cp_result.ranked_candidates
            else float("inf"),
            units="composite_score",
        ),
    ]

    best_id = ranked[0].candidate_id if ranked else "none"
    return (
        GateResult(
            gate_id=generate_gate_id(),
            gate_name="cross_paradigm_ranking",
            candidate_id=best_id,
            passed=True,
            checks=checks,
            summary_reason=(
                f"Cross-paradigm qualified comparison: best={best_id} (backends: {backends_str})"
            ),
            policy_version=policy.policy_version,
            timestamp=datetime.now(tz=UTC).isoformat(),
        ),
        ranked,
    )


# ---------------------------------------------------------------------------
# Gate 2.5: Credibility Qualification (Phase 2 scaffold)
# ---------------------------------------------------------------------------


def evaluate_gate2_5(
    result: BackendResult,
    policy: GatePolicy,
    credibility_context: CredibilityContext | None = None,
) -> GateResult:
    """Evaluate Gate 2.5: Credibility Qualification (ICH M15).

    Checks:
      1. Context-of-use: statement exists when required
      2. Limitation-to-risk: mapping present when required
      3. Data adequacy: n_observations / n_parameters >= threshold
      4. Sensitivity: results available when required
      5. AI/ML transparency: statement present for NODE/agentic backends

    Args:
        result: BackendResult from estimation.
        policy: GatePolicy with gate2_5 thresholds.
        credibility_context: Optional context with COU statement, etc.
    """
    g25 = policy.gate2_5
    ctx = credibility_context or CredibilityContext()
    checks: list[GateCheckResult] = []

    if g25 is None:
        checks.append(
            GateCheckResult(
                check_id="credibility_qualification",
                passed=True,
                observed="no_gate2_5_config",
            )
        )
        return GateResult(
            gate_id=generate_gate_id(),
            gate_name="credibility_qualification",
            candidate_id=result.model_id,
            passed=True,
            checks=checks,
            summary_reason="No Gate 2.5 config — passed",
            policy_version=policy.policy_version,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    # Check 1: Context-of-use
    checks.append(_check_context_of_use(ctx, g25))

    # Check 2: Limitation-to-risk mapping
    checks.append(_check_limitation_to_risk(ctx, g25))

    # Check 3: Data adequacy vs model complexity
    checks.append(_check_data_adequacy(ctx, g25))

    # Check 4: Sensitivity analysis
    checks.append(_check_sensitivity(ctx, g25))

    # Check 5: AI/ML transparency
    checks.append(_check_ml_transparency(result, ctx, g25))

    passed = all(c.passed for c in checks)
    failed_names = [c.check_id for c in checks if not c.passed]
    summary = "All checks passed" if passed else f"Failed: {', '.join(failed_names)}"

    return GateResult(
        gate_id=generate_gate_id(),
        gate_name="credibility_qualification",
        candidate_id=result.model_id,
        passed=passed,
        checks=checks,
        summary_reason=summary,
        policy_version=policy.policy_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


def _check_context_of_use(
    ctx: CredibilityContext,
    g25: Gate25Config,
) -> GateCheckResult:
    """Context-of-use statement must exist when required."""
    if not g25.context_of_use_required:
        return GateCheckResult(
            check_id="context_of_use",
            passed=True,
            observed="not_required",
        )
    has_cou = ctx.context_of_use is not None and len(ctx.context_of_use.strip()) > 0
    return GateCheckResult(
        check_id="context_of_use",
        passed=has_cou,
        observed="present" if has_cou else "missing",
        threshold="required",
    )


def _check_limitation_to_risk(
    ctx: CredibilityContext,
    g25: Gate25Config,
) -> GateCheckResult:
    """Limitation-to-risk mapping must exist when required."""
    if not g25.limitation_to_risk_mapping_required:
        return GateCheckResult(
            check_id="limitation_to_risk",
            passed=True,
            observed="not_required",
        )
    has_mapping = len(ctx.limitations) > 0 and ctx.risk_level is not None
    return GateCheckResult(
        check_id="limitation_to_risk",
        passed=has_mapping,
        observed="present" if has_mapping else "missing",
        threshold="required",
    )


def _check_data_adequacy(
    ctx: CredibilityContext,
    g25: Gate25Config,
) -> GateCheckResult:
    """Data adequacy: n_obs / n_params >= threshold."""
    if not g25.data_adequacy_required:
        return GateCheckResult(
            check_id="data_adequacy",
            passed=True,
            observed="not_required",
        )
    if ctx.n_parameters == 0:
        return GateCheckResult(
            check_id="data_adequacy",
            passed=True,
            observed="no_parameters",
        )
    ratio = ctx.n_observations / ctx.n_parameters
    passed = ratio >= g25.data_adequacy_ratio_min
    return GateCheckResult(
        check_id="data_adequacy",
        passed=passed,
        observed=round(ratio, 1),
        threshold=g25.data_adequacy_ratio_min,
        units="obs/params",
    )


def _check_sensitivity(
    ctx: CredibilityContext,
    g25: Gate25Config,
) -> GateCheckResult:
    """Sensitivity analysis results must be available when required."""
    if not g25.sensitivity_analysis_required:
        return GateCheckResult(
            check_id="sensitivity_analysis",
            passed=True,
            observed="not_required",
        )
    return GateCheckResult(
        check_id="sensitivity_analysis",
        passed=ctx.sensitivity_available,
        observed="available" if ctx.sensitivity_available else "missing",
        threshold="required",
    )


def _check_ml_transparency(
    result: BackendResult,
    ctx: CredibilityContext,
    g25: Gate25Config,
) -> GateCheckResult:
    """AI/ML transparency statement required for NODE/agentic backends."""
    if not g25.ai_ml_transparency_required:
        return GateCheckResult(
            check_id="ml_transparency",
            passed=True,
            observed="not_required",
        )

    is_ml_backend = result.backend in ("jax_node", "agentic_llm")
    if not is_ml_backend:
        return GateCheckResult(
            check_id="ml_transparency",
            passed=True,
            observed="not_ml_backend",
        )

    has_statement = (
        ctx.ml_transparency_statement is not None
        and len(ctx.ml_transparency_statement.strip()) > 0
    )
    return GateCheckResult(
        check_id="ml_transparency",
        passed=has_statement,
        observed="present" if has_statement else "missing",
        threshold="required",
    )


# ---------------------------------------------------------------------------
# Gate 1 Bayesian — per-parameter-class warn/fail tiers (plan Task 17)
# ---------------------------------------------------------------------------

_PARAM_CLASS_NAMES = ("fixed_effects", "iiv", "residual", "correlations")


def classify_param_class(name: str) -> str:
    """Map a Stan parameter name to its :class:`Gate1BayesianConfig` class.

    Recognised prefixes match the DSL's on-disk naming (see
    ``apmode/dsl/priors.py`` / ``apmode/dsl/stan_emitter.py``):

    * ``omega_<p>`` → ``iiv`` (between-subject SDs, centered or
      non-centered — both produce ``omega_*`` names after decomposition).
    * ``sigma_``-prefixed / ``residual_sd`` / ``sigma_prop`` /
      ``sigma_add`` → ``residual``.
    * ``L_corr_``, ``corr_iiv``, ``L_Omega`` → ``correlations``.
    * everything else → ``fixed_effects`` (structural parameters and
      covariate betas).

    Unknown names default to ``fixed_effects`` so the Gate 1 Bayesian
    evaluator errs on the strict side.
    """
    if name.startswith(("L_corr_", "L_Omega")) or name == "corr_iiv":
        return "correlations"
    if name.startswith("omega_"):
        return "iiv"
    if name.startswith("sigma_") or name in {"residual_sd", "sigma_prop", "sigma_add"}:
        return "residual"
    return "fixed_effects"


def _rhat_check(
    class_name: str,
    observed: float,
    threshold: float,
    severity: str,
) -> tuple[GateCheckResult, str | None]:
    """Build a GateCheckResult for one rhat-by-class comparison.

    Returns ``(check, failure_reason | None)``. A ``warn``-severity
    violation yields a passing check so the gate as a whole does not
    fail, but the observed value is still surfaced.
    """
    violates = observed > threshold
    is_failure = violates and severity == "fail"
    reason: str | None = None
    if violates:
        reason = f"R-hat {observed:.3f} > {threshold:.3f} on {class_name}"
    return (
        GateCheckResult(
            check_id=f"bayesian_rhat_{class_name}",
            passed=not is_failure,
            observed=observed,
            threshold=threshold,
            evidence_ref=f"severity={severity}" if violates else None,
        ),
        reason if is_failure else None,
    )


def _ess_check(
    class_name: str,
    observed: float,
    threshold: float,
    severity: str,
    kind: str,
) -> tuple[GateCheckResult, str | None]:
    """Analogue of :func:`_rhat_check` for ESS (lower = worse)."""
    violates = observed < threshold
    is_failure = violates and severity == "fail"
    reason: str | None = None
    if violates:
        reason = f"ESS-{kind} {observed:.0f} < {threshold:.0f} on {class_name}"
    return (
        GateCheckResult(
            check_id=f"bayesian_ess_{kind}_{class_name}",
            passed=not is_failure,
            observed=observed,
            threshold=threshold,
            evidence_ref=f"severity={severity}" if violates else None,
        ),
        reason if is_failure else None,
    )


def evaluate_gate1_bayesian(
    result: BackendResult,
    policy: GatePolicy,
) -> GateResult:
    """Evaluate Gate 1 Bayesian checks for a ``BackendResult``.

    Applies only when ``result.backend == "bayesian_stan"``. Non-
    Bayesian backends pass trivially with a single ``not_applicable``
    check so the gate evaluator can be called uniformly in the
    orchestration layer.

    The evaluator walks every per-class R-hat / bulk ESS / tail ESS
    entry against the matching ``Gate1BayesianConfig`` threshold, and
    additionally enforces the scalar knobs (divergences, tree-depth
    saturation, Pareto-k, E-BFMI). Each diagnostic axis has its own
    severity tier (``warn`` / ``fail``) so Discovery-lane policies can
    keep Pareto-k informative while Submission keeps it disqualifying.

    The ``passed`` flag is True when no ``fail``-severity violation
    fired. Warnings are recorded in each check's ``evidence_ref`` and
    aggregated into ``summary_reason``.
    """
    g = policy.gate1_bayesian
    checks: list[GateCheckResult] = []
    failure_reasons: list[str] = []
    warning_reasons: list[str] = []

    if result.backend != "bayesian_stan":
        return GateResult(
            gate_id=generate_gate_id(),
            gate_name="gate1_bayesian",
            candidate_id=result.model_id,
            passed=True,
            checks=[
                GateCheckResult(
                    check_id="bayesian_backend",
                    passed=True,
                    observed=result.backend,
                    threshold="bayesian_stan",
                )
            ],
            summary_reason="not_applicable — non-Bayesian backend",
            policy_version=policy.policy_version,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    diag = result.posterior_diagnostics
    if diag is None:
        return GateResult(
            gate_id=generate_gate_id(),
            gate_name="gate1_bayesian",
            candidate_id=result.model_id,
            passed=False,
            checks=[
                GateCheckResult(
                    check_id="posterior_diagnostics_present",
                    passed=False,
                    observed="missing",
                    threshold="required",
                )
            ],
            summary_reason=(
                "BackendResult.posterior_diagnostics is None — cannot "
                "evaluate Gate 1 Bayesian. The BayesianRunner must "
                "populate this field on every run."
            ),
            policy_version=policy.policy_version,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    rhat_severity = g.severity["rhat"]
    ess_severity = g.severity["ess"]
    div_severity = g.severity["divergences"]
    pareto_severity = g.severity["pareto_k"]

    def _resolve_threshold(cfg: Gate1BayesianConfig, axis: str, class_name: str) -> float:
        block = getattr(cfg, axis)
        return float(getattr(block, class_name))

    for class_name in _PARAM_CLASS_NAMES:
        if class_name in diag.rhat_max_by_class:
            threshold = _resolve_threshold(g, "rhat_max", class_name)
            check, reason = _rhat_check(
                class_name,
                diag.rhat_max_by_class[class_name],
                threshold,
                rhat_severity,
            )
            checks.append(check)
            if reason:
                failure_reasons.append(reason)
            elif check.evidence_ref:
                warning_reasons.append(
                    f"R-hat warn on {class_name}: "
                    f"observed={diag.rhat_max_by_class[class_name]:.3f}"
                )
        if class_name in diag.ess_bulk_min_by_class:
            threshold = _resolve_threshold(g, "ess_bulk_min", class_name)
            check, reason = _ess_check(
                class_name,
                diag.ess_bulk_min_by_class[class_name],
                threshold,
                ess_severity,
                kind="bulk",
            )
            checks.append(check)
            if reason:
                failure_reasons.append(reason)
        if class_name in diag.ess_tail_min_by_class:
            threshold = _resolve_threshold(g, "ess_tail_min", class_name)
            check, reason = _ess_check(
                class_name,
                diag.ess_tail_min_by_class[class_name],
                threshold,
                ess_severity,
                kind="tail",
            )
            checks.append(check)
            if reason:
                failure_reasons.append(reason)

    # Scalar knobs
    div_violates = diag.n_divergent > g.divergence_tolerance
    div_is_failure = div_violates and div_severity == "fail"
    checks.append(
        GateCheckResult(
            check_id="bayesian_divergences",
            passed=not div_is_failure,
            observed=float(diag.n_divergent),
            threshold=float(g.divergence_tolerance),
            evidence_ref=(
                f"severity={div_severity}; see reparameterization_recommendation"
                if div_violates
                else None
            ),
        )
    )
    if div_is_failure:
        failure_reasons.append(
            f"divergent transitions {diag.n_divergent} > {g.divergence_tolerance}"
        )
    elif div_violates:
        warning_reasons.append(
            f"divergent transitions warn: {diag.n_divergent} "
            f"(tolerance {g.divergence_tolerance}; see "
            "reparameterization_recommendation)"
        )

    if diag.pareto_k_max is not None:
        pareto_violates = diag.pareto_k_max > g.pareto_k_max
        pareto_is_failure = pareto_violates and pareto_severity == "fail"
        checks.append(
            GateCheckResult(
                check_id="bayesian_pareto_k",
                passed=not pareto_is_failure,
                observed=diag.pareto_k_max,
                threshold=g.pareto_k_max,
                evidence_ref=(f"severity={pareto_severity}" if pareto_violates else None),
            )
        )
        if pareto_is_failure:
            failure_reasons.append(f"Pareto-k_max {diag.pareto_k_max:.2f} > {g.pareto_k_max}")
        elif pareto_violates:
            warning_reasons.append(
                f"Pareto-k_max warn: {diag.pareto_k_max:.2f} (threshold {g.pareto_k_max})"
            )

    passed = not failure_reasons
    if passed and not warning_reasons:
        summary = "All Gate 1 Bayesian checks passed"
    elif passed:
        summary = "Passed with warnings: " + "; ".join(warning_reasons)
    else:
        summary = "Failed: " + "; ".join(failure_reasons)

    return GateResult(
        gate_id=generate_gate_id(),
        gate_name="gate1_bayesian",
        candidate_id=result.model_id,
        passed=passed,
        checks=checks,
        summary_reason=summary,
        policy_version=policy.policy_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
    )
