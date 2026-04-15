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

from apmode.bundle.models import CredibilityContext, GateCheckResult, GateResult, LOROMetrics
from apmode.governance.policy import Gate25Config  # noqa: TC001 — used at runtime
from apmode.ids import generate_gate_id

logger = structlog.get_logger(__name__)

_VALID_LANES = frozenset({"submission", "discovery", "optimization"})

if TYPE_CHECKING:
    from apmode.bundle.models import (
        BackendResult,
        ImputationStabilityEntry,
        MissingDataDirective,
    )
    from apmode.governance.policy import Gate1Config, Gate2Config, GatePolicy


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
      5. VPC coverage: within [lower, upper] policy bounds
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
    checks.append(_check_parameter_plausibility(result))
    checks.append(_check_state_trajectory(result, g1))
    checks.extend(_check_cwres(result, g1))
    checks.append(_check_vpc_coverage(result, g1))
    checks.append(_check_split_integrity(result, g1))
    checks.append(_check_seed_stability(result, seed_results, g1))
    checks.append(_check_imputation_stability(result, stability, directive))

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


def _check_parameter_plausibility(result: BackendResult) -> GateCheckResult:
    """Check 2: Are structural parameters plausible?

    Checks: no negative volumes or clearances, no RSE > 200% (when available),
    no estimates at boundary (0.01 or 10000 — our sanity bounds).
    """
    # Sanity bounds for boundary detection
    lower_bound = 1e-4
    upper_bound = 1e5

    issues: list[str] = []
    for name, pe in result.parameter_estimates.items():
        if pe.category != "structural":
            continue
        # Non-positive structural parameters are pharmacologically implausible
        # (CL, V, ka must all be > 0)
        if pe.estimate <= 0:
            issues.append(f"{name}={pe.estimate:.4g} (non-positive)")
        # Boundary estimates indicate structural misspecification
        elif pe.estimate <= lower_bound:
            issues.append(f"{name}={pe.estimate:.4g} (at lower bound)")
        elif pe.estimate >= upper_bound:
            issues.append(f"{name}={pe.estimate:.4g} (at upper bound)")
        # RSE > 200% means effectively unidentifiable
        if pe.rse is not None and pe.rse > 200:
            issues.append(f"{name} RSE={pe.rse:.1f}% (>200%)")

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


def _check_vpc_coverage(result: BackendResult, g1: Gate1Config) -> GateCheckResult:
    """Check 5: VPC coverage within policy thresholds.

    Each percentile band (p5, p50, p95) coverage must be within
    [vpc_coverage_lower, vpc_coverage_upper].
    """
    vpc = result.diagnostics.vpc
    if vpc is None:
        # When the policy does not require a VPC (e.g. Phase 1 backends
        # that do not yet populate VPC), pass the check explicitly. When
        # required, missing evidence fails — missing evidence ≠ passing.
        if not g1.vpc_required:
            return GateCheckResult(
                check_id="vpc_coverage",
                passed=True,
                observed="vpc_not_configured",
            )
        return GateCheckResult(
            check_id="vpc_coverage",
            passed=False,
            observed="vpc_not_available",
        )

    violations: list[str] = []
    for band, coverage in vpc.coverage.items():
        if coverage < g1.vpc_coverage_lower:
            violations.append(f"{band}={coverage:.3f} (<{g1.vpc_coverage_lower})")
        if coverage > g1.vpc_coverage_upper:
            violations.append(f"{band}={coverage:.3f} (>{g1.vpc_coverage_upper})")

    passed = len(violations) == 0
    return GateCheckResult(
        check_id="vpc_coverage",
        passed=passed,
        observed="; ".join(violations) if violations else "all_within_bounds",
        threshold=f"[{g1.vpc_coverage_lower}, {g1.vpc_coverage_upper}]",
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
        # No split diagnostics available — cannot evaluate, pass
        return GateCheckResult(
            check_id="split_integrity",
            passed=True,
            observed="no_split_diagnostics",
        )

    issues: list[str] = []

    # Test CWRES mean should not drift far from train
    cwres_drift = abs(sgof.test_cwres_mean) - abs(sgof.train_cwres_mean)
    if cwres_drift > 0.5:
        issues.append(
            f"cwres_drift={cwres_drift:.3f} "
            f"(test={sgof.test_cwres_mean:.3f}, train={sgof.train_cwres_mean:.3f})"
        )

    # Test outlier fraction should not be >2x train + buffer
    outlier_ratio_threshold = 2.0 * sgof.train_outlier_fraction + 0.05
    if sgof.test_outlier_fraction > outlier_ratio_threshold:
        issues.append(
            f"test_outliers={sgof.test_outlier_fraction:.3f} (>{outlier_ratio_threshold:.3f})"
        )

    passed = len(issues) == 0
    return GateCheckResult(
        check_id="split_integrity",
        passed=passed,
        observed="; ".join(issues) if issues else "train_test_consistent",
        threshold="cwres_drift<=0.5, outliers<=2x+0.05",
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
        # Not enough seed runs — fail (missing evidence is not a pass)
        n_have = 1 + (len(seed_results) if seed_results else 0)
        return GateCheckResult(
            check_id="seed_stability",
            passed=False,
            observed=f"insufficient_seeds ({n_have}/{g1.seed_stability_n})",
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


_IMPUTATION_MIN_CONVERGENCE_RATE: float = 0.5
"""Candidates converging in <50% of imputations fail Gate 1 regardless of
penalty weight — this is a hard data-quality floor, not a tunable threshold.
"""


def _check_imputation_stability(
    result: BackendResult,
    stability: ImputationStabilityEntry | None,
    directive: MissingDataDirective | None,
) -> GateCheckResult:
    """Check 8: Imputation stability (MI runs only).

    The rank-stability threshold is driven by
    ``directive.imputation_stability_penalty``:

      pass_threshold = max(0.0, 1.0 - penalty)

    Convergence rate is hard-gated at
    ``_IMPUTATION_MIN_CONVERGENCE_RATE`` when stability data is present,
    independent of the penalty weight — candidates that fail to converge on
    more than half the imputations cannot be trusted regardless of how
    permissive the rank-stability threshold is.

    When MI is not active (directive is None or method is not an MI-*
    variant, or no stability entry is available), the check is marked
    ``not_applicable`` and passes.
    """
    del result  # candidate is identified via stability.candidate_id

    if directive is None or stability is None or not directive.covariate_method.startswith("MI-"):
        return GateCheckResult(
            check_id="imputation_stability",
            passed=True,
            observed="not_applicable",
        )

    issues: list[str] = []

    if stability.convergence_rate < _IMPUTATION_MIN_CONVERGENCE_RATE:
        issues.append(
            f"convergence_rate={stability.convergence_rate:.2f} "
            f"(<{_IMPUTATION_MIN_CONVERGENCE_RATE})"
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
            f"conv_rate≥{_IMPUTATION_MIN_CONVERGENCE_RATE}, "
            f"rank_stability≥{max(0.0, 1.0 - penalty):.2f}"
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

    requires_sim, qualification_reason = ranking_requires_simulation_metrics(survivors)

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

    def _safe_bic(r: BackendResult) -> float:
        if r.bic is None or not np.isfinite(r.bic):
            return float("inf")
        return r.bic

    sorted_survivors = sorted(survivors, key=_safe_bic)

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

    cp_result = rank_cross_paradigm(survivors, qualification_reason=qualification_reason)

    def _safe_bic(r: BackendResult) -> float:
        if r.bic is None or not np.isfinite(r.bic):
            return float("inf")
        return r.bic

    survivors_by_id = {s.model_id: s for s in survivors}
    ranked: list[RankedCandidate] = []
    for i, m in enumerate(cp_result.ranked_candidates):
        # Find the original survivor to extract BIC/AIC/n_params.
        # A missing entry is a ranking-module invariant violation, not a
        # recoverable state — log and skip rather than crash the bundle.
        orig = survivors_by_id.get(m.candidate_id)
        if orig is None:
            logger.error(
                "cross_paradigm_orphan_candidate",
                candidate_id=m.candidate_id,
                known_survivors=list(survivors_by_id.keys()),
            )
            continue

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
