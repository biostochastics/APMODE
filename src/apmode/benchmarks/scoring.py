# SPDX-License-Identifier: GPL-2.0-or-later
"""Backend-agnostic benchmark scoring harness (PRD §5, §9).

Consumes run bundles (BackendResult), optional DSLSpec, and expert model
artifacts. Produces BenchmarkScore per case. Supports all three suite types:

  Suite A/A-ext: Structure recovery + parameter bias/coverage
  Suite B:       Perturbation resilience + dispatch correctness
  Suite C:       Expert comparison (NPE, PI calibration, fraction-beats-expert)
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import numpy as np

from apmode.benchmarks.models import (
    BenchmarkCase,
    BenchmarkScore,
    MetricValue,
    SuiteName,
    SuiteReport,
    SuiteSummary,
)

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, EvidenceManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec

# Smith (2000) FDA BE goalposts: a candidate is "bioequivalent per subject"
# when the GMR of its simulated exposure vs the NCA reference lies inside
# this interval for *both* AUC and Cmax.
_BE_GMR_LOWER: float = 0.80
_BE_GMR_UPPER: float = 1.25

__all__ = [
    "aggregate_suite",
    "compute_auc_cmax_be_score",
    "compute_fraction_beats_expert",
    "compute_npe",
    "compute_prediction_interval_calibration",
    "evaluate_expert_comparison",
    "is_nca_eligible_for_auc_cmax",
    "is_nca_eligible_per_subject",
    "score_case",
    "score_convergence",
    "score_parameter_bias",
    "score_parameter_coverage",
    "score_structure_recovery",
]


# ---------------------------------------------------------------------------
# Suite A: Structure + Parameter recovery scoring
# ---------------------------------------------------------------------------


def score_structure_recovery(
    case: BenchmarkCase,
    discovered_spec: DSLSpec | None = None,
) -> bool | None:
    """Check if the recovered model structure matches the expected ground truth.

    Compares the discovered DSLSpec's module types against the case's
    ExpectedStructure. Returns None if no spec or no expected structure
    is available (unscorable).
    """
    expected = case.expected_structure
    if expected is None:
        return None  # No structural assertion for this case

    if discovered_spec is None:
        return None  # No discovered spec available — unscorable

    # Compare absorption type
    if expected.absorption is not None:
        actual_abs = discovered_spec.absorption.type
        if actual_abs != expected.absorption:
            return False

    # Compare distribution type
    if expected.distribution is not None:
        actual_dist = discovered_spec.distribution.type
        if actual_dist != expected.distribution:
            return False

    # Compare elimination type
    if expected.elimination is not None:
        actual_elim = discovered_spec.elimination.type
        if actual_elim != expected.elimination:
            return False

    # Compare compartment count
    if expected.n_compartments is not None:
        dist_type = discovered_spec.distribution.type
        actual_cmt = {"OneCmt": 1, "TwoCmt": 2, "ThreeCmt": 3}.get(dist_type)
        if actual_cmt is not None and actual_cmt != expected.n_compartments:
            return False

    return True


def score_parameter_bias(
    case: BenchmarkCase,
    result: BackendResult,
) -> dict[str, float]:
    """Compute relative bias for each reference parameter.

    Missing reference params that are absent from estimates are flagged
    with bias = NaN (not silently skipped).

    A reference value of exactly ``0`` is ``NaN`` (unscorable) rather
    than falling back to ``abs(est)``. The old absolute-value fallback
    silently mixed unit-ful absolute error with unitless relative bias
    under the same dict key, which poisoned downstream ``max()``-style
    aggregation across parameters with different scales. Callers should
    treat a zero reference as "unscorable at this point" and rely on
    the missingness branch of the aggregator.
    """
    biases: dict[str, float] = {}
    for param_name, ref_value in case.reference_params.items():
        if param_name in result.parameter_estimates:
            est = result.parameter_estimates[param_name].estimate
            if ref_value != 0:
                biases[param_name] = abs(est - ref_value) / abs(ref_value)
            else:
                biases[param_name] = float("nan")  # ref==0 is unscorable relative bias
        else:
            biases[param_name] = float("nan")  # Missing estimate
    return biases


def score_parameter_coverage(
    case: BenchmarkCase,
    result: BackendResult,
) -> dict[str, bool | None]:
    """Check if 95% CI contains the true parameter value.

    Returns None for parameters where CI is unavailable (e.g., SAEM
    covariance step failure). Missing CI is unscorable, not a free pass —
    aggregators decide whether it counts as a failure.
    """
    coverage: dict[str, bool | None] = {}
    for param_name, ref_value in case.reference_params.items():
        if param_name in result.parameter_estimates:
            pe = result.parameter_estimates[param_name]
            if pe.ci95_lower is not None and pe.ci95_upper is not None:
                coverage[param_name] = pe.ci95_lower <= ref_value <= pe.ci95_upper
            else:
                coverage[param_name] = None  # CI unavailable — unscorable
        else:
            coverage[param_name] = None  # Missing estimate
    return coverage


# ---------------------------------------------------------------------------
# Suite B/C: Predictive performance scoring
# ---------------------------------------------------------------------------


def compute_npe(
    observed: np.ndarray,
    predicted_simulations: np.ndarray,
) -> float:
    """Compute nonparametric prediction error.

    NPE = median absolute prediction error from posterior predictive
    simulations on held-out subjects.

    Args:
        observed: (n_obs,) array of observed DV values.
        predicted_simulations: (n_sims, n_obs) array of simulated DV values.

    Returns:
        Median absolute prediction error (scalar).
    """
    if observed.shape[0] != predicted_simulations.shape[1]:
        msg = (
            f"Shape mismatch: observed ({observed.shape[0]}) vs "
            f"simulations ({predicted_simulations.shape[1]})"
        )
        raise ValueError(msg)
    point_pred = np.median(predicted_simulations, axis=0)
    abs_errors = np.abs(observed - point_pred)
    return float(np.median(abs_errors))


def compute_prediction_interval_calibration(
    observed: np.ndarray,
    predicted_simulations: np.ndarray,
    levels: tuple[float, ...] = (50.0, 80.0, 90.0, 95.0),
) -> dict[str, float]:
    """Compute empirical coverage of prediction intervals.

    For each nominal level (e.g., 90%), compute the fraction of
    observations falling within the central prediction interval.
    """
    calibration: dict[str, float] = {}
    for level in levels:
        alpha = (100.0 - level) / 2.0
        lower = np.percentile(predicted_simulations, alpha, axis=0)
        upper = np.percentile(predicted_simulations, 100.0 - alpha, axis=0)
        in_interval = (observed >= lower) & (observed <= upper)
        calibration[str(int(level))] = float(np.mean(in_interval))
    return calibration


def is_nca_eligible_for_auc_cmax(
    manifest: EvidenceManifest,
    *,
    max_blq_burden: float = 0.20,
) -> tuple[bool, str]:
    """Decide whether an observed-data NCA reference is admissible for Gate 3.

    AUC/Cmax bioequivalence scoring (``compute_auc_cmax_be_score``) compares
    each candidate's simulated exposure against a per-subject NCA reference
    derived from observed data. The NCA estimate is only trustworthy when:

    1. The absorption phase is adequately sampled (otherwise Cmax is biased
       by missing the peak).
    2. The elimination phase is adequately sampled (otherwise AUC(0-inf)
       extrapolation inflates error).
    3. BLQ burden is low enough that trapezoidal AUC is not censoring-biased.
       Default threshold 0.20 matches Thway 2018 recommendation; policies
       may tighten via ``Gate3Config.auc_cmax_nca_max_blq_burden``.

    Returns ``(eligible, reason)`` — the reason string is meant to be
    surfaced in the ranking audit trail when the metric is dropped.
    """
    if manifest.absorption_phase_coverage != "adequate":
        return False, "absorption phase coverage inadequate — Cmax from NCA is biased"
    if manifest.elimination_phase_coverage != "adequate":
        return False, "elimination phase coverage inadequate — AUC extrapolation is biased"
    if manifest.blq_burden > max_blq_burden:
        return False, (
            f"BLQ burden {manifest.blq_burden:.2%} > "
            f"auc_cmax_nca_max_blq_burden {max_blq_burden:.2%}: "
            "trapezoidal NCA AUC is censoring-biased (Thway 2018)"
        )
    return True, "observed-data NCA admissible"


def is_nca_eligible_per_subject(
    diagnostic: NCASubjectDiagnostic,
) -> tuple[bool, str]:
    """Per-subject NCA eligibility for AUC/Cmax bioequivalence scoring.

    Unlike :func:`is_nca_eligible_for_auc_cmax` (pooled across the whole
    manifest), this checks a single subject's observed-data NCA QC record.
    The eligibility signal is the QC verdict already encoded during NCA
    profiling (``NCASubjectDiagnostic.excluded``) — adj_R² on λz,
    AUC extrapolation fraction, span ratio, and minimum λz point count
    are all rolled into that bool by the profiler. Reusing it here keeps
    per-subject eligibility consistent with the initial-estimates path
    and avoids duplicating QC logic.

    Returns ``(eligible, reason)`` mirroring the pooled sibling so both
    can feed the audit trail. Ineligible subjects are mask-dropped (not
    counted as BE-fail) by :func:`compute_auc_cmax_be_score` when an
    eligibility mask is supplied — the QC is observed-data-only so
    candidates cannot influence the mask and there is no dodge surface.
    """
    if diagnostic.excluded:
        return False, diagnostic.excluded_reason or "subject excluded by NCA QC"
    return True, "per-subject observed-data NCA admissible"


def compute_auc_cmax_be_score(
    candidate_auc_per_subject: np.ndarray,
    candidate_cmax_per_subject: np.ndarray,
    nca_auc_per_subject: np.ndarray,
    nca_cmax_per_subject: np.ndarray,
    *,
    eligible_mask: np.ndarray | None = None,
    min_eligible: int | None = None,
    min_eligible_fraction: float | None = None,
) -> float | None:
    """Fraction of eligible subjects whose candidate-vs-NCA GMRs are bioequivalent.

    A subject counts as "BE-pass" when both its AUC and Cmax geometric mean
    ratios (candidate ÷ NCA) fall inside the Smith 2000 FDA goalposts
    ``[0.80, 1.25]``. The returned score is the mean pass rate across
    *eligible* subjects — a fraction in ``[0, 1]`` with useful dynamic range.

    All inputs are per-subject vectors (shape ``(n_subjects,)``). NaN or
    non-positive AUC/Cmax values on either side disqualify that subject
    within the BE check (contribute a 0 to the numerator) — this is
    separate from NCA eligibility, which is mask-controlled.

    **Eligibility (optional).** When ``eligible_mask`` is supplied, only
    subjects with ``eligible_mask[i] == True`` contribute to the score:

      * Numerator: count of BE-pass *AND* eligible subjects.
      * Denominator: count of eligible subjects (mask-drop, not BE-fail).

    Observed-data NCA QC (per subject) is the intended mask source —
    candidate-independent, so there is no dodge surface. When
    ``min_eligible`` is provided, the function returns ``None`` if fewer
    eligible subjects survive; ``min_eligible_fraction`` applies the
    same rule to the eligible fraction of the total cohort. Both floors
    are AND-combined so a policy can require both an absolute count and
    a representative fraction. Returning ``None`` triggers the uniform-
    drop path in :func:`apmode.governance.ranking.rank_cross_paradigm`.

    Empty cohort (``n == 0``) returns ``None`` — an empty cohort is
    semantically "undefined / no data," not "BE-failed." Callers that
    want "BE-fail" semantics should check for ``None`` explicitly. The
    uniform-drop rule in
    :func:`apmode.governance.ranking._apply_uniform_auc_cmax_drop`
    already treats ``None`` correctly.

    Raises ``ValueError`` when array lengths do not match or when the
    mask is supplied with a mismatched length.
    """
    n = candidate_auc_per_subject.shape[0]
    if not (
        candidate_cmax_per_subject.shape[0] == n
        and nca_auc_per_subject.shape[0] == n
        and nca_cmax_per_subject.shape[0] == n
    ):
        shapes = (
            candidate_auc_per_subject.shape[0],
            candidate_cmax_per_subject.shape[0],
            nca_auc_per_subject.shape[0],
            nca_cmax_per_subject.shape[0],
        )
        msg = f"Per-subject AUC/Cmax arrays must share length; got {shapes}"
        raise ValueError(msg)
    if eligible_mask is not None and eligible_mask.shape[0] != n:
        msg = (
            f"eligible_mask length {eligible_mask.shape[0]} must match "
            f"per-subject array length {n}"
        )
        raise ValueError(msg)
    if n == 0:
        # Empty cohort is undefined, not BE-failed. Return None so the
        # uniform-drop rule in rank_cross_paradigm drops the component
        # cleanly. Callers that need "BE-fail" semantics check for None.
        return None

    # Subjects with any non-finite or non-positive component are treated as
    # BE-fail within the BE check (separate concept from NCA eligibility).
    valid = (
        np.isfinite(candidate_auc_per_subject)
        & np.isfinite(candidate_cmax_per_subject)
        & np.isfinite(nca_auc_per_subject)
        & np.isfinite(nca_cmax_per_subject)
        & (candidate_auc_per_subject > 0)
        & (candidate_cmax_per_subject > 0)
        & (nca_auc_per_subject > 0)
        & (nca_cmax_per_subject > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        auc_gmr = candidate_auc_per_subject / nca_auc_per_subject
        cmax_gmr = candidate_cmax_per_subject / nca_cmax_per_subject
    auc_pass = (auc_gmr >= _BE_GMR_LOWER) & (auc_gmr <= _BE_GMR_UPPER)
    cmax_pass = (cmax_gmr >= _BE_GMR_LOWER) & (cmax_gmr <= _BE_GMR_UPPER)
    per_subject_pass = valid & auc_pass & cmax_pass

    if eligible_mask is None:
        # Legacy path: mean over all subjects. Floors-without-mask would be
        # ambiguous (no eligibility signal to compare against), so they are
        # a no-op here.
        return float(np.mean(per_subject_pass))

    eligible_bool = eligible_mask.astype(bool)
    n_eligible = int(eligible_bool.sum())
    if min_eligible is not None and n_eligible < min_eligible:
        return None
    if min_eligible_fraction is not None and (n_eligible / n) < min_eligible_fraction:
        return None
    if n_eligible == 0:
        return None
    wins = int((per_subject_pass & eligible_bool).sum())
    return wins / n_eligible


# ---------------------------------------------------------------------------
# Suite C: Expert comparison scoring
# ---------------------------------------------------------------------------


def evaluate_expert_comparison(
    apmode_npe: float,
    expert_npes: list[float],
    win_margin: float = 0.02,
) -> tuple[bool | None, float | None, float | None]:
    """Evaluate whether APMODE beats the median expert.

    Returns (beats_median, expert_median, npe_gap).
    Returns (None, None, None) if expert list is empty — the comparison
    is unscorable without a panel (published anchors don't count as panel).
    """
    if not expert_npes:
        return None, None, None

    expert_median = statistics.median(expert_npes)
    threshold = expert_median * (1 - win_margin)
    beats_median = apmode_npe <= threshold
    npe_gap = apmode_npe - expert_median
    return beats_median, expert_median, npe_gap


def compute_fraction_beats_expert(
    case_results: list[tuple[float, list[float]]],
    win_margin: float = 0.02,
) -> float:
    """Compute fraction of datasets where APMODE beats median expert.

    Skips cases where expert list is empty (unscorable).
    """
    if not case_results:
        return 0.0
    scorable = [
        (a, e)
        for a, e in case_results
        if e  # Skip empty expert lists
    ]
    if not scorable:
        return 0.0
    wins = sum(
        1
        for apmode_npe, expert_npes in scorable
        if evaluate_expert_comparison(apmode_npe, expert_npes, win_margin)[0]
    )
    return wins / len(scorable)


# ---------------------------------------------------------------------------
# Suite B: Dispatch assertion scoring
# ---------------------------------------------------------------------------


def score_dispatch_assertions(
    case: BenchmarkCase,
    dispatched_backends: list[str],
) -> bool | None:
    """Check whether dispatch decisions match expected assertions.

    Returns True if all includes are present and all excludes are absent.
    Returns None if no dispatch assertions are declared.
    """
    if not case.expected_dispatch_includes and not case.expected_dispatch_excludes:
        return None  # No dispatch assertions

    if not all(r in dispatched_backends for r in case.expected_dispatch_includes):
        return False
    return all(e not in dispatched_backends for e in case.expected_dispatch_excludes)


# ---------------------------------------------------------------------------
# Convergence and efficiency metrics
# ---------------------------------------------------------------------------


def score_convergence(
    results: list[BackendResult],
) -> tuple[float, dict[str, int]]:
    """Compute convergence rate and failure taxonomy."""
    if not results:
        return 0.0, {}

    converged = sum(1 for r in results if r.converged)
    rate = converged / len(results)

    failures: dict[str, int] = {}
    for r in results:
        if not r.converged:
            status = r.convergence_metadata.minimization_status
            failures[status] = failures.get(status, 0) + 1

    return rate, failures


# ---------------------------------------------------------------------------
# Full case scoring
# ---------------------------------------------------------------------------


def score_case(
    case: BenchmarkCase,
    result: BackendResult,
    all_candidate_results: list[BackendResult] | None = None,
    discovered_spec: DSLSpec | None = None,
    dispatched_backends: list[str] | None = None,
) -> BenchmarkScore:
    """Score a single benchmark case given the selected result.

    Args:
        case: The benchmark case definition.
        result: The selected (best) BackendResult.
        all_candidate_results: All candidate results for convergence stats.
        discovered_spec: The DSLSpec of the selected candidate (Suite A).
        dispatched_backends: Backends dispatched by the Lane Router (Suite B).
    """
    metrics: list[MetricValue] = []

    # --- Structure recovery (Suite A) ---
    structure_ok = score_structure_recovery(case, discovered_spec)
    if structure_ok is not None:
        metrics.append(
            MetricValue(
                name="structure_recovered",
                value=1.0 if structure_ok else 0.0,
                passed=structure_ok,
            )
        )

    # --- Parameter bias (Suite A) ---
    param_bias = score_parameter_bias(case, result)
    param_coverage = score_parameter_coverage(case, result)

    if param_bias:
        finite_biases = [v for v in param_bias.values() if not np.isnan(v)]
        max_bias = max(finite_biases) if finite_biases else 0.0
        has_missing = any(np.isnan(v) for v in param_bias.values())
        metrics.append(
            MetricValue(
                name="max_param_bias",
                value=max_bias,
                passed=(max_bias <= case.param_bias_tolerance) and not has_missing,
            )
        )

    if param_coverage:
        scored_coverage = [v for v in param_coverage.values() if v is not None]
        if scored_coverage:
            coverage_rate = sum(scored_coverage) / len(scored_coverage)
            metrics.append(
                MetricValue(
                    name="param_coverage_rate",
                    value=coverage_rate,
                    passed=coverage_rate >= 0.90,
                )
            )
        n_unscorable = sum(1 for v in param_coverage.values() if v is None)
        if n_unscorable > 0:
            metrics.append(
                MetricValue(
                    name="ci_unavailable_count",
                    value=float(n_unscorable),
                    passed=None,  # Informational
                )
            )

    # --- Dispatch assertions (Suite B) ---
    dispatch_correct: bool | None = None
    if dispatched_backends is not None:
        dispatch_correct = score_dispatch_assertions(case, dispatched_backends)
        if dispatch_correct is not None:
            metrics.append(
                MetricValue(
                    name="dispatch_correct",
                    value=1.0 if dispatch_correct else 0.0,
                    passed=dispatch_correct,
                )
            )

    # --- Convergence (all suites) ---
    all_results = all_candidate_results or [result]
    conv_rate, failure_classes = score_convergence(all_results)
    metrics.append(
        MetricValue(
            name="convergence_rate",
            value=conv_rate,
        )
    )

    # --- Gate passage ---
    gate1_passed = result.converged
    metrics.append(
        MetricValue(
            name="gate1_pass",
            value=1.0 if gate1_passed else 0.0,
            passed=gate1_passed,
        )
    )

    # --- Overall pass ---
    all_passed = all(m.passed for m in metrics if m.passed is not None)

    return BenchmarkScore(
        case_id=case.case_id,
        run_id=result.model_id,
        suite=case.suite,
        metrics=metrics,
        structure_recovered=structure_ok,
        param_bias=param_bias,
        param_coverage=param_coverage,
        dispatch_correct=dispatch_correct,
        wall_time_seconds=result.wall_time_seconds,
        candidates_evaluated=len(all_results),
        gate1_passed=gate1_passed,
        convergence_rate=conv_rate,
        failure_classes=failure_classes,
        overall_passed=all_passed,
    )


# ---------------------------------------------------------------------------
# Suite-level aggregation
# ---------------------------------------------------------------------------


def aggregate_suite(
    suite_name: SuiteName,
    scores: list[BenchmarkScore],
) -> SuiteReport:
    """Aggregate individual case scores into a suite report."""
    n_cases = len(scores)
    n_passed = sum(1 for s in scores if s.overall_passed)
    pass_rate = n_passed / n_cases if n_cases > 0 else 0.0

    wall_times = [s.wall_time_seconds for s in scores if s.wall_time_seconds is not None]
    mean_wall = statistics.mean(wall_times) if wall_times else None

    # Suite A specific
    structural_recovery_rate = None
    mean_param_bias = None
    if suite_name in ("A", "A_external"):
        recoveries = [s.structure_recovered for s in scores if s.structure_recovered is not None]
        if recoveries:
            structural_recovery_rate = sum(recoveries) / len(recoveries)
        all_biases = [v for s in scores for v in s.param_bias.values() if not np.isnan(v)]
        if all_biases:
            mean_param_bias = statistics.mean(all_biases)

    # Suite C specific
    fraction_beats = None
    if suite_name == "C":
        beats = [s.beats_median_expert for s in scores if s.beats_median_expert is not None]
        if beats:
            fraction_beats = sum(beats) / len(beats)

    return SuiteReport(
        suite=suite_name,
        scores=scores,
        summary=SuiteSummary(
            n_cases=n_cases,
            n_passed=n_passed,
            pass_rate=pass_rate,
            mean_wall_time_seconds=mean_wall,
            structural_recovery_rate=structural_recovery_rate,
            mean_param_bias=mean_param_bias,
            fraction_beats_expert=fraction_beats,
        ),
    )
