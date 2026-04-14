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
    from apmode.bundle.models import BackendResult
    from apmode.dsl.ast_models import DSLSpec

__all__ = [
    "aggregate_suite",
    "compute_fraction_beats_expert",
    "compute_npe",
    "compute_prediction_interval_calibration",
    "evaluate_expert_comparison",
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
    """
    biases: dict[str, float] = {}
    for param_name, ref_value in case.reference_params.items():
        if param_name in result.parameter_estimates:
            est = result.parameter_estimates[param_name].estimate
            if ref_value != 0:
                biases[param_name] = abs(est - ref_value) / abs(ref_value)
            else:
                biases[param_name] = abs(est)
        else:
            biases[param_name] = float("nan")  # Missing estimate
    return biases


def score_parameter_coverage(
    case: BenchmarkCase,
    result: BackendResult,
) -> dict[str, bool | None]:
    """Check if 95% CI contains the true parameter value.

    Returns None for parameters where CI is unavailable (e.g., SAEM
    covariance step failure). Per GPT-5.2-pro: missing CI is unscorable,
    not a free pass — aggregators decide whether it's a failure.
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
