# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression coverage for unavailable numeric artifact fields."""

from __future__ import annotations

from apmode.bundle.models import (
    ImputationStabilityEntry,
    LassoSelectionResult,
    LOROMetrics,
    PosteriorDiagnostics,
)
from apmode.search.stability import rubin_pool


def test_loro_unavailable_metrics_round_trip_as_null() -> None:
    metrics = LOROMetrics(
        n_folds=1,
        n_total_test_subjects=1,
        pooled_npde_mean=None,
        pooled_npde_variance=None,
        vpc_coverage_concordance=0.0,
    )
    assert LOROMetrics.model_validate_json(metrics.model_dump_json()) == metrics


def test_rubin_infinite_dof_is_persisted_as_unavailable() -> None:
    pooled = rubin_pool([1.0, 1.0], [0.1, 0.1])
    assert pooled[-1] is None
    entry = ImputationStabilityEntry(
        candidate_id="candidate",
        convergence_rate=1.0,
        rank_stability=1.0,
        pooled_parameters={
            "CL": {
                "pooled_estimate": pooled[0],
                "within_var": pooled[1],
                "between_var": pooled[2],
                "total_var": pooled[3],
                "dof": pooled[4],
            }
        },
    )
    assert ImputationStabilityEntry.model_validate_json(entry.model_dump_json()) == entry


def test_unavailable_ebfmi_round_trips() -> None:
    diagnostics = PosteriorDiagnostics(
        rhat_max=1.0,
        ess_bulk_min=500,
        ess_tail_min=500,
        n_divergent=0,
        n_max_treedepth=0,
        ebfmi_min=None,
    )
    assert PosteriorDiagnostics.model_validate_json(diagnostics.model_dump_json()) == diagnostics


def test_empty_lasso_selection_has_no_nan_bic() -> None:
    result = LassoSelectionResult(bic=None, derivative_r_squared=0.0)
    assert LassoSelectionResult.model_validate_json(result.model_dump_json()) == result
