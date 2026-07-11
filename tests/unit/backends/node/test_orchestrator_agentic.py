# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for orchestrator agentic backend integration."""

import pytest

from apmode.bundle.models import ImputationStabilityEntry, ImputationStabilityManifest
from apmode.orchestrator import _promote_mi_pooled_results
from apmode.routing import _LANE_BACKENDS
from apmode.search.engine import SearchOutcome, SearchResult
from tests._helpers.builders import make_backend_result as _make_backend_result


def test_discovery_lane_includes_agentic() -> None:
    assert "agentic_llm" in _LANE_BACKENDS["discovery"]


def test_optimization_lane_includes_agentic() -> None:
    assert "agentic_llm" in _LANE_BACKENDS["optimization"]


def test_submission_lane_excludes_agentic() -> None:
    assert "agentic_llm" not in _LANE_BACKENDS["submission"]


def test_mi_pooled_results_are_promoted_to_backend_result() -> None:
    result = _make_backend_result(ofv=100.0, aic=110.0, bic=120.0)
    outcome = SearchOutcome(
        results=[
            SearchResult(
                candidate_id=result.model_id,
                spec=None,  # type: ignore[arg-type]
                result=result,
                converged=True,
                bic=result.bic,
                aic=result.aic,
                n_params=len(result.parameter_estimates),
            )
        ]
    )
    stability = ImputationStabilityManifest(
        m=3,
        method="MI-PMM",
        entries=[
            ImputationStabilityEntry(
                candidate_id=result.model_id,
                pooled_ofv=90.0,
                pooled_aic=101.0,
                pooled_bic=112.0,
                convergence_rate=1.0,
                rank_stability=1.0,
                pooled_parameters={
                    "CL": {
                        "pooled_estimate": 6.0,
                        "within_var": 0.04,
                        "between_var": 0.01,
                        "total_var": 0.06,
                        "dof": 12.0,
                    }
                },
            )
        ],
    )

    _promote_mi_pooled_results(outcome, stability)

    promoted = outcome.results[0].result
    assert promoted is not None
    assert promoted.ofv == 90.0
    assert promoted.aic == 101.0
    assert promoted.bic == 112.0
    assert outcome.results[0].bic == 112.0
    assert promoted.parameter_estimates["CL"].estimate == 6.0
    assert promoted.parameter_estimates["CL"].se == pytest.approx(0.06**0.5)
