# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for diagnostic summarizer (agentic LLM context builder)."""

from apmode.backends.diagnostic_summarizer import summarize_diagnostics, summarize_for_llm
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
)


def _mock_result(
    cwres_mean: float = 0.05,
    outlier_frac: float = 0.02,
    converged: bool = True,
) -> BackendResult:
    return BackendResult(
        model_id="test-model",
        backend="nlmixr2",
        converged=converged,
        ofv=-100.0,
        aic=210.0,
        bic=220.0,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL", estimate=2.0, se=0.1, rse=5.0, category="structural"
            ),
            "V": ParameterEstimate(
                name="V", estimate=30.0, se=1.5, rse=5.0, category="structural"
            ),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=converged,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=cwres_mean,
                cwres_sd=1.05,
                outlier_fraction=outlier_frac,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "2.1.0"},
        initial_estimate_source="nca",
    )


def test_summarize_diagnostics_returns_dict() -> None:
    result = _mock_result()
    summary = summarize_diagnostics(result)
    assert "cwres_mean" in summary
    assert "parameters" in summary
    assert summary["converged"] is True


def test_summarize_for_llm_returns_string() -> None:
    result = _mock_result()
    text = summarize_for_llm(result, iteration=3, max_iterations=25)
    assert "Iteration 3/25" in text
    assert "CL" in text
    assert "CWRES" in text


def test_summarize_highlights_high_cwres() -> None:
    result = _mock_result(cwres_mean=0.8)
    text = summarize_for_llm(result, iteration=1, max_iterations=25)
    assert "bias" in text.lower() or "high" in text.lower()


def test_summarize_highlights_non_convergence() -> None:
    result = _mock_result(converged=False)
    text = summarize_for_llm(result, iteration=1, max_iterations=25)
    assert "not converge" in text.lower() or "did not converge" in text.lower()
