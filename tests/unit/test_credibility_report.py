# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for credibility report generator."""

from __future__ import annotations

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    CredibilityReport,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)
from apmode.report.credibility import generate_credibility_report


def _make_result(backend: str = "nlmixr2") -> BackendResult:
    return BackendResult(
        model_id="test_model",
        backend=backend,  # type: ignore[arg-type]
        converged=True,
        ofv=150.0,
        aic=160.0,
        bic=170.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=70.0, category="structural"),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=200,
            minimization_status="successful",
            wall_time_seconds=45.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.01, cwres_sd=1.0, outlier_fraction=0.02),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.92, "p50": 0.95, "p95": 0.91},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=45.0,
        backend_versions={"nlmixr2": "2.1.2"},
        initial_estimate_source="nca",
    )


class TestCredibilityReportGenerator:
    def test_generates_report_for_classical(self) -> None:
        result = _make_result("nlmixr2")
        report = generate_credibility_report(result, "submission", n_observations=100)
        assert isinstance(report, CredibilityReport)
        assert report.candidate_id == "test_model"
        assert "submission" in report.context_of_use
        assert report.ml_transparency is None

    def test_generates_ml_transparency_for_node(self) -> None:
        result = _make_result("jax_node")
        report = generate_credibility_report(result, "discovery", n_observations=200)
        assert report.ml_transparency is not None
        assert "NODE" in report.ml_transparency

    def test_data_adequacy_adequate(self) -> None:
        result = _make_result()
        report = generate_credibility_report(result, "submission", n_observations=100)
        assert report.data_adequacy == "adequate"

    def test_data_adequacy_marginal(self) -> None:
        result = _make_result()
        report = generate_credibility_report(result, "submission", n_observations=5)
        assert report.data_adequacy == "marginal"

    def test_context_of_use_populated(self) -> None:
        result = _make_result()
        report = generate_credibility_report(result, "discovery", n_observations=100)
        assert len(report.context_of_use) > 0

    def test_credibility_dict_has_fields(self) -> None:
        result = _make_result()
        report = generate_credibility_report(result, "submission", n_observations=100)
        assert "estimation_method" in report.model_credibility
        assert "converged" in report.model_credibility
        assert "n_parameters" in report.model_credibility

    def test_node_has_limitations(self) -> None:
        result = _make_result("jax_node")
        report = generate_credibility_report(result, "discovery", n_observations=100)
        assert len(report.limitations) > 0
        assert any("NODE" in lim for lim in report.limitations)
