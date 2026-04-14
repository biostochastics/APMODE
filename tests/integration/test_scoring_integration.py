# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for the benchmark scoring harness.

Tests score_case() with mock BackendResults against various BenchmarkCase
configurations from all three suites.
"""

from __future__ import annotations

import pytest

from apmode.benchmarks.models import (
    BenchmarkCase,
    ExpectedStructure,
)
from apmode.benchmarks.scoring import score_case, score_convergence
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)


def _mock_result(
    model_id: str = "test_model",
    converged: bool = True,
    bias: float = 0.02,
) -> BackendResult:
    """Build a mock BackendResult for scoring tests."""
    estimates = {
        "ka": ParameterEstimate(
            name="ka",
            estimate=1.5 * (1 + bias),
            se=0.15,
            rse=10.0,
            ci95_lower=1.2,
            ci95_upper=1.8,
            category="structural",
        ),
        "V": ParameterEstimate(
            name="V",
            estimate=70.0 * (1 + bias),
            se=7.0,
            rse=10.0,
            ci95_lower=56.0,
            ci95_upper=84.0,
            category="structural",
        ),
        "CL": ParameterEstimate(
            name="CL",
            estimate=5.0 * (1 + bias),
            se=0.5,
            rse=10.0,
            ci95_lower=4.0,
            ci95_upper=6.0,
            category="structural",
        ),
    }
    min_status = "successful" if converged else "terminated"
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=converged,
        ofv=500.0,
        aic=520.0,
        bic=540.0,
        parameter_estimates=estimates,
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=converged,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status=min_status,
            wall_time_seconds=60.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=0.02,
                cwres_sd=1.01,
                outlier_fraction=0.01,
                obs_vs_pred_r2=0.95,
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.92, "p50": 0.96, "p95": 0.93},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"ka": True, "V": True, "CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=60.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


@pytest.mark.integration
class TestScoreCase:
    """Test score_case with Suite A-style assertions."""

    def test_good_fit_passes(self) -> None:
        """A well-fitting result passes all assertions."""
        case = BenchmarkCase(
            case_id="test_a1",
            suite="A",
            dataset_id="test",
            description="Test case",
            lane="submission",
            reference_params={"ka": 1.5, "V": 70.0, "CL": 5.0},
            expected_structure=ExpectedStructure(
                absorption="FirstOrder",
                distribution="OneCmt",
                elimination="Linear",
            ),
        )
        result = _mock_result(bias=0.02)
        score = score_case(case, result)

        assert score.overall_passed
        # structure_recovered is None when no DSLSpec is passed
        assert score.structure_recovered is None
        assert all(b <= 0.20 for b in score.param_bias.values())
        assert score.convergence_rate == 1.0

    def test_biased_fit_fails(self) -> None:
        """A heavily biased result fails parameter tolerance."""
        case = BenchmarkCase(
            case_id="test_biased",
            suite="A",
            dataset_id="test",
            description="Test case",
            lane="submission",
            reference_params={"ka": 1.5, "V": 70.0, "CL": 5.0},
            param_bias_tolerance=0.10,  # Tighter threshold
        )
        result = _mock_result(bias=0.15)  # 15% bias
        score = score_case(case, result)

        assert not score.overall_passed
        assert max(score.param_bias.values()) > 0.10

    def test_no_reference_params(self) -> None:
        """Case with no reference params still produces a valid score."""
        case = BenchmarkCase(
            case_id="test_no_ref",
            suite="B",
            dataset_id="test",
            description="Test case",
            lane="discovery",
        )
        result = _mock_result()
        score = score_case(case, result)

        assert score.param_bias == {}
        assert score.case_id == "test_no_ref"


@pytest.mark.integration
class TestConvergenceScoring:
    """Test convergence rate and failure taxonomy."""

    def test_all_converged(self) -> None:
        """All converged → rate = 1.0, no failures."""
        results = [_mock_result(converged=True) for _ in range(5)]
        rate, failures = score_convergence(results)
        assert rate == 1.0
        assert failures == {}

    def test_mixed_convergence(self) -> None:
        """Mixed convergence computes correct rate."""
        results = [
            _mock_result(converged=True),
            _mock_result(converged=True),
            _mock_result(converged=False),
        ]
        rate, failures = score_convergence(results)
        assert rate == pytest.approx(2.0 / 3.0)
        assert "terminated" in failures  # Non-converged mock uses "terminated" status

    def test_empty_results(self) -> None:
        """Empty results → 0.0 rate."""
        rate, failures = score_convergence([])
        assert rate == 0.0
        assert failures == {}
