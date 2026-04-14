# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 2 LORO requirement check (Phase 3 P3.B).

Tests the replacement of the _check_loro_requirement placeholder with
real LOROMetrics threshold evaluation.
"""

from __future__ import annotations

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    LOROMetrics,
    ParameterEstimate,
)
from apmode.governance.gates import _check_loro_requirement
from apmode.governance.policy import Gate2Config


def _default_g2() -> Gate2Config:
    """Default optimization-lane Gate 2 config with LORO enabled."""
    return Gate2Config(
        interpretable_parameterization="not_required",
        reproducible_estimation="required",
        shrinkage_max=None,
        identifiability_required=False,
        node_eligible=True,
        loro_required=True,
        loro_npde_mean_max=0.3,
        loro_npde_variance_min=0.5,
        loro_npde_variance_max=1.5,
        loro_vpc_coverage_min=0.80,
        loro_min_folds=3,
    )


def _mock_result() -> BackendResult:
    return BackendResult(
        model_id="test",
        backend="nlmixr2",
        converged=True,
        ofv=-100.0,
        aic=210.0,
        bic=220.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=2.0, category="structural"),
        },
        eta_shrinkage={"CL": 10.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.05, cwres_sd=1.0, outlier_fraction=0.02),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0,
                profile_likelihood_ci={"CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "5.0.0"},
        initial_estimate_source="nca",
    )


def _good_metrics() -> LOROMetrics:
    """LOROMetrics that should pass Gate 2."""
    return LOROMetrics(
        n_folds=4,
        n_total_test_subjects=40,
        pooled_npde_mean=0.05,
        pooled_npde_variance=1.02,
        vpc_coverage_concordance=0.92,
        overall_pass=True,
    )


class TestCheckLoroRequirement:
    """Tests for _check_loro_requirement with real LOROMetrics."""

    def test_passes_when_not_optimization_lane(self) -> None:
        g2 = _default_g2()
        check = _check_loro_requirement(_mock_result(), g2, "discovery")
        assert check.passed is True

    def test_passes_when_loro_not_required(self) -> None:
        g2 = _default_g2()
        g2 = g2.model_copy(update={"loro_required": False})
        check = _check_loro_requirement(_mock_result(), g2, "optimization")
        assert check.passed is True

    def test_fails_when_no_metrics_provided(self) -> None:
        """Optimization lane with loro_required=True but no metrics → fail."""
        g2 = _default_g2()
        check = _check_loro_requirement(_mock_result(), g2, "optimization")
        assert check.passed is False
        assert "not_evaluated" in str(check.observed)

    def test_passes_with_good_metrics(self) -> None:
        g2 = _default_g2()
        check = _check_loro_requirement(
            _mock_result(), g2, "optimization", loro_metrics=_good_metrics()
        )
        assert check.passed is True

    def test_fails_high_npde_mean(self) -> None:
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"pooled_npde_mean": 0.5})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False
        assert "npde_mean" in str(check.observed)

    def test_fails_negative_npde_mean(self) -> None:
        """Absolute value check: negative NPDE mean beyond threshold."""
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"pooled_npde_mean": -0.4})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False

    def test_fails_low_npde_variance(self) -> None:
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"pooled_npde_variance": 0.3})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False
        assert "npde_var" in str(check.observed)

    def test_fails_high_npde_variance(self) -> None:
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"pooled_npde_variance": 2.0})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False

    def test_fails_low_vpc_coverage(self) -> None:
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"vpc_coverage_concordance": 0.65})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False
        assert "vpc_cov" in str(check.observed)

    def test_fails_insufficient_folds(self) -> None:
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(update={"n_folds": 2})
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False
        assert "folds" in str(check.observed).lower()

    def test_multiple_failures_reported(self) -> None:
        """Multiple threshold violations → all reported in observed."""
        g2 = _default_g2()
        metrics = _good_metrics().model_copy(
            update={
                "pooled_npde_mean": 0.5,
                "vpc_coverage_concordance": 0.60,
            }
        )
        check = _check_loro_requirement(_mock_result(), g2, "optimization", loro_metrics=metrics)
        assert check.passed is False
        assert "npde_mean" in str(check.observed)
        assert "vpc_cov" in str(check.observed)
