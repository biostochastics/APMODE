# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE functional distillation."""

from __future__ import annotations

import jax
import pytest

from apmode.backends.node_distillation import (
    DistillationReport,
    FidelityResult,
    SurrogateResult,
    distill,
    fit_parametric_surrogate,
    quantify_fidelity,
    visualize_sub_function,
)
from apmode.backends.node_ode import HybridPKODE, ODEConfig


def _make_model(seed: int = 0) -> HybridPKODE:
    return HybridPKODE(
        config=ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.0, "V": 30.0},
        ),
        key=jax.random.PRNGKey(seed),
    )


class TestSubFunctionVisualization:
    """Learned sub-function visualization."""

    def test_produces_x_y_data(self) -> None:
        model = _make_model()
        x, y = visualize_sub_function(model, n_points=20)
        assert len(x) == 20
        assert len(y) == 20

    def test_x_covers_range(self) -> None:
        model = _make_model()
        x, _y = visualize_sub_function(model, conc_range=(0.1, 50.0), n_points=10)
        assert x[0] == pytest.approx(0.1)
        assert x[-1] == pytest.approx(50.0)

    def test_y_values_finite(self) -> None:
        model = _make_model()
        _x, y = visualize_sub_function(model, n_points=50)
        import math

        assert all(math.isfinite(v) for v in y)

    def test_bounded_positive_output_positive(self) -> None:
        model = _make_model()
        _x, y = visualize_sub_function(model, n_points=50)
        assert all(v > 0 for v in y)


class TestSurrogateFitting:
    """Parametric surrogate fitting."""

    def test_fits_to_linear_data(self) -> None:
        x = [float(i) for i in range(1, 21)]
        y = [2.0 * xi + 1.0 for xi in x]
        result = fit_parametric_surrogate(x, y)
        assert isinstance(result, SurrogateResult)
        assert result.surrogate_type in ("linear", "michaelis_menten")
        assert result.r_squared > 0.9

    def test_fits_to_mm_data(self) -> None:
        x = [float(i) for i in range(1, 51)]
        vmax, km = 10.0, 5.0
        y = [vmax * xi / (km + xi) for xi in x]
        result = fit_parametric_surrogate(x, y)
        assert isinstance(result, SurrogateResult)
        # MM fit should be very good
        assert result.r_squared > 0.9

    def test_has_interpretable_params(self) -> None:
        x = [float(i) for i in range(1, 21)]
        y = [2.0 * xi + 1.0 for xi in x]
        result = fit_parametric_surrogate(x, y)
        assert len(result.params) > 0
        if result.surrogate_type == "linear":
            assert "slope" in result.params
            assert "intercept" in result.params
        else:
            assert "Vmax" in result.params
            assert "Km" in result.params


class TestFidelityQuantification:
    """AUC/Cmax 80-125% GMR bioequivalence."""

    def test_identical_passes(self) -> None:
        x = [float(i) for i in range(1, 21)]
        y = [2.0 * xi for xi in x]
        surr = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 2.0, "intercept": 0.0},
            residual_ss=0.0,
            r_squared=1.0,
        )
        result = quantify_fidelity(x, y, surr)
        assert isinstance(result, FidelityResult)
        assert result.auc_gmr == pytest.approx(1.0, abs=0.01)
        assert result.cmax_gmr == pytest.approx(1.0, abs=0.01)
        assert result.overall_pass is True

    def test_large_deviation_fails(self) -> None:
        x = [float(i) for i in range(1, 21)]
        y = [10.0 * xi for xi in x]  # NODE output
        surr = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 2.0, "intercept": 0.0},
            residual_ss=100.0,
            r_squared=0.5,
        )
        result = quantify_fidelity(x, y, surr)
        # Surrogate predicts 5x less → GMR ~0.2 → fails 80-125%
        assert result.overall_pass is False

    def test_gmr_within_bounds(self) -> None:
        x = [float(i) for i in range(1, 21)]
        y = [2.0 * xi for xi in x]
        surr = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 2.1, "intercept": 0.0},
            residual_ss=1.0,
            r_squared=0.99,
        )
        result = quantify_fidelity(x, y, surr)
        # 2.1/2.0 = 1.05 → within 80-125%
        assert result.auc_pass is True


class TestDistillPipeline:
    """Full distillation pipeline."""

    def test_produces_report(self) -> None:
        model = _make_model()
        report = distill(model, "test_candidate")

        assert isinstance(report, DistillationReport)
        assert report.candidate_id == "test_candidate"
        assert report.node_position == "elimination"
        assert len(report.sub_function_x) > 0
        assert len(report.sub_function_y) > 0
        assert report.surrogate is not None
        assert report.fidelity is not None

    def test_report_has_surrogate_params(self) -> None:
        model = _make_model()
        report = distill(model, "test_candidate")
        assert len(report.surrogate.params) > 0  # type: ignore[union-attr]

    def test_absorption_position(self) -> None:
        model = HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position="absorption",
                constraint_template="monotone_decreasing",
                node_dim=3,
            ),
            key=jax.random.PRNGKey(0),
        )
        report = distill(model, "abs_candidate")
        assert report.node_position == "absorption"
