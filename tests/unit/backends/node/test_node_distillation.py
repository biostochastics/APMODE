# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE functional distillation."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
from pydantic import BaseModel

from apmode.backends.node_distillation import (
    DistillationReport,
    FidelityResult,
    SurrogateResult,
    distill,
    distillation_passes_fidelity,
    fit_parametric_surrogate,
    quantify_fidelity,
    surrogate_to_formular,
    visualize_sub_function,
)
from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.dsl.ast_models import FirstOrder, LinearElim, OneCmt


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


class TestDistillationReportPydantic:
    """The distillation dataclasses are promoted to sealed Pydantic models."""

    def test_models_are_pydantic(self) -> None:
        assert issubclass(SurrogateResult, BaseModel)
        assert issubclass(FidelityResult, BaseModel)
        assert issubclass(DistillationReport, BaseModel)

    def test_report_json_roundtrips(self) -> None:
        report = DistillationReport(
            candidate_id="c001",
            node_position="elimination",
            sub_function_x=[0.1, 1.0],
            sub_function_y=[0.05, 0.05],
            surrogate=SurrogateResult(
                surrogate_type="linear",
                params={"slope": 0.0, "intercept": 0.05},
                residual_ss=0.0,
                r_squared=0.99,
            ),
            fidelity=FidelityResult(
                auc_gmr=1.0, cmax_gmr=1.0, auc_pass=True, cmax_pass=True, overall_pass=True
            ),
        )
        restored = DistillationReport.model_validate_json(report.model_dump_json())
        assert restored == report
        assert restored.promoted is False
        assert restored.promoted_model_id is None


class TestSurrogateToFormular:
    """Promote a fitted NODE surrogate to a classical, refit-able DSLSpec."""

    def test_linear_surrogate_maps_to_linear_elimination(self) -> None:
        surrogate = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 0.0, "intercept": 0.05},
            residual_ss=0.0,
            r_squared=0.99,
        )
        spec = surrogate_to_formular(
            surrogate,
            "elimination",
            model_id="m_distilled",
            mechanistic_params={"ka": 1.0, "V": 30.0, "CL": 2.0},
        )
        assert isinstance(spec.absorption, FirstOrder)
        assert isinstance(spec.distribution, OneCmt)
        assert isinstance(spec.elimination, LinearElim)
        # A constant NODE coefficient is CL/V, hence CL = coefficient * V.
        assert spec.initial["CL"] == pytest.approx(0.05 * 30.0)
        assert spec.initial["V"] == pytest.approx(30.0)
        assert spec.initial["ka"] == pytest.approx(1.0)
        # A promoted surrogate is a plain classical candidate — never a NODE spec.
        assert spec.has_node_modules() is False
        assert spec.experimental.node is False
        assert spec.metadata is not None
        assert spec.metadata.intent is not None and "distill" in spec.metadata.intent

    def test_michaelis_menten_shaped_coefficient_fails_closed(self) -> None:
        surrogate = SurrogateResult(
            surrogate_type="michaelis_menten",
            params={"Vmax": 12.0, "Km": 3.0},
            residual_ss=0.0,
            r_squared=0.98,
        )
        with pytest.raises(ValueError, match="per-unit rate"):
            surrogate_to_formular(
                surrogate,
                "elimination",
                model_id="m_mm",
                mechanistic_params={"ka": 1.0, "V": 25.0},
            )

    def test_nonzero_slope_is_not_frozen_at_reference_concentration(self) -> None:
        surrogate = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 0.05, "intercept": 0.01},
            residual_ss=0.0,
            r_squared=0.99,
        )
        with pytest.raises(ValueError, match="concentration-dependent"):
            surrogate_to_formular(
                surrogate,
                "elimination",
                model_id="m_bad",
                mechanistic_params={"ka": 1.0, "V": 25.0},
                reference_conc=10.0,
            )

    def test_absorption_position_is_out_of_scope(self) -> None:
        surrogate = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 0.0, "intercept": 0.05},
            residual_ss=0.0,
            r_squared=0.9,
        )
        with pytest.raises(NotImplementedError):
            surrogate_to_formular(
                surrogate, "absorption", model_id="x", mechanistic_params={"ka": 1.0, "V": 30.0}
            )


class TestFidelityGate:
    """Only distillations that clear fidelity may promote into Gate 3."""

    def _report(self, *, overall_pass: bool, r2: float) -> DistillationReport:
        return DistillationReport(
            candidate_id="c",
            node_position="elimination",
            surrogate=SurrogateResult(
                surrogate_type="linear",
                params={"slope": 0.0, "intercept": 0.05},
                residual_ss=0.0,
                r_squared=r2,
            ),
            fidelity=FidelityResult(
                auc_gmr=1.0,
                cmax_gmr=1.0,
                auc_pass=overall_pass,
                cmax_pass=overall_pass,
                overall_pass=overall_pass,
            ),
        )

    def test_passes_when_bioequivalent_and_high_r2(self) -> None:
        assert distillation_passes_fidelity(self._report(overall_pass=True, r2=0.95)) is True

    def test_fails_on_low_r2(self) -> None:
        assert distillation_passes_fidelity(self._report(overall_pass=True, r2=0.5)) is False

    def test_fails_when_not_bioequivalent(self) -> None:
        assert distillation_passes_fidelity(self._report(overall_pass=False, r2=0.95)) is False

    def test_fails_when_surrogate_or_fidelity_missing(self) -> None:
        empty = DistillationReport(candidate_id="c", node_position="elimination")
        assert distillation_passes_fidelity(empty) is False


class TestWriteDistillationReport:
    """BundleEmitter seals a distillation report to distillation/<id>.json.

    This is the method the orchestrator's NODE path calls (B1b seal), so the
    projected RO-Crate File entity has a real artifact to point at.
    """

    def test_writes_report_to_bundle(self, tmp_path: Path) -> None:
        import json

        from apmode.bundle.emitter import BundleEmitter

        emitter = BundleEmitter(tmp_path)
        emitter.initialize()

        report = DistillationReport(
            candidate_id="node_1",
            node_position="elimination",
            surrogate=SurrogateResult(
                surrogate_type="linear",
                params={"slope": 0.0, "intercept": 0.05},
                residual_ss=0.0,
                r_squared=0.95,
            ),
        )
        path = emitter.write_distillation_report(report)

        assert path.exists()
        assert path.parent.name == "distillation"
        assert path.name == "node_1.json"
        data = json.loads(path.read_text())
        assert data["candidate_id"] == "node_1"
        assert data["surrogate"]["surrogate_type"] == "linear"
