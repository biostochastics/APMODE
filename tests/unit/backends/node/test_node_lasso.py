# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE-LASSO structural selection and distillation (Bräm 2025).

Covers the candidate library (DSL-mappable annotation), column standardization,
LASSO structural recovery on synthetic linear / Michaelis-Menten derivatives, and
the ``distill_via_lasso`` integration that promotes a fully-mappable selected
structure to a classical ``DSLSpec`` (doi:10.1002/psp4.70285).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from apmode.backends.node_candidate_library import (
    build_candidate_library,
    standardize_columns,
)
from apmode.backends.node_distillation import (
    LassoDistillationReport,
    distill_via_lasso,
    lasso_fidelity,
    lasso_result_to_formular,
)
from apmode.backends.node_lasso_select import select_structure
from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.bundle.models import LassoSelectionResult
from apmode.dsl.ast_models import LinearElim


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


class TestCandidateLibraryMapping:
    """Every candidate column carries its DSL module (or None) per Bräm 2025 §2.3."""

    def test_module_annotations(self) -> None:
        inputs = np.linspace(0.5, 10.0, 40)
        library = build_candidate_library(
            inputs,
            u_exp_grid=[-2.0, -1.0, 1.0, 2.0],
            h_grid=[0.5, 1.0, 2.0],
            ue50_grid=[2.0, 4.0],
        )
        by_family: dict[str, list[str | None]] = {}
        for col in library:
            by_family.setdefault(col.family, []).append(col.mappable_module)

        # Only a constant NODE coefficient maps to CL/V. A concentration-linear
        # coefficient is not Formular's linear elimination law.
        assert all(m is None for m in by_family["linear"])
        assert all(m == "LinearElim" for m in by_family["constant"])
        # exponential is non-mappable.
        assert all(m is None for m in by_family["exponential"])

    def test_emax_is_not_mismapped_to_mm(self) -> None:
        inputs = np.linspace(0.5, 10.0, 40)
        library = build_candidate_library(
            inputs, u_exp_grid=[1.0], h_grid=[0.5, 1.0, 2.0], ue50_grid=[3.0]
        )
        for col in library:
            if col.family != "emax":
                continue
            assert col.mappable_module is None


class TestStandardizeColumns:
    """Standardization must not produce NaN even with a zero-variance column."""

    def test_constant_column_no_nan(self) -> None:
        inputs = np.linspace(0.5, 10.0, 30)
        # The library always contains a constant (zero-variance) column.
        library = build_candidate_library(inputs, u_exp_grid=[1.0], h_grid=[1.0], ue50_grid=[3.0])
        design, means, stds = standardize_columns(library)
        assert np.all(np.isfinite(design))
        assert np.all(np.isfinite(means))
        assert np.all(stds > 0.0)  # zero-variance stds coerced to 1.0
        # The constant column standardizes to an all-zero column (mean subtracted).
        const_idx = next(i for i, c in enumerate(library) if c.family == "constant")
        assert np.allclose(design[:, const_idx], 0.0)


class TestSelectStructureLinear:
    """A concentration-linear coefficient is recovered but marked non-mappable."""

    def _fit(self, seed: int) -> LassoSelectionResult:
        inputs = np.linspace(0.5, 8.0, 80)
        rng = np.random.default_rng(seed)
        derivatives = 0.7 * inputs + rng.normal(0.0, 0.01, size=inputs.shape)
        library = build_candidate_library(inputs)
        return select_structure(inputs, derivatives, library, n_bootstrap=20, rng_seed=123)

    def test_recovers_linear_coefficient(self) -> None:
        result = self._fit(seed=7)
        assert "linear" in result.selected_terms
        assert "linear" in result.rejected_nonmappable
        assert "linear" not in result.coefficients
        assert result.derivative_r_squared > 0.99

    def test_does_not_select_exponential(self) -> None:
        result = self._fit(seed=7)
        assert not any(t.startswith("exp_") for t in result.selected_terms)
        assert not any(t.startswith("exp_") for t in result.rejected_nonmappable)

    def test_deterministic(self) -> None:
        first = self._fit(seed=7)
        second = self._fit(seed=7)
        assert first.selected_terms == second.selected_terms
        assert first.coefficients == second.coefficients


class TestSelectStructureMichaelisMenten:
    """An Emax-shaped coefficient is selected but never mismapped to MM."""

    def test_recovers_mm_term(self) -> None:
        inputs = np.linspace(0.5, 12.0, 90)
        vmax, km = 2.0, 3.0
        rng = np.random.default_rng(3)
        derivatives = vmax * inputs / (km + inputs) + rng.normal(0.0, 0.005, size=inputs.shape)
        # Constrain the library so the h==1 Emax at the true Km is present.
        library = build_candidate_library(
            inputs, u_exp_grid=[-1.0, 1.0], h_grid=[1.0], ue50_grid=[km]
        )
        result = select_structure(inputs, derivatives, library, n_bootstrap=20, rng_seed=5)
        mm_terms = [t for t in result.selected_terms if t.startswith("emax_h=1")]
        assert mm_terms, f"expected an Emax term, got {result.selected_terms}"
        assert all(t in result.rejected_nonmappable for t in mm_terms)
        assert all(t not in result.coefficients for t in mm_terms)
        assert result.derivative_r_squared > 0.99


class TestLassoResultToFormular:
    """Only a constant first-order coefficient maps to LinearElim."""

    def test_constant_maps_to_linear_elim(self) -> None:
        result = LassoSelectionResult(
            selected_terms=["constant"],
            coefficients={"constant": 0.05},
            bic=1.0,
            derivative_r_squared=0.99,
        )
        spec = lasso_result_to_formular(
            result, "elimination", model_id="m", mechanistic_params={"V": 30.0, "ka": 1.0}
        )
        assert isinstance(spec.elimination, LinearElim)
        assert spec.initial["CL"] == 0.05 * 30.0

    @pytest.mark.parametrize(
        ("term", "coefficient"),
        [("linear", 0.05), ("emax_h=1.000_ue50=3", 2.0)],
    )
    def test_concentration_dependent_terms_fail_closed(
        self, term: str, coefficient: float
    ) -> None:
        result = LassoSelectionResult(
            selected_terms=[term],
            coefficients={term: coefficient},
            bic=1.0,
            derivative_r_squared=0.99,
        )
        with pytest.raises(ValueError, match="constant NODE rate"):
            lasso_result_to_formular(
                result, "elimination", model_id="m", mechanistic_params={"V": 30.0}
            )


class TestLassoFidelity:
    """Derivative-space R^2 is the primary fidelity metric; MARE is a companion."""

    def test_matches_selection_r_squared(self) -> None:
        inputs = np.linspace(0.5, 8.0, 80)
        rng = np.random.default_rng(1)
        derivatives = 0.7 * inputs + rng.normal(0.0, 0.01, size=inputs.shape)
        library = build_candidate_library(inputs)
        result = select_structure(inputs, derivatives, library, n_bootstrap=20, rng_seed=9)
        fidelity = lasso_fidelity(inputs, derivatives, result)
        assert fidelity["derivative_r_squared"] > 0.99
        assert fidelity["derivative_mare"] >= 0.0

    def test_empty_selection_returns_defaults(self) -> None:
        result = LassoSelectionResult(bic=float("nan"), derivative_r_squared=0.0)
        fidelity = lasso_fidelity(np.linspace(0.1, 1.0, 10), np.zeros(10), result)
        assert fidelity["derivative_r_squared"] == 0.0


class TestDistillViaLasso:
    """distill_via_lasso runs end-to-end and promotes only fully-mappable structures."""

    def test_returns_report_without_raising(self) -> None:
        model = _make_model(seed=0)
        report = distill_via_lasso(model, "node_1", n_bootstrap=20, rng_seed=0)
        assert isinstance(report, LassoDistillationReport)
        assert report.candidate_id == "node_1"
        assert report.node_position == "elimination"

    def test_promotion_yields_mappable_elimination(self) -> None:
        model = _make_model(seed=0)
        report = distill_via_lasso(model, "node_1", n_bootstrap=20, rng_seed=0)
        if report.promoted:
            assert report.promoted_spec is not None
            assert isinstance(report.promoted_spec.elimination, LinearElim)
            assert report.promoted_model_id == "node_1_lasso_distilled"
            assert report.timecourse_fidelity is not None
            assert report.timecourse_fidelity.overall_pass
        else:
            # A non-mappable selection must never be silently promoted.
            assert report.promoted_spec is None

    def test_synthetic_concentration_linear_term_is_not_promoted(self) -> None:
        result = LassoSelectionResult(
            selected_terms=["linear"],
            coefficients={"linear": 0.05},
            bic=1.0,
            derivative_r_squared=0.99,
        )
        with pytest.raises(ValueError, match="constant NODE rate"):
            lasso_result_to_formular(
                result,
                "elimination",
                model_id="node_1_lasso_distilled",
                mechanistic_params={"V": 30.0, "ka": 1.0},
            )
