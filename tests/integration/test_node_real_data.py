# SPDX-License-Identifier: GPL-2.0-or-later
"""Real-data integration tests for NODE backend.

Tests the full NODE pipeline (ODE build -> train -> distill) on real PK
datasets fetched from R's nlmixr2data package. Requires R with nlmixr2data.

Datasets:
  - theo_sd: 12 subjects, 1-cmt oral, single dose, linear elimination
  - Oral_2CPT: 120 subjects, 2-cmt oral, multi-dose (tests rejection guard)
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from apmode.backends.node_constraints import TEMPLATE_MAX_DIM
from apmode.backends.node_distillation import distill
from apmode.backends.node_init import reset_weight_library, transfer_from_classical
from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_runner import NodeBackendRunner
from apmode.backends.node_trainer import TrainingConfig, train_node
from apmode.bundle.models import ColumnMapping, DataManifest
from apmode.dsl.ast_models import (
    DSLSpec,
    FirstOrder,
    NODEElimination,
    OneCmt,
    Proportional,
    TwoCmt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_R_AVAILABLE = True
try:
    from apmode.data.datasets import fetch_dataset
except ImportError:
    _R_AVAILABLE = False

DATA_DIR = Path("/tmp/apmode_node_integration_tests")


def _theo_sd_path() -> Path:
    """Fetch theo_sd if not cached."""
    return fetch_dataset("theo_sd", DATA_DIR)


def _oral_2cpt_path() -> Path:
    """Fetch Oral_2CPT if not cached."""
    return fetch_dataset("Oral_2CPT", DATA_DIR)


def _make_theo_manifest() -> DataManifest:
    """DataManifest for theo_sd dataset."""
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
            cmt="CMT",
        ),
        n_subjects=12,
        n_observations=132,
        n_doses=12,
    )


def _make_oral_2cpt_manifest() -> DataManifest:
    """DataManifest for Oral_2CPT dataset."""
    return DataManifest(
        data_sha256="b" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
            cmt="CMT",
        ),
        n_subjects=120,
        n_observations=6600,
        n_doses=960,
    )


def _make_node_spec(*, n_cmt: int = 1, model_id: str = "test_node") -> DSLSpec:
    """Create a DSLSpec with NODE elimination."""
    dist = OneCmt(V=30.0) if n_cmt == 1 else TwoCmt(V1=30.0, V2=40.0, Q=5.0)
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(ka=1.0),
        distribution=dist,
        elimination=NODEElimination(dim=3, constraint_template="bounded_positive"),
        variability=[],
        observation=Proportional(sigma_prop=0.1),
    )


requires_r = pytest.mark.skipif(
    not _R_AVAILABLE,
    reason="R with nlmixr2data not available",
)


# ---------------------------------------------------------------------------
# 1-cmt real data: theo_sd
# ---------------------------------------------------------------------------


@requires_r
class TestTheoSDNodeFit:
    """NODE fit on theo_sd (12 subjects, 1-cmt, single dose, real data)."""

    def test_load_and_train_1cmt(self, tmp_path: Path) -> None:
        """Full pipeline: fetch data -> load CSV -> build ODE -> train -> verify."""
        reset_weight_library()

        csv_path = _theo_sd_path()
        manifest = _make_theo_manifest()

        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=50, learning_rate=5e-3),
        )

        # Load subjects directly to verify CSV parsing
        subjects = runner._load_subjects_from_csv(csv_path, manifest, n_cmt=1)

        assert len(subjects) == 12, f"Expected 12 subjects, got {len(subjects)}"

        for subj in subjects:
            assert subj["y0"].shape == (2,), "1-cmt y0 should be (2,)"
            assert float(subj["y0"][0]) > 0, "Dose should be positive"
            assert float(subj["y0"][1]) == 0.0, "Initial central amount should be 0"
            assert len(subj["times"]) > 0, "Should have observations"
            # Times should be sorted
            times_np = np.array(subj["times"])
            assert np.all(times_np[1:] >= times_np[:-1]), "Times should be sorted"

    def test_train_produces_convergence(self, tmp_path: Path) -> None:
        """Train on real theo_sd data and verify convergence."""
        reset_weight_library()

        csv_path = _theo_sd_path()
        manifest = _make_theo_manifest()

        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=100, learning_rate=5e-3),
        )
        subjects = runner._load_subjects_from_csv(csv_path, manifest, n_cmt=1)

        # Use only first 4 subjects for speed
        subjects = subjects[:4]

        config = ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.5, "V": 30.0, "CL": 3.0},
        )
        key = jax.random.PRNGKey(42)
        transfer = transfer_from_classical(
            config,
            classical_estimates={"ka": 1.5, "V": 30.0, "CL": 3.0},
            key=key,
        )

        result = train_node(
            transfer.model,
            subjects,
            TrainingConfig(epochs=100, learning_rate=5e-3),
        )

        assert result.n_epochs > 0
        assert result.final_loss < float("inf")
        assert np.isfinite(result.final_loss), "Loss should be finite"
        # Loss should decrease from initial
        assert result.loss_history[-1] < result.loss_history[0], "Loss should decrease"
        assert result.minimization_status in ("successful", "plateau", "max_evaluations")

    def test_distillation_on_trained_model(self, tmp_path: Path) -> None:
        """Train on real data then run functional distillation."""
        reset_weight_library()

        csv_path = _theo_sd_path()
        manifest = _make_theo_manifest()

        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=80, learning_rate=5e-3),
        )
        subjects = runner._load_subjects_from_csv(csv_path, manifest, n_cmt=1)[:4]

        config = ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.5, "V": 30.0, "CL": 3.0},
        )
        key = jax.random.PRNGKey(42)
        transfer = transfer_from_classical(
            config,
            classical_estimates={"ka": 1.5, "V": 30.0, "CL": 3.0},
            key=key,
        )
        result = train_node(transfer.model, subjects, TrainingConfig(epochs=80))

        # Distill
        report = distill(result.trained_model, "theo_node_1cmt")

        assert report.candidate_id == "theo_node_1cmt"
        assert report.node_position == "elimination"
        assert len(report.sub_function_x) == 100
        assert len(report.sub_function_y) == 100
        assert report.surrogate is not None
        assert report.surrogate.r_squared >= 0.0
        assert report.fidelity is not None
        assert report.fidelity.auc_gmr > 0

    async def test_full_runner_e2e(self, tmp_path: Path) -> None:
        """End-to-end NodeBackendRunner.run() on theo_sd."""
        reset_weight_library()

        csv_path = _theo_sd_path()
        manifest = _make_theo_manifest()
        spec = _make_node_spec(n_cmt=1, model_id="theo_node_e2e")

        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=50, learning_rate=5e-3),
        )

        result = await runner.run(
            spec=spec,
            data_manifest=manifest,
            initial_estimates={"ka": 1.5, "V": 30.0, "CL": 3.0},
            seed=42,
            data_path=csv_path,
        )

        assert result.backend == "jax_node"
        assert result.model_id == "theo_node_e2e"
        assert np.isfinite(result.ofv), "OFV should be finite"
        assert result.aic > result.ofv, "AIC >= OFV (penalty term > 0)"
        assert result.bic > result.ofv, "BIC >= OFV"
        assert "ka" in result.parameter_estimates
        assert "V" in result.parameter_estimates
        assert "sigma" in result.parameter_estimates
        assert result.parameter_estimates["ka"].estimate > 0
        assert result.parameter_estimates["V"].estimate > 0
        # Verify BIC uses actual trainable params (should be ~25, not ~5)
        n_params_implied = (result.bic - result.ofv) / np.log(132)
        assert n_params_implied > 10, (
            f"BIC implies {n_params_implied:.0f} params — should be >10 (MLP weights)"
        )


# ---------------------------------------------------------------------------
# 2-cmt: multi-dose rejection and mock-mode 2-cmt
# ---------------------------------------------------------------------------


@requires_r
class TestOral2CPTMultiDose:
    """Oral_2CPT is multi-dose — NODE runner should load it with event support."""

    def test_multi_dose_loads_subjects(self, tmp_path: Path) -> None:
        """Multi-dose data loads successfully with dose event handling."""
        csv_path = _oral_2cpt_path()
        manifest = _make_oral_2cpt_manifest()

        runner = NodeBackendRunner(work_dir=tmp_path)
        subjects = runner._load_subjects_from_csv(csv_path, manifest, n_cmt=2)

        assert len(subjects) > 0
        # Multi-dose subjects should have dose_events or single-dose y0
        for subj in subjects:
            assert "times" in subj
            assert "observations" in subj
            assert "y0" in subj


class TestTwoCmtMockMode:
    """2-cmt NODE in mock mode (no R needed)."""

    def test_mock_2cmt_y0_shape(self, tmp_path: Path) -> None:
        """Mock subjects for 2-cmt have 3-element y0."""
        manifest = DataManifest(
            data_sha256="c" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
            ),
            n_subjects=5,
            n_observations=40,
            n_doses=5,
        )

        runner = NodeBackendRunner(work_dir=tmp_path)
        subjects = runner._make_mock_subjects(
            manifest,
            {"ka": 1.0, "V": 30.0, "CL": 2.0, "V2": 40.0, "Q": 5.0},
            n_cmt=2,
        )

        assert len(subjects) == 5
        for subj in subjects:
            assert subj["y0"].shape == (3,), f"2-cmt y0 should be (3,), got {subj['y0'].shape}"
            assert float(subj["y0"][2]) == 0.0, "Initial peripheral amount should be 0"

    def test_train_2cmt_mock(self, tmp_path: Path) -> None:
        """Train a 2-cmt NODE on mock data (no crash)."""
        reset_weight_library()

        manifest = DataManifest(
            data_sha256="d" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
            ),
            n_subjects=5,
            n_observations=40,
            n_doses=5,
        )

        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=30),
        )
        subjects = runner._make_mock_subjects(
            manifest,
            {"ka": 1.0, "V": 30.0, "CL": 2.0, "V2": 40.0, "Q": 5.0},
            n_cmt=2,
        )

        config = ODEConfig(
            n_cmt=2,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.0, "V": 30.0, "CL": 2.0, "V2": 40.0, "Q": 5.0},
        )
        key = jax.random.PRNGKey(0)
        model = HybridPKODE(config=config, key=key)

        result = train_node(model, subjects, TrainingConfig(epochs=30))

        assert result.n_epochs > 0
        assert np.isfinite(result.final_loss)


# ---------------------------------------------------------------------------
# Constraint template coverage on real-shaped data
# ---------------------------------------------------------------------------


@requires_r
class TestConstraintTemplatesRealData:
    """Test all constraint templates on real data shapes from theo_sd."""

    @pytest.mark.parametrize("template", list(TEMPLATE_MAX_DIM.keys()))
    def test_template_trains_on_theo(self, template: str, tmp_path: Path) -> None:
        """Each constraint template trains without error on real data."""
        reset_weight_library()

        csv_path = _theo_sd_path()
        manifest = _make_theo_manifest()

        runner = NodeBackendRunner(work_dir=tmp_path)
        subjects = runner._load_subjects_from_csv(csv_path, manifest, n_cmt=1)[:2]

        max_dim = TEMPLATE_MAX_DIM[template]
        dim = min(3, max_dim)

        config = ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template=template,
            node_dim=dim,
            mechanistic_params={"ka": 1.5, "V": 30.0, "CL": 3.0},
        )
        key = jax.random.PRNGKey(42)
        model = HybridPKODE(config=config, key=key)

        result = train_node(model, subjects, TrainingConfig(epochs=20))

        assert result.n_epochs > 0
        assert np.isfinite(result.final_loss), f"Template {template} produced NaN loss"
