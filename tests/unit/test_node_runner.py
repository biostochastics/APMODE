# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NodeBackendRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

from apmode.backends.node_runner import NodeBackendRunner
from apmode.backends.node_trainer import TrainingConfig
from apmode.backends.protocol import BackendRunner
from apmode.bundle.models import BackendResult, ColumnMapping, DataManifest
from apmode.dsl.ast_models import (
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    NODEElimination,
    OneCmt,
)
from apmode.errors import InvalidSpecError


def _make_data_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="ID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=10,
        n_observations=80,
        n_doses=10,
    )


def _node_elim_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_node_elim",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=NODEElimination(dim=3, constraint_template="bounded_positive"),
        variability=[IIV(params=["ka", "V"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


def _node_abs_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_node_abs",
        absorption=NODEAbsorption(dim=3, constraint_template="monotone_decreasing"),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["V", "CL"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


def _classical_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_classical",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["ka", "V", "CL"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


class TestProtocolCompliance:
    """NodeBackendRunner satisfies BackendRunner protocol."""

    def test_is_runtime_checkable(self) -> None:
        runner = NodeBackendRunner(
            work_dir=Path("/tmp/test_node"),
            training_config=TrainingConfig(epochs=2),
        )
        assert isinstance(runner, BackendRunner)


class TestNodeElimination:
    """NODE elimination backend."""

    @pytest.mark.asyncio
    async def test_run_produces_backend_result(self, tmp_path: Path) -> None:
        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=5, early_stop_patience=10),
        )
        result = await runner.run(
            spec=_node_elim_spec(),
            data_manifest=_make_data_manifest(),
            initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
            seed=42,
        )

        assert isinstance(result, BackendResult)
        assert result.backend == "jax_node"
        assert result.model_id == "test_node_elim"
        assert result.wall_time_seconds > 0

    @pytest.mark.asyncio
    async def test_result_has_parameters(self, tmp_path: Path) -> None:
        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=5, early_stop_patience=10),
        )
        result = await runner.run(
            spec=_node_elim_spec(),
            data_manifest=_make_data_manifest(),
            initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
            seed=42,
        )

        assert "ka" in result.parameter_estimates
        assert "V" in result.parameter_estimates
        assert "sigma" in result.parameter_estimates
        assert result.parameter_estimates["sigma"].estimate > 0


class TestNodeAbsorption:
    """NODE absorption backend."""

    @pytest.mark.asyncio
    async def test_run_produces_backend_result(self, tmp_path: Path) -> None:
        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=5, early_stop_patience=10),
        )
        result = await runner.run(
            spec=_node_abs_spec(),
            data_manifest=_make_data_manifest(),
            initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
            seed=42,
        )

        assert isinstance(result, BackendResult)
        assert result.backend == "jax_node"


class TestInvalidSpec:
    """Rejects specs without NODE modules."""

    @pytest.mark.asyncio
    async def test_rejects_classical_spec(self, tmp_path: Path) -> None:
        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=2),
        )
        with pytest.raises(InvalidSpecError, match="NODE modules"):
            await runner.run(
                spec=_classical_spec(),
                data_manifest=_make_data_manifest(),
                initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
                seed=42,
            )


class TestConvergenceMetadata:
    """Convergence metadata is populated."""

    @pytest.mark.asyncio
    async def test_method_is_adam(self, tmp_path: Path) -> None:
        runner = NodeBackendRunner(
            work_dir=tmp_path,
            training_config=TrainingConfig(epochs=3, early_stop_patience=10),
        )
        result = await runner.run(
            spec=_node_elim_spec(),
            data_manifest=_make_data_manifest(),
            initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
            seed=42,
        )

        assert result.convergence_metadata.method == "adam"
        assert result.convergence_metadata.iterations > 0
        assert result.convergence_metadata.wall_time_seconds > 0
