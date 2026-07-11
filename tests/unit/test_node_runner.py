# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NodeBackendRunner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=NODEElimination(dim=3, constraint_template="bounded_positive"),
        variability=[IIV(params=["ka", "V"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


def _node_abs_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_node_abs",
        absorption=NODEAbsorption(dim=3, constraint_template="monotone_decreasing"),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["V", "CL"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


def _classical_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_classical",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
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


def _infusion_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="e" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            cmt="CMT",
            rate="RATE",
            ss="SS",
        ),
        n_subjects=1,
        n_observations=4,
        n_doses=1,
    )


def _manifest_no_ss() -> DataManifest:
    """Infusion-capable manifest that does NOT map a steady-state column."""
    return DataManifest(
        data_sha256="f" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            cmt="CMT",
            rate="RATE",
        ),
        n_subjects=1,
        n_observations=4,
        n_doses=1,
    )


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


class TestInfusionLoading:
    """The RATE>0 hard reject is gone; infusions load as dose events."""

    def test_loads_infusion_csv_with_stop_event(self, tmp_path: Path) -> None:
        # 100 mg over 10 h -> RATE=10 into central (CMT=2).
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1, 1],
                "TIME": [0.0, 2.0, 6.0, 10.0, 20.0],
                "EVID": [1, 0, 0, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 0.0, 0.0],
                "RATE": [10.0, 0.0, 0.0, 0.0, 0.0],
                "CMT": [2, 1, 1, 1, 1],
                "DV": [0.0, 1.5, 3.0, 3.2, 1.0],
            }
        )
        csv = _write_csv(tmp_path / "infusion.csv", df)

        runner = NodeBackendRunner(work_dir=tmp_path)
        subjects = runner._load_subjects_from_csv(csv, _infusion_manifest(), n_cmt=1)

        assert len(subjects) == 1
        events = subjects[0]["dose_events"]
        # An infusion start (positive rate) and a synthetic stop (negative rate).
        starts = [e for e in events if e[4] > 0]
        stops = [e for e in events if e[4] < 0]
        assert len(starts) == 1
        assert len(stops) == 1
        assert starts[0][4] == pytest.approx(10.0)
        assert stops[0][0] == pytest.approx(10.0)  # stop at TIME + DUR = 0 + 10


class TestRejectsUnsupportedRows:
    """SS, EVID=2, and non-central observations must fail loudly."""

    def test_rejects_ss_rows(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 2.0, 6.0],
                "EVID": [1, 0, 0],
                "AMT": [100.0, 0.0, 0.0],
                "RATE": [0.0, 0.0, 0.0],
                "CMT": [1, 1, 1],
                "SS": [1, 0, 0],
                "DV": [0.0, 3.0, 2.0],
            }
        )
        csv = _write_csv(tmp_path / "ss.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        with pytest.raises(InvalidSpecError, match="steady-state"):
            runner._load_subjects_from_csv(csv, _infusion_manifest(), n_cmt=1)

    def test_rejects_evid2_rows(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1],
                "TIME": [0.0, 2.0, 4.0, 6.0],
                "EVID": [1, 2, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 0.0],
                "RATE": [0.0, 0.0, 0.0, 0.0],
                "CMT": [1, 1, 1, 1],
                "SS": [0, 0, 0, 0],
                "DV": [0.0, 0.0, 3.0, 2.0],
            }
        )
        csv = _write_csv(tmp_path / "evid2.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        with pytest.raises(InvalidSpecError, match="EVID=2"):
            runner._load_subjects_from_csv(csv, _infusion_manifest(), n_cmt=1)

    def test_rejects_observation_cmt_not_1(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 2.0, 6.0],
                "EVID": [1, 0, 0],
                "AMT": [100.0, 0.0, 0.0],
                "RATE": [0.0, 0.0, 0.0],
                "CMT": [1, 2, 2],
                "SS": [0, 0, 0],
                "DV": [0.0, 3.0, 2.0],
            }
        )
        csv = _write_csv(tmp_path / "obscmt.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        with pytest.raises(InvalidSpecError, match="observation"):
            runner._load_subjects_from_csv(csv, _infusion_manifest(), n_cmt=1)

    def test_rejects_infusion_into_depot(self, tmp_path: Path) -> None:
        """An IV infusion labelled CMT=1 lands in the absorption depot (index 0)
        instead of central, producing an absorption-delayed, materially wrong
        curve. It must be rejected rather than silently mis-solved.
        """
        # 100 mg over 10 h, but the dose row is (wrongly) CMT=1 (depot).
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1, 1],
                "TIME": [0.0, 2.0, 6.0, 10.0, 20.0],
                "EVID": [1, 0, 0, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 0.0, 0.0],
                "RATE": [10.0, 0.0, 0.0, 0.0, 0.0],
                "CMT": [1, 1, 1, 1, 1],
                "DV": [0.0, 1.5, 3.0, 3.2, 1.0],
            }
        )
        csv = _write_csv(tmp_path / "iv_depot.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        with pytest.raises(InvalidSpecError, match="central compartment"):
            runner._load_subjects_from_csv(csv, _manifest_no_ss(), n_cmt=1)

    def test_infusion_into_central_is_accepted(self, tmp_path: Path) -> None:
        """The correct convention (infusion dose CMT=2 -> central) still loads."""
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1, 1],
                "TIME": [0.0, 2.0, 6.0, 10.0, 20.0],
                "EVID": [1, 0, 0, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 0.0, 0.0],
                "RATE": [10.0, 0.0, 0.0, 0.0, 0.0],
                "CMT": [2, 1, 1, 1, 1],
                "DV": [0.0, 1.5, 3.0, 3.2, 1.0],
            }
        )
        csv = _write_csv(tmp_path / "iv_central.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        subjects = runner._load_subjects_from_csv(csv, _manifest_no_ss(), n_cmt=1)
        assert len(subjects) == 1

    def test_benign_ss_column_not_rejected(self, tmp_path: Path) -> None:
        """A dataset with a benign covariate column literally named 'SS' that is
        NOT mapped as the steady-state control column must load, not be
        false-rejected as steady-state.
        """
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 2.0, 6.0],
                "EVID": [1, 0, 0],
                "AMT": [100.0, 0.0, 0.0],
                "RATE": [0.0, 0.0, 0.0],
                "CMT": [1, 1, 1],  # oral bolus into depot is fine
                "SS": [99, 99, 99],  # covariate, NOT a steady-state flag
                "DV": [0.0, 3.0, 2.0],
            }
        )
        csv = _write_csv(tmp_path / "benign_ss.csv", df)
        runner = NodeBackendRunner(work_dir=tmp_path)
        # _manifest_no_ss() does not map cm.ss, so the 'SS' column is a covariate.
        subjects = runner._load_subjects_from_csv(csv, _manifest_no_ss(), n_cmt=1)
        assert len(subjects) == 1


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
