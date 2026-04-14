# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for parallel model evaluation in SearchEngine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
)
from apmode.dsl.ast_models import IIV, DSLSpec, FirstOrder, LinearElim, OneCmt, Proportional
from apmode.search.engine import SearchEngine


def _make_spec(model_id: str) -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _make_result(model_id: str, bic: float = 200.0) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=True,
        ofv=-100.0,
        aic=190.0,
        bic=bic,
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
            converged=True,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=10.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.01, cwres_sd=1.0, outlier_fraction=0.02),
            identifiability=IdentifiabilityFlags(
                condition_number=10.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=10.0,
        n_observations=100,
        n_subjects=20,
        backend_versions={"nlmixr2": "5.0.0"},
        initial_estimate_source="nca",
    )


def _make_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        n_subjects=20,
        n_observations=100,
        n_doses=20,
        column_mapping=ColumnMapping(
            subject_id="ID",
            time="TIME",
            dv="DV",
            amt="AMT",
            evid="EVID",
        ),
    )


class TestSearchEngineParallel:
    """Test that max_concurrency controls parallel evaluation."""

    @pytest.mark.asyncio
    async def test_sequential_default(self) -> None:
        """max_concurrency=1 runs candidates sequentially."""
        call_order: list[str] = []

        async def mock_run(**kwargs: object) -> BackendResult:
            model_id = getattr(kwargs["spec"], "model_id", "unknown")
            call_order.append(f"start_{model_id}")
            await asyncio.sleep(0.01)
            call_order.append(f"end_{model_id}")
            return _make_result(model_id)

        runner = AsyncMock()
        runner.run = mock_run

        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=1,
        )

        specs = [_make_spec(f"m{i}") for i in range(3)]
        results = await engine._gather_evaluations([(s, {"CL": 2.0, "V": 30.0}) for s in specs])

        assert len(results) == 3
        # Sequential: each start-end pair completes before the next starts
        assert call_order == [
            "start_m0",
            "end_m0",
            "start_m1",
            "end_m1",
            "start_m2",
            "end_m2",
        ]

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        """max_concurrency>1 allows overlapping execution."""
        active = 0
        max_active = 0

        async def mock_run(**kwargs: object) -> BackendResult:
            nonlocal active, max_active
            model_id = getattr(kwargs["spec"], "model_id", "unknown")
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return _make_result(model_id)

        runner = AsyncMock()
        runner.run = mock_run

        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=4,
        )

        specs = [_make_spec(f"m{i}") for i in range(4)]
        results = await engine._gather_evaluations([(s, {"CL": 2.0, "V": 30.0}) for s in specs])

        assert len(results) == 4
        assert all(r.converged for r in results)
        # With 4 concurrent and 4 tasks, all should overlap
        assert max_active > 1

    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrency(self) -> None:
        """Semaphore limits active tasks to max_concurrency."""
        active = 0
        max_active = 0

        async def mock_run(**kwargs: object) -> BackendResult:
            nonlocal active, max_active
            model_id = getattr(kwargs["spec"], "model_id", "unknown")
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return _make_result(model_id)

        runner = AsyncMock()
        runner.run = mock_run

        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=2,
        )

        specs = [_make_spec(f"m{i}") for i in range(6)]
        results = await engine._gather_evaluations([(s, {"CL": 2.0, "V": 30.0}) for s in specs])

        assert len(results) == 6
        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_parallel_preserves_result_order(self) -> None:
        """Results maintain input order regardless of completion order."""
        import random

        async def mock_run(**kwargs: object) -> BackendResult:
            model_id = getattr(kwargs["spec"], "model_id", "unknown")
            # Random delay to ensure different completion order
            await asyncio.sleep(random.uniform(0.01, 0.05))
            bic = float(model_id.replace("m", "")) * 10 + 100
            return _make_result(model_id, bic=bic)

        runner = AsyncMock()
        runner.run = mock_run

        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=4,
        )

        specs = [_make_spec(f"m{i}") for i in range(5)]
        results = await engine._gather_evaluations([(s, {"CL": 2.0, "V": 30.0}) for s in specs])

        # Results must be in same order as input specs
        assert [r.candidate_id for r in results] == [f"m{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel(self) -> None:
        """Backend errors in one candidate don't crash others."""
        call_count = 0

        async def mock_run(**kwargs: object) -> BackendResult:
            nonlocal call_count
            model_id = getattr(kwargs["spec"], "model_id", "unknown")
            call_count += 1
            if model_id == "m1":
                from apmode.errors import BackendError

                raise BackendError("simulated failure")
            return _make_result(model_id)

        runner = AsyncMock()
        runner.run = mock_run

        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=3,
        )

        specs = [_make_spec(f"m{i}") for i in range(3)]
        results = await engine._gather_evaluations([(s, {"CL": 2.0, "V": 30.0}) for s in specs])

        assert len(results) == 3
        # m0 and m2 should succeed, m1 should have error
        assert results[0].converged is True
        assert results[1].error is not None
        assert results[2].converged is True

    @pytest.mark.asyncio
    async def test_empty_tasks(self) -> None:
        """Empty task list returns empty results."""
        runner = AsyncMock()
        engine = SearchEngine(
            runner=runner,
            data_manifest=_make_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            max_concurrency=4,
        )
        results = await engine._gather_evaluations([])
        assert results == []
