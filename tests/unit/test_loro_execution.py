# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for LORO-CV execution engine (Phase 3 P3.B-4).

Tests the evaluate_loro_cv() function with mocked BackendRunner.
"""

from __future__ import annotations

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
    LOROFoldResult,
    ParameterEstimate,
    SplitManifest,
    SubjectAssignment,
)
from apmode.dsl.ast_models import IIV, DSLSpec, FirstOrder, LinearElim, OneCmt, Proportional
from apmode.evaluation.loro_cv import _aggregate_loro_metrics, evaluate_loro_cv


def _base_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test-base",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _mock_backend_result(
    model_id: str = "test",
    bic: float = 220.0,
    converged: bool = True,
    cwres_mean: float = 0.05,
    cwres_sd: float = 1.0,
) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=converged,
        ofv=-100.0,
        aic=210.0,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(
                name="CL",
                estimate=2.0,
                se=0.1,
                rse=5.0,
                category="structural",
            ),
            "V": ParameterEstimate(
                name="V",
                estimate=30.0,
                se=1.5,
                rse=5.0,
                category="structural",
            ),
        },
        eta_shrinkage={"CL": 15.0, "V": 20.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=converged,
            iterations=500,
            minimization_status="successful",
            wall_time_seconds=30.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=cwres_mean, cwres_sd=cwres_sd, outlier_fraction=0.02),
            identifiability=IdentifiabilityFlags(
                condition_number=50.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=30.0,
        backend_versions={"nlmixr2": "5.0.0"},
        initial_estimate_source="nca",
    )


def _mock_data_manifest() -> DataManifest:
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
        n_subjects=40,
        n_observations=400,
        n_doses=40,
    )


def _make_folds(n_groups: int = 4, subjects_per_group: int = 10) -> list[SplitManifest]:
    """Create synthetic LORO folds."""
    all_subjects = list(range(1, n_groups * subjects_per_group + 1))
    folds = []
    for g in range(n_groups):
        test_start = g * subjects_per_group + 1
        test_end = test_start + subjects_per_group
        test_subjects = set(range(test_start, test_end))

        assignments = [
            SubjectAssignment(
                subject_id=str(s),
                fold="test" if s in test_subjects else "train",
            )
            for s in all_subjects
        ]
        folds.append(
            SplitManifest(
                split_seed=42,
                split_strategy="regimen_level",
                assignments=assignments,
            )
        )
    return folds


class TestEvaluateLoroCV:
    """Tests for evaluate_loro_cv() with mocked runner."""

    @pytest.mark.asyncio
    async def test_runs_all_folds(self, tmp_path: Path) -> None:
        """Runner is called once per fold."""
        folds = _make_folds(n_groups=3)
        runner = AsyncMock()
        runner.run = AsyncMock(return_value=_mock_backend_result())

        result = await evaluate_loro_cv(
            candidate_spec=_base_spec(),
            candidate_result=_mock_backend_result(),
            folds=folds,
            runner=runner,
            data_manifest=_mock_data_manifest(),
            data_path=tmp_path / "data.csv",
            initial_estimates={"CL": 2.0, "V": 30.0, "ka": 1.0},
            seed=42,
        )

        assert runner.run.call_count == 3
        assert len(result.fold_results) == 3

    @pytest.mark.asyncio
    async def test_returns_loro_cv_result(self, tmp_path: Path) -> None:
        folds = _make_folds(n_groups=4)
        runner = AsyncMock()
        runner.run = AsyncMock(return_value=_mock_backend_result())

        result = await evaluate_loro_cv(
            candidate_spec=_base_spec(),
            candidate_result=_mock_backend_result(model_id="cand_001"),
            folds=folds,
            runner=runner,
            data_manifest=_mock_data_manifest(),
            data_path=tmp_path / "data.csv",
            initial_estimates={"CL": 2.0, "V": 30.0},
            seed=42,
        )

        assert result.candidate_id == "cand_001"
        assert result.metrics.n_folds == 4
        assert result.wall_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_handles_fold_failure(self, tmp_path: Path) -> None:
        """If a fold raises, it's recorded as non-converged."""
        folds = _make_folds(n_groups=3)
        runner = AsyncMock()
        call_count = 0

        async def mock_run(**kwargs: object) -> BackendResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                msg = "Fold 2 failed"
                raise RuntimeError(msg)
            return _mock_backend_result()

        runner.run = mock_run

        result = await evaluate_loro_cv(
            candidate_spec=_base_spec(),
            candidate_result=_mock_backend_result(),
            folds=folds,
            runner=runner,
            data_manifest=_mock_data_manifest(),
            data_path=tmp_path / "data.csv",
            initial_estimates={"CL": 2.0, "V": 30.0},
            seed=42,
        )

        assert len(result.fold_results) == 3
        converged = [f.converged for f in result.fold_results]
        assert converged == [True, False, True]

    @pytest.mark.asyncio
    async def test_warm_starts_from_fitted_estimates(self, tmp_path: Path) -> None:
        """Runner receives fitted structural estimates, not initial NCA."""
        folds = _make_folds(n_groups=2)
        runner = AsyncMock()
        runner.run = AsyncMock(return_value=_mock_backend_result())

        candidate = _mock_backend_result()
        await evaluate_loro_cv(
            candidate_spec=_base_spec(),
            candidate_result=candidate,
            folds=folds,
            runner=runner,
            data_manifest=_mock_data_manifest(),
            data_path=tmp_path / "data.csv",
            initial_estimates={"CL": 1.0, "V": 20.0},  # Different from fitted
            seed=42,
        )

        # The runner should receive the candidate's fitted estimates (CL=2.0, V=30.0)
        call_kwargs = runner.run.call_args_list[0][1]
        assert call_kwargs["initial_estimates"]["CL"] == 2.0
        assert call_kwargs["initial_estimates"]["V"] == 30.0


class TestAggregateLoroMetrics:
    """Tests for _aggregate_loro_metrics()."""

    def test_all_converged_folds(self) -> None:
        folds = [
            LOROFoldResult(
                fold_index=i,
                regimen_group=f"fold_{i}",
                n_train_subjects=30,
                n_test_subjects=10,
                converged=True,
                test_npde_mean=0.05 * (i + 1),
                test_npde_variance=1.0 + 0.02 * i,
                test_bic=200.0 + i * 5,
            )
            for i in range(4)
        ]
        metrics = _aggregate_loro_metrics(folds)
        assert metrics.n_folds == 4
        assert metrics.overall_pass is True
        assert metrics.n_total_test_subjects == 40

    def test_no_converged_folds(self) -> None:
        folds = [
            LOROFoldResult(
                fold_index=0,
                regimen_group="fold_0",
                n_train_subjects=30,
                n_test_subjects=10,
                converged=False,
            ),
        ]
        metrics = _aggregate_loro_metrics(folds)
        assert metrics.overall_pass is False
        assert metrics.vpc_coverage_concordance == 0.0

    def test_partial_convergence(self) -> None:
        folds = [
            LOROFoldResult(
                fold_index=0,
                regimen_group="fold_0",
                n_train_subjects=30,
                n_test_subjects=10,
                converged=True,
                test_npde_mean=0.05,
                test_npde_variance=1.0,
            ),
            LOROFoldResult(
                fold_index=1,
                regimen_group="fold_1",
                n_train_subjects=30,
                n_test_subjects=10,
                converged=False,
            ),
        ]
        metrics = _aggregate_loro_metrics(folds)
        assert metrics.overall_pass is False  # Not all folds converged
        assert metrics.vpc_coverage_concordance == 0.5  # 1/2 converged
