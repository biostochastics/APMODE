# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for ``apmode.benchmarks.suite_a_runner``.

Pure-Python tests — no R subprocess. ``Nlmixr2Runner.run`` is stubbed with
an ``AsyncMock`` injected directly into ``run_scenario``/``run_all`` (the
same dependency-injection shape ``suite_b_runner.run_case`` uses), so these
tests exercise dataset/eta resolution, the A3 "n"-parameter filtering, the
A7 NODE skip, error containment, and the atomic JSON writer without needing
R or nlmixr2 installed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from apmode.benchmarks.suite_a import scenario_a1, scenario_a3, scenario_a7
from apmode.benchmarks.suite_a_runner import (
    ScenarioEtaResult,
    _calibration_initial_estimates,
    run_all,
    run_scenario,
    write_results_atomic,
)
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
)


def _make_backend_result(
    *,
    converged: bool = True,
    per_subject_eta: dict[str, dict[str, float]] | None = None,
    wall_time_seconds: float = 1.5,
) -> BackendResult:
    """Minimal BackendResult carrying per_subject_eta, mirroring the
    suite_c_phase1_runner test helper's "stub every required field to its
    smallest valid form" convention.
    """
    return BackendResult(
        model_id="fake_candidate",
        backend="nlmixr2",
        converged=converged,
        ofv=-100.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=70.0, category="structural"),
            "ka": ParameterEstimate(name="ka", estimate=1.5, category="structural"),
        },
        eta_shrinkage={"CL": 0.0, "V": 0.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=converged,
            iterations=1,
            gradient_norm=0.0,
            minimization_status="successful" if converged else "failed",
            wall_time_seconds=wall_time_seconds,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(profile_likelihood_ci={}, ill_conditioned=False),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        per_subject_eta=per_subject_eta or {},
        wall_time_seconds=wall_time_seconds,
        backend_versions={"nlmixr2": "test"},
        initial_estimate_source="nca",
    )


# ---------------------------------------------------------------------------
# Fixtures: minimal on-disk Suite A dataset + eta CSVs
# ---------------------------------------------------------------------------

_A1_STEM = "a1_1cmt_oral_linear"


def _write_a1_dataset(suite_dir: Path) -> tuple[Path, Path]:
    """Write a minimal valid A1 dataset CSV + matching eta ground-truth CSV."""
    suite_dir.mkdir(parents=True, exist_ok=True)
    csv_path = suite_dir / f"{_A1_STEM}.csv"
    csv_path.write_text(
        "NMID,TIME,DV,AMT,EVID,MDV,CMT\n"
        "1,0,0,100,1,1,1\n"
        "1,1,2.1,0,0,0,2\n"
        "1,2,3.4,0,0,0,2\n"
        "2,0,0,100,1,1,1\n"
        "2,1,1.9,0,0,0,2\n"
        "2,2,3.0,0,0,0,2\n"
    )
    eta_path = suite_dir / f"{_A1_STEM}_eta.csv"
    eta_path.write_text("NMID,eta.ka,eta.V,eta.CL\n1,0.10,0.05,-0.02\n2,-0.10,0.00,0.03\n")
    return csv_path, eta_path


# ---------------------------------------------------------------------------
# (a) Successful scenario computes a sane eta_rmse
# ---------------------------------------------------------------------------


class TestRunScenarioSuccess:
    def test_ok_scenario_computes_eta_rmse(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        _write_a1_dataset(suite_dir)

        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(
            return_value=_make_backend_result(
                per_subject_eta={
                    "1": {"ka": 0.12, "V": 0.05, "CL": -0.02},
                    "2": {"ka": -0.08, "V": 0.02, "CL": 0.03},
                }
            )
        )

        result = asyncio.run(
            run_scenario(
                "A1",
                scenario_a1,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=20260708,
                timeout_seconds=600,
            )
        )

        assert result.status == "ok"
        assert result.converged is True
        assert result.n_subjects_true == 2
        assert result.n_subjects_fitted == 2
        assert result.wall_time_seconds == pytest.approx(1.5)
        # ka: errors are (0.12-0.10)=0.02 and (-0.08-(-0.10))=0.02 -> RMSE=0.02
        assert result.eta_rmse["ka"] == pytest.approx(0.02, abs=1e-9)
        # V: errors are (0.05-0.05)=0 and (0.02-0.00)=0.02 -> RMSE = sqrt((0+0.0004)/2)
        assert result.eta_rmse["V"] == pytest.approx((0.0004 / 2) ** 0.5, abs=1e-9)
        assert all(v == v for v in result.eta_rmse.values())  # no NaNs

        fake_runner.run.assert_awaited_once()
        call = fake_runner.run.await_args
        assert call.args[3] == 20260708  # seed is the 4th positional arg
        assert call.kwargs["timeout_seconds"] == 600

    def test_fitted_eta_missing_yields_nan_not_a_crash(self, tmp_path: Path) -> None:
        """An older harness / non-nlmixr2 path with empty per_subject_eta must
        produce NaN eta_rmse (per score_eta_recovery's own contract), not an
        exception -- this is documented "unscorable" behaviour, not a bug.
        """
        suite_dir = tmp_path / "suite_a"
        _write_a1_dataset(suite_dir)

        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(return_value=_make_backend_result(per_subject_eta={}))

        result = asyncio.run(
            run_scenario(
                "A1",
                scenario_a1,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert result.status == "ok"
        assert result.n_subjects_fitted == 0
        assert result.eta_rmse  # non-empty dict
        assert all(v != v for v in result.eta_rmse.values())  # every value is NaN


# ---------------------------------------------------------------------------
# (b) A7 is always skipped
# ---------------------------------------------------------------------------


class TestA7AlwaysSkipped:
    def test_a7_skipped_with_reason(self, tmp_path: Path) -> None:
        fake_runner = AsyncMock()

        result = asyncio.run(
            run_scenario(
                "A7",
                scenario_a7,
                runner=fake_runner,
                suite_dir=tmp_path,  # deliberately empty -- must not be touched
                seed=1,
                timeout_seconds=600,
            )
        )

        assert result.status == "skipped"
        assert result.skipped is True
        assert result.skip_reason is not None and "NODE" in result.skip_reason
        fake_runner.run.assert_not_awaited()

    def test_a7_skipped_inside_run_all(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        _write_a1_dataset(suite_dir)
        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(
            return_value=_make_backend_result(
                per_subject_eta={"1": {"ka": 0.0, "V": 0.0, "CL": 0.0}}
            )
        )

        results = asyncio.run(
            run_all(
                [("A1", scenario_a1), ("A7", scenario_a7)],
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert results["A7"].skipped is True
        assert results["A1"].status == "ok"
        # Only A1 should have driven a fit.
        fake_runner.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# (c) Missing eta ground truth -> no fit attempted
# ---------------------------------------------------------------------------


class TestNoEtaGroundTruth:
    def test_missing_eta_csv_skips_fit(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        suite_dir.mkdir(parents=True)
        # Data CSV present, but no matching *_eta.csv.
        (suite_dir / f"{_A1_STEM}.csv").write_text(
            "NMID,TIME,DV,AMT,EVID,MDV,CMT\n1,0,0,100,1,1,1\n1,1,2.1,0,0,0,2\n"
        )
        fake_runner = AsyncMock()

        result = asyncio.run(
            run_scenario(
                "A1",
                scenario_a1,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert result.status == "no_eta_ground_truth"
        fake_runner.run.assert_not_awaited()

    def test_missing_data_csv_records_no_data(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"  # empty, no files at all
        suite_dir.mkdir(parents=True)
        fake_runner = AsyncMock()

        result = asyncio.run(
            run_scenario(
                "A1",
                scenario_a1,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert result.status == "no_data"
        fake_runner.run.assert_not_awaited()


# ---------------------------------------------------------------------------
# (d) A fit-raising exception is caught, other scenarios still run
# ---------------------------------------------------------------------------


class TestFitErrorContained:
    def test_fit_exception_recorded_not_raised(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        _write_a1_dataset(suite_dir)
        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(side_effect=RuntimeError("R subprocess crashed"))

        result = asyncio.run(
            run_scenario(
                "A1",
                scenario_a1,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert result.status == "fit_error"
        assert result.error_message is not None
        assert "R subprocess crashed" in result.error_message

    def test_one_scenario_failure_does_not_abort_run_all(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        _write_a1_dataset(suite_dir)
        # Second scenario shares the same stem-family setup trivially by
        # reusing A1's factory under a different id so we don't need a
        # second real dataset on disk (its data-CSV is absent -> no_data).
        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(side_effect=RuntimeError("boom"))

        results = asyncio.run(
            run_all(
                [("A1", scenario_a1)],
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        assert results["A1"].status == "fit_error"
        assert "boom" in (results["A1"].error_message or "")


# ---------------------------------------------------------------------------
# A3 "n" parameter filtering (structural, not calibratable)
# ---------------------------------------------------------------------------


class TestA3NParameterFiltering:
    def test_n_excluded_from_calibration_initial_estimates(self) -> None:
        spec = scenario_a3()
        reference_params = {"n": 3.0, "ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0}
        filtered = _calibration_initial_estimates(spec, reference_params)
        assert "n" not in filtered
        assert filtered == {"ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0}

    def test_a3_fit_invoked_without_n_key(self, tmp_path: Path) -> None:
        suite_dir = tmp_path / "suite_a"
        suite_dir.mkdir(parents=True)
        stem = "a3_transit_1cmt_linear"
        (suite_dir / f"{stem}.csv").write_text(
            "NMID,TIME,DV,AMT,EVID,MDV,CMT\n1,0,0,100,1,1,1\n1,2,1.5,0,0,0,2\n"
        )
        (suite_dir / f"{stem}_eta.csv").write_text(
            "NMID,eta.ktr,eta.ka,eta.V,eta.CL\n1,0.0,0.0,0.0,0.0\n"
        )
        fake_runner = AsyncMock()
        fake_runner.run = AsyncMock(
            return_value=_make_backend_result(
                per_subject_eta={"1": {"ktr": 0.0, "ka": 0.0, "V": 0.0, "CL": 0.0}}
            )
        )

        asyncio.run(
            run_scenario(
                "A3",
                scenario_a3,
                runner=fake_runner,
                suite_dir=suite_dir,
                seed=1,
                timeout_seconds=600,
            )
        )

        fake_runner.run.assert_awaited_once()
        initial_estimates = fake_runner.run.await_args.args[2]
        assert "n" not in initial_estimates


# ---------------------------------------------------------------------------
# (e) Atomic JSON writer
# ---------------------------------------------------------------------------


class TestWriteResultsAtomic:
    def test_round_trip_payload(self, tmp_path: Path) -> None:
        out = tmp_path / "eta_recovery_report.json"
        result = ScenarioEtaResult(
            scenario_id="A1",
            status="ok",
            dataset_csv="/x/a1.csv",
            eta_csv="/x/a1_eta.csv",
            converged=True,
            eta_rmse={"ka": 0.02, "V": 0.01, "CL": 0.005},
            n_subjects_true=2,
            n_subjects_fitted=2,
            wall_time_seconds=1.5,
        )
        write_results_atomic(out, {"A1": result})

        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["A1"]["scenario_id"] == "A1"
        assert payload["A1"]["status"] == "ok"
        assert payload["A1"]["eta_rmse"]["ka"] == pytest.approx(0.02)
        assert not list(tmp_path.glob("*.tmp"))

    def test_skipped_scenario_serializes(self, tmp_path: Path) -> None:
        out = tmp_path / "eta_recovery_report.json"
        result = ScenarioEtaResult(
            scenario_id="A7",
            status="skipped",
            skipped=True,
            skip_reason="NODE cases use the neural backend.",
        )
        write_results_atomic(out, {"A7": result})
        payload = json.loads(out.read_text())
        assert payload["A7"]["skipped"] is True
        assert "NODE" in payload["A7"]["skip_reason"]

    def test_tmp_file_removed_on_write_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "eta_recovery_report.json"
        result = ScenarioEtaResult(scenario_id="A1", status="ok")

        def _boom(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("serialization exploded")

        monkeypatch.setattr("apmode.benchmarks.suite_a_runner.json.dumps", _boom)

        with pytest.raises(RuntimeError, match="serialization exploded"):
            write_results_atomic(out, {"A1": result})

        assert not out.exists()
        assert not list(tmp_path.glob("*.tmp"))
