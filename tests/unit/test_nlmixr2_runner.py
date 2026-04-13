# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Nlmixr2Runner subprocess backend (ARCHITECTURE.md S4.2)."""

import json
from pathlib import Path

import pytest

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.backends.protocol import BackendRunner
from apmode.bundle.models import ColumnMapping, DataManifest
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.errors import ConvergenceError, CrashError


def _test_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_model_id_0000000",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _test_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=30,
        n_observations=300,
        n_doses=60,
    )


class TestNlmixr2RunnerInit:
    def test_default_r_executable(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"))
        assert runner.r_executable == "Rscript"

    def test_custom_r_executable(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"), r_executable="/usr/local/bin/Rscript")
        assert runner.r_executable == "/usr/local/bin/Rscript"

    def test_default_harness_path(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"))
        assert runner.harness_path.name == "harness.R"

    def test_default_estimation(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"))
        assert runner.estimation == ["saem", "focei"]

    def test_custom_estimation(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"), estimation=["focei"])
        assert runner.estimation == ["focei"]


class TestNlmixr2RunnerProtocol:
    def test_implements_backend_runner(self) -> None:
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"))
        assert isinstance(runner, BackendRunner)


class TestParseResponse:
    """Test _parse_response directly (no subprocess needed)."""

    def _make_success_response(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status": "success",
            "error_type": None,
            "result": {
                "model_id": "test_model_id_0000000",
                "backend": "nlmixr2",
                "converged": True,
                "ofv": -1234.5,
                "aic": -1220.5,
                "bic": -1210.5,
                "parameter_estimates": {
                    "CL": {
                        "name": "CL",
                        "estimate": 5.1,
                        "se": 0.3,
                        "rse": 5.9,
                        "ci95_lower": 4.5,
                        "ci95_upper": 5.7,
                        "fixed": False,
                        "category": "structural",
                    },
                },
                "eta_shrinkage": {"CL": 0.12},
                "convergence_metadata": {
                    "method": "saem",
                    "converged": True,
                    "iterations": 200,
                    "gradient_norm": 0.001,
                    "minimization_status": "successful",
                    "wall_time_seconds": 45.2,
                },
                "diagnostics": {
                    "gof": {
                        "cwres_mean": 0.01,
                        "cwres_sd": 1.02,
                        "outlier_fraction": 0.02,
                        "obs_vs_pred_r2": 0.95,
                    },
                    "vpc": None,
                    "identifiability": {
                        "condition_number": 12.5,
                        "profile_likelihood_ci": {"CL": True},
                        "ill_conditioned": False,
                    },
                    "blq": {
                        "method": "none",
                        "lloq": None,
                        "n_blq": 0,
                        "blq_fraction": 0.0,
                    },
                    "diagnostic_plots": {},
                },
                "wall_time_seconds": 45.2,
                "backend_versions": {"nlmixr2": "3.0.0", "R": "4.4.1"},
                "initial_estimate_source": "nca",
            },
            "r_session_info": {
                "r_version": "4.4.1",
                "nlmixr2_version": "3.0.0",
                "platform": "aarch64-apple-darwin",
                "packages": {},
            },
            "random_seed_state": [1, 2, 3],
        }

    def test_success_response(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"
        response_path.write_text(json.dumps(self._make_success_response()))

        result = runner._parse_response(response_path, 0, "test_model_id_0000000")
        assert result.converged is True
        assert result.ofv == -1234.5
        assert "CL" in result.parameter_estimates

    def test_missing_response_raises_crash(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "nonexistent.json"

        with pytest.raises(CrashError, match=r"no response\.json"):
            runner._parse_response(response_path, 139, "test_model")

    def test_convergence_error_response(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"
        response_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "error",
                    "error_type": "convergence",
                    "result": None,
                    "r_session_info": {
                        "r_version": "4.4.1",
                        "nlmixr2_version": "3.0.0",
                        "platform": "test",
                        "packages": {},
                    },
                    "random_seed_state": None,
                }
            )
        )

        with pytest.raises(ConvergenceError, match="convergence failure"):
            runner._parse_response(response_path, 1, "test_model")

    def test_crash_error_response(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"
        response_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "error",
                    "error_type": "crash",
                    "result": None,
                    "r_session_info": {
                        "r_version": "4.4.1",
                        "nlmixr2_version": "3.0.0",
                        "platform": "test",
                        "packages": {},
                    },
                    "random_seed_state": None,
                }
            )
        )

        with pytest.raises(CrashError, match="R backend error"):
            runner._parse_response(response_path, 1, "test_model")

    def test_success_with_null_result_raises_crash(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"
        response_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "success",
                    "error_type": None,
                    "result": None,
                    "r_session_info": {
                        "r_version": "4.4.1",
                        "nlmixr2_version": "3.0.0",
                        "platform": "test",
                        "packages": {},
                    },
                    "random_seed_state": None,
                }
            )
        )

        with pytest.raises(CrashError, match="no result payload"):
            runner._parse_response(response_path, 0, "test_model")

    def test_exit_code_in_crash_error(self, tmp_path: Path) -> None:
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "nonexistent.json"

        with pytest.raises(CrashError) as exc_info:
            runner._parse_response(response_path, 139, "test_model")
        assert exc_info.value.exit_code == 139


class TestNlmixr2RunnerRequestCreation:
    """Test that run() creates correct request.json (mock the subprocess)."""

    @pytest.mark.asyncio
    async def test_creates_request_json(self, tmp_path: Path) -> None:
        # Mock R harness: receives (harness_path, request_path, response_path)
        # The runner calls: r_executable harness_path request_path response_path
        # So in shell: $1=harness, $2=request, $3=response
        script = tmp_path / "noop.sh"
        script.write_text(
            "#!/bin/sh\n"
            "cat > \"$3\" << 'RESP'\n"
            '{"schema_version":"1.0","status":"error","error_type":"convergence",'
            '"result":null,"r_session_info":{"r_version":"4.4.1",'
            '"nlmixr2_version":"3.0.0","platform":"test","packages":{}},'
            '"random_seed_state":null}\n'
            "RESP\n"
        )
        script.chmod(0o755)

        runner = Nlmixr2Runner(
            work_dir=tmp_path / "work",
            r_executable=str(script),
            harness_path=Path("/dev/null"),  # mock ignores this
        )

        with pytest.raises(ConvergenceError):
            await runner.run(
                spec=_test_spec(),
                data_manifest=_test_manifest(),
                initial_estimates={"CL": 5.0, "V": 70.0},
                seed=42,
                data_path=Path("/data/test.csv"),
            )

        # Verify request.json was created
        work_dirs = list((tmp_path / "work").iterdir())
        assert len(work_dirs) == 1
        request_path = work_dirs[0] / "request.json"
        assert request_path.exists()

        req_data = json.loads(request_path.read_text())
        assert req_data["seed"] == 42
        assert req_data["candidate_id"] == "test_model_id_0000000"
        assert "compiled_r_code" in req_data
        assert "ini({" in req_data["compiled_r_code"]
