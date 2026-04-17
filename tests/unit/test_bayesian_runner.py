# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for BayesianRunner (src/apmode/backends/bayesian_runner.py).

Mocks the subprocess layer — does not require a live CmdStan installation.
Covers request/response schemas, error classification, NODE rejection, and
subprocess timeout handling.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from apmode.backends.bayesian_runner import (
    BayesianRunner,
    BayesianSubprocessRequest,
    BayesianSubprocessResponse,
)
from apmode.bundle.models import SamplerConfig
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    OneCmt,
    Proportional,
)
from apmode.errors import BackendTimeoutError, ConvergenceError, CrashError


def _classical_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_classical",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=20.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.3),
    )


def _node_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_node",
        absorption=NODEAbsorption(dim=4, constraint_template="bounded_positive"),
        distribution=OneCmt(V=20.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.3),
    )


def _mock_result_dict(model_id: str) -> dict[str, Any]:
    """Minimal valid dict that parses into BackendResult."""
    return {
        "model_id": model_id,
        "backend": "bayesian_stan",
        "converged": True,
        "parameter_estimates": {
            "CL": {
                "name": "CL",
                "estimate": 5.1,
                "posterior_sd": 0.2,
                "q05": 4.8,
                "q50": 5.1,
                "q95": 5.4,
                "category": "structural",
            }
        },
        "eta_shrinkage": {"CL": 0.02},
        "convergence_metadata": {
            "method": "nuts",
            "converged": True,
            "iterations": 1000,
            "minimization_status": "successful",
            "wall_time_seconds": 123.4,
        },
        "diagnostics": {
            "gof": {"cwres_mean": 0.0, "cwres_sd": 1.0, "outlier_fraction": 0.01},
            "identifiability": {
                "profile_likelihood_ci": {"CL": True},
                "ill_conditioned": False,
            },
            "blq": {"method": "none", "n_blq": 0, "blq_fraction": 0.0},
        },
        "wall_time_seconds": 123.4,
        "backend_versions": {"cmdstan": "2.36.0", "stan": "2.36.0"},
        "initial_estimate_source": "fallback",
    }


class TestSubprocessRequestSchema:
    def test_construct_minimal(self) -> None:
        cfg = SamplerConfig()
        req = BayesianSubprocessRequest(
            request_id="r1",
            run_id="r1",
            candidate_id="c1",
            spec={"model_id": "c1"},
            data_path="/tmp/data.csv",
            seed=42,
            initial_estimates={"CL": 5.0},
            compiled_stan_code="// stan",
            sampler_config=cfg,
            output_draws_path="/tmp/draws.parquet",
        )
        assert req.schema_version == "1.0"
        assert req.sampler_config.chains == 4


class TestSubprocessResponseSchema:
    def test_success_response(self) -> None:
        resp = BayesianSubprocessResponse(
            status="success",
            result=_mock_result_dict("m1"),
        )
        assert resp.status == "success"
        assert resp.error_type is None

    def test_error_response(self) -> None:
        resp = BayesianSubprocessResponse(
            status="error",
            error_type="convergence",
            error_detail="R-hat=2.3",
        )
        assert resp.error_type == "convergence"


class TestSamplerConfig:
    def test_defaults_aligned_with_stan_recommendations(self) -> None:
        cfg = SamplerConfig()
        assert cfg.chains == 4
        assert cfg.warmup == 1000
        assert cfg.sampling == 1000
        assert cfg.adapt_delta == 0.95
        assert cfg.max_treedepth == 12

    def test_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            SamplerConfig(chains=0)
        with pytest.raises(ValueError):
            SamplerConfig(adapt_delta=1.0)
        with pytest.raises(ValueError):
            SamplerConfig(max_treedepth=3)


class TestBayesianRunnerNODERejection:
    def test_run_rejects_node_spec(self, tmp_path: Path) -> None:
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(ValueError, match="NODE modules"):
            asyncio.run(
                runner.run(
                    spec=_node_spec(),
                    data_manifest=None,  # type: ignore[arg-type]
                    initial_estimates={"CL": 5.0},
                    seed=0,
                    data_path=tmp_path / "data.csv",
                )
            )

    def test_run_requires_data_path(self, tmp_path: Path) -> None:
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(ValueError, match="data_path"):
            asyncio.run(
                runner.run(
                    spec=_classical_spec(),
                    data_manifest=None,  # type: ignore[arg-type]
                    initial_estimates={"CL": 5.0},
                    seed=0,
                )
            )


class TestParseResponseErrorPaths:
    def test_missing_response_file_is_crash(self, tmp_path: Path) -> None:
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(CrashError):
            runner._parse_response(tmp_path / "nope.json", 137, _classical_spec())

    def test_error_response_maps_to_convergence(self, tmp_path: Path) -> None:
        response = {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "convergence",
            "error_detail": "R-hat=2.3",
            "session_info": {},
        }
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response))
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(ConvergenceError):
            runner._parse_response(path, 0, _classical_spec())

    def test_error_response_maps_to_crash(self, tmp_path: Path) -> None:
        response = {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "compile_error",
            "error_detail": "missing semicolon",
            "session_info": {},
        }
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response))
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(CrashError):
            runner._parse_response(path, 0, _classical_spec())

    def test_success_response_builds_backend_result(self, tmp_path: Path) -> None:
        response = {
            "schema_version": "1.0",
            "status": "success",
            "result": _mock_result_dict("m1"),
            "session_info": {},
        }
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response))
        runner = BayesianRunner(work_dir=tmp_path)
        result = runner._parse_response(path, 0, _classical_spec())
        assert result.backend == "bayesian_stan"
        assert result.converged is True

    def test_success_without_result_is_crash(self, tmp_path: Path) -> None:
        response = {
            "schema_version": "1.0",
            "status": "success",
            "session_info": {},
        }
        path = tmp_path / "response.json"
        path.write_text(json.dumps(response))
        runner = BayesianRunner(work_dir=tmp_path)
        with pytest.raises(CrashError):
            runner._parse_response(path, 0, _classical_spec())


class TestSubprocessTimeoutPath:
    def test_timeout_raises_backend_timeout_error(self, tmp_path: Path) -> None:
        """Simulate a subprocess that exceeds the timeout budget."""

        async def _mock_spawn(*args: object, **kwargs: object) -> object:
            class _Proc:
                pid = 12345
                returncode = None

                async def communicate(self) -> tuple[bytes, bytes]:
                    await asyncio.sleep(10)  # long enough to trigger timeout
                    return (b"", b"")

                async def wait(self) -> int:
                    return 0

            return _Proc()

        runner = BayesianRunner(work_dir=tmp_path)
        # Only patch the subprocess creation layer; the rest of _spawn_harness
        # handles the asyncio.wait_for + killpg wrapping.
        with (
            patch(
                "apmode.backends.bayesian_runner.asyncio.create_subprocess_exec",
                side_effect=_mock_spawn,
            ),
            patch("apmode.backends.bayesian_runner.os.killpg"),
            patch("apmode.backends.bayesian_runner.os.getpgid", return_value=12345),
            pytest.raises(BackendTimeoutError),
        ):
            asyncio.run(
                runner._spawn_harness(
                    request_path=tmp_path / "req.json",
                    response_path=tmp_path / "resp.json",
                    timeout_seconds=1,
                )
            )
