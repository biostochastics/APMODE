# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Nlmixr2Runner subprocess backend (ARCHITECTURE.md S4.2)."""

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not on PATH — #22 defence-in-depth makes Nlmixr2Runner.__init__ require it",
)

from apmode.backends.nlmixr2_runner import Nlmixr2Runner  # noqa: E402
from apmode.backends.protocol import BackendRunner  # noqa: E402
from apmode.bundle.models import ColumnMapping, DataManifest  # noqa: E402
from apmode.dsl.ast_models import (  # noqa: E402
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.errors import ConvergenceError, CrashError  # noqa: E402


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
        # #22: __init__ now resolves the executable to an absolute path
        # via shutil.which so subprocess spawning never performs a
        # runtime PATH lookup. The resolved path ends with "Rscript".
        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"))
        assert Path(runner.r_executable).name == "Rscript"
        assert Path(runner.r_executable).is_absolute()

    def test_custom_r_executable(self) -> None:
        # An absolute path must be accepted as-is (after an existence check).
        # Use sys.executable as a stable absolute-file fixture that exists
        # on any developer machine; we only care that the runner preserves
        # the path, not that it points at a real Rscript binary.
        import sys

        runner = Nlmixr2Runner(work_dir=Path("/tmp/apmode"), r_executable=sys.executable)
        assert runner.r_executable == sys.executable

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

        # The runner now pre-adapts the on-disk CSV (NMID -> ID, DVID
        # filter, categorical remap), so it must actually open the file.
        # An NMID-shaped two-row CSV exercises the rename path.
        train_csv = tmp_path / "train.csv"
        train_csv.write_text("NMID,TIME,DV,AMT,EVID,MDV,CMT\n1,0.0,0,1,1,1,1\n1,1.0,5,0,0,0,1\n")

        with pytest.raises(ConvergenceError):
            await runner.run(
                spec=_test_spec(),
                data_manifest=_test_manifest(),
                initial_estimates={"CL": 5.0, "V": 70.0},
                seed=42,
                data_path=train_csv,
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
        # New rc8 field: n_posterior_predictive_sims defaults to 0 when no
        # Gate3Config policy is threaded through (CLI one-shot runs).
        assert req_data["n_posterior_predictive_sims"] == 0
        # New honest-mode fields default to absent / False so the
        # request shape stays bit-identical to the v0.6 baseline when
        # the caller does not opt in.
        assert req_data.get("test_data_path") is None
        assert req_data.get("fixed_parameter") is False
        # The runner pre-adapts the on-disk CSV (NMID -> ID, DVID filter,
        # categorical remap) and writes the adapted copy to the run's
        # scratch ``data_nlmixr2.csv`` to keep nlmixr2's column-name
        # contract; the request should point at the adapted path so the
        # harness reads the renamed columns.
        adapted = work_dirs[0] / "data_nlmixr2.csv"
        assert adapted.exists()
        assert req_data["data_path"] == str(adapted)
        assert "ID," in adapted.read_text()
        assert "NMID," not in adapted.read_text()

    @pytest.mark.asyncio
    async def test_request_carries_test_data_and_fixed_parameter(self, tmp_path: Path) -> None:
        """Honest-mode kwargs survive the round trip to request.json verbatim.

        The R harness only honours these when present and explicitly set,
        so a regression that drops them silently would re-introduce the
        in-sample / warm-start tautology the v0.6.1 work fixed.
        """
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
            harness_path=Path("/dev/null"),
        )
        train_csv = tmp_path / "train.csv"
        train_csv.write_text("NMID,TIME,DV,AMT,EVID,MDV,CMT\n1,0.0,0,1,1,1,1\n1,1.0,5,0,0,0,1\n")
        test_csv = tmp_path / "held_out.csv"
        test_csv.write_text("NMID,TIME,DV,AMT,EVID,MDV,CMT\n99,0.0,0,1,1,1,1\n")
        with pytest.raises(ConvergenceError):
            await runner.run(
                spec=_test_spec(),
                data_manifest=_test_manifest(),
                initial_estimates={"CL": 5.0, "V": 70.0},
                seed=42,
                data_path=train_csv,
                test_data_path=test_csv,
                fixed_parameter=True,
            )
        work_dirs = list((tmp_path / "work").iterdir())
        assert len(work_dirs) == 1
        req_data = json.loads((work_dirs[0] / "request.json").read_text())
        # The runner adapts both train and test CSVs in the work_dir; the
        # request.json fields point at the adapted (NMID -> ID renamed)
        # copies so the harness reads canonical-ID-column data.
        adapted_train = work_dirs[0] / "data_nlmixr2.csv"
        adapted_test = work_dirs[0] / "test_data_nlmixr2.csv"
        assert adapted_train.exists()
        assert adapted_test.exists()
        assert req_data["data_path"] == str(adapted_train)
        assert req_data["test_data_path"] == str(adapted_test)
        assert req_data["fixed_parameter"] is True
        # Persisted train/test CSVs at the caller's path are byte-identical
        # to what they wrote — only the in-runner copy is adapted.
        assert "NMID," in train_csv.read_text()
        assert "NMID," in test_csv.read_text()

    @pytest.mark.asyncio
    async def test_test_data_path_must_be_absolute(self, tmp_path: Path) -> None:
        """Relative ``test_data_path`` is rejected at the runner boundary.

        Catches the bug class where a CLI tool passes a path resolved
        against a soon-to-change CWD; the harness reads paths verbatim
        so we want the failure surfaced before the subprocess spawn.
        """
        script = tmp_path / "noop.sh"
        script.write_text("#!/bin/sh\necho should-not-run\n")
        script.chmod(0o755)
        runner = Nlmixr2Runner(
            work_dir=tmp_path / "work",
            r_executable=str(script),
            harness_path=Path("/dev/null"),
        )
        with pytest.raises(ValueError, match="test_data_path must be absolute"):
            await runner.run(
                spec=_test_spec(),
                data_manifest=_test_manifest(),
                initial_estimates={"CL": 5.0, "V": 70.0},
                seed=42,
                data_path=Path("/data/test.csv"),
                test_data_path=Path("relative/test.csv"),
            )


class TestParseResponseWithPredictedSimulations:
    """rc8 wiring: Nlmixr2Runner populates VPC/NPE/AUC-Cmax from R harness sims."""

    def _base_result(self) -> dict[str, object]:
        return {
            "model_id": "test_model_id_0000000",
            "backend": "nlmixr2",
            "converged": True,
            "ofv": 100.0,
            "aic": 110.0,
            "bic": 120.0,
            "parameter_estimates": {
                "CL": {
                    "name": "CL",
                    "estimate": 5.0,
                    "se": 0.3,
                    "rse": 6.0,
                    "ci95_lower": 4.5,
                    "ci95_upper": 5.5,
                    "fixed": False,
                    "category": "structural",
                },
            },
            "eta_shrinkage": {"CL": 0.05},
            "convergence_metadata": {
                "method": "saem",
                "converged": True,
                "iterations": 200,
                "gradient_norm": 0.001,
                "minimization_status": "successful",
                "wall_time_seconds": 10.0,
            },
            "diagnostics": {
                "gof": {
                    "cwres_mean": 0.01,
                    "cwres_sd": 1.0,
                    "outlier_fraction": 0.02,
                    "obs_vs_pred_r2": 0.95,
                },
                "vpc": None,
                "identifiability": {
                    "condition_number": 10.0,
                    "profile_likelihood_ci": {"CL": True},
                    "ill_conditioned": False,
                },
                "blq": {"method": "none", "lloq": None, "n_blq": 0, "blq_fraction": 0.0},
                "diagnostic_plots": {},
            },
            "wall_time_seconds": 10.0,
            "backend_versions": {"nlmixr2": "3.0.0", "R": "4.4.1"},
            "initial_estimate_source": "nca",
        }

    def _predicted_sims_cohort(
        self, *, n_subjects: int, n_obs: int, n_sims: int
    ) -> list[dict[str, object]]:
        import numpy as np

        rng = np.random.default_rng(42)
        out: list[dict[str, object]] = []
        for i in range(n_subjects):
            times = list(np.linspace(0.5, 10.0, n_obs))
            observed = [5.0] * n_obs
            sims = rng.normal(loc=5.0, scale=0.1, size=(n_sims, n_obs)).tolist()
            out.append(
                {
                    "subject_id": f"s{i}",
                    "t_observed": times,
                    "observed_dv": observed,
                    "sims_at_observed": sims,
                }
            )
        return out

    def _wrap_response(self, result: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status": "success",
            "error_type": None,
            "result": result,
            "r_session_info": {
                "r_version": "4.4.1",
                "nlmixr2_version": "3.0.0",
                "platform": "test",
                "packages": {},
            },
            "random_seed_state": [1, 2, 3],
        }

    def test_without_policy_ignores_predicted_simulations(self, tmp_path: Path) -> None:
        """No Gate3Config → VPC/NPE stay None even if harness emitted sims."""
        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"

        result = self._base_result()
        result["predicted_simulations"] = self._predicted_sims_cohort(
            n_subjects=10, n_obs=5, n_sims=20
        )
        response_path.write_text(json.dumps(self._wrap_response(result)))

        backend_result = runner._parse_response(response_path, 0, "test_model_id_0000000")
        assert backend_result.diagnostics.vpc is None
        assert backend_result.diagnostics.npe_score is None
        assert backend_result.diagnostics.auc_cmax_be_score is None

    def test_with_policy_populates_all_three_diagnostics(self, tmp_path: Path) -> None:
        """Gate3Config + sims → VPC, npe_score, auc_cmax_be_score all set atomically."""
        from apmode.bundle.models import NCASubjectDiagnostic
        from apmode.governance.policy import Gate3Config

        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"

        # 12 subjects all admissible → passes the 8-floor AND 0.5-fraction.
        result = self._base_result()
        result["predicted_simulations"] = self._predicted_sims_cohort(
            n_subjects=12, n_obs=6, n_sims=30
        )
        response_path.write_text(json.dumps(self._wrap_response(result)))

        policy = Gate3Config(
            composite_method="weighted_sum",
            vpc_weight=0.5,
            npe_weight=0.5,
            bic_weight=0.0,
            auc_cmax_weight=0.0,
            n_posterior_predictive_sims=100,
            vpc_n_bins=4,
        )
        diagnostics = [NCASubjectDiagnostic(subject_id=f"s{i}", excluded=False) for i in range(12)]

        backend_result = runner._parse_response(
            response_path,
            0,
            "test_model_id_0000000",
            gate3_policy=policy,
            nca_diagnostics=diagnostics,
        )
        assert backend_result.diagnostics.vpc is not None
        assert backend_result.diagnostics.npe_score is not None
        # All subjects eligible + observed ≈ sim mean → score should pass BE.
        assert backend_result.diagnostics.auc_cmax_be_score is not None
        assert backend_result.diagnostics.auc_cmax_source == "observed_trapezoid"

    def test_null_predicted_simulations_keeps_baseline_diagnostics(self, tmp_path: Path) -> None:
        """R harness emitted predicted_simulations=null → non-fatal, no VPC/NPE."""
        from apmode.governance.policy import Gate3Config

        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"

        result = self._base_result()
        result["predicted_simulations"] = None  # R sim failed
        response_path.write_text(json.dumps(self._wrap_response(result)))

        policy = Gate3Config(
            composite_method="weighted_sum",
            vpc_weight=0.5,
            npe_weight=0.5,
            bic_weight=0.0,
            auc_cmax_weight=0.0,
            n_posterior_predictive_sims=100,
        )
        backend_result = runner._parse_response(
            response_path, 0, "test_model_id_0000000", gate3_policy=policy
        )
        # Baseline diagnostics still populated; VPC/NPE remain unset.
        assert backend_result.diagnostics.vpc is None
        assert backend_result.diagnostics.npe_score is None

    def test_below_floor_drops_auc_cmax_keeps_vpc_npe(self, tmp_path: Path) -> None:
        """Few eligible subjects → VPC + NPE still emit; auc_cmax_be_score None."""
        from apmode.bundle.models import NCASubjectDiagnostic
        from apmode.governance.policy import Gate3Config

        runner = Nlmixr2Runner(work_dir=tmp_path)
        response_path = tmp_path / "response.json"

        result = self._base_result()
        result["predicted_simulations"] = self._predicted_sims_cohort(
            n_subjects=12, n_obs=5, n_sims=20
        )
        response_path.write_text(json.dumps(self._wrap_response(result)))

        # 12 subjects in cohort but only 2 eligible (fraction 2/12 ≈ 0.17)
        # → below 0.5 fraction floor AND 8 absolute floor.
        diagnostics: list[NCASubjectDiagnostic] = []
        for i in range(12):
            diagnostics.append(
                NCASubjectDiagnostic(
                    subject_id=f"s{i}",
                    excluded=(i >= 2),
                    excluded_reason="auc_extrap>20%" if i >= 2 else None,
                )
            )

        policy = Gate3Config(
            composite_method="weighted_sum",
            vpc_weight=0.5,
            npe_weight=0.5,
            bic_weight=0.0,
            auc_cmax_weight=0.0,
            n_posterior_predictive_sims=100,
            vpc_n_bins=4,
            auc_cmax_nca_min_eligible=8,
            auc_cmax_nca_min_eligible_fraction=0.5,
        )
        backend_result = runner._parse_response(
            response_path,
            0,
            "test_model_id_0000000",
            gate3_policy=policy,
            nca_diagnostics=diagnostics,
        )
        assert backend_result.diagnostics.vpc is not None
        assert backend_result.diagnostics.npe_score is not None
        assert backend_result.diagnostics.auc_cmax_be_score is None
        assert backend_result.diagnostics.auc_cmax_source is None
