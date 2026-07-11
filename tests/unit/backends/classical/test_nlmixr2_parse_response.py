# SPDX-License-Identifier: GPL-2.0-or-later
"""R-less coverage for ``Nlmixr2Runner._parse_response``.

The sibling ``tests/unit/test_nlmixr2_runner.py`` carries a *file-level*
``pytest.mark.skipif(shutil.which("Rscript") is None)`` because
``Nlmixr2Runner.__init__`` resolves and existence-checks an R executable
(#22 defence-in-depth). That skip hides two classes that parse *fabricated*
JSON and need no live R at all: ``TestParseResponse`` and
``TestParseResponseWithPredictedSimulations``.

This module re-exercises the same ``_parse_response`` contract without any
Rscript dependency, so the JSON-only parse path (rc8 VPC/NPE/AUC/NPDE ->
``DiagnosticBundle`` wiring, error/crash/convergence dispatch) runs on
R-less CI. It constructs the runner with ``r_executable=sys.executable`` —
an absolute, existing binary that ``__init__`` accepts unchanged (mirrors
``test_nlmixr2_runner.py::TestNlmixr2RunnerInit::test_custom_r_executable``).
``_parse_response`` never references ``r_executable``, so the substitution
is semantically inert for everything under test here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.errors import ConvergenceError, CrashError


def _make_runner(work_dir: Path) -> Nlmixr2Runner:
    """Construct a runner without requiring Rscript on PATH.

    ``sys.executable`` is an absolute file that exists on any machine; the
    parse path under test never spawns it.
    """
    return Nlmixr2Runner(work_dir=work_dir, r_executable=sys.executable)


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
        runner = _make_runner(tmp_path)
        response_path = tmp_path / "response.json"
        response_path.write_text(json.dumps(self._make_success_response()))

        result = runner._parse_response(response_path, 0, "test_model_id_0000000")
        assert result.converged is True
        assert result.ofv == -1234.5
        assert "CL" in result.parameter_estimates

    def test_missing_response_raises_crash(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        response_path = tmp_path / "nonexistent.json"

        with pytest.raises(CrashError, match=r"no response\.json"):
            runner._parse_response(response_path, 139, "test_model")

    def test_convergence_error_response(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
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
        runner = _make_runner(tmp_path)
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
        runner = _make_runner(tmp_path)
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
        runner = _make_runner(tmp_path)
        response_path = tmp_path / "nonexistent.json"

        with pytest.raises(CrashError) as exc_info:
            runner._parse_response(response_path, 139, "test_model")
        assert exc_info.value.exit_code == 139


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
        runner = _make_runner(tmp_path)
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

        runner = _make_runner(tmp_path)
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

        runner = _make_runner(tmp_path)
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

        runner = _make_runner(tmp_path)
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

    def test_diagnostics_npde_mapped_from_predictive_bundle(self, tmp_path: Path) -> None:
        """npde is populated onto DiagnosticBundle atomically alongside
        vpc/pit_calibration/npe_score whenever predicted_simulations is
        present (mirrors the existing vpc/npe_score assertions above)."""
        from apmode.governance.policy import Gate3Config

        runner = _make_runner(tmp_path)
        response_path = tmp_path / "response.json"

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
        backend_result = runner._parse_response(
            response_path,
            0,
            "test_model_id_0000000",
            gate3_policy=policy,
        )
        assert backend_result.diagnostics.npde is not None
        assert backend_result.diagnostics.npde.n_subjects == 12

    def test_vpc_include_prediction_corrected_survives_full_path(self, tmp_path: Path) -> None:
        """Gate3Config(vpc_include_prediction_corrected=True) reaches
        DiagnosticBundle.vpc.prediction_corrected through the full
        Nlmixr2Runner._parse_response -> build_predictive_diagnostics path
        — pins that the policy flag isn't dropped between RSubprocessRequest
        plumbing and the diagnostic bundle."""
        from apmode.governance.policy import Gate3Config

        runner = _make_runner(tmp_path)
        response_path = tmp_path / "response.json"

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
            vpc_include_prediction_corrected=True,
        )
        backend_result = runner._parse_response(
            response_path,
            0,
            "test_model_id_0000000",
            gate3_policy=policy,
        )
        assert backend_result.diagnostics.vpc is not None
        assert backend_result.diagnostics.vpc.prediction_corrected is True

    def test_vpc_include_prediction_corrected_default_false_survives_full_path(
        self, tmp_path: Path
    ) -> None:
        """Default (False) policy also survives the full path unchanged —
        companion regression pin to the True-flag assertion above."""
        from apmode.governance.policy import Gate3Config

        runner = _make_runner(tmp_path)
        response_path = tmp_path / "response.json"

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
        backend_result = runner._parse_response(
            response_path,
            0,
            "test_model_id_0000000",
            gate3_policy=policy,
        )
        assert backend_result.diagnostics.vpc is not None
        assert backend_result.diagnostics.vpc.prediction_corrected is False
