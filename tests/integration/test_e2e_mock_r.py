# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end integration test with mock R subprocess.

Validates the full pipeline:
  ingest CSV → DataManifest → DSLSpec → emit_nlmixr2 →
  RSubprocessRequest → mock R script → RSubprocessResponse →
  BackendResult → BundleEmitter writes full bundle

Does NOT require nlmixr2 / R to be installed — uses a mock R
script that reads request.json and writes a canned response.json.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.benchmarks.suite_a import REFERENCE_PARAMS, scenario_a1
from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    BackendResult,
    BackendVersions,
    SeedRegistry,
)
from apmode.data.adapters import to_nlmixr2_format
from apmode.data.ingest import ingest_nonmem_csv
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2

FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "pk_data" / "simple_1cmt.csv"


def _make_mock_r_script(script_path: Path) -> None:
    """Write a Python script that mimics the R harness.

    Reads request.json, emits a canned successful response.json
    with realistic BackendResult fields.
    """
    script_path.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"Mock R harness — reads request.json, writes response.json.\"\"\"
        import json
        import sys

        request_path = sys.argv[1]
        response_path = sys.argv[2]

        with open(request_path) as f:
            req = json.load(f)

        candidate_id = req["candidate_id"]
        initial_estimates = req.get("initial_estimates", {})

        # Build a realistic BackendResult
        param_estimates = {}
        for name, value in initial_estimates.items():
            param_estimates[name] = {
                "name": name,
                "estimate": value * 1.02,  # slight perturbation
                "se": value * 0.1,
                "rse": 10.0,
                "ci95_lower": value * 0.8,
                "ci95_upper": value * 1.2,
                "fixed": False,
                "category": "structural",
            }

        response = {
            "schema_version": "1.0",
            "status": "success",
            "error_type": None,
            "result": {
                "model_id": candidate_id,
                "backend": "nlmixr2",
                "converged": True,
                "ofv": 150.5,
                "aic": 160.5,
                "bic": 170.5,
                "parameter_estimates": param_estimates,
                "eta_shrinkage": {"CL": 0.05, "V": 0.08, "ka": 0.12},
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
                        "cwres_mean": 0.02,
                        "cwres_sd": 1.01,
                        "outlier_fraction": 0.01,
                        "obs_vs_pred_r2": 0.95,
                    },
                    "vpc": None,
                    "identifiability": {
                        "condition_number": 15.3,
                        "profile_likelihood_ci": {
                            "CL": True, "V": True, "ka": True,
                        },
                        "ill_conditioned": False,
                    },
                    "blq": {
                        "method": "none",
                        "lloq": None,
                        "n_blq": 0,
                        "blq_fraction": 0.0,
                    },
                },
                "wall_time_seconds": 45.2,
                "backend_versions": {
                    "nlmixr2": "2.1.2",
                    "R": "4.4.1",
                },
                "initial_estimate_source": "fallback",
            },
            "r_session_info": {
                "r_version": "4.4.1",
                "nlmixr2_version": "2.1.2",
                "platform": "x86_64-pc-linux-gnu",
                "packages": {"rxode2": "2.1.3"},
            },
            "random_seed_state": [42, 1, 2, 3],
        }

        with open(response_path, "w") as f:
            json.dump(response, f, indent=2)

        sys.exit(0)
        """)
    )


@pytest.mark.integration
class TestEndToEndMockR:
    """Full pipeline integration test using a mock R subprocess."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Ingest → DSLSpec → emit → mock R → BackendResult → bundle."""
        # --- Stage 1: Data Ingestion ---
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)

        assert manifest.n_subjects == 2
        assert manifest.n_observations == 12
        assert manifest.n_doses == 2
        assert manifest.ingestion_format == "nonmem_csv"
        assert len(manifest.data_sha256) == 64

        # --- Stage 2: Data Adapter ---
        nlmixr2_df = to_nlmixr2_format(df)
        assert "ID" in nlmixr2_df.columns
        assert "NMID" not in nlmixr2_df.columns

        # Write adapted data for the runner
        data_csv = tmp_path / "pk_data.csv"
        nlmixr2_df.to_csv(data_csv, index=False)

        # --- Stage 3: DSLSpec + R Code Emission ---
        spec = scenario_a1()
        r_code = emit_nlmixr2(spec, initial_estimates=REFERENCE_PARAMS["A1"])

        assert "ini({" in r_code
        assert "model({" in r_code
        assert "lCL <- log(5.0)" in r_code
        assert "lV <- log(70.0)" in r_code
        assert "lka <- log(1.5)" in r_code

        # --- Stage 4: Mock R Subprocess ---
        mock_script = tmp_path / "mock_harness.py"
        _make_mock_r_script(mock_script)

        runner = Nlmixr2Runner(
            work_dir=tmp_path / "work",
            r_executable=sys.executable,  # use Python to run mock
            harness_path=mock_script,
            estimation=["saem"],
        )

        import asyncio

        result: BackendResult = asyncio.run(
            runner.run(
                spec=spec,
                data_manifest=manifest,
                initial_estimates=REFERENCE_PARAMS["A1"],
                seed=42,
                timeout_seconds=30,
                data_path=data_csv,
            )
        )

        # --- Stage 5: Validate BackendResult ---
        assert result.converged is True
        assert result.backend == "nlmixr2"
        assert result.model_id == spec.model_id
        assert result.ofv == pytest.approx(150.5)
        assert result.aic == pytest.approx(160.5)
        assert "CL" in result.parameter_estimates
        assert "V" in result.parameter_estimates
        assert "ka" in result.parameter_estimates
        assert result.convergence_metadata.method == "saem"
        assert result.convergence_metadata.converged is True

        # --- Stage 6: Bundle Emission ---
        bundle_dir = tmp_path / "bundles"
        emitter = BundleEmitter(bundle_dir, run_id="test_e2e_run")
        run_dir = emitter.initialize()

        # Write all bundle artifacts
        emitter.write_data_manifest(manifest)
        emitter.write_seed_registry(
            SeedRegistry(
                root_seed=42,
                r_seed=42,
                r_rng_kind="L'Ecuyer-CMRG",
                np_seed=42,
            )
        )
        emitter.write_backend_versions(
            BackendVersions(
                apmode_version="0.1.0",
                python_version="3.12.0",
                r_version="4.4.1",
                nlmixr2_version="2.1.2",
            )
        )
        json_path, r_path = emitter.write_compiled_spec(
            spec, initial_estimates=REFERENCE_PARAMS["A1"]
        )

        # Write result to results/
        result_path = run_dir / "results" / f"{spec.model_id}.json"
        result_path.write_text(result.model_dump_json(indent=2))

        # --- Stage 7: Verify Bundle Structure ---
        assert (run_dir / "data_manifest.json").exists()
        assert (run_dir / "seed_registry.json").exists()
        assert (run_dir / "backend_versions.json").exists()
        assert json_path.exists()
        assert r_path is not None and r_path.exists()
        assert result_path.exists()
        assert (run_dir / "gate_decisions").is_dir()

        # Verify data_manifest.json is valid
        dm_data = json.loads((run_dir / "data_manifest.json").read_text())
        assert dm_data["n_subjects"] == 2
        assert dm_data["ingestion_format"] == "nonmem_csv"

        # Verify compiled spec JSON roundtrips
        spec_data = json.loads(json_path.read_text())
        assert spec_data["model_id"] == spec.model_id

        # Verify result JSON is valid BackendResult
        result_data = json.loads(result_path.read_text())
        roundtrip = BackendResult.model_validate(result_data)
        assert roundtrip.converged is True

    def test_pipeline_convergence_error(self, tmp_path: Path) -> None:
        """Verify ConvergenceError propagates correctly through the pipeline."""
        # Write a mock script that returns convergence error
        mock_script = tmp_path / "mock_fail.py"
        mock_script.write_text(
            textwrap.dedent("""\
            import json, sys
            response = {
                "schema_version": "1.0",
                "status": "error",
                "error_type": "convergence",
                "result": None,
                "r_session_info": {
                    "r_version": "4.4.1",
                    "nlmixr2_version": "2.1.2",
                    "platform": "test",
                    "packages": {},
                },
                "random_seed_state": None,
            }
            with open(sys.argv[2], "w") as f:
                json.dump(response, f)
            sys.exit(1)
            """)
        )

        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        data_csv = tmp_path / "pk_data.csv"
        to_nlmixr2_format(df).to_csv(data_csv, index=False)

        runner = Nlmixr2Runner(
            work_dir=tmp_path / "work",
            r_executable=sys.executable,
            harness_path=mock_script,
        )

        import asyncio

        from apmode.errors import ConvergenceError

        with pytest.raises(ConvergenceError, match="convergence failure"):
            asyncio.run(
                runner.run(
                    spec=scenario_a1(),
                    data_manifest=manifest,
                    initial_estimates=REFERENCE_PARAMS["A1"],
                    seed=42,
                    data_path=data_csv,
                )
            )

    def test_pipeline_crash_no_response(self, tmp_path: Path) -> None:
        """Verify CrashError when R process exits without writing response."""
        mock_script = tmp_path / "mock_crash.py"
        mock_script.write_text("import sys; sys.exit(139)  # SIGSEGV\n")

        manifest, _ = ingest_nonmem_csv(FIXTURE_CSV)
        data_csv = tmp_path / "pk_data.csv"
        # Write a minimal CSV (runner needs data_path to exist)
        data_csv.write_text("ID,TIME,DV\n1,0,0\n")

        runner = Nlmixr2Runner(
            work_dir=tmp_path / "work",
            r_executable=sys.executable,
            harness_path=mock_script,
        )

        import asyncio

        from apmode.errors import CrashError

        with pytest.raises(CrashError, match=r"no response\.json"):
            asyncio.run(
                runner.run(
                    spec=scenario_a1(),
                    data_manifest=manifest,
                    initial_estimates={},
                    seed=42,
                    data_path=data_csv,
                )
            )
