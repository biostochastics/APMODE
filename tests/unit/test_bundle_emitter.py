# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for bundle emitter scaffolding (ARCHITECTURE.md §5)."""

import json
from pathlib import Path

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    BackendVersions,
    ColumnMapping,
    DataManifest,
    SeedRegistry,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEAbsorption,
    OneCmt,
    Proportional,
)


def _test_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_model_emitter_000",
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
            subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=20,
        n_observations=200,
        n_doses=40,
    )


def _test_seed_registry() -> SeedRegistry:
    return SeedRegistry(
        root_seed=42,
        r_seed=42,
        r_rng_kind="L'Ecuyer-CMRG",
        np_seed=42,
    )


def _test_versions() -> BackendVersions:
    return BackendVersions(
        apmode_version="0.1.0",
        python_version="3.12.0",
        r_version="4.4.1",
        nlmixr2_version="3.0.0",
    )


class TestBundleEmitter:
    def test_initialize_creates_structure(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_init")
        run_dir = emitter.initialize()
        assert run_dir.exists()
        assert (run_dir / "compiled_specs").is_dir()
        assert (run_dir / "gate_decisions").is_dir()
        assert (run_dir / "results").is_dir()

    def test_write_data_manifest(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_manifest")
        emitter.initialize()
        path = emitter.write_data_manifest(_test_manifest())
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_subjects"] == 20
        assert data["ingestion_format"] == "nonmem_csv"

    def test_write_seed_registry(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_seed")
        emitter.initialize()
        path = emitter.write_seed_registry(_test_seed_registry())
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["root_seed"] == 42
        assert data["r_rng_kind"] == "L'Ecuyer-CMRG"

    def test_write_backend_versions(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_versions")
        emitter.initialize()
        path = emitter.write_backend_versions(_test_versions())
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["apmode_version"] == "0.1.0"

    def test_write_compiled_spec_creates_json_and_r(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_spec")
        emitter.initialize()
        json_path, r_path = emitter.write_compiled_spec(_test_spec())

        assert json_path.exists()
        assert r_path is not None
        assert r_path.exists()
        assert json_path.suffix == ".json"
        assert r_path.suffix == ".R"

        # JSON should be valid DSLSpec
        data = json.loads(json_path.read_text())
        roundtripped = DSLSpec.model_validate(data)
        assert roundtripped.model_id == "test_model_emitter_000"

        # R code should be valid nlmixr2
        r_code = r_path.read_text()
        assert "function()" in r_code
        assert "ini({" in r_code
        assert "model({" in r_code

    def test_write_policy_file(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_policy")
        emitter.initialize()
        policy = {"policy_version": "0.1.0", "lane": "submission"}
        path = emitter.write_policy_file(policy)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["lane"] == "submission"

    def test_multiple_specs_in_same_bundle(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_multi")
        emitter.initialize()

        spec1 = DSLSpec(
            model_id="candidate_001_000000",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        spec2 = DSLSpec(
            model_id="candidate_002_000000",
            absorption=FirstOrder(ka=2.0),
            distribution=OneCmt(V=50.0),
            elimination=LinearElim(CL=3.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.15),
        )

        emitter.write_compiled_spec(spec1)
        emitter.write_compiled_spec(spec2)

        specs_dir = emitter.run_dir / "compiled_specs"
        json_files = list(specs_dir.glob("*.json"))
        r_files = list(specs_dir.glob("*.R"))
        assert len(json_files) == 2
        assert len(r_files) == 2

    def test_write_compiled_spec_node_skips_r(self, tmp_path: Path) -> None:
        """NODE specs emit JSON only, no R code."""
        emitter = BundleEmitter(tmp_path, run_id="test_run_node")
        emitter.initialize()
        node_spec = DSLSpec(
            model_id="node_test_model_0000",
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        json_path, r_path = emitter.write_compiled_spec(node_spec)
        assert json_path.exists()
        assert r_path is None

    def test_auto_generated_run_id(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path)
        assert len(emitter.run_id) == 21  # sparkid length

    def test_full_bundle_workflow(self, tmp_path: Path) -> None:
        """End-to-end: initialize, write all artifacts, verify structure."""
        emitter = BundleEmitter(tmp_path, run_id="full_workflow_test")
        run_dir = emitter.initialize()

        emitter.write_data_manifest(_test_manifest())
        emitter.write_seed_registry(_test_seed_registry())
        emitter.write_backend_versions(_test_versions())
        emitter.write_compiled_spec(_test_spec())
        emitter.write_policy_file({"policy_version": "0.1.0", "lane": "submission"})

        # Verify all expected files exist
        assert (run_dir / "data_manifest.json").exists()
        assert (run_dir / "seed_registry.json").exists()
        assert (run_dir / "backend_versions.json").exists()
        assert (run_dir / "policy_file.json").exists()
        assert len(list((run_dir / "compiled_specs").glob("*.json"))) == 1
        assert len(list((run_dir / "compiled_specs").glob("*.R"))) == 1
