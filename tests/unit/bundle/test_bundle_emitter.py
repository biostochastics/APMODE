# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for bundle emitter scaffolding (ARCHITECTURE.md §5)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apmode.bundle.emitter import BundleEmitter, BundleNotSealedError, _compute_bundle_digest
from apmode.bundle.models import (
    BackendResult,
    BackendVersions,
    BLQHandling,
    CandidateLineage,
    CandidateLineageEntry,
    ColumnMapping,
    ConvergenceMetadata,
    CurationStep,
    DataManifest,
    DataProvenance,
    DiagnosticBundle,
    EvidenceManifest,
    FailedCandidate,
    GateCheckResult,
    GateOverride,
    GateResult,
    GOFMetrics,
    IdentifiabilityFlags,
    InitialEstimateEntry,
    InitialEstimates,
    ParameterEstimate,
    RankedCandidateEntry,
    Ranking,
    ReviewerAttestation,
    SearchTrajectoryEntry,
    SeedRegistry,
    SplitManifest,
    SubjectAssignment,
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
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
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


def _test_provenance() -> DataProvenance:
    return DataProvenance(
        source_system="NONMEM dataset",
        time_zero_definition="first dose administration, protocol-defined",
        blq_handling_method="M3_likelihood",
        curation_steps=[
            CurationStep(description="Removed duplicate dosing records", applied_by="J. Analyst")
        ],
        source_file_reference="Study XYZ-101 SDTM PC domain, extracted 2026-03-01",
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

    def test_write_compiled_spec_carries_metadata_into_fingerprints(self, tmp_path: Path) -> None:
        """Metadata (P1.2) is written into fingerprints.json alongside digests."""
        from apmode.dsl.ast_models import Metadata

        emitter = BundleEmitter(tmp_path, run_id="test_run_spec_meta")
        emitter.initialize()
        spec = _test_spec().model_copy(
            update={
                "model_id": "test_model_emitter_meta",
                "metadata": Metadata(title="Test model", analyte="drugX"),
            }
        )
        emitter.write_compiled_spec(spec)

        fingerprints_path = (
            tmp_path
            / "test_run_spec_meta"
            / "compiled_specs"
            / "test_model_emitter_meta_fingerprints.json"
        )
        assert fingerprints_path.exists()
        data = json.loads(fingerprints_path.read_text())
        assert data["metadata"] == {
            "title": "Test model",
            "intent": None,
            "context_of_use": None,
            "analyte": "drugX",
            "version": None,
        }

    def test_write_compiled_spec_metadata_none_when_absent(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_run_spec_no_meta")
        emitter.initialize()
        emitter.write_compiled_spec(_test_spec())

        fingerprints_path = (
            tmp_path
            / "test_run_spec_no_meta"
            / "compiled_specs"
            / "test_model_emitter_000_fingerprints.json"
        )
        data = json.loads(fingerprints_path.read_text())
        assert data["metadata"] is None

    def test_write_compiled_spec_pins_grammar_version_in_fingerprints(
        self, tmp_path: Path
    ) -> None:
        """dsl_grammar_version defaults to apmode.dsl.grammar.grammar_version()."""
        from apmode.dsl.grammar import grammar_version

        emitter = BundleEmitter(tmp_path, run_id="test_run_grammar_version")
        emitter.initialize()
        emitter.write_compiled_spec(_test_spec())

        fingerprints_path = (
            tmp_path
            / "test_run_grammar_version"
            / "compiled_specs"
            / "test_model_emitter_000_fingerprints.json"
        )
        data = json.loads(fingerprints_path.read_text())
        assert data["dsl_grammar_version"] == grammar_version()
        assert len(data["dsl_grammar_version"]) == 64

    def test_write_compiled_spec_grammar_version_override(self, tmp_path: Path) -> None:
        """Explicit dsl_grammar_version overrides the default (caller-supplied pin)."""
        emitter = BundleEmitter(tmp_path, run_id="test_run_grammar_override")
        emitter.initialize()
        emitter.write_compiled_spec(_test_spec(), dsl_grammar_version="deadbeef" * 8)

        fingerprints_path = (
            tmp_path
            / "test_run_grammar_override"
            / "compiled_specs"
            / "test_model_emitter_000_fingerprints.json"
        )
        data = json.loads(fingerprints_path.read_text())
        assert data["dsl_grammar_version"] == "deadbeef" * 8

    def test_write_compiled_spec_grammar_version_is_additive_not_content(
        self, tmp_path: Path
    ) -> None:
        """Different dsl_grammar_version pins must not perturb spec-content
        fingerprints (structure_fingerprint / spec_fingerprint) — proves
        grammar identity is compiler provenance, not model content, matching
        the ``macros_used`` exclusion-by-omission precedent in canonical.py.
        """
        emitter_a = BundleEmitter(tmp_path, run_id="test_run_grammar_a")
        emitter_a.initialize()
        emitter_a.write_compiled_spec(_test_spec(), dsl_grammar_version="a" * 64)

        emitter_b = BundleEmitter(tmp_path, run_id="test_run_grammar_b")
        emitter_b.initialize()
        emitter_b.write_compiled_spec(_test_spec(), dsl_grammar_version="b" * 64)

        data_a = json.loads(
            (
                tmp_path
                / "test_run_grammar_a"
                / "compiled_specs"
                / "test_model_emitter_000_fingerprints.json"
            ).read_text()
        )
        data_b = json.loads(
            (
                tmp_path
                / "test_run_grammar_b"
                / "compiled_specs"
                / "test_model_emitter_000_fingerprints.json"
            ).read_text()
        )

        assert data_a["dsl_grammar_version"] != data_b["dsl_grammar_version"]
        assert data_a["structure_fingerprint"] == data_b["structure_fingerprint"]
        assert data_a["spec_fingerprint"] == data_b["spec_fingerprint"]
        assert data_a["initial_fingerprint"] == data_b["initial_fingerprint"]
        assert data_a["justification_hash"] == data_b["justification_hash"]

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
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        spec2 = DSLSpec(
            model_id="candidate_002_000000",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.15),
        )

        emitter.write_compiled_spec(spec1)
        emitter.write_compiled_spec(spec2)

        specs_dir = emitter.run_dir / "compiled_specs"
        json_files = list(specs_dir.glob("*.json"))
        r_files = list(specs_dir.glob("*.R"))
        fingerprints_files = list(specs_dir.glob("*_fingerprints.json"))
        # 2 candidates x (spec.json + fingerprints.json)
        assert len(json_files) == 4
        assert len(r_files) == 2
        assert len(fingerprints_files) == 2

    def test_write_compiled_spec_no_macros_omits_expanded_formular(self, tmp_path: Path) -> None:
        """No `expanded.formular` artifact when `macros_used` is empty (the common case)."""
        emitter = BundleEmitter(tmp_path, run_id="test_run_no_macro")
        emitter.initialize()
        spec = _test_spec()
        assert spec.macros_used == []
        emitter.write_compiled_spec(spec)
        expanded_path = emitter.run_dir / "compiled_specs" / spec.model_id / "expanded.formular"
        assert not expanded_path.exists()

    def test_write_compiled_spec_with_macros_writes_expanded_formular(
        self, tmp_path: Path
    ) -> None:
        """`expanded.formular` is written iff `spec.macros_used` is non-empty (P2.1)."""
        from apmode.dsl.grammar import compile_dsl
        from apmode.dsl.serializer import serialize_spec

        emitter = BundleEmitter(tmp_path, run_id="test_run_macro")
        emitter.initialize()
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
                use pkstd.standard_iiv
            }
            """
        )
        assert spec.macros_used == ["pkstd.standard_iiv@v1"]
        emitter.write_compiled_spec(spec)

        expanded_path = emitter.run_dir / "compiled_specs" / spec.model_id / "expanded.formular"
        assert expanded_path.exists()
        assert expanded_path.read_text() == serialize_spec(spec)
        assert "variability: {" in expanded_path.read_text()

    def test_write_compiled_spec_node_skips_r(self, tmp_path: Path) -> None:
        """NODE specs emit JSON only, no R code."""
        emitter = BundleEmitter(tmp_path, run_id="test_run_node")
        emitter.initialize()
        node_spec = DSLSpec(
            model_id="node_test_model_0000",
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
            distribution=OneCmt(),
            elimination=LinearElim(),
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
        # spec.json + fingerprints.json for the single candidate
        assert len(list((run_dir / "compiled_specs").glob("*.json"))) == 2
        assert len(list((run_dir / "compiled_specs").glob("*.R"))) == 1
        assert (run_dir / "compiled_specs" / "test_model_emitter_000_fingerprints.json").exists()

    def test_write_seed_result(self, tmp_path: Path) -> None:
        """Seed stability results are persisted as {cid}_seed_{n}_result.json."""
        emitter = BundleEmitter(tmp_path, run_id="test_seed_persist")
        emitter.initialize()
        result = BackendResult(
            model_id="seed_run_model",
            backend="nlmixr2",
            converged=True,
            ofv=152.0,
            aic=162.0,
            bic=172.0,
            parameter_estimates={
                "CL": ParameterEstimate(name="CL", estimate=5.1, category="structural"),
            },
            eta_shrinkage={"CL": 0.05},
            convergence_metadata=ConvergenceMetadata(
                method="saem",
                converged=True,
                iterations=200,
                minimization_status="successful",
                wall_time_seconds=40.0,
            ),
            diagnostics=DiagnosticBundle(
                gof=GOFMetrics(cwres_mean=0.01, cwres_sd=1.0, outlier_fraction=0.01),
                identifiability=IdentifiabilityFlags(
                    profile_likelihood_ci={"CL": True},
                    ill_conditioned=False,
                ),
                blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            ),
            wall_time_seconds=40.0,
            backend_versions={"nlmixr2": "2.1.2"},
            initial_estimate_source="nca",
        )
        path = emitter.write_seed_result(result, "cand_001", 1)
        assert path.exists()
        assert path.name == "cand_001_seed_1_result.json"
        data = json.loads(path.read_text())
        assert data["ofv"] == 152.0

    def test_write_ranking(self, tmp_path: Path) -> None:
        """ranking.json with full ordered candidate list."""
        emitter = BundleEmitter(tmp_path, run_id="test_ranking")
        emitter.initialize()
        ranking = Ranking(
            ranked_candidates=[
                RankedCandidateEntry(
                    candidate_id="c1", rank=1, bic=160.0, n_params=3, backend="nlmixr2"
                ),
                RankedCandidateEntry(
                    candidate_id="c2", rank=2, bic=170.0, n_params=4, backend="nlmixr2"
                ),
            ],
            best_candidate_id="c1",
            ranking_metric="bic",
            n_survivors=2,
        )
        path = emitter.write_ranking(ranking)
        assert path.exists()
        assert path.name == "ranking.json"
        data = json.loads(path.read_text())
        assert data["best_candidate_id"] == "c1"
        assert len(data["ranked_candidates"]) == 2
        assert data["ranked_candidates"][0]["rank"] == 1


class TestDataProvenanceSidecar:
    """DataProvenance is an optional input-lineage sidecar (data_provenance.json)."""

    def test_write_data_provenance(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test_provenance")
        emitter.initialize()
        path = emitter.write_data_provenance(_test_provenance())
        assert path.exists()
        assert path.name == "data_provenance.json"
        data = json.loads(path.read_text())
        assert data["source_system"] == "NONMEM dataset"
        assert data["blq_handling_method"] == "M3_likelihood"
        assert len(data["curation_steps"]) == 1

    def test_omitted_sidecar_does_not_change_digest(self, tmp_path: Path) -> None:
        """A bundle that never calls write_data_provenance is bit-identical
        (digest-wise) to a bundle produced before this feature existed —
        no new required-artifact check, no exclusion-set entry needed."""
        emitter_a = BundleEmitter(tmp_path, run_id="run_without_provenance")
        emitter_a.initialize()
        emitter_a.write_data_manifest(_test_manifest())
        emitter_a.write_seed_registry(_test_seed_registry())

        emitter_b = BundleEmitter(tmp_path, run_id="run_without_provenance_dup")
        emitter_b.initialize()
        emitter_b.write_data_manifest(_test_manifest())
        emitter_b.write_seed_registry(_test_seed_registry())

        assert _compute_bundle_digest(emitter_a.run_dir) == _compute_bundle_digest(
            emitter_b.run_dir
        )

    def test_present_sidecar_changes_digest(self, tmp_path: Path) -> None:
        """When present, data_provenance.json participates in the sealed
        digest like any other bundle file — no exemption."""
        emitter_without = BundleEmitter(tmp_path, run_id="run_no_prov")
        emitter_without.initialize()
        emitter_without.write_data_manifest(_test_manifest())
        emitter_without.write_seed_registry(_test_seed_registry())
        digest_without = _compute_bundle_digest(emitter_without.run_dir)

        emitter_with = BundleEmitter(tmp_path, run_id="run_with_prov")
        emitter_with.initialize()
        emitter_with.write_data_manifest(_test_manifest())
        emitter_with.write_seed_registry(_test_seed_registry())
        emitter_with.write_data_provenance(_test_provenance())
        digest_with = _compute_bundle_digest(emitter_with.run_dir)

        assert digest_with != digest_without

    def test_sidecar_participates_in_seal_digest(self, tmp_path: Path) -> None:
        """Sealing a bundle with the sidecar present includes it in the
        sentinel's recorded digest (round-trip via seal + re-compute)."""
        emitter = BundleEmitter(tmp_path, run_id="run_seal_prov")
        emitter.initialize()
        emitter.write_data_manifest(_test_manifest())
        emitter.write_seed_registry(_test_seed_registry())
        emitter.write_data_provenance(_test_provenance())
        run_dir = emitter.seal()

        sentinel = json.loads((run_dir / "_COMPLETE").read_text())
        assert sentinel["sha256"] == _compute_bundle_digest(run_dir)


def _test_attestation() -> ReviewerAttestation:
    return ReviewerAttestation(
        reviewer_id="jdoe",
        reviewer_role="PK reviewer",
        timestamp="2026-07-10T00:00:00+00:00",
        decision="approved",
        rationale="Reviewed gate decisions and diagnostics; no concerns.",
        gate_overrides=[
            GateOverride(
                gate_id="gate2",
                check_id="shrinkage_max",
                original_passed=False,
                override_justification="Sparse design; expected shrinkage.",
                authorized_by="senior_pharmacometrician",
            )
        ],
    )


def _seal_minimal_bundle(tmp_path: Path, run_id: str) -> BundleEmitter:
    emitter = BundleEmitter(tmp_path, run_id=run_id)
    emitter.initialize()
    emitter.write_data_manifest(_test_manifest())
    emitter.write_seed_registry(_test_seed_registry())
    emitter.seal()
    return emitter


class TestAttestationSidecar:
    """attestation.json — human-in-the-loop reviewer sign-off (QA/QC remediation)."""

    def test_refuses_on_unsealed_bundle(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="run_attest_unsealed")
        emitter.initialize()
        emitter.write_data_manifest(_test_manifest())
        with pytest.raises(BundleNotSealedError, match="not sealed"):
            emitter.write_attestation(_test_attestation())

    def test_write_attestation_on_sealed_bundle(self, tmp_path: Path) -> None:
        emitter = _seal_minimal_bundle(tmp_path, "run_attest_sealed")
        path = emitter.write_attestation(_test_attestation())
        assert path.exists()
        assert path.name == "attestation.json"
        data = json.loads(path.read_text())
        assert data["reviewer_id"] == "jdoe"
        assert data["decision"] == "approved"
        assert len(data["gate_overrides"]) == 1

    def test_attestation_does_not_change_digest(self, tmp_path: Path) -> None:
        emitter = _seal_minimal_bundle(tmp_path, "run_attest_digest")
        digest_before = _compute_bundle_digest(emitter.run_dir)
        emitter.write_attestation(_test_attestation())
        digest_after = _compute_bundle_digest(emitter.run_dir)
        assert digest_before == digest_after

    def test_attestation_does_not_invalidate_sentinel(self, tmp_path: Path) -> None:
        emitter = _seal_minimal_bundle(tmp_path, "run_attest_sentinel")
        sentinel_before = json.loads((emitter.run_dir / "_COMPLETE").read_text())
        emitter.write_attestation(_test_attestation())
        sentinel_after = json.loads((emitter.run_dir / "_COMPLETE").read_text())
        assert sentinel_before == sentinel_after
        assert sentinel_after["sha256"] == _compute_bundle_digest(emitter.run_dir)

    def test_double_write_without_force_raises(self, tmp_path: Path) -> None:
        emitter = _seal_minimal_bundle(tmp_path, "run_attest_double")
        emitter.write_attestation(_test_attestation())
        with pytest.raises(FileExistsError):
            emitter.write_attestation(_test_attestation())

    def test_double_write_with_force_overwrites(self, tmp_path: Path) -> None:
        emitter = _seal_minimal_bundle(tmp_path, "run_attest_force")
        emitter.write_attestation(_test_attestation())
        updated = _test_attestation().model_copy(update={"decision": "rejected"})
        path = emitter.write_attestation(updated, force=True)
        data = json.loads(path.read_text())
        assert data["decision"] == "rejected"


class TestBundleEmitterFull:
    """Tests for all bundle artifact writers (all artifacts per §5)."""

    def test_write_evidence_manifest(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        em = EvidenceManifest(
            data_sha256="a" * 64,
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=2,
            covariate_correlated=False,
            blq_burden=0.05,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        path = emitter.write_evidence_manifest(em)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["richness_category"] == "rich"
        assert data["data_sha256"] == "a" * 64

    def test_write_initial_estimates(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        ie = InitialEstimates(
            entries={
                "cand_001": InitialEstimateEntry(
                    candidate_id="cand_001",
                    source="nca",
                    estimates={"CL": 5.0, "V": 70.0, "ka": 1.5},
                    inputs_used=["per_subject_nca"],
                )
            }
        )
        path = emitter.write_initial_estimates(ie)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cand_001" in data["entries"]

    def test_write_split_manifest(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        split = SplitManifest(
            split_seed=42,
            split_strategy="subject_level",
            assignments=[
                SubjectAssignment(subject_id="1", fold="train"),
                SubjectAssignment(subject_id="2", fold="test"),
            ],
        )
        path = emitter.write_split_manifest(split)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["split_seed"] == 42

    def test_write_gate_decision(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        gr = GateResult(
            gate_id="gate_001",
            gate_name="technical_validity",
            candidate_id="cand_001",
            passed=True,
            checks=[GateCheckResult(check_id="convergence", passed=True, observed=True)],
            summary_reason="All checks passed",
            policy_version="0.1.0",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        path = emitter.write_gate_decision(gr, gate_number=1)
        assert path.exists()
        # Gate filename invariant (CLAUDE.md): Gate 1 = gate1_<id>.json.
        assert "gate1_cand_001.json" in path.name
        data = json.loads(path.read_text())
        assert data["passed"] is True

    def test_append_search_trajectory(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        e1 = SearchTrajectoryEntry(
            candidate_id="a",
            backend="nlmixr2",
            converged=True,
            bic=100.0,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        e2 = SearchTrajectoryEntry(
            candidate_id="b",
            backend="nlmixr2",
            converged=False,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_search_trajectory(e1)
        emitter.append_search_trajectory(e2)
        path = emitter.run_dir / "search_trajectory.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["candidate_id"] == "a"
        assert json.loads(lines[1])["converged"] is False

    def test_append_failed_candidate(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        fc = FailedCandidate(
            candidate_id="bad_001",
            backend="nlmixr2",
            gate_failed="gate1",
            failed_checks=["convergence", "cwres_mean"],
            summary_reason="Failed: convergence, cwres_mean",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_failed_candidate(fc)
        path = emitter.run_dir / "failed_candidates.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["gate_failed"] == "gate1"
        assert len(data["failed_checks"]) == 2

    def test_write_candidate_lineage(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        lineage = CandidateLineage(
            entries=[
                CandidateLineageEntry(candidate_id="root_1"),
                CandidateLineageEntry(
                    candidate_id="child_1", parent_id="root_1", transform="add_cov_WT"
                ),
            ]
        )
        path = emitter.write_candidate_lineage(lineage)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["entries"]) == 2
        assert data["entries"][1]["parent_id"] == "root_1"
