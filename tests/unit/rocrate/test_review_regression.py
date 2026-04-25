# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression tests that lock in the multi-model-review fixes.

Each test is labelled with the original finding tag (B1 / H-series /
M-series / L1) so future reviewers can trace a failure back to the
decision that motivated it. Do **not** collapse these with the existing
suite — they exist to prevent silent reintroduction of the original
bug, which passing unit tests alone did not catch before.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from apmode.bundle.rocrate import (
    BundleNotSealedError,
    RoCrateEmitter,
    RoCrateExportOptions,
)

from ._fixtures import build_submission_bundle


class TestH3_AIModelInvocation:
    def test_metadata_uses_ai_model_invocation(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(
            tmp_path,
            candidate_ids=("c001",),
            add_agentic=True,
        )
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(
                include_provagent=True,
                date_published="2026-04-17T10:00:00Z",
            ),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        meta_entities = [
            e
            for e in metadata["@graph"]
            if isinstance(e.get("@id"), str) and "_meta.json" in e["@id"] and "additionalType" in e
        ]
        assert meta_entities, "no agentic meta entity was projected with additionalType"
        for entity in meta_entities:
            assert entity["additionalType"] == "provagent:AIModelInvocation", (
                f"expected AIModelInvocation per PROV-AGENT eScience 2025; got "
                f"{entity['additionalType']!r}"
            )


class TestH4_DatePublishedStability:
    def test_reads_sealed_at_from_complete(self, tmp_path: Path) -> None:
        """When ``options.date_published`` is unset the projector uses ``_COMPLETE.sealed_at``."""
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        # Deliberately skew mtime to prove we are NOT reading it.
        sentinel = bundle / "_COMPLETE"
        os.utime(sentinel, (0, 0))
        assert sentinel.stat().st_mtime == 0

        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(),  # no explicit date_published
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        root = next(e for e in metadata["@graph"] if e["@id"] == "./")
        # sealed_at from the fixture is 2026-04-17T10:00:00Z
        assert root["datePublished"] == "2026-04-17T10:00:00Z", (
            "datePublished should track _COMPLETE.sealed_at, not filesystem mtime"
        )


class TestH5_CaseInsensitiveExclusion:
    def test_uppercase_sidecar_round_trips(self, tmp_path: Path) -> None:
        from apmode.bundle.rocrate.importer import import_crate

        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        # Simulate a case-insensitive filesystem round-trip by planting
        # an upper-case variant of the SBOM filename inside the crate.
        (out / "BOM.CDX.JSON").write_text("{}")
        target = tmp_path / "imported"
        # If the excluded set were case-sensitive, this file would be
        # hashed into the digest and the sentinel check would fail.
        import_crate(out, target)
        assert (target / "BOM.CDX.JSON").is_file()


class TestM1_CredibilityOrphanGuard:
    def test_skips_credibility_when_no_result(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(
            tmp_path,
            candidate_ids=("c001",),
            add_credibility=True,
        )
        # Remove the backend result so c001 has *no* CreateAction in the
        # graph; the credibility File must not be emitted as an orphan.
        (bundle / "results" / "c001_result.json").unlink()
        # Re-seal so digest matches.
        import hashlib

        digest = hashlib.sha256()
        for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
            if not p.is_file() or p.name in ("_COMPLETE", "bom.cdx.json", "sbc_manifest.json"):
                continue
            digest.update(p.relative_to(bundle).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(p.read_bytes())
        (bundle / "_COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "orphan-credibility-test",
                    "sha256": digest.hexdigest(),
                    "sealed_at": "2026-04-17T10:00:00Z",
                },
                indent=2,
            )
            + "\n"
        )

        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        credibility_refs = [
            e
            for e in metadata["@graph"]
            if isinstance(e.get("@id"), str) and e["@id"].startswith("credibility/")
        ]
        assert credibility_refs == [], (
            f"credibility orphan emitted despite no backend result: {credibility_refs!r}"
        )


class TestM3_ContentSizeString:
    def test_contentSize_is_string(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        sized = [e for e in metadata["@graph"] if "contentSize" in e]
        assert sized, "no File entity emitted with contentSize"
        for entity in sized:
            assert isinstance(entity["contentSize"], str), (
                f"contentSize must be a string per schema.org Text range: {entity['@id']}"
            )


class TestM6_SentinelParseFailure:
    def test_refuses_unparseable_sentinel(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        (bundle / "_COMPLETE").write_text("not valid json ::")
        with pytest.raises(BundleNotSealedError, match="unparseable"):
            RoCrateEmitter().export_from_sealed_bundle(
                bundle,
                tmp_path / "crate",
                RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
            )

    def test_refuses_non_object_sentinel(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        (bundle / "_COMPLETE").write_text('["wrong shape"]')
        with pytest.raises(BundleNotSealedError, match="JSON object"):
            RoCrateEmitter().export_from_sealed_bundle(
                bundle,
                tmp_path / "crate",
                RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
            )


class TestM7_PublishExistenceCheck:
    def test_publish_stub_validates_bundle_exists(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from apmode.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bundle",
                "publish",
                str(tmp_path / "does-not-exist"),
                "--workflowhub",
            ],
        )
        assert result.exit_code == 1, result.stdout
        assert "bundle_dir not found" in result.stdout or "bundle_dir not found" in (
            result.stderr or ""
        )


class TestM8_AgenticSequenceOrdering:
    def test_sequence_number_beats_lexicographic(self, tmp_path: Path) -> None:
        """iter10 must follow iter2 when sequence_number says so."""
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        trace = bundle / "agentic_trace"
        trace.mkdir()
        for iter_id, seq in (("iter10", 2), ("iter2", 1)):
            for kind, extra in (
                ("input", {}),
                ("output", {"parsed_transforms": []}),
                (
                    "meta",
                    {
                        "model_id": "stub-model",
                        "sequence_number": seq,
                    },
                ),
            ):
                payload: dict[str, Any] = {"iteration_id": iter_id}
                payload.update(extra)
                (trace / f"{iter_id}_{kind}.json").write_text(json.dumps(payload) + "\n")
        # Re-seal bundle to include the agentic_trace files in digest.
        import hashlib

        digest = hashlib.sha256()
        for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
            if not p.is_file() or p.name in ("_COMPLETE", "bom.cdx.json", "sbc_manifest.json"):
                continue
            digest.update(p.relative_to(bundle).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(p.read_bytes())
        (bundle / "_COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "agentic-seq",
                    "sha256": digest.hexdigest(),
                    "sealed_at": "2026-04-17T10:00:00Z",
                },
                indent=2,
            )
            + "\n"
        )

        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        iter10 = next(e for e in metadata["@graph"] if e["@id"] == "#agentic-iter10")
        # iter10 should reference iter2 via wasInformedBy (because
        # sequence_number puts iter2 first), not the other way round.
        informed_by = iter10.get("prov:wasInformedBy")
        assert informed_by == {"@id": "#agentic-iter2"} or informed_by == [
            {"@id": "#agentic-iter2"}
        ], f"expected iter10 to follow iter2; got {informed_by!r}"


class TestL1_WorkflowRoCrateBaseProfile:
    def test_conformsTo_includes_base_workflow_profile(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        root = next(e for e in metadata["@graph"] if e["@id"] == "./")
        ids = {ref["@id"] for ref in root["conformsTo"] if isinstance(ref, dict)}
        assert "https://w3id.org/workflowhub/workflow-ro-crate/1.0" in ids

    def test_profile_creativework_versions_are_correct(self, tmp_path: Path) -> None:
        """Each conformsTo profile entity carries its own version string.

        Earlier versions used a heuristic ``"0.5" if "/0.5" in uri else "1.1"``
        which would have stamped the new ``workflow-ro-crate/1.0``
        entity with version ``1.1`` (wrong). Lock the explicit triples
        in so a regression is caught.
        """
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        expected = {
            "https://w3id.org/ro/wfrun/provenance/0.5": "0.5",
            "https://w3id.org/ro/wfrun/workflow/0.5": "0.5",
            "https://w3id.org/ro/wfrun/process/0.5": "0.5",
            "https://w3id.org/workflowhub/workflow-ro-crate/1.0": "1.0",
            "https://w3id.org/ro/crate/1.1": "1.1",
        }
        for uri, version in expected.items():
            entity = next((e for e in metadata["@graph"] if e["@id"] == uri), None)
            assert entity is not None, f"missing profile CreativeWork: {uri}"
            assert entity["version"] == version, (
                f"profile {uri} version {entity['version']!r} != {version!r}"
            )


class TestRereviewPolish:
    """Polish fixes from the second-pass review (gemini/droid/glm-5)."""

    def test_h4_invalid_sealed_at_falls_back(self, tmp_path: Path) -> None:
        """A garbage ``sealed_at`` must not silently corrupt datePublished."""
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        sentinel = bundle / "_COMPLETE"
        payload = json.loads(sentinel.read_text())
        payload["sealed_at"] = "not-an-iso-date"
        sentinel.write_text(json.dumps(payload, indent=2) + "\n")

        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        root = next(e for e in metadata["@graph"] if e["@id"] == "./")
        # Falls through to mtime-derived ISO timestamp; guarantee it
        # at least *parses* as ISO-8601 rather than echoing the garbage.
        from datetime import datetime as _datetime

        _datetime.fromisoformat(root["datePublished"])

    def test_m8_mixed_sequence_falls_back_to_lex(self, tmp_path: Path) -> None:
        """When some iterations have sequence_number and some don't, fall back to lex.

        Avoids the prior 'tier 0 wins' behaviour where a single
        sequence-numbered iteration was stamped first regardless of
        true trace order.
        """
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        trace = bundle / "agentic_trace"
        trace.mkdir()
        for iter_id, meta_extra in (
            ("iterA", {"started_at": "2026-04-17T10:00:00Z"}),
            # iterB has sequence_number but iterA does not — mixed signal.
            ("iterB", {"sequence_number": 99, "started_at": "2026-04-17T10:05:00Z"}),
        ):
            for kind, extra in (
                ("input", {}),
                ("output", {"parsed_transforms": []}),
                ("meta", {"model_id": "stub"} | meta_extra),
            ):
                p: dict[str, Any] = {"iteration_id": iter_id}
                p.update(extra)
                (trace / f"{iter_id}_{kind}.json").write_text(json.dumps(p) + "\n")
        # Re-seal.
        import hashlib

        digest = hashlib.sha256()
        for entry in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
            if not entry.is_file() or entry.name in (
                "_COMPLETE",
                "bom.cdx.json",
                "sbc_manifest.json",
            ):
                continue
            digest.update(entry.relative_to(bundle).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(entry.read_bytes())
        (bundle / "_COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "mixed-agentic",
                    "sha256": digest.hexdigest(),
                    "sealed_at": "2026-04-17T10:00:00Z",
                },
                indent=2,
            )
            + "\n"
        )

        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        # Both signals are mixed → expect timestamp fallback (both
        # iterations carry started_at). iterA timestamp is earlier so
        # iterB wasInformedBy iterA.
        iter_b = next(e for e in metadata["@graph"] if e["@id"] == "#agentic-iterB")
        informed_by = iter_b.get("prov:wasInformedBy")
        assert informed_by == {"@id": "#agentic-iterA"} or informed_by == [
            {"@id": "#agentic-iterA"}
        ], f"expected iterB → iterA via timestamp; got {informed_by!r}"

    def test_b1_rejects_malicious_main_entity_path(self, tmp_path: Path) -> None:
        """A crafted mainEntity with `..` is treated as 'no synthetic file'.

        Returning ``None`` from ``_read_synthetic_workflow_path`` means
        every workflow/* entry is preserved on import — the safer
        outcome than skipping a sensitive path. The digest check still
        catches any mismatch.
        """
        from apmode.bundle.rocrate.importer import _read_synthetic_workflow_path

        crate = tmp_path / "evil"
        crate.mkdir()
        (crate / "ro-crate-metadata.json").write_text(
            json.dumps(
                {
                    "@context": "https://w3id.org/ro/crate/1.1/context",
                    "@graph": [
                        {
                            "@id": "./",
                            "@type": "Dataset",
                            "mainEntity": {"@id": "../../etc/passwd"},
                        }
                    ],
                }
            )
        )
        assert _read_synthetic_workflow_path(crate) is None

    def test_m6_binary_corrupted_sentinel_rejected(self, tmp_path: Path) -> None:
        """A non-UTF-8 sentinel raises BundleNotSealedError, not UnicodeDecodeError."""
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        (bundle / "_COMPLETE").write_bytes(b"\xff\xfe\xfd not utf-8")
        with pytest.raises(BundleNotSealedError, match="unparseable"):
            RoCrateEmitter().export_from_sealed_bundle(
                bundle,
                tmp_path / "crate",
                RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
            )


class TestH1_H2_BackendEntities:
    def test_backend_howtostep_and_engine_emitted(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        out = tmp_path / "crate"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        metadata = json.loads((out / "ro-crate-metadata.json").read_text())
        step = next(
            (e for e in metadata["@graph"] if e.get("@id") == "#step-backend-nlmixr2"),
            None,
        )
        engine = next(
            (e for e in metadata["@graph"] if e.get("@id") == "#engine-nlmixr2"),
            None,
        )
        assert step is not None and engine is not None
        # workExample on the backend step must point at the engine.
        assert step["workExample"] == {"@id": "#engine-nlmixr2"}
        # Engine must carry a softwareVersion from backend_versions.json.
        assert engine.get("softwareVersion") == "3.0.0"

        # Workflow.hasPart must include the engine so the
        # ProvRCToolRequired invariant holds.
        workflow_id = "workflows/submission-lane.apmode"
        workflow = next(e for e in metadata["@graph"] if e["@id"] == workflow_id)
        has_part_ids = {ref["@id"] for ref in workflow["hasPart"] if isinstance(ref, dict)}
        assert "#engine-nlmixr2" in has_part_ids
