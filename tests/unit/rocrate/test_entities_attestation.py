# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the reviewer attestation sidecar projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import attestation as ent_attestation

from ._fixtures import build_submission_bundle


def _root(graph: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in graph if e.get("@id") == "./")


_ATTESTATION_JSON = (
    '{"attestation_schema_version": "1.0", "reviewer_id": "jdoe", '
    '"reviewer_role": "PK reviewer", "timestamp": "2026-07-10T00:00:00+00:00", '
    '"decision": "approved", "rationale": "ok", "gate_overrides": []}\n'
)


class TestAddAttestation:
    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        """No attestation sidecar -> projector is a no-op."""
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        assert ent_attestation.add_attestation(graph, bundle) is None
        assert "hasPart" not in _root(graph)

    def test_projects_file_and_additional_type(self, tmp_path: Path) -> None:
        """When attestation.json exists, it is projected with apmode:attestation."""
        bundle = build_submission_bundle(tmp_path)
        (bundle / "attestation.json").write_text(_ATTESTATION_JSON)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        fid = ent_attestation.add_attestation(graph, bundle)

        assert fid == "attestation.json"
        entity = next(e for e in graph if e["@id"] == "attestation.json")
        assert entity["@type"] == "File"
        assert entity["additionalType"] == "apmode:attestation"
        assert entity["encodingFormat"] == "application/json"
        assert "sha256" in entity
        assert "contentSize" in entity
        assert {"@id": "attestation.json"} in _root(graph)["hasPart"]

    def test_attestation_does_not_invalidate_sealed_digest(self, tmp_path: Path) -> None:
        """Adding attestation.json after sealing must not break _COMPLETE verification.

        The emitter/importer digest excludes attestation.json explicitly
        so that ``apmode attest`` can drop the sidecar into a sealed
        bundle without re-sealing. Mirror that guarantee in a unit test
        against the importer's verifier.
        """
        from apmode.bundle.rocrate.importer import _verify_sentinel

        bundle = build_submission_bundle(tmp_path)
        # Sealed fixture already has a valid _COMPLETE; drop an
        # attestation after-the-fact and verify the digest still matches.
        (bundle / "attestation.json").write_text(_ATTESTATION_JSON)
        # Must not raise.
        _verify_sentinel(bundle)
