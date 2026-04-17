# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the CycloneDX SBOM sidecar projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import sbom as ent_sbom

from ._fixtures import build_submission_bundle


def _root(graph: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in graph if e.get("@id") == "./")


class TestAddSbom:
    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        """No SBOM sidecar -> projector is a no-op."""
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        assert ent_sbom.add_sbom(graph, bundle) is None
        assert "hasPart" not in _root(graph)

    def test_projects_file_and_additional_type(self, tmp_path: Path) -> None:
        """When bom.cdx.json exists, it is projected with apmode:sbom."""
        bundle = build_submission_bundle(tmp_path)
        (bundle / "bom.cdx.json").write_text('{"bomFormat": "CycloneDX"}\n')
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        fid = ent_sbom.add_sbom(graph, bundle)

        assert fid == "bom.cdx.json"
        entity = next(e for e in graph if e["@id"] == "bom.cdx.json")
        assert entity["@type"] == "File"
        assert entity["additionalType"] == "apmode:sbom"
        assert entity["encodingFormat"] == "application/vnd.cyclonedx+json"
        assert "sha256" in entity
        assert "contentSize" in entity
        assert {"@id": "bom.cdx.json"} in _root(graph)["hasPart"]

    def test_sbom_does_not_invalidate_sealed_digest(self, tmp_path: Path) -> None:
        """Adding bom.cdx.json after sealing must not break _COMPLETE verification.

        The emitter/importer digest excludes bom.cdx.json explicitly so
        that CI or ``apmode bundle sbom`` can drop the SBOM into a
        sealed bundle without re-sealing. Mirror that guarantee in a
        unit test against the importer's verifier.
        """
        import json as _json

        from apmode.bundle.rocrate.importer import _verify_sentinel

        bundle = build_submission_bundle(tmp_path)
        # Sealed fixture already has a valid _COMPLETE; drop an SBOM
        # after-the-fact and verify the digest still matches.
        (bundle / "bom.cdx.json").write_text(
            _json.dumps({"bomFormat": "CycloneDX", "components": []}) + "\n"
        )
        # Must not raise.
        _verify_sentinel(bundle)
