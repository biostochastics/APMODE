# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the data-manifest / split-manifest / seed-registry projectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import data as ent_data

from ._fixtures import build_submission_bundle


def _root(graph: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in graph if e.get("@id") == "./")


class TestAddDataManifest:
    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]
        assert ent_data.add_data_manifest(graph, tmp_path) is None

    def test_projects_file_and_has_part(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        fid = ent_data.add_data_manifest(graph, bundle)

        assert fid == "data_manifest.json"
        file_entity = next(e for e in graph if e["@id"] == "data_manifest.json")
        assert file_entity["@type"] == "File"
        assert "sha256" in file_entity
        root = _root(graph)
        assert {"@id": "data_manifest.json"} in root["hasPart"]

    def test_carries_source_sha256_as_identifier(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        ent_data.add_data_manifest(graph, bundle)

        file_entity = next(e for e in graph if e["@id"] == "data_manifest.json")
        assert file_entity["identifier"].startswith("sha256:")


class TestAddEvidenceManifest:
    def test_projects_additional_type(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        fid = ent_data.add_evidence_manifest(graph, bundle)

        assert fid == "evidence_manifest.json"
        entity = next(e for e in graph if e["@id"] == "evidence_manifest.json")
        assert entity["additionalType"] == "apmode:evidenceClass"


class TestAddSplitManifest:
    def test_carries_split_type(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        ent_data.add_split_manifest(graph, bundle)

        entity = next(e for e in graph if e["@id"] == "split_manifest.json")
        assert entity["additionalType"] == "apmode:split"
