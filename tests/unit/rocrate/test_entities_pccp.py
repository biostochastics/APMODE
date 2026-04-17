# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for PCCP / regulatory projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import pccp as ent_pccp

from ._fixtures import build_submission_bundle


class TestRegulatoryArtifacts:
    def test_no_op_when_absent(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        added = ent_pccp.add_regulatory_artifacts(graph, bundle)

        assert added is False

    def test_sets_pccp_context_when_present(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, add_regulatory=True)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        added = ent_pccp.add_regulatory_artifacts(graph, bundle)

        assert added is True
        root = next(e for e in graph if e["@id"] == "./")
        assert root["apmode:regulatoryContext"] == "pccp-ai-dsf"
        assert root["apmode:modificationDescription"] == {"@id": "regulatory/md.json"}
        assert root["apmode:traceabilityTable"] == {"@id": "regulatory/traceability.csv"}
