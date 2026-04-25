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

    def test_projects_files_without_context_default(self, tmp_path: Path) -> None:
        """Regulatory files project, but the context slot stays untouched.

        As of the M2 fix the projector no longer silently assumes
        ``pccp-ai-dsf`` whenever any ``regulatory/`` file is present.
        The context slot is left for the orchestrator (which applies
        the ``research-only`` default) or for the operator to set
        explicitly via ``--regulatory-context``.
        """
        bundle = build_submission_bundle(tmp_path, add_regulatory=True)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        added = ent_pccp.add_regulatory_artifacts(graph, bundle)

        assert added is True
        root = next(e for e in graph if e["@id"] == "./")
        assert "apmode:regulatoryContext" not in root
        assert root["apmode:modificationDescription"] == {"@id": "regulatory/md.json"}
        assert root["apmode:traceabilityTable"] == {"@id": "regulatory/traceability.csv"}

    def test_sets_context_when_explicitly_provided(self, tmp_path: Path) -> None:
        """An explicit ``regulatory_context`` override is applied verbatim."""
        bundle = build_submission_bundle(tmp_path, add_regulatory=True)
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        added = ent_pccp.add_regulatory_artifacts(
            graph,
            bundle,
            regulatory_context="ai-act-article-12",
        )

        assert added is True
        root = next(e for e in graph if e["@id"] == "./")
        assert root["apmode:regulatoryContext"] == "ai-act-article-12"
