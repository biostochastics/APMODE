# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for PROV-derivation projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import lineage as ent_lineage

from ._fixtures import build_submission_bundle


class TestLineageDerivations:
    def test_one_derivation_per_edge(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("root", "child1", "child2"))
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        n = ent_lineage.add_candidate_lineage_derivations(graph, bundle)

        assert n == 2  # root has no parent
        c1 = next(e for e in graph if e["@id"] == "#candidate-child1")
        assert c1["prov:wasDerivedFrom"] == [{"@id": "#candidate-root"}]
        assert c1["apmode:dslTransform"] == ["add_second_cmt"]
