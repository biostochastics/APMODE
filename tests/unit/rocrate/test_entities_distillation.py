# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for distillation/<id>.json RO-Crate projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities import distillation as ent_distillation


def _seed_backend_action(graph: list[dict[str, Any]], candidate_id: str) -> None:
    graph.append({"@id": f"#backend-create-{candidate_id}", "@type": "CreateAction", "result": []})


def test_add_distillation_reports_projects_file(tmp_path: Path) -> None:
    d = tmp_path / "distillation"
    d.mkdir()
    (d / "cand-1.json").write_text(
        json.dumps(
            {
                "candidate_id": "cand-1",
                "node_position": "elimination",
                "surrogate": {"surrogate_type": "michaelis_menten"},
            }
        )
    )
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]
    _seed_backend_action(graph, "cand-1")

    added = ent_distillation.add_distillation_reports(graph, tmp_path)

    assert len(added) == 1
    entity = next(e for e in graph if e["@id"] == added[0])
    assert entity["additionalType"] == vocab.DISTILLATION_REPORT
    assert entity["encodingFormat"] == "application/json"
    # roc-validator REQUIRED: the File must be referenced from BOTH the root
    # hasPart and the candidate's CreateAction.result (no orphan).
    action = next(e for e in graph if e["@id"] == "#backend-create-cand-1")
    assert {"@id": added[0]} in action["result"]
    root = next(e for e in graph if e["@id"] == "./")
    assert {"@id": added[0]} in root["hasPart"]


def test_add_distillation_reports_skips_orphan(tmp_path: Path) -> None:
    d = tmp_path / "distillation"
    d.mkdir()
    (d / "cand-orphan.json").write_text(json.dumps({"candidate_id": "cand-orphan"}))
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]  # no backend action

    added = ent_distillation.add_distillation_reports(graph, tmp_path)
    assert added == []


def test_add_distillation_reports_missing_dir_returns_empty(tmp_path: Path) -> None:
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]
    added = ent_distillation.add_distillation_reports(graph, tmp_path)
    assert added == []
