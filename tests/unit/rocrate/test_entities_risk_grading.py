# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for risk_grading/<id>.json RO-Crate projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities import risk_grading as ent_risk_grading


def _seed_backend_action(graph: list[dict[str, Any]], candidate_id: str) -> None:
    graph.append({"@id": f"#backend-create-{candidate_id}", "@type": "CreateAction", "result": []})


def test_add_risk_grading_reports_projects_file(tmp_path: Path) -> None:
    rg_dir = tmp_path / "risk_grading"
    rg_dir.mkdir()
    (rg_dir / "cand-1.json").write_text(json.dumps({"context_of_use": "cou", "risk_tier": "high"}))
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]
    _seed_backend_action(graph, "cand-1")

    added = ent_risk_grading.add_risk_grading_reports(graph, tmp_path)

    assert len(added) == 1
    entity = next(e for e in graph if e["@id"] == added[0])
    assert entity["additionalType"] == vocab.RISK_GRADING_REPORT

    action = next(e for e in graph if e["@id"] == "#backend-create-cand-1")
    assert {"@id": added[0]} in action["result"]

    root = next(e for e in graph if e["@id"] == "./")
    assert {"@id": added[0]} in root["hasPart"]


def test_add_risk_grading_reports_skips_orphan(tmp_path: Path) -> None:
    rg_dir = tmp_path / "risk_grading"
    rg_dir.mkdir()
    (rg_dir / "cand-orphan.json").write_text(json.dumps({"risk_tier": "low"}))
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]  # no backend-create action

    added = ent_risk_grading.add_risk_grading_reports(graph, tmp_path)
    assert added == []


def test_add_risk_grading_reports_missing_dir_returns_empty(tmp_path: Path) -> None:
    graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]
    added = ent_risk_grading.add_risk_grading_reports(graph, tmp_path)
    assert added == []
