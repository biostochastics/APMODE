# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for gate-decision → ControlAction projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import gate as ent_gate

from ._fixtures import build_submission_bundle


class TestParseGateFilename:
    def test_gate1(self) -> None:
        assert ent_gate.parse_gate_filename("gate1_cand001.json") == ("1", "cand001")

    def test_gate_half(self) -> None:
        assert ent_gate.parse_gate_filename("gate2.5_cand001.json") == ("2.5", "cand001")

    def test_invalid(self) -> None:
        assert ent_gate.parse_gate_filename("gateZ_x.json") is None


class TestGateControlAction:
    def test_projects_howtostep_and_control_action(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [
            {"@id": "./", "@type": "Dataset"},
            {
                "@id": "workflows/submission-lane.apmode",
                "@type": ["File", "SoftwareSourceCode", "ComputationalWorkflow", "HowTo"],
                "step": [],
            },
            {"@id": "#run-organize-action", "@type": "OrganizeAction", "object": []},
        ]

        gate_path = bundle / "gate_decisions" / "gate1_c001.json"
        action_id = ent_gate.add_gate_control_action(
            graph,
            bundle,
            gate_path,
            workflow_id="workflows/submission-lane.apmode",
            organize_action_id="#run-organize-action",
        )

        assert action_id == "#gate1-control-c001"
        action = next(e for e in graph if e["@id"] == action_id)
        assert action["apmode:gate"] == "gate1"
        assert action["apmode:gateRationale"] == {"@id": "gate_decisions/gate1_c001.json"}
        workflow = next(e for e in graph if e["@id"] == "workflows/submission-lane.apmode")
        assert {"@id": "#step-gate1"} in workflow["step"]


class TestIterGateDecisions:
    def test_sorted_results(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001", "c002"))
        paths = ent_gate.iter_gate_decisions(bundle)
        assert all(p.parent.name == "gate_decisions" for p in paths)
        assert paths == sorted(paths)
