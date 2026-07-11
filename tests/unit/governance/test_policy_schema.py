# SPDX-License-Identifier: GPL-2.0-or-later
"""Policy JSON schema assertions for lane policy files (PRD §4.3.1)."""

from __future__ import annotations

import json
from pathlib import Path

_POLICIES = Path(__file__).parent.parent.parent.parent / "policies"


def test_submission_has_gate2_5_block() -> None:
    data = json.loads((_POLICIES / "submission.json").read_text())
    assert "gate2_5" in data, "submission.json must include gate2_5 block"
    g25 = data["gate2_5"]
    for key in (
        "context_of_use_required",
        "limitation_to_risk_mapping_required",
        "data_adequacy_required",
        "data_adequacy_ratio_min",
        "sensitivity_analysis_required",
        "ai_ml_transparency_required",
    ):
        assert key in g25, f"gate2_5 missing '{key}'"


def test_submission_has_risk_grading_block() -> None:
    data = json.loads((_POLICIES / "submission.json").read_text())
    rg = data["gate2_5"]["risk_grading"]
    assert rg["enabled"] is True
    assert set(rg["matrix"]["high"]) == {"low", "medium", "high"}
    assert "high" in rg["credibility_factors"]


def test_submission_policy_version_bumped_v071() -> None:
    # 0.7.0 bump: V&V40-style risk-grading matrix added to gate2_5
    # (context-of-use x consequence-of-wrong-decision -> rigor floors).
    # 0.7.1 bump: agentic_compliance block added for trajectory-level
    # reward-hacking / rationale-coherence QA (agentic-LLM backend).
    data = json.loads((_POLICIES / "submission.json").read_text())
    assert data["policy_version"] == "0.7.1"
