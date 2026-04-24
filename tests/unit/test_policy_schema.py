# SPDX-License-Identifier: GPL-2.0-or-later
"""Policy JSON schema assertions for lane policy files (PRD §4.3.1)."""

from __future__ import annotations

import json
from pathlib import Path

_POLICIES = Path(__file__).parent.parent.parent / "policies"


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


def test_submission_policy_version_bumped() -> None:
    # 0.6.0 bump: Gate 2 prior-data conflict + prior-sensitivity hard-gates
    # added (plan Tasks 20 + 21). Submission requires both; Discovery /
    # Optimization default the new ``*_required`` knobs to False.
    data = json.loads((_POLICIES / "submission.json").read_text())
    assert data["policy_version"] == "0.6.0", "expected policy_version bump to 0.6.0"
