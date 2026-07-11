# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared policy-loading helpers for the test suite.

Single home for ``POLICY_DIR`` + ``load_policy(lane)`` (faithful copy of the
former ``tests/unit/test_gates.py::_load_policy``) so gate/ranking tests do not
import them cross-module from a sibling test file.
"""

from __future__ import annotations

import json
from pathlib import Path

from apmode.governance.policy import GatePolicy

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def load_policy(lane: str) -> GatePolicy:
    """Load a policy from the policies directory."""
    path = POLICY_DIR / f"{lane}.json"
    return GatePolicy.model_validate(json.loads(path.read_text()))
