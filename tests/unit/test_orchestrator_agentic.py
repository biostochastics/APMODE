# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for orchestrator agentic backend integration."""

from apmode.routing import _LANE_BACKENDS


def test_discovery_lane_includes_agentic() -> None:
    assert "agentic_llm" in _LANE_BACKENDS["discovery"]


def test_optimization_lane_includes_agentic() -> None:
    assert "agentic_llm" in _LANE_BACKENDS["optimization"]


def test_submission_lane_excludes_agentic() -> None:
    assert "agentic_llm" not in _LANE_BACKENDS["submission"]
