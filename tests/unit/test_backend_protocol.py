# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for BackendRunner protocol and Lane enum."""

from typing import runtime_checkable

from apmode.backends.protocol import BackendRunner, Lane


class TestLaneEnum:
    def test_values(self) -> None:
        assert Lane.SUBMISSION.value == "submission"
        assert Lane.DISCOVERY.value == "discovery"
        assert Lane.OPTIMIZATION.value == "optimization"

    def test_all_lanes(self) -> None:
        assert len(Lane) == 3


class TestBackendRunnerProtocol:
    def test_is_runtime_checkable(self) -> None:
        assert runtime_checkable(BackendRunner)

    def test_protocol_has_run_method(self) -> None:
        assert hasattr(BackendRunner, "run")
