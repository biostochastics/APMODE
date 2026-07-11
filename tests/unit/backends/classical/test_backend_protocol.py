# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for BackendRunner protocol and Lane enum."""

from apmode.backends.protocol import BackendRunner, Lane


class TestLaneEnum:
    def test_values(self) -> None:
        assert Lane.SUBMISSION.value == "submission"
        assert Lane.DISCOVERY.value == "discovery"
        assert Lane.OPTIMIZATION.value == "optimization"

    def test_all_lanes(self) -> None:
        assert len(Lane) == 3


class TestBackendRunnerProtocol:
    def test_runtime_isinstance_requires_run_method(self) -> None:
        """The @runtime_checkable decorator makes ``isinstance`` do a real
        structural check: an object exposing ``run`` conforms, one without it
        does not. This exercises the contract instead of asserting that the
        (idempotent) decorator returns a truthy value.
        """

        class _Conforming:
            async def run(self, *args: object, **kwargs: object) -> object:
                return None

        class _MissingRun:
            async def fit(self, *args: object, **kwargs: object) -> object:
                return None

        assert isinstance(_Conforming(), BackendRunner)
        assert not isinstance(_MissingRun(), BackendRunner)

    def test_protocol_run_is_a_coroutine_contract(self) -> None:
        """The protocol's ``run`` member is an async coroutine function —
        callers are contractually allowed to ``await`` it.
        """
        import inspect

        assert inspect.iscoroutinefunction(BackendRunner.run)
