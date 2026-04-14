# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the BackendError hierarchy."""

import pytest

from apmode.errors import (
    BackendError,
    BackendTimeoutError,
    ConvergenceError,
    CrashError,
    InvalidSpecError,
)


class TestBackendErrorHierarchy:
    """All backend errors inherit from BackendError."""

    def test_backend_error_is_exception(self) -> None:
        assert issubclass(BackendError, Exception)

    @pytest.mark.parametrize(
        "error_cls",
        [ConvergenceError, BackendTimeoutError, CrashError, InvalidSpecError],
    )
    def test_subclass_of_backend_error(self, error_cls: type) -> None:
        assert issubclass(error_cls, BackendError)

    @pytest.mark.parametrize(
        "error_cls",
        [ConvergenceError, BackendTimeoutError, CrashError, InvalidSpecError],
    )
    def test_can_catch_as_backend_error(self, error_cls: type) -> None:
        with pytest.raises(BackendError):
            raise error_cls("test message")


class TestConvergenceError:
    def test_fields(self) -> None:
        err = ConvergenceError(
            "SAEM did not converge",
            method="saem",
            iterations=500,
            gradient_norm=1.2e-3,
        )
        assert err.method == "saem"
        assert err.iterations == 500
        assert err.gradient_norm == pytest.approx(1.2e-3)
        assert "SAEM did not converge" in str(err)

    def test_defaults(self) -> None:
        err = ConvergenceError("fail")
        assert err.method is None
        assert err.iterations is None
        assert err.gradient_norm is None


class TestBackendTimeoutError:
    def test_fields(self) -> None:
        err = BackendTimeoutError("R process timed out", timeout_seconds=600, pid=12345)
        assert err.timeout_seconds == 600
        assert err.pid == 12345

    def test_does_not_shadow_builtin(self) -> None:
        """Our error class is distinct from and does not shadow builtins.TimeoutError."""
        assert BackendTimeoutError is not builtins.TimeoutError
        assert BackendTimeoutError.__name__ != "TimeoutError"


class TestCrashError:
    def test_fields(self) -> None:
        err = CrashError("Segfault", exit_code=139, stderr_tail="segfault at 0x0")
        assert err.exit_code == 139
        assert err.stderr_tail == "segfault at 0x0"


class TestInvalidSpecError:
    def test_fields(self) -> None:
        err = InvalidSpecError(
            "Negative volume",
            spec_id="abc123",
            violations=["V1 < 0", "CL < 0"],
        )
        assert err.spec_id == "abc123"
        assert err.violations == ["V1 < 0", "CL < 0"]


import builtins  # noqa: E402
