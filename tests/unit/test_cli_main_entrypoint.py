# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for ``apmode.__main__.main`` — the typed-error entrypoint catcher.

Key behaviours pinned here:

* ``--json`` argv pre-scan flips the renderer before Click parses anything
  (so usage errors raised during parsing render as a JSON envelope, not
  a Rich panel).
* ``raise typer.Exit(N)`` from a command body still returns ``N`` (the
  legacy compatibility bridge that lets PR2 ship without migrating any
  existing call sites).
* ``APModeCLIError`` subclasses render as the typed envelope and exit
  with the class's ``code``.
* Unknown command / bad parameter (Click ``UsageError``) renders as
  ``kind="usage"`` and exits ``2``.
* Ctrl-C → ``user_abort`` + exit ``130``.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import click
import pytest
import typer

from apmode.__main__ import _argv_has_json_flag, main
from apmode.cli_errors import (
    APModeCLIError,
    BundleInvalidError,
    BundleNotFoundError,
    ConfigError,
    PolicyValidationError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# argv pre-scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--json"],
        ["validate", "./bundle", "--json"],
        ["--json", "validate", "./bundle"],
        ["log", "--json", "--top", "3", "./bundle"],
    ],
)
def test_argv_pre_scan_detects_json_anywhere(argv: list[str]) -> None:
    assert _argv_has_json_flag(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["validate", "./bundle"],
        ["doctor"],
        # ``--json=true`` is intentionally NOT recognised: every command
        # in this CLI models ``--json`` as a flag, not as a value option.
        ["validate", "--json=true", "./bundle"],
    ],
)
def test_argv_pre_scan_misses_non_flag_forms(argv: list[str]) -> None:
    assert _argv_has_json_flag(argv) is False


# ---------------------------------------------------------------------------
# Helpers — invoke main() with a stubbed Typer app so each test pins a
# specific exception path without needing a real CLI command to fail.
# ---------------------------------------------------------------------------


class _StubApp:
    """Minimal stand-in for ``apmode.cli.app`` used by the entrypoint tests."""

    def __init__(self, side_effect: Exception | None = None) -> None:
        self._side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})
        if self._side_effect is not None:
            raise self._side_effect


def _run_main(
    argv: list[str],
    *,
    side_effect: Exception | None,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    """Drive ``main()`` with a stubbed ``app`` and return (code, stdout, stderr)."""
    stub = _StubApp(side_effect=side_effect)
    with patch("apmode.cli.app", stub), pytest.raises(SystemExit) as excinfo:
        main(argv)
    captured = capsys.readouterr()
    return int(excinfo.value.code or 0), captured.out, captured.err


# ---------------------------------------------------------------------------
# Happy path (no exception)
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean_completion(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = _run_main([], side_effect=None, capsys=capsys)
    assert code == 0
    assert stdout == ""
    assert stderr == ""


# ---------------------------------------------------------------------------
# Legacy ``raise typer.Exit(N)`` bridge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 1, 2, 7, 99])
def test_typer_exit_propagates_through_main(
    exit_code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defensive catch path: if Click ever re-raises ``Exit``, we honour it."""
    code, _, _ = _run_main(
        [],
        side_effect=typer.Exit(code=exit_code),
        capsys=capsys,
    )
    assert code == exit_code


class _ReturnValueApp:
    """Stub that simulates Click 8.x's actual ``standalone_mode=False``
    behaviour: ``raise typer.Exit(N)`` is converted to a return value
    of ``N`` rather than re-raised. The entrypoint reads this return
    value and forwards it to ``sys.exit``."""

    def __init__(self, return_code: int | None) -> None:
        self.return_code = return_code

    def __call__(self, *args: Any, **kwargs: Any) -> int | None:
        return self.return_code


@pytest.mark.parametrize("exit_code", [0, 1, 2, 7, 99])
def test_click_return_value_is_honoured_as_exit_code(
    exit_code: int,
) -> None:
    """Click 8.x converts ``raise typer.Exit(N)`` to a return value; the
    entrypoint must turn that into ``sys.exit(N)``. Without this the new
    catcher would silently demote every legacy ``typer.Exit`` to exit 0."""
    stub = _ReturnValueApp(return_code=exit_code)
    with patch("apmode.cli.app", stub), pytest.raises(SystemExit) as excinfo:
        main([])
    assert int(excinfo.value.code or 0) == exit_code


def test_none_return_value_means_exit_zero() -> None:
    """A command that returns normally (no Exit, no return value) → exit 0."""
    stub = _ReturnValueApp(return_code=None)
    with patch("apmode.cli.app", stub), pytest.raises(SystemExit) as excinfo:
        main([])
    assert int(excinfo.value.code or 0) == 0


# ---------------------------------------------------------------------------
# Typed errors — Rich panel path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (BundleNotFoundError("./missing"), 10),
        (BundleInvalidError("bad seal"), 11),
        (PolicyValidationError("schema"), 12),
    ],
)
def test_typed_error_renders_to_stderr_and_uses_class_code(
    exc: APModeCLIError,
    expected_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, stdout, stderr = _run_main([], side_effect=exc, capsys=capsys)
    assert code == expected_code
    # Default (non-JSON) mode: stderr carries the message, stdout is silent.
    assert stdout == ""
    assert exc.kind in stderr
    assert exc.message in stderr


def test_typed_error_includes_details_block_in_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exc = ConfigError(
        "contradictory flags",
        details={"flags": ["--agentic", "--lane", "submission"]},
    )
    _, _, stderr = _run_main([], side_effect=exc, capsys=capsys)
    assert "details:" in stderr
    assert "agentic" in stderr


# ---------------------------------------------------------------------------
# Typed errors — JSON envelope path
# ---------------------------------------------------------------------------


def test_typed_error_emits_json_envelope_when_json_flag_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exc = BundleNotFoundError(
        "./missing",
        details={"searched": ["./runs", "."]},
    )
    code, stdout, stderr = _run_main(
        ["validate", "./missing", "--json"],
        side_effect=exc,
        capsys=capsys,
    )
    assert code == 10
    # JSON renderer writes to stdout, stderr stays empty.
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "ok": False,
        "error": {
            "kind": "bundle_not_found",
            "code": 10,
            "message": "./missing",
            "details": {"searched": ["./runs", "."]},
        },
    }


def test_typed_error_json_path_works_when_flag_precedes_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pre-scan must succeed regardless of flag position."""
    exc = BundleInvalidError("missing _COMPLETE")
    code, stdout, stderr = _run_main(
        ["--json", "validate", "./bundle"],
        side_effect=exc,
        capsys=capsys,
    )
    assert code == 11
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["error"]["kind"] == "bundle_invalid"


# ---------------------------------------------------------------------------
# Click usage errors → kind="usage", exit 2
# ---------------------------------------------------------------------------


def test_click_usage_error_human_path(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, stderr = _run_main(
        [],
        side_effect=click.UsageError("unknown command 'frobnicate'"),
        capsys=capsys,
    )
    assert code == 2
    # Click's own renderer is reused so the wording matches pre-PR2.
    assert "frobnicate" in stderr or "Error" in stderr


def test_click_usage_error_json_path(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = _run_main(
        ["--json"],
        side_effect=click.UsageError("missing argument 'BUNDLE'"),
        capsys=capsys,
    )
    assert code == 2
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload == {
        "ok": False,
        "error": {
            "kind": "usage",
            "code": 2,
            "message": "missing argument 'BUNDLE'",
            "details": {},
        },
    }


# ---------------------------------------------------------------------------
# KeyboardInterrupt → user_abort, exit 130
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_human_path(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, stderr = _run_main([], side_effect=KeyboardInterrupt(), capsys=capsys)
    assert code == 130
    assert "user_abort" in stderr


def test_keyboard_interrupt_json_path(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = _run_main(
        ["--json"],
        side_effect=KeyboardInterrupt(),
        capsys=capsys,
    )
    assert code == 130
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["error"]["kind"] == "user_abort"
    assert payload["error"]["code"] == 130


# ---------------------------------------------------------------------------
# Sys.argv default
# ---------------------------------------------------------------------------


def test_argv_pre_scan_respects_double_dash_sentinel() -> None:
    """``--json`` after ``--`` is a positional argument, not a flag."""
    assert _argv_has_json_flag(["validate", "--", "--json"]) is False
    assert _argv_has_json_flag(["validate", "--json", "--"]) is True
    assert _argv_has_json_flag(["--", "--json", "--anything"]) is False


def test_click_exception_exit_code_is_honoured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``ClickException`` subclasses with custom ``exit_code`` (e.g.
    ``FileError``) propagate their declared code rather than being
    flattened to 2."""

    class CustomClickError(click.ClickException):
        exit_code = 7

    code, _, _stderr = _run_main(
        [],
        side_effect=CustomClickError("custom failure"),
        capsys=capsys,
    )
    assert code == 7
    # JSON path also respects the exit code.
    code_j, stdout_j, _ = _run_main(
        ["--json"],
        side_effect=CustomClickError("custom failure"),
        capsys=capsys,
    )
    assert code_j == 7
    payload = json.loads(stdout_j)
    assert payload["error"]["code"] == 7


def test_click_abort_renders_as_user_abort(capsys: pytest.CaptureFixture[str]) -> None:
    """``click.Abort`` (raised by ``click.confirm`` on Ctrl-C inside a
    prompt) must surface as ``user_abort`` + exit 130, not propagate as
    an uncaught traceback."""
    code, _, stderr = _run_main([], side_effect=click.Abort(), capsys=capsys)
    assert code == 130
    assert "user_abort" in stderr


def test_main_defaults_to_sys_argv_when_argv_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``main()`` without args should use ``sys.argv[1:]``."""
    monkeypatch.setattr("sys.argv", ["apmode", "--json"])
    stub = _StubApp(side_effect=None)
    with patch("apmode.cli.app", stub), pytest.raises(SystemExit) as excinfo:
        main()
    assert int(excinfo.value.code or 0) == 0
    # Confirm Click was given the right slice (everything past argv[0]).
    assert stub.calls[0]["kwargs"]["args"] == ["--json"]


def test_packaged_console_script_points_at_typed_entrypoint() -> None:
    """The installed `apmode` command must use the same catcher as `python -m apmode`."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["apmode"] == "apmode.__main__:main"
