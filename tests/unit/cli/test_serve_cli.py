# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI-layer tests for ``apmode serve`` (plan Task 35).

The command is the thin typer wrapper around
:func:`apmode.api.app.build_app`. Tests verify three load-bearing
behaviours:

* the loopback default + the ``--allow-public`` gate (security policy:
  no auth on the API, bundles may carry patient data)
* the [api]-extras-missing path
* that uvicorn is invoked with the expected ``host`` / ``port`` /
  ``timeout_graceful_shutdown`` kwargs

We do **not** spin up a real HTTP server in unit tests — `uvicorn.Server`
is patched so the command returns synchronously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from apmode.cli import _is_loopback_host, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# _is_loopback_host helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "::1", "127.5.6.7"],
)
def test_loopback_host_accepted(host: str) -> None:
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "192.168.1.10",
        "10.0.0.5",
        "example.com",
        "::",
        "fe80::1",
    ],
)
def test_non_loopback_host_rejected(host: str) -> None:
    assert _is_loopback_host(host) is False


# ---------------------------------------------------------------------------
# serve command — defaults bind to loopback
# ---------------------------------------------------------------------------


def _patched_serve_invocation(args: list[str]) -> Any:
    """Invoke ``apmode serve`` with build_app + uvicorn patched.

    Returns the ``CliResult`` plus the captured Config + Server mocks so
    individual tests can assert against the kwargs uvicorn was given.
    """
    fake_app = MagicMock(name="fastapi_app")
    config_mock = MagicMock(name="uvicorn.Config")
    server_mock = MagicMock(name="uvicorn.Server")
    server_mock.return_value.run.return_value = None

    with (
        patch("apmode.api.app.build_app", return_value=fake_app) as build_app_mock,
        patch("uvicorn.Config", config_mock),
        patch("uvicorn.Server", server_mock),
    ):
        result = runner.invoke(app, ["serve", *args])

    return result, build_app_mock, config_mock, server_mock


def test_serve_default_bind_is_loopback(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "runs" / ".apmode_runs.sqlite3"
    result, build_app_mock, config_mock, server_mock = _patched_serve_invocation(
        ["--runs-dir", str(runs_dir), "--db-path", str(db_path)]
    )

    assert result.exit_code == 0, result.output
    # build_app is called exactly once, with resolved paths and the
    # default backend allowlist (nlmixr2 only).
    build_app_mock.assert_called_once()
    kwargs = build_app_mock.call_args.kwargs
    assert kwargs["runs_dir"] == runs_dir.resolve()
    assert kwargs["db_path"] == db_path.resolve()
    assert kwargs["allow_backends"] == ("nlmixr2",)

    # uvicorn.Config receives the loopback host, our chosen port, and
    # the graceful-shutdown budget.
    config_mock.assert_called_once()
    config_kwargs = config_mock.call_args.kwargs
    assert config_kwargs["host"] == "127.0.0.1"
    assert config_kwargs["port"] == 8765
    assert config_kwargs["timeout_graceful_shutdown"] == 30
    assert config_kwargs["log_level"] == "info"

    # Server.run is invoked exactly once.
    server_mock.return_value.run.assert_called_once_with()


def test_serve_allow_bayesian_extends_backend_allowlist(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    db_path = tmp_path / "runs" / ".apmode_runs.sqlite3"
    result, build_app_mock, _, _ = _patched_serve_invocation(
        [
            "--runs-dir",
            str(runs_dir),
            "--db-path",
            str(db_path),
            "--allow-bayesian",
        ]
    )

    assert result.exit_code == 0, result.output
    assert build_app_mock.call_args.kwargs["allow_backends"] == (
        "nlmixr2",
        "bayesian_stan",
    )


def test_serve_custom_port_and_shutdown_budget(tmp_path: Path) -> None:
    result, _, config_mock, _ = _patched_serve_invocation(
        [
            "--runs-dir",
            str(tmp_path / "runs"),
            "--db-path",
            str(tmp_path / "runs" / "db.sqlite3"),
            "--port",
            "9001",
            "--timeout-graceful-shutdown",
            "45",
            "--log-level",
            "warning",
        ]
    )
    assert result.exit_code == 0, result.output
    config_kwargs = config_mock.call_args.kwargs
    assert config_kwargs["port"] == 9001
    assert config_kwargs["timeout_graceful_shutdown"] == 45
    assert config_kwargs["log_level"] == "warning"


# ---------------------------------------------------------------------------
# Loopback-policy gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "10.0.0.5"],
)
def test_serve_refuses_non_loopback_without_allow_public(host: str, tmp_path: Path) -> None:
    result, build_app_mock, _, server_mock = _patched_serve_invocation(
        [
            "--runs-dir",
            str(tmp_path / "runs"),
            "--db-path",
            str(tmp_path / "runs" / "db.sqlite3"),
            "--host",
            host,
        ]
    )

    assert result.exit_code == 2, result.output
    assert "non-loopback" in result.output
    assert "--allow-public" in result.output
    # Critical: build_app must NOT be called when the gate fires —
    # otherwise we leak a SQLite handle and a runs_dir mkdir.
    build_app_mock.assert_not_called()
    server_mock.return_value.run.assert_not_called()


def test_serve_allow_public_proceeds_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--allow-public`` proceeds only when its hard requirements are met:

    * ``APMODE_API_KEY`` set so the API-key dependency rejects
      unauthenticated requests.
    * ``--dataset-root`` set so a remote caller cannot probe arbitrary
      paths via ``POST /runs``.

    Both are enforced by the serve command and exit ``2`` if missing —
    the test sets them and asserts the success path emits the warning
    and dispatches to ``build_app`` + ``uvicorn.Server.run``.
    """
    monkeypatch.setenv("APMODE_API_KEY", "test-api-key-32-bytes-of-secure-tokens")
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()

    result, build_app_mock, config_mock, server_mock = _patched_serve_invocation(
        [
            "--runs-dir",
            str(tmp_path / "runs"),
            "--db-path",
            str(tmp_path / "runs" / "db.sqlite3"),
            "--host",
            "0.0.0.0",
            "--allow-public",
            "--dataset-root",
            str(dataset_root),
        ]
    )

    assert result.exit_code == 0, result.output
    # The warning line is emitted to stderr, which CliRunner mixes into
    # ``output`` by default — check for the substring.
    assert "non-loopback" in result.output or "warning" in result.output.lower()
    build_app_mock.assert_called_once()
    config_mock.assert_called_once()
    assert config_mock.call_args.kwargs["host"] == "0.0.0.0"
    server_mock.return_value.run.assert_called_once_with()


# ---------------------------------------------------------------------------
# [api] extras-not-installed path
# ---------------------------------------------------------------------------


def test_serve_handles_missing_api_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``apmode.api.app`` cannot be imported, exit 1 with a hint."""
    import builtins

    real_import = builtins.__import__

    def _import_blocking_api(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "apmode.api.app" or (name.startswith("apmode.api") and "build_app" in fromlist):
            raise ImportError("No module named 'fastapi'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_blocking_api)

    result = runner.invoke(
        app,
        [
            "serve",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--db-path",
            str(tmp_path / "runs" / "db.sqlite3"),
        ],
    )

    assert result.exit_code == 1, result.output
    assert "uv sync --extra api" in result.output


# ---------------------------------------------------------------------------
# Help text shape (smoke)
# ---------------------------------------------------------------------------


def test_serve_help_lists_endpoints() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    # The help text advertises the endpoint contract so users don't
    # have to dig into routes.py to know what they get.
    assert "/healthz" in result.output
    assert "/runs" in result.output
    assert "Retry-After" in result.output
