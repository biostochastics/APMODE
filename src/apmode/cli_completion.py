# SPDX-License-Identifier: GPL-2.0-or-later
"""``apmode completion {install,show,uninstall}`` Typer sub-app.

Three commands, one shared shell-detection step:

* ``install`` writes the per-shell completion block (or file-drop) and
  reports the action taken.
* ``show`` renders the completion source on stdout so an operator can
  pipe it into a file of their choosing.
* ``uninstall`` removes the marker block (or the file-drop) without
  touching surrounding bytes.

Each command honours the ``--json`` flag wired through
:mod:`apmode._json_ctx`. The JSON shape::

    {"ok": true, "shell": "zsh", "path": "/Users/x/.zshrc",
     "action": "installed|already_installed|updated|uninstalled|absent",
     "marker": ">>> apmode completion >>>"}

Shell detection priority:

1. ``--shell`` flag (operator override).
2. ``$SHELL`` env var basename, when set.
3. ``shellingham.detect_shell()`` (which inspects the parent process
   so `apmode completion install` correctly works from inside `nohup`,
   IDE terminals, etc.).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from apmode.cli_errors import ConfigError
from apmode.shells import (
    InstallResult,
    UninstallResult,
    UnsupportedShellError,
    get_strategy,
)

# Sub-app registered into the main typer app at
# ``apmode.cli`` via ``app.add_typer(completion_app, name="completion")``.
completion_app = typer.Typer(
    name="completion",
    help="Install / show / uninstall shell completion for the apmode CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


_SUPPORTED_SHELLS = ("bash", "zsh", "fish", "powershell", "pwsh")


def _detect_shell(explicit: str | None) -> str:
    """Resolve the shell name we should target.

    Order of precedence:

    1. The operator's explicit ``--shell`` flag (lower-cased).
    2. The basename of ``$SHELL`` if it matches a supported value.
    3. ``shellingham.detect_shell()`` — parent-process inspection,
       which works inside IDE terminals / ``nohup`` / wrappers where
       ``$SHELL`` may be stale or unset.

    Raises :class:`apmode.cli_errors.ConfigError` when nothing resolves.
    """
    if explicit is not None:
        candidate = explicit.lower().strip()
        if candidate not in _SUPPORTED_SHELLS:
            raise ConfigError(
                f"unsupported --shell value: {explicit!r}",
                details={
                    "supported": list(_SUPPORTED_SHELLS),
                    "got": explicit,
                },
            )
        return candidate

    sh_env = os.environ.get("SHELL", "").strip()
    if sh_env:
        basename = Path(sh_env).name.lower()
        if basename in _SUPPORTED_SHELLS:
            return basename

    # Fall back to shellingham — its parent-process detection is more
    # robust than $SHELL on macOS (Terminal.app sometimes leaves $SHELL
    # pointing at /bin/bash even when the user is running zsh).
    try:
        import shellingham

        detected, _ = shellingham.detect_shell()
    except Exception as exc:  # pragma: no cover - shellingham failure path
        raise ConfigError(
            "could not detect the active shell. Pass --shell explicitly.",
            details={"reason": str(exc)},
        ) from exc

    name: str = str(detected).lower()
    if name not in _SUPPORTED_SHELLS:
        raise ConfigError(
            f"detected shell {name!r} is not supported by apmode completion.",
            details={"detected": name, "supported": list(_SUPPORTED_SHELLS)},
        )
    return name


def _install_envelope(result: InstallResult) -> dict[str, object]:
    return {
        "ok": True,
        "shell": result.shell,
        "path": str(result.path),
        "action": result.action,
        "marker": result.marker,
    }


def _uninstall_envelope(result: UninstallResult) -> dict[str, object]:
    return {
        "ok": True,
        "shell": result.shell,
        "path": str(result.path),
        "action": result.action,
        "marker": result.marker,
    }


@completion_app.command("install")
def install(
    shell: Annotated[
        str | None,
        typer.Option(
            "--shell",
            help=(
                "Override shell auto-detection. Accepted values: "
                "bash, zsh, fish, powershell, pwsh."
            ),
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a structured JSON envelope on stdout instead of Rich text.",
        ),
    ] = False,
) -> None:
    """Install shell completion for the apmode CLI.

    Idempotent: re-running on an already-installed system reports
    ``action="already_installed"``. After an APMODE upgrade the block
    is rewritten in place (action ``"updated"``) so the rendered
    source stays current.
    """
    shell_name = _detect_shell(shell)
    try:
        strategy = get_strategy(shell_name)
    except UnsupportedShellError as exc:
        raise ConfigError(
            str(exc),
            details={"detected": shell_name, "supported": list(_SUPPORTED_SHELLS)},
        ) from exc

    result = strategy.install()
    if output_json:
        print(json.dumps(_install_envelope(result), ensure_ascii=False))
        return

    # Rich text path. Use plain ``print`` so the output is shell-script
    # friendly (no ANSI codes when stdout is not a TTY).
    typer.echo(
        f"apmode completion {result.action} for {result.shell} at {result.path}\n"
        f"Open a new shell or `source {result.path}` to activate."
    )


@completion_app.command("show")
def show(
    shell: Annotated[
        str | None,
        typer.Option(
            "--shell",
            help="Override shell auto-detection (see `install --help`).",
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Wrap the script in a JSON envelope on stdout.",
        ),
    ] = False,
) -> None:
    """Print the completion source for ``shell`` on stdout, no install.

    Useful for piping into a custom location, or for inspecting what
    ``install`` would write.
    """
    shell_name = _detect_shell(shell)
    try:
        strategy = get_strategy(shell_name)
    except UnsupportedShellError as exc:
        raise ConfigError(
            str(exc),
            details={"detected": shell_name, "supported": list(_SUPPORTED_SHELLS)},
        ) from exc

    script = strategy.completion_script()
    if output_json:
        envelope = {"ok": True, "shell": strategy.name, "script": script}
        print(json.dumps(envelope, ensure_ascii=False))
        return
    typer.echo(script)


@completion_app.command("uninstall")
def uninstall(
    shell: Annotated[
        str | None,
        typer.Option(
            "--shell",
            help="Override shell auto-detection (see `install --help`).",
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a structured JSON envelope on stdout instead of Rich text.",
        ),
    ] = False,
) -> None:
    """Remove the apmode completion block (or file-drop).

    Surrounding bytes in the rc file are preserved exactly. Reports
    ``action="absent"`` when nothing was found to remove — never an
    error, since a no-op uninstall is a successful uninstall.
    """
    shell_name = _detect_shell(shell)
    try:
        strategy = get_strategy(shell_name)
    except UnsupportedShellError as exc:
        raise ConfigError(
            str(exc),
            details={"detected": shell_name, "supported": list(_SUPPORTED_SHELLS)},
        ) from exc

    result = strategy.uninstall()
    if output_json:
        print(json.dumps(_uninstall_envelope(result), ensure_ascii=False))
        return
    if result.action == "absent":
        typer.echo(f"apmode completion not found for {result.shell} at {result.path}")
    else:
        typer.echo(f"apmode completion removed from {result.path}")


__all__ = ["completion_app"]
