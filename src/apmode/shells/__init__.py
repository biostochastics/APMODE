# SPDX-License-Identifier: GPL-2.0-or-later
"""Per-shell strategies for ``apmode completion install|show|uninstall``.

Every supported shell exposes a :class:`ShellStrategy` whose four methods —
``rc_path``, ``completion_script``, ``install``, ``uninstall`` — encapsulate
the differences between bash (mark a block in ``~/.bashrc``), zsh
(file-drop into oh-my-zsh's custom dir if present, else mark a block in
``~/.zshrc``), fish (file-drop into ``$XDG_CONFIG_HOME/fish/completions``),
and PowerShell (mark a block in ``$PROFILE``).

The CLI front-end at :mod:`apmode.cli_completion` is shell-agnostic: it
detects the active shell via ``shellingham`` and dispatches to a strategy
returned from :func:`get_strategy`. Adding a new shell is therefore a
matter of dropping a new strategy module under this package and
registering it in the dispatch table — no front-end changes needed.

Why a registry and not subclasses? Each strategy holds shell-specific
heuristics (``ZDOTDIR``, oh-my-zsh detection, PS5-vs-PS7 ``$PROFILE``)
that do not generalise. A ``Protocol`` keeps the surface flat while
mypy still verifies every strategy implements every method.
"""

from __future__ import annotations

from typing import Protocol

# The marker pair below is the canonical fence around every block this
# package writes into a shell rc file. The form ``>>> apmode completion >>>``
# (and the reverse close) intentionally matches the convention popularised
# by ``conda init`` so an operator who has seen one knows what the other
# means at a glance. Both lines must be byte-stable so ``uninstall`` can
# delete *exactly* what ``install`` wrote and leave the rest of the rc
# file unchanged.
MARKER_OPEN = "# >>> apmode completion >>>"
MARKER_CLOSE = "# <<< apmode completion <<<"

# A version signature embedded in the rendered block. When ``install`` is
# re-run after an APMODE upgrade the existing block is rewritten only if
# the signature has changed, so a no-op upgrade does not bump the rc
# file's mtime. The constant lives here (not in ``cli_completion``) to
# keep the strategy modules free of CLI-layer imports.
COMPLETION_SCHEMA = "1"


class ShellStrategy(Protocol):
    """Common surface every shell-specific strategy must implement.

    The methods are deliberately small. ``install`` and ``uninstall``
    return a payload that the JSON renderer in
    :mod:`apmode.cli_completion` turns into the canonical envelope:

        {"ok": true, "shell": "<name>", "path": "<rc-file>",
         "action": "installed|already_installed|updated|uninstalled|absent",
         "marker": "..."}

    ``name`` is declared as a ``@property`` so subclasses can satisfy
    it via either a class-level attribute (bash, zsh, fish) or a
    computed property (PowerShell, where the value depends on whether
    we were instantiated for PowerShell 5 or PowerShell 7+).
    """

    @property
    def name(self) -> str:
        """Shell identifier (e.g. ``"bash"``, ``"zsh"``, ``"fish"``, ``"powershell"``)."""

    def rc_path(self) -> _PathLike:
        """Resolved absolute path that ``install`` will write to."""

    def completion_script(self, prog_name: str = "apmode") -> str:
        """Generate the shell-specific completion source as a string."""

    def install(self, prog_name: str = "apmode") -> InstallResult:
        """Idempotently write the completion block; return action + path."""

    def uninstall(self) -> UninstallResult:
        """Remove the marker block (or file). Surrounding bytes preserved."""


# These names are imported lazily by strategies to avoid circular imports.
# ``_PathLike`` is just an alias for ``pathlib.Path`` exported at module
# scope so the ``Protocol`` annotation is type-checkable without forcing
# every strategy module to re-import ``pathlib``.

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

_PathLike = Path


@dataclass(frozen=True)
class InstallResult:
    """Return value from :meth:`ShellStrategy.install`."""

    shell: str
    path: Path
    action: str  # "installed" | "already_installed" | "updated"
    marker: str = MARKER_OPEN


@dataclass(frozen=True)
class UninstallResult:
    """Return value from :meth:`ShellStrategy.uninstall`."""

    shell: str
    path: Path
    action: str  # "uninstalled" | "absent"
    marker: str = MARKER_OPEN


def get_strategy(shell_name: str) -> ShellStrategy:
    """Look up the strategy for ``shell_name`` (case-insensitive).

    ``shellingham`` returns names like ``"bash"`` / ``"zsh"`` / ``"fish"``;
    on Windows it returns ``"powershell"`` (Windows PowerShell 5) or
    ``"pwsh"`` (PowerShell Core 7+). Both forms route to the same
    ``PowerShellStrategy`` and the strategy resolves the correct
    ``$PROFILE`` path internally.
    """
    name = shell_name.lower().strip()

    # Imports are deferred to avoid pulling shell-specific modules into
    # the import graph for a CLI run that doesn't touch completion.
    if name == "bash":
        from apmode.shells.bash import BashStrategy

        return BashStrategy()
    if name == "zsh":
        from apmode.shells.zsh import ZshStrategy

        return ZshStrategy()
    if name == "fish":
        from apmode.shells.fish import FishStrategy

        return FishStrategy()
    if name in {"powershell", "pwsh"}:
        from apmode.shells.powershell import PowerShellStrategy

        return PowerShellStrategy(is_pwsh=(name == "pwsh"))
    raise UnsupportedShellError(name)


class UnsupportedShellError(Exception):
    """Raised by :func:`get_strategy` when the shell name is not handled.

    The CLI front-end translates this into a typed
    :class:`apmode.cli_errors.ConfigError` so the user sees a structured
    envelope rather than a stack trace.
    """

    def __init__(self, shell_name: str) -> None:
        super().__init__(f"unsupported shell: {shell_name!r}")
        self.shell_name = shell_name


__all__ = [
    "COMPLETION_SCHEMA",
    "MARKER_CLOSE",
    "MARKER_OPEN",
    "InstallResult",
    "ShellStrategy",
    "UninstallResult",
    "UnsupportedShellError",
    "get_strategy",
]
