# SPDX-License-Identifier: GPL-2.0-or-later
"""fish completion installer.

fish auto-loads completions from ``$XDG_CONFIG_HOME/fish/completions/``
(or ``~/.config/fish/completions/`` when ``XDG_CONFIG_HOME`` is unset).
Per fish convention every command gets its own ``<command>.fish`` file
in that directory; we drop ``apmode.fish`` and that's it. No marker
block, no rc-file edits — uninstall just deletes the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from apmode.shells import InstallResult, UninstallResult
from apmode.shells._rcfile import atomic_write, backup_once


class FishStrategy:
    """``apmode completion install`` for fish."""

    name = "fish"

    def _config_dir(self) -> Path:
        env = os.environ.get("XDG_CONFIG_HOME")
        return Path(env) if env else Path.home() / ".config"

    def rc_path(self) -> Path:
        return self._config_dir() / "fish" / "completions" / "apmode.fish"

    def completion_script(self, prog_name: str = "apmode") -> str:
        env_var = f"_{prog_name.upper()}_COMPLETE"
        # ``env VAR=value command | source`` is the portable form. Plain
        # ``VAR=value command`` (POSIX-style) only became valid in fish
        # 4.0 (Jan 2025); fish 3.x parses ``_APMODE_COMPLETE=fish_source``
        # as a literal command name and silently emits no completions.
        # The ``env`` prefix works on every fish release we support.
        return f"env {env_var}=fish_source {prog_name} | source"

    def install(self, prog_name: str = "apmode") -> InstallResult:
        path = self.rc_path()
        new_content = self.completion_script(prog_name) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") == new_content:
            return InstallResult(shell=self.name, path=path, action="already_installed")
        existed = path.exists()
        backup_once(path)
        atomic_write(path, new_content)
        return InstallResult(
            shell=self.name,
            path=path,
            action="updated" if existed else "installed",
        )

    def uninstall(self) -> UninstallResult:
        path = self.rc_path()
        if not path.exists():
            return UninstallResult(shell=self.name, path=path, action="absent")
        backup_once(path)
        path.unlink()
        return UninstallResult(shell=self.name, path=path, action="uninstalled")
