# SPDX-License-Identifier: GPL-2.0-or-later
"""bash completion installer.

Strategy: write a marker block into ``~/.bashrc``. The block sources
the output of ``_APMODE_COMPLETE=bash_source apmode`` — Click's standard
shell-completion contract. Sourcing dynamically keeps the block tiny
and means a future change to APMODE's command surface picks itself up
automatically (no re-install required).

Why ``~/.bashrc`` and not ``~/.bash_profile`` / ``~/.profile``? On
Linux + WSL ``~/.bashrc`` is sourced for every interactive non-login
shell, which is the dominant case for shell-completion users. macOS
``Terminal.app`` runs login shells, but every modern dotfile setup
sources ``~/.bashrc`` from ``~/.bash_profile`` (and operators who don't
get this far rarely have completion problems anyway).
"""

from __future__ import annotations

from pathlib import Path

from apmode.shells import InstallResult, UninstallResult
from apmode.shells._rcfile import (
    atomic_write,
    backup_once,
    read_rc,
    remove_block,
    upsert_block,
)


class BashStrategy:
    """``apmode completion install`` for bash."""

    name = "bash"

    def rc_path(self) -> Path:
        return Path.home() / ".bashrc"

    def completion_script(self, prog_name: str = "apmode") -> str:
        # ``eval`` of the dynamic source is the canonical Click idiom.
        # The leading variable lets bash render an environment-prefixed
        # command without requiring a subshell quirks dance.
        env_var = f"_{prog_name.upper()}_COMPLETE"
        return f'eval "$({env_var}=bash_source {prog_name})"'

    def install(self, prog_name: str = "apmode") -> InstallResult:
        path = self.rc_path()
        existing = read_rc(path)
        backup_once(path)
        update = upsert_block(existing, self.completion_script(prog_name))
        if update.action != "already_installed":
            atomic_write(path, update.new_text)
        return InstallResult(shell=self.name, path=path, action=update.action)

    def uninstall(self) -> UninstallResult:
        path = self.rc_path()
        existing = read_rc(path)
        if not existing:
            return UninstallResult(shell=self.name, path=path, action="absent")
        backup_once(path)
        removal = remove_block(existing)
        if removal.action == "uninstalled":
            atomic_write(path, removal.new_text)
        return UninstallResult(shell=self.name, path=path, action=removal.action)
