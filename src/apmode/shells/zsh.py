# SPDX-License-Identifier: GPL-2.0-or-later
"""zsh completion installer.

Two install strategies depending on what the operator has:

1. **Vanilla zsh.** Mark a block in ``${ZDOTDIR:-$HOME}/.zshrc``. Same
   shape as the bash strategy.
2. **oh-my-zsh.** When ``~/.oh-my-zsh`` exists, drop a single file at
   ``${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/apmode-completion.zsh``.
   oh-my-zsh auto-sources every ``*.zsh`` in that directory, so a
   file-drop is the idiomatic option — operators who run oh-my-zsh
   expect plugins to live as files, not as marker blocks in their
   ``.zshrc``. Uninstall deletes the file.

The detection precedence — file-drop first, marker block second — means
an operator who installs oh-my-zsh after a vanilla install ends up with
two artefacts. ``uninstall`` cleans both.
"""

from __future__ import annotations

import os
from pathlib import Path

from apmode.shells import InstallResult, UninstallResult
from apmode.shells._rcfile import (
    atomic_write,
    backup_once,
    read_rc,
    remove_block,
    upsert_block,
)


class ZshStrategy:
    """``apmode completion install`` for zsh."""

    name = "zsh"

    def _zdotdir(self) -> Path:
        return Path(os.environ.get("ZDOTDIR") or str(Path.home()))

    def _rc_path_marker(self) -> Path:
        return self._zdotdir() / ".zshrc"

    def _ohmyzsh_root(self) -> Path | None:
        omz = Path.home() / ".oh-my-zsh"
        return omz if omz.is_dir() else None

    def _ohmyzsh_custom_dir(self) -> Path:
        env = os.environ.get("ZSH_CUSTOM")
        if env:
            return Path(env)
        return Path.home() / ".oh-my-zsh" / "custom"

    def _ohmyzsh_file(self) -> Path:
        return self._ohmyzsh_custom_dir() / "apmode-completion.zsh"

    def rc_path(self) -> Path:
        # Reported to the operator. The file-drop is preferred under
        # oh-my-zsh; otherwise the rc file is reported.
        if self._ohmyzsh_root() is not None:
            return self._ohmyzsh_file()
        return self._rc_path_marker()

    def completion_script(self, prog_name: str = "apmode") -> str:
        env_var = f"_{prog_name.upper()}_COMPLETE"
        return f'eval "$({env_var}=zsh_source {prog_name})"'

    def install(self, prog_name: str = "apmode") -> InstallResult:
        script = self.completion_script(prog_name)
        if self._ohmyzsh_root() is not None:
            return self._install_file_drop(script)
        return self._install_marker(script)

    def _install_file_drop(self, script: str) -> InstallResult:
        path = self._ohmyzsh_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        new_content = script + "\n"
        if path.exists() and path.read_text(encoding="utf-8") == new_content:
            return InstallResult(shell=self.name, path=path, action="already_installed")
        backup_once(path)
        existed = path.exists()
        atomic_write(path, new_content)
        return InstallResult(
            shell=self.name,
            path=path,
            action="updated" if existed else "installed",
        )

    def _install_marker(self, script: str) -> InstallResult:
        path = self._rc_path_marker()
        existing = read_rc(path)
        backup_once(path)
        update = upsert_block(existing, script)
        if update.action != "already_installed":
            atomic_write(path, update.new_text)
        return InstallResult(shell=self.name, path=path, action=update.action)

    def uninstall(self) -> UninstallResult:
        # Try both potential locations so a switch from vanilla to
        # oh-my-zsh (or back) leaves no stale block / file behind.
        cleaned: list[tuple[Path, str]] = []

        marker_path = self._rc_path_marker()
        if marker_path.exists():
            text = read_rc(marker_path)
            removal = remove_block(text)
            if removal.action == "uninstalled":
                backup_once(marker_path)
                atomic_write(marker_path, removal.new_text)
                cleaned.append((marker_path, "uninstalled"))

        file_drop = self._ohmyzsh_file()
        if file_drop.exists():
            backup_once(file_drop)
            file_drop.unlink()
            cleaned.append((file_drop, "uninstalled"))

        if not cleaned:
            # Report whichever path we *would* have written to so the
            # operator can verify the right place was checked.
            return UninstallResult(shell=self.name, path=self.rc_path(), action="absent")
        # Prefer the file-drop in the report when both were present —
        # it's the more visible artefact under oh-my-zsh.
        path, action = cleaned[-1]
        return UninstallResult(shell=self.name, path=path, action=action)
