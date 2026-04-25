# SPDX-License-Identifier: GPL-2.0-or-later
"""PowerShell completion installer.

PowerShell stores user customisations in ``$PROFILE``. The exact path
depends on which PowerShell flavour is running:

* **Windows PowerShell 5.1** (the version that ships with Windows
  baseline): ``$HOME/Documents/WindowsPowerShell/Microsoft.PowerShell_profile.ps1``.
* **PowerShell 7+** (Core, cross-platform): the same path but under
  ``Documents/PowerShell/`` instead. On macOS / Linux the documents
  folder may be at ``~/.config/powershell/`` instead — PS Core resolves
  it via its own configuration cascade.

We don't try to be clever: ``shellingham`` reports ``"powershell"`` for
WPS5 and ``"pwsh"`` for PS7 and we use that as the discriminator. On
non-Windows platforms ``$HOME/.config/powershell/`` is the canonical
PS Core profile location.

Otherwise the strategy is the same shape as bash: write a marker block
into ``$PROFILE`` so existing prelude / postlude is preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

from apmode.shells import InstallResult, UninstallResult
from apmode.shells._rcfile import (
    atomic_write,
    backup_once,
    read_rc,
    remove_block,
    upsert_block,
)


class PowerShellStrategy:
    """``apmode completion install`` for Windows PowerShell or PowerShell Core."""

    def __init__(self, *, is_pwsh: bool) -> None:
        self.is_pwsh = is_pwsh

    @property
    def name(self) -> str:
        return "pwsh" if self.is_pwsh else "powershell"

    def rc_path(self) -> Path:
        home = Path.home()
        if self.is_pwsh:
            # PS Core 7+. On Windows it lives under ``Documents/PowerShell/``;
            # on macOS / Linux PS Core uses ``~/.config/powershell/``.
            if sys.platform.startswith("win"):
                return home / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1"
            return home / ".config" / "powershell" / "Microsoft.PowerShell_profile.ps1"
        # Windows PowerShell 5.x. Always under ``Documents/WindowsPowerShell/``.
        return home / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1"

    def completion_script(self, prog_name: str = "apmode") -> str:
        # Click's PowerShell completion handler is invoked by setting the
        # env var and piping the output into Invoke-Expression. We wrap
        # in a try/catch so a missing apmode binary does not break the
        # operator's profile entirely.
        env_var = f"_{prog_name.upper()}_COMPLETE"
        return (
            f"$env:{env_var} = 'pwsh_source'\n"
            f"try {{ {prog_name} | Out-String | Invoke-Expression }} "
            f"catch {{ Write-Verbose 'apmode completion unavailable: $_' }}\n"
            f"Remove-Item Env:{env_var}"
        )

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
