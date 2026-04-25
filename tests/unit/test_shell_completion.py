# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for ``apmode completion {install,show,uninstall}``.

Coverage layout:

* :class:`TestRcFileHelpers` — the shared marker / atomic-write
  machinery in ``apmode.shells._rcfile``.
* :class:`TestBashStrategy`, :class:`TestZshStrategy` (vanilla +
  oh-my-zsh), :class:`TestFishStrategy`, :class:`TestPowerShellStrategy`
  — per-shell behaviour driven through monkeypatched ``$HOME``,
  ``$XDG_CONFIG_HOME``, ``$ZDOTDIR`` so the tests never touch the real
  user environment.
* :class:`TestCliCompletion` — end-to-end through ``CliRunner``
  exercising shell-detection precedence, ``--shell`` override,
  ``--json`` envelope, idempotency, and uninstall byte-equality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apmode.cli import app
from apmode.shells import (
    COMPLETION_SCHEMA,
    MARKER_CLOSE,
    MARKER_OPEN,
    UnsupportedShellError,
    get_strategy,
)
from apmode.shells._rcfile import (
    atomic_write,
    backup_once,
    read_rc,
    remove_block,
    reset_backup_state_for_tests,
    upsert_block,
)
from apmode.shells.bash import BashStrategy
from apmode.shells.fish import FishStrategy
from apmode.shells.powershell import PowerShellStrategy
from apmode.shells.zsh import ZshStrategy

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_backup_state() -> None:
    """Per-process backup tracker is shared across tests; reset before each."""
    reset_backup_state_for_tests()


# ---------------------------------------------------------------------------
# _rcfile helpers — pure functions, no I/O except the atomic-write tests
# ---------------------------------------------------------------------------


class TestRcFileHelpers:
    def test_upsert_block_appends_when_absent(self) -> None:
        before = "# user content\n"
        update = upsert_block(before, "echo hello")
        assert update.action == "installed"
        assert MARKER_OPEN in update.new_text
        assert MARKER_CLOSE in update.new_text
        assert "echo hello" in update.new_text
        # Original content survives byte-for-byte at the start of the file.
        assert update.new_text.startswith("# user content\n")

    def test_upsert_block_is_idempotent(self) -> None:
        once = upsert_block("# pre\n", "echo hello").new_text
        twice = upsert_block(once, "echo hello")
        assert twice.action == "already_installed"
        assert twice.new_text == once

    def test_upsert_block_rewrites_on_body_change(self) -> None:
        once = upsert_block("# pre\n", "echo old").new_text
        twice = upsert_block(once, "echo new")
        assert twice.action == "updated"
        assert "echo new" in twice.new_text
        assert "echo old" not in twice.new_text
        # Prelude survived the rewrite.
        assert twice.new_text.startswith("# pre\n")

    def test_upsert_block_writes_version_signature(self) -> None:
        update = upsert_block("", "echo hello")
        assert f"# version: {COMPLETION_SCHEMA}" in update.new_text

    def test_upsert_block_preserves_postlude(self) -> None:
        before = "# pre\n# post-marker-text\n"
        update = upsert_block(before, "echo hello")
        # Re-uninstall must restore the postlude exactly.
        removal = remove_block(update.new_text)
        assert removal.action == "uninstalled"
        # The pre-existing newline structure is preserved; the
        # block introduces a single blank line before itself which is
        # collapsed back on remove.
        assert removal.new_text == before

    def test_remove_block_no_op_when_absent(self) -> None:
        before = "# user content\n"
        removal = remove_block(before)
        assert removal.action == "absent"
        assert removal.new_text == before

    def test_remove_block_collapses_introduced_blank_line(self) -> None:
        """A pristine install→uninstall cycle returns the file to its prior bytes."""
        original = "# pre\n"
        installed = upsert_block(original, "echo hello").new_text
        uninstalled = remove_block(installed).new_text
        assert uninstalled == original

    def test_atomic_write_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "file.txt"
        atomic_write(path, "hello world\n")
        assert path.read_text() == "hello world\n"
        # Parent directory was created.
        assert path.parent.is_dir()
        # No stray temp file left behind.
        assert not (path.parent / "file.txt.tmp").exists()

    def test_backup_once_makes_one_copy(self, tmp_path: Path) -> None:
        path = tmp_path / "rcfile"
        path.write_text("first\n")
        first = backup_once(path)
        assert first is not None
        assert first.name == "rcfile.bak"
        assert first.read_text() == "first\n"
        # Mutate the file; second backup_once should be a no-op.
        path.write_text("second\n")
        second = backup_once(path)
        assert second is None
        # First backup is preserved unchanged.
        assert first.read_text() == "first\n"

    def test_read_rc_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        assert read_rc(tmp_path / "nope") == ""


# ---------------------------------------------------------------------------
# BashStrategy
# ---------------------------------------------------------------------------


class TestBashStrategy:
    def test_rc_path_is_home_bashrc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        assert BashStrategy().rc_path() == tmp_path / ".bashrc"

    def test_install_writes_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        result = BashStrategy().install()
        assert result.action == "installed"
        assert result.path == tmp_path / ".bashrc"
        text = result.path.read_text()
        assert MARKER_OPEN in text
        assert "_APMODE_COMPLETE=bash_source apmode" in text

    def test_install_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        first = BashStrategy().install()
        second = BashStrategy().install()
        assert first.action == "installed"
        assert second.action == "already_installed"

    def test_install_then_uninstall_byte_equality(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        rc = tmp_path / ".bashrc"
        rc.write_text("# my custom rc\nexport FOO=bar\n")
        original_bytes = rc.read_text()

        BashStrategy().install()
        assert MARKER_OPEN in rc.read_text()  # block now present

        BashStrategy().uninstall()
        # Surrounding bytes restored exactly.
        assert rc.read_text() == original_bytes

    def test_uninstall_when_absent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        result = BashStrategy().uninstall()
        assert result.action == "absent"


# ---------------------------------------------------------------------------
# ZshStrategy — vanilla and oh-my-zsh paths
# ---------------------------------------------------------------------------


class TestZshStrategyVanilla:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Must be a HOME with no oh-my-zsh present.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("ZDOTDIR", raising=False)
        monkeypatch.delenv("ZSH_CUSTOM", raising=False)
        # Ensure ~/.oh-my-zsh does not exist.
        omz = tmp_path / ".oh-my-zsh"
        if omz.exists():
            for child in omz.iterdir():
                child.unlink()
            omz.rmdir()

    def test_rc_path_uses_zdotdir_if_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zdot = tmp_path / "zdot"
        zdot.mkdir()
        monkeypatch.setenv("ZDOTDIR", str(zdot))
        assert ZshStrategy().rc_path() == zdot / ".zshrc"

    def test_install_writes_block(self, tmp_path: Path) -> None:
        result = ZshStrategy().install()
        assert result.action == "installed"
        assert result.path == tmp_path / ".zshrc"
        assert "_APMODE_COMPLETE=zsh_source apmode" in result.path.read_text()


class TestZshStrategyOhMyZsh:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("ZDOTDIR", raising=False)
        monkeypatch.delenv("ZSH_CUSTOM", raising=False)
        # Materialise an oh-my-zsh layout.
        (tmp_path / ".oh-my-zsh" / "custom").mkdir(parents=True)

    def test_rc_path_is_custom_dir_file(self, tmp_path: Path) -> None:
        assert ZshStrategy().rc_path() == (
            tmp_path / ".oh-my-zsh" / "custom" / "apmode-completion.zsh"
        )

    def test_install_writes_file_drop(self, tmp_path: Path) -> None:
        result = ZshStrategy().install()
        assert result.action == "installed"
        assert result.path.exists()
        # File-drop content has no marker block — the file IS the block.
        text = result.path.read_text()
        assert MARKER_OPEN not in text
        assert "_APMODE_COMPLETE=zsh_source apmode" in text

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        first = ZshStrategy().install()
        second = ZshStrategy().install()
        assert first.action == "installed"
        assert second.action == "already_installed"

    def test_uninstall_removes_file(self, tmp_path: Path) -> None:
        ZshStrategy().install()
        result = ZshStrategy().uninstall()
        assert result.action == "uninstalled"
        assert not result.path.exists()

    def test_uninstall_cleans_both_marker_and_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Operator who switches between vanilla / oh-my-zsh: nothing left behind."""
        # Pre-populate marker block in .zshrc (simulating prior vanilla install).
        rc = tmp_path / ".zshrc"
        rc.write_text(
            f"{MARKER_OPEN}\n# version: {COMPLETION_SCHEMA}\nold-marker\n{MARKER_CLOSE}\n"
        )
        # And install the oh-my-zsh file too.
        ZshStrategy().install()
        # Both should be cleaned by uninstall.
        result = ZshStrategy().uninstall()
        assert result.action == "uninstalled"
        assert MARKER_OPEN not in rc.read_text()
        assert not (tmp_path / ".oh-my-zsh" / "custom" / "apmode-completion.zsh").exists()


# ---------------------------------------------------------------------------
# FishStrategy
# ---------------------------------------------------------------------------


class TestFishStrategy:
    def test_rc_path_uses_xdg_config_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert (
            FishStrategy().rc_path() == tmp_path / "xdg" / "fish" / "completions" / "apmode.fish"
        )

    def test_rc_path_falls_back_to_home_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        assert (
            FishStrategy().rc_path()
            == tmp_path / ".config" / "fish" / "completions" / "apmode.fish"
        )

    def test_install_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        result = FishStrategy().install()
        assert result.action == "installed"
        text = result.path.read_text()
        assert "_APMODE_COMPLETE=fish_source apmode | source" in text

    def test_install_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        FishStrategy().install()
        result = FishStrategy().install()
        assert result.action == "already_installed"

    def test_uninstall_deletes_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        FishStrategy().install()
        result = FishStrategy().uninstall()
        assert result.action == "uninstalled"
        assert not result.path.exists()


# ---------------------------------------------------------------------------
# PowerShellStrategy — PS5 vs PS7 path resolution
# ---------------------------------------------------------------------------


class TestPowerShellStrategy:
    def test_pwsh_path_on_posix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("sys.platform", "darwin")
        path = PowerShellStrategy(is_pwsh=True).rc_path()
        assert path == tmp_path / ".config" / "powershell" / "Microsoft.PowerShell_profile.ps1"

    def test_pwsh_path_on_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("sys.platform", "win32")
        path = PowerShellStrategy(is_pwsh=True).rc_path()
        assert path == (tmp_path / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1")

    def test_powershell_5_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        path = PowerShellStrategy(is_pwsh=False).rc_path()
        assert path == (
            tmp_path / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1"
        )

    def test_install_uses_marker_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("sys.platform", "darwin")
        result = PowerShellStrategy(is_pwsh=True).install()
        assert result.action == "installed"
        text = result.path.read_text()
        assert MARKER_OPEN in text
        assert "_APMODE_COMPLETE" in text


# ---------------------------------------------------------------------------
# get_strategy registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("bash", BashStrategy),
        ("zsh", ZshStrategy),
        ("fish", FishStrategy),
    ],
)
def test_get_strategy_returns_expected_class(name: str, cls: type) -> None:
    assert isinstance(get_strategy(name), cls)


@pytest.mark.parametrize("name", ["powershell", "pwsh", "POWERSHELL", "Pwsh"])
def test_get_strategy_returns_powershell(name: str) -> None:
    s = get_strategy(name)
    assert isinstance(s, PowerShellStrategy)
    assert s.is_pwsh == (name.lower() == "pwsh")


def test_get_strategy_rejects_unknown() -> None:
    with pytest.raises(UnsupportedShellError) as excinfo:
        get_strategy("tcsh")
    assert excinfo.value.shell_name == "tcsh"


# ---------------------------------------------------------------------------
# CLI front-end
# ---------------------------------------------------------------------------


class TestCliCompletion:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("ZDOTDIR", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("ZSH_CUSTOM", raising=False)
        monkeypatch.setenv("SHELL", "/bin/bash")
        # Make sure we don't trigger the oh-my-zsh path.
        omz = tmp_path / ".oh-my-zsh"
        if omz.exists():
            import shutil as _sh

            _sh.rmtree(omz)

    def test_install_uses_explicit_shell_flag(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "bash"])
        assert result.exit_code == 0, result.output
        assert "installed for bash" in result.output
        assert (tmp_path / ".bashrc").exists()

    def test_install_with_unsupported_shell_flag_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "tcsh"])
        # Click reports nonzero (typed ConfigError → exit 14 via the new
        # entrypoint when invoked through `python -m apmode`; via
        # CliRunner the legacy path raises typer.Exit so the code is 1).
        assert result.exit_code != 0

    def test_install_json_envelope(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "bash", "--json"])
        assert result.exit_code == 0, result.output
        # Pull just the JSON line; some Rich init noise might prefix it
        # in CI, but in this test the only stdout is the envelope.
        envelope = json.loads(result.output.strip().splitlines()[-1])
        assert envelope == {
            "ok": True,
            "shell": "bash",
            "path": str(tmp_path / ".bashrc"),
            "action": "installed",
            "marker": MARKER_OPEN,
        }

    def test_install_idempotency_through_cli(self) -> None:
        first = runner.invoke(app, ["completion", "install", "--shell", "bash", "--json"])
        second = runner.invoke(app, ["completion", "install", "--shell", "bash", "--json"])
        assert first.exit_code == 0
        assert second.exit_code == 0
        first_envelope = json.loads(first.output.strip().splitlines()[-1])
        second_envelope = json.loads(second.output.strip().splitlines()[-1])
        assert first_envelope["action"] == "installed"
        assert second_envelope["action"] == "already_installed"

    def test_show_prints_completion_source(self) -> None:
        result = runner.invoke(app, ["completion", "show", "--shell", "bash"])
        assert result.exit_code == 0
        assert "_APMODE_COMPLETE=bash_source apmode" in result.output

    def test_show_json_includes_script(self) -> None:
        result = runner.invoke(app, ["completion", "show", "--shell", "fish", "--json"])
        assert result.exit_code == 0
        envelope = json.loads(result.output.strip().splitlines()[-1])
        assert envelope["ok"] is True
        assert envelope["shell"] == "fish"
        assert "_APMODE_COMPLETE=fish_source apmode | source" in envelope["script"]

    def test_uninstall_after_install_round_trip(self, tmp_path: Path) -> None:
        rc = tmp_path / ".bashrc"
        rc.write_text("# my custom rc\nexport FOO=bar\n")
        original = rc.read_text()
        runner.invoke(app, ["completion", "install", "--shell", "bash"])
        runner.invoke(app, ["completion", "uninstall", "--shell", "bash"])
        assert rc.read_text() == original

    def test_uninstall_when_absent_reports_absent(self) -> None:
        result = runner.invoke(app, ["completion", "uninstall", "--shell", "bash", "--json"])
        assert result.exit_code == 0
        envelope = json.loads(result.output.strip().splitlines()[-1])
        assert envelope["action"] == "absent"

    def test_install_falls_back_to_shell_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHELL", "/usr/local/bin/zsh")
        # No --shell, no shellingham invocation expected.
        result = runner.invoke(app, ["completion", "install"])
        assert result.exit_code == 0
        # zsh strategy points at .zshrc (vanilla path here)
        assert (tmp_path / ".zshrc").exists()


# ---------------------------------------------------------------------------
# Cross-shell smoke
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shell_name", ["bash", "zsh", "fish"])
def test_completion_script_contains_correct_env_var(shell_name: str) -> None:
    """Each shell produces a string containing ``_APMODE_COMPLETE``."""
    strategy = get_strategy(shell_name)
    assert "_APMODE_COMPLETE" in strategy.completion_script()


def test_fish_completion_uses_env_prefix_for_compat() -> None:
    """fish < 4.0 rejects ``VAR=value cmd``; the portable form is ``env VAR=value cmd``."""
    script = FishStrategy().completion_script()
    assert script.startswith("env ")


def test_atomic_write_uses_pid_suffix_to_avoid_concurrent_clobber(
    tmp_path: Path,
) -> None:
    """Two concurrent installs must not race on a shared ``<file>.tmp``.

    We can't easily exercise the race in-process, but we can pin the
    invariant that the temp filename includes the PID so two processes
    deterministically pick distinct names.
    """
    target = tmp_path / "sub" / "rcfile"
    atomic_write(target, "hello\n")
    # No stale .tmp.<pid> left behind.
    leftovers = [p for p in target.parent.iterdir() if p.name.startswith("rcfile.tmp.")]
    assert leftovers == []
    # The implementation embeds os.getpid() in the temp name.
    import inspect

    from apmode.shells._rcfile import atomic_write as fn

    assert "os.getpid()" in inspect.getsource(fn)


def test_backup_once_handles_file_disappearing_mid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOCTOU: file gone between exists() and copy2 → silent no-op."""
    path = tmp_path / "rcfile"
    path.write_text("first\n")
    # Patch shutil.copy2 to raise FileNotFoundError as if the file
    # vanished between the existence check and the copy.
    import shutil as _sh

    def _raise_fnf(*_a: object, **_kw: object) -> None:
        raise FileNotFoundError("vanished")

    monkeypatch.setattr(_sh, "copy2", _raise_fnf)
    assert backup_once(path) is None  # no crash
    # And we still record the file as "tried" so we don't try again.
    assert backup_once(path) is None


@pytest.mark.skipif(os.name != "nt", reason="powershell paths are Windows-specific")
def test_powershell_install_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # pragma: no cover - exercised on Windows only
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    PowerShellStrategy(is_pwsh=True).install()
    assert (tmp_path / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1").exists()
