# SPDX-License-Identifier: GPL-2.0-or-later
"""Security-focused tests for :mod:`apmode.bundle.rocrate.importer`.

Covers ZIP-slip (CVE-class path traversal) and symlink-in-archive
attacks. Every test builds a deliberately malicious archive or
directory and asserts that :func:`import_crate` refuses to write
outside the target.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from apmode.bundle.rocrate.importer import RoCrateImportError, import_crate


def _write_malicious_zip_with_traversal(path: Path, evil_name: str) -> None:
    """Write a ZIP whose entry name escapes the extraction root."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(evil_name, b"pwned")
        zf.writestr("ro-crate-metadata.json", b"{}")
        zf.writestr("_COMPLETE", b'{"sha256":"x"}')


class TestZipSlipProtection:
    def test_rejects_parent_dir_traversal(self, tmp_path: Path) -> None:
        evil = tmp_path / "evil.zip"
        _write_malicious_zip_with_traversal(evil, "../../pwned.txt")
        target = tmp_path / "imported"

        with pytest.raises(RoCrateImportError, match="escapes staging"):
            import_crate(evil, target)

    def test_rejects_absolute_path_entry(self, tmp_path: Path) -> None:
        evil = tmp_path / "evil.zip"
        _write_malicious_zip_with_traversal(evil, "/etc/passwd")
        target = tmp_path / "imported"

        with pytest.raises(RoCrateImportError, match="unsafe ZIP entry"):
            import_crate(evil, target)

    def test_rejects_windows_drive_prefix(self, tmp_path: Path) -> None:
        evil = tmp_path / "evil.zip"
        _write_malicious_zip_with_traversal(evil, "C:/pwned.txt")
        target = tmp_path / "imported"

        with pytest.raises(RoCrateImportError, match="unsafe ZIP entry"):
            import_crate(evil, target)

    def test_rejects_symlink_entry_in_zip(self, tmp_path: Path) -> None:
        """A ZIP entry with symlink mode bits must be rejected.

        The zipfile library does not expose symlink creation directly,
        so we craft one via ``ZipInfo.external_attr`` — the upper 16
        bits of ``external_attr`` carry the Unix mode, and 0o120000
        (0xA000) is the symlink file type.
        """
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w", zipfile.ZIP_DEFLATED) as zf:
            info = zipfile.ZipInfo("evil-link")
            # 0o120777 <<16 = symlink + rwxrwxrwx
            info.external_attr = (0o120777 & 0xFFFF) << 16
            zf.writestr(info, b"/etc/passwd")
            zf.writestr("_COMPLETE", b'{"sha256":"x"}')
        target = tmp_path / "imported"

        with pytest.raises(RoCrateImportError, match="non-regular ZIP entry"):
            import_crate(evil, target)


class TestSymlinkInDirectoryImport:
    def test_rejects_symlink_inside_source_dir(self, tmp_path: Path) -> None:
        """Directory-form import must refuse a symlink inside the crate."""
        crate = tmp_path / "crate"
        crate.mkdir()
        (crate / "_COMPLETE").write_text('{"sha256": "doesnt-matter"}')
        (crate / "data_manifest.json").write_text('{"ok": true}')
        # Plant a symlink pointing out of the crate
        (crate / "escape").symlink_to("/etc/passwd")

        target = tmp_path / "imported"
        with pytest.raises(RoCrateImportError, match="symlink"):
            import_crate(crate, target)
