# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the RO-Crate importer (round-trip path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.rocrate import RoCrateEmitter, RoCrateExportOptions
from apmode.bundle.rocrate.importer import RoCrateImportError, import_crate

from ._fixtures import build_submission_bundle


def _export_crate(
    tmp_path: Path,
    *,
    form: str,
    scenario: dict[str, object] | None = None,
) -> Path:
    orig = tmp_path / "orig"
    orig.mkdir()
    bundle = build_submission_bundle(orig, **(scenario or {}))
    out = tmp_path / ("crate.zip" if form == "zip" else "crate_dir")
    RoCrateEmitter().export_from_sealed_bundle(
        bundle,
        out,
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    return out


class TestImportCrateDirectory:
    def test_round_trip_produces_sealed_bundle(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="dir")
        target = tmp_path / "imported"

        result = import_crate(crate, target)

        assert result == target
        assert (target / "_COMPLETE").is_file()

    def test_excludes_rocrate_metadata_file(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="dir")
        target = tmp_path / "imported"

        import_crate(crate, target)

        assert not (target / "ro-crate-metadata.json").exists()
        assert not (target / "workflows").exists()

    def test_digest_verified(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="dir")
        target = tmp_path / "imported"

        import_crate(crate, target)

        sentinel = json.loads((target / "_COMPLETE").read_text())
        assert sentinel.get("sha256")


class TestImportCrateZip:
    def test_round_trip_from_zip(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="zip")
        target = tmp_path / "imported_from_zip"

        import_crate(crate, target)

        assert (target / "data_manifest.json").is_file()
        assert (target / "_COMPLETE").is_file()


class TestSafeguards:
    def test_refuses_non_empty_target(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="dir")
        target = tmp_path / "nonempty"
        target.mkdir()
        (target / "file.txt").write_text("keep me")

        with pytest.raises(FileExistsError):
            import_crate(crate, target)

    def test_raises_on_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_crate(tmp_path / "does-not-exist", tmp_path / "target")

    def test_detects_tampered_bundle(self, tmp_path: Path) -> None:
        crate = _export_crate(tmp_path, form="dir")
        # Tamper with a file inside the crate before import
        tampered = crate / "data_manifest.json"
        tampered.write_text(json.dumps({"tampered": True}))

        target = tmp_path / "imported_tampered"
        with pytest.raises(RoCrateImportError):
            import_crate(crate, target)
