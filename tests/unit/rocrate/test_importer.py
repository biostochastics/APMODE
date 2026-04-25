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


class TestWorkflowsSubtreePreserved:
    """B1 regression — user-owned ``workflows/*`` is not dropped on import.

    The exporter materialises a synthetic stub at ``workflows/<lane>-lane.apmode``
    to back ``mainEntity``, but any other ``workflows/*`` entry the source
    bundle happens to have (e.g. an orchestrator asset) is legitimate
    user content and must round-trip.
    """

    def test_user_workflows_file_survives_directory_import(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig"
        orig.mkdir()
        bundle = build_submission_bundle(orig)
        # Plant a user-owned file inside ``workflows/`` *before* sealing.
        # Re-seal the bundle so its sha256 matches what's on disk.
        (bundle / "workflows").mkdir(exist_ok=True)
        user_payload = b"user-owned-workflow-asset\n"
        (bundle / "workflows" / "user-notes.txt").write_bytes(user_payload)
        import hashlib
        import json

        digest = hashlib.sha256()
        for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
            if not p.is_file() or p.name in ("_COMPLETE", "bom.cdx.json", "sbc_manifest.json"):
                continue
            digest.update(p.relative_to(bundle).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(p.read_bytes())
        (bundle / "_COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "with-user-workflows",
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )

        out = tmp_path / "crate_dir"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        target = tmp_path / "imported"
        import_crate(out, target)

        preserved = target / "workflows" / "user-notes.txt"
        synthetic = target / "workflows" / "submission-lane.apmode"
        assert preserved.is_file(), "user-owned workflows/ file was silently dropped"
        assert preserved.read_bytes() == user_payload
        assert not synthetic.exists(), "synthetic workflow stub should not be imported"

    def test_user_workflows_file_survives_zip_import(self, tmp_path: Path) -> None:
        orig = tmp_path / "orig"
        orig.mkdir()
        bundle = build_submission_bundle(orig)
        (bundle / "workflows").mkdir(exist_ok=True)
        user_payload = b'{"note": "zip form"}\n'
        (bundle / "workflows" / "user.json").write_bytes(user_payload)
        import hashlib
        import json

        digest = hashlib.sha256()
        for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
            if not p.is_file() or p.name in ("_COMPLETE", "bom.cdx.json", "sbc_manifest.json"):
                continue
            digest.update(p.relative_to(bundle).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(p.read_bytes())
        (bundle / "_COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "run_id": "zip-user-workflows",
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )

        out_zip = tmp_path / "crate.zip"
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            out_zip,
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )
        target = tmp_path / "imported_zip"
        import_crate(out_zip, target)

        preserved = target / "workflows" / "user.json"
        synthetic = target / "workflows" / "submission-lane.apmode"
        assert preserved.is_file()
        assert preserved.read_bytes() == user_payload
        assert not synthetic.exists()


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
