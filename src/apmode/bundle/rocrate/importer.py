# SPDX-License-Identifier: GPL-2.0-or-later
"""Import a Workflow Run RO-Crate back to an APMODE bundle directory.

The exporter in :mod:`apmode.bundle.rocrate.projector` writes every
bundle file unchanged into the crate alongside the emitted
``ro-crate-metadata.json``. Import is therefore a mechanical unpack
followed by integrity verification:

1. If the source is a ZIP, extract it to the target.
2. If the source is a directory, copy its bundle-owned files to the
   target (excluding the RO-Crate-owned artifacts ``ro-crate-metadata.json``
   and the virtual ``workflows/`` definition).
3. Verify the ``_COMPLETE`` sentinel digest matches the extracted
   bundle — this confirms the bundle is identical to the one that was
   sealed pre-export.

The import helper does not mutate the crate; it only reads. The
target directory must be empty or non-existent so that imports never
silently overwrite user data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

# The name is repeated here (rather than imported from
# ``apmode.bundle.emitter``) so the importer stays light-weight and
# does not pull in Lark + the DSL stack.
_COMPLETE_SENTINEL = "_COMPLETE"

_ROCRATE_OWNED_FILES: frozenset[str] = frozenset(
    {
        "ro-crate-metadata.json",
    }
)
_ROCRATE_OWNED_DIRS: frozenset[str] = frozenset(
    {
        "workflows",
    }
)


class RoCrateImportError(RuntimeError):
    """Raised when crate import fails (missing sentinel, digest mismatch, ...)."""


def import_crate(source: Path, target: Path) -> Path:
    """Import an RO-Crate at ``source`` into a bundle directory at ``target``.

    Args:
        source: Directory-form crate or ``.zip`` file.
        target: Destination bundle directory. Must not exist or must be
            empty so we never overwrite user files.

    Returns:
        The ``Path`` of the imported bundle directory (== ``target``).

    Raises:
        FileNotFoundError: if ``source`` does not exist.
        FileExistsError: if ``target`` is a non-empty directory.
        RoCrateImportError: if the crate is missing ``_COMPLETE`` or
            the digest does not match the extracted tree.
    """
    if not source.exists():
        msg = f"crate source not found: {source}"
        raise FileNotFoundError(msg)

    if target.exists():
        if not target.is_dir():
            msg = f"target must be a directory: {target}"
            raise FileExistsError(msg)
        if any(target.iterdir()):
            msg = f"target directory is not empty: {target}"
            raise FileExistsError(msg)
    else:
        target.mkdir(parents=True)

    if source.is_file() and source.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as td:
            staging = Path(td)
            _safe_extract_zip(source, staging)
            _copy_bundle_files(staging, target)
    elif source.is_dir():
        _copy_bundle_files(source, target)
    else:
        msg = f"unsupported crate source: {source}"
        raise RoCrateImportError(msg)

    _verify_sentinel(target)
    return target


def _safe_extract_zip(source: Path, staging: Path) -> None:
    """Extract ``source`` into ``staging`` with per-entry path validation.

    Guards against ZIP-slip (CVE-class path traversal via entry names
    like ``../../etc/passwd``): each entry's resolved target must remain
    strictly inside ``staging``. Symlinks and hard links are rejected
    outright — an APMODE crate only contains regular files and
    directories, so a link is always evidence of a malicious archive.
    """
    staging_resolved = staging.resolve()
    with zipfile.ZipFile(source) as zf:
        for entry in zf.infolist():
            # Reject directory traversal attempts: absolute paths,
            # empty names, ``..`` segments, drive letters (Windows),
            # and archive entries that resolve outside ``staging``.
            name = entry.filename
            if not name or name.startswith(("/", "\\")) or ":" in name:
                msg = f"rejecting unsafe ZIP entry: {name!r}"
                raise RoCrateImportError(msg)

            # Reject symlinks + other special file types. The
            # Unix-mode field on ZipInfo encodes the entry kind; the
            # upper 16 bits carry the stat mode, so 0xA000 is a symlink.
            mode = (entry.external_attr >> 16) & 0xF000
            if mode in (0xA000, 0xC000, 0x6000):  # symlink, socket, block
                msg = f"rejecting non-regular ZIP entry: {name!r}"
                raise RoCrateImportError(msg)

            target = (staging / name).resolve()
            try:
                target.relative_to(staging_resolved)
            except ValueError as exc:
                msg = f"ZIP entry escapes staging: {name!r}"
                raise RoCrateImportError(msg) from exc

        zf.extractall(staging)


def _copy_bundle_files(crate_root: Path, dest: Path) -> None:
    """Copy bundle-owned files from the crate to the destination.

    RO-Crate-owned files (``ro-crate-metadata.json`` and the virtual
    ``workflows/`` definition created by the exporter) are excluded so
    the resulting directory is a valid APMODE bundle again.

    Symlinks inside the staging tree are also rejected — bundles only
    contain regular files, and a symlink is either a malicious ZIP
    entry or filesystem corruption.
    """
    crate_resolved = crate_root.resolve()
    dest_resolved = dest.resolve()
    for src in sorted(crate_root.rglob("*")):
        # ``rglob`` traverses symlinks; reject anything that is not a
        # regular file. ``is_file()`` follows symlinks, but
        # ``is_symlink()`` does not — combine them for a safe test.
        if src.is_symlink():
            msg = f"refusing to copy symlink inside crate: {src}"
            raise RoCrateImportError(msg)
        if not src.is_file():
            continue
        # Re-verify the source path is inside the crate root even after
        # symlink following — defence in depth against a symlink that
        # survived the first check.
        try:
            src.resolve().relative_to(crate_resolved)
        except ValueError as exc:
            msg = f"crate entry escapes crate root: {src}"
            raise RoCrateImportError(msg) from exc
        rel = src.relative_to(crate_root)
        rel_posix = rel.as_posix()
        if rel_posix in _ROCRATE_OWNED_FILES:
            continue
        top = rel.parts[0] if rel.parts else ""
        if top in _ROCRATE_OWNED_DIRS:
            continue
        dst = (dest / rel).resolve()
        try:
            dst.relative_to(dest_resolved)
        except ValueError as exc:
            msg = f"destination escapes target: {rel_posix}"
            raise RoCrateImportError(msg) from exc
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _verify_sentinel(bundle: Path) -> None:
    """Re-compute the SHA-256 digest and match it against ``_COMPLETE``.

    Mirrors :func:`apmode.bundle.emitter._compute_bundle_digest` but
    avoids the import to keep the importer dependency-free.
    """
    sentinel_path = bundle / _COMPLETE_SENTINEL
    if not sentinel_path.is_file():
        msg = f"imported bundle has no _COMPLETE sentinel: {bundle}"
        raise RoCrateImportError(msg)
    try:
        sentinel = json.loads(sentinel_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"_COMPLETE sentinel at {sentinel_path} is unreadable: {exc}"
        raise RoCrateImportError(msg) from exc

    expected = sentinel.get("sha256")
    if not isinstance(expected, str):
        msg = f"_COMPLETE sentinel at {sentinel_path} missing sha256 digest"
        raise RoCrateImportError(msg)

    digest = hashlib.sha256()
    for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
        # bom.cdx.json is a producer-side sidecar that may be generated
        # post-seal (via ``apmode bundle sbom``); it is excluded from the
        # digest just like the sentinel so its presence does not trip the
        # tamper check on import.
        if not p.is_file() or p.name in (_COMPLETE_SENTINEL, "bom.cdx.json"):
            continue
        digest.update(p.relative_to(bundle).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(p.read_bytes())
    observed = digest.hexdigest()
    if observed != expected:
        msg = (
            f"bundle digest mismatch after import: sentinel claims {expected}, observed {observed}"
        )
        raise RoCrateImportError(msg)
