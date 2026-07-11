# SPDX-License-Identifier: GPL-2.0-or-later
"""Reusable bundle fixtures for the RO-Crate projection tests.

The bundle builder itself now lives in :mod:`tests._helpers.bundles` so
the golden and integration tiers can import it without reaching into this
unit-test package; it is re-exported here for the RO-Crate unit tests
that already import ``_fixtures.build_submission_bundle``.

This module retains :func:`_digest_bundle` — the deliberately bespoke
sentinel-digest helper that mirrors
``apmode.bundle.emitter._DIGEST_EXCLUDED_RELATIVE_PATHS``. Keep it here
as the lockstep pin: any change to the emitter's digest-exclusion set
must update this helper (and the importer verifier) in lockstep.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests._helpers.bundles import build_submission_bundle

__all__ = ["build_submission_bundle"]


def _digest_bundle(run_dir: Path) -> str:
    digest = hashlib.sha256()
    # Mirror ``apmode.bundle.emitter._DIGEST_EXCLUDED_RELATIVE_PATHS``: sentinel
    # itself + post-seal sidecars (CycloneDX SBOM, SBC manifest, reviewer
    # attestation) that are explicitly excluded so regenerating them
    # never invalidates ``_COMPLETE``.
    excluded = {"_COMPLETE", "bom.cdx.json", "sbc_manifest.json", "attestation.json"}
    for p in sorted(run_dir.rglob("*"), key=lambda q: q.relative_to(run_dir).as_posix()):
        if not p.is_file() or p.name in excluded:
            continue
        digest.update(p.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(p.read_bytes())
    return digest.hexdigest()
