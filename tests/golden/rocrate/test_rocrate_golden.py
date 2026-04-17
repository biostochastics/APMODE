# SPDX-License-Identifier: GPL-2.0-or-later
"""Golden snapshot for the canonical Submission-lane RO-Crate projection.

Renders ``ro-crate-metadata.json`` for a fixed Submission-lane bundle
and asserts it byte-matches a syrupy snapshot. Intentional changes are
reviewed by running ``uv run pytest tests/ --snapshot-update`` and
inspecting the diff.

The bundle used here is the minimal reproducible fixture from
``tests/unit/rocrate/_fixtures.py``. Keep it boring — snapshot
reviewability is proportional to how little the fixture changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from apmode.bundle.rocrate import RoCrateEmitter, RoCrateExportOptions
from tests.unit.rocrate._fixtures import build_submission_bundle


def test_canonical_submission_metadata(tmp_path: Path, snapshot: object) -> None:
    bundle = build_submission_bundle(
        tmp_path,
        run_id="golden-submission-run",
        candidate_ids=("c001",),
    )
    out = tmp_path / "crate"
    emitter = RoCrateEmitter()
    emitter.export_from_sealed_bundle(
        bundle,
        out,
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    metadata = json.loads((out / "ro-crate-metadata.json").read_text())
    # The SHA-256 hashes of File entities depend on the bundle fixture,
    # which is deterministic (syrupy snapshot covers them). A projection
    # change that moves a hash flags the fixture drift for review.
    assert metadata == snapshot  # type: ignore[comparison-overlap]
