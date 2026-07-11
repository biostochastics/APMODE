# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the SBC manifest schema + emitter (plan Task 26).

The Talts 2018 Simulation-Based Calibration roll-up is the artefact the
nightly runner (Task 27) populates with rank histograms. The producer-
side emitter writes a stub ``priors=[]`` manifest on every Bayesian run
— the artefact's *presence* signals the Bayesian path ran end-to-end,
not its content.

Tests cover:

* ``SBCPriorEntry`` schema invariants (rank histogram length matches the
  declared bin count; counts are non-negative; ``ks_pvalue`` ∈ [0, 1]).
* ``SBCManifest`` round-trips through ``model_dump_json``.
* ``BundleEmitter.write_sbc_manifest`` writes to
  ``artifacts/sbc/sbc_manifest.json``.
* The SBC manifest is excluded from the sealed-bundle digest so a
  nightly rewrite does not invalidate ``_COMPLETE`` — the file shows up
  in the bundle directory but does not appear in the digest input set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.emitter import BundleEmitter, _compute_bundle_digest
from apmode.bundle.models import SBCManifest, SBCPriorEntry


def _stub_manifest() -> SBCManifest:
    return SBCManifest(
        run_id="run_sbc_001",
        sbc_runner_commit="abc1234",
        priors=[],
        generated_at="2026-04-24T00:00:00+00:00",
    )


def _populated_entry() -> SBCPriorEntry:
    return SBCPriorEntry(
        target="CL",
        family="LogNormal",
        n_simulations=200,
        rank_histogram_bins=10,
        rank_histogram_counts=[20, 19, 21, 18, 22, 20, 19, 21, 20, 20],
        ks_pvalue=0.42,
        passed=True,
    )


# --- Schema invariants ---------------------------------------------------


def test_rank_histogram_length_must_match_bins() -> None:
    with pytest.raises(ValueError, match=r"does not match"):
        SBCPriorEntry(
            target="CL",
            family="LogNormal",
            n_simulations=200,
            rank_histogram_bins=10,
            rank_histogram_counts=[1, 2, 3],  # length 3, not 10
            passed=False,
        )


def test_rank_histogram_counts_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match=r"non-negative"):
        SBCPriorEntry(
            target="CL",
            family="LogNormal",
            n_simulations=200,
            rank_histogram_bins=4,
            rank_histogram_counts=[10, -1, 5, 8],
            passed=False,
        )


def test_ks_pvalue_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"less than or equal to 1"):
        SBCPriorEntry(
            target="CL",
            family="LogNormal",
            n_simulations=200,
            rank_histogram_bins=4,
            rank_histogram_counts=[10, 20, 15, 25],
            ks_pvalue=1.5,
            passed=False,
        )


def test_n_simulations_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"greater than or equal to 1"):
        SBCPriorEntry(
            target="CL",
            family="LogNormal",
            n_simulations=0,
            rank_histogram_bins=4,
            rank_histogram_counts=[10, 20, 15, 25],
            passed=False,
        )


def test_populated_entry_round_trips() -> None:
    entry = _populated_entry()
    blob = entry.model_dump_json()
    restored = SBCPriorEntry.model_validate_json(blob)
    assert restored == entry


def test_manifest_with_no_priors_is_valid_stub() -> None:
    """The orchestrator emits this on every Bayesian run."""
    manifest = _stub_manifest()
    assert manifest.priors == []
    assert manifest.schema_version == "1.0"


# --- Emitter -------------------------------------------------------------


def test_emitter_writes_to_artifacts_sbc(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_sbc")
    em.initialize()
    path = em.write_sbc_manifest(_stub_manifest())
    assert path.relative_to(em.run_dir).as_posix() == "artifacts/sbc/sbc_manifest.json"
    body = json.loads(path.read_text())
    assert body["run_id"] == "run_sbc_001"
    assert body["priors"] == []


def test_sbc_manifest_excluded_from_sealed_digest(tmp_path: Path) -> None:
    """A nightly rewrite of the SBC manifest must not invalidate _COMPLETE."""
    em = BundleEmitter(base_dir=tmp_path, run_id="run_sbc_digest")
    em.initialize()
    # Write something to the bundle so the digest is non-trivial
    (em.run_dir / "data_manifest.json").write_text('{"version": 1}')
    em.write_sbc_manifest(_stub_manifest())
    digest_before = _compute_bundle_digest(em.run_dir)
    # Overwrite the SBC manifest with a populated payload — simulating
    # a nightly runner update.
    populated = SBCManifest(
        run_id="run_sbc_001",
        sbc_runner_commit="def5678",
        priors=[_populated_entry()],
        generated_at="2026-04-25T03:00:00+00:00",
    )
    em.write_sbc_manifest(populated)
    digest_after = _compute_bundle_digest(em.run_dir)
    assert digest_before == digest_after


def test_emitter_handles_populated_manifest(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_sbc_populated")
    em.initialize()
    populated = SBCManifest(
        run_id="run_sbc_populated",
        sbc_runner_commit="abcdef0",
        priors=[_populated_entry()],
        generated_at="2026-04-24T12:00:00+00:00",
    )
    path = em.write_sbc_manifest(populated)
    body = json.loads(path.read_text())
    assert len(body["priors"]) == 1
    assert body["priors"][0]["target"] == "CL"
    assert body["priors"][0]["passed"] is True
