# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end RO-Crate export + roc-validator CI gate.

Implements the v0.6 acceptance criterion: 5 Submission-lane bundles are
exported and each must pass ``roc-validator`` at REQUIRED severity
against the ``provenance-run-crate-0.5`` profile. A failure here blocks
the merge.

Runs programmatic validator calls (no shell-out) so that the test is
self-contained and portable. If roc-validator becomes unavailable the
tests are skipped — the dev group pin guarantees it is installed on CI.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("rocrate_validator")

from rocrate_validator import models as _rcv_models
from rocrate_validator import services as _rcv_services

from apmode.bundle.rocrate import (
    RoCrateEmitter,
    RoCrateExportOptions,
)

# Reuse the unit-test bundle fixture so the integration test exercises
# the same Pydantic-compatible artifacts the rest of the suite uses.
from tests.unit.rocrate._fixtures import build_submission_bundle


def _validate_crate(crate_path: Path) -> tuple[bool, list[str]]:
    """Validate a directory-form crate at REQUIRED severity."""
    settings = _rcv_models.ValidationSettings(
        rocrate_uri=str(crate_path),
        profile_identifier="provenance-run-crate-0.5",
        requirement_severity=_rcv_models.Severity.REQUIRED,
    )
    result = _rcv_services.validate(settings)
    messages = [f"{issue.severity.name}: {issue.message}" for issue in result.get_issues()]
    return bool(result.passed()), messages


# Parametrised matrix covering the five acceptance scenarios from plan §H:
#   1. Minimal 1-candidate run
#   2. 2-candidate run with lineage edges
#   3. Bayesian-augmented run (prior/simulation/MCMC)
#   4. Agentic run with regulatory files (PCCP)
#   5. Full mixed-mode run (credibility + bayesian + agentic + regulatory)
_SCENARIOS = [
    pytest.param(
        {"candidate_ids": ("c001",)},
        id="minimal-submission",
    ),
    pytest.param(
        {"candidate_ids": ("c001", "c002")},
        id="two-candidates-with-lineage",
    ),
    pytest.param(
        {"candidate_ids": ("c001",), "add_credibility": True, "add_bayesian": True},
        id="credibility-and-bayesian",
    ),
    pytest.param(
        {
            "candidate_ids": ("c001", "c002"),
            "add_agentic": True,
            "add_regulatory": True,
        },
        id="agentic-and-pccp",
    ),
    pytest.param(
        {
            "candidate_ids": ("c001", "c002", "c003"),
            "add_credibility": True,
            "add_bayesian": True,
            "add_agentic": True,
            "add_regulatory": True,
        },
        id="full-mixed",
    ),
]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_rocrate_directory_form_validates(
    scenario: dict[str, object],
    tmp_path: Path,
) -> None:
    """Every Suite-A Submission bundle exports to a valid directory crate."""
    include_provagent = bool(scenario.get("add_agentic"))
    bundle = build_submission_bundle(tmp_path, **scenario)
    out = tmp_path / "crate_dir"
    emitter = RoCrateEmitter()
    result_path = emitter.export_from_sealed_bundle(
        bundle,
        out,
        RoCrateExportOptions(
            date_published="2026-04-17T10:00:00Z",
            include_provagent=include_provagent,
        ),
    )

    assert result_path == out
    assert (out / "ro-crate-metadata.json").is_file()

    ok, messages = _validate_crate(out)
    assert ok, "REQUIRED-level issues:\n  " + "\n  ".join(messages)


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_rocrate_zip_form_validates(
    scenario: dict[str, object],
    tmp_path: Path,
) -> None:
    """Every scenario exports to a valid ZIP that survives extraction."""
    include_provagent = bool(scenario.get("add_agentic"))
    bundle = build_submission_bundle(tmp_path, **scenario)
    out_zip = tmp_path / "crate.zip"
    emitter = RoCrateEmitter()
    result_path = emitter.export_from_sealed_bundle(
        bundle,
        out_zip,
        RoCrateExportOptions(
            date_published="2026-04-17T10:00:00Z",
            include_provagent=include_provagent,
        ),
    )

    assert result_path == out_zip
    assert out_zip.is_file()

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(out_zip) as zf:
        zf.extractall(extracted)
    assert (extracted / "ro-crate-metadata.json").is_file()

    ok, messages = _validate_crate(extracted)
    assert ok, "REQUIRED-level issues:\n  " + "\n  ".join(messages)


def test_bundle_not_modified_after_export(tmp_path: Path) -> None:
    """Source bundle directory must be byte-identical after export."""
    bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))

    def _snapshot(b: Path) -> dict[str, bytes]:
        return {str(p.relative_to(b)): p.read_bytes() for p in sorted(b.rglob("*")) if p.is_file()}

    before = _snapshot(bundle)
    emitter = RoCrateEmitter()
    emitter.export_from_sealed_bundle(
        bundle,
        tmp_path / "crate",
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    emitter.export_from_sealed_bundle(
        bundle,
        tmp_path / "crate.zip",
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    after = _snapshot(bundle)
    assert before == after


def test_complete_sentinel_is_a_file_entity_with_additional_type(
    tmp_path: Path,
) -> None:
    """_COMPLETE is projected as File with apmode:completeSentinel additionalType."""
    import json

    bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
    out = tmp_path / "crate"
    RoCrateEmitter().export_from_sealed_bundle(
        bundle,
        out,
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    meta = json.loads((out / "ro-crate-metadata.json").read_text())

    sentinel = next((e for e in meta["@graph"] if e["@id"] == "_COMPLETE"), None)
    assert sentinel is not None
    assert sentinel["@type"] == "File"
    assert sentinel["additionalType"] == "apmode:completeSentinel"
    # Root Dataset.hasPart must include _COMPLETE
    root = next(e for e in meta["@graph"] if e["@id"] == "./")
    ids = {ref["@id"] for ref in root["hasPart"] if isinstance(ref, dict)}
    assert "_COMPLETE" in ids


def test_refuses_unsealed_bundle(tmp_path: Path) -> None:
    """Export must refuse a bundle missing the _COMPLETE sentinel."""
    from apmode.bundle.rocrate.projector import BundleNotSealedError

    bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
    (bundle / "_COMPLETE").unlink()

    with pytest.raises(BundleNotSealedError):
        RoCrateEmitter().export_from_sealed_bundle(
            bundle,
            tmp_path / "crate",
            RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
        )


def test_cli_export_command(tmp_path: Path) -> None:
    """Smoke test for ``apmode bundle rocrate export``."""
    from typer.testing import CliRunner

    from apmode.cli import app

    bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
    out = tmp_path / "crate_cli"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bundle",
            "rocrate",
            "export",
            str(bundle),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "ro-crate-metadata.json").is_file()


def test_cli_validate_with_rocrate(tmp_path: Path) -> None:
    """``apmode validate --rocrate`` runs roc-validator on the crate."""
    from typer.testing import CliRunner

    from apmode.cli import app

    bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
    crate = tmp_path / "crate_cli"
    RoCrateEmitter().export_from_sealed_bundle(
        bundle,
        crate,
        RoCrateExportOptions(date_published="2026-04-17T10:00:00Z"),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["validate", str(bundle), "--rocrate", "--crate", str(crate), "--json"],
    )
    # ``validate`` returns 0 when every check passes; the crate must
    # validate and the bundle must pass its own checks.
    import json as _json

    payload = _json.loads(result.stdout)
    assert payload["ok"] is True, payload
    assert payload["rocrate"]["ok"] is True, payload["rocrate"]
