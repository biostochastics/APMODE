# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for ``apmode attest`` (QA/QC remediation: human-in-the-loop
reviewer attestation / override-audit gate before export).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    ColumnMapping,
    DataManifest,
    GateCheckResult,
    GateResult,
    SeedRegistry,
)
from apmode.cli import app

runner = CliRunner()


def _gate_result(gate_id: str, gate_name: str, check_id: str, *, passed: bool) -> GateResult:
    """Minimal sealed gate decision an override in these tests can reference.

    The attest CLI validates each ``--gate-override`` against the sealed gate
    files: the override's ``gate_id``/``check_id`` must exist and its
    ``original_passed`` must equal the sealed ``check.passed``.
    """
    return GateResult(
        gate_id=gate_id,
        gate_name=gate_name,
        candidate_id="cand_attest",
        passed=passed,
        checks=[
            GateCheckResult(check_id=check_id, passed=passed, observed=passed, threshold=None)
        ],
        summary_reason="ok" if passed else f"Failed: {check_id}",
        policy_version="0.7.1",
        timestamp="2026-07-11T00:00:00+00:00",
    )


def _seal_bundle(tmp_path: Path, run_id: str) -> Path:
    emitter = BundleEmitter(tmp_path, run_id=run_id)
    emitter.initialize()
    emitter.write_data_manifest(
        DataManifest(
            data_sha256="a" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
            ),
            n_subjects=20,
            n_observations=200,
            n_doses=40,
        )
    )
    emitter.write_seed_registry(
        SeedRegistry(root_seed=42, r_seed=42, r_rng_kind="L'Ecuyer-CMRG", np_seed=42)
    )
    # Gate decisions the override tests reference: gate1/cwres_mean_max (passed)
    # and gate2/shrinkage_max (failed — the override lifts it). The attest CLI
    # validates each override against these sealed files.
    emitter.write_gate_decision(
        _gate_result("gate1", "technical_validity", "cwres_mean_max", passed=True), 1
    )
    emitter.write_gate_decision(
        _gate_result("gate2", "lane_admissibility", "shrinkage_max", passed=False), 2
    )
    return emitter.seal()


class TestAttestCLIUnsealed:
    def test_unsealed_bundle_exits_1(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="run_unsealed")
        emitter.initialize()
        result = runner.invoke(
            app,
            [
                "attest",
                str(emitter.run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "Looks good.",
            ],
        )
        assert result.exit_code == 1
        assert "not sealed" in result.output.lower()

    def test_missing_bundle_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "attest",
                str(tmp_path / "does_not_exist"),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "Looks good.",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestAttestCLISealed:
    def test_writes_attestation_and_exits_0(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "Reviewed gate decisions; no concerns.",
            ],
        )
        assert result.exit_code == 0, result.output
        att_path = run_dir / "attestation.json"
        assert att_path.exists()
        data = json.loads(att_path.read_text())
        assert data["reviewer_id"] == "jdoe"
        assert data["decision"] == "approved"
        assert data["gate_overrides"] == []

    def test_json_output(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_json")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "rejected",
                "--rationale",
                "CWRES trend unresolved.",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["ok"] is True
        assert payload["path"] == str(run_dir / "attestation.json")

    def test_gate_override_parsed(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_override")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved_with_conditions",
                "--rationale",
                "Shrinkage borderline but justified by sparse design.",
                "--gate-override",
                "gate2:shrinkage_max:false:Sparse design; expected shrinkage:senior_reviewer",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads((run_dir / "attestation.json").read_text())
        assert len(data["gate_overrides"]) == 1
        override = data["gate_overrides"][0]
        assert override["gate_id"] == "gate2"
        assert override["check_id"] == "shrinkage_max"
        assert override["original_passed"] is False
        assert override["authorized_by"] == "senior_reviewer"

    def test_gate_override_authorized_by_defaults_to_reviewer(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_override_default")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "ok",
                "--gate-override",
                "gate1:cwres_mean_max:true:Recorded for audit trail",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads((run_dir / "attestation.json").read_text())
        assert data["gate_overrides"][0]["authorized_by"] == "jdoe"

    def test_gate_override_justification_may_contain_colons(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_override_colons")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "ok",
                "--gate-override",
                "gate1:cwres_mean_max:true:Reason: sparse design with no author override",
            ],
        )
        assert result.exit_code == 0, result.output
        data = json.loads((run_dir / "attestation.json").read_text())
        override = data["gate_overrides"][0]
        assert override["override_justification"] == (
            "Reason: sparse design with no author override"
        )
        assert override["authorized_by"] == "jdoe"

    def test_malformed_gate_override_exits_1(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_bad_override")
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "ok",
                "--gate-override",
                "not-enough-fields",
            ],
        )
        assert result.exit_code == 1

    def test_double_attest_without_force_exits_1(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_double")
        args = [
            "attest",
            str(run_dir),
            "--reviewer-id",
            "jdoe",
            "--reviewer-role",
            "PK reviewer",
            "--decision",
            "approved",
            "--rationale",
            "ok",
        ]
        first = runner.invoke(app, args)
        assert first.exit_code == 0
        second = runner.invoke(app, args)
        assert second.exit_code == 1

    def test_double_attest_with_force_succeeds(self, tmp_path: Path) -> None:
        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_force")
        base_args = [
            "attest",
            str(run_dir),
            "--reviewer-id",
            "jdoe",
            "--reviewer-role",
            "PK reviewer",
            "--decision",
            "approved",
            "--rationale",
            "ok",
        ]
        first = runner.invoke(app, base_args)
        assert first.exit_code == 0
        second = runner.invoke(app, [*base_args, "--decision", "rejected", "--force"])
        assert second.exit_code == 0
        data = json.loads((run_dir / "attestation.json").read_text())
        assert data["decision"] == "rejected"

    def test_attestation_does_not_change_bundle_digest(self, tmp_path: Path) -> None:
        from apmode.bundle.emitter import _compute_bundle_digest

        run_dir = _seal_bundle(tmp_path, "run_sealed_attest_digest")
        digest_before = _compute_bundle_digest(run_dir)
        result = runner.invoke(
            app,
            [
                "attest",
                str(run_dir),
                "--reviewer-id",
                "jdoe",
                "--reviewer-role",
                "PK reviewer",
                "--decision",
                "approved",
                "--rationale",
                "ok",
            ],
        )
        assert result.exit_code == 0, result.output
        digest_after = _compute_bundle_digest(run_dir)
        assert digest_before == digest_after
