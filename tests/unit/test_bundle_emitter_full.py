# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the completed Bundle Emitter (all artifacts per §5)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    CandidateLineage,
    CandidateLineageEntry,
    EvidenceManifest,
    FailedCandidate,
    GateCheckResult,
    GateResult,
    InitialEstimateEntry,
    InitialEstimates,
    SearchTrajectoryEntry,
    SplitManifest,
    SubjectAssignment,
)


class TestBundleEmitterFull:
    """Tests for all bundle artifact writers."""

    def test_write_evidence_manifest(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        em = EvidenceManifest(
            data_sha256="a" * 64,
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_signature=False,
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=2,
            covariate_correlated=False,
            blq_burden=0.05,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        path = emitter.write_evidence_manifest(em)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["richness_category"] == "rich"
        assert data["data_sha256"] == "a" * 64

    def test_write_initial_estimates(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        ie = InitialEstimates(
            entries={
                "cand_001": InitialEstimateEntry(
                    candidate_id="cand_001",
                    source="nca",
                    estimates={"CL": 5.0, "V": 70.0, "ka": 1.5},
                    inputs_used=["per_subject_nca"],
                )
            }
        )
        path = emitter.write_initial_estimates(ie)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "cand_001" in data["entries"]

    def test_write_split_manifest(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        split = SplitManifest(
            split_seed=42,
            split_strategy="subject_level",
            assignments=[
                SubjectAssignment(subject_id="1", fold="train"),
                SubjectAssignment(subject_id="2", fold="test"),
            ],
        )
        path = emitter.write_split_manifest(split)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["split_seed"] == 42

    def test_write_gate_decision(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        gr = GateResult(
            gate_id="gate_001",
            gate_name="technical_validity",
            candidate_id="cand_001",
            passed=True,
            checks=[GateCheckResult(check_id="convergence", passed=True, observed=True)],
            summary_reason="All checks passed",
            policy_version="0.1.0",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        path = emitter.write_gate_decision(gr, gate_number=1)
        assert path.exists()
        assert "gate1_cand_001.json" in path.name
        data = json.loads(path.read_text())
        assert data["passed"] is True

    def test_append_search_trajectory(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        e1 = SearchTrajectoryEntry(
            candidate_id="a",
            backend="nlmixr2",
            converged=True,
            bic=100.0,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        e2 = SearchTrajectoryEntry(
            candidate_id="b",
            backend="nlmixr2",
            converged=False,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_search_trajectory(e1)
        emitter.append_search_trajectory(e2)
        path = emitter.run_dir / "search_trajectory.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["candidate_id"] == "a"
        assert json.loads(lines[1])["converged"] is False

    def test_append_failed_candidate(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        fc = FailedCandidate(
            candidate_id="bad_001",
            backend="nlmixr2",
            gate_failed="gate1",
            failed_checks=["convergence", "cwres_mean"],
            summary_reason="Failed: convergence, cwres_mean",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_failed_candidate(fc)
        path = emitter.run_dir / "failed_candidates.jsonl"
        data = json.loads(path.read_text().strip())
        assert data["gate_failed"] == "gate1"
        assert len(data["failed_checks"]) == 2

    def test_write_candidate_lineage(self, tmp_path: Path) -> None:
        emitter = BundleEmitter(tmp_path, run_id="test")
        emitter.initialize()
        lineage = CandidateLineage(
            entries=[
                CandidateLineageEntry(candidate_id="root_1"),
                CandidateLineageEntry(
                    candidate_id="child_1", parent_id="root_1", transform="add_cov_WT"
                ),
            ]
        )
        path = emitter.write_candidate_lineage(lineage)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["entries"]) == 2
        assert data["entries"][1]["parent_id"] == "root_1"
