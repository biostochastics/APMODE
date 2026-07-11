# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI-layer tests for the bundle-reading commands ``inspect``, ``log``,
``diff``, and their machine-readable ``--json`` envelopes.

Split out of the former monolithic ``test_cli.py``. Deep-inspection commands
(trace, lineage, graph) live in ``test_cli_trace_lineage_graph.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from apmode.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Bundle fixture helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def _make_minimal_bundle(tmp_path: Path, name: str = "run_min") -> Path:
    """Bundle with only the required files — enough for ``validate`` to pass."""
    from apmode.bundle.emitter import _compute_bundle_digest

    bundle = tmp_path / name
    bundle.mkdir()
    _write_json(bundle / "data_manifest.json", {"n_subjects": 10, "n_observations": 100})
    _write_json(bundle / "seed_registry.json", {"root_seed": 753849})
    _write_json(bundle / "backend_versions.json", {"nlmixr2": "3.0.0"})
    # ``evidence_manifest`` and ``candidate_lineage`` are required artifacts.
    _write_json(
        bundle / "evidence_manifest.json",
        {
            "richness_category": "rich",
            "route_certainty": "oral",
            "nonlinear_clearance_evidence_strength": "none",
        },
    )
    _write_json(bundle / "candidate_lineage.json", {"entries": []})
    # Write the ``_COMPLETE`` sentinel with a matching digest last.
    _write_json(
        bundle / "_COMPLETE",
        {
            "schema_version": 1,
            "run_id": name,
            "sha256": _compute_bundle_digest(bundle),
        },
    )
    return bundle


def _make_full_bundle(tmp_path: Path, name: str = "run_full") -> Path:
    """Bundle with optional files, search trajectory, gates, ranking — exercises
    ``inspect``, ``diff``, ``log`` default/failed/gate/top paths."""
    bundle = _make_minimal_bundle(tmp_path, name=name)

    _write_json(
        bundle / "evidence_manifest.json",
        {
            "richness_category": "rich",
            "route_certainty": "oral",
            "nonlinear_clearance_evidence_strength": "none",
        },
    )
    _write_json(bundle / "initial_estimates.json", {"CL": 3.0, "V": 30.0})
    _write_json(bundle / "split_manifest.json", {"train": 8, "test": 2})
    _write_json(bundle / "policy_file.json", {"gate1": {"min_subjects": 5}})
    _write_json(
        bundle / "candidate_lineage.json",
        {
            "nodes": [
                {"candidate_id": "cand_001", "backend": "nlmixr2", "converged": True},
                {"candidate_id": "cand_002", "backend": "nlmixr2", "converged": True},
            ],
            "edges": [
                {
                    "parent_id": "cand_001",
                    "child_id": "cand_002",
                    "transform": "add_covariate(CL, WT)",
                },
            ],
        },
    )

    _write_jsonl(
        bundle / "search_trajectory.jsonl",
        [
            {"candidate_id": "cand_001", "converged": True, "bic": 400.0, "n_params": 4},
            {"candidate_id": "cand_002", "converged": True, "bic": 380.0, "n_params": 5},
            {"candidate_id": "cand_003", "converged": False, "bic": None, "n_params": 5},
        ],
    )
    _write_jsonl(
        bundle / "failed_candidates.jsonl",
        [
            {"model_id": "cand_003", "failed_gate": "gate1", "reason": "non-convergence"},
        ],
    )

    # Gate decisions
    gd = bundle / "gate_decisions"
    gd.mkdir()
    _write_json(gd / "gate1_cand_001.json", {"candidate_id": "cand_001", "passed": True})
    _write_json(gd / "gate1_cand_002.json", {"candidate_id": "cand_002", "passed": True})
    _write_json(gd / "gate1_cand_003.json", {"candidate_id": "cand_003", "passed": False})
    _write_json(gd / "gate2_cand_001.json", {"candidate_id": "cand_001", "passed": True})
    _write_json(gd / "gate2_cand_002.json", {"candidate_id": "cand_002", "passed": False})

    _write_json(
        bundle / "ranking.json",
        {
            "ranked_candidates": [
                {"model_id": "cand_002", "bic": 380.0, "score": 0.9},
                {"model_id": "cand_001", "bic": 400.0, "score": 0.8},
            ],
        },
    )

    # Compiled specs (used by `lineage` helper, also exercised via `log --top`)
    specs = bundle / "compiled_specs"
    specs.mkdir()
    _write_json(
        specs / "cand_001.json",
        {
            "absorption": {"model": "FirstOrder"},
            "distribution": {"model": "OneCompartment"},
            "elimination": {"model": "Linear"},
            "parameters": {"CL": 3.0, "V": 30.0},
        },
    )
    _write_json(
        specs / "cand_002.json",
        {
            "absorption": {"model": "FirstOrder"},
            "distribution": {"model": "OneCompartment"},
            "elimination": {"model": "Linear"},
            "parameters": {"CL": 3.0, "V": 30.0, "WT_ON_CL": 0.75},
        },
    )

    results = bundle / "results"
    results.mkdir()
    _write_json(
        results / "cand_001.json",
        {"parameters": {"CL": 3.0, "V": 30.0}, "bic": 400.0},
    )
    _write_json(
        results / "cand_002.json",
        {"parameters": {"CL": 3.0, "V": 30.0, "WT_ON_CL": 0.75}, "bic": 380.0},
    )
    return bundle


# ---------------------------------------------------------------------------
# `inspect` command
# ---------------------------------------------------------------------------


class TestInspect:
    def test_missing_directory(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["inspect", str(tmp_path / "nope")])
        assert result.exit_code == 1

    def test_minimal_bundle_renders(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0

    def test_full_bundle_renders_trajectory_and_gates(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        # Evidence / gate panels should be populated:
        assert "cand_00" in result.output or "Gate" in result.output


class TestInspectAttestation:
    """QA/QC remediation: ``apmode inspect`` surfaces attestation.json."""

    def _attestation_payload(self) -> dict[str, Any]:
        return {
            "attestation_schema_version": "1.0",
            "reviewer_id": "jdoe",
            "reviewer_role": "PK reviewer",
            "timestamp": "2026-07-10T00:00:00+00:00",
            "decision": "approved_with_conditions",
            "rationale": "Shrinkage borderline but justified by sparse design.",
            "gate_overrides": [
                {
                    "gate_id": "gate2",
                    "check_id": "shrinkage_max",
                    "original_passed": False,
                    "override_justification": "Sparse design; expected shrinkage.",
                    "authorized_by": "senior_reviewer",
                }
            ],
        }

    def test_json_output_includes_none_when_absent(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["inspect", str(bundle), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "attestation" in payload
        assert payload["attestation"] is None

    def test_json_output_includes_parsed_attestation(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        _write_json(bundle / "attestation.json", self._attestation_payload())
        result = runner.invoke(app, ["inspect", str(bundle), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["attestation"]["reviewer_id"] == "jdoe"
        assert payload["attestation"]["decision"] == "approved_with_conditions"
        assert len(payload["attestation"]["gate_overrides"]) == 1

    def test_rich_output_renders_attestation_panel(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        _write_json(bundle / "attestation.json", self._attestation_payload())
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        assert "Reviewer Attestation" in result.output
        assert "jdoe" in result.output
        assert "approved_with_conditions" in result.output

    def test_rich_output_omits_panel_when_absent(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        assert "Reviewer Attestation" not in result.output

    def test_pre_existing_bundle_without_attestation_does_not_crash(self, tmp_path: Path) -> None:
        """Bundles produced before this feature existed have no attestation.json
        — ``inspect`` must degrade gracefully, not crash."""
        bundle = _make_full_bundle(tmp_path)
        assert not (bundle / "attestation.json").exists()
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `diff` command
# ---------------------------------------------------------------------------


class TestDiff:
    def test_missing_bundle_a(self, tmp_path: Path) -> None:
        b = _make_minimal_bundle(tmp_path, name="b")
        result = runner.invoke(app, ["diff", str(tmp_path / "nope"), str(b)])
        assert result.exit_code == 1
        assert "bundle a" in result.output.lower()

    def test_missing_bundle_b(self, tmp_path: Path) -> None:
        a = _make_minimal_bundle(tmp_path, name="a")
        result = runner.invoke(app, ["diff", str(a), str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "bundle b" in result.output.lower()

    def test_two_full_bundles_compare(self, tmp_path: Path) -> None:
        a = _make_full_bundle(tmp_path, name="run_a")
        b = _make_full_bundle(tmp_path, name="run_b")
        # Tweak b's evidence to force a mismatch row
        _write_json(
            b / "evidence_manifest.json",
            {
                "richness_category": "sparse",
                "route_certainty": "oral",
                "nonlinear_clearance_evidence_strength": "none",
            },
        )
        result = runner.invoke(app, ["diff", str(a), str(b)])
        assert result.exit_code == 0
        assert "richness_category" in result.output


# ---------------------------------------------------------------------------
# `log` command
# ---------------------------------------------------------------------------


class TestLog:
    def test_missing_directory_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["log", str(tmp_path / "nope")])
        assert result.exit_code == 1

    def test_default_overview(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle)])
        assert result.exit_code == 0
        assert "Candidates" in result.output or "Gate" in result.output

    def test_corrupt_search_trajectory_does_not_crash(self, tmp_path: Path) -> None:
        """Regression: ``json.loads`` on the trajectory was previously
        unguarded; a corrupt line crashed ``log``."""
        bundle = _make_full_bundle(tmp_path)
        # Append a garbage line to the JSONL
        with (bundle / "search_trajectory.jsonl").open("a") as f:
            f.write("not-json-at-all\n")
        result = runner.invoke(app, ["log", str(bundle)])
        assert result.exit_code == 0
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_failed_flag(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--failed"])
        assert result.exit_code == 0
        assert "cand_003" in result.output

    def test_failed_flag_when_empty(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--failed"])
        assert result.exit_code == 0
        assert (
            "no failed candidates" in result.output.lower()
            or "failed_candidates" in result.output.lower()
        ), result.output

    def test_empty_trajectory_not_counted_as_one(self, tmp_path: Path) -> None:
        """Regression: ``''.split('\\n') == ['']`` used to make empty
        trajectories render as "1 total, 0 converged"."""
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "search_trajectory.jsonl").write_text("")
        result = runner.invoke(app, ["log", str(bundle)])
        assert result.exit_code == 0
        assert "1 total" not in result.output, result.output

    def test_non_dict_jsonl_rows_are_skipped(self, tmp_path: Path) -> None:
        """Regression: ``failed_candidates.jsonl`` containing ``[]`` or ``1``
        used to crash ``log --failed`` with AttributeError when ``.get()`` was
        called on the non-dict value."""
        bundle = _make_full_bundle(tmp_path)
        (bundle / "failed_candidates.jsonl").write_text('[]\n1\n"bad"\n')
        result = runner.invoke(app, ["log", str(bundle), "--failed"])
        assert result.exit_code == 0
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_non_dict_json_does_not_crash(self, tmp_path: Path) -> None:
        """Regression: ``_load_json`` used to return a list if the file
        contained one, then callers crashed with AttributeError on ``.get()``."""
        bundle = _make_full_bundle(tmp_path)
        (bundle / "ranking.json").write_text("[1, 2, 3]")
        result = runner.invoke(app, ["log", str(bundle), "--top", "2"])
        assert result.exit_code == 0
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_gate_filter(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--gate", "gate1"])
        assert result.exit_code == 0
        # Gate-specific output must actually render — not just a silent OK:
        assert "cand_00" in result.output, result.output

    def test_top_n(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--top", "2"])
        assert result.exit_code == 0
        assert "cand_002" in result.output

    def test_top_n_zero_suppresses_top_table(self, tmp_path: Path) -> None:
        """--top 0 is the documented "disabled" sentinel. Must succeed AND
        must not render the top-N table (which would appear for --top >= 1)."""
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--top", "0"])
        assert result.exit_code == 0
        # The top-N table includes a "Top N Candidates" rule; default overview does not.
        assert "top" not in result.output.lower().split("candidates")[0][-30:]

    def test_top_n_negative_rejected(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--top", "-1"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Machine-readable `--json` outputs
# ---------------------------------------------------------------------------


class TestJsonOutputs:
    """Ensure every read command honors --json with a parseable envelope.

    The contract: stdout is a single JSON object, ``ok`` is a bool, and Rich
    output is suppressed. Errors travel through the JSON envelope, not stderr,
    when --json is set.
    """

    def test_datasets_list_json(self) -> None:
        result = runner.invoke(app, ["datasets", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert isinstance(payload["datasets"], list)
        assert payload["count"] == len(payload["datasets"])

    def test_datasets_unknown_json(self) -> None:
        result = runner.invoke(app, ["datasets", "no_such_dataset_xyz", "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload == {
            "ok": False,
            "error": "unknown_dataset",
            "name": "no_such_dataset_xyz",
            "available": payload["available"],
        }
        assert isinstance(payload["available"], list)

    def test_doctor_json_envelope(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])
        # Exit code may be 0 or 1 depending on local env; both are valid as
        # long as the envelope is well-formed.
        assert result.exit_code in (0, 1)
        payload = json.loads(result.stdout)
        assert isinstance(payload["ok"], bool)
        assert isinstance(payload["components"], list)
        # Required taxonomy keys present on every component.
        for c in payload["components"]:
            assert {"name", "status", "detail", "required"} <= set(c.keys())

    def test_policies_json(self) -> None:
        result = runner.invoke(app, ["policies", "--json"])
        # 0 or 1 depending on whether policies/ is on disk in CI.
        assert result.exit_code in (0, 1)
        payload = json.loads(result.stdout)
        assert "ok" in payload
        assert "policies" in payload or "error" in payload

    def test_policies_single_lane_json_includes_raw(self) -> None:
        result = runner.invoke(app, ["policies", "submission", "--json"])
        if result.exit_code != 0:
            pytest.skip("policies/submission.json not present in this checkout")
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert len(payload["policies"]) == 1
        # Single-lane mode embeds the raw policy doc for jq drill-downs.
        assert "raw" in payload["policies"][0]

    def test_log_missing_bundle_json(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["log", str(tmp_path / "nope"), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["error"] == "not_a_directory"

    def test_diff_missing_bundle_json(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["diff", str(tmp_path / "a"), str(tmp_path / "b"), "--json"],
        )
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["error"] == "bundle_not_found"
        assert payload["missing"] == "A"

    def test_log_overview_json_on_full_bundle(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = runner.invoke(app, ["log", str(bundle), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["view"] == "overview"
        assert payload["trajectory"]["n_total"] >= 0

    def test_diff_full_bundles_json(self, tmp_path: Path) -> None:
        a = _make_full_bundle(tmp_path, name="run_a")
        b = _make_full_bundle(tmp_path, name="run_b")
        result = runner.invoke(app, ["diff", str(a), str(b), "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert "evidence_diff" in payload or "ranking_changes" in payload
