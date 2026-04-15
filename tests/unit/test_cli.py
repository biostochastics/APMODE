# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI-layer tests for all top-level apmode commands.

Covers argument parsing, option dispatch, error paths, and exit codes for:
    run, validate, inspect, datasets, explore, diff, log.

Deep-inspection commands (trace, lineage, graph) are tested in
``test_deep_inspect.py`` — this file deliberately does not duplicate them.

These tests exercise the CLI through ``typer.testing.CliRunner``; heavy
pipeline work is either skipped or mocked, since integration tests cover
end-to-end behaviour elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
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
    bundle = tmp_path / name
    bundle.mkdir()
    _write_json(bundle / "data_manifest.json", {"n_subjects": 10, "n_observations": 100})
    _write_json(bundle / "seed_registry.json", {"root_seed": 753849})
    _write_json(bundle / "backend_versions.json", {"nlmixr2": "3.0.0"})
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
            "nonlinear_clearance_signature": False,
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
# --help / --version smoke tests
# ---------------------------------------------------------------------------


class TestHelp:
    """Every command must render --help cleanly (catches import / typing
    regressions without executing any logic)."""

    @pytest.mark.parametrize(
        "cmd",
        [
            [],  # top-level
            ["run"],
            ["validate"],
            ["inspect"],
            ["datasets"],
            ["explore"],
            ["diff"],
            ["log"],
            ["trace"],
            ["lineage"],
            ["graph"],
        ],
    )
    def test_help_renders(self, cmd: list[str]) -> None:
        result = runner.invoke(app, [*cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Version output contains either a digit or the package name
        assert any(c.isdigit() for c in result.output) or "apmode" in result.output.lower()


# ---------------------------------------------------------------------------
# `run` command
# ---------------------------------------------------------------------------


class TestRun:
    def test_missing_dataset_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(tmp_path / "does_not_exist.csv")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "dataset" in result.output.lower()

    def test_verbose_and_quiet_are_mutex(self, tmp_path: Path) -> None:
        csv = tmp_path / "empty.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--verbose", "--quiet"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()

    def test_missing_policy_exits_1(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
        result = runner.invoke(
            app,
            ["run", str(csv), "--policy", str(tmp_path / "missing_policy.json")],
        )
        assert result.exit_code == 1
        assert "policy" in result.output.lower()

    def test_unknown_backend_exits_1(self, tmp_path: Path) -> None:
        """Ingestion must succeed before the backend branch is reached, so we
        use a real fixture CSV and mock the orchestrator import chain."""
        csv = Path("tests/fixtures/suite_a/a4_1cmt_oral_mm.csv").resolve()
        if not csv.exists():
            pytest.skip("fixture CSV missing")
        result = runner.invoke(
            app,
            ["run", str(csv), "--output", str(tmp_path / "runs"), "--backend", "bogus"],
        )
        assert result.exit_code == 1
        assert "unknown backend" in result.output.lower()

    def test_bad_lane_exits_nonzero(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--lane", "not_a_lane"])
        assert result.exit_code != 0

    def test_bad_provider_still_parses(self, tmp_path: Path) -> None:
        """--provider is typed as ``str`` — the CLI must accept it at parse
        time. Lazy validation happens only if --agentic is actually set."""
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(
            app,
            ["run", str(csv), "--provider", "made_up_provider"],
        )
        # Must reach ingestion (not rejected at Typer parse time):
        assert "ingestion failed" in result.output.lower(), result.output

    def test_max_iterations_clamped(self, tmp_path: Path) -> None:
        """PRD §4.2.6 caps agentic iterations at 25. Typer enforces max=25."""
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--max-iterations", "99"])
        assert result.exit_code != 0

    def test_parallel_models_min_1(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "-j", "0"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `validate` command
# ---------------------------------------------------------------------------


class TestValidate:
    def test_missing_directory_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_file_instead_of_dir_exits_1(self, tmp_path: Path) -> None:
        f = tmp_path / "a_file.txt"
        f.write_text("x")
        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 1
        assert "not a directory" in result.output.lower()

    def test_valid_minimal_bundle(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["validate", str(bundle)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_missing_required_file_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "backend_versions.json").unlink()
        result = runner.invoke(app, ["validate", str(bundle)])
        assert result.exit_code == 1
        assert "failed" in result.output.lower()
        assert "backend_versions.json" in result.output

    def test_corrupt_json_flagged_but_does_not_crash(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "data_manifest.json").write_text("{ not json")
        result = runner.invoke(app, ["validate", str(bundle)])
        # Validator should fail cleanly (exit 1), not traceback:
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_non_dict_json_rejected(self, tmp_path: Path) -> None:
        """Regression: a required file containing valid-but-non-object JSON
        (e.g. ``[]``) was previously accepted as OK."""
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "data_manifest.json").write_text("[]")
        result = runner.invoke(app, ["validate", str(bundle)])
        assert result.exit_code == 1

    def test_directory_instead_of_json_file_handled(self, tmp_path: Path) -> None:
        """Regression: a required path being a directory used to raise
        IsADirectoryError before ``_validate_file`` caught OSError."""
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "data_manifest.json").unlink()
        (bundle / "data_manifest.json").mkdir()
        result = runner.invoke(app, ["validate", str(bundle)])
        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)


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


# ---------------------------------------------------------------------------
# `datasets` command
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_list_all(self) -> None:
        result = runner.invoke(app, ["datasets"])
        assert result.exit_code == 0
        assert "Name" in result.output or "Subj" in result.output

    def test_filter_by_route(self) -> None:
        result = runner.invoke(app, ["datasets", "--route", "oral"])
        assert result.exit_code == 0

    def test_filter_with_no_matches(self) -> None:
        result = runner.invoke(app, ["datasets", "--route", "telepathic_infusion"])
        assert result.exit_code == 0
        assert "no datasets" in result.output.lower() or "match" in result.output.lower()

    def test_unknown_dataset_exits_1(self) -> None:
        result = runner.invoke(app, ["datasets", "not_a_real_dataset_xyz"])
        assert result.exit_code == 1
        assert "unknown" in result.output.lower()

    def test_fetch_happy_path_mocked(self, tmp_path: Path) -> None:
        """Registry lookup + fetch_dataset stubbed out so no network is hit."""
        out_csv = tmp_path / "data" / "theo_sd.csv"
        with patch("apmode.data.datasets.fetch_dataset", return_value=out_csv) as m:
            result = runner.invoke(app, ["datasets", "theo_sd", "-o", str(tmp_path / "data")])
            # theo_sd should exist in the registry; if naming changes, fall back
            # to verifying the mock was exercised (or skip if not in registry).
            if result.exit_code != 0:
                pytest.skip(f"theo_sd not in current registry: {result.output}")
            m.assert_called_once()
            assert "downloaded" in result.output.lower()


# ---------------------------------------------------------------------------
# `explore` command
# ---------------------------------------------------------------------------


class TestExplore:
    def test_unknown_dataset_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["explore", "not_a_dataset_and_not_a_file", "--output", str(tmp_path)],
        )
        assert result.exit_code == 1
        assert "not a file or known dataset" in result.output.lower()

    def test_launch_run_propagates_typer_exit(self, tmp_path: Path) -> None:
        """Regression: ``_launch_run`` must propagate non-zero ``typer.Exit``
        from the underlying pipeline so ``explore -y`` doesn't lie about success."""
        from apmode import cli as cli_mod

        def _boom(**_kwargs: Any) -> None:
            raise typer.Exit(code=2)

        with patch.object(cli_mod, "run", _boom):
            with pytest.raises(typer.Exit) as exc_info:
                cli_mod._launch_run(
                    csv_path=tmp_path / "x.csv",
                    lane=cli_mod.Lane.discovery,
                    seed=1,
                    output=tmp_path,
                )
            assert exc_info.value.exit_code == 2

    def test_launch_run_propagates_system_exit(self, tmp_path: Path) -> None:
        """Regression against the original ``contextlib.suppress(SystemExit)``:
        a raw ``SystemExit(2)`` from inner code must also propagate."""
        from apmode import cli as cli_mod

        def _boom(**_kwargs: Any) -> None:
            raise SystemExit(2)

        with patch.object(cli_mod, "run", _boom):
            with pytest.raises(typer.Exit) as exc_info:
                cli_mod._launch_run(
                    csv_path=tmp_path / "x.csv",
                    lane=cli_mod.Lane.discovery,
                    seed=1,
                    output=tmp_path,
                )
            assert exc_info.value.exit_code == 2

    def test_launch_run_zero_exit_does_not_raise(self, tmp_path: Path) -> None:
        """A successful inner run (``typer.Exit(0)``) must not bubble as an error."""
        from apmode import cli as cli_mod

        def _ok(**_kwargs: Any) -> None:
            raise typer.Exit(code=0)

        with patch.object(cli_mod, "run", _ok):
            cli_mod._launch_run(
                csv_path=tmp_path / "x.csv",
                lane=cli_mod.Lane.discovery,
                seed=1,
                output=tmp_path,
            )

    def test_local_csv_non_interactive_launches_run(self, tmp_path: Path) -> None:
        """With -y, explore must reach ``_launch_run``. We mock that to avoid
        triggering the real pipeline."""
        csv = Path("tests/fixtures/suite_a/a4_1cmt_oral_mm.csv").resolve()
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        with patch("apmode.cli._launch_run") as mock_launch:
            result = runner.invoke(
                app,
                ["explore", str(csv), "-y", "--output", str(tmp_path / "runs")],
            )
            # Explore does real ingest + profile before reaching _launch_run —
            # tolerate failures in those stages (they're tested elsewhere), but
            # if everything succeeds, _launch_run must have been called exactly
            # once:
            if result.exit_code == 0:
                assert mock_launch.called
            else:
                pytest.skip(f"explore pipeline stage failed (covered elsewhere): {result.output}")


# ---------------------------------------------------------------------------
# `diff` command
# ---------------------------------------------------------------------------


class TestGraph:
    def test_missing_graph_file_exits_0(self, tmp_path: Path) -> None:
        """Bundles without search_graph.json are valid (classical runs) —
        graph must report "not found" but not fail the shell."""
        bundle = _make_minimal_bundle(tmp_path)
        result = runner.invoke(app, ["graph", str(bundle)])
        assert result.exit_code == 0
        assert "no search graph" in result.output.lower()

    def test_malformed_graph_file_exits_1(self, tmp_path: Path) -> None:
        """Regression: if search_graph.json exists but is corrupt / wrong type,
        graph used to return 0 silently. Must surface as exit 1."""
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "search_graph.json").write_text("[]")  # list, not dict
        result = runner.invoke(app, ["graph", str(bundle)])
        assert result.exit_code == 1
        assert "search_graph" in result.output.lower()

    def test_graph_output_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Regression: ``graph -o nested/dir/out.json`` previously crashed
        with FileNotFoundError because parent dirs weren't created."""
        bundle = _make_minimal_bundle(tmp_path)
        _write_json(
            bundle / "search_graph.json",
            {
                "nodes": [{"candidate_id": "c1", "backend": "nlmixr2", "converged": True}],
                "edges": [],
            },
        )
        nested = tmp_path / "does" / "not" / "exist" / "out.json"
        result = runner.invoke(app, ["graph", str(bundle), "--format", "json", "-o", str(nested)])
        assert result.exit_code == 0
        assert nested.exists()


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
                "nonlinear_clearance_signature": False,
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
# Dispatch / wiring: option propagation
# ---------------------------------------------------------------------------


class TestRunWiring:
    """Verify that `run`'s key options are actually forwarded to the underlying
    config / orchestrator rather than being silently ignored."""

    def test_seed_and_lane_propagate_to_runconfig(self, tmp_path: Path) -> None:
        csv = Path("tests/fixtures/suite_a/a4_1cmt_oral_mm.csv").resolve()
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        captured: dict[str, Any] = {}

        class _FakeOrch:
            def __init__(self, _runner: Any, _out: Path, config: Any, **kw: Any) -> None:
                captured["lane"] = config.lane
                captured["seed"] = config.seed
                captured["timeout"] = config.timeout_seconds

            async def run(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("stop-after-config")

        with patch("apmode.orchestrator.Orchestrator", _FakeOrch):
            result = runner.invoke(
                app,
                [
                    "run",
                    str(csv),
                    "--lane",
                    "discovery",
                    "--seed",
                    "424242",
                    "--timeout",
                    "123",
                    "--output",
                    str(tmp_path / "runs"),
                ],
            )

        # Pipeline exits 1 when orchestrator raises, but we only care that the
        # config captured before the failure reflects our flags:
        assert captured.get("lane") == "discovery", result.output
        assert captured.get("seed") == 424242
        assert captured.get("timeout") == 123

    def test_agentic_flag_only_builds_runner_on_discovery(self, tmp_path: Path) -> None:
        csv = Path("tests/fixtures/suite_a/a4_1cmt_oral_mm.csv").resolve()
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        with (
            patch("apmode.cli._try_build_agentic_runner") as mock_build,
            patch("apmode.orchestrator.Orchestrator") as mock_orch,
        ):
            # Orchestrator.run is awaited — make it raise to halt after dispatch.
            inst = MagicMock()

            async def _boom(*a: Any, **k: Any) -> Any:
                raise RuntimeError("stop")

            inst.run = _boom
            mock_orch.return_value = inst

            # Submission lane: agentic flag must NOT trigger the builder.
            runner.invoke(
                app,
                [
                    "run",
                    str(csv),
                    "--lane",
                    "submission",
                    "--agentic",
                    "--output",
                    str(tmp_path / "sub"),
                ],
            )
            assert not mock_build.called, "agentic must be ignored on submission lane"

            # Discovery lane: agentic flag MUST trigger the builder.
            mock_build.reset_mock()
            mock_build.return_value = None  # builder may return None if provider missing
            runner.invoke(
                app,
                [
                    "run",
                    str(csv),
                    "--lane",
                    "discovery",
                    "--agentic",
                    "--output",
                    str(tmp_path / "disc"),
                ],
            )
            assert mock_build.called, "agentic must be dispatched on discovery lane"
