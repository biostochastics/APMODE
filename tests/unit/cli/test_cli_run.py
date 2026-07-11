# SPDX-License-Identifier: GPL-2.0-or-later
"""CLI-layer tests for ``apmode run`` — argument parsing, option dispatch,
config/orchestrator wiring, and APMODE_* environment-variable bindings.

Split out of the former monolithic ``test_cli.py``. These tests exercise the
CLI through ``typer.testing.CliRunner``; heavy pipeline work is mocked, since
integration tests cover end-to-end behaviour elsewhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from apmode.cli import app
from tests._helpers.fixtures_data import FIXTURES_DIR

runner = CliRunner()


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
        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
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

    def test_binary_encode_invalid_no_equals(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--binary-encode", "SEX"])
        assert result.exit_code == 1
        assert "invalid --binary-encode" in result.output.lower()

    def test_binary_encode_invalid_target(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--binary-encode", "SEX=M:3,F:1"])
        assert result.exit_code == 1
        assert "invalid --binary-encode target" in result.output.lower()

    def test_binary_encode_valid_parses(self, tmp_path: Path) -> None:
        # Valid flag should parse and then fail later at ingestion (empty CSV).
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--binary-encode", "SEX=M:0,F:1"])
        # Must get past parsing into the pipeline.
        assert "invalid --binary-encode" not in result.output.lower()

    def test_provenance_missing_file_exits_1(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(
            app,
            ["run", str(csv), "--provenance", str(tmp_path / "missing_provenance.json")],
        )
        assert result.exit_code == 1
        assert "provenance" in result.output.lower()

    def test_provenance_invalid_json_exits_1(self, tmp_path: Path) -> None:
        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
        if not csv.exists():
            pytest.skip("fixture CSV missing")
        bad_provenance = tmp_path / "bad_provenance.json"
        bad_provenance.write_text(json.dumps({"source_system": "NONMEM dataset"}))
        result = runner.invoke(
            app,
            [
                "run",
                str(csv),
                "--output",
                str(tmp_path / "runs"),
                "--provenance",
                str(bad_provenance),
            ],
        )
        assert result.exit_code == 1
        assert "invalid --provenance" in result.output.lower()

    def test_model_influence_invalid_value_exits_1(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--model-influence", "extreme"])
        assert result.exit_code == 1
        assert "--model-influence" in result.output.lower()

    def test_decision_consequence_invalid_value_exits_1(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(app, ["run", str(csv), "--decision-consequence", "extreme"])
        assert result.exit_code == 1
        assert "--decision-consequence" in result.output.lower()

    def test_model_influence_and_decision_consequence_valid_parse(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n")
        result = runner.invoke(
            app,
            [
                "run",
                str(csv),
                "--model-influence",
                "high",
                "--decision-consequence",
                "high",
            ],
        )
        # Must get past parsing/validation into the pipeline (empty CSV
        # then fails at ingestion, not at flag validation).
        assert "--model-influence" not in result.output.lower()
        assert "--decision-consequence" not in result.output.lower()


# ---------------------------------------------------------------------------
# Dispatch / wiring: option propagation
# ---------------------------------------------------------------------------


class TestRunWiring:
    """Verify that `run`'s key options are actually forwarded to the underlying
    config / orchestrator rather than being silently ignored."""

    def test_seed_and_lane_propagate_to_runconfig(self, tmp_path: Path) -> None:
        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        captured: dict[str, Any] = {}

        class _FakeOrch:
            def __init__(self, _runner: Any, _out: Path, config: Any, **_kw: Any) -> None:
                # ``_runner``/``_out``/``_kw`` mirror the real Orchestrator
                # signature so isinstance/positional callers still bind; only
                # ``config`` is interesting for this test.
                del _runner, _out, _kw
                captured["lane"] = config.lane
                captured["seed"] = config.seed
                captured["timeout"] = config.timeout_seconds

            async def run(self, *_args: Any, **_kwargs: Any) -> Any:
                del _args, _kwargs
                raise RuntimeError("stop-after-config")

        with (
            patch("apmode.backends.nlmixr2_runner.Nlmixr2Runner") as _fake_runner,
            patch("apmode.orchestrator.Orchestrator", _FakeOrch),
        ):
            _fake_runner.return_value = MagicMock()
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

    def test_provenance_flag_forwarded_to_orchestrator_run(self, tmp_path: Path) -> None:
        """--provenance is parsed and forwarded as a DataProvenance kwarg to
        Orchestrator.run; omitting the flag forwards None (byte-identical
        bundle to a --provenance-less run — no new required artifact)."""
        from apmode.bundle.models import DataProvenance

        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        provenance_path = tmp_path / "provenance.json"
        provenance_path.write_text(
            DataProvenance(
                source_system="NONMEM dataset",
                time_zero_definition="first dose administration, protocol-defined",
                blq_handling_method="M3_likelihood",
            ).model_dump_json()
        )

        captured: dict[str, Any] = {}

        class _FakeOrch:
            def __init__(self, _runner: Any, _out: Path, _config: Any, **_kw: Any) -> None:
                del _runner, _out, _config, _kw

            async def run(self, *_args: Any, **kwargs: Any) -> Any:
                captured["data_provenance"] = kwargs.get("data_provenance")
                raise RuntimeError("stop-after-capture")

        with (
            patch("apmode.backends.nlmixr2_runner.Nlmixr2Runner") as _fake_runner,
            patch("apmode.orchestrator.Orchestrator", _FakeOrch),
        ):
            _fake_runner.return_value = MagicMock()
            runner.invoke(
                app,
                [
                    "run",
                    str(csv),
                    "--provenance",
                    str(provenance_path),
                    "--output",
                    str(tmp_path / "with_prov"),
                ],
            )
        assert isinstance(captured.get("data_provenance"), DataProvenance)
        assert captured["data_provenance"].source_system == "NONMEM dataset"

        captured_no_prov: dict[str, Any] = {}

        class _FakeOrchNoProv:
            def __init__(self, _runner: Any, _out: Path, _config: Any, **_kw: Any) -> None:
                del _runner, _out, _config, _kw

            async def run(self, *_args: Any, **kwargs: Any) -> Any:
                captured_no_prov["data_provenance"] = kwargs.get("data_provenance")
                raise RuntimeError("stop-after-capture")

        with (
            patch("apmode.backends.nlmixr2_runner.Nlmixr2Runner") as _fake_runner,
            patch("apmode.orchestrator.Orchestrator", _FakeOrchNoProv),
        ):
            _fake_runner.return_value = MagicMock()
            runner.invoke(
                app,
                ["run", str(csv), "--output", str(tmp_path / "without_prov")],
            )
        assert captured_no_prov.get("data_provenance") is None

    def test_agentic_flag_only_builds_runner_on_discovery(self, tmp_path: Path) -> None:
        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        with (
            patch("apmode.backends.nlmixr2_runner.Nlmixr2Runner") as _fake_runner,
            patch("apmode.cli._try_build_agentic_runner") as mock_build,
            patch("apmode.orchestrator.Orchestrator") as mock_orch,
        ):
            _fake_runner.return_value = MagicMock()
            # Orchestrator.run is awaited — make it raise to halt after dispatch.
            inst = MagicMock()

            async def _boom(*_a: Any, **_k: Any) -> Any:
                del _a, _k
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
            # --yes bypasses the data-sharing confirmation prompt added for
            # non-local providers (required since the test runner provides no stdin).
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
                    "--yes",
                    "--api-base",
                    "https://llm.example/v1",
                    "--output",
                    str(tmp_path / "disc"),
                ],
            )
            assert mock_build.called, "agentic must be dispatched on discovery lane"
            assert mock_build.call_args.kwargs["api_base"] == "https://llm.example/v1"

    def test_agentic_builder_passes_api_base_to_llm_config(self, tmp_path: Path) -> None:
        from apmode.cli import _try_build_agentic_runner

        captured = {}

        def _fake_create(config: Any) -> MagicMock:
            captured["config"] = config
            client = MagicMock()

            async def _complete(*_a: Any, **_k: Any) -> Any:
                del _a, _k

            client.complete = _complete
            return client

        with patch("apmode.backends.llm_providers.create_llm_client", side_effect=_fake_create):
            runner_obj = _try_build_agentic_runner(
                inner_runner=MagicMock(),
                provider="ollama",
                model_name="llama3.1:8b",
                api_base="http://localhost:11435",
                max_iterations=1,
                lane="discovery",
                trace_dir=tmp_path / "agentic_trace",
                quiet=True,
            )

        assert runner_obj is not None
        assert captured["config"].api_base == "http://localhost:11435"

    def test_agentic_builder_allows_litellm_fallback_provider(self, tmp_path: Path) -> None:
        from apmode.cli import _try_build_agentic_runner

        captured = {}

        def _fake_create(config: Any) -> MagicMock:
            captured["config"] = config
            client = MagicMock()

            async def _complete(*_a: Any, **_k: Any) -> Any:
                del _a, _k

            client.complete = _complete
            return client

        with (
            patch("apmode.backends.llm_providers.available_providers", return_value=["litellm"]),
            patch("apmode.backends.llm_providers.create_llm_client", side_effect=_fake_create),
        ):
            runner_obj = _try_build_agentic_runner(
                inner_runner=MagicMock(),
                provider="custom_provider",
                model_name="custom/model",
                api_base=None,
                max_iterations=1,
                lane="discovery",
                trace_dir=tmp_path / "agentic_trace",
                quiet=True,
            )

        assert runner_obj is not None
        assert captured["config"].provider == "custom_provider"
        assert captured["config"].model == "custom/model"


# ---------------------------------------------------------------------------
# Environment-variable bindings on `apmode run`
# ---------------------------------------------------------------------------


class TestEnvVarBindings:
    """`apmode run` must respect APMODE_* env vars when CLI flags are omitted."""

    def test_apmode_lane_envvar(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        csv = FIXTURES_DIR / "suite_a" / "a4_1cmt_oral_mm.csv"
        if not csv.exists():
            pytest.skip("fixture CSV missing")

        captured: dict[str, Any] = {}

        class _FakeOrch:
            def __init__(self, _runner: Any, _out: Path, config: Any, **_kw: Any) -> None:
                del _runner, _out, _kw
                captured["lane"] = config.lane
                captured["seed"] = config.seed

            async def run(self, *_args: Any, **_kwargs: Any) -> Any:
                del _args, _kwargs
                raise RuntimeError("stop-after-config")

        monkeypatch.setenv("APMODE_LANE", "discovery")
        monkeypatch.setenv("APMODE_SEED", "999")

        with (
            patch("apmode.backends.nlmixr2_runner.Nlmixr2Runner") as _fake_runner,
            patch("apmode.orchestrator.Orchestrator", _FakeOrch),
        ):
            _fake_runner.return_value = MagicMock()
            runner.invoke(app, ["run", str(csv), "--output", str(tmp_path / "runs")])

        assert captured.get("lane") == "discovery"
        assert captured.get("seed") == 999

    def test_envvars_documented_in_help(self) -> None:
        """Every documented APMODE_* var must show up in `run --help`.

        Rich wraps and may ellipsis-truncate long identifiers inside option
        panels, so we widen the rendering terminal and collapse whitespace
        before checking. Truncation manifests as a trailing ``…`` so we also
        accept a short prefix match for the longest identifier.
        """
        # Click's CliRunner exposes terminal width via env COLUMNS — Rich
        # honors this when rendering tables, eliminating ellipsis truncation
        # for our longer identifiers.
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "240"})
        # Strip box-drawing borders, then collapse all whitespace runs into a
        # single space so "APMODE_PARALLEL_\nMODELS" still matches.
        haystack = re.sub(r"\s+", " ", re.sub(r"[│╭╮╰╯─]", " ", result.output))
        for var in (
            "APMODE_LANE",
            "APMODE_SEED",
            "APMODE_TIMEOUT",
            "APMODE_OUTPUT_DIR",
            "APMODE_BACKEND",
            "APMODE_PROVIDER",
            "APMODE_MODEL",
            "APMODE_AGENTIC_MAX_ITER",
            "APMODE_PARALLEL_MODELS",
            "APMODE_POLICY",
        ):
            assert var in haystack, f"{var} missing from `run --help`"

    def test_run_output_short_flag(self) -> None:
        """`run -o <dir>` and `--output-dir <dir>` must both be accepted
        as aliases for ``--output`` (per skill docs).

        Asserts on functional behaviour (typer parses both forms
        without raising "no such option") rather than on rich's
        rendered ``--help`` output: rich/typer minor-version drift
        (CI's environment vs. local) sometimes hides the second alias
        from the help table even when the option *is* registered.
        Functional acceptance is the actual contract — we exercise it
        directly here.
        """
        # Functional acceptance: invoke `apmode run` with each alias
        # against a non-existent dataset. typer's "no such option"
        # parsing error exits with code 2 and a "No such option"
        # diagnostic; an *accepted* alias proceeds past argument
        # parsing and surfaces a *different* error (missing dataset
        # file). We assert that none of the three aliases hits the
        # parser-error path.
        for alias in ("-o", "--output", "--output-dir"):
            result = runner.invoke(
                app,
                ["run", "/nonexistent/dataset.csv", alias, "/tmp/out"],
                env={"COLUMNS": "240"},
            )
            # Typer's option-parsing error returns exit code 2 with
            # "No such option" in stderr/stdout. Anything else
            # (missing dataset, validation error, ...) means the
            # alias was accepted.
            output = (result.output or "") + (
                result.stderr_bytes.decode()
                if hasattr(result, "stderr_bytes") and result.stderr_bytes
                else ""
            )
            assert "No such option" not in output, (
                f"{alias!r} not accepted by `apmode run` — typer reports "
                f"'No such option' which means the alias is missing from "
                f"the option's decl list. Output: {output[:200]}"
            )

        # And the help text exposes -o (the most-likely-rendered alias
        # under any rich version); we don't pin --output-dir's
        # rendering because rich condenses long-option lists.
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "240"})
        assert "-o" in result.output
