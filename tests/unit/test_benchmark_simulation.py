# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Benchmark Suite A simulation scaffolding.

Validates:
- R simulation script exists and has correct structure
- Reference params align with DSLSpec scenarios
- Generated CSV files (when available) can be ingested
"""

from __future__ import annotations

from pathlib import Path

from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    scenario_a1,
    scenario_a2,
    scenario_a3,
    scenario_a4,
)

SUITE_A_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "suite_a"


class TestSimulationScaffolding:
    """Validate the simulation R script and reference structure."""

    def test_simulation_script_exists(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        assert script.exists(), "simulate_all.R should exist in benchmarks/suite_a/"

    def test_simulation_script_has_all_scenarios(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        # A20a/A20b share one R simulator function (sim_A20) -- see
        # SCENARIO_FILENAME_STEMS / ALL_SCENARIOS in suite_a.py.
        for scenario_id in (
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
            "A17",
            "A18",
            "A19",
            "A20",
            "A21",
        ):
            assert f"sim_{scenario_id} <-" in content, (
                f"Simulator function sim_{scenario_id} missing"
            )

    def test_simulation_script_uses_rxode2(self) -> None:
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "library(rxode2)" in content

    def test_simulation_script_uses_reproducible_seeds(self) -> None:
        """Per-scenario, per-replicate seeds derived from BASE_SEED."""
        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        assert "BASE_SEED" in content
        assert "scn_seed" in content

    def test_simulation_output_filenames(self) -> None:
        """Check the R script advertises the expected filename stems."""
        from apmode.benchmarks.suite_a import SCENARIO_FILENAME_STEMS

        script = SUITE_A_DIR / "simulate_all.R"
        content = script.read_text()
        for stem in SCENARIO_FILENAME_STEMS.values():
            assert stem in content, f"Filename stem {stem} not referenced in R script"
        assert "reference_params.json" in content


class TestReferenceParamsAlignWithSpecs:
    """Ensure REFERENCE_PARAMS match DSLSpec structural_param_names()."""

    def test_a1_params_match_spec(self) -> None:
        spec = scenario_a1()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A1"].keys())
        assert spec_params == ref_params

    def test_a2_params_match_spec(self) -> None:
        spec = scenario_a2()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A2"].keys())
        assert spec_params == ref_params

    def test_a3_params_match_spec(self) -> None:
        spec = scenario_a3()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A3"].keys())
        assert spec_params == ref_params

    def test_a4_params_match_spec(self) -> None:
        spec = scenario_a4()
        spec_params = set(spec.structural_param_names())
        ref_params = set(REFERENCE_PARAMS["A4"].keys())
        assert spec_params == ref_params

    def test_all_scenarios_have_reference_params(self) -> None:
        for name, _ in ALL_SCENARIOS:
            assert name in REFERENCE_PARAMS, f"Missing reference params for {name}"

    def test_reference_param_values_are_positive(self) -> None:
        for name, params in REFERENCE_PARAMS.items():
            for param_name, value in params.items():
                assert value > 0, f"{name}.{param_name} should be positive, got {value}"


class TestMultiReplicateDiscovery:
    """The dataset discovery helpers must handle both single-replicate and
    multi-replicate output layouts produced by simulate_all.R."""

    def test_stems_cover_every_scenario(self) -> None:
        from apmode.benchmarks.suite_a import (
            SCENARIO_FILENAME_STEMS as stems,
        )

        assert set(stems) == {
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "A16",
            "A17",
            "A18",
            "A19",
            "A20a",
            "A20b",
            "A21",
        }
        for scenario_id, stem in stems.items():
            # A20a/A20b share the "a20_..." stem (paired benchmark unit
            # against one CSV) so the naive lower-cased-id prefix check
            # doesn't apply to them.
            if scenario_id in ("A20a", "A20b"):
                assert stem == "a20_1cmt_oral_blq_elevated_lloq"
                continue
            assert stem.startswith(scenario_id.lower() + "_"), (
                f"Stem {stem} should start with {scenario_id.lower()}_"
            )

    def test_scenario_dataset_paths_single_replicate(self, tmp_path: Path) -> None:
        from apmode.benchmarks.suite_a import (
            SCENARIO_FILENAME_STEMS,
            scenario_dataset_paths,
        )

        stem = SCENARIO_FILENAME_STEMS["A1"]
        csv = tmp_path / f"{stem}.csv"
        csv.write_text("NMID,TIME,DV\n1,0,0\n")
        eta = tmp_path / f"{stem}_eta.csv"
        eta.write_text("NMID,eta.CL\n1,0.1\n")

        assert scenario_dataset_paths(tmp_path, "A1") == [csv]
        assert scenario_dataset_paths(tmp_path, "A1", include_eta=True) == [csv, eta]

    def test_scenario_dataset_paths_multi_replicate(self, tmp_path: Path) -> None:
        from apmode.benchmarks.suite_a import (
            SCENARIO_FILENAME_STEMS,
            scenario_dataset_paths,
        )

        stem = SCENARIO_FILENAME_STEMS["A8"]
        rep_paths: list[Path] = []
        for i in range(1, 4):
            csv = tmp_path / f"{stem}_rep{i:02d}.csv"
            csv.write_text("NMID,TIME,DV\n")
            eta = tmp_path / f"{stem}_rep{i:02d}_eta.csv"
            eta.write_text("NMID,eta.CL\n")
            rep_paths.append(csv)

        assert scenario_dataset_paths(tmp_path, "A8") == rep_paths
        got = scenario_dataset_paths(tmp_path, "A8", include_eta=True)
        assert len(got) == 6
        # csv/eta pairs are interleaved
        assert got[0].name.endswith("_rep01.csv")
        assert got[1].name.endswith("_rep01_eta.csv")

    def test_scenario_dataset_paths_missing_returns_empty(self, tmp_path: Path) -> None:
        from apmode.benchmarks.suite_a import scenario_dataset_paths

        assert scenario_dataset_paths(tmp_path, "A5") == []

    def test_scenario_dataset_paths_unknown_scenario_raises(self, tmp_path: Path) -> None:
        from apmode.benchmarks.suite_a import scenario_dataset_paths

        try:
            scenario_dataset_paths(tmp_path, "A99")
        except KeyError:
            return
        raise AssertionError("Unknown scenario id should raise KeyError")

    def test_manifest_covers_all_scenarios(self, tmp_path: Path) -> None:
        from apmode.benchmarks.suite_a import (
            SCENARIO_FILENAME_STEMS,
            suite_a_manifest,
        )

        # Only A1 and A6 are materialized; the manifest still lists all eight.
        (tmp_path / f"{SCENARIO_FILENAME_STEMS['A1']}.csv").write_text("x")
        (tmp_path / f"{SCENARIO_FILENAME_STEMS['A6']}_rep05.csv").write_text("x")

        manifest = suite_a_manifest(tmp_path)
        assert set(manifest) == set(SCENARIO_FILENAME_STEMS)
        assert len(manifest["A1"]) == 1
        assert len(manifest["A6"]) == 1
        assert manifest["A2"] == []
        assert manifest["A8"] == []

    def test_current_suite_a_directory_is_discoverable(self) -> None:
        """The real suite_a directory should be enumerable end-to-end."""
        from apmode.benchmarks.suite_a import suite_a_manifest

        if not (SUITE_A_DIR / "simulate_all.R").exists():
            return  # scaffolding not present in this checkout
        manifest = suite_a_manifest(SUITE_A_DIR)
        # Whether the sim has been run in this checkout is optional; the
        # manifest must at least return every expected key.
        from apmode.benchmarks.suite_a import SCENARIO_FILENAME_STEMS

        assert set(manifest) == set(SCENARIO_FILENAME_STEMS)
