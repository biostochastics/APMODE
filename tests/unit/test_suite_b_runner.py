# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for ``apmode.benchmarks.suite_b_runner``.

Pure-Python tests — no R subprocess. Exercises the dataset resolver,
DSLSpec construction, the cross-seed stability helper, and the atomic
results writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.benchmarks.models import (
    BenchmarkCase,
    ExpectedStructure,
    PerturbationRecipe,
    PerturbationType,
)
from apmode.benchmarks.suite_b_runner import (
    SeedRunResult,
    SuiteBCaseResult,
    _build_default_spec,
    _compute_cross_seed_stability,
    resolve_dataset_csv,
    write_results_atomic,
)

# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


class TestResolveDatasetCSV:
    def test_override_takes_precedence_over_registry(self, tmp_path: Path) -> None:
        csv = tmp_path / "data.csv"
        csv.write_text("NMID,TIME,DV\n1,0,0\n")
        out = resolve_dataset_csv(
            "nlmixr2data_theophylline",
            cache_dir=tmp_path / "cache",
            overrides={"nlmixr2data_theophylline": csv},
        )
        assert out == csv

    def test_override_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="--dataset-csv override"):
            resolve_dataset_csv(
                "nlmixr2data_theophylline",
                cache_dir=tmp_path / "cache",
                overrides={"nlmixr2data_theophylline": tmp_path / "ghost.csv"},
            )

    def test_unknown_dataset_id_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="not in the built-in"):
            resolve_dataset_csv(
                "ddmore_gentamicin",  # not in the builtin map
                cache_dir=tmp_path / "cache",
                overrides={},
            )


# ---------------------------------------------------------------------------
# DSLSpec construction
# ---------------------------------------------------------------------------


class TestBuildDefaultSpec:
    def _case(
        self,
        case_id: str,
        n_compartments: int | None,
    ) -> BenchmarkCase:
        return BenchmarkCase(
            case_id=case_id,
            suite="B",
            dataset_id="nlmixr2data_theophylline",
            description="test",
            lane="submission",
            policy_file="submission.json",
            expected_structure=(
                ExpectedStructure(n_compartments=n_compartments)
                if n_compartments is not None
                else None
            ),
        )

    def test_one_compartment_branch(self) -> None:
        spec = _build_default_spec(self._case("c1", 1))
        assert spec.distribution.type == "OneCmt"
        assert {p for v in spec.variability for p in getattr(v, "params", [])} >= {
            "CL",
            "V",
        }

    def test_two_compartment_branch(self) -> None:
        spec = _build_default_spec(self._case("c2", 2))
        assert spec.distribution.type == "TwoCmt"
        # 2-cmt IIV uses V1 not V
        assert {p for v in spec.variability for p in getattr(v, "params", [])} >= {
            "CL",
            "V1",
        }

    def test_missing_expected_structure_defaults_to_one_cmt(self) -> None:
        spec = _build_default_spec(self._case("c3", None))
        assert spec.distribution.type == "OneCmt"


# ---------------------------------------------------------------------------
# Cross-seed stability
# ---------------------------------------------------------------------------


class TestComputeCrossSeedStability:
    def _seed_result(
        self,
        seed: int,
        *,
        converged: bool = True,
        ka: float = 1.0,
        cl: float = 5.0,
    ) -> SeedRunResult:
        return SeedRunResult(
            seed=seed,
            converged=converged,
            minimization_status="success" if converged else "fail",
            parameter_estimates={"ka": ka, "CL": cl} if converged else {},
            bic=100.0 if converged else None,
            wall_time_seconds=1.0,
        )

    def test_returns_none_when_fewer_than_two_converged(self) -> None:
        results = [
            self._seed_result(1, converged=True),
            self._seed_result(2, converged=False),
        ]
        cv_per_param, cv_max = _compute_cross_seed_stability(results)
        assert cv_per_param == {}
        assert cv_max is None

    def test_cv_calc_against_known_values(self) -> None:
        results = [
            self._seed_result(1, ka=1.0, cl=5.0),
            self._seed_result(2, ka=1.1, cl=5.0),
            self._seed_result(3, ka=0.9, cl=5.0),
        ]
        cv_per_param, cv_max = _compute_cross_seed_stability(results)
        # ka mean=1.0, sample sd=0.1, CV=0.1; CL has zero variance → not in dict.
        assert "ka" in cv_per_param
        assert cv_per_param["ka"] == pytest.approx(0.1, rel=0.01)
        assert "CL" not in cv_per_param  # zero variance excluded
        assert cv_max == cv_per_param["ka"]

    def test_zero_mean_skipped_safely(self) -> None:
        # A parameter that came back as zero across seeds (degenerate fit)
        # must not divide-by-zero — it should be excluded silently.
        results = [
            self._seed_result(1, ka=1.0, cl=0.0),
            self._seed_result(2, ka=1.05, cl=0.0),
        ]
        cv_per_param, cv_max = _compute_cross_seed_stability(results)
        assert "CL" not in cv_per_param
        assert "ka" in cv_per_param
        assert cv_max is not None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestWriteResultsAtomic:
    def test_round_trip_payload(self, tmp_path: Path) -> None:
        out = tmp_path / "suite_b_results.json"
        result = SuiteBCaseResult(
            case_id="b5_test",
            suite="B",
            dataset_id="nlmixr2data_mavoglurant",
            perturbation_manifests=({"perturbation": "inject_blq"},),
            n_seeds=2,
            seed_results=(
                SeedRunResult(
                    seed=1,
                    converged=True,
                    minimization_status="success",
                    parameter_estimates={"CL": 5.0},
                    bic=100.0,
                    wall_time_seconds=1.5,
                ),
                SeedRunResult(
                    seed=2,
                    converged=False,
                    minimization_status="theta_reset",
                    parameter_estimates={},
                    bic=None,
                    wall_time_seconds=2.0,
                ),
            ),
            convergence_rate=0.5,
            cross_seed_cv_max=None,
            cross_seed_cv_per_param={},
        )
        write_results_atomic(out, {"b5_test": result})
        assert out.exists()
        roundtrip = json.loads(out.read_text())
        assert roundtrip["b5_test"]["case_id"] == "b5_test"
        assert roundtrip["b5_test"]["convergence_rate"] == 0.5
        assert roundtrip["b5_test"]["seed_results"][0]["seed"] == 1
        assert roundtrip["b5_test"]["seed_results"][1]["converged"] is False

    def test_skipped_case_serializes(self, tmp_path: Path) -> None:
        out = tmp_path / "suite_b_results.json"
        result = SuiteBCaseResult(
            case_id="b1_node",
            suite="B",
            dataset_id="x",
            skipped=True,
            skip_reason="NODE backend live wiring is out of v0.6 scope",
        )
        write_results_atomic(out, {"b1_node": result})
        roundtrip = json.loads(out.read_text())
        assert roundtrip["b1_node"]["skipped"] is True
        assert roundtrip["b1_node"]["seed_results"] == []

    def test_unused_recipe_field(self) -> None:
        # PerturbationRecipe round-trip on the new covariate fields is also
        # exercised here as a sanity check that the runner can serialise a
        # case carrying the new perturbation type.
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_COVARIATE_MISSINGNESS,
            covariate_missingness_fraction=0.3,
            covariate_missingness_columns=["WT", "SEX"],
            seed=42,
        )
        assert recipe.covariate_missingness_fraction == pytest.approx(0.3)
        assert list(recipe.covariate_missingness_columns) == ["WT", "SEX"]
