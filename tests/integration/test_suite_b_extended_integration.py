# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for Suite B Extended (real-data anchors + perturbations).

Tests case definitions, perturbation recipes, and dispatch assertions.
Does NOT require real data files — validates the Python-side specs and
perturbation logic using synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.benchmarks.models import PerturbationRecipe, PerturbationType
from apmode.benchmarks.perturbations import apply_perturbation, apply_perturbations
from apmode.benchmarks.suite_b_extended import (
    ALL_EXTENDED_CASES,
    CASE_B4_THEO_NODE,
    CASE_B5_MAVO_BLQ25,
    CASE_B7_MAVO_SPARSE_ABS,
    CASE_B8_MAVO_NULL_COV,
    CASE_B9_GENTA_IOV,
    NIGHTLY_CASES,
)

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def _make_synthetic_pk_data(
    n_subjects: int = 20,
    n_obs_per: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic NONMEM-style PK data for perturbation testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0][:n_obs_per]
    for subj in range(1, n_subjects + 1):
        # Dosing event
        rows.append(
            {
                "NMID": subj,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0,
                "CMT": 1,
                "WT": float(rng.normal(70, 15)),
            }
        )
        # Observation events with realistic PK profile
        for t in times:
            conc = 50.0 * np.exp(-0.1 * t) * (1 - np.exp(-1.5 * t))
            conc *= 1 + rng.normal(0, 0.15)  # 15% proportional error
            rows.append(
                {
                    "NMID": subj,
                    "TIME": t,
                    "DV": max(0.01, conc),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                    "WT": rows[-1]["WT"],  # Same WT as dose row
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Case definition tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteBExtendedCases:
    """Validate extended Suite B case definitions."""

    def test_all_cases_count(self) -> None:
        """7 extended cases (B4-B9, with B5 having two BLQ variants)."""
        assert len(ALL_EXTENDED_CASES) == 7

    def test_all_cases_are_suite_b(self) -> None:
        """All extended cases belong to Suite B."""
        for case in ALL_EXTENDED_CASES:
            assert case.suite == "B"

    def test_b4_is_discovery_lane(self) -> None:
        """B4 theophylline NODE uses discovery lane."""
        assert CASE_B4_THEO_NODE.lane == "discovery"
        assert CASE_B4_THEO_NODE.dataset_id == "nlmixr2data_theophylline"

    def test_b5_has_blq_perturbation(self) -> None:
        """B5 cases inject BLQ at different fractions."""
        assert len(CASE_B5_MAVO_BLQ25.perturbations) == 1
        assert CASE_B5_MAVO_BLQ25.perturbations[0].blq_fraction == 0.25

    def test_b7_expects_node_exclusion(self) -> None:
        """B7 sparse absorption should exclude NODE from dispatch."""
        assert "jax_node" in CASE_B7_MAVO_SPARSE_ABS.expected_dispatch_excludes

    def test_b8_has_null_covariates(self) -> None:
        """B8 adds 5 null random covariates."""
        assert len(CASE_B8_MAVO_NULL_COV.perturbations) == 1
        assert CASE_B8_MAVO_NULL_COV.perturbations[0].null_covariate_n == 5

    def test_b9_has_split_strategy(self) -> None:
        """B9 gentamicin IOV uses 5-fold CV."""
        assert CASE_B9_GENTA_IOV.split_strategy is not None
        assert CASE_B9_GENTA_IOV.split_strategy.n_folds == 5

    def test_nightly_subset(self) -> None:
        """Nightly subset includes only per_pr and nightly cadence cases."""
        for case in NIGHTLY_CASES:
            assert case.ci_cadence in ("per_pr", "nightly")


# ---------------------------------------------------------------------------
# Perturbation logic tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPerturbationBLQ:
    """Test BLQ injection perturbation."""

    def test_inject_blq_25pct(self) -> None:
        """25% BLQ injection produces ~25% censored observations."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.25,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert "BLQ_FLAG" in result.columns
        assert "LLOQ" in result.columns
        assert "CENS" in result.columns  # M3-style censoring column
        assert manifest["perturbation"] == "inject_blq"
        assert manifest["method"] == "M3"
        assert manifest["n_censored"] > 0
        # Actual fraction should be approximately 25%
        assert 0.10 <= manifest["actual_fraction"] <= 0.50
        # DV should be LLOQ for censored rows, not 0.0
        blq_rows = result[result["BLQ_FLAG"] == 1]
        assert (blq_rows["DV"] == manifest["lloq"]).all()
        assert (blq_rows["CENS"] == 1).all()

    def test_inject_blq_preserves_dose_rows(self) -> None:
        """BLQ injection does not modify dose events."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.30,
            seed=42,
        )
        result, _ = apply_perturbation(df, recipe)

        dose_mask = result["EVID"] == 1
        assert (result.loc[dose_mask, "BLQ_FLAG"] == 0).all()


@pytest.mark.integration
class TestPerturbationOutliers:
    """Test outlier injection perturbation."""

    def test_inject_outliers_5pct(self) -> None:
        """5% outlier injection modifies the right number of observations."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_OUTLIERS,
            outlier_fraction=0.05,
            outlier_magnitude=5.0,
            seed=44,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert manifest["perturbation"] == "inject_outliers"
        assert manifest["n_outliers"] > 0
        assert manifest["magnitude"] == 5.0
        # Row count should be preserved
        assert len(result) == len(df)


@pytest.mark.integration
class TestPerturbationSparsify:
    """Test sparsification perturbation."""

    def test_sparsify_to_3_obs(self) -> None:
        """Sparsification reduces to target obs/subject."""
        df = _make_synthetic_pk_data(n_subjects=10, n_obs_per=8)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.SPARSIFY,
            target_obs_per_subject=3,
            seed=45,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert manifest["perturbation"] == "sparsify"
        assert manifest["final_observations"] < manifest["original_observations"]

        # Check each subject has at most 3 observations
        obs_result = result[result["EVID"] == 0]
        for _, group in obs_result.groupby("NMID"):
            assert len(group) <= 3

    def test_sparsify_preserves_doses(self) -> None:
        """Sparsification never removes dosing events."""
        df = _make_synthetic_pk_data(n_subjects=10)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.SPARSIFY,
            target_obs_per_subject=2,
            seed=45,
        )
        result, _ = apply_perturbation(df, recipe)

        original_doses = len(df[df["EVID"] == 1])
        result_doses = len(result[result["EVID"] == 1])
        assert result_doses == original_doses


@pytest.mark.integration
class TestPerturbationNullCovariates:
    """Test null covariate injection."""

    def test_add_5_null_covariates(self) -> None:
        """Adding 5 null covariates produces 5 new columns."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_NULL_COVARIATES,
            null_covariate_n=5,
            seed=46,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert manifest["n_covariates"] == 5
        for name in manifest["covariate_names"]:
            assert name in result.columns

    def test_null_covariates_constant_within_subject(self) -> None:
        """Null covariates are constant within each subject."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_NULL_COVARIATES,
            null_covariate_n=3,
            seed=46,
        )
        result, manifest = apply_perturbation(df, recipe)

        for cov_name in manifest["covariate_names"]:
            for _, group in result.groupby("NMID"):
                assert group[cov_name].nunique() == 1


@pytest.mark.integration
class TestPerturbationAbsorption:
    """Test absorption sample removal."""

    def test_remove_absorption_samples(self) -> None:
        """Removing absorption samples drops early timepoints."""
        df = _make_synthetic_pk_data()
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.REMOVE_ABSORPTION_SAMPLES,
            absorption_time_cutoff=2.0,
            seed=45,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert manifest["n_removed"] > 0
        obs = result[result["EVID"] == 0]
        assert (obs["TIME"] >= 2.0).all()


@pytest.mark.integration
class TestPerturbationChaining:
    """Test sequential application of multiple perturbations."""

    def test_blq_then_sparsify(self) -> None:
        """Applying BLQ then sparsification produces valid data."""
        df = _make_synthetic_pk_data()
        recipes = [
            PerturbationRecipe(
                perturbation_type=PerturbationType.INJECT_BLQ,
                blq_fraction=0.20,
                seed=42,
            ),
            PerturbationRecipe(
                perturbation_type=PerturbationType.SPARSIFY,
                target_obs_per_subject=4,
                seed=43,
            ),
        ]
        result, manifests = apply_perturbations(df, recipes)

        assert len(manifests) == 2
        assert manifests[0]["perturbation"] == "inject_blq"
        assert manifests[1]["perturbation"] == "sparsify"
        assert len(result) < len(df)


@pytest.mark.integration
class TestPerturbationProtocolPooling:
    """Test protocol pooling perturbation."""

    def test_basic_protocol_assignment(self) -> None:
        """Protocol pooling assigns STUDY_ID to all subjects."""
        df = _make_synthetic_pk_data(n_subjects=20)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_PROTOCOL_POOLING,
            n_protocols=3,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert "STUDY_ID" in result.columns
        assert manifest["perturbation"] == "add_protocol_pooling"
        assert manifest["n_protocols"] == 3
        # Every row should have a STUDY_ID
        assert result["STUDY_ID"].notna().all()
        # STUDY_ID values should be in [1, n_protocols]
        assert result["STUDY_ID"].min() >= 1
        assert result["STUDY_ID"].max() <= 3

    def test_protocol_pooling_constant_within_subject(self) -> None:
        """STUDY_ID is constant within each subject."""
        df = _make_synthetic_pk_data(n_subjects=15)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_PROTOCOL_POOLING,
            n_protocols=2,
            seed=42,
        )
        result, _ = apply_perturbation(df, recipe)

        for _, group in result.groupby("NMID"):
            assert group["STUDY_ID"].nunique() == 1

    def test_protocol_pooling_with_sampling_variation(self) -> None:
        """vary_sampling drops some observations per protocol."""
        df = _make_synthetic_pk_data(n_subjects=20)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_PROTOCOL_POOLING,
            n_protocols=2,
            vary_sampling=True,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)

        assert manifest["vary_sampling"] is True
        # Some observations should have been dropped
        assert len(result) <= len(df)


@pytest.mark.integration
class TestPerturbationRecipeValidation:
    """Test PerturbationRecipe cross-field validators."""

    def test_blq_requires_fraction(self) -> None:
        """BLQ perturbation without blq_fraction raises."""
        with pytest.raises(ValueError, match="blq_fraction required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.INJECT_BLQ,
            )

    def test_outlier_requires_fraction(self) -> None:
        """Outlier perturbation without outlier_fraction raises."""
        with pytest.raises(ValueError, match="outlier_fraction required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.INJECT_OUTLIERS,
            )

    def test_sparsify_requires_target(self) -> None:
        """Sparsify without target_obs_per_subject raises."""
        with pytest.raises(ValueError, match="target_obs_per_subject required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.SPARSIFY,
            )

    def test_absorption_requires_cutoff(self) -> None:
        """Absorption removal without cutoff raises."""
        with pytest.raises(ValueError, match="absorption_time_cutoff required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.REMOVE_ABSORPTION_SAMPLES,
            )

    def test_null_cov_requires_count_or_names(self) -> None:
        """Null covariates without count or names raises."""
        with pytest.raises(ValueError, match="null_covariate_n or"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.ADD_NULL_COVARIATES,
            )

    def test_blq_fraction_bounds(self) -> None:
        """blq_fraction > 1.0 raises validation error."""
        with pytest.raises(ValueError):
            PerturbationRecipe(
                perturbation_type=PerturbationType.INJECT_BLQ,
                blq_fraction=1.5,
            )

    def test_valid_recipe_passes(self) -> None:
        """A well-formed recipe passes validation."""
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.25,
            seed=42,
        )
        assert recipe.blq_fraction == 0.25
