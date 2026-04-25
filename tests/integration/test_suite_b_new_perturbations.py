# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the v0.6.x Suite B perturbation completions.

Covers the four PRD §10 stress-surface perturbations that previously
raised ``NotImplementedError`` (``scale_bsv_variances``,
``saturate_clearance``, ``tmdd``, ``flip_flop``) plus the new
``inject_covariate_missingness`` (PRD §5 conformance) and the two
correctness fixes on existing perturbations (``add_occasion_labels``
groupby, ``add_protocol_pooling`` post-drop manifest).

Pure data-side tests on synthetic NONMEM frames — no R subprocess,
fast enough to run on every PR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.benchmarks.models import PerturbationRecipe, PerturbationType
from apmode.benchmarks.perturbations import apply_perturbation


def _synthetic_pk_frame(
    n_subjects: int = 20,
    n_obs_per: int = 6,
    *,
    with_covariates: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Minimal NONMEM-style frame with one dose + N observations per subject."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0][:n_obs_per]
    for subj in range(1, n_subjects + 1):
        wt = float(rng.normal(70.0, 10.0))
        sex = "M" if subj % 2 == 0 else "F"
        rows.append(
            {
                "NMID": subj,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0,
                "CMT": 1,
                **({"WT": wt, "SEX": sex} if with_covariates else {}),
            }
        )
        for t in times:
            conc = 50.0 * np.exp(-0.1 * t) * (1.0 - np.exp(-1.5 * t))
            rows.append(
                {
                    "NMID": subj,
                    "TIME": float(t),
                    "DV": float(max(0.01, conc)),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                    **({"WT": wt, "SEX": sex} if with_covariates else {}),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# scale_bsv_variances
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScaleBSVVariances:
    def test_preserves_geomean_in_expectation(self) -> None:
        """log-multiplier mean is 0, so DV geomean is preserved at large N."""
        df = _synthetic_pk_frame(n_subjects=200, seed=123)
        original_geomean = float(np.exp(np.log(df.loc[df["EVID"] == 0, "DV"]).mean()))
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.SCALE_BSV_VARIANCES,
            bsv_scale_factor=0.5,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        new_geomean = float(np.exp(np.log(result.loc[result["EVID"] == 0, "DV"]).mean()))
        # 200 subjects, sigma=0.5: expected log-mean SE ≈ 0.5/sqrt(200) ≈ 0.035,
        # so geomean ratio is within ~10% of 1 with very high probability.
        assert 0.85 < new_geomean / original_geomean < 1.15
        assert manifest["perturbation"] == "scale_bsv_variances"
        assert manifest["bsv_scale_factor"] == pytest.approx(0.5)
        assert manifest["n_subjects"] == 200

    def test_inflates_between_subject_variance(self) -> None:
        """A larger bsv_scale_factor produces a wider per-subject DV distribution."""
        df = _synthetic_pk_frame(n_subjects=80, seed=7)

        def _per_subject_geomean_log_sd(frame: pd.DataFrame) -> float:
            obs = frame[frame["EVID"] == 0]
            per_subj = obs.groupby("NMID")["DV"].apply(lambda s: float(np.log(s).mean()))
            return float(per_subj.std(ddof=1))

        baseline = _per_subject_geomean_log_sd(df)
        small = apply_perturbation(
            df,
            PerturbationRecipe(
                perturbation_type=PerturbationType.SCALE_BSV_VARIANCES,
                bsv_scale_factor=0.2,
                seed=11,
            ),
        )[0]
        large = apply_perturbation(
            df,
            PerturbationRecipe(
                perturbation_type=PerturbationType.SCALE_BSV_VARIANCES,
                bsv_scale_factor=0.8,
                seed=11,
            ),
        )[0]
        assert _per_subject_geomean_log_sd(large) > _per_subject_geomean_log_sd(small) > baseline

    def test_validator_rejects_missing_factor(self) -> None:
        with pytest.raises(ValueError, match="bsv_scale_factor required"):
            PerturbationRecipe(perturbation_type=PerturbationType.SCALE_BSV_VARIANCES)


# ---------------------------------------------------------------------------
# saturate_clearance
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSaturateClearance:
    def test_high_dv_inflated_low_dv_largely_unchanged(self) -> None:
        df = _synthetic_pk_frame(n_subjects=10, seed=1)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.SATURATE_CLEARANCE,
            saturation_km=10.0,
            saturation_vmax=10_000.0,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        obs_orig = df.loc[df["EVID"] == 0, "DV"].to_numpy()
        obs_new = result.loc[result["EVID"] == 0, "DV"].to_numpy()
        # Each new value is >= original (factor 1 + DV/Km >= 1).
        assert np.all(obs_new >= obs_orig - 1e-9)
        # High DV is inflated more than low DV (in absolute terms).
        diffs = obs_new - obs_orig
        assert diffs[obs_orig.argmax()] > diffs[obs_orig.argmin()]
        assert manifest["perturbation"] == "saturate_clearance"

    def test_vmax_clip_caps_perturbed_dv(self) -> None:
        df = _synthetic_pk_frame(n_subjects=5, seed=2)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.SATURATE_CLEARANCE,
            saturation_km=1.0,
            saturation_vmax=20.0,  # tight cap
            seed=42,
        )
        result, _ = apply_perturbation(df, recipe)
        obs_new = result.loc[result["EVID"] == 0, "DV"]
        assert obs_new.max() <= 20.0 + 1e-9

    def test_validator_rejects_missing_params(self) -> None:
        with pytest.raises(ValueError, match="saturation_km and saturation_vmax required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.SATURATE_CLEARANCE,
                saturation_km=1.0,
            )


# ---------------------------------------------------------------------------
# tmdd
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTMDD:
    def test_low_dv_depressed_high_dv_largely_unchanged(self) -> None:
        df = _synthetic_pk_frame(n_subjects=10, seed=3)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.TMDD,
            tmdd_kss=2.0,
            tmdd_r0=1.0,
            seed=42,
        )
        result, _ = apply_perturbation(df, recipe)
        obs_orig = df.loc[df["EVID"] == 0, "DV"].to_numpy()
        obs_new = result.loc[result["EVID"] == 0, "DV"].to_numpy()
        # Every value is bounded above by the original.
        assert np.all(obs_new <= obs_orig + 1e-9)
        # No negatives.
        assert np.all(obs_new >= 0.0)
        # Low-DV samples lose a *larger fraction* of their value than high-DV samples.
        ratio = np.where(obs_orig > 0, obs_new / obs_orig, 1.0)
        low_indices = np.argsort(obs_orig)[:5]
        high_indices = np.argsort(obs_orig)[-5:]
        assert ratio[low_indices].mean() < ratio[high_indices].mean()

    def test_validator_rejects_missing_params(self) -> None:
        with pytest.raises(ValueError, match="tmdd_kss and tmdd_r0 required"):
            PerturbationRecipe(perturbation_type=PerturbationType.TMDD, tmdd_r0=1.0)


# ---------------------------------------------------------------------------
# flip_flop
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFlipFlop:
    def test_early_phase_unchanged_late_phase_stretched(self) -> None:
        df = _synthetic_pk_frame(n_subjects=5, seed=4)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.FLIP_FLOP,
            flip_flop_ka=2.0,  # delay = 0.5h
            flip_flop_ke_ratio=0.5,  # stretch terminal phase
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        obs_orig = df.loc[df["EVID"] == 0]
        obs_new = result.loc[result["EVID"] == 0]
        # Times before delay (t <= 0.5h) are within numerical tolerance.
        early = obs_orig["TIME"] <= 0.5
        np.testing.assert_allclose(
            obs_new.loc[early, "DV"].to_numpy(),
            obs_orig.loc[early, "DV"].to_numpy(),
            rtol=1e-9,
        )
        # ke_ratio=0.5 < 1 stretches the terminal phase: DV at t > delay
        # should be *higher* than original (slower decay).
        late = obs_orig["TIME"] > 1.0
        assert (obs_new.loc[late, "DV"].to_numpy() > obs_orig.loc[late, "DV"].to_numpy()).all()
        assert manifest["post_absorption_threshold_h"] == pytest.approx(0.5)

    def test_validator_rejects_missing_params(self) -> None:
        with pytest.raises(ValueError, match="flip_flop_ka and flip_flop_ke_ratio required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.FLIP_FLOP,
                flip_flop_ka=1.0,
            )


# ---------------------------------------------------------------------------
# inject_covariate_missingness
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInjectCovariateMissingness:
    def test_drops_specified_fraction_at_subject_level(self) -> None:
        df = _synthetic_pk_frame(n_subjects=40, with_covariates=True, seed=5)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_COVARIATE_MISSINGNESS,
            covariate_missingness_fraction=0.25,
            covariate_missingness_columns=["WT"],
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        # Exactly 10 subjects out of 40 (25%) lose WT for *all* their rows.
        wt_missing_per_subject = result.groupby("NMID")["WT"].apply(lambda s: bool(s.isna().all()))
        assert int(wt_missing_per_subject.sum()) == 10
        assert manifest["n_subjects_dropped_per_column"] == 10
        # Subject-level: rows for any given subject either all have WT or all NaN.
        per_subj = result.groupby("NMID")["WT"].apply(lambda s: int(s.isna().nunique()))
        assert (per_subj == 1).all()

    def test_canonical_columns_excluded_when_no_explicit_list(self) -> None:
        df = _synthetic_pk_frame(n_subjects=10, with_covariates=True, seed=6)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_COVARIATE_MISSINGNESS,
            covariate_missingness_fraction=0.5,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        # NONMEM canonical cols (NMID/TIME/DV/...) must never lose values.
        for col in ("NMID", "TIME", "DV", "EVID", "AMT", "MDV"):
            assert result[col].notna().all(), f"canonical column {col} got NaN'd"
        # WT/SEX are constant within subject and should be in the candidate set.
        assert "WT" in manifest["covariate_columns"]
        assert "SEX" in manifest["covariate_columns"]

    def test_validator_rejects_missing_fraction(self) -> None:
        with pytest.raises(ValueError, match="covariate_missingness_fraction required"):
            PerturbationRecipe(
                perturbation_type=PerturbationType.INJECT_COVARIATE_MISSINGNESS,
            )


# ---------------------------------------------------------------------------
# Existing-perturbation correctness fixes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAddOccasionLabelsCorrectness:
    """The fix replaced the .name accident with a proper temp-column groupby."""

    def test_occasion_increments_per_dose(self) -> None:
        df = _synthetic_pk_frame(n_subjects=3, seed=8)
        # Inject a second dose row per subject at TIME=24
        extra_doses = df[df["EVID"] == 1].copy()
        extra_doses["TIME"] = 24.0
        df = pd.concat([df, extra_doses], ignore_index=True)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_OCCASION_LABELS,
            seed=42,
        )
        result, manifest = apply_perturbation(df, recipe)
        assert "OCCASION" in result.columns
        # Each subject should reach OCCASION=2 after the second dose.
        for _, sub in result.groupby("NMID"):
            assert int(sub["OCCASION"].max()) == 2
        assert manifest["n_occasions"] == 2

    def test_occasion_labels_no_dose_flag_column_left_behind(self) -> None:
        """Regression guard: the temp `_DOSE_FLAG` column must be dropped."""
        df = _synthetic_pk_frame(n_subjects=2, seed=9)
        result, _ = apply_perturbation(
            df,
            PerturbationRecipe(
                perturbation_type=PerturbationType.ADD_OCCASION_LABELS,
                seed=42,
            ),
        )
        assert "_DOSE_FLAG" not in result.columns


@pytest.mark.integration
class TestAddProtocolPoolingManifest:
    """The fix added a post-drop count alongside the assigned count."""

    def test_manifest_reports_both_assigned_and_post_drop(self) -> None:
        df = _synthetic_pk_frame(n_subjects=20, seed=10)
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_PROTOCOL_POOLING,
            n_protocols=4,
            vary_sampling=True,
            seed=42,
        )
        _, manifest = apply_perturbation(df, recipe)
        assert "subjects_per_protocol_assigned" in manifest
        assert "subjects_per_protocol_post_drop" in manifest
        # Block-randomised allocation gives exactly 5 per protocol pre-drop.
        for pid in (1, 2, 3, 4):
            assert manifest["subjects_per_protocol_assigned"][str(pid)] == 5
        # Post-drop counts must be ≤ assigned counts (rows can be removed).
        for pid in (1, 2, 3, 4):
            assert (
                manifest["subjects_per_protocol_post_drop"][str(pid)]
                <= manifest["subjects_per_protocol_assigned"][str(pid)]
            )
