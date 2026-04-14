# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for LORO-CV fold generation and metrics (Phase 3 P3.B).

Tests cover:
  - Regimen-signature-based fold generation
  - Fold completeness (no overlap, full coverage)
  - Determinism (same seed → same folds)
  - Minimum fold count enforcement
  - Edge cases (too few regimens, single-dose)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apmode.bundle.models import LOROCVResult, LOROFoldResult, LOROMetrics, SplitManifest
from apmode.data.splitter import loro_cv_splits


def _make_pk_data(
    n_per_group: int = 10,
    dose_levels: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic PK data with distinct dose groups."""
    if dose_levels is None:
        dose_levels = [10.0, 30.0, 60.0, 120.0]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    subject_id = 1

    for dose in dose_levels:
        for _ in range(n_per_group):
            # Dosing event
            rows.append(
                {
                    "NMID": subject_id,
                    "TIME": 0.0,
                    "DV": 0.0,
                    "EVID": 1,
                    "AMT": dose,
                    "MDV": 1,
                }
            )
            # Observation events at standard times
            for t in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]:
                rows.append(
                    {
                        "NMID": subject_id,
                        "TIME": t,
                        "DV": float(rng.lognormal(mean=np.log(dose * 0.5), sigma=0.3)),
                        "EVID": 0,
                        "AMT": 0.0,
                        "MDV": 0,
                    }
                )
            subject_id += 1

    return pd.DataFrame(rows)


class TestLoroCvSplits:
    """Tests for loro_cv_splits() fold generation."""

    def test_returns_k_folds_for_k_dose_groups(self) -> None:
        """One fold per regimen group."""
        df = _make_pk_data(dose_levels=[10.0, 30.0, 60.0, 120.0])
        folds = loro_cv_splits(df, seed=42)
        assert len(folds) == 4

    def test_each_fold_is_split_manifest(self) -> None:
        df = _make_pk_data()
        folds = loro_cv_splits(df, seed=42)
        for fold in folds:
            assert isinstance(fold, SplitManifest)
            assert fold.split_strategy == "regimen_level"

    def test_folds_are_exhaustive_no_overlap(self) -> None:
        """Each subject appears as test in exactly one fold."""
        df = _make_pk_data(n_per_group=5, dose_levels=[10.0, 30.0, 60.0])
        folds = loro_cv_splits(df, seed=42)

        subjects = sorted(df["NMID"].unique().tolist())
        test_counts: dict[int, int] = {s: 0 for s in subjects}

        for fold in folds:
            for a in fold.assignments:
                if a.fold == "test":
                    test_counts[int(a.subject_id)] += 1

        # Every subject is test exactly once
        assert all(c == 1 for c in test_counts.values()), f"Test counts: {test_counts}"

    def test_each_fold_has_train_and_test(self) -> None:
        """Every fold has both train and test subjects."""
        df = _make_pk_data()
        folds = loro_cv_splits(df, seed=42)
        for fold in folds:
            folds_set = {a.fold for a in fold.assignments}
            assert "train" in folds_set
            assert "test" in folds_set

    def test_deterministic_with_same_seed(self) -> None:
        """Same data + seed → identical folds."""
        df = _make_pk_data()
        folds_a = loro_cv_splits(df, seed=42)
        folds_b = loro_cv_splits(df, seed=42)

        for a, b in zip(folds_a, folds_b, strict=True):
            test_a = sorted(x.subject_id for x in a.assignments if x.fold == "test")
            test_b = sorted(x.subject_id for x in b.assignments if x.fold == "test")
            assert test_a == test_b

    def test_different_seed_same_folds(self) -> None:
        """LORO folds are determined by regimen groups, not by random splitting.
        Different seeds should yield the same fold structure (groups are deterministic)."""
        df = _make_pk_data()
        folds_a = loro_cv_splits(df, seed=42)
        folds_b = loro_cv_splits(df, seed=99)

        # Same number of folds
        assert len(folds_a) == len(folds_b)
        # Same test subjects per fold (order may differ across folds)
        test_sets_a = {
            frozenset(x.subject_id for x in f.assignments if x.fold == "test") for f in folds_a
        }
        test_sets_b = {
            frozenset(x.subject_id for x in f.assignments if x.fold == "test") for f in folds_b
        }
        assert test_sets_a == test_sets_b

    def test_insufficient_regimens_raises(self) -> None:
        """Fewer than 3 regimen groups → error."""
        df = _make_pk_data(dose_levels=[10.0, 30.0])
        with pytest.raises(ValueError, match=r"[Ii]nsufficient|[Rr]egimen"):
            loro_cv_splits(df, seed=42, min_folds=3)

    def test_single_dose_level_raises(self) -> None:
        """Only one dose level → insufficient regimens."""
        df = _make_pk_data(dose_levels=[10.0])
        with pytest.raises(ValueError, match=r"[Ii]nsufficient|[Rr]egimen"):
            loro_cv_splits(df, seed=42, min_folds=3)

    def test_min_folds_two_allows_two_groups(self) -> None:
        """With min_folds=2, two dose levels should work."""
        df = _make_pk_data(dose_levels=[10.0, 30.0])
        folds = loro_cv_splits(df, seed=42, min_folds=2)
        assert len(folds) == 2


class TestLoroModels:
    """Tests for LORO Pydantic models."""

    def test_loro_fold_result_validates(self) -> None:
        result = LOROFoldResult(
            fold_index=0,
            regimen_group="10.0mg",
            n_train_subjects=30,
            n_test_subjects=10,
            converged=True,
            test_npde_mean=0.05,
            test_npde_variance=1.02,
        )
        assert result.fold_index == 0
        assert result.regimen_group == "10.0mg"

    def test_loro_metrics_validates(self) -> None:
        metrics = LOROMetrics(
            n_folds=4,
            n_total_test_subjects=40,
            pooled_npde_mean=0.02,
            pooled_npde_variance=1.05,
            vpc_coverage_concordance=0.92,
            overall_pass=True,
        )
        assert metrics.overall_pass is True
        assert metrics.evaluation_mode == "fixed_parameter"

    def test_loro_cv_result_validates(self) -> None:
        metrics = LOROMetrics(
            n_folds=3,
            n_total_test_subjects=30,
            pooled_npde_mean=-0.01,
            pooled_npde_variance=0.98,
            vpc_coverage_concordance=0.88,
            overall_pass=True,
        )
        result = LOROCVResult(
            candidate_id="test_cand_001",
            metrics=metrics,
            fold_results=[],
            wall_time_seconds=120.0,
            regimen_groups=["10mg", "30mg", "60mg"],
            seed=42,
        )
        assert result.candidate_id == "test_cand_001"


class TestLoroEmitter:
    """Tests for LORO bundle emission."""

    def test_write_loro_cv_result(self, tmp_path: Path) -> None:
        import json

        from apmode.bundle.emitter import BundleEmitter

        emitter = BundleEmitter(tmp_path)
        emitter.initialize()

        metrics = LOROMetrics(
            n_folds=3,
            n_total_test_subjects=30,
            pooled_npde_mean=0.05,
            pooled_npde_variance=1.02,
            vpc_coverage_concordance=0.90,
            overall_pass=True,
        )
        result = LOROCVResult(
            candidate_id="cand_001",
            metrics=metrics,
            fold_results=[],
            wall_time_seconds=60.0,
        )

        path = emitter.write_loro_cv_result(result)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["candidate_id"] == "cand_001"
        assert data["metrics"]["pooled_npde_mean"] == 0.05
