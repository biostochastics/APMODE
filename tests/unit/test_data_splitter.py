# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for data splitting (PRD §4.3.2)."""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from apmode.bundle.models import SplitManifest
from apmode.data.splitter import k_fold_split, split_subjects


def _make_pk_df(n_subjects: int = 10, n_obs: int = 5) -> pd.DataFrame:
    """Create a simple PK DataFrame for testing."""
    rows: list[dict[str, object]] = []
    for subj in range(1, n_subjects + 1):
        # Dose
        rows.append(
            {
                "NMID": subj,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0 * subj,
                "CMT": 1,
            }
        )
        # Observations
        for i in range(n_obs):
            rows.append(
                {
                    "NMID": subj,
                    "TIME": float(i + 1),
                    "DV": float(10 - i),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
    return pd.DataFrame(rows)


class TestSplitSubjects:
    """Subject-level splitting."""

    def test_returns_split_manifest(self) -> None:
        df = _make_pk_df()
        result = split_subjects(df, seed=42)
        assert isinstance(result, SplitManifest)

    def test_all_subjects_assigned(self) -> None:
        df = _make_pk_df(n_subjects=10)
        result = split_subjects(df, seed=42)
        assigned_ids = {a.subject_id for a in result.assignments}
        expected_ids = {str(i) for i in range(1, 11)}
        assert assigned_ids == expected_ids

    def test_test_fraction(self) -> None:
        df = _make_pk_df(n_subjects=20)
        result = split_subjects(df, seed=42, test_fraction=0.25)
        test_count = sum(1 for a in result.assignments if a.fold == "test")
        train_count = sum(1 for a in result.assignments if a.fold == "train")
        assert test_count == 5
        assert train_count == 15

    def test_deterministic_with_seed(self) -> None:
        df = _make_pk_df()
        r1 = split_subjects(df, seed=42)
        r2 = split_subjects(df, seed=42)
        ids1 = [(a.subject_id, a.fold) for a in r1.assignments]
        ids2 = [(a.subject_id, a.fold) for a in r2.assignments]
        assert ids1 == ids2

    def test_different_seeds_differ(self) -> None:
        df = _make_pk_df(n_subjects=20)
        r1 = split_subjects(df, seed=42)
        r2 = split_subjects(df, seed=99)
        folds1 = [a.fold for a in sorted(r1.assignments, key=lambda a: a.subject_id)]
        folds2 = [a.fold for a in sorted(r2.assignments, key=lambda a: a.subject_id)]
        # Very unlikely to be identical with different seeds
        assert folds1 != folds2

    def test_at_least_one_test_subject(self) -> None:
        df = _make_pk_df(n_subjects=3)
        result = split_subjects(df, seed=42, test_fraction=0.1)
        test_count = sum(1 for a in result.assignments if a.fold == "test")
        assert test_count >= 1

    def test_split_strategy_recorded(self) -> None:
        df = _make_pk_df()
        result = split_subjects(df, seed=42, strategy="subject_level")
        assert result.split_strategy == "subject_level"

    def test_seed_recorded(self) -> None:
        df = _make_pk_df()
        result = split_subjects(df, seed=12345)
        assert result.split_seed == 12345


class TestKFoldSplit:
    """K-fold cross-validation."""

    def test_returns_k_manifests(self) -> None:
        df = _make_pk_df(n_subjects=10)
        manifests = k_fold_split(df, seed=42, k=5)
        assert len(manifests) == 5

    def test_each_fold_has_test_subjects(self) -> None:
        df = _make_pk_df(n_subjects=10)
        manifests = k_fold_split(df, seed=42, k=5)
        for m in manifests:
            test_count = sum(1 for a in m.assignments if a.fold == "test")
            assert test_count >= 1

    def test_all_subjects_appear_in_test_exactly_once(self) -> None:
        df = _make_pk_df(n_subjects=10)
        manifests = k_fold_split(df, seed=42, k=5)
        test_ids: list[str] = []
        for m in manifests:
            for a in m.assignments:
                if a.fold == "test":
                    test_ids.append(a.subject_id)
        # Each subject should be in test exactly once
        assert len(test_ids) == 10
        assert len(set(test_ids)) == 10

    def test_deterministic(self) -> None:
        df = _make_pk_df(n_subjects=10)
        m1 = k_fold_split(df, seed=42, k=5)
        m2 = k_fold_split(df, seed=42, k=5)
        for a, b in zip(m1, m2, strict=True):
            ids_a = [(x.subject_id, x.fold) for x in a.assignments]
            ids_b = [(x.subject_id, x.fold) for x in b.assignments]
            assert ids_a == ids_b

    def test_json_roundtrip(self) -> None:
        df = _make_pk_df(n_subjects=10)
        manifests = k_fold_split(df, seed=42, k=3)
        for m in manifests:
            json_str = m.model_dump_json()
            roundtrip = SplitManifest.model_validate_json(json_str)
            assert len(roundtrip.assignments) == len(m.assignments)
