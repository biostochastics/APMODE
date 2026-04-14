# SPDX-License-Identifier: GPL-2.0-or-later
"""Property-based tests for LORO-CV fold generation (Phase 3 P3.B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from apmode.data.splitter import loro_cv_splits


@st.composite
def pk_data_with_regimens(
    draw: st.DrawFn,
    min_groups: int = 3,
    max_groups: int = 6,
) -> pd.DataFrame:
    """Generate synthetic PK data with variable number of dose groups."""
    n_groups = draw(st.integers(min_value=min_groups, max_value=max_groups))
    n_per_group = draw(st.integers(min_value=2, max_value=8))
    doses = sorted(
        draw(
            st.lists(
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=n_groups,
                max_size=n_groups,
                unique=True,
            )
        )
    )

    rows: list[dict[str, object]] = []
    subject_id = 1
    for dose in doses:
        for _ in range(n_per_group):
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
            for t in [1.0, 4.0, 12.0]:
                rows.append(
                    {
                        "NMID": subject_id,
                        "TIME": t,
                        "DV": float(np.random.default_rng(subject_id).lognormal(0, 1)),
                        "EVID": 0,
                        "AMT": 0.0,
                        "MDV": 0,
                    }
                )
            subject_id += 1

    return pd.DataFrame(rows)


class TestLoroFoldProperties:
    """Hypothesis property tests for LORO fold generation."""

    @given(data=pk_data_with_regimens())
    @settings(max_examples=30, deadline=5000)
    def test_every_subject_test_exactly_once(self, data: pd.DataFrame) -> None:
        """Each subject appears as test in exactly one fold across all folds."""
        folds = loro_cv_splits(data, seed=42, min_folds=3)
        subjects = sorted(data["NMID"].unique().tolist())

        test_count: dict[int, int] = {s: 0 for s in subjects}
        for fold in folds:
            for a in fold.assignments:
                if a.fold == "test":
                    test_count[int(a.subject_id)] += 1

        for subj, count in test_count.items():
            assert count == 1, f"Subject {subj} was test in {count} folds (expected 1)"

    @given(data=pk_data_with_regimens())
    @settings(max_examples=30, deadline=5000)
    def test_every_fold_has_both_train_and_test(self, data: pd.DataFrame) -> None:
        """Every fold must have at least one train and one test subject."""
        folds = loro_cv_splits(data, seed=42, min_folds=3)
        for i, fold in enumerate(folds):
            fold_types = {a.fold for a in fold.assignments}
            assert "train" in fold_types, f"Fold {i} has no train subjects"
            assert "test" in fold_types, f"Fold {i} has no test subjects"

    @given(data=pk_data_with_regimens())
    @settings(max_examples=30, deadline=5000)
    def test_all_subjects_assigned_in_every_fold(self, data: pd.DataFrame) -> None:
        """Every fold must include all subjects (as either train or test)."""
        folds = loro_cv_splits(data, seed=42, min_folds=3)
        subjects = set(str(s) for s in data["NMID"].unique())

        for i, fold in enumerate(folds):
            fold_subjects = {a.subject_id for a in fold.assignments}
            assert fold_subjects == subjects, (
                f"Fold {i}: missing subjects {subjects - fold_subjects}"
            )

    @given(data=pk_data_with_regimens())
    @settings(max_examples=30, deadline=5000)
    def test_folds_are_deterministic(self, data: pd.DataFrame) -> None:
        """Same data → same folds regardless of seed (regimen groups are deterministic)."""
        folds_a = loro_cv_splits(data, seed=1, min_folds=3)
        folds_b = loro_cv_splits(data, seed=999, min_folds=3)

        assert len(folds_a) == len(folds_b)
        test_sets_a = {
            frozenset(a.subject_id for a in f.assignments if a.fold == "test") for f in folds_a
        }
        test_sets_b = {
            frozenset(a.subject_id for a in f.assignments if a.fold == "test") for f in folds_b
        }
        assert test_sets_a == test_sets_b

    @given(
        n_groups=st.integers(min_value=1, max_value=2),
        n_per_group=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=10, deadline=5000)
    def test_insufficient_groups_raises(self, n_groups: int, n_per_group: int) -> None:
        """Fewer than min_folds groups must raise ValueError."""
        rows: list[dict[str, object]] = []
        sid = 1
        for g in range(n_groups):
            dose = float((g + 1) * 10)
            for _ in range(n_per_group):
                rows.append(
                    {
                        "NMID": sid,
                        "TIME": 0.0,
                        "DV": 0.0,
                        "EVID": 1,
                        "AMT": dose,
                        "MDV": 1,
                    }
                )
                rows.append(
                    {
                        "NMID": sid,
                        "TIME": 1.0,
                        "DV": 1.0,
                        "EVID": 0,
                        "AMT": 0.0,
                        "MDV": 0,
                    }
                )
                sid += 1
        df = pd.DataFrame(rows)

        import pytest

        with pytest.raises(ValueError, match=r"[Ii]nsufficient|[Rr]egimen"):
            loro_cv_splits(df, seed=42, min_folds=3)
