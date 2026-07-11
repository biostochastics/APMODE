# SPDX-License-Identifier: GPL-2.0-or-later
"""Data splitting for cross-validation (PRD §4.3.2, ARCHITECTURE.md §3).

Subject-level splitting with stratification support.
Produces SplitManifest for the reproducibility bundle.

Split strategies:
  - subject_level: random subject-level train/test split
  - stratified: stratified by dose group, study, or other factor
  - regimen_level: leave-one-regimen-out (LORO) for Optimization lane
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apmode.bundle.models import SplitManifest, SubjectAssignment


def split_subjects(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float = 0.2,
    strategy: str = "subject_level",
    stratify_by: str | None = None,
) -> SplitManifest:
    """Split subjects into train/test folds.

    Args:
        df: Validated PK DataFrame with NMID column.
        seed: Random seed for reproducibility.
        test_fraction: Fraction of subjects for test set (0-1).
        strategy: Split strategy (subject_level, stratified, regimen_level).
        stratify_by: Column name for stratification (required if strategy=stratified).

    Returns:
        SplitManifest with per-subject fold assignments.

    Raises:
        ValueError: If test_fraction is not in (0, 1).
    """
    if not (0 < test_fraction < 1):
        msg = f"test_fraction must be in (0, 1), got {test_fraction}"
        raise ValueError(msg)
    valid_strategies = {"subject_level", "stratified", "regimen_level"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown split strategy {strategy!r}; expected {sorted(valid_strategies)}"
        )
    if strategy == "stratified":
        if stratify_by is None:
            raise ValueError("stratify_by is required when strategy='stratified'")
        if stratify_by not in df.columns:
            raise ValueError(f"stratify_by column {stratify_by!r} is not present")
    rng = np.random.default_rng(seed)
    subjects = sorted(df["NMID"].unique().tolist())
    n_subjects = len(subjects)
    if n_subjects < 2:
        raise ValueError("At least two subjects are required for a train/test split")
    n_test = min(n_subjects - 1, max(1, int(n_subjects * test_fraction)))

    if strategy == "stratified" and stratify_by is not None:
        assignments = _stratified_split(df, subjects, n_test, rng, stratify_by)
    elif strategy == "regimen_level":
        assignments = _regimen_level_split(df, subjects, rng)
    else:
        assignments = _random_split(subjects, n_test, rng)

    return SplitManifest(
        split_seed=seed,
        split_strategy=strategy,
        assignments=assignments,
    )


def k_fold_split(
    df: pd.DataFrame,
    seed: int,
    k: int = 5,
) -> list[SplitManifest]:
    """Generate K-fold cross-validation splits.

    Returns k SplitManifests, each with a different test fold.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(df["NMID"].unique().tolist())
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if k > len(subjects):
        raise ValueError(f"k={k} exceeds the number of subjects ({len(subjects)})")

    # Shuffle subjects
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    # Assign to folds
    fold_assignments: list[list[str]] = [[] for _ in range(k)]
    for i, subj in enumerate(shuffled):
        fold_assignments[i % k].append(str(subj))

    manifests: list[SplitManifest] = []
    for fold_idx in range(k):
        assignments: list[SubjectAssignment] = []
        for i, fold_subjects in enumerate(fold_assignments):
            fold_type = "test" if i == fold_idx else "train"
            for subj in fold_subjects:
                assignments.append(SubjectAssignment(subject_id=subj, fold=fold_type))

        manifests.append(
            SplitManifest(
                split_seed=seed,
                split_strategy="subject_level",
                assignments=assignments,
            )
        )

    return manifests


def loro_cv_splits(
    df: pd.DataFrame,
    seed: int,
    min_folds: int = 3,
) -> list[SplitManifest]:
    """Generate leave-one-regimen-out cross-validation folds.

    Groups subjects by regimen signature (modal dose amount from EVID==1),
    then generates K folds where K = number of unique regimen groups.
    Each fold holds out one regimen group as "test", rest as "train".

    Uses regimen signature (modal dose per subject) rather than total AMT
    to better capture true regimen identity.

    Args:
        df: Validated PK DataFrame with NMID, EVID, AMT columns.
        seed: Random seed (stored in manifest; folds are deterministic).
        min_folds: Minimum number of regimen groups required. Default 3.

    Returns:
        List of SplitManifest, one per regimen group (fold).

    Raises:
        ValueError: If fewer than min_folds unique regimen groups exist.
    """
    subjects = sorted(df["NMID"].unique().tolist())
    regimen_groups = _identify_regimen_groups(df)
    unique_group_labels = sorted(set(regimen_groups.values()))
    n_groups = len(unique_group_labels)

    if n_groups < min_folds:
        msg = (
            f"Insufficient regimen groups for LORO-CV: found {n_groups} "
            f"unique regimen groups, but min_folds={min_folds} required. "
            f"Groups: {unique_group_labels}"
        )
        raise ValueError(msg)

    manifests: list[SplitManifest] = []
    for holdout_group in unique_group_labels:
        assignments: list[SubjectAssignment] = []
        for subj in subjects:
            subj_group = regimen_groups.get(str(subj), regimen_groups.get(subj, "unknown"))
            fold = "test" if subj_group == holdout_group else "train"
            assignments.append(SubjectAssignment(subject_id=str(subj), fold=fold))
        manifests.append(
            SplitManifest(
                split_seed=seed,
                split_strategy="regimen_level",
                assignments=assignments,
            )
        )

    return manifests


def _identify_regimen_groups(df: pd.DataFrame) -> dict[str, str]:
    """Identify regimen group for each subject via modal dose amount.

    Uses the most frequent dose amount from EVID==1 records per subject
    as the regimen signature (more robust than total AMT).
    """
    doses = df[df["EVID"].isin([1, 4])].copy()
    if doses.empty:
        subjects = sorted(df["NMID"].unique().tolist())
        return {str(s): "no_dose" for s in subjects}

    per_subj_modal_dose: dict[str, float] = {}
    for subj_id, group in doses.groupby("NMID"):
        amt_values = group["AMT"].dropna()
        if len(amt_values) == 0:
            per_subj_modal_dose[str(subj_id)] = 0.0
        else:
            mode_val = amt_values.mode()
            per_subj_modal_dose[str(subj_id)] = float(mode_val.iloc[0])

    unique_doses = sorted(set(per_subj_modal_dose.values()))
    dose_to_label = {dose: f"{dose}mg" for dose in unique_doses}

    return {subj: dose_to_label[dose] for subj, dose in per_subj_modal_dose.items()}


# ---------------------------------------------------------------------------
# Internal split implementations
# ---------------------------------------------------------------------------


def _random_split(
    subjects: list[object],
    n_test: int,
    rng: np.random.Generator,
) -> list[SubjectAssignment]:
    """Simple random subject-level split."""
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    assignments: list[SubjectAssignment] = []
    for i, subj in enumerate(shuffled):
        fold = "test" if i < n_test else "train"
        assignments.append(SubjectAssignment(subject_id=str(subj), fold=fold))
    return assignments


def _stratified_split(
    df: pd.DataFrame,
    subjects: list[object],
    n_test: int,
    rng: np.random.Generator,
    stratify_by: str,
) -> list[SubjectAssignment]:
    """Stratified split: proportional test allocation per stratum."""
    # Get stratum for each subject
    subj_strata = df.groupby("NMID")[stratify_by].first()
    grouped: list[list[str]] = []
    for stratum in subj_strata.unique():
        stratum_subjs = [str(s) for s in subjects if str(subj_strata.get(s, None)) == str(stratum)]
        rng.shuffle(stratum_subjs)
        grouped.append(stratum_subjs)

    # Largest-remainder allocation hits the requested global test size
    # exactly. In particular, singleton strata do not each force one test
    # subject and consume the entire training set.
    ideals = [len(group) * n_test / len(subjects) for group in grouped]
    allocations = [int(np.floor(ideal)) for ideal in ideals]
    remainder_order = sorted(
        range(len(grouped)), key=lambda i: ideals[i] - allocations[i], reverse=True
    )
    remaining = n_test - sum(allocations)
    for idx in remainder_order:
        if remaining == 0:
            break
        if allocations[idx] < len(grouped[idx]):
            allocations[idx] += 1
            remaining -= 1

    test_subjects = {
        subject
        for group, allocation in zip(grouped, allocations, strict=True)
        for subject in group[:allocation]
    }

    assignments: list[SubjectAssignment] = []
    for subj in subjects:
        fold = "test" if str(subj) in test_subjects else "train"
        assignments.append(SubjectAssignment(subject_id=str(subj), fold=fold))
    return assignments


def _regimen_level_split(
    df: pd.DataFrame,
    subjects: list[object],
    rng: np.random.Generator,
) -> list[SubjectAssignment]:
    """Leave-one-regimen-out: group by dose level, hold one group out.

    For Optimization lane LORO-CV (PRD §3.3).
    """
    # Determine dose groups from dosing records
    doses = df[df["EVID"].isin([1, 4])].copy()
    if doses.empty:
        return _random_split(subjects, max(1, len(subjects) // 5), rng)

    # Per-subject total dose as proxy for regimen
    per_subj_dose = doses.groupby("NMID")["AMT"].sum()
    if per_subj_dose.nunique() < 2:
        return _random_split(subjects, max(1, min(len(subjects) - 1, len(subjects) // 5)), rng)

    # Bin into dose groups (quartiles)
    try:
        dose_groups = pd.qcut(per_subj_dose, q=4, labels=False, duplicates="drop")
    except ValueError:
        # Too few unique values for quartiles
        dose_groups = pd.cut(per_subj_dose, bins=2, labels=False)

    # Hold out the largest dose group (index = max group)
    max_group = dose_groups.max()
    test_subjs = set(str(s) for s in dose_groups[dose_groups == max_group].index)
    if not test_subjs or len(test_subjs) == len(subjects):
        return _random_split(subjects, max(1, min(len(subjects) - 1, len(subjects) // 5)), rng)

    assignments: list[SubjectAssignment] = []
    for subj in subjects:
        fold = "test" if str(subj) in test_subjs else "train"
        assignments.append(SubjectAssignment(subject_id=str(subj), fold=fold))
    return assignments
