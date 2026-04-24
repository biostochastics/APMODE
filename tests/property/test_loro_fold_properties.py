# SPDX-License-Identifier: GPL-2.0-or-later
"""Property tests for LORO-CV fold generation (PRD §4.3.1, plan P3.B-6).

Complements ``test_loro_property.py`` with invariants specific to fold
arithmetic: one fold per regimen when eligible, and assignment stability
under DataFrame row permutation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from apmode.data.splitter import loro_cv_splits


def _synth_data(n_subjects: int, n_regimens: int, seed: int) -> pd.DataFrame:
    """Build a synthetic canonical PK dataset with ``n_regimens`` modal-dose groups."""
    rng = np.random.default_rng(seed)
    amts = rng.choice([100.0, 200.0, 400.0, 800.0], size=n_regimens, replace=False)
    rows: list[dict[str, float]] = []
    for sid in range(1, n_subjects + 1):
        amt = float(amts[(sid - 1) % n_regimens])
        rows.append({"NMID": sid, "TIME": 0.0, "DV": 0.0, "EVID": 1, "AMT": amt, "MDV": 1})
        for t in (0.5, 1.0, 2.0, 4.0, 8.0):
            rows.append(
                {
                    "NMID": sid,
                    "TIME": t,
                    "DV": float(rng.lognormal(0, 0.2)),
                    "EVID": 0,
                    "AMT": 0.0,
                    "MDV": 0,
                }
            )
    return pd.DataFrame(rows)


@given(
    n_subjects=st.integers(min_value=8, max_value=40),
    n_regimens=st.integers(min_value=3, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=25, deadline=None)
def test_fold_count_matches_regimens_when_eligible(
    n_subjects: int, n_regimens: int, seed: int
) -> None:
    """When min_folds ≤ n_regimens, ``loro_cv_splits`` emits exactly one fold per regimen."""
    df = _synth_data(n_subjects, n_regimens, seed)
    folds = loro_cv_splits(df, seed=seed, min_folds=3)
    assert len(folds) == n_regimens


@given(
    n_subjects=st.integers(min_value=8, max_value=40),
    n_regimens=st.integers(min_value=3, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=25, deadline=None)
def test_regimen_signature_stable_under_row_permutation(
    n_subjects: int, n_regimens: int, seed: int
) -> None:
    """Shuffling rows must not change the per-fold held-out subject set."""
    df = _synth_data(n_subjects, n_regimens, seed)
    rng = np.random.default_rng(seed + 1)
    shuffled = df.sample(frac=1.0, random_state=int(rng.integers(0, 1 << 30))).reset_index(
        drop=True
    )
    folds_a = loro_cv_splits(df, seed=seed, min_folds=3)
    folds_b = loro_cv_splits(shuffled, seed=seed, min_folds=3)

    def _test_sets(ms: list) -> set:  # type: ignore[type-arg]
        return {frozenset(a.subject_id for a in m.assignments if a.fold == "test") for m in ms}

    assert _test_sets(folds_a) == _test_sets(folds_b)
