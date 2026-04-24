# SPDX-License-Identifier: GPL-2.0-or-later
"""Posterior_summary.parquet emitter (plan Task 11)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from apmode.bundle.emitter import BundleEmitter

_CANONICAL_COLUMNS = ("param", "rhat", "ess_bulk", "ess_tail", "mean", "sd", "q05", "q50", "q95")


def _summary_frame(n_params: int = 3) -> pd.DataFrame:
    base_rhat = [1.0, 1.01, 1.002, 1.004, 1.003]
    base_mean = [2.0, 30.0, 0.6, 4.5, 1.1]
    return pd.DataFrame(
        {
            "param": [f"theta_{i}" for i in range(n_params)],
            "rhat": base_rhat[:n_params],
            "ess_bulk": [4000, 3200, 5000, 4600, 3800][:n_params],
            "ess_tail": [3800, 3000, 4800, 4400, 3600][:n_params],
            "mean": base_mean[:n_params],
            "sd": [m * 0.05 for m in base_mean[:n_params]],
            "q05": [m * 0.9 for m in base_mean[:n_params]],
            "q50": base_mean[:n_params],
            "q95": [m * 1.1 for m in base_mean[:n_params]],
        }
    )


def test_write_posterior_summary_round_trip(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    df = _summary_frame(n_params=3)
    path = emitter.write_posterior_summary(candidate_id="cand_001", summary_df=df)
    tbl = pq.read_table(path)
    assert set(tbl.column_names) == set(_CANONICAL_COLUMNS)
    assert tbl.num_rows == 3
    assert path.name == "cand_001_posterior_summary.parquet"


def test_write_posterior_summary_rejects_missing_columns(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    df = _summary_frame(n_params=2).drop(columns=["rhat"])
    with pytest.raises(ValueError, match="rhat"):
        emitter.write_posterior_summary(candidate_id="cand_bad", summary_df=df)


def test_write_posterior_summary_rejects_invalid_candidate_id(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    df = _summary_frame(n_params=2)
    with pytest.raises(ValueError, match="candidate_id"):
        emitter.write_posterior_summary(candidate_id="../escape", summary_df=df)
