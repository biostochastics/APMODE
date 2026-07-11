# SPDX-License-Identifier: GPL-2.0-or-later
"""In-memory posterior_draws.parquet emitter (plan Task 10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from apmode.bundle.emitter import BundleEmitter


def test_write_posterior_draws_thinned(tmp_path: Path) -> None:
    """Thinning every 10th draw yields N/10 rows per chain per param."""
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    draws = {
        "CL": np.random.default_rng(0).normal(size=(4, 1000)),
        "V": np.random.default_rng(1).normal(size=(4, 1000)),
    }
    path = emitter.write_posterior_draws(
        candidate_id="cand_001",
        draws_by_param=draws,
        thin_every=10,
    )
    tbl = pq.read_table(path)
    assert set(tbl.column_names) == {"chain", "iter", "param", "value"}
    # 1000 / 10 = 100 per chain per param; 4 chains x 2 params x 100 = 800
    assert tbl.num_rows == 800
    assert path.name == "cand_001_posterior_draws.parquet"


def test_write_posterior_draws_no_thin_default(tmp_path: Path) -> None:
    """thin_every=1 (default) preserves all draws."""
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    draws = {
        "CL": np.arange(12, dtype=float).reshape(3, 4),
    }
    path = emitter.write_posterior_draws(candidate_id="cand_042", draws_by_param=draws)
    tbl = pq.read_table(path)
    assert tbl.num_rows == 12  # 3 chains x 4 iters x 1 param


def test_write_posterior_draws_rejects_invalid_candidate_id(tmp_path: Path) -> None:
    """Candidate IDs are validated as path components (no '..' or '/')."""
    import pytest

    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    draws = {"CL": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="candidate_id"):
        emitter.write_posterior_draws(candidate_id="../escape", draws_by_param=draws)
