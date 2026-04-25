# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for ``apmode.bayes.harness._build_stan_data`` guardrails.

Covers defence-in-depth validations:

* ``data_path`` must point to an existing regular file (CWE-22 guard).
* Non-positive ``DV`` rows with ``MDV=0`` must raise ``ValueError`` rather
  than being silently dropped (lognormal likelihood incompatibility).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apmode.bayes.harness import _build_stan_data


def _base_request(data_path: Path) -> dict[str, object]:
    return {
        "data_path": str(data_path),
        "spec": {
            "absorption": {"type": "FirstOrder"},
            "distribution": {"type": "OneCmt"},
            "elimination": {"type": "Linear"},
        },
        "candidate_id": "cand001",
        "compiled_stan_code": "data{}",
        "output_draws_path": str(data_path.parent / "draws.parquet"),
        "sampler_config": {
            "chains": 2,
            "warmup": 10,
            "sampling": 10,
            "adapt_delta": 0.8,
            "max_treedepth": 8,
            "seed": 1,
        },
    }


def _valid_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {"NMID": 1, "TIME": 0.0, "DV": 0.0, "MDV": 1, "EVID": 1, "AMT": 100.0, "CMT": 1},
            {"NMID": 1, "TIME": 1.0, "DV": 5.0, "MDV": 0, "EVID": 0, "AMT": 0.0, "CMT": 1},
            {"NMID": 2, "TIME": 0.0, "DV": 0.0, "MDV": 1, "EVID": 1, "AMT": 100.0, "CMT": 1},
            {"NMID": 2, "TIME": 1.0, "DV": 4.5, "MDV": 0, "EVID": 0, "AMT": 0.0, "CMT": 1},
        ]
    ).to_csv(path, index=False)


def test_missing_data_path_raises(tmp_path: Path) -> None:
    req = _base_request(tmp_path / "nonexistent.csv")
    with pytest.raises(ValueError, match="regular file"):
        _build_stan_data(req)


def test_non_string_data_path_raises(tmp_path: Path) -> None:
    req = _base_request(tmp_path / "unused.csv")
    req["data_path"] = 42  # type: ignore[assignment]
    with pytest.raises(ValueError, match="data_path"):
        _build_stan_data(req)


def test_directory_data_path_raises(tmp_path: Path) -> None:
    req = _base_request(tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        _build_stan_data(req)


def test_nonpositive_dv_observation_escalates_to_valueerror(tmp_path: Path) -> None:
    """DV<=0 with MDV=0 is incompatible with the lognormal likelihood and
    must raise rather than silently shrink the dataset.
    """
    csv = tmp_path / "nonpos.csv"
    pd.DataFrame(
        [
            {"NMID": 1, "TIME": 0.0, "DV": 0.0, "MDV": 1, "EVID": 1, "AMT": 100.0, "CMT": 1},
            {"NMID": 1, "TIME": 0.25, "DV": 0.0, "MDV": 0, "EVID": 0, "AMT": 0.0, "CMT": 1},
            {"NMID": 1, "TIME": 1.0, "DV": 5.0, "MDV": 0, "EVID": 0, "AMT": 0.0, "CMT": 1},
        ]
    ).to_csv(csv, index=False)
    req = _base_request(csv)
    with pytest.raises(ValueError, match="non-positive DV"):
        _build_stan_data(req)


def test_valid_data_round_trip(tmp_path: Path) -> None:
    """Well-formed input still builds the Stan data dict."""
    csv = tmp_path / "ok.csv"
    _valid_csv(csv)
    stan_data = _build_stan_data(_base_request(csv))
    assert stan_data["N"] == 2
    assert stan_data["N_subjects"] == 2
    assert stan_data["N_events"] == 2
