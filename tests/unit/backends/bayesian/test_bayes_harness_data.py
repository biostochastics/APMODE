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

from apmode.bayes.harness import _build_stan_data, _compute_eta_shrinkage, _extract_iiv_names


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


def test_observations_are_grouped_by_subject_and_sorted_by_time(tmp_path: Path) -> None:
    csv = tmp_path / "interleaved.csv"
    pd.DataFrame(
        [
            {"ID": 1, "EVID": 0, "TIME": 2.0, "DV": -1.0},
            {"ID": 2, "EVID": 0, "TIME": 2.0, "DV": 2.0},
            {"ID": 1, "EVID": 0, "TIME": 1.0, "DV": 1.0},
            {"ID": 2, "EVID": 0, "TIME": 1.0, "DV": -2.0},
        ]
    ).to_csv(csv, index=False)
    request = _base_request(csv)
    request["spec"] = {
        "observation": {"type": "Additive", "sigma_add": 1.0},
        "covariates": [],
    }
    data = _build_stan_data(request)
    assert data["subject"] == [1, 1, 2, 2]
    assert data["time"] == [1.0, 2.0, 1.0, 2.0]
    assert data["dv"] == [1.0, -1.0, -2.0, 2.0]


def test_blq_cens_row_is_retained_even_when_mdv_is_one(tmp_path: Path) -> None:
    csv = tmp_path / "blq.csv"
    pd.DataFrame(
        [
            {"ID": 1, "EVID": 0, "MDV": 1, "CENS": 1, "TIME": 1.0, "DV": 0.0},
            {"ID": 1, "EVID": 0, "MDV": 0, "CENS": 0, "TIME": 2.0, "DV": 2.0},
        ]
    ).to_csv(csv, index=False)
    request = _base_request(csv)
    request["spec"] = {
        "observation": {
            "type": "BLQ_M3",
            "loq_value": 0.5,
            "error_model": "proportional",
        },
        "covariates": [],
    }
    data = _build_stan_data(request)
    assert data["N"] == 2
    assert data["cens"] == [1, 0]


def test_categorical_covariate_uses_declared_reference(tmp_path: Path) -> None:
    csv = tmp_path / "categorical.csv"
    pd.DataFrame(
        [
            {"ID": 1, "EVID": 0, "TIME": 1.0, "DV": 1.0, "SEX": "F"},
            {"ID": 2, "EVID": 0, "TIME": 1.0, "DV": 2.0, "SEX": "M"},
        ]
    ).to_csv(csv, index=False)
    request = _base_request(csv)
    request["spec"] = {
        "observation": {"type": "Proportional", "sigma_prop": 0.1},
        "covariates": [
            {
                "type": "CovariateLink",
                "param": "CL",
                "covariate": "SEX",
                "form": "categorical",
                "reference": "F",
            }
        ],
    }
    data = _build_stan_data(request)
    assert data["SEX"] == [0.0, 1.0]


def test_eta_shrinkage_uses_iiv_declaration_order() -> None:
    import numpy as np

    class _Fit:
        def stan_variables(self) -> dict[str, object]:
            return {"eta_raw": object(), "omega_CL": object()}

        def stan_variable(self, name: str) -> object:
            if name == "eta_raw":
                return np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
            return np.array([0.5, 0.5])

    spec = {"variability": [{"type": "IIV", "params": ["CL"], "structure": "diagonal"}]}
    assert _extract_iiv_names(spec) == ["CL"]
    assert set(_compute_eta_shrinkage(_Fit(), ["CL"])) == {"CL"}
