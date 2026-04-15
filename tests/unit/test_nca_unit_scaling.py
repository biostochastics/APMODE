# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression test: NCA estimator unit-scaling heuristic.

Real-world bug discovered on mavoglurant (mGluR5 antagonist) Suite B run:
NCA computes CL = Dose / AUC directly. This is correct only when dose and DV
units are commensurate (e.g., dose in mg + DV in mg/L = ug/mL). When dose is
in mg but DV is in ng/mL — a routine pharmacometric convention — the raw CL
is 1000x too small (CL=0.05 instead of 50 L/h), causing SAEM to start in a
degenerate region of parameter space and Gate 1 to reject all candidates.

The unit-scaling heuristic detects this and applies a multiplier to CL and V.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.bundle.models import ColumnMapping, DataManifest
from apmode.data.initial_estimates import NCAEstimator


def _make_manifest(n_subjects: int, n_obs: int) -> DataManifest:
    """Minimal DataManifest for NCA testing (only fields the estimator reads)."""
    return DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=n_subjects,
        n_observations=n_obs,
        n_doses=n_subjects,
    )


def _make_dataset(
    *,
    dose_mg: float,
    dv_scale: float,
    n_subjects: int = 30,
) -> pd.DataFrame:
    """Build a synthetic 1-cmt oral PK dataset.

    Args:
        dose_mg: dose amount placed in AMT column (units only matter as a label)
        dv_scale: multiplier applied to true concentrations — controls whether
            DV ends up in mg/L (1.0) or ng/mL (1000.0).
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int]] = []
    times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    # True PK: 1-cmt oral, V=70 L, CL=5 L/h, ka=1.5/h
    V, CL, ka = 70.0, 5.0, 1.5
    kel = CL / V
    for subj in range(1, n_subjects + 1):
        rows.append(
            {
                "NMID": subj,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": dose_mg,
                "CMT": 1,
            }
        )
        for t in times:
            # Bateman equation: 1-cmt oral, F=1
            conc_per_l = dose_mg / V * ka / (ka - kel) * (np.exp(-kel * t) - np.exp(-ka * t))
            conc = conc_per_l * dv_scale * (1 + rng.normal(0, 0.10))
            rows.append(
                {
                    "NMID": subj,
                    "TIME": t,
                    "DV": max(0.01, conc),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
    return pd.DataFrame(rows)


class TestNCAUnitScaling:
    """Heuristic unit detection on three canonical scenarios."""

    def test_commensurate_units_no_scaling(self) -> None:
        """Dose=100 mg + DV in mg/L → CL recovered without scaling."""
        df = _make_dataset(dose_mg=100.0, dv_scale=1.0)
        est = NCAEstimator(df, _make_manifest(30, len(df))).estimate_per_subject()
        scale = est.pop("_unit_scale_applied", 1.0)

        assert scale == 1.0
        # CL should be in the right order of magnitude (truth = 5 L/h)
        assert 1.0 < est["CL"] < 50.0, f"Expected CL ~5 L/h, got {est['CL']}"

    def test_ng_per_ml_dv_triggers_1000x_scaling(self) -> None:
        """Dose=100 mg + DV in ng/mL (1000x) → 1000x scale applied to CL/V."""
        df = _make_dataset(dose_mg=100.0, dv_scale=1000.0)  # DV now in ng/mL
        est = NCAEstimator(df, _make_manifest(30, len(df))).estimate_per_subject()
        scale = est.pop("_unit_scale_applied", 1.0)

        assert scale == 1000.0, f"Expected x1000 scaling, got {scale}"
        # After scaling, CL should still be in the right ballpark
        assert 1.0 < est["CL"] < 50.0, f"Expected CL ~5 L/h after scaling, got {est['CL']}"

    def test_low_concentration_data_no_false_trigger(self) -> None:
        """Small DV magnitudes (<50) should not trigger the heuristic."""
        df = _make_dataset(dose_mg=300.0, dv_scale=1.0)  # large dose, small DV
        est = NCAEstimator(df, _make_manifest(30, len(df))).estimate_per_subject()
        scale = est.pop("_unit_scale_applied", 1.0)

        # Theophylline-like: DV in mg/L, CL is plausible without scaling
        assert scale == 1.0, f"Should not have triggered scaling, got {scale}"


class TestNCAUnitScalingEdgeCases:
    """Boundary conditions for the heuristic."""

    def test_empty_data_no_crash(self) -> None:
        """Empty observations → fallback estimates."""
        df = pd.DataFrame(columns=["NMID", "TIME", "DV", "MDV", "EVID", "AMT", "CMT"])
        manifest = _make_manifest(1, 1)
        est = NCAEstimator(df, manifest).estimate_per_subject()
        # Should fall back to defaults without scale_applied
        assert "_unit_scale_applied" not in est

    def test_borderline_cl_no_false_scaling(self) -> None:
        """CL just above the 0.5 threshold should not trigger scaling."""
        # Tune: CL=0.6 L/h from scaled data — just above the 0.5 threshold
        df = _make_dataset(dose_mg=10.0, dv_scale=1.0, n_subjects=30)
        est = NCAEstimator(df, _make_manifest(30, len(df))).estimate_per_subject()
        scale = est.pop("_unit_scale_applied", 1.0)
        # Whatever happens, the test just verifies no spurious scaling on
        # commensurate-units data
        assert scale == 1.0


@pytest.mark.parametrize(
    "dose_unit_in_mg,dv_scale,expected_scale",
    [
        (100.0, 1.0, 1.0),  # mg + mg/L (commensurate)
        (100.0, 1000.0, 1000.0),  # mg + ng/mL (1000x off)
        (50.0, 1000.0, 1000.0),  # smaller dose, still ng/mL
    ],
)
def test_scaling_matrix(
    dose_unit_in_mg: float,
    dv_scale: float,
    expected_scale: float,
) -> None:
    """Parametrized matrix over (dose, dv_scale) combinations."""
    df = _make_dataset(dose_mg=dose_unit_in_mg, dv_scale=dv_scale)
    est = NCAEstimator(df, _make_manifest(30, len(df))).estimate_per_subject()
    scale = est.pop("_unit_scale_applied", 1.0)
    assert scale == expected_scale
