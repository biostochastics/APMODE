# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for multi-dose expansion and event table construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from apmode.data.dosing import (
    INFUSION_STOP_EVID,
    build_event_table,
    expand_addl,
    expand_infusion_events,
    extract_subject_events,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_dose_df() -> pd.DataFrame:
    """Single-dose theophylline-like data (no ADDL/II)."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 1, 1],
            "TIME": [0.0, 0.5, 1.0, 2.0, 4.0],
            "EVID": [1, 0, 0, 0, 0],
            "AMT": [320.0, 0.0, 0.0, 0.0, 0.0],
            "CMT": [1, 1, 1, 1, 1],
            "DV": [0.0, 5.0, 8.0, 6.0, 3.0],
            "MDV": [1, 0, 0, 0, 0],
        }
    )


def _multi_dose_addl_df() -> pd.DataFrame:
    """Multi-dose with ADDL/II: 320mg Q12H x3 additional doses."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 1, 1, 1],
            "TIME": [0.0, 6.0, 12.0, 18.0, 24.0, 36.0],
            "EVID": [1, 0, 0, 0, 0, 0],
            "AMT": [320.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "CMT": [1, 1, 1, 1, 1, 1],
            "DV": [0.0, 5.0, 8.0, 10.0, 9.0, 4.0],
            "MDV": [1, 0, 0, 0, 0, 0],
            "ADDL": [3, 0, 0, 0, 0, 0],
            "II": [12.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _two_subject_addl_df() -> pd.DataFrame:
    """Two subjects with different ADDL regimens."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 2, 2, 2],
            "TIME": [0.0, 6.0, 24.0, 0.0, 12.0, 48.0],
            "EVID": [1, 0, 0, 1, 0, 0],
            "AMT": [100.0, 0.0, 0.0, 200.0, 0.0, 0.0],
            "CMT": [1, 1, 1, 1, 1, 1],
            "DV": [0.0, 5.0, 2.0, 0.0, 8.0, 3.0],
            "MDV": [1, 0, 0, 1, 0, 0],
            "ADDL": [2, 0, 0, 1, 0, 0],
            "II": [8.0, 0.0, 0.0, 24.0, 0.0, 0.0],
        }
    )


def _infusion_df() -> pd.DataFrame:
    """Infusion data with RATE."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 1],
            "TIME": [0.0, 1.0, 3.0, 6.0],
            "EVID": [1, 0, 0, 0],
            "AMT": [500.0, 0.0, 0.0, 0.0],
            "CMT": [2, 2, 2, 2],
            "DV": [0.0, 100.0, 80.0, 40.0],
            "MDV": [1, 0, 0, 0],
            "RATE": [250.0, 0.0, 0.0, 0.0],
        }
    )


def _infusion_addl_df() -> pd.DataFrame:
    """Infusion with ADDL: 500mg infusion over 2h, Q12H x1."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 1, 1],
            "TIME": [0.0, 3.0, 6.0, 15.0, 24.0],
            "EVID": [1, 0, 0, 0, 0],
            "AMT": [500.0, 0.0, 0.0, 0.0, 0.0],
            "CMT": [2, 2, 2, 2, 2],
            "DV": [0.0, 80.0, 40.0, 90.0, 30.0],
            "MDV": [1, 0, 0, 0, 0],
            "RATE": [250.0, 0.0, 0.0, 0.0, 0.0],
            "ADDL": [1, 0, 0, 0, 0],
            "II": [12.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _reset_dose_df() -> pd.DataFrame:
    """Data with EVID=4 (reset+dose)."""
    return pd.DataFrame(
        {
            "NMID": [1, 1, 1, 1, 1],
            "TIME": [0.0, 6.0, 12.0, 18.0, 24.0],
            "EVID": [1, 0, 4, 0, 0],
            "AMT": [100.0, 0.0, 200.0, 0.0, 0.0],
            "CMT": [1, 1, 1, 1, 1],
            "DV": [0.0, 5.0, 0.0, 8.0, 4.0],
            "MDV": [1, 0, 1, 0, 0],
        }
    )


# ---------------------------------------------------------------------------
# expand_addl tests
# ---------------------------------------------------------------------------


class TestExpandAddl:
    """Tests for ADDL/II dose expansion."""

    def test_no_addl_columns_returns_copy(self) -> None:
        df = _single_dose_df()
        result = expand_addl(df)
        pd.testing.assert_frame_equal(result, df)

    def test_no_addl_values_returns_same_rows(self) -> None:
        df = _single_dose_df()
        df["ADDL"] = 0
        df["II"] = 0.0
        result = expand_addl(df)
        assert len(result) == len(df)

    def test_basic_addl_expansion(self) -> None:
        df = _multi_dose_addl_df()
        result = expand_addl(df)

        # Original: 1 dose + 5 obs = 6 rows
        # Expansion: 3 additional doses -> 6 + 3 = 9 rows
        assert len(result) == 9

        # Check dose rows
        doses = result[result["EVID"] == 1]
        assert len(doses) == 4  # 1 original + 3 expanded

        # Check dose times: 0, 12, 24, 36
        dose_times = sorted(doses["TIME"].values)
        np.testing.assert_array_almost_equal(dose_times, [0.0, 12.0, 24.0, 36.0])

        # All dose amounts should be 320
        assert all(doses["AMT"] == 320.0)

        # All expanded rows should have ADDL=0
        assert all(doses["ADDL"] == 0)

    def test_two_subject_expansion(self) -> None:
        df = _two_subject_addl_df()
        result = expand_addl(df)

        # Subject 1: 1 dose + 2 obs + 2 additional = 5
        # Subject 2: 1 dose + 2 obs + 1 additional = 4
        # Total: 9
        assert len(result) == 9

        s1 = result[result["NMID"] == 1]
        s1_doses = s1[s1["EVID"] == 1]
        assert len(s1_doses) == 3
        s1_times = sorted(s1_doses["TIME"].values)
        np.testing.assert_array_almost_equal(s1_times, [0.0, 8.0, 16.0])

        s2 = result[result["NMID"] == 2]
        s2_doses = s2[s2["EVID"] == 1]
        assert len(s2_doses) == 2
        s2_times = sorted(s2_doses["TIME"].values)
        np.testing.assert_array_almost_equal(s2_times, [0.0, 24.0])

    def test_sort_order_preserved(self) -> None:
        df = _multi_dose_addl_df()
        result = expand_addl(df)

        # Within each subject, times should be non-decreasing
        for _sid, sdf in result.groupby("NMID"):
            times = sdf["TIME"].values
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_evid_priority_within_same_time(self) -> None:
        """Dose events should come before observation events at same time."""
        df = _multi_dose_addl_df()
        result = expand_addl(df)

        # At t=12 and t=24, there should be both dose and obs events
        # Dose should come before obs (lower sort priority)
        at_12 = result[result["TIME"] == 12.0]
        if len(at_12) > 1:
            evids = at_12["EVID"].values
            # First row should be dose (1), second should be obs (0)
            dose_idx = np.where(evids == 1)[0]
            obs_idx = np.where(evids == 0)[0]
            if len(dose_idx) > 0 and len(obs_idx) > 0:
                assert dose_idx[0] < obs_idx[0]


# ---------------------------------------------------------------------------
# expand_infusion_events tests
# ---------------------------------------------------------------------------


class TestExpandInfusionEvents:
    """Tests for infusion stop event generation."""

    def test_no_infusion_no_change(self) -> None:
        df = _single_dose_df()
        result = expand_infusion_events(df)
        assert "_INF_RATE" in result.columns
        assert all(result["_INF_RATE"] == 0.0)

    def test_infusion_generates_stop_event(self) -> None:
        df = _infusion_df()
        result = expand_infusion_events(df)

        # Should have 1 extra row (infusion stop)
        assert len(result) == len(df) + 1

        # Stop event
        stops = result[result["EVID"] == INFUSION_STOP_EVID]
        assert len(stops) == 1
        # DUR = AMT / RATE = 500 / 250 = 2h
        assert stops.iloc[0]["TIME"] == pytest.approx(2.0)
        # Stop has negative rate
        assert stops.iloc[0]["_INF_RATE"] == pytest.approx(-250.0)

    def test_infusion_with_addl_generates_multiple_stops(self) -> None:
        df = _infusion_addl_df()
        # First expand ADDL, then infusion events
        expanded = expand_addl(df)
        result = expand_infusion_events(expanded)

        # Original dose at t=0 + expanded dose at t=12
        # Each generates a stop event: t=2 and t=14
        stops = result[result["EVID"] == INFUSION_STOP_EVID]
        assert len(stops) == 2
        stop_times = sorted(stops["TIME"].values)
        np.testing.assert_array_almost_equal(stop_times, [2.0, 14.0])


# ---------------------------------------------------------------------------
# build_event_table tests
# ---------------------------------------------------------------------------


class TestBuildEventTable:
    """Tests for complete event table construction."""

    def test_single_dose_passthrough(self) -> None:
        df = _single_dose_df()
        result = build_event_table(df)
        # No ADDL/II columns, no infusions -> should add _INF_RATE and return
        assert "_INF_RATE" in result.columns
        assert len(result) == len(df)

    def test_multi_dose_complete(self) -> None:
        df = _multi_dose_addl_df()
        result = build_event_table(df)

        # 9 rows from expand_addl + _INF_RATE column
        assert "_INF_RATE" in result.columns
        doses = result[result["EVID"] == 1]
        assert len(doses) == 4

    def test_infusion_addl_complete(self) -> None:
        df = _infusion_addl_df()
        result = build_event_table(df)

        # 2 doses (after ADDL expansion) + 4 obs + 2 infusion stops = 8
        doses = result[result["EVID"] == 1]
        assert len(doses) == 2
        stops = result[result["EVID"] == INFUSION_STOP_EVID]
        assert len(stops) == 2

    def test_reset_dose_preserved(self) -> None:
        df = _reset_dose_df()
        result = build_event_table(df)

        evid4 = result[result["EVID"] == 4]
        assert len(evid4) == 1
        assert evid4.iloc[0]["TIME"] == 12.0
        assert evid4.iloc[0]["AMT"] == 200.0


# ---------------------------------------------------------------------------
# extract_subject_events tests
# ---------------------------------------------------------------------------


class TestExtractSubjectEvents:
    """Tests for per-subject event array extraction."""

    def test_single_subject_extraction(self) -> None:
        df = _multi_dose_addl_df()
        event_table = build_event_table(df)
        events = extract_subject_events(event_table, subject_id=1)

        assert "event_times" in events
        assert "event_evids" in events
        assert "event_amts" in events
        assert "obs_indices" in events

        # 4 doses + 5 obs = 9 events total
        assert len(events["event_times"]) == 9
        assert len(events["obs_indices"]) == 5

    def test_obs_indices_correct(self) -> None:
        df = _multi_dose_addl_df()
        event_table = build_event_table(df)
        events = extract_subject_events(event_table, subject_id=1)

        for idx in events["obs_indices"]:
            assert events["event_evids"][idx] == 0

    def test_two_subjects_independent(self) -> None:
        df = _two_subject_addl_df()
        event_table = build_event_table(df)

        e1 = extract_subject_events(event_table, subject_id=1)
        e2 = extract_subject_events(event_table, subject_id=2)

        # Subject 1: 3 doses + 2 obs = 5
        assert len(e1["event_times"]) == 5
        # Subject 2: 2 doses + 2 obs = 4
        assert len(e2["event_times"]) == 4


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for ADDL/II/SS cross-column validation."""

    def test_addl_requires_ii(self) -> None:
        """ADDL > 0 with II = 0 should fail validation."""
        from apmode.data.schema import CanonicalPKSchema

        df = pd.DataFrame(
            {
                "NMID": [1],
                "TIME": [0.0],
                "DV": [0.0],
                "MDV": [1],
                "EVID": [1],
                "AMT": [100.0],
                "CMT": [1],
                "ADDL": [3],
                "II": [0.0],  # Invalid: ADDL>0 but II=0
            }
        )
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            CanonicalPKSchema.validate(df, lazy=True)

    def test_valid_addl_ii_passes(self) -> None:
        """Valid ADDL/II combination should pass."""
        from apmode.data.schema import CanonicalPKSchema

        df = pd.DataFrame(
            {
                "NMID": [1, 1],
                "TIME": [0.0, 6.0],
                "DV": [0.0, 5.0],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
                "ADDL": [3, 0],
                "II": [12.0, 0.0],
            }
        )
        result = CanonicalPKSchema.validate(df, lazy=True)
        assert len(result) == 2

    def test_addl_on_obs_row_fails(self) -> None:
        """ADDL > 0 on observation row (EVID=0) should fail."""
        from apmode.data.schema import CanonicalPKSchema

        df = pd.DataFrame(
            {
                "NMID": [1],
                "TIME": [1.0],
                "DV": [5.0],
                "MDV": [0],
                "EVID": [0],
                "AMT": [0.0],
                "CMT": [1],
                "ADDL": [2],  # Invalid on obs row
                "II": [12.0],
            }
        )
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            CanonicalPKSchema.validate(df, lazy=True)
