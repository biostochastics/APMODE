# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Pandera canonical PK data schema (PRD §4.2.0)."""

import pandas as pd
import pandera as pa
import pytest

from apmode.data.schema import CanonicalPKSchema


class TestCanonicalPKSchema:
    """Validate the canonical schema accepts well-formed PK data."""

    def _minimal_observation_row(self, **overrides: object) -> dict:
        base = {
            "NMID": 1,
            "TIME": 1.0,
            "DV": 5.5,
            "MDV": 0,
            "EVID": 0,
            "AMT": 0.0,
            "CMT": 1,
        }
        base.update(overrides)
        return base

    def _minimal_dose_row(self, **overrides: object) -> dict:
        base = {
            "NMID": 1,
            "TIME": 0.0,
            "DV": 0.0,
            "MDV": 1,
            "EVID": 1,
            "AMT": 100.0,
            "CMT": 1,
        }
        base.update(overrides)
        return base

    def test_valid_observation(self) -> None:
        df = pd.DataFrame([self._minimal_observation_row()])
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert len(validated) == 1

    def test_valid_dose(self) -> None:
        df = pd.DataFrame([self._minimal_dose_row()])
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert len(validated) == 1

    def test_valid_mixed_dataset(self) -> None:
        df = pd.DataFrame(
            [
                self._minimal_dose_row(NMID=1, TIME=0.0),
                self._minimal_observation_row(NMID=1, TIME=1.0),
                self._minimal_observation_row(NMID=1, TIME=2.0),
                self._minimal_dose_row(NMID=2, TIME=0.0),
                self._minimal_observation_row(NMID=2, TIME=0.5),
            ]
        )
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert len(validated) == 5

    def test_negative_time_rejected(self) -> None:
        df = pd.DataFrame([self._minimal_observation_row(TIME=-1.0)])
        with pytest.raises(pa.errors.SchemaErrors):
            CanonicalPKSchema.validate(df, lazy=True)

    def test_invalid_evid_rejected(self) -> None:
        df = pd.DataFrame([self._minimal_observation_row(EVID=99)])
        with pytest.raises(pa.errors.SchemaErrors):
            CanonicalPKSchema.validate(df, lazy=True)

    def test_invalid_mdv_rejected(self) -> None:
        df = pd.DataFrame([self._minimal_observation_row(MDV=5)])
        with pytest.raises(pa.errors.SchemaErrors):
            CanonicalPKSchema.validate(df, lazy=True)

    def test_optional_blq_fields(self) -> None:
        df = pd.DataFrame(
            [
                self._minimal_observation_row(BLQ_FLAG=0, LLOQ=0.1),
            ]
        )
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert "BLQ_FLAG" in validated.columns

    def test_optional_occasion_field(self) -> None:
        df = pd.DataFrame(
            [
                self._minimal_observation_row(OCCASION=1),
            ]
        )
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert "OCCASION" in validated.columns

    def test_optional_rate_dur_fields(self) -> None:
        df = pd.DataFrame(
            [
                self._minimal_dose_row(RATE=50.0, DUR=2.0),
            ]
        )
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert "RATE" in validated.columns
