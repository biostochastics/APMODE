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

    def test_ss_accepts_acop_99_sentinel(self) -> None:
        """ACOP-2016 simulated datasets (Oral_1CPT, Bolus_1CPT, ...) carry
        ``SS=99`` on observation / non-dose rows as a "not applicable"
        sentinel. Without ``99`` in the schema's allowlist the
        canonical-PK validator rejects every row of those fixtures and
        the Suite-C Phase-1 weekly run aborts the entire fixture before
        any per-fold scorecard is produced. The downstream
        ``ss_requires_ii_and_dose`` cross-check uses ``isin([1, 2])``
        so SS=99 is correctly treated as not-SS.
        """
        df = pd.DataFrame(
            [
                self._minimal_observation_row(SS=99),
                self._minimal_dose_row(SS=99),
                self._minimal_observation_row(NMID=2, TIME=1.0, SS=0),
                self._minimal_dose_row(NMID=2, SS=1, II=12.0),
            ]
        )
        validated = CanonicalPKSchema.validate(df, lazy=True)
        assert "SS" in validated.columns
        assert set(validated["SS"].tolist()) == {99, 0, 1}

    def test_ss_rejects_unknown_value(self) -> None:
        """Other non-canonical SS values (e.g. ``5``) are still rejected;
        the ``99`` allowlist entry is a deliberate sentinel, not a
        general "any int" relaxation.
        """
        df = pd.DataFrame([self._minimal_observation_row(SS=5)])
        with pytest.raises(pa.errors.SchemaErrors):
            CanonicalPKSchema.validate(df, lazy=True)
