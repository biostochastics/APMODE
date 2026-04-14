# SPDX-License-Identifier: GPL-2.0-or-later
"""Pandera DataFrameModel for canonical PK data schema (PRD §4.2.0).

All ingested data (NONMEM CSV, nlmixr2 eventTable, CDISC ADaM) normalize to
this canonical schema. Validation uses lazy=True to surface all violations.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pandera as pa
from pandera.typing import Series  # noqa: TC002 — runtime use in field annotations


class CanonicalPKSchema(pa.DataFrameModel):
    """Canonical internal PK data representation.

    Required columns: NMID, TIME, DV, MDV, EVID, AMT, CMT.
    Optional columns: RATE, DUR, BLQ_FLAG, LLOQ, OCCASION, STUDY_ID,
                      OBS_TYPE, plus covariates.
    """

    NMID: Series[int] = pa.Field(description="Subject identifier")
    TIME: Series[float] = pa.Field(ge=0.0, description="Time relative to first dose")
    DV: Series[float] = pa.Field(description="Dependent variable (observation value)")
    MDV: Series[int] = pa.Field(isin=[0, 1], description="Missing DV flag")
    EVID: Series[int] = pa.Field(
        isin=[0, 1, 2, 3, 4],
        description="Event ID (0=obs, 1=dose, 2=other, 3=reset, 4=reset+dose)",
    )
    AMT: Series[float] = pa.Field(ge=0.0, description="Dose amount")
    CMT: Series[int] = pa.Field(ge=1, description="Compartment number")

    # Optional columns — validated only when present
    RATE: Series[float] | None = pa.Field(ge=0.0, nullable=True)
    DUR: Series[float] | None = pa.Field(ge=0.0, nullable=True)
    ADDL: Series[int] | None = pa.Field(ge=0, description="Number of additional doses")
    II: Series[float] | None = pa.Field(ge=0.0, nullable=True, description="Inter-dose interval")
    SS: Series[int] | None = pa.Field(
        isin=[0, 1, 2], description="Steady-state flag (0=none, 1=SS, 2=SS+superposition)"
    )
    BLQ_FLAG: Series[int] | None = pa.Field(isin=[0, 1])
    LLOQ: Series[float] | None = pa.Field(ge=0.0, nullable=True)
    OCCASION: Series[int] | None = pa.Field(ge=0)
    STUDY_ID: Series[str] | None = pa.Field()
    OBS_TYPE: Series[str] | None = pa.Field(
        description="Observation type (e.g., parent, metabolite)"
    )

    # Cross-column checks (dataframe-level)
    @pa.dataframe_check
    def dose_amt_positive_when_evid_1(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When EVID=1 (dose), AMT must be > 0."""
        return ~((df["EVID"] == 1) & (df["AMT"] <= 0))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def obs_amt_zero_when_evid_0(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When EVID=0 (observation), AMT should be 0."""
        return ~((df["EVID"] == 0) & (df["AMT"] != 0))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def addl_requires_ii(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When ADDL > 0, II must be > 0."""
        if "ADDL" not in df.columns or "II" not in df.columns:
            return pd.Series(True, index=df.index)  # type: ignore[no-any-return]
        addl = df["ADDL"].fillna(0)
        ii = df["II"].fillna(0.0)
        return ~((addl > 0) & (ii <= 0))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def ss_requires_ii_and_dose(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When SS in {1, 2}, II must be > 0 and EVID must be 1 (dose)."""
        if "SS" not in df.columns:
            return pd.Series(True, index=df.index)  # type: ignore[no-any-return]
        ss = df["SS"].fillna(0)
        ii = df["II"].fillna(0.0) if "II" in df.columns else pd.Series(0.0, index=df.index)
        has_ss = ss.isin([1, 2])
        return ~(has_ss & ((ii <= 0) | (~df["EVID"].isin([1, 4]))))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def addl_only_on_dose_rows(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """ADDL must be 0 on non-dose rows (EVID not in {1, 4})."""
        if "ADDL" not in df.columns:
            return pd.Series(True, index=df.index)  # type: ignore[no-any-return]
        addl = df["ADDL"].fillna(0)
        return ~((addl > 0) & (~df["EVID"].isin([1, 4])))  # type: ignore[no-any-return]

    class Config:
        strict = False  # allow extra covariate columns
        coerce = True
