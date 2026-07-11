# SPDX-License-Identifier: GPL-2.0-or-later
"""Pandera DataFrameModel for canonical PK data schema (PRD §4.2.0).

All ingested data (NONMEM CSV, nlmixr2 eventTable, CDISC ADaM) normalize to
this canonical schema. Validation uses lazy=True to surface all violations.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series  # noqa: TC002 — runtime use in field annotations


class CanonicalPKSchema(pa.DataFrameModel):
    """Canonical internal PK data representation.

    Required columns: NMID, TIME, DV, MDV, EVID, AMT, CMT.
    Optional columns: RATE, DUR, BLQ_FLAG, LLOQ, OCCASION, STUDY_ID,
                      OBS_TYPE, plus covariates.
    """

    NMID: Series[int] = pa.Field(description="Subject identifier")
    TIME: Series[float] = pa.Field(ge=0.0, description="Time relative to first dose")
    # nullable=True: the standard NONMEM convention for a genuinely missing
    # sample (dropout, unscheduled-visit miss) is DV=NaN with MDV=1 — not to
    # be confused with BLQ-censoring, which carries a real numeric value
    # (typically the LLOQ) with BLQ=1, MDV=0 (see Suite A scenario A20).
    # dv_present_when_mdv_0 below still fails closed on a NaN DV for an
    # actual observation row (MDV=0).
    DV: Series[float] = pa.Field(
        nullable=True, description="Dependent variable (observation value)"
    )
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
    # Nullable integer-like NONMEM fields are represented as floats at the
    # pandas boundary because numpy int64 cannot hold NaN. Cross-column logic
    # treats missing as zero and ``optional_integer_fields_are_integral``
    # preserves integer semantics for non-missing values.
    ADDL: Series[float] | None = pa.Field(
        ge=0, nullable=True, description="Number of additional doses"
    )
    II: Series[float] | None = pa.Field(ge=0.0, nullable=True, description="Inter-dose interval")
    # NONMEM steady-state flag. Standard values: 0=none, 1=SS,
    # 2=SS+superposition. 99 is an ACOP-style "not applicable" sentinel
    # that the nlmixr2data ACOP-2016 simulated datasets use on
    # observation / non-dose rows; downstream checks treat 99 as
    # equivalent to 0 (not-SS). Without 99 in the allowlist the
    # canonical-PK validator rejects every row of Oral_1CPT /
    # Bolus_1CPT / Infusion_1CPT (and the MM variants), which makes
    # those fixtures unusable in Phase-1.
    SS: Series[float] | None = pa.Field(
        isin=[0, 1, 2, 99],
        nullable=True,
        description=(
            "Steady-state flag (0=none, 1=SS, 2=SS+superposition, "
            "99=not applicable / ACOP-style sentinel — treated as 0)"
        ),
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
    def dose_amt_positive_when_evid_is_dose(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """Dose and reset+dose events must carry a positive amount."""
        return ~(df["EVID"].isin([1, 4]) & (df["AMT"] <= 0))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def obs_amt_zero_when_evid_0(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When EVID=0 (observation), AMT should be 0."""
        return ~((df["EVID"] == 0) & (df["AMT"] != 0))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def dv_present_when_mdv_0(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When MDV=0 (DV is meaningful), DV must not be NaN.

        DV's own nullability only permits the MDV=1 "genuinely missing
        sample" convention; this check keeps that permissive without
        silently accepting a NaN on a row the fit is actually meant to use.
        """
        return ~((df["MDV"] == 0) & df["DV"].isna())  # type: ignore[no-any-return]

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

    @pa.dataframe_check
    def infusion_rate_duration_match_amount(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        """When RATE and DUR are both supplied, their product must equal AMT."""
        if "RATE" not in df.columns or "DUR" not in df.columns:
            return pd.Series(True, index=df.index)  # type: ignore[no-any-return]
        rate = df["RATE"].fillna(0.0)
        dur = df["DUR"].fillna(0.0)
        both = df["EVID"].isin([1, 4]) & (rate > 0) & (dur > 0)
        expected = rate * dur
        tolerance = 1e-8 * df["AMT"].abs().clip(lower=1.0)
        return ~(both & ((expected - df["AMT"]).abs() > tolerance))  # type: ignore[no-any-return]

    @pa.dataframe_check
    def optional_integer_fields_are_integral(cls, df: pd.DataFrame) -> Series[bool]:  # type: ignore[misc]
        valid = pd.Series(True, index=df.index)
        for column in ("ADDL", "SS"):
            if column in df.columns:
                values = df[column]
                valid &= values.isna() | (values % 1 == 0)
        return valid  # type: ignore[no-any-return]

    class Config:
        strict = False  # allow extra covariate columns
        coerce = True
