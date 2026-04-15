# SPDX-License-Identifier: GPL-2.0-or-later
"""Data format adapters for backend-specific column naming (PRD S4.2.0).

Converts between the canonical PK schema (CanonicalPKSchema) and
backend-specific expectations (e.g., nlmixr2 uses 'ID' not 'NMID').
"""

from __future__ import annotations

import pandas as pd
import structlog

from apmode.data.categorical_encoding import auto_remap_binary_columns

_logger = structlog.get_logger(__name__)

# nlmixr2 expects 'ID' for subject identifier; all other canonical columns
# (TIME, DV, MDV, EVID, AMT, CMT, RATE, DUR) are already nlmixr2-compatible.
_CANONICAL_TO_NLMIXR2: dict[str, str] = {
    "NMID": "ID",
}

# DVID values that nlmixr2's single-endpoint models treat as the primary
# PK concentration. Mirrors ``apmode.data.profiler._PK_DVIDS`` but is kept
# in-sync manually because ``adapters`` must not import from ``profiler``
# (profiler depends on adapters transitively via bundle artifacts).
_PK_DVID_ALLOWLIST: frozenset[str] = frozenset({"1", "conc", "concentration", "cp"})

# Canonical PK columns — never remap these even if their dtype looks
# categorical. Everything else is treated as a candidate covariate.
_RESERVED_COLUMNS: frozenset[str] = frozenset(
    {
        "NMID",
        "ID",
        "TIME",
        "DV",
        "AMT",
        "EVID",
        "CMT",
        "MDV",
        "BLQ_FLAG",
        "DVID",
        "RATE",
        "DUR",
        "LLOQ",
        "STUDY_ID",
    }
)


def to_nlmixr2_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical PK DataFrame to nlmixr2-ready form.

    Performs three transformations that nlmixr2's SAEM engine requires
    but that the canonical APMODE schema does not enforce:

    1. ``NMID`` → ``ID`` rename.
    2. Non-PK observation rows (``DVID`` outside the PK allowlist — e.g.
       ``pca`` rows mixed into warfarin) are filtered. When the resulting
       ``DVID`` column has ≤1 unique non-null value, the column is dropped
       entirely so single-endpoint models pass nlmixr2's
       "mis-match in nbr endpoints" check.
    3. String / 2-level categorical covariates (e.g. ``SEX="male"/"female"``)
       are remapped to integer 0/1 via
       :func:`apmode.data.categorical_encoding.auto_remap_binary_columns`.
       nlmixr2 cannot consume string-typed covariates.

    Raises:
        ValueError: If both NMID and ID columns exist (would create duplicates).
    """
    if "NMID" in df.columns and "ID" in df.columns:
        msg = "DataFrame has both 'NMID' and 'ID' columns; rename would create duplicates"
        raise ValueError(msg)

    out = df.rename(columns=_CANONICAL_TO_NLMIXR2).copy()

    if "DVID" in out.columns:
        evid = out["EVID"] if "EVID" in out.columns else pd.Series(0, index=out.index)
        obs_mask = evid == 0
        dvid_str = out["DVID"].astype(str).str.strip().str.lower()
        keep_mask = (~obs_mask) | dvid_str.isin(_PK_DVID_ALLOWLIST)
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            dropped_values = sorted({v for v in dvid_str[~keep_mask].unique() if v})
            _logger.info(
                "adapter.dropped_non_pk_dvid_rows",
                n_rows=n_dropped,
                dropped_dvid_values=dropped_values,
                allowlist=sorted(_PK_DVID_ALLOWLIST),
            )
            out = out.loc[keep_mask].reset_index(drop=True)
        remaining = out.loc[obs_mask.reindex(out.index, fill_value=False), "DVID"]
        unique_remaining = {str(v).strip().lower() for v in remaining.dropna().unique()}
        if len(unique_remaining) <= 1:
            out = out.drop(columns=["DVID"])

    from pandas.api.types import is_numeric_dtype

    to_remap = sorted(
        c for c in out.columns if c not in _RESERVED_COLUMNS and not is_numeric_dtype(out[c])
    )
    if to_remap:
        out, _hints = auto_remap_binary_columns(out, to_remap)

    return out
