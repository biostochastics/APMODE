# SPDX-License-Identifier: GPL-2.0-or-later
"""Data format adapters for backend-specific column naming (PRD S4.2.0).

Converts between the canonical PK schema (CanonicalPKSchema) and
backend-specific expectations (e.g., nlmixr2 uses 'ID' not 'NMID').
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]  # noqa: TC002 — runtime use

# nlmixr2 expects 'ID' for subject identifier; all other canonical columns
# (TIME, DV, MDV, EVID, AMT, CMT, RATE, DUR) are already nlmixr2-compatible.
_CANONICAL_TO_NLMIXR2: dict[str, str] = {
    "NMID": "ID",
}


def to_nlmixr2_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical PK DataFrame to nlmixr2-expected column names.

    Renames NMID -> ID. All other canonical columns are already compatible.
    Covariate columns are passed through unchanged.

    Raises:
        ValueError: If both NMID and ID columns exist (would create duplicates).
    """
    if "NMID" in df.columns and "ID" in df.columns:
        msg = "DataFrame has both 'NMID' and 'ID' columns; rename would create duplicates"
        raise ValueError(msg)
    return df.rename(columns=_CANONICAL_TO_NLMIXR2)
