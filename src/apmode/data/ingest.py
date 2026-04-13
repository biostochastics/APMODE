# SPDX-License-Identifier: GPL-2.0-or-later
"""NONMEM CSV data ingestion (PRD S4.2.0).

Reads NONMEM-style CSV, validates against CanonicalPKSchema,
computes SHA-256, detects covariates, and produces DataManifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path  # noqa: TC003 — used at runtime in function signature

import pandas as pd  # type: ignore[import-untyped]

from apmode.bundle.models import ColumnMapping, CovariateMetadata, DataManifest
from apmode.data.schema import CanonicalPKSchema

# Canonical column names that are NOT covariates
_CANONICAL_COLUMNS = frozenset(
    {
        "NMID",
        "TIME",
        "DV",
        "MDV",
        "EVID",
        "AMT",
        "CMT",
        "RATE",
        "DUR",
        "BLQ_FLAG",
        "LLOQ",
        "OCCASION",
        "STUDY_ID",
        "OBS_TYPE",
    }
)

_REQUIRED_COLUMNS = frozenset({"NMID", "TIME", "DV", "MDV", "EVID", "AMT", "CMT"})


def ingest_nonmem_csv(
    path: Path,
    column_mapping: dict[str, str] | None = None,
) -> tuple[DataManifest, pd.DataFrame]:
    """Ingest a NONMEM-style CSV file.

    Args:
        path: Path to the CSV file.
        column_mapping: Optional dict mapping source column names to canonical names.
                       If None, assumes columns already use canonical names.

    Returns:
        Tuple of (DataManifest, validated DataFrame).

    Raises:
        ValueError: If required columns are missing or validation fails.
    """
    raw: pd.DataFrame = pd.read_csv(path)

    # Apply column mapping if provided
    if column_mapping:
        raw = raw.rename(columns=column_mapping)

    # Check required columns
    missing = _REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    # Validate with Pandera
    df: pd.DataFrame = CanonicalPKSchema.validate(raw, lazy=True)

    # Compute SHA-256 of the raw file
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    # Detect covariates (columns not in canonical set)
    covariates: list[CovariateMetadata] = []
    for col in raw.columns:
        if col not in _CANONICAL_COLUMNS:
            cov_type: str = (
                "categorical" if pd.api.types.is_string_dtype(raw[col]) else "continuous"
            )
            covariates.append(CovariateMetadata(name=col, type=cov_type))

    # Build column mapping
    cols = set(raw.columns)
    col_map = ColumnMapping(
        subject_id="NMID",
        time="TIME",
        dv="DV",
        evid="EVID",
        amt="AMT",
        mdv="MDV",
        cmt="CMT",
        rate="RATE" if "RATE" in cols else None,
        dur="DUR" if "DUR" in cols else None,
        blq_flag="BLQ_FLAG" if "BLQ_FLAG" in cols else None,
        lloq="LLOQ" if "LLOQ" in cols else None,
        occasion="OCCASION" if "OCCASION" in cols else None,
        study_id="STUDY_ID" if "STUDY_ID" in cols else None,
    )

    manifest = DataManifest(
        data_sha256=sha256,
        ingestion_format="nonmem_csv",
        column_mapping=col_map,
        n_subjects=int(raw["NMID"].nunique()),
        n_observations=int((raw["EVID"] == 0).sum()),
        n_doses=int((raw["EVID"] == 1).sum()),
        covariates=covariates,
    )

    return manifest, df
