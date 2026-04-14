# SPDX-License-Identifier: GPL-2.0-or-later
"""Canonical name normalization for DSL parameters and data columns.

The DSL grammar uses mixed-case parameter names (ka, CL, V1, KD, etc.)
that are case-sensitive in the AST. This module provides a single source
of truth for resolving case-insensitive input to canonical forms, so that
LLM output, user input, and programmatic references all resolve correctly.

Column name normalization ensures the Python ingestion path matches the
R download path (toupper) for NONMEM-style data.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Structural parameter canonical names
# ---------------------------------------------------------------------------

# Every structural parameter that can appear in a DSLSpec, mapped from
# lowercase to canonical form. This is the authoritative list.
_PARAM_CANONICAL: dict[str, str] = {
    # Absorption
    "ka": "ka",
    "dur": "dur",
    "tlag": "tlag",
    "n": "n",
    "ktr": "ktr",
    "frac": "frac",
    # Distribution
    "v": "V",
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "q": "Q",
    "q2": "Q2",
    "q3": "Q3",
    "r0": "R0",
    "kon": "kon",
    "koff": "koff",
    "kint": "kint",
    "kd": "KD",
    # Elimination
    "cl": "CL",
    "vmax": "Vmax",
    "km": "Km",
    "kdecay": "kdecay",
}


def normalize_param_name(name: str) -> str:
    """Resolve a parameter name to its canonical form.

    Case-insensitive lookup against the known structural parameter set.
    Returns the canonical form if found, otherwise returns the original
    name unchanged (so that downstream validation can report the error).

    >>> normalize_param_name("CL")
    'CL'
    >>> normalize_param_name("cl")
    'CL'
    >>> normalize_param_name("Ka")
    'ka'
    >>> normalize_param_name("nonexistent")
    'nonexistent'
    """
    return _PARAM_CANONICAL.get(name.lower(), name)


def normalize_param_list(names: list[str]) -> list[str]:
    """Normalize a list of parameter names, preserving order."""
    return [normalize_param_name(n) for n in names]


# ---------------------------------------------------------------------------
# Data column name normalization
# ---------------------------------------------------------------------------

# Canonical NONMEM column names (all uppercase)
CANONICAL_COLUMNS: frozenset[str] = frozenset(
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
        "ADDL",
        "II",
        "SS",
        "BLQ_FLAG",
        "LLOQ",
        "OCCASION",
        "STUDY_ID",
        "OBS_TYPE",
    }
)

# Common aliases that should map to canonical names
_COLUMN_ALIASES: dict[str, str] = {
    "ID": "NMID",
    "SUBJ": "NMID",
    "SUBJECT": "NMID",
    "SUBJID": "NMID",
    "SUBJECT_ID": "NMID",
}


def normalize_column_name(name: str) -> str:
    """Normalize a single column name to canonical form.

    1. Uppercase the name (NONMEM convention).
    2. Check for known aliases (ID -> NMID, etc.).
    3. Return the result.

    Covariate columns (not in the canonical set) are uppercased but
    otherwise passed through.

    >>> normalize_column_name("time")
    'TIME'
    >>> normalize_column_name("id")
    'NMID'
    >>> normalize_column_name("wt")
    'WT'
    """
    upper = name.upper()
    return _COLUMN_ALIASES.get(upper, upper)


def normalize_columns(columns: list[str]) -> dict[str, str]:
    """Build a rename mapping for a list of column names.

    Returns a dict mapping original names to normalized names.
    Only includes entries where the name actually changes.

    >>> normalize_columns(["id", "time", "dv"])
    {'id': 'NMID', 'time': 'TIME', 'dv': 'DV'}
    """
    mapping: dict[str, str] = {}
    for col in columns:
        normalized = normalize_column_name(col)
        if normalized != col:
            mapping[col] = normalized
    return mapping
