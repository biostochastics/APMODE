# SPDX-License-Identifier: GPL-2.0-or-later
"""Data format adapters for backend-specific column naming (PRD S4.2.0).

Converts between the canonical PK schema (CanonicalPKSchema) and
backend-specific expectations (e.g., nlmixr2 uses 'ID' not 'NMID').
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import structlog

from apmode.data.categorical_encoding import auto_remap_binary_columns

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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
#
# Exported (no leading underscore) so ``apmode.data.initial_estimates``
# can apply the same PK-row filter inside its NCA estimator. Without the
# shared filter, NCA on mixed-endpoint datasets (e.g. warfarin's
# ``DVID="cp"`` PK + ``DVID="pca"`` PD rows) sees PD concentrations as
# PK noise, the per-subject lambda_z fits collapse for >50% of subjects,
# and the estimator falls all the way to ``_default_estimates()``.
PK_DVID_ALLOWLIST: frozenset[str] = frozenset({"1", "conc", "concentration", "cp"})

# Backwards-compatible alias for any in-tree consumer that grew an
# import on the underscored name before it was promoted. New call sites
# should use the public name.
_PK_DVID_ALLOWLIST = PK_DVID_ALLOWLIST

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
        "SUMIG_DOSE",
        "SUMIG_T0",
        "STUDY_ID",
    }
)

# Columns whose names collide with common PK parameter names. The
# nlmixr2data ACOP-2016 simulated datasets (Oral_1CPT, Bolus_1CPT, ...)
# carry the simulator's true per-subject parameter values in columns
# named exactly ``V``, ``CL``, ``KA``, etc. When rxode2 / nlmixr2 sees
# a data column with the same name as a model parameter, it silently
# uses the *data* column instead of the compiled model parameter.
# That is harmless during estimation (data columns ignored unless the
# model explicitly references them) but breaks the
# ``rxode2::rxSolve(object=fit, events=data, nStud=n_sims)`` posterior-
# predictive call inside ``r/harness.R::.simulate_posterior_predictive``:
# rxSolve treats the matching-name data columns as time-varying inputs,
# the per-subject ``V/CL/KA`` columns are constant for every observation,
# and the resulting trajectory shape mismatches the expected n_sims x
# n_obs matrix. The harness's ``tryCatch`` then returns NULL and Gate 3
# ends up with ``npe_score: None``, which the Phase-1 runner's strict
# ``_extract_npe`` rightly refuses.
#
# These names are NEVER legitimate NONMEM event-table columns, so
# stripping them at the adapter is always safe.
_PK_PARAM_COLLISION_COLUMNS: frozenset[str] = frozenset(
    {
        # Volumes
        "V",
        "VC",
        "V1",
        "V2",
        "V3",
        "VP",
        "VSS",
        # Clearances
        "CL",
        "CLR",
        "CLT",
        "CLP",
        "Q",
        "Q2",
        "Q3",
        # Absorption
        "KA",
        "KTR",
        "MTT",
        # Elimination rates / Michaelis-Menten
        "KE",
        "KEL",
        "VM",
        "VMAX",
        "KM",
        # Bioavailability / lag
        "F",
        "F1",
        "TLAG",
        # ACOP-2016 simulator metadata (not a parameter, but
        # similarly safe to strip — the model never references it)
        "DOSE",
        "SD",
    }
)


def _normalize_dvid_allowlist(values: Iterable[object]) -> frozenset[str]:
    """Normalize DVID tokens the same way row values are normalized."""
    return frozenset(str(v).strip().lower() for v in values)


def to_nlmixr2_format(
    df: pd.DataFrame,
    *,
    dvid_allowlist: Iterable[object] | None = None,
    categorical_references: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert canonical PK DataFrame to nlmixr2-ready form.

    Performs three transformations that nlmixr2's SAEM engine requires
    but that the canonical APMODE schema does not enforce:

    1. ``NMID`` → ``ID`` rename.
    2. Observation rows whose ``DVID`` is outside ``dvid_allowlist`` are
       filtered. By default that allowlist is the single-endpoint PK
       allowlist (``DVID`` values such as ``1``/``cp``), preserving the
       historical behavior. Multi-analyte callers pass the DSL-declared
       endpoint DVIDs so all modeled endpoints survive adaptation. When the
       resulting ``DVID`` column has ≤1 unique non-null value, the column is
       dropped entirely so single-endpoint models pass nlmixr2's
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

    # Strip parameter-name-collision columns (V, CL, KA, ...) the
    # ACOP-2016 simulator ships. See _PK_PARAM_COLLISION_COLUMNS for
    # the rationale; without this strip the harness's
    # ``rxode2::rxSolve(object=fit, events=data)`` returns NULL and
    # Gate 3 sees ``npe_score: None``.
    collision_cols = sorted(set(out.columns) & _PK_PARAM_COLLISION_COLUMNS)
    if collision_cols:
        _logger.info(
            "adapter.dropped_pk_param_collision_columns",
            n_columns=len(collision_cols),
            dropped_columns=collision_cols,
        )
        out = out.drop(columns=collision_cols)

    active_dvid_allowlist = (
        PK_DVID_ALLOWLIST if dvid_allowlist is None else _normalize_dvid_allowlist(dvid_allowlist)
    )

    if "DVID" in out.columns:
        evid = out["EVID"] if "EVID" in out.columns else pd.Series(0, index=out.index)
        obs_mask = evid == 0
        dvid_str = out["DVID"].astype(str).str.strip().str.lower()
        keep_mask = (~obs_mask) | dvid_str.isin(active_dvid_allowlist)
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            dropped_values = sorted({v for v in dvid_str[~keep_mask].unique() if v})
            _logger.info(
                "adapter.dropped_non_pk_dvid_rows",
                n_rows=n_dropped,
                dropped_dvid_values=dropped_values,
                allowlist=sorted(active_dvid_allowlist),
            )
            out = out.loc[keep_mask].reset_index(drop=True)
        # Filtering/reset_index changes positional labels; recompute from the
        # retained frame instead of reindexing the pre-filter mask.
        remaining_obs_mask = (
            out["EVID"].eq(0) if "EVID" in out.columns else pd.Series(True, index=out.index)
        )
        remaining = out.loc[remaining_obs_mask, "DVID"]
        unique_remaining = {str(v).strip().lower() for v in remaining.dropna().unique()}
        if len(unique_remaining) <= 1:
            out = out.drop(columns=["DVID"])

    # SumIG lowering needs the dose scalar at every RHS evaluation. rxode2's
    # reserved `amt` is only populated on event rows, so provide a persistent
    # per-subject single-dose column the SumIG model can reference. The SumIG
    # validator/data-bound checks reject multi-dose use; the adapter stays
    # conservative and only adds the helper when each dosed subject has exactly
    # one positive dose.
    if "SUMIG_DOSE" not in out.columns and {"ID", "EVID", "AMT"}.issubset(out.columns):
        dose_rows = out[(out["EVID"].isin([1, 4])) & (out["AMT"] > 0)]
        if (
            not dose_rows.empty
            and dose_rows.groupby("ID").size().eq(1).all()
            and "TIME" in dose_rows.columns
        ):
            out = out.merge(
                dose_rows[["ID", "AMT", "TIME"]].rename(
                    columns={"AMT": "SUMIG_DOSE", "TIME": "SUMIG_T0"}
                ),
                on="ID",
                how="left",
            )

    from pandas.api.types import is_numeric_dtype

    to_remap = sorted(
        c for c in out.columns if c not in _RESERVED_COLUMNS and not is_numeric_dtype(out[c])
    )
    if to_remap:
        overrides: dict[str, dict[object, int]] = {}
        for requested_col, reference in (categorical_references or {}).items():
            actual_col = next(
                (col for col in out.columns if col.upper() == requested_col.upper()), None
            )
            if actual_col is None or actual_col not in to_remap:
                continue
            observed = out[actual_col].dropna().unique().tolist()
            if len(observed) != 2:
                raise ValueError(
                    f"Categorical reference for {actual_col!r} requires exactly two "
                    f"observed levels; got {sorted(observed, key=str)}"
                )
            matches = [
                value
                for value in observed
                if str(value).strip().casefold() == reference.strip().casefold()
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Categorical reference {reference!r} for {actual_col!r} does not "
                    f"match exactly one observed level {sorted(observed, key=str)}"
                )
            baseline = matches[0]
            overrides[actual_col] = {value: 0 if value == baseline else 1 for value in observed}
        out, _hints = auto_remap_binary_columns(out, to_remap, overrides=overrides)

    return out
