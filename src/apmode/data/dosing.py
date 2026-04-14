# SPDX-License-Identifier: GPL-2.0-or-later
"""Multi-dose expansion and event table construction (PRD §4.2.0).

Provides:
- expand_addl(): Materializes ADDL/II into explicit dose rows.
- expand_infusion_events(): Generates infusion stop events.
- build_event_table(): Merges all events with deterministic sort priority.

nlmixr2/rxode2 handles ADDL/II/SS natively; these functions are used by
backends that require explicit events (Stan, NODE/JAX).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Event sort priority within the same TIME (lower = earlier).
# Matches NONMEM/rxode2 convention: resets before doses before observations.
_EVID_SORT_PRIORITY: dict[int, int] = {
    3: 0,  # reset
    4: 1,  # reset + dose
    1: 2,  # dose (bolus or infusion start)
    9: 3,  # infusion stop (synthetic)
    2: 4,  # other
    0: 5,  # observation
}

# Internal event types for infusion bookkeeping
INFUSION_STOP_EVID = 9  # synthetic; filtered before final output


def expand_addl(
    df: pd.DataFrame,
    *,
    col_time: str = "TIME",
    col_id: str = "NMID",
    col_evid: str = "EVID",
    col_addl: str = "ADDL",
    col_ii: str = "II",
) -> pd.DataFrame:
    """Expand ADDL/II into explicit dose rows.

    For each dose row (EVID in {1, 4}) with ADDL > 0 and II > 0, generate
    ADDL additional rows at TIME + k*II for k = 1..ADDL. Generated rows
    copy AMT, CMT, RATE, DUR from the source row. ADDL is set to 0 on
    generated rows to prevent re-expansion.

    Args:
        df: DataFrame with canonical PK columns.
        col_time: Time column name.
        col_id: Subject ID column name.
        col_evid: Event ID column name.
        col_addl: ADDL column name.
        col_ii: II column name.

    Returns:
        DataFrame with expanded dose rows, stable-sorted by ID/TIME/EVID priority.
    """
    if col_addl not in df.columns or col_ii not in df.columns:
        return df.copy()

    addl = df[col_addl].fillna(0).astype(int)
    ii = df[col_ii].fillna(0.0).astype(float)

    # Identify rows to expand: dose events with ADDL > 0 and II > 0
    mask = df[col_evid].isin([1, 4]) & (addl > 0) & (ii > 0)

    if not mask.any():
        return df.copy()

    new_rows: list[pd.Series] = []
    for idx in df.index[mask]:
        row = df.loc[idx]
        n_addl = int(addl.loc[idx])
        interval = float(ii.loc[idx])
        base_time = float(row[col_time])

        for k in range(1, n_addl + 1):
            new_row = row.copy()
            new_row[col_time] = base_time + k * interval
            new_row[col_addl] = 0
            # Mark as expanded for traceability
            new_rows.append(new_row)

    if not new_rows:
        return df.copy()

    # Set ADDL=0 on original expanding rows (they keep their original time)
    result = df.copy()
    result.loc[mask, col_addl] = 0

    expanded = pd.DataFrame(new_rows)
    result = pd.concat([result, expanded], ignore_index=True)

    return _sort_event_table(result, col_id=col_id, col_time=col_time, col_evid=col_evid)


def expand_infusion_events(
    df: pd.DataFrame,
    *,
    col_time: str = "TIME",
    col_id: str = "NMID",
    col_evid: str = "EVID",
    col_amt: str = "AMT",
    col_rate: str = "RATE",
    col_dur: str = "DUR",
) -> pd.DataFrame:
    """Generate explicit infusion stop events for rate-based infusions.

    For each dose row with a positive RATE or DUR, computes the infusion
    duration and creates a stop event at TIME + DUR. Infusion rates are
    stored in a _INF_RATE column for piecewise integration.

    Args:
        df: DataFrame (typically after expand_addl).

    Returns:
        DataFrame with added infusion stop rows (EVID=9, synthetic).
    """
    result = df.copy()

    # Compute infusion rate and duration
    has_rate = col_rate in result.columns
    has_dur = col_dur in result.columns

    if not has_rate and not has_dur:
        # No infusions — add helper column as zero
        result["_INF_RATE"] = 0.0
        return result

    rate = result[col_rate].fillna(0.0) if has_rate else pd.Series(0.0, index=result.index)
    dur = result[col_dur].fillna(0.0) if has_dur else pd.Series(0.0, index=result.index)
    amt = result[col_amt].fillna(0.0)

    # Compute infusion rate where possible
    inf_rate = rate.copy()
    # If DUR given but RATE not: RATE = AMT / DUR
    needs_rate = (inf_rate <= 0) & (dur > 0) & (amt > 0)
    inf_rate.loc[needs_rate] = amt.loc[needs_rate] / dur.loc[needs_rate]

    # Compute DUR where needed: DUR = AMT / RATE
    inf_dur = dur.copy()
    needs_dur = (inf_dur <= 0) & (inf_rate > 0) & (amt > 0)
    inf_dur.loc[needs_dur] = amt.loc[needs_dur] / inf_rate.loc[needs_dur]

    result["_INF_RATE"] = inf_rate

    # Find infusion dose rows
    is_infusion = result[col_evid].isin([1, 4]) & (inf_rate > 0) & (inf_dur > 0)

    if not is_infusion.any():
        return result

    stop_rows: list[dict[str, object]] = []
    for idx in result.index[is_infusion]:
        row = result.loc[idx]
        stop_time = float(row[col_time]) + float(inf_dur.loc[idx])
        stop_rows.append(
            {
                col_id: row[col_id],
                col_time: stop_time,
                col_evid: INFUSION_STOP_EVID,
                col_amt: 0.0,
                "CMT": row.get("CMT", 1),
                "_INF_RATE": -float(inf_rate.loc[idx]),  # negative = stop
            }
        )

    if stop_rows:
        stops = pd.DataFrame(stop_rows)
        # Fill missing columns with defaults (DV=NaN for synthetic events)
        for col in result.columns:
            if col not in stops.columns:
                if col in ("DV",):
                    stops[col] = np.nan
                else:
                    stops[col] = 0 if result[col].dtype in ("int64", "int32") else 0.0
        result = pd.concat([result, stops], ignore_index=True)

    return _sort_event_table(result, col_id=col_id, col_time=col_time, col_evid=col_evid)


def build_event_table(
    df: pd.DataFrame,
    *,
    col_time: str = "TIME",
    col_id: str = "NMID",
    col_evid: str = "EVID",
    col_addl: str = "ADDL",
    col_ii: str = "II",
    col_amt: str = "AMT",
    col_rate: str = "RATE",
    col_dur: str = "DUR",
    col_dv: str = "DV",
) -> pd.DataFrame:
    """Build a complete, sorted event table for explicit-event backends.

    Pipeline: expand_addl → expand_infusion_events → sort.

    Args:
        df: Raw canonical PK DataFrame (may contain ADDL/II/SS).

    Returns:
        Sorted DataFrame with all events explicit. Includes _INF_RATE column
        for infusion rate tracking. SS rows are preserved (backends must
        handle or reject them).
    """
    expanded = expand_addl(
        df,
        col_time=col_time,
        col_id=col_id,
        col_evid=col_evid,
        col_addl=col_addl,
        col_ii=col_ii,
    )
    with_infusions = expand_infusion_events(
        expanded,
        col_time=col_time,
        col_id=col_id,
        col_evid=col_evid,
        col_amt=col_amt,
        col_rate=col_rate,
        col_dur=col_dur,
    )
    return _sort_event_table(with_infusions, col_id=col_id, col_time=col_time, col_evid=col_evid)


def _sort_event_table(
    df: pd.DataFrame,
    *,
    col_id: str = "NMID",
    col_time: str = "TIME",
    col_evid: str = "EVID",
) -> pd.DataFrame:
    """Stable sort event table by subject, time, then event priority.

    Within same subject and time:
    - Reset (EVID=3) first
    - Reset+dose (EVID=4) second
    - Dose/infusion start (EVID=1) third
    - Infusion stop (EVID=9) before observations but after doses
    - Other (EVID=2) fourth
    - Observation (EVID=0) last
    """
    result = df.copy()
    result["_sort_priority"] = result[col_evid].map(lambda e: _EVID_SORT_PRIORITY.get(int(e), 3))
    result = result.sort_values(
        [col_id, col_time, "_sort_priority"],
        kind="mergesort",  # stable sort preserves original row order as tiebreaker
    ).reset_index(drop=True)
    result = result.drop(columns=["_sort_priority"])
    return result


def extract_subject_events(
    event_table: pd.DataFrame,
    subject_id: object,
    *,
    col_id: str = "NMID",
    col_time: str = "TIME",
    col_evid: str = "EVID",
    col_amt: str = "AMT",
    col_cmt: str = "CMT",
    col_dv: str = "DV",
) -> dict[str, np.ndarray]:
    """Extract per-subject event arrays for Stan/JAX integration.

    Returns a dict with arrays suitable for piecewise ODE integration:
    - event_times: sorted event times
    - event_evids: event type (0=obs, 1=dose, 3=reset, 4=reset+dose, 9=inf_stop)
    - event_amts: dose amounts (0 for obs/reset)
    - event_cmts: compartment numbers
    - event_dvs: observed DV (NaN for non-obs events)
    - event_inf_rates: infusion rate deltas (+rate=start, -rate=stop)
    - obs_indices: indices into event arrays where EVID=0
    """
    sdf = event_table[event_table[col_id] == subject_id].copy()

    inf_rate_col = "_INF_RATE" if "_INF_RATE" in sdf.columns else None

    return {
        "event_times": sdf[col_time].values.astype(np.float64),
        "event_evids": sdf[col_evid].values.astype(np.int32),
        "event_amts": sdf[col_amt].fillna(0.0).values.astype(np.float64),
        "event_cmts": sdf[col_cmt].fillna(1).values.astype(np.int32),
        "event_dvs": sdf[col_dv].fillna(np.nan).values.astype(np.float64),
        "event_inf_rates": (
            sdf[inf_rate_col].fillna(0.0).values.astype(np.float64)
            if inf_rate_col
            else np.zeros(len(sdf), dtype=np.float64)
        ),
        "obs_indices": np.where(sdf[col_evid].values == 0)[0].astype(np.int32),
    }
