# SPDX-License-Identifier: GPL-2.0-or-later
"""Categorical-covariate encoding detection + auto-remap (PRD §4.2.0).

APMODE's downstream consumers — most notably the FREM emitter's
``transform="binary"`` validator and the BLQ M3/M4 observation
models — assume canonical 0/1 coding for binary categorical
covariates. Real-world PK datasets routinely use other conventions:

  - "M" / "F", "Male" / "Female"  (warfarin)
  - 1 / 2  (mavoglurant; 1-indexed integer codes)
  - "Yes" / "No", "Y" / "N"  (clinical screening flags)
  - True / False booleans
  - "Pos" / "Neg", "Positive" / "Negative"

Rather than failing late inside the FREM emitter with a generic
``ValueError``, this module inspects each candidate categorical
column and either auto-remaps to 0/1 with a logged note or returns
an actionable diagnostic so the caller (CLI ``inspect``,
``validate``, or the orchestrator) can suggest a fix.

Convention for auto-remap: when two levels are detected, the
**lexicographically/numerically smaller** value maps to 0 and the
larger maps to 1. Booleans map ``False → 0`` and ``True → 1`` (the
Python convention). This is deterministic, stable across reruns, and
easy to override per-dataset.

Multi-level categorical covariates (>2 distinct values) are **not**
auto-remapped — the FREM-style additive-normal endpoint compromise
needs k-1 one-hot indicators chosen and named by the analyst, not
guessed by the loader. The diagnostic returned for those cases names
the levels and points the user at the one-hot recipe.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


_logger = structlog.get_logger(__name__)


EncodingKind = Literal[
    "binary_zero_one",
    "binary_one_two",
    "binary_string_pair",
    "binary_boolean",
    "multi_level",
    "continuous",
    "constant",
    "all_missing",
]


@dataclass(frozen=True)
class CategoricalEncodingHint:
    """Diagnostic + suggested remap for one column's encoding.

    Attributes:
        column: Source column name.
        detected_encoding: One of the ``EncodingKind`` literals.
        unique_values: Sorted list of distinct non-null values
            actually present in the column. Truncated to the first
            ten for ``multi_level`` / ``continuous`` to keep the
            diagnostic compact.
        suggested_remap: Mapping native value → ``0``/``1`` when the
            column is binary in a recognized form. ``None`` when the
            column is already 0/1, multi-level, or continuous.
        applied: Whether the remap was actually applied to the data
            (vs. only suggested in a report). Set by
            ``auto_remap_binary_columns`` based on caller policy.
        rationale: Human-readable one-liner used in CLI / log
            output. Examples: "two-level string column 'sex'
            ('female','male') auto-mapped to 0/1".
    """

    column: str
    detected_encoding: EncodingKind
    unique_values: list[object]
    suggested_remap: dict[object, int] | None
    applied: bool
    rationale: str


# Standard ordered pairs we recognize as binary categorical "labels".
# Each pair is (zero_label, one_label) — the value that becomes 0 vs
# the value that becomes 1. We match case-insensitively.
_KNOWN_BINARY_PAIRS: list[tuple[str, str]] = [
    ("f", "m"),
    ("female", "male"),
    ("no", "yes"),
    ("n", "y"),
    ("false", "true"),
    ("neg", "pos"),
    ("negative", "positive"),
    ("absent", "present"),
    ("control", "case"),
    ("placebo", "active"),
    ("0", "1"),
]


def _is_numeric_like(value: object) -> bool:
    """True iff the value is an int/float (excluding bool, which is a separate case)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_pair(values: list[object]) -> tuple[str, str] | None:
    """Return the lower-cased ``(min, max)`` pair when both values are stringy."""
    strings = [str(v).strip().lower() for v in values]
    if len(strings) != 2 or strings[0] == strings[1]:
        return None
    sorted_pair = tuple(sorted(strings))
    return sorted_pair  # type: ignore[return-value]


def detect_encoding(series: pd.Series, *, column: str | None = None) -> CategoricalEncodingHint:
    """Inspect a single column and classify its encoding.

    Args:
        series: The column to inspect. NaNs are dropped before
            analysis (so ``all_missing`` is a special case).
        column: Optional override for the rationale label. Falls back
            to ``series.name``.

    Returns:
        A ``CategoricalEncodingHint`` describing the encoding. The
        ``applied`` field is always ``False`` here — applying the
        remap is the caller's job (typically via
        ``auto_remap_binary_columns``).
    """
    name = column or str(series.name) or "<unnamed>"
    observed = series.dropna()
    if observed.empty:
        return CategoricalEncodingHint(
            column=name,
            detected_encoding="all_missing",
            unique_values=[],
            suggested_remap=None,
            applied=False,
            rationale=f"column {name!r} has no non-NaN values",
        )

    raw_unique = observed.unique().tolist()
    # Mixed-type columns (object dtype mixing strings and numerics) cannot be
    # robustly remapped — explicit mapping is required. Codex review:
    # ``["M", 1]`` would otherwise stringify and produce an alphabetic
    # remap that is almost certainly wrong.
    if len(raw_unique) > 0:
        types_seen = {type(v) for v in raw_unique}
        # bool is a subclass of int; treat them as one dtype for this check.
        non_str = {t for t in types_seen if not issubclass(t, str)}
        has_str = any(issubclass(t, str) for t in types_seen)
        if has_str and non_str and not all(issubclass(t, bool) for t in non_str):
            return CategoricalEncodingHint(
                column=name,
                detected_encoding="multi_level",
                unique_values=sorted(raw_unique, key=str)[:10],
                suggested_remap=None,
                applied=False,
                rationale=(
                    f"column {name!r} mixes string and numeric values "
                    f"({sorted(raw_unique, key=str)[:5]}); auto-remap refuses "
                    f"to guess. Pass an explicit override or canonicalize "
                    f"upstream."
                ),
            )
        # Gemini review: object-dtype columns mixing bool and non-bool
        # numeric (e.g., ``pd.Series([True, 1], dtype=object)``) hash-
        # collapse on equality and would otherwise be reported as
        # ``constant``. Treat them as ambiguous multi_level so the
        # caller is forced to canonicalize upstream.
        has_bool = any(issubclass(t, bool) for t in types_seen)
        has_non_bool_numeric = any(
            issubclass(t, (int, float)) and not issubclass(t, bool) for t in types_seen
        )
        if has_bool and has_non_bool_numeric:
            return CategoricalEncodingHint(
                column=name,
                detected_encoding="multi_level",
                unique_values=sorted(raw_unique, key=str)[:10],
                suggested_remap=None,
                applied=False,
                rationale=(
                    f"column {name!r} mixes bool and non-bool numeric values "
                    f"({sorted(raw_unique, key=str)[:5]}); these equate under "
                    f"Python semantics (True == 1) and cannot be safely "
                    f"remapped. Canonicalize to a single dtype upstream."
                ),
            )
    n_unique = len(raw_unique)

    if n_unique == 1:
        return CategoricalEncodingHint(
            column=name,
            detected_encoding="constant",
            unique_values=raw_unique,
            suggested_remap=None,
            applied=False,
            rationale=f"column {name!r} is constant ({raw_unique[0]!r})",
        )

    # Booleans → {False: 0, True: 1} via Python convention.
    if n_unique == 2 and all(isinstance(v, bool) for v in raw_unique):
        # ``dict[object, int]`` annotation needed so the dataclass field
        # accepts the boolean keys without a variance-mismatch error.
        remap: dict[object, int] = {False: 0, True: 1}
        return CategoricalEncodingHint(
            column=name,
            detected_encoding="binary_boolean",
            unique_values=sorted(raw_unique),
            suggested_remap=remap,
            applied=False,
            rationale=(f"column {name!r} is boolean True/False → mapped to 1/0"),
        )

    # Numeric-like binary cases: {0, 1} (no remap) or {1, 2} (1-indexed).
    if n_unique == 2 and all(_is_numeric_like(v) for v in raw_unique):
        sorted_nums = sorted(raw_unique, key=float)
        lo, hi = float(sorted_nums[0]), float(sorted_nums[1])
        if (lo, hi) == (0.0, 1.0):
            return CategoricalEncodingHint(
                column=name,
                detected_encoding="binary_zero_one",
                unique_values=[0, 1],
                suggested_remap=None,
                applied=False,
                rationale=(f"column {name!r} already in canonical 0/1 form; no remap needed"),
            )
        if (lo, hi) == (1.0, 2.0):
            remap_int = {sorted_nums[0]: 0, sorted_nums[1]: 1}
            return CategoricalEncodingHint(
                column=name,
                detected_encoding="binary_one_two",
                unique_values=sorted_nums,
                suggested_remap=remap_int,
                applied=False,
                rationale=(
                    f"column {name!r} uses 1-indexed integer codes 1/2 → mapped to 0/1 "
                    f"(lower value → 0, higher → 1)"
                ),
            )
        # Two distinct numbers but neither {0,1} nor {1,2}: treat as
        # continuous so we don't accidentally mis-remap quantitative data.
        return CategoricalEncodingHint(
            column=name,
            detected_encoding="continuous",
            unique_values=sorted_nums[:10],
            suggested_remap=None,
            applied=False,
            rationale=(
                f"column {name!r} has 2 distinct numeric values "
                f"({sorted_nums[0]}, {sorted_nums[1]}) but they are not the canonical "
                "0/1 or 1/2 binary encodings; treating as continuous to avoid mis-remap"
            ),
        )

    # Case-folded deduplication for string columns. ``["M", "m", "F"]``
    # would otherwise be classified ``multi_level`` (3 distinct raw
    # values) when it is semantically binary with a casing
    # inconsistency. We reduce to one representative per case-folded
    # class and broadcast the resulting remap back to every original
    # variant so ``{"M": 1, "m": 1, "F": 0}`` is what gets returned.
    case_variant_groups: dict[str, list[object]] | None = None
    if all(isinstance(v, str) for v in raw_unique):
        folded: dict[str, list[object]] = {}
        for v in raw_unique:
            folded.setdefault(str(v).strip().lower(), []).append(v)
        if len(folded) == 2:
            case_variant_groups = folded
            raw_unique = [variants[0] for variants in folded.values()]
            n_unique = 2

    # Two-level string columns: try to match against the known pairs.
    if n_unique == 2:
        pair = _normalize_pair(raw_unique)
        if pair is not None:
            zero_label_lower, one_label_lower = pair
            for known_zero, known_one in _KNOWN_BINARY_PAIRS:
                if {zero_label_lower, one_label_lower} == {known_zero, known_one}:
                    # Build the remap from canonical case-folded labels;
                    # broadcast across all raw variants when case folding
                    # collapsed multiple originals (``["M", "m"] → 1``).
                    sources = case_variant_groups or {
                        str(v).strip().lower(): [v] for v in raw_unique
                    }
                    remap_str: dict[object, int] = {}
                    for canonical, variants in sources.items():
                        target = 0 if canonical == known_zero else 1
                        for raw in variants:
                            remap_str[raw] = target
                    all_variants = sorted(remap_str.keys(), key=str)
                    return CategoricalEncodingHint(
                        column=name,
                        detected_encoding="binary_string_pair",
                        unique_values=all_variants,
                        suggested_remap=remap_str,
                        applied=False,
                        rationale=(
                            f"column {name!r} is recognised binary string pair "
                            f"({all_variants}) → mapped to 0/1 "
                            f"({known_zero!r}=0, {known_one!r}=1)"
                        ),
                    )
            # Unknown two-level string pair: still binary in spirit, but
            # we won't guess the polarity. Default to alphabetic-order
            # 0/1 and emit a UserWarning so programmatic callers see it
            # (the rationale string alone doesn't propagate to log
            # handlers that downstream consumers actually monitor).
            sorted_strs = sorted(raw_unique, key=str)
            sources = case_variant_groups or {str(v).strip().lower(): [v] for v in raw_unique}
            # Polarity by canonical (case-folded) sort.
            sorted_canonicals = sorted(sources.keys())
            zero_canon, one_canon = sorted_canonicals
            remap_unknown: dict[object, int] = {}
            for canonical, variants in sources.items():
                target = 0 if canonical == zero_canon else 1
                for raw in variants:
                    remap_unknown[raw] = target
            warnings.warn(
                f"Categorical column {name!r}: unknown two-level pair "
                f"{sorted_strs}; defaulting to alphabetic-order remap "
                f"({zero_canon!r}=0, {one_canon!r}=1). Pass "
                f"``overrides={{'{name}': {{'<value>': 0/1, ...}}}}`` to "
                f"force the desired polarity.",
                UserWarning,
                stacklevel=2,
            )
            return CategoricalEncodingHint(
                column=name,
                detected_encoding="binary_string_pair",
                unique_values=sorted(remap_unknown.keys(), key=str),
                suggested_remap=remap_unknown,
                applied=False,
                rationale=(
                    f"column {name!r} is a two-level string column "
                    f"({sorted_strs}) not in the recognised pair list; "
                    f"defaulting to {zero_canon!r}=0, {one_canon!r}=1 "
                    f"(alphabetic order). Override via overrides= "
                    f"if the polarity matters."
                ),
            )

    # >2 distinct values OR mixed numeric/string we can't map.
    if n_unique > 2 and n_unique <= 10:
        # Treat small-cardinality columns as multi-level categorical.
        return CategoricalEncodingHint(
            column=name,
            detected_encoding="multi_level",
            unique_values=sorted(raw_unique, key=str),
            suggested_remap=None,
            applied=False,
            rationale=(
                f"column {name!r} has {n_unique} distinct values "
                f"{sorted(raw_unique, key=str)[:10]}; FREM binary endpoints "
                f"require k-1 one-hot indicators chosen and named by the "
                f"analyst — auto-remap deliberately not attempted"
            ),
        )

    # Many distinct values: assume continuous.
    return CategoricalEncodingHint(
        column=name,
        detected_encoding="continuous",
        unique_values=raw_unique[:10],
        suggested_remap=None,
        applied=False,
        rationale=(f"column {name!r} has {n_unique} distinct values; treating as continuous"),
    )


def auto_remap_binary_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    overrides: dict[str, dict[object, int]] | None = None,
    apply: bool = True,
) -> tuple[pd.DataFrame, list[CategoricalEncodingHint]]:
    """Detect + (optionally) apply binary remaps for the named columns.

    Args:
        df: Source DataFrame. Returned unchanged when ``apply=False``.
        columns: Column names to inspect. Missing columns are skipped
            with a hint of kind ``all_missing`` so callers can surface
            them in a single diagnostic pass.
        overrides: Optional per-column explicit remap dict that takes
            precedence over the auto-detected one. The override is
            applied verbatim (with the same applied/rationale
            bookkeeping). Use this when the auto-detected polarity is
            wrong for the analysis.
        apply: When True (default), modifies ``df`` in place by
            applying the remap (auto or override) and returns the
            modified DataFrame. When False, returns ``df`` unchanged
            and the hints are advisory only — useful for ``apmode
            inspect`` / ``validate`` reports.

    Returns:
        ``(possibly-modified df, list of per-column hints)``. Each
        hint has ``applied=True`` iff the column was remapped in
        place (auto or override).
    """
    overrides = overrides or {}
    hints: list[CategoricalEncodingHint] = []
    out = df.copy() if apply else df

    for col in columns:
        if col not in df.columns:
            hints.append(
                CategoricalEncodingHint(
                    column=col,
                    detected_encoding="all_missing",
                    unique_values=[],
                    suggested_remap=None,
                    applied=False,
                    rationale=f"column {col!r} not present in DataFrame",
                )
            )
            continue

        explicit_remap = overrides.get(col)
        hint = detect_encoding(df[col], column=col)

        # Decide what to apply: override > auto-suggested > nothing.
        remap_to_apply: dict[object, int] | None
        if explicit_remap is not None:
            remap_to_apply = explicit_remap
            rationale = (
                f"column {col!r}: caller-supplied override remap applied ({explicit_remap})"
            )
        elif hint.suggested_remap is not None:
            remap_to_apply = hint.suggested_remap
            rationale = hint.rationale
        else:
            remap_to_apply = None
            rationale = hint.rationale

        if apply and remap_to_apply is not None:
            _logger.info(
                "categorical_encoding.remap_applied",
                column=col,
                detected_encoding=hint.detected_encoding,
                source="override" if explicit_remap is not None else "auto",
                remap={str(k): int(v) for k, v in remap_to_apply.items()},
            )
            # Validate remap completeness BEFORE applying. ``Series.map``
            # silently produces NaN for any source value not in the
            # remap dict — that would drop subjects from FREM without
            # warning. Codex/Crush both flagged this as the
            # highest-impact bug. Targets are also constrained to
            # ``{0, 1}`` so a malformed override cannot poison the
            # additive-normal endpoint interpretation downstream.
            observed_values = set(df[col].dropna().unique().tolist())
            unmapped = observed_values - set(remap_to_apply.keys())
            if unmapped:
                msg = (
                    f"Remap for column {col!r} is incomplete: "
                    f"observed values {sorted(unmapped, key=str)} have no "
                    f"target in {remap_to_apply}. Add explicit entries to "
                    f"the override or remove the rows upstream."
                )
                raise ValueError(msg)
            invalid_targets = {t for t in remap_to_apply.values() if t not in (0, 1)}
            if invalid_targets:
                msg = (
                    f"Remap for column {col!r} has non-binary target values "
                    f"{sorted(invalid_targets)}; the FREM binary endpoint "
                    f"requires targets in {{0, 1}}."
                )
                raise ValueError(msg)
            out[col] = df[col].map(remap_to_apply)

        hints.append(
            CategoricalEncodingHint(
                column=hint.column,
                detected_encoding=hint.detected_encoding,
                unique_values=hint.unique_values,
                suggested_remap=remap_to_apply,
                applied=apply and remap_to_apply is not None,
                rationale=rationale,
            )
        )

    return out, hints


def hints_to_provenance_entries(
    hints: Sequence[CategoricalEncodingHint],
) -> list[CategoricalEncodingEntry]:
    """Convert in-memory hints to the bundle-emittable Pydantic form.

    JSON does not distinguish ``int`` vs ``str`` keys, so the provenance
    artifact stringifies every original value. Reviewers reading
    ``categorical_encoding_provenance.json`` see exactly which raw value
    became which 0/1 target — the audit trail the multi-CLI review
    asked for after the silent-NaN-on-incomplete-remap class of bugs.
    """
    from apmode.bundle.models import CategoricalEncodingEntry

    entries: list[CategoricalEncodingEntry] = []
    for h in hints:
        if h.applied and h.suggested_remap is not None:
            source: Literal["auto", "override", "no_remap"] = (
                "override" if "override" in h.rationale.lower() else "auto"
            )
        else:
            source = "no_remap"
        applied_remap_str = (
            {str(k): int(v) for k, v in h.suggested_remap.items()}
            if h.applied and h.suggested_remap is not None
            else {}
        )
        entries.append(
            CategoricalEncodingEntry(
                column=h.column,
                detected_encoding=h.detected_encoding,
                unique_values=[str(v) for v in h.unique_values],
                applied_remap=applied_remap_str,
                applied=h.applied,
                source=source,
                rationale=h.rationale,
            )
        )
    return entries


if TYPE_CHECKING:
    from apmode.bundle.models import CategoricalEncodingEntry


# Documentation surface — referenced by README and CLI ``inspect``.
EXPECTED_BINARY_FORMAT: str = (
    "Binary categorical covariates must be encoded as the integers "
    "{0, 1}. APMODE auto-detects and remaps the following common "
    "alternatives at ingest time: True/False booleans, the integers "
    "{1, 2} (1-indexed), and the recognized string pairs M/F, "
    "Male/Female, Yes/No, Y/N, True/False, Pos/Neg, Positive/Negative, "
    "Absent/Present, Control/Case, Placebo/Active. Unknown two-level "
    "string columns get a deterministic alphabetical-order remap with "
    "a warning. Override the polarity via "
    "RunConfig.binary_encode_overrides or the per-call "
    "auto_remap_binary_columns(overrides=...) argument. Multi-level "
    "categorical columns (>2 distinct values) are NOT auto-remapped — "
    "one-hot encode to k-1 binary indicators upstream."
)
