# SPDX-License-Identifier: GPL-2.0-or-later
"""Best-effort text-pattern migrator: pre-Phase-1 Formular syntax -> current grammar.

Formular sharpening plan §4 Phase 1 (P1.11). Phase 1 replaced two pieces of
surface syntax outright rather than dual-supporting old and new forms (see
``docs/plans/2026-07-08-formular-sharpening-and-adoption-design.md`` §4):
inline calibration values on structural declarations, and the compact
``CovariateLink(...)`` function-call syntax. The compiler
(:mod:`apmode.dsl.grammar`) no longer parses either old form at all, so
there is no live Lark grammar to reuse for migration — resurrecting a
second full grammar just to migrate a handful of syntactic patterns would
be disproportionate to the problem. This module is instead a standalone
regex/string rewriter for exactly those two changes:

1. Inline ``name=value`` calibration arguments move out of structural
   declarations into a new ``initial: { ... }`` block:
   ``FirstOrder(ka=1.5)`` becomes ``FirstOrder(ka)`` plus
   ``initial: { ka = 1.5 }``. Structural (non-calibration) arguments —
   ``Transit``/``Erlang``'s ``n``, ``SumIG``'s ``k``, ``TimeVarying``'s
   ``decay_fn`` — are left inline unchanged; they were never moved.

2. The compact ``CovariateLink(param=P, covariate=C, form=F)`` call
   (formerly a ``variability:`` item) moves to the arrow-syntax
   ``covariates: { P <- C.F(...) }`` block. The old syntax carried no
   per-form reference values at all (they were hardcoded inside the
   nlmixr2 emitter); ``power``/``exponential``/``linear``/``maturation``
   get defaults that reproduce those *exact* old hardcoded emitter
   constants (0.75/70 allometric default for power — Anderson & Holford
   2008 — 0.0 for exponential/linear, 1.0/1.0 for maturation's
   hill/tm50), so a migrated-then-compiled spec lowers to numerically
   identical backend code. ``form=categorical`` has no old-syntax
   equivalent for its now-required ``reference`` (baseline level name)
   field — that information never existed anywhere in the pre-Phase-1
   AST — so a categorical link is intentionally left unmigrated with a
   :class:`MigrationWarning` rather than guessed at.

See ``docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md`` for the full recipe with
worked examples, and the module docstring above each helper for the exact
scope this best-effort rewriter does and does not cover.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class MigrationWarning(BaseModel):
    """One construct the migrator recognized but could not safely rewrite."""

    model_config = ConfigDict(frozen=True)

    line: int
    message: str


class MigrationResult(BaseModel):
    """Output of :func:`migrate_v06_to_v07`: rewritten text plus any warnings."""

    model_config = ConfigDict(frozen=True)

    text: str
    warnings: list[MigrationWarning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Structural-declaration calibration-value stripping
# ---------------------------------------------------------------------------

# Every pre-Phase-1 structural function-call keyword that carried at least
# one inline calibration value. IVBolus / NODE_Absorption / NODE_Elimination
# are omitted: they never had calibration arguments and are unaffected by
# this rewrite, so leaving them out of the match set is a no-op for those
# lines (not an oversight).
_STRUCTURAL_KEYWORDS = (
    "FirstOrder",
    "ZeroOrder",
    "LaggedFirstOrder",
    "Transit",
    "MixedFirstZero",
    "Erlang",
    "ParallelFirstOrder",
    "SumIG",
    "OneCmt",
    "TwoCmt",
    "ThreeCmt",
    "TMDD_Core",
    "TMDD_QSS",
    "Linear",
    "MichaelisMenten",
    "ParallelLinearMM",
    "TimeVarying",
)

# Parameter names are unique across every structural keyword above (no two
# keywords share a calibration name with different meaning), so
# classification is a flat name -> kind lookup rather than a per-keyword
# positional schema. Names NOT in this set (n, k, decay_fn) are structural
# and stay inline as ``name=value`` verbatim.
_CALIBRATION_PARAM_NAMES = frozenset(
    {
        "ka",
        "dur",
        "tlag",
        "ktr",
        "frac",
        "ka1",
        "ka2",
        "MT_1",
        "MT_2",
        "RD2_1",
        "RD2_2",
        "weight_1",
        "V",
        "V1",
        "V2",
        "V3",
        "Q",
        "Q2",
        "Q3",
        "R0",
        "kon",
        "koff",
        "kint",
        "KD",
        "CL",
        "Vmax",
        "Km",
        "kdecay",
    }
)

_STRUCTURAL_CALL_RE = re.compile(r"\b(" + "|".join(_STRUCTURAL_KEYWORDS) + r")\s*\(([^()]*)\)")


def _rewrite_structural_call(match: re.Match[str]) -> tuple[str, list[tuple[str, str]]]:
    """Strip calibration values from one matched structural call.

    Returns the rewritten call text and the ``(name, raw_value)`` pairs
    removed from it, in left-to-right order. Args with no ``=`` (e.g. a
    line already migrated, or a bare structural name a second ``--migrate``
    pass encounters) pass through unchanged and contribute no calibration
    entry.
    """
    keyword, inner = match.group(1), match.group(2)
    pieces = [p.strip() for p in inner.split(",") if p.strip()]
    rebuilt: list[str] = []
    calibration: list[tuple[str, str]] = []
    for piece in pieces:
        if "=" not in piece:
            rebuilt.append(piece)
            continue
        name, _, value = piece.partition("=")
        name, value = name.strip(), value.strip()
        if name == "kdecay":
            # TimeVarying's new grammar has no place for kdecay in the call
            # at all -- ``time_varying_elim: "TimeVarying" "(" "CL" ","
            # "decay_fn" "=" DECAY_FN ")"`` -- it becomes a pure `initial:`
            # value (optional; defaults to 0.1 via DSLSpec.get_initial), so
            # it is dropped from the call entirely rather than kept bare.
            calibration.append((name, value))
            continue
        if name in _CALIBRATION_PARAM_NAMES:
            rebuilt.append(name)
            calibration.append((name, value))
        else:
            rebuilt.append(f"{name}={value}")
    return f"{keyword}({', '.join(rebuilt)})", calibration


def _rewrite_structural_calls(line: str) -> tuple[str, list[tuple[str, str]]]:
    """Apply :func:`_rewrite_structural_call` to every match on one line."""
    calibration: list[tuple[str, str]] = []

    def _sub(match: re.Match[str]) -> str:
        rewritten, cal = _rewrite_structural_call(match)
        calibration.extend(cal)
        return rewritten

    return _STRUCTURAL_CALL_RE.sub(_sub, line), calibration


# ---------------------------------------------------------------------------
# CovariateLink -> covariates: arrow-syntax migration
# ---------------------------------------------------------------------------

_COVARIATE_LINK_RE = re.compile(
    r"CovariateLink\s*\(\s*param\s*=\s*(\w+)\s*,\s*covariate\s*=\s*(\w+)\s*,\s*form\s*=\s*(\w+)\s*\)"
)

# Value-preserving defaults reproducing the pre-P1.6 hardcoded emitter
# constants exactly (see apmode/dsl/nlmixr2_emitter.py git history:
# power -> 0.75 coefficient / 70 reference weight, Anderson & Holford 2008;
# exponential/linear -> 0 coefficient via the old catch-all branch;
# maturation -> hill=1, tm50=1). ``categorical`` is deliberately absent: the
# old syntax never carried a baseline-level name anywhere, so there is no
# value to preserve or reasonably default to.
_COVARIATE_FORM_DEFAULTS: dict[str, str] = {
    "power": "theta=0.75, ref=70",
    "exponential": "theta=0.0",
    "linear": "theta=0.0",
    "maturation": "tm50=1.0, hill=1.0",
}

# A line whose stripped content becomes one of these after removing its sole
# CovariateLink call had no other content of its own — drop it entirely
# rather than leave a dangling ``variability:`` (now-legal empty) or blank
# line. Multiple declarations sharing one line (e.g.
# ``variability: { IIV(...), CovariateLink(...) }`` on a single line) are
# outside this best-effort migrator's scope and are left as-is; see
# docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md.
_DROPPABLE_STRIPPED_LINES = frozenset({"", "variability:"})


def _rewrite_covariate_links(
    line: str, line_no: int
) -> tuple[str, list[str], list[MigrationWarning]]:
    """Rewrite (or flag) every ``CovariateLink(...)`` call on one line.

    Handled forms are removed from ``line`` and turned into a
    ``covariates:`` entry string; ``form=categorical`` (or any unrecognized
    form) is left untouched in ``line`` and reported as a
    :class:`MigrationWarning` instead — the caller must not silently drop
    or guess at it.
    """
    entries: list[str] = []
    warnings: list[MigrationWarning] = []
    spans_to_blank: list[tuple[int, int]] = []

    for match in _COVARIATE_LINK_RE.finditer(line):
        param, covariate, form = match.group(1), match.group(2), match.group(3)
        defaults = _COVARIATE_FORM_DEFAULTS.get(form)
        if defaults is None:
            warnings.append(
                MigrationWarning(
                    line=line_no,
                    message=(
                        f"could not auto-migrate this construct near line {line_no}, "
                        f"please review manually: CovariateLink(param={param}, "
                        f"covariate={covariate}, form={form}) has no equivalent "
                        f"reference value under the old syntax for form={form!r}; "
                        "add a `covariates:` entry by hand "
                        "(see docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md)."
                    ),
                )
            )
            continue
        entries.append(f"{param} <- {covariate}.{form}({defaults})")
        spans_to_blank.append(match.span())

    new_line = line
    for start, end in reversed(spans_to_blank):
        new_line = new_line[:start] + new_line[end:]
    return new_line, entries, warnings


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

_TOP_LEVEL_BLOCK_RE = re.compile(
    r"^(\s*)(absorption|distribution|elimination|observation|variability)\s*:"
)


def _detect_indent(lines: list[str]) -> str:
    """Reuse the indentation of an existing top-level block, else 4 spaces."""
    for line in lines:
        found = _TOP_LEVEL_BLOCK_RE.match(line)
        if found:
            return found.group(1)
    return "    "


def _find_final_closing_brace(lines: list[str]) -> int:
    """Index of the last line whose stripped content is exactly ``}``.

    Assumes a single ``model { ... }`` per file, matching every existing
    Formular fixture/benchmark/test spec in this repository.
    """
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            return i
    msg = "could not find a closing `}` for the model block"
    raise ValueError(msg)


def migrate_v06_to_v07(text: str) -> MigrationResult:
    """Best-effort rewrite of pre-Phase-1 Formular text to the current grammar.

    Handles exactly the two breaking syntax changes documented in
    ``docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md``: inline calibration values
    move into a synthesized ``initial: { ... }`` block, and
    ``CovariateLink(...)`` calls move into a synthesized
    ``covariates: { ... }`` block. Constructs the migrator does not
    recognize (or cannot safely default, e.g. ``form=categorical``) are
    left untouched in the output text and reported via
    :attr:`MigrationResult.warnings` — never silently corrupted or
    dropped. Every other line passes through byte-for-byte.
    """
    lines = text.split("\n")
    calibration: dict[str, str] = {}
    covariate_entries: list[str] = []
    warnings: list[MigrationWarning] = []
    out_lines: list[str] = []

    for line_no, line in enumerate(lines, start=1):
        rewritten, cal = _rewrite_structural_calls(line)
        for name, value in cal:
            calibration.setdefault(name, value)
        rewritten, entries, line_warnings = _rewrite_covariate_links(rewritten, line_no)
        covariate_entries.extend(entries)
        warnings.extend(line_warnings)

        if entries and rewritten.strip() in _DROPPABLE_STRIPPED_LINES:
            continue
        out_lines.append(rewritten)

    if not calibration and not covariate_entries:
        return MigrationResult(text="\n".join(out_lines), warnings=warnings)

    indent = _detect_indent(out_lines)
    insert_at = _find_final_closing_brace(out_lines)

    new_blocks: list[str] = []
    if calibration:
        body = ", ".join(f"{name} = {value}" for name, value in calibration.items())
        new_blocks.append(f"{indent}initial: {{ {body} }}")
    if covariate_entries:
        body = ", ".join(covariate_entries)
        new_blocks.append(f"{indent}covariates: {{ {body} }}")

    out_lines[insert_at:insert_at] = new_blocks
    return MigrationResult(text="\n".join(out_lines), warnings=warnings)


__all__ = ["MigrationResult", "MigrationWarning", "migrate_v06_to_v07"]
