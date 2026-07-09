# SPDX-License-Identifier: GPL-2.0-or-later
"""Seven-level Formular DSL validation API (Formular sharpening plan §4 Phase 1, P1.8).

:func:`validate` is the single public entry point every caller — CLI,
agentic transform loop, programmatic API consumer — should use; the CLI
(P1.9) is a thin renderer over its :class:`ValidationReport`, never a
second logic path.

Layering: ``syntax``/``ast``/``semantic``/``lane_bound`` are *tags* on the
checks :func:`apmode.dsl.validator.validate_dsl` already runs (one pass
over the compiled spec, then filtered by each error's :class:`FrmCode`
taxon — see :data:`_TAXON_TO_LEVEL`); ``data_bound``/``backend_bound``/
``policy_bound`` are new checks (:func:`apmode.dsl.validator.validate_data_bound`
/ ``validate_backend_bound`` / ``validate_policy_bound``) that only run
when their required input (``data``/``backend``/``policy``) is supplied.
Physically splitting ``validate_dsl`` into four separate functions was
rejected: every one of its checks already walks the same AST once, and a
real split would mean either four full spec traversals per call or a
shared-state refactor with no behavioural benefit — filtering a single
pass's output by ``FrmCode`` taxon is cheap (a compiled spec is small) and
keeps ``validate_dsl`` itself a stable, independently-tested unit.

Selecting a bound level whose required input is not supplied (e.g.
``level=ValidationLevel.DATA_BOUND`` with ``data=None``) marks that level
as skipped in :attr:`ValidationReport.skipped_levels`; ``ok`` is false for
reports with skipped requested levels. This prevents CI/CLI callers from
mistaking "not run" for "clean".
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from apmode.dsl.validator import (
    ValidationError,
    validate_backend_bound,
    validate_data_bound,
    validate_dsl,
    validate_policy_bound,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

    from apmode.dsl.ast_models import DSLSpec
    from apmode.dsl.lane import Lane
    from apmode.governance.policy import GatePolicy


class ValidationLevel(StrEnum):
    """One named layer of Formular DSL validation.

    ``ALL`` is a selector sentinel, not a checkable level in its own
    right — :func:`resolve_levels` expands it to every member of
    :data:`_CHECKABLE_LEVELS` and it never appears as a key on
    :attr:`ValidationReport.by_level`.
    """

    SYNTAX = "syntax"
    AST = "ast"
    SEMANTIC = "semantic"
    DATA_BOUND = "data_bound"
    LANE_BOUND = "lane_bound"
    BACKEND_BOUND = "backend_bound"
    POLICY_BOUND = "policy_bound"
    ALL = "all"


_CHECKABLE_LEVELS: tuple[ValidationLevel, ...] = (
    ValidationLevel.SYNTAX,
    ValidationLevel.AST,
    ValidationLevel.SEMANTIC,
    ValidationLevel.DATA_BOUND,
    ValidationLevel.LANE_BOUND,
    ValidationLevel.BACKEND_BOUND,
    ValidationLevel.POLICY_BOUND,
)

# ``validate_dsl`` runs in one pass; a returned error's level is derived
# from its FrmCode taxon (the "FRM-{TAXON}-NNN" prefix). PRIOR errors are
# not emitted by validate_dsl today (see apmode.dsl.errors module
# docstring), but PRIOR-family checks are semantic/numeric in character
# (justification length, family/target-kind mismatch) so they are mapped
# to SEMANTIC for when that changes. Any taxon absent from this mapping
# (there is none, today) would default to SEMANTIC via .get(...,
# ValidationLevel.SEMANTIC) — a deliberately conservative fallback so an
# unmapped future taxon is never silently dropped from every level filter.
_TAXON_TO_LEVEL: dict[str, ValidationLevel] = {
    "SYN": ValidationLevel.SYNTAX,
    "AST": ValidationLevel.AST,
    "SEM": ValidationLevel.SEMANTIC,
    "LANE": ValidationLevel.LANE_BOUND,
    "BE": ValidationLevel.BACKEND_BOUND,
    "DATA": ValidationLevel.DATA_BOUND,
    "POLICY": ValidationLevel.POLICY_BOUND,
    "PRIOR": ValidationLevel.SEMANTIC,
}


def _level_for_code(code: str) -> ValidationLevel:
    """Map a ``FrmCode`` value (``"FRM-{TAXON}-NNN"``) to its ``ValidationLevel``."""
    parts = code.split("-")
    taxon = parts[1] if len(parts) >= 3 else ""
    return _TAXON_TO_LEVEL.get(taxon, ValidationLevel.SEMANTIC)


def resolve_levels(
    level: ValidationLevel | Iterable[ValidationLevel],
) -> frozenset[ValidationLevel]:
    """Expand a level selector (single level / iterable of levels / ALL).

    ``ValidationLevel.ALL`` — alone or anywhere inside an iterable —
    expands to every member of :data:`_CHECKABLE_LEVELS`.
    """
    selected: set[ValidationLevel] = {level} if isinstance(level, ValidationLevel) else set(level)
    if ValidationLevel.ALL in selected:
        return frozenset(_CHECKABLE_LEVELS)
    return frozenset(selected)


class ValidationReport(BaseModel):
    """Per-level results of one :func:`validate` call.

    ``by_level`` only ever carries keys for the levels that were actually
    selected (see :func:`resolve_levels`) — a level absent from the
    selector never appears as a key, distinguishing "not run" from "run,
    found nothing".
    """

    model_config = ConfigDict(frozen=True)

    levels_run: frozenset[ValidationLevel]
    by_level: dict[ValidationLevel, list[ValidationError]]
    skipped_levels: dict[ValidationLevel, str] = Field(default_factory=dict)

    @property
    def all_errors(self) -> list[ValidationError]:
        """Every error across every run level, in a stable level order."""
        flattened: list[ValidationError] = []
        for lvl in _CHECKABLE_LEVELS:
            flattened.extend(self.by_level.get(lvl, []))
        return flattened

    @property
    def ok(self) -> bool:
        """True iff no run level produced a ``severity="error"`` entry.

        Requested levels skipped for missing required inputs fail ``ok``.
        Warnings/info-severity entries never fail ``ok`` — matching
        :class:`apmode.dsl.validator.ValidationError`'s existing
        ``severity`` field semantics.
        """
        return not self.skipped_levels and not any(e.severity == "error" for e in self.all_errors)


def validate(
    spec: DSLSpec,
    *,
    level: ValidationLevel | Iterable[ValidationLevel] = ValidationLevel.ALL,
    lane: Lane,
    data: pd.DataFrame | None = None,
    backend: str | None = None,
    policy: GatePolicy | None = None,
) -> ValidationReport:
    """Validate ``spec`` at one or more named levels.

    Args:
        spec: The compiled DSL spec to validate.
        level: A single :class:`ValidationLevel`, an iterable of them, or
            ``ValidationLevel.ALL`` (default) for every level.
        lane: Operating lane — required by ``syntax``/``ast``/``semantic``/
            ``lane_bound`` (via :func:`apmode.dsl.validator.validate_dsl`)
            and by ``policy_bound`` (lane/policy match check).
        data: Bound dataset for ``data_bound`` checks. When ``None``,
            ``data_bound`` (if selected) is marked skipped.
        backend: Emitter name (``"nlmixr2"``/``"stan"``/``"frem"``) for
            ``backend_bound`` checks. When ``None``, ``backend_bound``
            (if selected) is marked skipped.
        policy: A loaded ``GatePolicy`` for ``policy_bound`` checks. When
            ``None``, ``policy_bound`` (if selected) is marked skipped.

    Returns:
        A :class:`ValidationReport` keyed by the levels actually selected.
    """
    levels = resolve_levels(level)
    by_level: dict[ValidationLevel, list[ValidationError]] = {lvl: [] for lvl in levels}
    skipped_levels: dict[ValidationLevel, str] = {}

    spec_bound_levels = levels & {
        ValidationLevel.SYNTAX,
        ValidationLevel.AST,
        ValidationLevel.SEMANTIC,
        ValidationLevel.LANE_BOUND,
    }
    if spec_bound_levels:
        for err in validate_dsl(spec, lane=lane):
            lvl = _level_for_code(str(err.code))
            if lvl in spec_bound_levels:
                by_level[lvl].append(err)

    if ValidationLevel.DATA_BOUND in levels:
        if data is None:
            skipped_levels[ValidationLevel.DATA_BOUND] = (
                "data_bound selected but no data was supplied"
            )
        else:
            by_level[ValidationLevel.DATA_BOUND].extend(validate_data_bound(spec, data))

    if ValidationLevel.BACKEND_BOUND in levels:
        if backend is None:
            skipped_levels[ValidationLevel.BACKEND_BOUND] = (
                "backend_bound selected but no backend was supplied"
            )
        else:
            by_level[ValidationLevel.BACKEND_BOUND].extend(validate_backend_bound(spec, backend))

    if ValidationLevel.POLICY_BOUND in levels:
        if policy is None:
            skipped_levels[ValidationLevel.POLICY_BOUND] = (
                "policy_bound selected but no policy was supplied"
            )
        else:
            by_level[ValidationLevel.POLICY_BOUND].extend(
                validate_policy_bound(spec, lane, policy)
            )

    return ValidationReport(levels_run=levels, by_level=by_level, skipped_levels=skipped_levels)


__all__ = ["ValidationLevel", "ValidationReport", "resolve_levels", "validate"]
