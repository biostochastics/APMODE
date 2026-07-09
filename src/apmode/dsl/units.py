# SPDX-License-Identifier: GPL-2.0-or-later
"""Dimensional-homogeneity checker for the Formular DSL (Formular sharpening
plan §4 Phase 1, P1.3).

**This is a dimensional-homogeneity checker, not a unit-conversion
library.** It never converts a number from one unit to another and does
not depend on Pint (or any unit-algebra package). It answers exactly two
questions:

1. Given a spec's optional top-level ``units: { time, amount,
   concentration, volume }`` block (see
   ``apmode.dsl.ast_models.UnitsDeclaration``), is the declaration
   internally self-consistent -- i.e. is ``volume`` dimensionally
   reachable from ``amount``/``concentration`` via
   ``Volume = Amount / Concentration``?
2. For each calibration parameter in the spec (the flat
   ``DSLSpec.initial`` values plus the inline observation sigmas), what
   dimension does its structural *role* imply, and does that dimension
   depend on the (possibly broken) ``volume`` reachability above?

Formular has no per-parameter unit annotation syntax -- units are declared
once, globally, for the whole spec. The role -> dimension lookup below
(``_ROLE_DIMENSIONS``) is therefore how "does CL make dimensional sense"
gets answered: CL's role is Clearance = Volume/Time, so it inherits
whatever the global volume-reachability check concluded.

Recognized unit vocabulary (deliberately small; PK units are
conventionally metric-prefixed mass/volume/time)
---------------------------------------------------
- Mass tokens: ``g``, ``mg``, ``mcg``, ``ug``, ``ng``.
- Volume tokens: ``L``, ``mL``.
- Time tokens (used only to positively confirm a token is time-shaped
  when checking for a *wrong-category* mismatch, never for arithmetic):
  ``h``, ``hr``, ``min``, ``day``, ``d``.

Every prefixed mass token collapses to the single category ``"mass"`` and
every volume token to ``"volume"`` -- the *conversion factor* between
e.g. ``mg`` and ``ng`` is irrelevant to a homogeneity check and is never
computed here. Concentration is expected to be written as a compound
``<mass>/<volume>`` token (e.g. ``"ng/mL"``); the grammar's ``unit_expr``
rule already splits this into two NAME tokens joined by ``/``.

What happens on an unrecognized token
--------------------------------------
A token outside the three vocabularies above (e.g. an author using
``"lb"`` or a typo) makes that field's category unresolvable. The
checker never guesses or raises a false-positive mismatch in that case:
the field (and every calibration parameter that depends on it) is marked
**unresolved**, which the coverage report surfaces as "unchecked" rather
than "mismatched". Only a token that resolves to a *recognized but wrong*
category (e.g. ``volume = "mg"``, which is mass-shaped) is a genuine
``FRM-SEM-010`` mismatch. A concentration with no ``/`` separator is
always a mismatch (not merely unresolved) since ``Amount/Volume`` cannot
be expressed as a single bare token by construction.

sigma_prop / sigma_add convention: standard deviation, not variance
---------------------------------------------------------------------
CHANGELOG.md (Suite A benchmark simulator notes) documents a real,
previously-live confusion point: "sigma_prop and sigma_add in the
simulator and reference_params.json are standard deviations on the data
scale. NONMEM's SIGMA block uses variance; square before comparing."
Formular's ``sigma_prop``/``sigma_add`` fields follow the simulator's
convention -- **standard-deviation scale, not variance** -- for both
``Proportional``/``Additive``/``Combined`` and the ``BLQM3``/``BLQM4``
always-present sigma fields. A spec author (or an agentic transform)
porting a NONMEM ``$SIGMA`` block must take the square root before
writing it into Formular's ``initial:``-adjacent sigma fields, not copy
the value verbatim.

``_sigma_prop_heuristic_warnings`` below adds one narrowly-scoped,
non-fatal heuristic for this exact failure mode: a ``sigma_prop`` (an
SD-scale *fraction*) above 1.0 -- i.e. implying a residual SD larger than
the predicted concentration itself -- is unusual enough in real PK
practice to warrant a warning, and is exactly the shape of value you get
by pasting a NONMEM proportional *variance* (e.g. 0.09) straight in
without squaring... except squaring makes it *smaller*, so the more
common real mistake is the reverse: an SD copied in as if it were a
variance produces a suspiciously *small* value, not a large one. The
threshold here only catches the large-value direction (a plausible typo
or unit-scale error), is deliberately conservative (misses many real
SD/variance mix-ups), and is a warning, never a hard error -- there is no
data-free way to distinguish "unusually large but intended" from
"mistaken" here, so this does not use a fabricated confidence threshold
beyond documenting exactly what it catches and why.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    Additive,
    Combined,
    Proportional,
)

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec, ObservationModule, UnitsDeclaration

# ---------------------------------------------------------------------------
# Dimension representation
# ---------------------------------------------------------------------------
#
# A Dimension is an exponent triple over the three base quantities Formular
# actually needs: (time_exponent, amount_exponent, concentration_exponent).
# Derived dimensions are plain tuple arithmetic over that basis -- there is
# no unit-algebra engine, just a handful of hand-written constants.

Dimension = tuple[int, int, int]

TIME: Dimension = (1, 0, 0)
AMOUNT: Dimension = (0, 1, 0)
CONCENTRATION: Dimension = (0, 0, 1)
UNITLESS: Dimension = (0, 0, 0)
VOLUME: Dimension = (0, 1, -1)  # Amount / Concentration
CLEARANCE: Dimension = (-1, 1, -1)  # Volume / Time
RATE: Dimension = (-1, 0, 0)  # 1 / Time
AMOUNT_PER_TIME: Dimension = (-1, 1, 0)  # Amount / Time (Vmax)

_DIMENSION_LABELS: dict[Dimension, str] = {
    TIME: "Time",
    AMOUNT: "Amount",
    CONCENTRATION: "Concentration",
    UNITLESS: "Unitless",
    VOLUME: "Volume",
    CLEARANCE: "Clearance",
    RATE: "Rate",
    AMOUNT_PER_TIME: "Amount/Time",
}

# Which declared-units fields a role's dimension transitively depends on.
# "volume"/"amount"/"concentration_num"/"concentration_den" are the four
# leaf checks performed by _field_statuses(); "time" is its own leaf.
_DIMENSION_DEPENDENCIES: dict[Dimension, tuple[str, ...]] = {
    CLEARANCE: ("volume", "amount", "concentration_num", "concentration_den", "time"),
    VOLUME: ("volume", "amount", "concentration_num", "concentration_den"),
    AMOUNT_PER_TIME: ("amount", "time"),
    CONCENTRATION: ("concentration_num", "concentration_den"),
    RATE: ("time",),
    TIME: ("time",),
    UNITLESS: (),
}

# Per the task brief: exactly this role table, no more. Parameters with no
# entry here (Q/Q2/Q3, TMDD's R0/kon/koff/kint/KD, SumIG/ParallelFirstOrder
# component params, structural n/k/dim/decay_fn, NODE weights, ...) are
# reported as "unchecked" -- Phase 2 candidates, not silently guessed at.
_ROLE_DIMENSIONS: dict[str, Dimension] = {
    "CL": CLEARANCE,
    "V": VOLUME,
    "V1": VOLUME,
    "V2": VOLUME,
    "V3": VOLUME,
    "ka": RATE,
    "ktr": RATE,
    "kdecay": RATE,
    "Vmax": AMOUNT_PER_TIME,
    "Km": CONCENTRATION,
    "tlag": TIME,
    "dur": TIME,
    "frac": UNITLESS,
    "sigma_prop": UNITLESS,
    "sigma_add": CONCENTRATION,
}

# ---------------------------------------------------------------------------
# Recognized unit-token vocabulary
# ---------------------------------------------------------------------------

_MASS_UNITS = frozenset({"g", "mg", "mcg", "ug", "ng"})
_VOLUME_UNITS = frozenset({"L", "mL"})
_TIME_UNITS = frozenset({"h", "hr", "min", "day", "d"})

_Category = Literal["mass", "volume", "time"]


def _classify(token: str) -> _Category | None:
    """Map a bare unit token to its recognized category, or ``None``."""
    if token in _MASS_UNITS:
        return "mass"
    if token in _VOLUME_UNITS:
        return "volume"
    if token in _TIME_UNITS:
        return "time"
    return None


_FieldStatus = Literal["ok", "mismatch", "unresolved"]


def _field_statuses(units: UnitsDeclaration) -> tuple[dict[str, _FieldStatus], dict[str, str]]:
    """Classify each leaf field of a units block against its expected category."""
    status: dict[str, _FieldStatus] = {}
    detail: dict[str, str] = {}

    def _check(field: str, token: str, expected: _Category) -> None:
        cat = _classify(token)
        if cat is None:
            status[field] = "unresolved"
            detail[field] = (
                f"'{token}' is not a recognized {expected} unit token; "
                "homogeneity for dependent parameters cannot be verified"
            )
        elif cat != expected:
            status[field] = "mismatch"
            detail[field] = f"expected a {expected} unit, got '{token}' which is {cat}-dimensioned"
        else:
            status[field] = "ok"

    _check("time", units.time, "time")
    _check("amount", units.amount, "mass")
    _check("volume", units.volume, "volume")

    if "/" in units.concentration:
        num, _, den = units.concentration.partition("/")
        _check("concentration_num", num, "mass")
        _check("concentration_den", den, "volume")
    else:
        msg = (
            "concentration must be a compound mass/volume unit (e.g. 'ng/mL'); "
            f"got '{units.concentration}' with no '/' separator"
        )
        status["concentration_num"] = "mismatch"
        status["concentration_den"] = "mismatch"
        detail["concentration_num"] = msg
        detail["concentration_den"] = msg

    return status, detail


class UnitConsistencyResult(BaseModel):
    """Result of checking a :class:`UnitsDeclaration` for self-consistency."""

    model_config = ConfigDict(frozen=True)

    status: Literal["consistent", "mismatched", "unresolved"]
    field_status: dict[str, str] = Field(default_factory=dict)
    field_detail: dict[str, str] = Field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-line summary of every non-``"ok"`` field."""
        problems = [
            f"{field}: {self.field_detail[field]}"
            for field, status in sorted(self.field_status.items())
            if status != "ok" and field in self.field_detail
        ]
        return "; ".join(problems) if problems else "units are self-consistent"


def check_units_consistency(units: UnitsDeclaration) -> UnitConsistencyResult:
    """Check whether ``units.volume`` is reachable from ``amount``/``concentration``.

    Returns ``status="mismatched"`` only when a field resolved to a
    recognized-but-wrong category (a genuine dimensional error);
    ``status="unresolved"`` when at least one token fell outside the
    recognized vocabulary (inconclusive, never a false-positive mismatch);
    ``status="consistent"`` otherwise.
    """
    field_status, field_detail = _field_statuses(units)
    if any(s == "mismatch" for s in field_status.values()):
        overall: Literal["consistent", "mismatched", "unresolved"] = "mismatched"
    elif any(s == "unresolved" for s in field_status.values()):
        overall = "unresolved"
    else:
        overall = "consistent"
    return UnitConsistencyResult(
        status=overall, field_status=field_status, field_detail=field_detail
    )


def _resolve_role_status(
    dimension: Dimension, field_status: dict[str, _FieldStatus]
) -> Literal["checked", "unchecked", "mismatched"]:
    deps = _DIMENSION_DEPENDENCIES[dimension]
    if not deps:
        return "checked"
    statuses = {field_status[d] for d in deps}
    if "mismatch" in statuses:
        return "mismatched"
    if "unresolved" in statuses:
        return "unchecked"
    return "checked"


class UnitMismatch(BaseModel):
    """One calibration parameter whose role dimension is broken by the units declaration."""

    model_config = ConfigDict(frozen=True)

    param: str
    expected_dimension: str
    detail: str


class UnitCoverageReport(BaseModel):
    """Per-parameter dimensional-homogeneity coverage for a compiled spec.

    ``status="not_declared"`` (all other fields empty) for a spec with no
    ``units:`` block. Otherwise ``status="checked"``: ``checked`` lists
    parameters whose role dimension was verified consistent, ``unchecked``
    lists parameters with no role lookup entry or an unresolvable
    (unrecognized-token) dependency, and ``mismatched`` lists genuine
    dimensional errors.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["checked", "not_declared"] = "not_declared"
    checked: list[str] = Field(default_factory=list)
    unchecked: list[str] = Field(default_factory=list)
    mismatched: list[UnitMismatch] = Field(default_factory=list)
    sigma_prop_warnings: list[str] = Field(default_factory=list)


def _observation_sigma_names(observation: ObservationModule) -> list[str]:
    """Return the sigma field names actually active on an ObservationModule.

    ``BLQM3``/``BLQM4`` always carry both ``sigma_prop``/``sigma_add`` on
    the Pydantic model (see their docstrings) but only the subset selected
    by ``error_model`` enters the likelihood -- ``active_sigmas()`` avoids
    reporting the vestigial default as "checked"/"mismatched".
    """
    if isinstance(observation, Proportional):
        return ["sigma_prop"]
    if isinstance(observation, Additive):
        return ["sigma_add"]
    if isinstance(observation, Combined):
        return ["sigma_prop", "sigma_add"]
    if isinstance(observation, BLQM3 | BLQM4):
        return observation.active_sigmas()
    return []


# SD-scale sigma_prop above this value implies a residual SD larger than
# the predicted concentration itself -- see module docstring for exactly
# what this catches (and does not catch).
_SIGMA_PROP_WARNING_THRESHOLD = 1.0


def _sigma_prop_heuristic_warnings(spec: DSLSpec) -> list[str]:
    warnings: list[str] = []
    observation = spec.observation
    sigma_prop: float | None = None
    if isinstance(observation, Proportional | Combined) or (
        isinstance(observation, BLQM3 | BLQM4) and "sigma_prop" in observation.active_sigmas()
    ):
        sigma_prop = observation.sigma_prop
    if sigma_prop is not None and sigma_prop > _SIGMA_PROP_WARNING_THRESHOLD:
        warnings.append(
            f"sigma_prop={sigma_prop} exceeds {_SIGMA_PROP_WARNING_THRESHOLD} "
            "(sigma_prop is standard-deviation scale, not variance -- see "
            "CHANGELOG.md Suite A note; check for an un-square-rooted NONMEM "
            "SIGMA variance value)"
        )
    return warnings


def unit_coverage_report(spec: DSLSpec) -> UnitCoverageReport:
    """Build the :class:`UnitCoverageReport` for a compiled spec.

    Enumerates :meth:`DSLSpec.calibration_param_names` (the flat
    ``initial:`` block roles) plus the observation module's active sigma
    fields (which live inline, not in ``initial:``) -- see
    :func:`_observation_sigma_names`.
    """
    if spec.units is None:
        return UnitCoverageReport(status="not_declared")

    consistency = check_units_consistency(spec.units)
    field_status = cast("dict[str, _FieldStatus]", consistency.field_status)

    checked: list[str] = []
    unchecked: list[str] = []
    mismatched: list[UnitMismatch] = []

    names = [*spec.calibration_param_names(), *_observation_sigma_names(spec.observation)]
    for name in names:
        dimension = _ROLE_DIMENSIONS.get(name)
        if dimension is None:
            unchecked.append(name)
            continue
        role_status = _resolve_role_status(dimension, field_status)
        if role_status == "checked":
            checked.append(name)
        elif role_status == "unchecked":
            unchecked.append(name)
        else:
            mismatched.append(
                UnitMismatch(
                    param=name,
                    expected_dimension=_DIMENSION_LABELS[dimension],
                    detail=consistency.summary(),
                )
            )

    return UnitCoverageReport(
        status="checked",
        checked=sorted(set(checked)),
        unchecked=sorted(set(unchecked)),
        mismatched=mismatched,
        sigma_prop_warnings=_sigma_prop_heuristic_warnings(spec),
    )


__all__ = [
    "AMOUNT",
    "AMOUNT_PER_TIME",
    "CLEARANCE",
    "CONCENTRATION",
    "RATE",
    "TIME",
    "UNITLESS",
    "VOLUME",
    "Dimension",
    "UnitConsistencyResult",
    "UnitCoverageReport",
    "UnitMismatch",
    "check_units_consistency",
    "unit_coverage_report",
]
