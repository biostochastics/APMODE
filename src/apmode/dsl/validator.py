# SPDX-License-Identifier: GPL-2.0-or-later
"""DSL semantic validator (ARCHITECTURE.md §2.2).

Enforces the constraint table from PRD §4.2.5:
- Volumes > 0, rates > 0, sigmas > 0
- NODE variant requires ``experimental.node`` opt-in before backend/lane checks
- NODE dim <= constraint_template max dim
- NODE dim <= lane ceiling
- NODE not admissible in Submission lane
- frac ∈ (0, 1), tlag >= 0, transit n >= 1
- Block IIV requires >= 2 params

Surfaces ALL violations (not fail-fast), matching the Pandera lazy=True philosophy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd

    from apmode.governance.policy import GatePolicy

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    DSLSpec,
    Erlang,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    ObservationModule,
    OneCmt,
    ParallelFirstOrder,
    ParallelLinearMM,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.capabilities import registered_emitters
from apmode.dsl.capabilities import report as capability_report
from apmode.dsl.errors import FrmCode
from apmode.dsl.lane import Lane
from apmode.dsl.normalize import normalize_param_name
from apmode.dsl.priors import prior_target_kinds, validate_priors
from apmode.dsl.spans import SourceSpan
from apmode.dsl.units import check_units_consistency

ValidationSeverity = Literal["error", "warning", "info"]

# NODE constraint template max dims (PRD §4.2.5 table)
_TEMPLATE_MAX_DIM: dict[str, int] = {
    "monotone_increasing": 4,
    "monotone_decreasing": 4,
    "bounded_positive": 6,
    "saturable": 4,
    "unconstrained_smooth": 8,
}

# Lane NODE dimension ceilings (PRD §4.2.5 table)
_LANE_DIM_CEILING: dict[Lane, int | None] = {
    Lane.SUBMISSION: None,  # NODE not admissible
    Lane.DISCOVERY: 8,
    Lane.OPTIMIZATION: 4,
}

# Absorption-form lane admissibility (ADR-0003 D6)
# True = admissible; False = rejected at Gate 2 with actionable error.
# All forms not enumerated here are admissible in every lane.
_LANE_ABSORPTION_INADMISSIBLE: dict[Lane, frozenset[str]] = {
    Lane.SUBMISSION: frozenset({"SumIG"}),  # SumIG academic-grade; not regulatory practice
    Lane.DISCOVERY: frozenset(),
    Lane.OPTIMIZATION: frozenset(),
}

# Erlang chain length cap (ADR-0003 D2): longer chains add little resolution
# and inflate state count quadratically in the explicit-chain emitter.
_ERLANG_MAX_N: int = 7

# SumIG component cap (ADR-0003 D1). Raising this is a validator-only change
# once the model contract supports more components.
_SUMIG_MAX_K: int = 2


class ValidationError(BaseModel):
    """A single semantic validation error."""

    model_config = ConfigDict(frozen=True)

    module: str
    param: str
    constraint: str
    message: str
    # #P0.1 (Formular sharpening plan §4 Phase 0): best-available source
    # position for the block this error concerns. DSLSpec.source_meta is
    # block-level (keyed by "absorption"/"distribution"/"elimination"/
    # "observation"/"variability[i]"), so a parameter-specific error still
    # only carries its enclosing block's span — there is no per-parameter
    # position tracking in the grammar today. None when the spec was built
    # programmatically (no parse tree, hence an empty source_meta).
    source_span: SourceSpan | None = None
    # #P0.2 (Formular sharpening plan §4 Phase 0): stable FRM-{TAXON}-NNN
    # identifier (see apmode.dsl.errors.FrmCode) so callers can pattern-match
    # on "which check fired" instead of parsing message prose. Every
    # ValidationError constructed by validate_dsl sets this explicitly;
    # there is no sentinel default because every check in this module is
    # already coded — an uncoded construction site would be a bug.
    code: str
    severity: ValidationSeverity = "error"
    # Short, actionable fix hint (e.g. "remove IIV(n) or use IIV(ktr, ka)").
    # None when no single fix is obviously correct (e.g. the SumIG
    # disposition-fixed check, which has several valid remediations).
    remediation: str | None = None


def _span_for(spec: DSLSpec, key: str) -> SourceSpan | None:
    """Look up the best-available :class:`SourceSpan` for a block key.

    ``key`` matches a ``DSLSpec.source_meta`` key: a top-level module name
    (``"absorption"``, ``"distribution"``, ``"elimination"``,
    ``"observation"``) or a ``"variability[i]"`` index. Returns ``None``
    when the spec has no parse-tree provenance (e.g. built programmatically).
    """
    point = spec.source_meta.get(key)
    if point is None:
        return None
    line, column = point
    return SourceSpan.from_point(line, column)


def validate_dsl(spec: DSLSpec, *, lane: Lane) -> list[ValidationError]:
    """Validate a DSLSpec against the constraint table for a given lane.

    Returns a list of all violations (empty list if valid).
    """
    errors: list[ValidationError] = []
    _validate_units(spec, errors)
    _validate_initial_values(spec, errors)
    _validate_absorption(spec, errors)
    _validate_distribution(spec, errors)
    _validate_elimination(spec, errors)
    _validate_observation(spec, errors)
    _validate_observations_multi(spec, errors)
    _validate_variability(spec, errors)
    _validate_covariates(spec, errors)
    _validate_priors(spec, errors)
    _validate_module_compatibility(spec, errors)
    _validate_node_experimental_gate(spec, errors)
    _validate_node_constraints(spec, lane, errors)
    _validate_lane_absorption_admissibility(spec, lane, errors)
    return errors


# Formular sharpening plan §4 Phase 1 (P1.4): ``kdecay`` (TimeVaryingElim) is
# the one calibration parameter that keeps a conventional default (0.1) when
# omitted from ``initial:`` — mirroring the pre-Phase-1 Pydantic field
# default that lived on the module itself. It is therefore excluded from
# the "missing" check but still counts as a valid name for the "unused"
# check (declaring it explicitly in ``initial:`` is not an error).
_OPTIONAL_INITIAL_PARAMS: frozenset[str] = frozenset({"kdecay"})
_INITIAL_DEFAULTS: dict[str, float] = {"kdecay": 0.1}


def _validate_initial_values(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Cross-check ``DSLSpec.initial`` against the spec's calibration params.

    Two directions, per Formular sharpening plan §4 Phase 1 (P1.4):
    a structural module referencing a calibration parameter absent from
    ``initial:`` (FRM-AST-012), or ``initial:`` declaring a value no
    structural module references (FRM-AST-013).
    """
    required = set(spec.calibration_param_names())
    present = set(spec.initial)

    initial_span = _span_for(spec, "initial")

    missing = sorted(required - present - _OPTIONAL_INITIAL_PARAMS)
    for name in missing:
        errors.append(
            ValidationError(
                module="initial",
                param=f"initial.{name}",
                constraint="initial_value_missing",
                message=(
                    f"Parameter '{name}' is used by a structural module but has "
                    "no value in the initial: block"
                ),
                source_span=initial_span,
                code=FrmCode.AST_INITIAL_VALUE_MISSING.value,
                remediation=f"Add '{name} = <value>' to the initial: block.",
            )
        )

    unused = sorted(present - required)
    for name in unused:
        errors.append(
            ValidationError(
                module="initial",
                param=f"initial.{name}",
                constraint="initial_value_unused",
                message=(
                    f"initial: declares a value for '{name}' but no structural "
                    "module references it"
                ),
                source_span=initial_span,
                code=FrmCode.AST_INITIAL_VALUE_UNUSED.value,
                remediation=(
                    f"Remove '{name}' from the initial: block, or reference it "
                    "from a structural module's parameter list."
                ),
            )
        )


def _validate_units(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Reject a self-inconsistent ``units:`` declaration (Formular sharpening plan §4 P1.3).

    Only ``status="mismatched"`` (a recognized-but-wrong unit category, or a
    malformed ``concentration``) fails here -- ``status="unresolved"``
    (an unrecognized token) is deliberately *not* an error: see
    ``apmode.dsl.units`` module docstring for why a false-positive
    mismatch on an unfamiliar-but-valid token would be worse than staying
    silent. A spec with no ``units:`` block is unaffected (nothing to
    check) -- see ``apmode.dsl.units.unit_coverage_report`` for the
    parallel non-fatal per-parameter coverage report.
    """
    if spec.units is None:
        return
    result = check_units_consistency(spec.units)
    if result.status != "mismatched":
        return
    errors.append(
        ValidationError(
            module="units",
            param="units",
            constraint="units_dimensional_homogeneity",
            message=f"units: block is dimensionally inconsistent: {result.summary()}",
            source_span=_span_for(spec, "units"),
            code=FrmCode.SEM_UNITS_INCONSISTENT.value,
            remediation=(
                "Declare volume as a metric volume unit reachable from "
                "amount/concentration via Volume = Amount / Concentration "
                "(e.g. amount=mg, concentration=ng/mL, volume=L)."
            ),
        )
    )


def _validate_node_experimental_gate(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Reject NODE variants that are not opted into via ``experimental.node``.

    No emitter (nlmixr2, Stan, FREM) has a working code path for
    ``NODEAbsorption``/``NODEElimination`` yet — they all raise
    ``NotImplementedError`` (see ``apmode.dsl.capabilities`` NODE tags).
    This is a hard semantic-level failure, independent of and prior to
    any lane-based rejection (:func:`_validate_node_constraints` below
    only rejects NODE in Submission lane; Discovery/Optimization would
    otherwise silently accept a construct with no working backend).
    """
    if spec.experimental.node:
        return

    node_modules: list[tuple[str, NODEAbsorption | NODEElimination]] = []
    if isinstance(spec.absorption, NODEAbsorption):
        node_modules.append(("absorption", spec.absorption))
    if isinstance(spec.elimination, NODEElimination):
        node_modules.append(("elimination", spec.elimination))

    for mod_name, _node in node_modules:
        errors.append(
            ValidationError(
                module=mod_name,
                param=f"{mod_name}.type",
                constraint="node_experimental_gate",
                message="NODE variant requires experimental.node opt-in",
                source_span=_span_for(spec, mod_name),
                code=FrmCode.LANE_NODE_EXPERIMENTAL_GATE.value,
                remediation=(
                    "add experimental node flag set to true to opt in, or remove the NODE variant"
                ),
            )
        )


def _validate_lane_absorption_admissibility(
    spec: DSLSpec, lane: Lane, errors: list[ValidationError]
) -> None:
    """Reject lane-inadmissible absorption forms at Gate 2 (ADR-0003 D6).

    SumIG is academic-grade and not yet standard regulatory practice;
    Submission-lane bundles must use a regulatorily conventional form.
    Discovery and Optimization admit every absorption variant.
    """
    inadmissible = _LANE_ABSORPTION_INADMISSIBLE.get(lane, frozenset())
    abs_type = spec.absorption.type
    if abs_type in inadmissible:
        errors.append(
            ValidationError(
                module="absorption",
                param="absorption.type",
                constraint="lane_absorption_admissibility",
                message=(
                    f"Absorption form '{abs_type}' is not admissible in "
                    f"{lane.value} lane (ADR-0003 D6). Use a regulatorily "
                    "conventional form (FirstOrder / LaggedFirstOrder / "
                    "Transit / Erlang / ParallelFirstOrder / MixedFirstZero) "
                    "for Submission, or run in Discovery/Optimization."
                ),
                source_span=_span_for(spec, "absorption"),
                code=FrmCode.LANE_ABSORPTION_ADMISSIBILITY.value,
                remediation=(
                    "Use FirstOrder / LaggedFirstOrder / Transit / Erlang / "
                    "ParallelFirstOrder / MixedFirstZero, or run in Discovery/"
                    "Optimization lane instead of Submission."
                ),
            )
        )


def _positive(
    module: str,
    param_name: str,
    value: float,
    errors: list[ValidationError],
    source_span: SourceSpan | None = None,
) -> None:
    if not math.isfinite(value) or value <= 0:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="positive",
                message=f"{param_name} must be > 0, got {value}",
                source_span=source_span,
                code=FrmCode.SEM_POSITIVE.value,
                remediation=f"Set {module}.{param_name} to a value > 0.",
            )
        )


def _non_negative(
    module: str,
    param_name: str,
    value: float,
    errors: list[ValidationError],
    source_span: SourceSpan | None = None,
) -> None:
    if not math.isfinite(value) or value < 0:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="non_negative",
                message=f"{param_name} must be >= 0, got {value}",
                source_span=source_span,
                code=FrmCode.SEM_NON_NEGATIVE.value,
                remediation=f"Set {module}.{param_name} to a value >= 0.",
            )
        )


def _unit_interval(
    module: str,
    param_name: str,
    value: float,
    errors: list[ValidationError],
    source_span: SourceSpan | None = None,
) -> None:
    """Strictly in (0, 1) — exclusive bounds."""
    if not math.isfinite(value) or value <= 0 or value >= 1:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="unit_interval",
                message=f"{param_name} must be in (0, 1), got {value}",
                source_span=source_span,
                code=FrmCode.SEM_UNIT_INTERVAL.value,
                remediation=f"Set {module}.{param_name} strictly between 0 and 1.",
            )
        )


def _positive_int(
    module: str,
    param_name: str,
    value: int,
    errors: list[ValidationError],
    source_span: SourceSpan | None = None,
) -> None:
    if value < 1:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="positive_int",
                message=f"{param_name} must be >= 1, got {value}",
                source_span=source_span,
                code=FrmCode.SEM_POSITIVE_INT.value,
                remediation=f"Set {module}.{param_name} to an integer >= 1.",
            )
        )


# --- Module-level validators ---


def _validate_absorption(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.absorption
    mod = "absorption"
    span = _span_for(spec, mod)

    if isinstance(m, FirstOrder):
        ka = spec.get_initial("ka")
        if ka is not None:
            _positive(mod, "ka", ka, errors, span)
    elif isinstance(m, ZeroOrder):
        dur = spec.get_initial("dur")
        if dur is not None:
            _positive(mod, "dur", dur, errors, span)
    elif isinstance(m, LaggedFirstOrder):
        ka = spec.get_initial("ka")
        if ka is not None:
            _positive(mod, "ka", ka, errors, span)
        tlag = spec.get_initial("tlag")
        if tlag is not None:
            _non_negative(mod, "tlag", tlag, errors, span)
    elif isinstance(m, Transit):
        _positive_int(mod, "n", m.n, errors, span)
        ktr = spec.get_initial("ktr")
        if ktr is not None:
            _positive(mod, "ktr", ktr, errors, span)
        ka = spec.get_initial("ka")
        if ka is not None:
            _positive(mod, "ka", ka, errors, span)
    elif isinstance(m, MixedFirstZero):
        ka = spec.get_initial("ka")
        if ka is not None:
            _positive(mod, "ka", ka, errors, span)
        dur = spec.get_initial("dur")
        if dur is not None:
            _positive(mod, "dur", dur, errors, span)
        frac = spec.get_initial("frac")
        if frac is not None:
            _unit_interval(mod, "frac", frac, errors, span)
    elif isinstance(m, Erlang):
        _positive_int(mod, "n", m.n, errors, span)
        if m.n > _ERLANG_MAX_N:
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"{mod}.n",
                    constraint="erlang_max_n",
                    message=(
                        f"Erlang.n={m.n} exceeds cap of {_ERLANG_MAX_N}; "
                        "longer chains add little resolution and inflate state "
                        "count. Use Transit absorption for n>7."
                    ),
                    source_span=span,
                    code=FrmCode.SEM_ERLANG_MAX_N.value,
                    remediation=(
                        f"Reduce Erlang.n to <= {_ERLANG_MAX_N}, or switch to "
                        "Transit absorption for longer chains."
                    ),
                )
            )
        ktr = spec.get_initial("ktr")
        if ktr is not None:
            _positive(mod, "ktr", ktr, errors, span)
    elif isinstance(m, ParallelFirstOrder):
        ka1 = spec.get_initial("ka1")
        if ka1 is not None:
            _positive(mod, "ka1", ka1, errors, span)
        ka2 = spec.get_initial("ka2")
        if ka2 is not None:
            _positive(mod, "ka2", ka2, errors, span)
        frac = spec.get_initial("frac")
        if frac is not None:
            _unit_interval(mod, "frac", frac, errors, span)
    elif isinstance(m, SumIG):
        _validate_sumig(spec, m, mod, errors)
    elif isinstance(m, NODEAbsorption):
        _positive_int(mod, "dim", m.dim, errors, span)
    elif isinstance(m, IVBolus):
        # No tunable structural parameters; nothing to validate.
        # An explicit branch keeps the audit trail complete and prevents
        # silent fall-through if new IVBolus fields are added later.
        pass


def _validate_sumig(
    spec: DSLSpec,
    m: SumIG,
    mod: str,
    errors: list[ValidationError],
) -> None:
    """SumIG-specific validation (ADR-0003 D1, D5).

    - k restricted to {1, 2}
    - All MT, RD2 strictly positive
    - weight_1 strictly in (0, 1)
    - MT_1 < MT_2 (positive-difference parameterisation; prevents label switching)
    - For k >= 2: disposition (CL/V/Q) must be fixed externally — checked
      against fixed-prior signal in spec.priors. A manifest-level
      disposition_fixed flag was planned as an additional dispatch-time
      signal (ADR-0003 D7) but is not yet implemented on EvidenceManifest;
      the priors-based check below is the only enforcement path today.
    """
    span = _span_for(spec, mod)

    # k in {1, 2} (structural; stays inline on the module)
    if m.k < 1 or m.k > _SUMIG_MAX_K:
        errors.append(
            ValidationError(
                module=mod,
                param=f"{mod}.k",
                constraint="sumig_k_range",
                message=(
                    f"SumIG.k={m.k} out of range; supported k ∈ "
                    f"[1, {_SUMIG_MAX_K}]. Path to k=3 is gated behind "
                    "the sumig_max_k policy knob (see ADR-0003 D1)."
                ),
                source_span=span,
                code=FrmCode.SEM_SUMIG_K_RANGE.value,
                remediation=f"Set SumIG.k to a value in [1, {_SUMIG_MAX_K}].",
            )
        )

    mt_1 = spec.get_initial("MT_1")
    mt_2 = spec.get_initial("MT_2")
    rd2_1 = spec.get_initial("RD2_1")
    rd2_2 = spec.get_initial("RD2_2")
    weight_1 = spec.get_initial("weight_1")

    if mt_1 is not None:
        _positive(mod, "MT_1", mt_1, errors, span)
    if m.k >= 2 and mt_2 is not None:
        _positive(mod, "MT_2", mt_2, errors, span)
    if rd2_1 is not None:
        _positive(mod, "RD2_1", rd2_1, errors, span)
    if m.k >= 2 and rd2_2 is not None:
        _positive(mod, "RD2_2", rd2_2, errors, span)
    if m.k >= 2 and weight_1 is not None:
        _unit_interval(mod, "weight_1", weight_1, errors, span)

    # Label-switching guard: MT_1 < MT_2 enforces a canonical ordering.
    # The emitter pairs MT_1 with weight_1 and MT_2 with (1-weight_1), so
    # without this constraint the same density is reachable from two
    # parameter combinations and the FOCEI gradient is ill-conditioned.
    if m.k >= 2 and mt_1 is not None and mt_2 is not None and mt_1 >= mt_2:
        errors.append(
            ValidationError(
                module=mod,
                param=f"{mod}.MT_1,MT_2",
                constraint="sumig_mt_ordering",
                message=(
                    f"SumIG MT_1={mt_1} must be < MT_2={mt_2} "
                    "(positive-difference parameterisation; prevents label switching)"
                ),
                source_span=span,
                code=FrmCode.SEM_SUMIG_MT_ORDERING.value,
                remediation="Swap MT_1/MT_2 (or their weight_1 pairing) so MT_1 < MT_2.",
            )
        )

    # Disposition-fixed cross-module check (ADR-0003 D5)
    if m.k >= 2 and not _disposition_priors_fixed(spec):
        errors.append(
            ValidationError(
                module=mod,
                param=f"{mod}.k",
                constraint="sumig_disposition_fixed",
                message=(
                    f"SumIG.k={m.k} requires disposition (CL/V/Q) to be fixed "
                    'externally via priors with source="fixed_external" on '
                    "all disposition params. (A manifest-driven IV-reference "
                    "path is planned per ADR-0003 D7 but not yet "
                    "implemented.) See ADR-0003 D5 / Csajka 2005 §4 / "
                    "Weiss 2022 §5."
                ),
                source_span=span,
                code=FrmCode.SEM_SUMIG_DISPOSITION_FIXED.value,
                remediation=(
                    'Add priors with source="fixed_external" on every '
                    "disposition parameter (CL/V/Q). (Manifest-driven "
                    "IV-reference detection is planned per ADR-0003 D7 but "
                    "not yet available.)"
                ),
            )
        )


def _disposition_priors_fixed(spec: DSLSpec) -> bool:
    """Check whether all disposition parameters have fixed-external priors.

    Returns True iff every disposition parameter present in the structural
    spec has a corresponding ``PriorSpec`` whose ``source ==
    "fixed_external"`` (or whose family is degenerate enough to imply
    fixation, e.g. a Normal with sigma ≤ 1e-6 — handled conservatively
    here as "explicit source tag only" so the gate is unambiguous).
    """
    disposition_params = {"CL", "V", "V1", "V2", "V3", "Q", "Q2", "Q3"}
    structural = set(spec.structural_param_names())
    required = disposition_params & structural
    if not required:
        return True  # nothing to fix
    fixed: set[str] = set()
    for prior in spec.priors:
        if getattr(prior, "source", None) == "fixed_external" and prior.target in required:
            fixed.add(prior.target)
    return required.issubset(fixed)


def _validate_distribution(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.distribution
    mod = "distribution"
    span = _span_for(spec, mod)

    if isinstance(m, OneCmt):
        v = spec.get_initial("V")
        if v is not None:
            _positive(mod, "V", v, errors, span)
    elif isinstance(m, TwoCmt):
        for name in ("V1", "V2", "Q"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)
    elif isinstance(m, ThreeCmt):
        for name in ("V1", "V2", "V3", "Q2", "Q3"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)
    elif isinstance(m, TMDDCore):
        for name in ("V", "R0", "kon", "koff", "kint"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)
    elif isinstance(m, TMDDQSS):
        for name in ("V", "R0", "KD", "kint"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)


def _validate_elimination(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.elimination
    mod = "elimination"
    span = _span_for(spec, mod)

    if isinstance(m, LinearElim):
        cl = spec.get_initial("CL")
        if cl is not None:
            _positive(mod, "CL", cl, errors, span)
    elif isinstance(m, MichaelisMenten):
        for name in ("Vmax", "Km"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)
    elif isinstance(m, ParallelLinearMM):
        for name in ("CL", "Vmax", "Km"):
            value = spec.get_initial(name)
            if value is not None:
                _positive(mod, name, value, errors, span)
    elif isinstance(m, TimeVaryingElim):
        cl = spec.get_initial("CL")
        if cl is not None:
            _positive(mod, "CL", cl, errors, span)
        # kdecay always has a value here (default 0.1 — see
        # _INITIAL_DEFAULTS), so this fires whenever a user supplies a
        # non-positive override.
        kdecay = spec.get_initial("kdecay", _INITIAL_DEFAULTS["kdecay"])
        assert kdecay is not None
        _positive(mod, "kdecay", kdecay, errors, span)
        # All three decay forms are supported as of v0.5.0 (plan §4 / #9):
        #   exponential:  CL(t) = CL * exp(-kdecay * t)
        #   half_life:    CL(t) = CL / (1 + kdecay * t)       (kdecay = ln(2)/t_half)
        #   linear:       CL(t) = max(CL * (1 - kdecay * t), 0)  [floored at 0]
        #
        # Linear-decay caveat: the max/fmax floor at t = 1/kdecay produces
        # a C0 kink. rxode2 LSODA with
        # FOCEI forward sensitivities handles the step-size reduction
        # natively; Stan's ode_rk45 error estimator assumes C1 smoothness
        # and may over-refine near the zero-crossing, producing HMC
        # divergences at large kdecay. When an initial-estimate kdecay
        # implies the zero-crossing falls within the typical observation
        # window (1/kdecay < t_obs_max), the emitter still runs — but
        # operators should expect reduced convergence reliability in the
        # Stan backend. A t_max-aware Gate 1 warning is a tracked
        # follow-on (M3+ infrastructure needs the manifest's t_obs_max
        # threaded through to the validator). The math itself is
        # correct — negative CL would be worse than a kink — so we
        # accept the adjoint limitation.
    elif isinstance(m, NODEElimination):
        _positive_int(mod, "dim", m.dim, errors, span)


def _validate_observation_module(
    mod: str,
    m: ObservationModule,
    span: SourceSpan | None,
    errors: list[ValidationError],
) -> None:
    """Validate one observation module's numeric constraints.

    Shared between the singular ``observation:`` field and every entry's
    ``error`` field in a multi-analyte ``observations:`` block so both syntax
    forms get identical sigma/loq_value positivity checking — see
    :func:`_validate_observation` / :func:`_validate_observations_multi`.
    """
    if isinstance(m, Proportional):
        _positive(mod, "sigma_prop", m.sigma_prop, errors, span)
    elif isinstance(m, Additive):
        _positive(mod, "sigma_add", m.sigma_add, errors, span)
    elif isinstance(m, Combined):
        _positive(mod, "sigma_prop", m.sigma_prop, errors, span)
        _positive(mod, "sigma_add", m.sigma_add, errors, span)
    elif isinstance(m, (BLQM3, BLQM4)):
        _positive(mod, "loq_value", m.loq_value, errors, span)
        # error_model selects which residual SD is live. Validate the
        # corresponding sigma(s); a zero or negative SD silently produces
        # degenerate likelihoods in the emitters, so catch it at the DSL
        # boundary. See _research/ROCRATE_INTEGRATION_PLAN.md and
        # dsl/ast_models.py:BLQM3/BLQM4 for the error_model contract.
        if m.error_model in {"proportional", "combined"}:
            _positive(mod, "sigma_prop", m.sigma_prop, errors, span)
        if m.error_model in {"additive", "combined"}:
            _positive(mod, "sigma_add", m.sigma_add, errors, span)


def _validate_observation(spec: DSLSpec, errors: list[ValidationError]) -> None:
    span = _span_for(spec, "observation")
    _validate_observation_module("observation", spec.observation, span, errors)


def _validate_observations_multi(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Validate a multi-analyte ``observations:`` block, if present (P1.7).

    Beyond the shared numeric checks (:func:`_validate_observation_module`),
    this checks two cross-entry/spec-level invariants no Pydantic model
    validator can see in isolation: every entry's ``dvid`` must be unique
    within the block (``FrmCode.AST_OBSERVATIONS_DVID_COLLISION``), and
    every entry's ``prediction`` must name a known state variable of the
    compiled model (``FrmCode.AST_OBSERVATIONS_PREDICTION_UNKNOWN``; see
    ``DSLSpec.known_prediction_variables``).
    """
    if spec.observations is None:
        return
    mod = "observations"
    known = spec.known_prediction_variables()
    seen_dvid: dict[int, str] = {}

    for i, (name, endpoint) in enumerate(spec.observations.items()):
        span = _span_for(spec, f"observations[{i}]")
        _validate_observation_module(f"{mod}.{name}", endpoint.error, span, errors)

        if endpoint.dvid in seen_dvid:
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"{mod}.{name}.dvid",
                    constraint="observations_dvid_collision",
                    message=(
                        f"observations: entry '{name}' claims dvid={endpoint.dvid}, "
                        f"already used by entry '{seen_dvid[endpoint.dvid]}'"
                    ),
                    source_span=span,
                    code=FrmCode.AST_OBSERVATIONS_DVID_COLLISION.value,
                    remediation=(
                        f"Assign a distinct dvid to '{name}' (or to '{seen_dvid[endpoint.dvid]}')."
                    ),
                )
            )
        else:
            seen_dvid[endpoint.dvid] = name

        if endpoint.prediction not in known:
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"{mod}.{name}.prediction",
                    constraint="observations_prediction_unknown",
                    message=(
                        f"observations: entry '{name}' references prediction "
                        f"'{endpoint.prediction}', which is not a known state "
                        f"variable of the compiled model; valid: {sorted(known)}"
                    ),
                    source_span=span,
                    code=FrmCode.AST_OBSERVATIONS_PREDICTION_UNKNOWN.value,
                    remediation=f"Set prediction to one of: {sorted(known)}.",
                )
            )

    declared = set(seen_dvid)
    expected = set(range(1, len(spec.observations) + 1))
    if declared != expected:
        errors.append(
            ValidationError(
                module=mod,
                param=f"{mod}.dvid",
                constraint="observations_dvid_sequence",
                message=(
                    "observations: DVID values must form the contiguous sequence "
                    f"1..{len(spec.observations)} for backend endpoint routing; "
                    f"got {sorted(declared)}"
                ),
                source_span=_span_for(spec, "observations"),
                code=FrmCode.AST_OBSERVATIONS_DVID_COLLISION.value,
                remediation=(
                    "Assign DVID values 1, 2, ... in the desired endpoint routing order."
                ),
            )
        )


_NO_VARIABILITY_PARAMS: frozenset[str] = frozenset({"n"})
"""Structural parameters that cannot have IIV/IOV.

Transit 'n' is structural topology, not an estimated parameter. Keep a
dedicated error so requests for IIV/IOV on n do not collapse into a generic
"unknown parameter" diagnostic.
"""


def _validate_variability(spec: DSLSpec, errors: list[ValidationError]) -> None:
    mod = "variability"
    valid_params = set(spec.structural_param_names())

    # Check for duplicate IIV parameters across all IIV blocks
    # Normalize param names for case-insensitive matching
    all_iiv_params: list[str] = []
    for i, item in enumerate(spec.variability):
        if isinstance(item, IIV):
            for p in item.params:
                np = normalize_param_name(p)
                if np in all_iiv_params:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param="variability.IIV.params",
                            constraint="iiv_no_duplicate_params",
                            message=(
                                f"Parameter '{p}' appears in multiple IIV blocks; "
                                f"each parameter may have IIV in at most one block"
                            ),
                            source_span=_span_for(spec, f"variability[{i}]"),
                            code=FrmCode.AST_IIV_NO_DUPLICATE_PARAMS.value,
                            remediation=(
                                f"Remove '{p}' from all but one IIV block, or "
                                "merge the blocks it appears in."
                            ),
                        )
                    )
                else:
                    all_iiv_params.append(np)

    for i, item in enumerate(spec.variability):
        item_span = _span_for(spec, f"variability[{i}]")
        if isinstance(item, IIV):
            if len(item.params) == 0:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].params",
                        constraint="non_empty",
                        message="IIV params must not be empty",
                        source_span=item_span,
                        code=FrmCode.AST_NON_EMPTY_PARAMS.value,
                        remediation=(
                            "Add at least one structural parameter to the IIV block's params list."
                        ),
                    )
                )
            if item.structure == "block" and len(item.params) < 2:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].structure",
                        constraint="block_min_params",
                        message="block structure requires >= 2 params",
                        source_span=item_span,
                        code=FrmCode.AST_BLOCK_MIN_PARAMS.value,
                        remediation=(
                            "Add a second parameter to the block, or switch "
                            'structure to "diagonal" for a single param.'
                        ),
                    )
                )
            for p in item.params:
                np = normalize_param_name(p)
                if np in _NO_VARIABILITY_PARAMS:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="no_variability_on_param",
                            message=(
                                f"Parameter '{p}' cannot have IIV; "
                                "it is structural topology, not an estimated parameter"
                            ),
                            source_span=item_span,
                            code=FrmCode.AST_NO_VARIABILITY_ON_PARAM.value,
                            remediation=f"Remove '{p}' from this IIV block's params list.",
                        )
                    )
                elif np not in valid_params:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="iiv_param_exists",
                            message=(
                                f"IIV param '{p}' does not match any structural "
                                f"parameter; valid: {sorted(valid_params)}"
                            ),
                            source_span=item_span,
                            code=FrmCode.AST_IIV_PARAM_EXISTS.value,
                            remediation=(
                                f"Replace '{p}' with one of the structural "
                                f"parameters: {sorted(valid_params)}."
                            ),
                        )
                    )
        elif isinstance(item, IOV):
            if len(item.params) == 0:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].params",
                        constraint="non_empty",
                        message="IOV params must not be empty",
                        source_span=item_span,
                        code=FrmCode.AST_NON_EMPTY_PARAMS.value,
                        remediation=(
                            "Add at least one structural parameter to the IOV block's params list."
                        ),
                    )
                )
            for p in item.params:
                np = normalize_param_name(p)
                if np in _NO_VARIABILITY_PARAMS:
                    # Mirror the IIV check: Transit ``n`` is topology, not a
                    # parameter with an eta-bearing back-transform.
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="no_variability_on_param",
                            message=(
                                f"Parameter '{p}' cannot have IOV; "
                                "it is structural topology, not an estimated parameter"
                            ),
                            source_span=item_span,
                            code=FrmCode.AST_NO_VARIABILITY_ON_PARAM.value,
                            remediation=f"Remove '{p}' from this IOV block's params list.",
                        )
                    )
                elif np not in valid_params:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="iov_param_exists",
                            message=(
                                f"IOV param '{p}' does not match any structural "
                                f"parameter; valid: {sorted(valid_params)}"
                            ),
                            source_span=item_span,
                            code=FrmCode.AST_IOV_PARAM_EXISTS.value,
                            remediation=(
                                f"Replace '{p}' with one of the structural "
                                f"parameters: {sorted(valid_params)}."
                            ),
                        )
                    )


def _validate_covariates(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Validate the top-level ``covariates:`` list (Formular sharpening plan §4 P1.6).

    Checks (moved here from ``_validate_variability`` now that
    ``CovariateLink`` lives in its own top-level ``spec.covariates`` list,
    not among ``IIV``/``IOV`` in ``spec.variability``):
    - no duplicate ``(param, covariate)`` pair (case-insensitive)
    - ``param`` must resolve to a structural parameter
    - form-specific reference-value bounds: ``power.ref`` and
      ``maturation.tm50``/``maturation.hill`` must be > 0 when present
      (the Pydantic-level ``CovariateLink`` validator only enforces field
      *presence* per form, not numeric ranges).
    """
    mod = "covariates"
    valid_params = set(spec.structural_param_names())

    seen_cov_links: set[tuple[str, str]] = set()
    for i, item in enumerate(spec.covariates):
        item_span = _span_for(spec, f"covariates[{i}]")
        key = (normalize_param_name(item.param), item.covariate.upper())
        if key in seen_cov_links:
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"covariates[{i}]",
                    constraint="covariate_link_no_duplicate",
                    message=(
                        f"Duplicate CovariateLink: {item.param}~{item.covariate} "
                        f"appears more than once"
                    ),
                    source_span=item_span,
                    code=FrmCode.AST_COVARIATE_LINK_NO_DUPLICATE.value,
                    remediation=(
                        f"Remove the duplicate {item.param}~{item.covariate} "
                        "covariate declaration."
                    ),
                )
            )
        else:
            seen_cov_links.add(key)

        np = normalize_param_name(item.param)
        if np not in valid_params:
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"covariates[{i}].param",
                    constraint="covariate_param_exists",
                    message=(
                        f"Covariate param '{item.param}' does not match "
                        f"any structural parameter; valid: {sorted(valid_params)}"
                    ),
                    source_span=item_span,
                    code=FrmCode.AST_COVARIATE_PARAM_EXISTS.value,
                    remediation=(
                        f"Replace '{item.param}' with one of the structural "
                        f"parameters: {sorted(valid_params)}."
                    ),
                )
            )

        numeric_checks: list[tuple[str, float | None]] = []
        if item.form == "power":
            numeric_checks.append(("ref", item.ref))
        elif item.form == "maturation":
            numeric_checks.append(("tm50", item.tm50))
            numeric_checks.append(("hill", item.hill))
        for field_name, value in numeric_checks:
            if value is not None and (not math.isfinite(value) or value <= 0):
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"covariates[{i}].{field_name}",
                        constraint="positive",
                        message=f"{field_name} must be > 0, got {value}",
                        source_span=item_span,
                        code=FrmCode.SEM_POSITIVE.value,
                        remediation=f"Set covariates[{i}].{field_name} to a value > 0.",
                    )
                )


def _validate_priors(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Validate programmatically constructed priors against the full spec.

    Grammar-authored priors pass through the same checks while lowering, but
    ``DSLSpec`` remains a public programmatic API. Running the canonical prior
    validator here closes that second construction path and also verifies the
    exact IIV/IOV/covariate/residual target namespace.
    """
    for message in validate_priors(
        spec.priors,
        set(spec.structural_param_names()),
        target_kinds=prior_target_kinds(spec),
    ):
        errors.append(
            ValidationError(
                module="priors",
                param="priors",
                constraint="prior_invalid_declaration",
                message=message,
                source_span=_span_for(spec, "priors"),
                code=FrmCode.PRIOR_INVALID_DECLARATION.value,
                remediation="Remove or correct the invalid prior declaration.",
            )
        )


def _validate_module_compatibility(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Validate cross-module compatibility constraints.

    TMDD distribution models route the selected classical elimination module
    through the free-drug amount in both emitters. NODE elimination has no
    TMDD lowering and would otherwise fall through to an undefined CL symbol.
    """
    if isinstance(spec.distribution, (TMDDCore, TMDDQSS)) and isinstance(
        spec.elimination, NODEElimination
    ):
        errors.append(
            ValidationError(
                module="distribution",
                param="distribution.type",
                constraint="tmdd_rejects_node_elim",
                message=(
                    f"TMDD distribution ({spec.distribution.type}) cannot be "
                    "paired with NODE elimination; TMDD lowering supports "
                    "classical Linear, MichaelisMenten, ParallelLinearMM, and "
                    f"TimeVarying elimination, got {spec.elimination.type}"
                ),
                source_span=_span_for(spec, "distribution"),
                code=FrmCode.AST_TMDD_REQUIRES_LINEAR_ELIM.value,
                remediation=("Use a classical elimination module when using a TMDD distribution."),
            )
        )


def _validate_node_constraints(spec: DSLSpec, lane: Lane, errors: list[ValidationError]) -> None:
    """Validate NODE module admissibility and dimension constraints."""
    node_modules: list[tuple[str, NODEAbsorption | NODEElimination]] = []
    if isinstance(spec.absorption, NODEAbsorption):
        node_modules.append(("absorption", spec.absorption))
    if isinstance(spec.elimination, NODEElimination):
        node_modules.append(("elimination", spec.elimination))

    for mod_name, node in node_modules:
        mod_span = _span_for(spec, mod_name)
        # Lane admissibility
        if lane == Lane.SUBMISSION:
            errors.append(
                ValidationError(
                    module=mod_name,
                    param=f"{mod_name}.type",
                    constraint="node_lane_admissibility",
                    message="NODE modules are not admissible in Submission lane",
                    source_span=mod_span,
                    code=FrmCode.LANE_NODE_ADMISSIBILITY.value,
                    remediation=(
                        "Replace the NODE module with a classical form, or "
                        "run this spec in Discovery/Optimization lane."
                    ),
                )
            )
            continue  # skip dim checks since NODE is inadmissible

        dim = node.dim
        template = node.constraint_template

        # Template max dim
        template_max = _TEMPLATE_MAX_DIM[template]
        if dim > template_max:
            errors.append(
                ValidationError(
                    module=mod_name,
                    param=f"{mod_name}.dim",
                    constraint="node_template_max_dim",
                    message=(
                        f"dim={dim} exceeds max dim={template_max} "
                        f"for constraint_template '{template}'"
                    ),
                    source_span=mod_span,
                    code=FrmCode.SEM_NODE_TEMPLATE_MAX_DIM.value,
                    remediation=(
                        f"Reduce {mod_name}.dim to <= {template_max}, or choose "
                        "a constraint_template with a higher max dim."
                    ),
                )
            )

        # Lane ceiling
        ceiling = _LANE_DIM_CEILING[lane]
        if ceiling is not None and dim > ceiling:
            errors.append(
                ValidationError(
                    module=mod_name,
                    param=f"{mod_name}.dim",
                    constraint="node_lane_dim_ceiling",
                    message=(f"dim={dim} exceeds {lane.value} lane ceiling of {ceiling}"),
                    source_span=mod_span,
                    code=FrmCode.LANE_NODE_DIM_CEILING.value,
                    remediation=(
                        f"Reduce {mod_name}.dim to <= {ceiling}, or run in a "
                        "lane with a higher NODE dimension ceiling."
                    ),
                )
            )


# --- Seven-level validator API ---
#
# The functions below are the DATA_BOUND / BACKEND_BOUND / POLICY_BOUND
# checks consumed by ``apmode.dsl.validation_levels.validate``. They are
# kept in this module (rather than ``validation_levels.py``) specifically
# so every ``FrmCode`` member they emit is discoverable by
# ``tests/unit/test_dsl_error_codes.py``'s source scan, which greps
# ``validator.py``/``grammar.py`` for ``FrmCode.<MEMBER>`` references —
# the same reason the existing syntax/ast/semantic/lane-bound checks live
# here rather than being physically split into seven separate modules.


def validate_data_bound(spec: DSLSpec, data: pd.DataFrame) -> list[ValidationError]:
    """Data-bound checks: does ``data`` satisfy what the spec declares.

    Scoped to checks that need the actual bound dataframe and are directly
    implied by the compiled spec:

    - A multi-analyte ``observations:`` block (``spec.observations``) needs
      a ``DVID`` column in ``data`` to route rows to each named endpoint.
    - Every ``covariates:`` entry's ``covariate`` name needs a matching
      column (exact or upper-cased — canonical PK data columns are
      conventionally upper-case, e.g. ``WT``/``SEX``; see
      ``apmode.data.adapters._RESERVED_COLUMNS``).

    - ``SumIG`` absorption needs dose metadata (``EVID``/``AMT`` plus
      ``NMID`` or ``ID``) and is single-dose-only here.

    Richer profiling (missingness, sparsity/richness classification,
    covariate balance) belongs to ``apmode.data.profiler``; this validator
    only emits spec-bound data contract errors.
    """
    errors: list[ValidationError] = []
    columns = set(data.columns)

    if spec.observations is not None and "DVID" not in columns:
        errors.append(
            ValidationError(
                module="data",
                param="data.DVID",
                constraint="data_dvid_column_missing",
                message=(
                    f"spec declares a multi-analyte observations: block "
                    f"({len(spec.observations)} entries) but the bound dataset "
                    "has no DVID column to route rows to each endpoint"
                ),
                code=FrmCode.DATA_REQUIRED_COLUMN_MISSING.value,
                remediation="Add a DVID column to the dataset identifying each analyte's rows.",
            )
        )

    for i, cov in enumerate(spec.covariates):
        if cov.covariate not in columns and cov.covariate.upper() not in columns:
            errors.append(
                ValidationError(
                    module="data",
                    param=f"covariates[{i}].covariate",
                    constraint="data_covariate_column_missing",
                    message=(
                        f"covariates: entry references covariate '{cov.covariate}' "
                        "which has no matching column in the bound dataset"
                    ),
                    code=FrmCode.DATA_COVARIATE_COLUMN_MISSING.value,
                    remediation=(
                        f"Add a '{cov.covariate}' column to the dataset, or "
                        "remove this covariate link."
                    ),
                )
            )

    if isinstance(spec.absorption, SumIG):
        required = {"EVID", "AMT"}
        missing = sorted(required - columns)
        if missing:
            errors.append(
                ValidationError(
                    module="data",
                    param="data",
                    constraint="data_sumig_dose_columns_missing",
                    message=f"SumIG requires dose columns {sorted(required)}; missing {missing}",
                    code=FrmCode.DATA_REQUIRED_COLUMN_MISSING.value,
                    remediation="Bind data with EVID and AMT columns before fitting SumIG.",
                )
            )
        else:
            subject_col = "NMID" if "NMID" in columns else "ID" if "ID" in columns else None
            if subject_col is None:
                errors.append(
                    ValidationError(
                        module="data",
                        param="data.ID",
                        constraint="data_sumig_subject_column_missing",
                        message=(
                            "SumIG single-dose validation requires NMID or ID in the bound dataset"
                        ),
                        code=FrmCode.DATA_REQUIRED_COLUMN_MISSING.value,
                        remediation="Add an NMID or ID subject column to the dataset.",
                    )
                )
            else:
                dose_rows = data[(data["EVID"].isin([1, 4])) & (data["AMT"] > 0)]
                subjects = data[subject_col].dropna().unique().tolist()
                dose_counts = dose_rows.groupby(subject_col).size().reindex(subjects, fill_value=0)
                bad_subjects = dose_counts[dose_counts != 1]
                has_implicit_doses = bool(
                    "ADDL" in columns and (dose_rows["ADDL"].fillna(0) > 0).any()
                )
                if dose_rows.empty or not bad_subjects.empty or has_implicit_doses:
                    details: list[str] = []
                    if not bad_subjects.empty:
                        details.append(
                            "subjects with dose-count != 1: "
                            + ", ".join(
                                f"{subject}={int(count)}"
                                for subject, count in bad_subjects.items()
                            )
                        )
                    if has_implicit_doses:
                        details.append("ADDL>0 creates implicit repeat doses")
                    errors.append(
                        ValidationError(
                            module="data",
                            param="data.AMT",
                            constraint="data_sumig_single_dose",
                            message=(
                                "SumIG is single-dose only in this Formular version; "
                                "every subject must have exactly one positive explicit dose "
                                "event and ADDL must be zero"
                                + (f" ({'; '.join(details)})" if details else "")
                            ),
                            code=FrmCode.SEM_SUMIG_DISPOSITION_FIXED.value,
                            remediation=(
                                "Use a single-dose dataset for SumIG or choose a supported "
                                "multi-dose absorption module."
                            ),
                        )
                    )

    return errors


def validate_backend_bound(spec: DSLSpec, backend: str) -> list[ValidationError]:
    """Backend-bound checks: can ``backend`` actually emit code for ``spec``.

    Delegates entirely to :func:`apmode.dsl.capabilities.report` (the
    code-derived capability matrix) — this function adds no new capability
    knowledge, it just turns a non-``"supported"`` status for any tag the
    spec exercises into a coded :class:`ValidationError`. From "can this
    backend emit this spec", ``"explicitly_unsupported"``, ``"unknown_gap"``,
    and ``"experimental_no_stable_backend"`` all answer no.
    """
    emitter_names = {e.name for e in registered_emitters()}
    if backend not in emitter_names:
        return [
            ValidationError(
                module="backend",
                param="backend",
                constraint="backend_unknown",
                message=(
                    f"Backend '{backend}' is not a registered emitter; "
                    f"known emitters: {sorted(emitter_names)}"
                ),
                code=FrmCode.BE_UNKNOWN_BACKEND.value,
                remediation=f"Use one of the registered emitters: {sorted(emitter_names)}.",
            )
        ]

    status_by_tag = capability_report(spec).get(backend, {})
    errors: list[ValidationError] = []
    for tag_value, status in sorted(status_by_tag.items()):
        if status == "supported":
            continue
        errors.append(
            ValidationError(
                module="backend",
                param=f"backend.{tag_value}",
                constraint="backend_capability_unsupported",
                message=(
                    f"Backend '{backend}' does not support capability "
                    f"'{tag_value}' (status={status})"
                ),
                code=FrmCode.BE_CAPABILITY_UNSUPPORTED.value,
                remediation=(
                    f"Remove the construct tagged '{tag_value}' from the spec, "
                    f"or choose a backend that supports it."
                ),
            )
        )
    return errors


def validate_policy_bound(spec: DSLSpec, lane: Lane, policy: GatePolicy) -> list[ValidationError]:
    """Policy-bound checks: the spec against a loaded ``GatePolicy``.

    These checks are cheap, unambiguous, and require only the policy object
    and the spec/lane already in hand.

    - ``policy.lane`` must match the lane validation was requested for —
      catches an operator accidentally loading e.g. ``discovery.json``
      while validating for Submission.
    - A NODE absorption/elimination module present while
      ``policy.gate2.node_eligible`` is false — this generalizes the
      hard-coded PRD §3 Submission-lane NODE ban
      (``FrmCode.LANE_NODE_ADMISSIBILITY``, always Submission-only) to
      whatever the *loaded policy* actually says, so a Discovery/
      Optimization deployment that has locally tightened
      ``gate2.node_eligible=false`` is also caught.

    Gate-threshold validation against actual candidate metrics (CWRES, VPC
    coverage, Gate 3 composite weights, etc.) needs a fitted candidate
    result, not just the compiled spec; see ``apmode.governance.gates`` for
    those checks.
    """
    errors: list[ValidationError] = []

    if policy.lane != lane:
        errors.append(
            ValidationError(
                module="policy",
                param="policy.lane",
                constraint="policy_lane_mismatch",
                message=(
                    f"Loaded policy targets lane '{policy.lane.value}' but "
                    f"validation was requested for lane '{lane.value}'"
                ),
                code=FrmCode.POLICY_LANE_MISMATCH.value,
                remediation=(
                    f"Load the policy for '{lane.value}' lane, or pass "
                    f"lane='{policy.lane.value}' to match the loaded policy."
                ),
            )
        )

    node_modules = [
        mod_name
        for mod_name, present in (
            ("absorption", isinstance(spec.absorption, NODEAbsorption)),
            ("elimination", isinstance(spec.elimination, NODEElimination)),
        )
        if present
    ]
    if node_modules and not policy.gate2.node_eligible:
        errors.append(
            ValidationError(
                module="policy",
                param="policy.gate2.node_eligible",
                constraint="policy_node_ineligible",
                message=(
                    f"Spec uses NODE module(s) ({', '.join(node_modules)}) but "
                    "the loaded policy sets gate2.node_eligible=false"
                ),
                code=FrmCode.POLICY_NODE_INELIGIBLE.value,
                remediation=(
                    "Set gate2.node_eligible=true in the policy, or remove "
                    "the NODE module(s) from the spec."
                ),
            )
        )

    return errors
