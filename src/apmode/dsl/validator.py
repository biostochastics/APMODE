# SPDX-License-Identifier: GPL-2.0-or-later
"""DSL semantic validator (ARCHITECTURE.md §2.2).

Enforces the constraint table from PRD §4.2.5:
- Volumes > 0, rates > 0, sigmas > 0
- NODE dim <= constraint_template max dim
- NODE dim <= lane ceiling
- NODE not admissible in Submission lane
- frac ∈ (0, 1), tlag >= 0, transit n >= 1
- Block IIV requires >= 2 params

Surfaces ALL violations (not fail-fast), matching the Pandera lazy=True philosophy.
"""

from __future__ import annotations

from dataclasses import dataclass

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.normalize import normalize_param_name

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


@dataclass(frozen=True)
class ValidationError:
    """A single semantic validation error."""

    module: str
    param: str
    constraint: str
    message: str


def validate_dsl(spec: DSLSpec, *, lane: Lane) -> list[ValidationError]:
    """Validate a DSLSpec against the constraint table for a given lane.

    Returns a list of all violations (empty list if valid).
    """
    errors: list[ValidationError] = []
    _validate_absorption(spec, errors)
    _validate_distribution(spec, errors)
    _validate_elimination(spec, errors)
    _validate_observation(spec, errors)
    _validate_variability(spec, errors)
    _validate_module_compatibility(spec, errors)
    _validate_node_constraints(spec, lane, errors)
    return errors


def _positive(module: str, param_name: str, value: float, errors: list[ValidationError]) -> None:
    if value <= 0:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="positive",
                message=f"{param_name} must be > 0, got {value}",
            )
        )


def _non_negative(
    module: str, param_name: str, value: float, errors: list[ValidationError]
) -> None:
    if value < 0:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="non_negative",
                message=f"{param_name} must be >= 0, got {value}",
            )
        )


def _unit_interval(
    module: str, param_name: str, value: float, errors: list[ValidationError]
) -> None:
    """Strictly in (0, 1) — exclusive bounds."""
    if value <= 0 or value >= 1:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="unit_interval",
                message=f"{param_name} must be in (0, 1), got {value}",
            )
        )


def _positive_int(module: str, param_name: str, value: int, errors: list[ValidationError]) -> None:
    if value < 1:
        errors.append(
            ValidationError(
                module=module,
                param=f"{module}.{param_name}",
                constraint="positive_int",
                message=f"{param_name} must be >= 1, got {value}",
            )
        )


# --- Module-level validators ---


def _validate_absorption(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.absorption
    mod = "absorption"

    if isinstance(m, FirstOrder):
        _positive(mod, "ka", m.ka, errors)
    elif isinstance(m, ZeroOrder):
        _positive(mod, "dur", m.dur, errors)
    elif isinstance(m, LaggedFirstOrder):
        _positive(mod, "ka", m.ka, errors)
        _non_negative(mod, "tlag", m.tlag, errors)
    elif isinstance(m, Transit):
        _positive_int(mod, "n", m.n, errors)
        _positive(mod, "ktr", m.ktr, errors)
        _positive(mod, "ka", m.ka, errors)
    elif isinstance(m, MixedFirstZero):
        _positive(mod, "ka", m.ka, errors)
        _positive(mod, "dur", m.dur, errors)
        _unit_interval(mod, "frac", m.frac, errors)
    elif isinstance(m, NODEAbsorption):
        _positive_int(mod, "dim", m.dim, errors)


def _validate_distribution(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.distribution
    mod = "distribution"

    if isinstance(m, OneCmt):
        _positive(mod, "V", m.V, errors)
    elif isinstance(m, TwoCmt):
        _positive(mod, "V1", m.V1, errors)
        _positive(mod, "V2", m.V2, errors)
        _positive(mod, "Q", m.Q, errors)
    elif isinstance(m, ThreeCmt):
        _positive(mod, "V1", m.V1, errors)
        _positive(mod, "V2", m.V2, errors)
        _positive(mod, "V3", m.V3, errors)
        _positive(mod, "Q2", m.Q2, errors)
        _positive(mod, "Q3", m.Q3, errors)
    elif isinstance(m, TMDDCore):
        _positive(mod, "V", m.V, errors)
        _positive(mod, "R0", m.R0, errors)
        _positive(mod, "kon", m.kon, errors)
        _positive(mod, "koff", m.koff, errors)
        _positive(mod, "kint", m.kint, errors)
    elif isinstance(m, TMDDQSS):
        _positive(mod, "V", m.V, errors)
        _positive(mod, "R0", m.R0, errors)
        _positive(mod, "KD", m.KD, errors)
        _positive(mod, "kint", m.kint, errors)


def _validate_elimination(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.elimination
    mod = "elimination"

    if isinstance(m, LinearElim):
        _positive(mod, "CL", m.CL, errors)
    elif isinstance(m, MichaelisMenten):
        _positive(mod, "Vmax", m.Vmax, errors)
        _positive(mod, "Km", m.Km, errors)
    elif isinstance(m, ParallelLinearMM):
        _positive(mod, "CL", m.CL, errors)
        _positive(mod, "Vmax", m.Vmax, errors)
        _positive(mod, "Km", m.Km, errors)
    elif isinstance(m, TimeVaryingElim):
        _positive(mod, "CL", m.CL, errors)
        _positive(mod, "kdecay", m.kdecay, errors)
        if m.decay_fn != "exponential":
            errors.append(
                ValidationError(
                    module=mod,
                    param=f"{mod}.decay_fn",
                    constraint="supported_decay_fn",
                    message=(
                        f"decay_fn '{m.decay_fn}' is not yet implemented; "
                        f"only 'exponential' is supported in Phase 1"
                    ),
                )
            )
    elif isinstance(m, NODEElimination):
        _positive_int(mod, "dim", m.dim, errors)


def _validate_observation(spec: DSLSpec, errors: list[ValidationError]) -> None:
    m = spec.observation
    mod = "observation"

    if isinstance(m, Proportional):
        _positive(mod, "sigma_prop", m.sigma_prop, errors)
    elif isinstance(m, Additive):
        _positive(mod, "sigma_add", m.sigma_add, errors)
    elif isinstance(m, Combined):
        _positive(mod, "sigma_prop", m.sigma_prop, errors)
        _positive(mod, "sigma_add", m.sigma_add, errors)
    elif isinstance(m, (BLQM3, BLQM4)):
        _positive(mod, "loq_value", m.loq_value, errors)


_NO_VARIABILITY_PARAMS: frozenset[str] = frozenset({"n"})
"""Structural parameters that cannot have IIV/IOV/CovariateLink.

Transit 'n' is estimated via log/exp transform but the nlmixr2/Stan emitters
do not apply eta or covariate effects to its back-transform. Allowing
variability on n would produce code that silently ignores the effect.
"""


def _validate_variability(spec: DSLSpec, errors: list[ValidationError]) -> None:
    mod = "variability"
    valid_params = set(spec.structural_param_names())

    # Check for duplicate IIV parameters across all IIV blocks
    # Normalize param names for case-insensitive matching
    all_iiv_params: list[str] = []
    for item in spec.variability:
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
                        )
                    )
                else:
                    all_iiv_params.append(np)

    # Check for duplicate CovariateLinks (same param+covariate, case-insensitive)
    seen_cov_links: set[tuple[str, str]] = set()
    for i, item in enumerate(spec.variability):
        if isinstance(item, CovariateLink):
            key = (normalize_param_name(item.param), item.covariate.upper())
            if key in seen_cov_links:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}]",
                        constraint="covariate_link_no_duplicate",
                        message=(
                            f"Duplicate CovariateLink: {item.param}~{item.covariate} "
                            f"appears more than once"
                        ),
                    )
                )
            else:
                seen_cov_links.add(key)

    for i, item in enumerate(spec.variability):
        if isinstance(item, IIV):
            if len(item.params) == 0:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].params",
                        constraint="non_empty",
                        message="IIV params must not be empty",
                    )
                )
            if item.structure == "block" and len(item.params) < 2:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].structure",
                        constraint="block_min_params",
                        message="block structure requires >= 2 params",
                    )
                )
            for p in item.params:
                np = normalize_param_name(p)
                if np not in valid_params:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="iiv_param_exists",
                            message=(
                                f"IIV param '{p}' does not match any structural "
                                f"parameter; valid: {sorted(valid_params)}"
                            ),
                        )
                    )
                elif np in _NO_VARIABILITY_PARAMS:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="no_variability_on_param",
                            message=(
                                f"Parameter '{p}' cannot have IIV; "
                                f"emitter does not apply eta to its back-transform"
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
                    )
                )
            for p in item.params:
                np = normalize_param_name(p)
                if np not in valid_params:
                    errors.append(
                        ValidationError(
                            module=mod,
                            param=f"variability[{i}].params",
                            constraint="iov_param_exists",
                            message=(
                                f"IOV param '{p}' does not match any structural "
                                f"parameter; valid: {sorted(valid_params)}"
                            ),
                        )
                    )
        elif isinstance(item, CovariateLink):
            # Covariate column name is checked at data-binding time, but
            # param name must reference a structural parameter in the spec
            np = normalize_param_name(item.param)
            if np not in valid_params:
                errors.append(
                    ValidationError(
                        module=mod,
                        param=f"variability[{i}].param",
                        constraint="covariate_param_exists",
                        message=(
                            f"CovariateLink param '{item.param}' does not match "
                            f"any structural parameter; valid: {sorted(valid_params)}"
                        ),
                    )
                )


def _validate_module_compatibility(spec: DSLSpec, errors: list[ValidationError]) -> None:
    """Validate cross-module compatibility constraints.

    TMDD distribution models emit kel = CL/V in the dynamics, so they
    require an elimination module that provides CL (LinearElim only).
    ParallelLinearMM has CL but its MM term is not wired into TMDD dynamics,
    so allowing it would silently drop Vmax/Km.
    """
    if isinstance(spec.distribution, (TMDDCore, TMDDQSS)) and not isinstance(
        spec.elimination, LinearElim
    ):
        errors.append(
            ValidationError(
                module="distribution",
                param="distribution.type",
                constraint="tmdd_requires_linear_elim",
                message=(
                    f"TMDD distribution ({spec.distribution.type}) requires "
                    f"Linear elimination (provides CL for kel = CL/V); "
                    f"got {spec.elimination.type}"
                ),
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
        # Lane admissibility
        if lane == Lane.SUBMISSION:
            errors.append(
                ValidationError(
                    module=mod_name,
                    param=f"{mod_name}.type",
                    constraint="node_lane_admissibility",
                    message="NODE modules are not admissible in Submission lane",
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
                )
            )
