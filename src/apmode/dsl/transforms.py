# SPDX-License-Identifier: GPL-2.0-or-later
"""Formular transform types for the agentic backend (PRD §4.2.5).

Six allowed agent transforms that produce new DSLSpec instances from existing ones.
Each transform is validated before application. The agent cannot escape the grammar
or propose structures outside these typed operations.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apmode.dsl.ast_models import (
    IIV,
    IOV,
    AbsorptionModule,
    CovariateLink,
    DistributionModule,
    DSLSpec,
    EliminationModule,
    FirstOrder,
    LaggedFirstOrder,
    NODEAbsorption,
    NODEElimination,
    ObservationModule,
    Transit,
)
from apmode.dsl.normalize import normalize_param_name
from apmode.dsl.prior_transforms import SetPrior, apply_set_prior, validate_set_prior
from apmode.ids import generate_candidate_id

# ---------------------------------------------------------------------------
# Transform types (PRD §4.2.5 enumerated agent transforms)
# ---------------------------------------------------------------------------


class SwapModule(BaseModel):
    """swap_module(position, new_module) — replace an entire axis module."""

    model_config = ConfigDict(frozen=True)
    type: Literal["swap_module"] = "swap_module"
    position: Literal["absorption", "distribution", "elimination", "observation"]
    new_module: AbsorptionModule | DistributionModule | EliminationModule | ObservationModule


class AddCovariateLink(BaseModel):
    """add_covariate_link(param, covariate, form)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["add_covariate_link"] = "add_covariate_link"
    param: str
    covariate: str
    form: Literal["power", "exponential", "linear", "categorical", "maturation"]


class AdjustVariability(BaseModel):
    """adjust_variability(param, action: add|remove|upgrade_to_block)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["adjust_variability"] = "adjust_variability"
    param: str
    action: Literal["add", "remove", "upgrade_to_block"]


class SetTransitN(BaseModel):
    """set_transit_n(n) — change transit compartment count."""

    model_config = ConfigDict(frozen=True)
    type: Literal["set_transit_n"] = "set_transit_n"
    n: int = Field(ge=1)


class ToggleLag(BaseModel):
    """toggle_lag(on|off) — add or remove lag time on first-order absorption."""

    model_config = ConfigDict(frozen=True)
    type: Literal["toggle_lag"] = "toggle_lag"
    on: bool


class ReplaceWithNODE(BaseModel):
    """replace_submodel_with_NODE(position, constraint_template, dim).

    Discovery lane only; dim ≤ lane ceiling; template from enumerated set.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["replace_with_node"] = "replace_with_node"
    position: Literal["absorption", "elimination"]
    constraint_template: Literal[
        "monotone_increasing",
        "monotone_decreasing",
        "bounded_positive",
        "saturable",
        "unconstrained_smooth",
    ]
    dim: int = Field(ge=1, le=8)


FormularTransform = Annotated[
    SwapModule
    | AddCovariateLink
    | AdjustVariability
    | SetTransitN
    | ToggleLag
    | ReplaceWithNODE
    | SetPrior,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_swap_position(transform: SwapModule) -> str | None:
    """Validate that new_module type is appropriate for the target position."""
    from apmode.dsl import ast_models as m

    pos_types: dict[str, tuple[type, ...]] = {
        "absorption": (
            m.IVBolus,
            m.FirstOrder,
            m.ZeroOrder,
            m.LaggedFirstOrder,
            m.Transit,
            m.MixedFirstZero,
            m.NODEAbsorption,
        ),
        "distribution": (
            m.OneCmt,
            m.TwoCmt,
            m.ThreeCmt,
            m.TMDDCore,
            m.TMDDQSS,
        ),
        "elimination": (
            m.LinearElim,
            m.MichaelisMenten,
            m.ParallelLinearMM,
            m.TimeVaryingElim,
            m.NODEElimination,
        ),
        "observation": (
            m.Proportional,
            m.Additive,
            m.Combined,
            m.BLQM3,
            m.BLQM4,
        ),
    }
    expected = pos_types.get(transform.position)
    if expected is not None and not isinstance(transform.new_module, expected):
        return (
            f"Module {type(transform.new_module).__name__} is not valid "
            f"for position '{transform.position}'"
        )
    return None


def validate_transform(spec: DSLSpec, transform: FormularTransform) -> list[str]:
    """Validate a transform against the current spec. Returns error strings."""
    errors: list[str] = []

    if isinstance(transform, SwapModule):
        # Validate module type matches position
        _valid = _validate_swap_position(transform)
        if _valid:
            errors.append(_valid)

    elif isinstance(transform, AddCovariateLink):
        valid_params = set(spec.structural_param_names())
        np = normalize_param_name(transform.param)
        if np not in valid_params:
            errors.append(
                f"CovariateLink param '{transform.param}' not in structural params "
                f"{sorted(valid_params)}"
            )
        # Check for duplicate covariate link on same param+covariate (case-insensitive)
        for item in spec.variability:
            if (
                isinstance(item, CovariateLink)
                and normalize_param_name(item.param) == np
                and item.covariate.upper() == transform.covariate.upper()
            ):
                errors.append(
                    f"Duplicate CovariateLink: {transform.param}~{transform.covariate} "
                    f"already exists"
                )

    elif isinstance(transform, AdjustVariability):
        valid_params = set(spec.structural_param_names())
        np = normalize_param_name(transform.param)
        if transform.action in ("add", "upgrade_to_block") and np not in valid_params:
            errors.append(
                f"AdjustVariability param '{transform.param}' not in structural "
                f"params {sorted(valid_params)}"
            )

    elif isinstance(transform, SetTransitN):
        if not isinstance(spec.absorption, Transit):
            errors.append("set_transit_n requires Transit absorption module")

    elif isinstance(transform, ToggleLag):
        if transform.on and not isinstance(spec.absorption, (FirstOrder, LaggedFirstOrder)):
            errors.append("toggle_lag(on) requires FirstOrder or LaggedFirstOrder absorption")

    elif isinstance(transform, ReplaceWithNODE):
        if transform.position not in ("absorption", "elimination"):
            errors.append(
                f"ReplaceWithNODE position must be 'absorption' or 'elimination', "
                f"got '{transform.position}'"
            )

    elif isinstance(transform, SetPrior):
        errors.extend(validate_set_prior(spec, transform))

    return errors


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_transform(spec: DSLSpec, transform: FormularTransform) -> DSLSpec:
    """Apply a single transform to a spec, returning a new spec with a fresh model_id.

    Raises ValueError if the transform is invalid for the spec.
    """
    errors = validate_transform(spec, transform)
    if errors:
        msg = f"Transform validation failed: {'; '.join(errors)}"
        raise ValueError(msg)

    new_id = generate_candidate_id()
    absorption: AbsorptionModule = spec.absorption
    distribution: DistributionModule = spec.distribution
    elimination: EliminationModule = spec.elimination
    variability: list[IIV | CovariateLink | object] = list(spec.variability)
    observation: ObservationModule = spec.observation

    if isinstance(transform, SwapModule):
        if transform.position == "absorption":
            absorption = transform.new_module  # type: ignore[assignment]
        elif transform.position == "distribution":
            distribution = transform.new_module  # type: ignore[assignment]
        elif transform.position == "elimination":
            elimination = transform.new_module  # type: ignore[assignment]
        elif transform.position == "observation":
            observation = transform.new_module  # type: ignore[assignment]

    elif isinstance(transform, AddCovariateLink):
        variability = list(spec.variability)
        variability.append(
            CovariateLink(
                param=normalize_param_name(transform.param),
                covariate=transform.covariate,
                form=transform.form,
            )
        )

    elif isinstance(transform, AdjustVariability):
        # Normalize param before applying so case-insensitive references work
        normalized_transform = AdjustVariability(
            param=normalize_param_name(transform.param),
            action=transform.action,
        )
        variability = _apply_adjust_variability(spec, normalized_transform)

    elif isinstance(transform, SetTransitN):
        abs_mod = spec.absorption
        if isinstance(abs_mod, Transit):
            absorption = Transit(n=transform.n, ktr=abs_mod.ktr, ka=abs_mod.ka)

    elif isinstance(transform, ToggleLag):
        absorption = _apply_toggle_lag(spec, transform)

    elif isinstance(transform, ReplaceWithNODE):
        if transform.position == "absorption":
            absorption = NODEAbsorption(
                dim=transform.dim, constraint_template=transform.constraint_template
            )
        else:
            elimination = NODEElimination(
                dim=transform.dim, constraint_template=transform.constraint_template
            )

    elif isinstance(transform, SetPrior):
        # Delegates to apply_set_prior which handles replace-or-append semantics.
        # Return early — SetPrior does not touch structural modules or variability.
        return apply_set_prior(spec, transform)

    # Preserve priors across all non-SetPrior transforms — structural swaps
    # may orphan individual priors (pruned below via _prune_stale_variability),
    # but other transforms (AddCovariateLink, AdjustVariability, SetTransitN,
    # ToggleLag) keep the full prior set intact.
    new_spec = DSLSpec(
        model_id=new_id,
        absorption=absorption,
        distribution=distribution,
        elimination=elimination,
        variability=variability,
        observation=observation,
        priors=spec.priors,
    )

    # Prune stale variability AND priors after structural module swaps
    if isinstance(transform, (SwapModule, ReplaceWithNODE)):
        new_spec = _prune_stale_variability(new_spec)

    return new_spec


def _prune_stale_variability(spec: DSLSpec) -> DSLSpec:
    """Remove variability AND priors referring to params that no longer exist.

    Called after SwapModule/ReplaceWithNODE to keep IIV/CovariateLink/priors
    consistent. Preserves any priors that still target a valid parameter,
    drops orphaned ones (e.g., prior on ``ka`` after swap to IVBolus).
    """
    valid_params = set(spec.structural_param_names())
    cleaned: list[object] = []

    for item in spec.variability:
        if isinstance(item, IIV):
            kept = [p for p in item.params if p in valid_params]
            if kept:
                structure = item.structure
                if structure == "block" and len(kept) < 2:
                    structure = "diagonal"
                cleaned.append(IIV(params=kept, structure=structure))
        elif isinstance(item, IOV):
            kept = [p for p in item.params if p in valid_params]
            if kept:
                cleaned.append(IOV(params=kept, occasions=item.occasions))
        elif isinstance(item, CovariateLink):
            if item.param in valid_params:
                cleaned.append(item)
        else:
            cleaned.append(item)

    # Prune stale priors — priors targeting parameters removed by the structural swap.
    pruned_priors = [p for p in spec.priors if _prior_target_still_valid(p.target, valid_params)]

    return DSLSpec(
        model_id=spec.model_id,
        absorption=spec.absorption,
        distribution=spec.distribution,
        elimination=spec.elimination,
        variability=cleaned,
        observation=spec.observation,
        priors=pruned_priors,
    )


def _prior_target_still_valid(target: str, structural_params: set[str]) -> bool:
    """Check if a prior target still resolves to an existing parameter.

    - structural targets (e.g. "CL"): must be in structural_params
    - "omega_X" / "omega_iov_X": underlying X must be in structural_params
    - "beta_X_COV": underlying X must be in structural_params
    - "sigma_prop", "sigma_add", "corr_iiv": always kept (tied to obs/correlation block)
    """
    if target in structural_params:
        return True
    if target in ("sigma_prop", "sigma_add", "corr_iiv"):
        return True
    if target.startswith("omega_iov_"):
        return target[len("omega_iov_") :] in structural_params
    if target.startswith("omega_"):
        return target[len("omega_") :] in structural_params
    if target.startswith("beta_"):
        # beta_<PARAM>_<COVARIATE>; param may contain digits/underscores.
        # Conservative: keep if *any* structural param matches the prefix.
        rest = target[len("beta_") :]
        return any(rest.startswith(f"{p}_") for p in structural_params)
    return False


def _apply_adjust_variability(spec: DSLSpec, transform: AdjustVariability) -> list[object]:
    """Adjust IIV variability items.

    Targets the first IIV block containing (or appropriate for) the param.
    upgrade_to_block only upgrades the block containing the specified param.
    """
    var_list: list[object] = []
    action_applied = False

    for item in spec.variability:
        if isinstance(item, IIV):
            params = list(item.params)
            structure = item.structure

            if not action_applied:
                if transform.action == "add" and transform.param not in params:
                    params.append(transform.param)
                    action_applied = True
                elif transform.action == "remove" and transform.param in params:
                    params.remove(transform.param)
                    action_applied = True
                elif transform.action == "upgrade_to_block" and transform.param in params:
                    structure = "block"
                    action_applied = True

            if params:  # don't add empty IIV
                var_list.append(IIV(params=params, structure=structure))
        else:
            var_list.append(item)

    # If "add" and no IIV existed or none was modified, create one
    if transform.action == "add" and not action_applied:
        var_list.insert(0, IIV(params=[transform.param], structure="diagonal"))

    return var_list


def _apply_toggle_lag(spec: DSLSpec, transform: ToggleLag) -> AbsorptionModule:
    """Toggle lag time on first-order absorption."""
    abs_mod = spec.absorption

    if transform.on:
        if isinstance(abs_mod, FirstOrder):
            return LaggedFirstOrder(ka=abs_mod.ka, tlag=0.5)
        return abs_mod  # already lagged or other type
    else:
        if isinstance(abs_mod, LaggedFirstOrder):
            return FirstOrder(ka=abs_mod.ka)
        return abs_mod
