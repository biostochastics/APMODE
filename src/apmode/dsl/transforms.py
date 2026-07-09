# SPDX-License-Identifier: GPL-2.0-or-later
"""Formular transform types for the agentic backend (PRD §4.2.5).

Six allowed agent transforms that produce new DSLSpec instances from existing ones.
Each transform is validated before application. The agent cannot escape the grammar
or propose structures outside these typed operations.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apmode.dsl.ast_models import (
    IIV,
    IOV,
    AbsorptionModule,
    CovariateLink,
    DistributionModule,
    DSLSpec,
    EliminationModule,
    Erlang,
    FirstOrder,
    LaggedFirstOrder,
    NODEAbsorption,
    NODEElimination,
    ObservationModule,
    OccasionByStudy,
    OccasionSpec,
    ParallelFirstOrder,
    SumIG,
    Transit,
    _require_covariate_fields,
)
from apmode.dsl.normalize import normalize_param_name
from apmode.dsl.prior_transforms import SetPrior, apply_set_prior, validate_set_prior
from apmode.ids import generate_candidate_id

# ---------------------------------------------------------------------------
# Transform types (PRD §4.2.5 enumerated agent transforms)
# ---------------------------------------------------------------------------


class SwapModule(BaseModel):
    """swap_module(position, new_module, initial_overrides) — replace an entire axis module.

    ``new_module`` is structural-only (Formular sharpening plan §4 Phase 1,
    P1.4 split calibration values out of every structural module), so when
    the swap introduces calibration parameters the old module did not have
    (e.g. absorption FirstOrder -> ZeroOrder needs ``dur``, not ``ka``),
    the agent must supply their values via ``initial_overrides``.
    ``apply_transform`` rejects the transform if any calibration parameter
    required by ``new_module`` is still missing after merging
    ``initial_overrides`` on top of the carried-forward ``spec.initial``.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["swap_module"] = "swap_module"
    position: Literal["absorption", "distribution", "elimination", "observation"]
    new_module: AbsorptionModule | DistributionModule | EliminationModule | ObservationModule
    initial_overrides: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


class AddCovariateLink(BaseModel):
    """add_covariate_link(param, covariate, form, ...).

    Formular sharpening plan §4 Phase 1 (P1.6): mirrors
    :class:`~apmode.dsl.ast_models.CovariateLink`'s per-form field
    contract exactly (same required/forbidden field set per ``form``,
    enforced by the same :func:`~apmode.dsl.ast_models._require_covariate_fields`
    helper) so the agent must supply the same reference-value fields a
    human author would in ``covariates:`` text.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["add_covariate_link"] = "add_covariate_link"
    param: str
    covariate: str
    form: Literal["power", "exponential", "linear", "categorical", "maturation"]
    theta: float | None = None
    ref: float | None = None
    reference: str | None = None
    tm50: float | None = None
    hill: float | None = None
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_field_shape(self) -> AddCovariateLink:
        _require_covariate_fields(
            self.form,
            theta=self.theta,
            ref=self.ref,
            reference=self.reference,
            tm50=self.tm50,
            hill=self.hill,
        )
        return self


class AdjustVariability(BaseModel):
    """adjust_variability(param, action: add|remove|upgrade_to_block|add_iov|remove_iov).

    ``add_iov``/``remove_iov`` construct or mutate an :class:`~apmode.dsl.ast_models.IOV`
    item instead of the IIV items the other three actions target. ``occasions``
    is only meaningful for ``add_iov``; when omitted it defaults to
    :class:`~apmode.dsl.ast_models.OccasionByStudy`, matching
    ``search/candidates.py``'s ``force_iov`` root-candidate default.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["adjust_variability"] = "adjust_variability"
    param: str
    action: Literal["add", "remove", "upgrade_to_block", "add_iov", "remove_iov"]
    occasions: OccasionSpec | None = None
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


class SetTransitN(BaseModel):
    """set_transit_n(n) — change transit compartment count."""

    model_config = ConfigDict(frozen=True)
    type: Literal["set_transit_n"] = "set_transit_n"
    n: int = Field(ge=1)
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


class ToggleLag(BaseModel):
    """toggle_lag(on|off) — add or remove lag time on first-order absorption."""

    model_config = ConfigDict(frozen=True)
    type: Literal["toggle_lag"] = "toggle_lag"
    on: bool
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


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
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


# v0.7 SOTA absorption transforms (ADR-0003 D2, D3, D5).
# These are the *only* path the agent has to reach the new absorption
# variants — initial placement of arbitrary new absorption modules still
# happens via SwapModule, but each transform below is allowlisted as a
# narrow agent move so search-space expansion is bounded.


class ConvertTransitToErlang(BaseModel):
    """Convert Transit → Erlang(n, ktr).

    Drops the terminal first-order ka step and locks n to an integer.
    Requires the current absorption to be Transit. The agent's only path
    to Erlang (ADR-0003 D2 — keeps search-space expansion bounded).
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["convert_transit_to_erlang"] = "convert_transit_to_erlang"
    n: int = Field(ge=1, le=7)
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


class AddParallelRoute(BaseModel):
    """Convert FirstOrder → ParallelFirstOrder(ka1, ka2, frac).

    Splits a single first-order absorption into two parallel routes
    (fast + slow). Requires current absorption to be FirstOrder.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["add_parallel_route"] = "add_parallel_route"
    ka2: float = Field(gt=0)
    frac: float = Field(gt=0, lt=1)
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


class SetSumIGComponents(BaseModel):
    """Set/update SumIG component parameters.

    Requires the current absorption to already be SumIG. v0.7 hard-codes
    k=2; the path to k=3 is a future validator-only change.

    The validator enforces MT_1 < MT_2 (label-switching guard) and the
    cross-module disposition_fixed check (ADR-0003 D5).
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["set_sumig_components"] = "set_sumig_components"
    MT_1: float = Field(gt=0)
    MT_2: float = Field(gt=0)
    RD2_1: float = Field(gt=0)
    RD2_2: float = Field(gt=0)
    weight_1: float = Field(gt=0, lt=1)
    rationale: str = ""
    expected_diagnostic_effect: list[str] = Field(default_factory=list)


FormularTransform = Annotated[
    SwapModule
    | AddCovariateLink
    | AdjustVariability
    | SetTransitN
    | ToggleLag
    | ReplaceWithNODE
    | ConvertTransitToErlang
    | AddParallelRoute
    | SetSumIGComponents
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
            m.Erlang,
            m.ParallelFirstOrder,
            m.SumIG,
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
        for item in spec.covariates:
            if (
                normalize_param_name(item.param) == np
                and item.covariate.upper() == transform.covariate.upper()
            ):
                errors.append(
                    f"Duplicate CovariateLink: {transform.param}~{transform.covariate} "
                    f"already exists"
                )

    elif isinstance(transform, AdjustVariability):
        valid_params = set(spec.structural_param_names())
        np = normalize_param_name(transform.param)
        if transform.action in ("add", "upgrade_to_block", "add_iov") and np not in valid_params:
            errors.append(
                f"AdjustVariability param '{transform.param}' not in structural "
                f"params {sorted(valid_params)}"
            )
        if transform.action == "add_iov":
            target_occasions = _resolve_iov_occasions(transform)
            for var_item in spec.variability:
                if (
                    isinstance(var_item, IOV)
                    and np in var_item.params
                    and var_item.occasions != target_occasions
                ):
                    errors.append(
                        f"AdjustVariability add_iov: param '{transform.param}' already "
                        f"has IOV under occasions {var_item.occasions!r}, cannot add under "
                        f"{target_occasions!r} — a param cannot have two different IOV "
                        "occasion partitions at once"
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

    elif isinstance(transform, ConvertTransitToErlang):
        if not isinstance(spec.absorption, Transit):
            errors.append(
                "convert_transit_to_erlang requires Transit absorption "
                f"(got {type(spec.absorption).__name__})"
            )

    elif isinstance(transform, AddParallelRoute):
        if not isinstance(spec.absorption, FirstOrder):
            errors.append(
                "add_parallel_route requires FirstOrder absorption "
                f"(got {type(spec.absorption).__name__})"
            )

    elif isinstance(transform, SetSumIGComponents):
        if not isinstance(spec.absorption, SumIG):
            errors.append(
                "set_sumig_components requires SumIG absorption "
                f"(got {type(spec.absorption).__name__})"
            )
        if transform.MT_1 >= transform.MT_2:
            errors.append(
                f"set_sumig_components: MT_1={transform.MT_1} must be < "
                f"MT_2={transform.MT_2} (label-switching guard)"
            )

    elif isinstance(transform, SetPrior):
        errors.extend(validate_set_prior(spec, transform))

    return errors


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_transform(spec: DSLSpec, transform: FormularTransform) -> DSLSpec:
    """Apply a single transform to a spec, returning a new spec with a fresh model_id.

    Raises ValueError if the transform is invalid for the spec, or if the
    resulting spec would leave a calibration parameter without an
    ``initial:`` value (Formular sharpening plan §4 Phase 1, P1.4 — every
    structural-module calibration parameter must resolve to a value in
    ``DSLSpec.initial``).
    """
    errors = validate_transform(spec, transform)
    if errors:
        msg = f"Transform validation failed: {'; '.join(errors)}"
        raise ValueError(msg)

    new_id = generate_candidate_id()
    absorption: AbsorptionModule = spec.absorption
    distribution: DistributionModule = spec.distribution
    elimination: EliminationModule = spec.elimination
    variability: list[IIV | IOV | object] = list(spec.variability)
    covariates: list[CovariateLink] = list(spec.covariates)
    observation: ObservationModule = spec.observation
    # Calibration values this transform introduces/changes on top of
    # spec.initial. Only names that are new or whose value the transform
    # explicitly sets go here — unchanged names carry forward automatically
    # via the prune-and-keep pass below.
    initial_overrides: dict[str, float] = {}

    if isinstance(transform, SwapModule):
        if transform.position == "absorption":
            absorption = transform.new_module  # type: ignore[assignment]
        elif transform.position == "distribution":
            distribution = transform.new_module  # type: ignore[assignment]
        elif transform.position == "elimination":
            elimination = transform.new_module  # type: ignore[assignment]
        elif transform.position == "observation":
            observation = transform.new_module  # type: ignore[assignment]
        initial_overrides = dict(transform.initial_overrides)

    elif isinstance(transform, AddCovariateLink):
        covariates = list(spec.covariates)
        covariates.append(
            CovariateLink(
                param=normalize_param_name(transform.param),
                covariate=transform.covariate,
                form=transform.form,
                theta=transform.theta,
                ref=transform.ref,
                reference=transform.reference,
                tm50=transform.tm50,
                hill=transform.hill,
            )
        )

    elif isinstance(transform, AdjustVariability):
        # Normalize param before applying so case-insensitive references work
        normalized_transform = AdjustVariability(
            param=normalize_param_name(transform.param),
            action=transform.action,
            occasions=transform.occasions,
        )
        variability = _apply_adjust_variability(spec, normalized_transform)

    elif isinstance(transform, SetTransitN):
        abs_mod = spec.absorption
        if isinstance(abs_mod, Transit):
            absorption = Transit(n=transform.n)

    elif isinstance(transform, ToggleLag):
        absorption = _apply_toggle_lag(spec, transform)
        if transform.on and isinstance(absorption, LaggedFirstOrder):
            initial_overrides = {"tlag": spec.initial.get("tlag", 0.5)}

    elif isinstance(transform, ReplaceWithNODE):
        if transform.position == "absorption":
            absorption = NODEAbsorption(
                dim=transform.dim, constraint_template=transform.constraint_template
            )
        else:
            elimination = NODEElimination(
                dim=transform.dim, constraint_template=transform.constraint_template
            )

    elif isinstance(transform, ConvertTransitToErlang):
        # Drop terminal ka, lock n to integer. ktr carries forward under the
        # same key (Transit.ktr and Erlang.ktr are both named "ktr" in
        # spec.initial) via the prune-and-keep pass below.
        prev = spec.absorption
        if isinstance(prev, Transit):
            absorption = Erlang(n=transform.n)

    elif isinstance(transform, AddParallelRoute):
        # Convert FirstOrder(ka) → ParallelFirstOrder(ka1, ka2, frac).
        prev = spec.absorption
        if isinstance(prev, FirstOrder):
            absorption = ParallelFirstOrder()
            initial_overrides = {
                "ka1": spec.initial.get("ka", 1.0),
                "ka2": transform.ka2,
                "frac": transform.frac,
            }

    elif isinstance(transform, SetSumIGComponents):
        # k (structural) is preserved from the current spec; the
        # calibration values below replace whatever was previously in
        # spec.initial for this SumIG's components.
        prev = spec.absorption
        if isinstance(prev, SumIG):
            absorption = SumIG(k=prev.k)
            initial_overrides = {
                "MT_1": transform.MT_1,
                "MT_2": transform.MT_2,
                "RD2_1": transform.RD2_1,
                "RD2_2": transform.RD2_2,
                "weight_1": transform.weight_1,
            }

    elif isinstance(transform, SetPrior):
        # Delegates to apply_set_prior which handles replace-or-append semantics.
        # Return early — SetPrior does not touch structural modules or variability.
        return apply_set_prior(spec, transform)

    # Preserve priors across all non-SetPrior transforms — structural swaps
    # may orphan individual priors (pruned below via _prune_stale_variability),
    # but other transforms (AddCovariateLink, AdjustVariability, SetTransitN,
    # ToggleLag) keep the full prior set intact.
    candidate = DSLSpec(
        model_id=new_id,
        absorption=absorption,
        distribution=distribution,
        elimination=elimination,
        variability=variability,
        covariates=covariates,
        observation=observation,
        priors=spec.priors,
        experimental=spec.experimental,
        metadata=spec.metadata,
        initial={},
    )

    required = set(candidate.calibration_param_names())
    new_initial = {name: value for name, value in spec.initial.items() if name in required}
    new_initial.update(initial_overrides)
    missing = required - set(new_initial)
    if missing:
        msg = f"Transform {transform.type!r} leaves initial values missing for: {sorted(missing)}"
        raise ValueError(msg)

    new_spec = candidate.model_copy(update={"initial": new_initial})

    # Prune stale variability AND priors after structural module swaps.
    # ConvertTransitToErlang and AddParallelRoute also change the structural
    # parameter set (drop ka / split into ka1+ka2), so they need pruning too.
    if isinstance(
        transform,
        (SwapModule, ReplaceWithNODE, ConvertTransitToErlang, AddParallelRoute),
    ):
        new_spec = _prune_stale_variability(new_spec)

    return new_spec


def _prune_stale_variability(spec: DSLSpec) -> DSLSpec:
    """Remove variability, covariates, AND priors referring to params that no longer exist.

    Called after SwapModule/ReplaceWithNODE to keep IIV/IOV/covariates/priors
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
        else:
            cleaned.append(item)

    # Prune covariates targeting a param removed by the structural swap
    # (e.g. a CovariateLink on ``V`` after swap to TwoCmt(V1, V2, Q)).
    cleaned_covariates = [c for c in spec.covariates if c.param in valid_params]

    # Prune stale priors — priors targeting parameters removed by the structural swap.
    pruned_priors = [p for p in spec.priors if _prior_target_still_valid(p.target, valid_params)]

    return DSLSpec(
        model_id=spec.model_id,
        absorption=spec.absorption,
        distribution=spec.distribution,
        elimination=spec.elimination,
        variability=cleaned,
        covariates=cleaned_covariates,
        observation=spec.observation,
        priors=pruned_priors,
        experimental=spec.experimental,
        metadata=spec.metadata,
        initial=spec.initial,
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


def _resolve_iov_occasions(transform: AdjustVariability) -> OccasionSpec:
    """Default add_iov's occasions to OccasionByStudy, matching search/candidates.py's
    force_iov root-candidate default."""
    return transform.occasions if transform.occasions is not None else OccasionByStudy()


def _apply_adjust_variability(spec: DSLSpec, transform: AdjustVariability) -> list[object]:
    """Adjust IIV or IOV variability items.

    add/remove/upgrade_to_block target IIV blocks: the first block containing
    (or appropriate for) the param. add_iov/remove_iov are dispatched to
    :func:`_apply_adjust_iov` and target IOV items instead.
    """
    if transform.action in ("add_iov", "remove_iov"):
        return _apply_adjust_iov(spec, transform)

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


def _apply_adjust_iov(spec: DSLSpec, transform: AdjustVariability) -> list[object]:
    """Adjust IOV variability items.

    add_iov merges the param into an existing IOV item under the same
    occasions (union, no duplicates), or creates a new item if none matches.
    Conflicting-occasions requests are rejected earlier by validate_transform,
    so by the time this runs any item already covering the param is known to
    share the target occasions. remove_iov drops the param from any item that
    has it, dropping the item entirely once its params list empties; removing
    a param with no existing IOV is a no-op.
    """
    target_occasions = _resolve_iov_occasions(transform)
    var_list: list[object] = []
    action_applied = False

    for item in spec.variability:
        if not isinstance(item, IOV):
            var_list.append(item)
            continue

        params = list(item.params)
        if transform.action == "add_iov":
            if not action_applied and transform.param in params:
                action_applied = True
                var_list.append(item)
            elif not action_applied and item.occasions == target_occasions:
                params.append(transform.param)
                var_list.append(IOV(params=params, occasions=item.occasions))
                action_applied = True
            else:
                var_list.append(item)
        else:  # remove_iov
            if transform.param in params:
                params.remove(transform.param)
                action_applied = True
            if params:
                var_list.append(IOV(params=params, occasions=item.occasions))

    if transform.action == "add_iov" and not action_applied:
        var_list.append(IOV(params=[transform.param], occasions=target_occasions))

    return var_list


def _apply_toggle_lag(spec: DSLSpec, transform: ToggleLag) -> AbsorptionModule:
    """Toggle lag time on first-order absorption.

    ``ka`` carries forward automatically (both ``FirstOrder`` and
    ``LaggedFirstOrder`` use the same ``spec.initial["ka"]`` key) via the
    caller's prune-and-keep pass; only ``tlag`` needs explicit handling
    (added when toggling on, pruned away — as no longer required — when
    toggling off).
    """
    abs_mod = spec.absorption

    if transform.on:
        if isinstance(abs_mod, FirstOrder):
            return LaggedFirstOrder()
        return abs_mod  # already lagged or other type
    else:
        if isinstance(abs_mod, LaggedFirstOrder):
            return FirstOrder()
        return abs_mod
