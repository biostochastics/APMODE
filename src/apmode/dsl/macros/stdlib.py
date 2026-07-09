# SPDX-License-Identifier: GPL-2.0-or-later
"""Vetted standard-library Formular macros (`use pkstd.*`).

Formular sharpening plan §4 Phase 2 (P2.1). Each macro below is a pure
``DSLSpec -> DSLSpec`` function, applied by
:func:`apmode.dsl.macros.expand_macros` after the full spec has been
assembled by the Lark transform. None of these macros expresses anything a
hand-authored ``variability:``/``priors:`` block could not already say —
they are sugar over the existing AST, added only to remove common
boilerplate for the near-universal starting configuration of a population-
PK model (Sheiner & Beal 1980).

Every macro is idempotent-safe in the "no-op if already covered" sense
(documented per-function below) but ``apmode.dsl.macros.expand_macros``
still rejects a *duplicate* ``use`` of the same macro name in one spec —
the no-op guard here is about pre-existing hand-authored coverage, not
permission to invoke the same macro twice.
"""

from __future__ import annotations

import math

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    Additive,
    Combined,
    DSLSpec,
    Proportional,
)
from apmode.dsl.macros import MacroDef, register_macro
from apmode.dsl.normalize import normalize_param_name
from apmode.dsl.priors import (
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    PriorSpec,
    build_prior_spec,
)

# ---------------------------------------------------------------------------
# pkstd.standard_iiv
# ---------------------------------------------------------------------------


def _covered_variability_params(spec: DSLSpec) -> set[str]:
    """Every structural param name already covered by an existing IIV/IOV entry.

    Names are resolved through :func:`apmode.dsl.normalize.normalize_param_name`
    before comparison: IIV/IOV parameter names are NOT case-normalized at
    compile time (unlike priors-block targets, which are forced to canonical
    case earlier in ``compile_dsl`` via ``classify_target``), so a
    hand-authored block using a non-canonical case (e.g. ``cl`` instead of
    ``CL``) must still be recognized as covering the canonical parameter
    name returned by :meth:`~apmode.dsl.ast_models.DSLSpec.structural_param_names`.
    Without this normalization the macro would append a second IIV entry for
    the same parameter, which then fails the validator's duplicate-IIV check
    (``FrmCode.AST_IIV_NO_DUPLICATE_PARAMS``) — silently violating this
    macro's own documented no-op guarantee.
    """
    covered: set[str] = set()
    for item in spec.variability:
        covered.update(normalize_param_name(p) for p in item.params)
    return covered


def _standard_iiv(spec: DSLSpec) -> DSLSpec:
    """pkstd.standard_iiv (v1): diagonal IIV on every uncovered structural parameter.

    Rationale: a single diagonal ``IIV(structure="diagonal")`` block
    covering every structural parameter is the near-universal starting
    point for a population-PK model. Adds one new ``IIV`` entry naming
    every structural parameter (per
    :meth:`~apmode.dsl.ast_models.DSLSpec.structural_param_names`) NOT
    already covered by any existing IIV/IOV entry in ``spec.variability``.

    "n" (Transit/Erlang chain length) is excluded even though it is a
    structural-param name: it is the sole name the semantic validator
    disallows IIV/IOV on
    (``apmode.dsl.validator._NO_VARIABILITY_PARAMS`` ->
    ``FrmCode.AST_NO_VARIABILITY_ON_PARAM``). NODE ``node_abs_w*``/
    ``node_elim_w*`` weight names ARE eligible (see
    ``DSLSpec.structural_param_names`` docstring) and are not excluded.

    Documented no-op: if every eligible structural parameter is already
    covered, returns ``spec`` unchanged rather than raising or appending an
    empty ``IIV(params=[], ...)`` (which the validator would reject via
    ``FrmCode.AST_NON_EMPTY_PARAMS``).
    """
    eligible = [p for p in spec.structural_param_names() if p != "n"]
    covered = _covered_variability_params(spec)
    uncovered = [p for p in eligible if p not in covered]
    if not uncovered:
        return spec
    new_iiv = IIV(params=uncovered, structure="diagonal")
    return spec.model_copy(update={"variability": [*spec.variability, new_iiv]})


# ---------------------------------------------------------------------------
# pkstd.standard_priors
# ---------------------------------------------------------------------------


def _default_structural_family(spec: DSLSpec, param: str) -> NormalPrior | LogNormalPrior:
    """Choose a weakly-informative prior family for one structural parameter.

    NODE input-layer weights (``node_abs_w*``/``node_elim_w*``) may be
    positive or negative depending on the owning NODE module's
    ``constraint_template`` (e.g. ``"unconstrained_smooth"``); a LogNormal
    default would silently bias every weight positive, so ``Normal(0, 1)``
    is used there instead.

    Every other structural parameter (CL, V, ka, ...) is conventionally
    positive, so ``LogNormal`` is used: its median is centered at
    ``exp(mu)``, chosen as the spec's own ``initial:`` value for ``param``
    when one is declared and positive (``mu = log(initial)``), or ``0.0``
    (median 1.0 on the natural scale) as a neutral fallback otherwise.
    ``LogNormal`` is valid for every ``"structural"`` target kind per
    ``apmode.dsl.priors._VALID_FAMILIES`` regardless of which structural
    parameter it targets, so this choice can never fail
    ``build_prior_spec``'s family/target-kind compatibility check.
    """
    if param.startswith(("node_abs_w", "node_elim_w")):
        return NormalPrior(mu=0.0, sigma=1.0)
    initial = spec.get_initial(param)
    mu = math.log(initial) if initial is not None and initial > 0 else 0.0
    return LogNormalPrior(mu=mu, sigma=1.0)


def _standard_priors(spec: DSLSpec) -> DSLSpec:
    """pkstd.standard_priors (v1): weakly-informative prior on every uncovered param.

    For every name in :meth:`~apmode.dsl.ast_models.DSLSpec.structural_param_names`
    lacking a declared prior in ``spec.priors`` (matched by exact
    ``PriorSpec.target`` string), adds one via the canonical
    :func:`apmode.dsl.priors.build_prior_spec` factory — never a
    hand-constructed ``PriorSpec`` — so this macro's output is governed by
    the exact same ``classify_target``/``validate_prior_family``
    invariants as a human-authored ``priors:`` entry or an agentic
    ``SetPrior`` transform (P1.5 parity guarantee). Family selection is
    per :func:`_default_structural_family`.

    "n" (Transit/Erlang chain length) is skipped: it is
    structural-but-not-calibrated (excluded from
    ``DSLSpec.calibration_param_names``) and there is no meaningful
    continuous weakly-informative prior for a fixed integer topology
    choice.

    Documented no-op: if every eligible structural parameter already has a
    declared prior, returns ``spec`` unchanged.
    """
    structural = set(spec.structural_param_names())
    already_targeted = {p.target for p in spec.priors}
    new_priors: list[PriorSpec] = list(spec.priors)
    changed = False
    for param in spec.structural_param_names():
        if param == "n" or param in already_targeted:
            continue
        family = _default_structural_family(spec, param)
        new_priors.append(
            build_prior_spec(
                target=param,
                family=family,
                source="weakly_informative",
                structural_params=structural,
            )
        )
        already_targeted.add(param)
        changed = True
    if not changed:
        return spec
    return spec.model_copy(update={"priors": new_priors})


# ---------------------------------------------------------------------------
# pkstd.standard_error_model
# ---------------------------------------------------------------------------


def _standard_error_model(spec: DSLSpec) -> DSLSpec:
    """pkstd.standard_error_model (v1): weakly-informative prior on the residual-error SD(s).

    Maps ``spec.observation``'s type to the residual-error SD target(s) it
    implies: ``Proportional`` -> ``{"sigma_prop"}``, ``Additive`` ->
    ``{"sigma_add"}``, ``Combined`` -> both. ``BLQM3``/``BLQM4`` use
    :meth:`~apmode.dsl.ast_models.BLQM3.active_sigmas` so only the sigma(s)
    that actually enter the likelihood under the declared ``error_model``
    are targeted (mirrors the parameter-count convention documented on
    that method). Adds a ``HalfNormal(sigma=1.0)`` prior (valid for every
    ``"residual_sd"`` target per ``apmode.dsl.priors._VALID_FAMILIES``) via
    :func:`apmode.dsl.priors.build_prior_spec` for each target not already
    declared in ``spec.priors``.

    Multi-analyte case (``spec.observations`` set): this is a **documented
    no-op**. The ``sigma_prop``/``sigma_add`` prior-target namespace is
    flat — ``apmode.dsl.priors.classify_target`` has no per-endpoint
    disambiguation, so every endpoint's residual-error prior would collide
    on the identical target name. Applying this macro per-endpoint could
    silently produce a prior that is correct for at most one endpoint and
    wrong (or simply redundant/overwritten) for every other. Rather than
    guess at a disambiguation scheme, this macro declines to act at all
    when ``spec.observations`` is populated; adding per-endpoint sigma
    prior targets to the DSL is a Phase 2+ candidate outside this macro's
    scope.

    Documented no-op: also returns ``spec`` unchanged (single-endpoint
    case) if every implied sigma target already has a declared prior.
    """
    if spec.observations:
        return spec

    obs = spec.observation
    if isinstance(obs, Proportional):
        targets = ["sigma_prop"]
    elif isinstance(obs, Additive):
        targets = ["sigma_add"]
    elif isinstance(obs, Combined):
        targets = ["sigma_prop", "sigma_add"]
    elif isinstance(obs, BLQM3 | BLQM4):
        targets = obs.active_sigmas()
    else:  # pragma: no cover — exhaustive per ObservationModule union
        msg = f"unhandled ObservationModule variant: {obs!r}"
        raise TypeError(msg)

    structural = set(spec.structural_param_names())
    already_targeted = {p.target for p in spec.priors}
    new_priors: list[PriorSpec] = list(spec.priors)
    changed = False
    for target in targets:
        if target in already_targeted:
            continue
        new_priors.append(
            build_prior_spec(
                target=target,
                family=HalfNormalPrior(sigma=1.0),
                source="weakly_informative",
                structural_params=structural,
            )
        )
        already_targeted.add(target)
        changed = True
    if not changed:
        return spec
    return spec.model_copy(update={"priors": new_priors})


register_macro(
    MacroDef(
        name="pkstd.standard_iiv",
        version="v1",
        description="Diagonal IIV on every structural parameter not already covered.",
    ),
    _standard_iiv,
)
register_macro(
    MacroDef(
        name="pkstd.standard_priors",
        version="v1",
        description=("Weakly-informative prior on every structural parameter lacking one."),
    ),
    _standard_priors,
)
register_macro(
    MacroDef(
        name="pkstd.standard_error_model",
        version="v1",
        description=(
            "Weakly-informative prior on the residual-error sigma(s) implied by "
            "the observation model (single-endpoint specs only)."
        ),
    ),
    _standard_error_model,
)

__all__: list[str] = []
