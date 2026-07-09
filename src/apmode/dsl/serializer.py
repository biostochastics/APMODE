# SPDX-License-Identifier: GPL-2.0-or-later
"""Canonical Formular DSL text serializer (Formular sharpening plan §4 Phase 1, P1.9).

:func:`serialize_spec` is the single implementation ``apmode formular fmt``
and ``apmode formular diff`` both build on: ``fmt`` prints its output
directly (or writes it back in-place), and ``diff`` compiles both specs and
compares the block-keyed projection from :func:`spec_blocks` (not raw
text), so that reordering top-level blocks — or declaring the same
``variability:``/``covariates:``/``priors:`` entries in a different order —
never shows up as a spurious diff.

Canonical block order (also the order :func:`serialize_spec` emits blocks
in): ``metadata, units, absorption, distribution, elimination, variability,
covariates, priors, observation-or-observations, initial``. This is a
*display* order only — per ``pk_grammar.lark``, top-level blocks may appear
in any order in source text, and ``DSLSpec.source_meta`` (not this module)
is what preserves the user's original order for diagnostics. Re-running
``fmt`` on already-canonical text is idempotent, but running it on
hand-authored text discards the original block order and line positions —
see the CLI's ``--in-place`` warning.

Every module-call site below reproduces the exact literal-token syntax
``pk_grammar.lark`` defines for that variant (e.g. ``first_order: "FirstOrder"
"(" "ka" ")"`` — the ``ka`` token is a fixed keyword the grammar hard-codes,
not a name captured from the parse, so the calibration value itself is
never written into the module call; it lives in the ``initial:`` block).
Because of this, serialization is driven entirely by ``type(module)``
dispatch, never by reading calibration values off the module itself
(P1.4 already moved every calibration field into ``DSLSpec.initial``).

Known gap: ``DSLSpec.experimental`` (the NODE opt-in gate, P0.8) has no
grammar syntax at all — there is no ``experimental:`` block in
``pk_grammar.lark`` — so a spec with ``experimental.node=True`` cannot be
round-tripped through :func:`serialize_spec` followed by
:func:`apmode.dsl.grammar.compile_dsl`: the reparsed spec always gets
``experimental.node=False`` and fails ``FrmCode.LANE_NODE_EXPERIMENTAL_GATE``
on the next ``validate_dsl`` call. This is a pre-existing grammar
limitation (Phase 2 candidate — adding text syntax for the experimental
gate), not something this module can paper over.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    Erlang,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
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
from apmode.dsl.priors import (
    BetaPrior,
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
)

if TYPE_CHECKING:
    from apmode.dsl.ast_models import (
        AbsorptionModule,
        CovariateLink,
        DistributionModule,
        DSLSpec,
        EliminationModule,
        Metadata,
        ObservationModule,
        OccasionSpec,
        UnitsDeclaration,
        VariabilityItem,
    )
    from apmode.dsl.canonical import JSONValue
    from apmode.dsl.priors import PriorFamily, PriorSpec

CANONICAL_BLOCK_ORDER: tuple[str, ...] = (
    "metadata",
    "units",
    "absorption",
    "distribution",
    "elimination",
    "variability",
    "covariates",
    "priors",
    "observation",
    "initial",
)
"""Display order used by :func:`serialize_spec` and the diff-block keys."""


def _num(value: float) -> str:
    """Render a float using Python's own round-tripping ``repr``.

    ``repr(70.0) == "70.0"``, ``repr(0.1) == "0.1"`` — both are valid
    ``NUMBER`` tokens per ``pk_grammar.lark``.
    """
    return repr(float(value))


def _string(value: str) -> str:
    """Render a Python string as a grammar ``STRING`` (``ESCAPED_STRING``) token."""
    return json.dumps(value)


# ---------------------------------------------------------------------------
# Absorption / Distribution / Elimination — fixed literal-token module calls
# ---------------------------------------------------------------------------


def serialize_absorption_module(mod: AbsorptionModule) -> str:
    if isinstance(mod, IVBolus):
        return "IVBolus()"
    if isinstance(mod, FirstOrder):
        return "FirstOrder(ka)"
    if isinstance(mod, ZeroOrder):
        return "ZeroOrder(dur)"
    if isinstance(mod, LaggedFirstOrder):
        return "LaggedFirstOrder(ka, tlag)"
    if isinstance(mod, Transit):
        return f"Transit(n={mod.n}, ktr, ka)"
    if isinstance(mod, MixedFirstZero):
        return "MixedFirstZero(ka, dur, frac)"
    if isinstance(mod, Erlang):
        return f"Erlang(n={mod.n}, ktr)"
    if isinstance(mod, ParallelFirstOrder):
        return "ParallelFirstOrder(ka1, ka2, frac)"
    if isinstance(mod, SumIG):
        return f"SumIG(k={mod.k}, MT_1, MT_2, RD2_1, RD2_2, weight_1)"
    if isinstance(mod, NODEAbsorption):
        return f"NODE_Absorption(dim={mod.dim}, constraint_template={mod.constraint_template})"
    msg = f"unhandled AbsorptionModule variant: {mod!r}"
    raise TypeError(msg)


def serialize_distribution_module(mod: DistributionModule) -> str:
    if isinstance(mod, OneCmt):
        return "OneCmt(V)"
    if isinstance(mod, TwoCmt):
        return "TwoCmt(V1, V2, Q)"
    if isinstance(mod, ThreeCmt):
        return "ThreeCmt(V1, V2, V3, Q2, Q3)"
    if isinstance(mod, TMDDCore):
        return "TMDD_Core(V, R0, kon, koff, kint)"
    if isinstance(mod, TMDDQSS):
        return "TMDD_QSS(V, R0, KD, kint)"
    msg = f"unhandled DistributionModule variant: {mod!r}"
    raise TypeError(msg)


def serialize_elimination_module(mod: EliminationModule) -> str:
    if isinstance(mod, LinearElim):
        return "Linear(CL)"
    if isinstance(mod, MichaelisMenten):
        return "MichaelisMenten(Vmax, Km)"
    if isinstance(mod, ParallelLinearMM):
        return "ParallelLinearMM(CL, Vmax, Km)"
    if isinstance(mod, TimeVaryingElim):
        return f"TimeVarying(CL, decay_fn={mod.decay_fn})"
    if isinstance(mod, NODEElimination):
        return f"NODE_Elimination(dim={mod.dim}, constraint_template={mod.constraint_template})"
    msg = f"unhandled EliminationModule variant: {mod!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Variability (IIV/IOV + occasions)
# ---------------------------------------------------------------------------


def _serialize_occasion(spec: OccasionSpec) -> str:
    if isinstance(spec, OccasionByStudy):
        return "ByStudy"
    if isinstance(spec, OccasionByVisit):
        return f"ByVisit({spec.column})"
    if isinstance(spec, OccasionByDoseEpoch):
        return f"ByDoseEpoch({spec.column})"
    if isinstance(spec, OccasionCustom):
        return f"Custom({spec.column})"
    msg = f"unhandled OccasionSpec variant: {spec!r}"
    raise TypeError(msg)


def serialize_variability_item(item: VariabilityItem) -> str:
    # params is semantically an unordered set (apmode.dsl.canonical
    # sorts it for fingerprinting for the same reason) -- sorted here too
    # so fmt output is deterministic regardless of declaration order.
    params = ", ".join(sorted(item.params))
    if isinstance(item, IIV):
        return f"IIV(params=[{params}], structure={item.structure})"
    if isinstance(item, IOV):
        return f"IOV(params=[{params}], occasions={_serialize_occasion(item.occasions)})"
    msg = f"unhandled VariabilityItem variant: {item!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------


def serialize_covariate_link(item: CovariateLink) -> str:
    if item.form == "power":
        assert item.theta is not None and item.ref is not None
        return (
            f"{item.param} <- {item.covariate}.power("
            f"theta={_num(item.theta)}, ref={_num(item.ref)})"
        )
    if item.form == "exponential":
        assert item.theta is not None
        return f"{item.param} <- {item.covariate}.exponential(theta={_num(item.theta)})"
    if item.form == "linear":
        assert item.theta is not None
        return f"{item.param} <- {item.covariate}.linear(theta={_num(item.theta)})"
    if item.form == "categorical":
        assert item.reference is not None
        return f"{item.param} <- {item.covariate}.categorical(reference={_string(item.reference)})"
    if item.form == "maturation":
        assert item.tm50 is not None and item.hill is not None
        return (
            f"{item.param} <- {item.covariate}.maturation("
            f"tm50={_num(item.tm50)}, hill={_num(item.hill)})"
        )
    msg = f"unhandled CovariateLink.form value: {item.form!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def serialize_prior_family(family: PriorFamily) -> str:
    if isinstance(family, NormalPrior):
        return f"Normal(mu={_num(family.mu)}, sigma={_num(family.sigma)})"
    if isinstance(family, LogNormalPrior):
        return f"LogNormal(mu={_num(family.mu)}, sigma={_num(family.sigma)})"
    if isinstance(family, HalfNormalPrior):
        return f"HalfNormal(sigma={_num(family.sigma)})"
    if isinstance(family, HalfCauchyPrior):
        return f"HalfCauchy(scale={_num(family.scale)})"
    if isinstance(family, GammaPrior):
        return f"Gamma(alpha={_num(family.alpha)}, beta={_num(family.beta)})"
    if isinstance(family, InvGammaPrior):
        return f"InvGamma(alpha={_num(family.alpha)}, beta={_num(family.beta)})"
    if isinstance(family, BetaPrior):
        return f"Beta(alpha={_num(family.alpha)}, beta={_num(family.beta)})"
    if isinstance(family, LKJPrior):
        return f"LKJ(eta={_num(family.eta)})"
    if isinstance(family, MixturePrior):
        components = ", ".join(serialize_prior_family(c) for c in family.components)
        weights = ", ".join(_num(w) for w in family.weights)
        return f"Mixture(components=[{components}], weights=[{weights}])"
    if isinstance(family, HistoricalBorrowingPrior):
        refs = ", ".join(_string(r) for r in family.historical_refs)
        return (
            f"HistoricalBorrowing(map_mean={_num(family.map_mean)}, "
            f"map_sd={_num(family.map_sd)}, robust_weight={_num(family.robust_weight)}, "
            f"historical_refs=[{refs}])"
        )
    msg = f"unhandled PriorFamily variant: {family!r}"
    raise TypeError(msg)


def serialize_prior(prior: PriorSpec) -> str:
    parts = [f"{prior.target} ~ {serialize_prior_family(prior.family)}"]
    if prior.source != "weakly_informative":
        parts.append(f"source={prior.source}")
    if prior.doi is not None:
        parts.append(f"doi={_string(prior.doi)}")
    if prior.justification:
        parts.append(f"justification={_string(prior.justification)}")
    if prior.historical_refs:
        refs = ", ".join(_string(r) for r in prior.historical_refs)
        parts.append(f"historical_refs=[{refs}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def serialize_observation_module(mod: ObservationModule) -> str:
    if isinstance(mod, Proportional):
        return f"Proportional(sigma_prop={_num(mod.sigma_prop)})"
    if isinstance(mod, Additive):
        return f"Additive(sigma_add={_num(mod.sigma_add)})"
    if isinstance(mod, Combined):
        return f"Combined(sigma_prop={_num(mod.sigma_prop)}, sigma_add={_num(mod.sigma_add)})"
    if isinstance(mod, BLQM3):
        return (
            f"BLQ_M3(loq_value={_num(mod.loq_value)}, error_model={mod.error_model}, "
            f"sigma_prop={_num(mod.sigma_prop)}, sigma_add={_num(mod.sigma_add)})"
        )
    if isinstance(mod, BLQM4):
        return (
            f"BLQ_M4(loq_value={_num(mod.loq_value)}, error_model={mod.error_model}, "
            f"sigma_prop={_num(mod.sigma_prop)}, sigma_add={_num(mod.sigma_add)})"
        )
    msg = f"unhandled ObservationModule variant: {mod!r}"
    raise TypeError(msg)


def _serialize_metadata(meta: Metadata) -> str:
    fields: list[str] = []
    for name in ("title", "intent", "context_of_use", "analyte", "version"):
        value = getattr(meta, name)
        if value is not None:
            fields.append(f"{name} = {_string(value)}")
    inner = ", ".join(fields)
    return f"metadata: {{ {inner} }}" if inner else "metadata: {}"


def _serialize_units(units: UnitsDeclaration) -> str:
    return (
        f"units: {{ time = {units.time}, amount = {units.amount}, "
        f"concentration = {units.concentration}, volume = {units.volume} }}"
    )


def _serialize_initial(initial: dict[str, float]) -> str:
    inner = ", ".join(f"{name} = {_num(value)}" for name, value in sorted(initial.items()))
    return f"initial: {{ {inner} }}"


def serialize_spec(spec: DSLSpec) -> str:
    """Render ``spec`` as canonical Formular DSL source text.

    Block order is fixed (see :data:`CANONICAL_BLOCK_ORDER`), independent
    of ``spec.source_meta`` — this is a *canonicalizing* formatter, not a
    format-preserving one (Formular sharpening plan §4 Phase 1, invariant
    #6: canonical order is used only in the fingerprint path and in
    ``formular fmt`` output).
    """
    lines: list[str] = ["model {"]

    if spec.metadata is not None:
        lines.append(f"    {_serialize_metadata(spec.metadata)}")
    if spec.units is not None:
        lines.append(f"    {_serialize_units(spec.units)}")

    lines.append(f"    absorption: {serialize_absorption_module(spec.absorption)}")
    lines.append(f"    distribution: {serialize_distribution_module(spec.distribution)}")
    lines.append(f"    elimination: {serialize_elimination_module(spec.elimination)}")

    if spec.variability:
        lines.append("    variability: {")
        for entry in sorted(serialize_variability_item(item) for item in spec.variability):
            lines.append(f"        {entry}")
        lines.append("    }")

    if spec.covariates:
        lines.append("    covariates: {")
        entries = sorted(serialize_covariate_link(c) for c in spec.covariates)
        for i, entry in enumerate(entries):
            suffix = "," if i < len(entries) - 1 else ""
            lines.append(f"        {entry}{suffix}")
        lines.append("    }")

    if spec.priors:
        lines.append("    priors: {")
        for entry in sorted(serialize_prior(p) for p in spec.priors):
            lines.append(f"        {entry}")
        lines.append("    }")

    if spec.observations:
        lines.append("    observations: {")
        names = sorted(spec.observations)
        for i, name in enumerate(names):
            ep = spec.observations[name]
            suffix = "," if i < len(names) - 1 else ""
            lines.append(
                f"        {name}: {{ dvid={ep.dvid}, prediction={ep.prediction}, "
                f"error={serialize_observation_module(ep.error)} }}{suffix}"
            )
        lines.append("    }")
    else:
        lines.append(f"    observation: {serialize_observation_module(spec.observation)}")

    lines.append(f"    {_serialize_initial(spec.initial)}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Block-keyed projection for order-insensitive diffing
# ---------------------------------------------------------------------------


def _canonical_sort(dumps: list[JSONValue]) -> list[JSONValue]:
    """Sort already-``model_dump``-ed values by their canonical JSON string.

    Gives a stable, order-insensitive comparison key for lists that are
    semantically unordered sets from a diffing point of view
    (``variability``/``covariates``/``priors`` entries) — mirrors
    ``apmode.dsl.canonical._sorted_by_canonical_json``.
    """
    return sorted(dumps, key=lambda v: json.dumps(v, sort_keys=True))


def _variability_item_dump(item: VariabilityItem) -> JSONValue:
    """Dump one variability item with its (semantically unordered) ``params`` sorted.

    Without this, two ``IIV(params=[CL, V], ...)`` /
    ``IIV(params=[V, CL], ...)`` declarations — identical per
    ``pk_grammar.lark``'s ``param_list`` semantics — would register as a
    spurious diff.
    """
    dumped = cast("dict[str, JSONValue]", item.model_dump(mode="json"))
    params = dumped.get("params")
    if isinstance(params, list):
        dumped["params"] = cast("JSONValue", sorted(cast("list[str]", params)))
    return dumped


def _observation_block(spec: DSLSpec) -> JSONValue:
    if spec.observations:
        return {
            "kind": "multi",
            "endpoints": {
                name: cast("JSONValue", ep.model_dump(mode="json"))
                for name, ep in sorted(spec.observations.items())
            },
        }
    module_dump = cast("JSONValue", spec.observation.model_dump(mode="json"))
    return {"kind": "single", "module": module_dump}


def spec_blocks(spec: DSLSpec) -> dict[str, JSONValue]:
    """Project ``spec`` onto a block-keyed dict, order-insensitive within each block.

    Used by :func:`diff_specs` so that two specs differing only in
    declaration order (top-level block order, or the order of
    ``variability:``/``covariates:``/``priors:`` entries) compare equal.
    Unlike :mod:`apmode.dsl.canonical`, this projection is NOT
    fingerprint-scoped: it includes ``metadata``/``units``/prior
    ``justification``/``doi`` because a text-level diff tool should show
    every real difference, not just modeling-relevant ones.
    """
    return {
        "metadata": (
            cast("JSONValue", spec.metadata.model_dump(mode="json"))
            if spec.metadata is not None
            else None
        ),
        "units": (
            cast("JSONValue", spec.units.model_dump(mode="json"))
            if spec.units is not None
            else None
        ),
        "absorption": cast("JSONValue", spec.absorption.model_dump(mode="json")),
        "distribution": cast("JSONValue", spec.distribution.model_dump(mode="json")),
        "elimination": cast("JSONValue", spec.elimination.model_dump(mode="json")),
        "variability": _canonical_sort([_variability_item_dump(i) for i in spec.variability]),
        "covariates": _canonical_sort(
            [cast("JSONValue", c.model_dump(mode="json")) for c in spec.covariates]
        ),
        "priors": _canonical_sort(
            [cast("JSONValue", p.model_dump(mode="json")) for p in spec.priors]
        ),
        "observation": _observation_block(spec),
        "initial": cast("JSONValue", dict(sorted(spec.initial.items()))),
    }


# ---------------------------------------------------------------------------
# Compact model signature (Formular sharpening plan §4 Phase 2, P2.4)
# ---------------------------------------------------------------------------
#
# A one-line, grep/pipe-able summary of a spec's module choices — distinct
# from serialize_spec's canonical multi-line DSL text. Short codes below
# reuse serialize_*_module's vocabulary where that vocabulary is already
# terse (e.g. "Linear", "TMDD_QSS") but favour readability over exact
# grammar-token fidelity ("Linear CL" not "Linear(CL)") since this view is
# for human scanning, not round-tripping.

_ABSORPTION_SHORT_CODES: dict[type, str] = {
    IVBolus: "IV bolus",
    FirstOrder: "FO",
    ZeroOrder: "ZO",
    LaggedFirstOrder: "FO+lag",
    MixedFirstZero: "FO+ZO mixed",
    ParallelFirstOrder: "Parallel-FO",
    NODEAbsorption: "NODE-abs",
}

_DISTRIBUTION_SHORT_CODES: dict[type, str] = {
    OneCmt: "1CMT",
    TwoCmt: "2CMT",
    ThreeCmt: "3CMT",
    TMDDCore: "TMDD-full",
    TMDDQSS: "TMDD-QSS",
}

_ELIMINATION_SHORT_CODES: dict[type, str] = {
    LinearElim: "Linear CL",
    MichaelisMenten: "MM",
    ParallelLinearMM: "Linear+MM",
    TimeVaryingElim: "TimeVarying CL",
    NODEElimination: "NODE-elim",
}

_OBSERVATION_SHORT_CODES: dict[type, str] = {
    Proportional: "Prop error",
    Additive: "Add error",
    Combined: "Combined error",
    BLQM3: "BLQ-M3",
    BLQM4: "BLQ-M4",
}


def _absorption_short_code(mod: AbsorptionModule) -> str:
    # Transit/Erlang/SumIG carry a structural int (n or k) inline on the
    # module, so they are keyed by isinstance rather than the plain type
    # lookup table above.
    if isinstance(mod, Transit):
        return f"Transit({mod.n})"
    if isinstance(mod, Erlang):
        return f"Erlang({mod.n})"
    if isinstance(mod, SumIG):
        return f"SumIG({mod.k})"
    code = _ABSORPTION_SHORT_CODES.get(type(mod))
    if code is None:
        msg = f"unhandled AbsorptionModule variant: {mod!r}"
        raise TypeError(msg)
    return code


def _distribution_short_code(mod: DistributionModule) -> str:
    code = _DISTRIBUTION_SHORT_CODES.get(type(mod))
    if code is None:
        msg = f"unhandled DistributionModule variant: {mod!r}"
        raise TypeError(msg)
    return code


def _elimination_short_code(mod: EliminationModule) -> str:
    code = _ELIMINATION_SHORT_CODES.get(type(mod))
    if code is None:
        msg = f"unhandled EliminationModule variant: {mod!r}"
        raise TypeError(msg)
    return code


def _observation_short_code(mod: ObservationModule) -> str:
    code = _OBSERVATION_SHORT_CODES.get(type(mod))
    if code is None:
        msg = f"unhandled ObservationModule variant: {mod!r}"
        raise TypeError(msg)
    return code


def _iiv_signature_segment(spec: DSLSpec) -> str | None:
    """Build the ``IIV(param,param,...) diag`` segment, or ``None`` if no IIV items.

    Only ``IIV`` variability items are summarized (not ``IOV`` — the
    compact signature is scoped to the between-subject-variability axis
    the plan calls out). Multiple ``IIV`` entries (e.g. one ``diagonal``
    group and one ``block`` group) are joined with ``"; "``; each entry's
    ``params`` is sorted for determinism, matching
    :func:`serialize_variability_item`'s convention.
    """
    iiv_items = [item for item in spec.variability if isinstance(item, IIV)]
    if not iiv_items:
        return None
    structure_code = {"diagonal": "diag", "block": "block"}
    parts = [
        f"IIV({','.join(sorted(item.params))}) {structure_code[item.structure]}"
        for item in iiv_items
    ]
    return "; ".join(parts)


def _observation_signature_segment(spec: DSLSpec) -> str:
    """Build the trailing observation-model segment.

    Legacy singular ``observation:`` specs render as the single short code
    (e.g. ``"Prop error"``). Multi-analyte ``observations:`` specs instead
    render as ``"<n> endpoints (<unique error-model codes>)"`` — the plan
    left the exact multi-analyte format to this implementation's
    discretion; endpoint count plus the distinct error-model codes in use
    (in endpoint-name-sorted order, de-duplicated) was chosen over listing
    every endpoint by name to keep the line short and grep-able even for
    specs with many endpoints.
    """
    if not spec.observations:
        return _observation_short_code(spec.observation)
    codes: list[str] = []
    seen: set[str] = set()
    for name in sorted(spec.observations):
        code = _observation_short_code(spec.observations[name].error)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    noun = "endpoint" if len(spec.observations) == 1 else "endpoints"
    return f"{len(spec.observations)} {noun} ({', '.join(codes)})"


def build_signature(spec: DSLSpec) -> str:
    """Render a compact, one-line, pipe-delimited summary of ``spec``'s module choices.

    e.g. ``"FO absorption | 1CMT | Linear CL | IIV(CL,V,ka) diag | Prop error"``.
    Intended for ``apmode formular signature``, ``apmode ls``, report
    headers, and DAG-viewer node labels — anywhere a single grep/pipe-able
    line is more useful than the full multi-line ``apmode formular explain``
    table. Segments, in order: absorption (suffixed ``" absorption"`` for
    readability since its short codes alone — ``"FO"``, ``"ZO"`` — read as
    unlabeled abbreviations more than distribution/elimination's do),
    distribution, elimination, IIV summary (omitted entirely when
    ``spec.variability`` has no ``IIV`` items), observation.

    The elimination segment is omitted entirely when ``spec.distribution``
    is ``TMDDCore``/``TMDDQSS``: per
    ``nlmixr2_emitter._emit_tmdd_core_odes``/``_emit_tmdd_qss_odes`` (and
    the matching disclosed note in ``equations.py``), those dynamics
    ignore ``spec.elimination`` completely and always use ``kel = CL/V``.
    Rendering the declared elimination module's short code alongside a
    TMDD distribution segment (e.g. ``"TMDD-QSS | MM"``) would imply
    Michaelis-Menten kinetics are active when the emitter never reads
    Vmax/Km in that branch — `compile_dsl` does not enforce
    ``FrmCode.AST_TMDD_REQUIRES_LINEAR_ELIM`` (that lives only in
    ``validator.py``, run by ``lint``/``validate``, not ``signature``), so
    a TMDD spec with a non-linear declared elimination module compiles and
    reaches this function.
    """
    segments = [
        f"{_absorption_short_code(spec.absorption)} absorption",
        _distribution_short_code(spec.distribution),
    ]
    if not isinstance(spec.distribution, (TMDDCore, TMDDQSS)):
        segments.append(_elimination_short_code(spec.elimination))
    iiv_segment = _iiv_signature_segment(spec)
    if iiv_segment is not None:
        segments.append(iiv_segment)
    segments.append(_observation_signature_segment(spec))
    return " | ".join(segments)


def diff_specs(a: DSLSpec, b: DSLSpec) -> dict[str, tuple[JSONValue, JSONValue]]:
    """Return ``{block_name: (a_value, b_value)}`` for every block that differs.

    Both specs are projected through :func:`spec_blocks` first, so
    reordering alone (top-level blocks, or entries within a
    ``variability:``/``covariates:``/``priors:`` block) never appears as
    a diff — only genuine content differences do. A block absent from the
    return value is identical between ``a`` and ``b``.
    """
    a_blocks = spec_blocks(a)
    b_blocks = spec_blocks(b)
    diffs: dict[str, tuple[JSONValue, JSONValue]] = {}
    for key in CANONICAL_BLOCK_ORDER:
        a_val = a_blocks.get(key)
        b_val = b_blocks.get(key)
        if a_val != b_val:
            diffs[key] = (a_val, b_val)
    return diffs


__all__ = [
    "CANONICAL_BLOCK_ORDER",
    "build_signature",
    "diff_specs",
    "serialize_spec",
    "spec_blocks",
]
