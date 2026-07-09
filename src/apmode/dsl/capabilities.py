# SPDX-License-Identifier: GPL-2.0-or-later
"""Code-derived capability matrix for the DSL emitters (Phase 0, P0.7).

Every AST variant/feature across the five module axes (Absorption x
Distribution x Elimination x Variability x Observation, PRD §4.2.5) gets one
:class:`CapabilityTag`. Each emitter module (``nlmixr2_emitter``,
``stan_emitter``, ``frem_emitter``) declares two module-level frozensets:

- ``SUPPORTS`` — tags with a real, correct lowering code path.
- ``EXPLICITLY_UNSUPPORTED`` — tags the emitter deliberately rejects
  (either via a ``NotImplementedError`` raise site, or — for FREM's
  ``CovariateLink`` — a documented design choice to strip the capability).

A tag missing from *both* sets on a given emitter is a silent gap: the
emitter would either crash unpredictably or, worse, silently emit a model
that does not honour the requested capability. ``scripts/verify_capability_coverage.py``
(and ``tests/unit/test_dsl_capabilities.py``) fail CI when such a gap exists,
so the matrix in this module must be updated whenever a new AST variant, a
new emitter, or a new emitter code path is added.

This module intentionally has no import-time dependency on the emitter
modules — :func:`registered_emitters` imports them lazily (and caches the
result) so that ``nlmixr2_emitter.py`` / ``stan_emitter.py`` /
``frem_emitter.py`` can import :class:`CapabilityTag` from here without
creating an import cycle.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

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


class CapabilityTag(StrEnum):
    """One tag per AST variant (or cross-cutting feature) in the PK DSL.

    Grouped by module axis (Absorption / Distribution / Elimination /
    Variability / Observation) to mirror ``src/apmode/dsl/ast_models.py``.
    Values are stable dotted strings — treat them as part of the DSL
    capability contract; renaming a member changes ``report()`` output.
    """

    # --- Absorption --------------------------------------------------
    ABSORPTION_IV_BOLUS = "absorption.iv_bolus"
    ABSORPTION_FIRST_ORDER = "absorption.first_order"
    ABSORPTION_ZERO_ORDER = "absorption.zero_order"
    ABSORPTION_LAGGED_FIRST_ORDER = "absorption.lagged_first_order"
    ABSORPTION_TRANSIT = "absorption.transit"
    ABSORPTION_MIXED_FIRST_ZERO = "absorption.mixed_first_zero"
    ABSORPTION_ERLANG = "absorption.erlang"
    ABSORPTION_PARALLEL_FIRST_ORDER = "absorption.parallel_first_order"
    ABSORPTION_SUM_IG = "absorption.sum_ig"
    ABSORPTION_NODE = "absorption.node"

    # --- Distribution --------------------------------------------------
    DISTRIBUTION_ONE_CMT = "distribution.one_cmt"
    DISTRIBUTION_TWO_CMT = "distribution.two_cmt"
    DISTRIBUTION_THREE_CMT = "distribution.three_cmt"
    DISTRIBUTION_TMDD_CORE = "distribution.tmdd_core"
    DISTRIBUTION_TMDD_QSS = "distribution.tmdd_qss"

    # --- Elimination --------------------------------------------------
    ELIMINATION_LINEAR = "elimination.linear"
    ELIMINATION_MICHAELIS_MENTEN = "elimination.michaelis_menten"
    ELIMINATION_PARALLEL_LINEAR_MM = "elimination.parallel_linear_mm"
    ELIMINATION_TIME_VARYING = "elimination.time_varying"
    ELIMINATION_NODE = "elimination.node"

    # --- Variability (item kinds + cross-cutting features) ------------
    VARIABILITY_IIV = "variability.iiv"
    VARIABILITY_IIV_BLOCK_STRUCTURE = "variability.iiv_block_structure"
    VARIABILITY_IOV = "variability.iov"
    VARIABILITY_COVARIATE_LINK = "variability.covariate_link"
    VARIABILITY_COVARIATE_MATURATION_FORM = "variability.covariate_maturation_form"

    # --- Observation --------------------------------------------------
    OBSERVATION_PROPORTIONAL = "observation.proportional"
    OBSERVATION_ADDITIVE = "observation.additive"
    OBSERVATION_COMBINED = "observation.combined"
    OBSERVATION_BLQ_M3 = "observation.blq_m3"
    OBSERVATION_BLQ_M4 = "observation.blq_m4"
    OBSERVATION_MULTI_ANALYTE = "observation.multi_analyte"


_ABSORPTION_TAGS: dict[type[Any], CapabilityTag] = {
    IVBolus: CapabilityTag.ABSORPTION_IV_BOLUS,
    FirstOrder: CapabilityTag.ABSORPTION_FIRST_ORDER,
    ZeroOrder: CapabilityTag.ABSORPTION_ZERO_ORDER,
    LaggedFirstOrder: CapabilityTag.ABSORPTION_LAGGED_FIRST_ORDER,
    Transit: CapabilityTag.ABSORPTION_TRANSIT,
    MixedFirstZero: CapabilityTag.ABSORPTION_MIXED_FIRST_ZERO,
    Erlang: CapabilityTag.ABSORPTION_ERLANG,
    ParallelFirstOrder: CapabilityTag.ABSORPTION_PARALLEL_FIRST_ORDER,
    SumIG: CapabilityTag.ABSORPTION_SUM_IG,
    NODEAbsorption: CapabilityTag.ABSORPTION_NODE,
}

_DISTRIBUTION_TAGS: dict[type[Any], CapabilityTag] = {
    OneCmt: CapabilityTag.DISTRIBUTION_ONE_CMT,
    TwoCmt: CapabilityTag.DISTRIBUTION_TWO_CMT,
    ThreeCmt: CapabilityTag.DISTRIBUTION_THREE_CMT,
    TMDDCore: CapabilityTag.DISTRIBUTION_TMDD_CORE,
    TMDDQSS: CapabilityTag.DISTRIBUTION_TMDD_QSS,
}

_ELIMINATION_TAGS: dict[type[Any], CapabilityTag] = {
    LinearElim: CapabilityTag.ELIMINATION_LINEAR,
    MichaelisMenten: CapabilityTag.ELIMINATION_MICHAELIS_MENTEN,
    ParallelLinearMM: CapabilityTag.ELIMINATION_PARALLEL_LINEAR_MM,
    TimeVaryingElim: CapabilityTag.ELIMINATION_TIME_VARYING,
    NODEElimination: CapabilityTag.ELIMINATION_NODE,
}

_OBSERVATION_TAGS: dict[type[Any], CapabilityTag] = {
    Proportional: CapabilityTag.OBSERVATION_PROPORTIONAL,
    Additive: CapabilityTag.OBSERVATION_ADDITIVE,
    Combined: CapabilityTag.OBSERVATION_COMBINED,
    BLQM3: CapabilityTag.OBSERVATION_BLQ_M3,
    BLQM4: CapabilityTag.OBSERVATION_BLQ_M4,
}


def tags_for_spec(spec: DSLSpec) -> frozenset[CapabilityTag]:
    """Derive the set of :class:`CapabilityTag` exercised by a compiled spec.

    One tag per populated axis (absorption/distribution/elimination/
    observation), plus per-item variability tags. ``IIV.structure == "block"``
    additionally sets :attr:`CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE`;
    ``CovariateLink.form == "maturation"`` additionally sets
    :attr:`CapabilityTag.VARIABILITY_COVARIATE_MATURATION_FORM`.
    """
    tags: set[CapabilityTag] = set()

    abs_tag = _ABSORPTION_TAGS.get(type(spec.absorption))
    if abs_tag is not None:
        tags.add(abs_tag)

    dist_tag = _DISTRIBUTION_TAGS.get(type(spec.distribution))
    if dist_tag is not None:
        tags.add(dist_tag)

    elim_tag = _ELIMINATION_TAGS.get(type(spec.elimination))
    if elim_tag is not None:
        tags.add(elim_tag)

    obs_tag = _OBSERVATION_TAGS.get(type(spec.observation))
    if obs_tag is not None:
        tags.add(obs_tag)

    # P1.7: a multi-analyte ``observations:`` block additionally tags every
    # entry's error-module type (so e.g. a Combined + Proportional
    # two-endpoint spec surfaces both OBSERVATION_COMBINED and
    # OBSERVATION_PROPORTIONAL) plus the cross-cutting
    # OBSERVATION_MULTI_ANALYTE feature tag itself.
    if spec.observations:
        tags.add(CapabilityTag.OBSERVATION_MULTI_ANALYTE)
        for endpoint in spec.observations.values():
            endpoint_tag = _OBSERVATION_TAGS.get(type(endpoint.error))
            if endpoint_tag is not None:
                tags.add(endpoint_tag)

    for item in spec.variability:
        if isinstance(item, IIV):
            tags.add(CapabilityTag.VARIABILITY_IIV)
            if item.structure == "block":
                tags.add(CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE)
        elif isinstance(item, IOV):
            tags.add(CapabilityTag.VARIABILITY_IOV)

    # CovariateLink lives in its own top-level ``spec.covariates`` list as
    # of Formular sharpening plan §4 Phase 1 (P1.6), no longer among
    # ``spec.variability``'s IIV/IOV items. Tag names keep their
    # "variability.covariate_*" wire-format values unchanged (avoids a
    # gratuitous rename of a stable capability-contract string) even though
    # covariates are no longer structurally part of the variability module.
    for cov in spec.covariates:
        tags.add(CapabilityTag.VARIABILITY_COVARIATE_LINK)
        if cov.form == "maturation":
            tags.add(CapabilityTag.VARIABILITY_COVARIATE_MATURATION_FORM)

    return frozenset(tags)


class EmitterCapabilities(NamedTuple):
    """A registered emitter's declared support for the capability matrix."""

    name: str
    supports: frozenset[CapabilityTag]
    explicitly_unsupported: frozenset[CapabilityTag]


@functools.lru_cache(maxsize=1)
def _registry() -> tuple[EmitterCapabilities, ...]:
    """Build the emitter registry, importing emitter modules lazily.

    Deferred (rather than module-level) import: the emitter modules import
    ``CapabilityTag`` from *this* module, so eagerly importing them at
    ``capabilities`` module-load time would create an import cycle. Calling
    this only on first use (and caching the result) sidesteps that entirely.
    """
    from apmode.dsl import frem_emitter, nlmixr2_emitter, stan_emitter

    return (
        EmitterCapabilities(
            "nlmixr2", nlmixr2_emitter.SUPPORTS, nlmixr2_emitter.EXPLICITLY_UNSUPPORTED
        ),
        EmitterCapabilities("stan", stan_emitter.SUPPORTS, stan_emitter.EXPLICITLY_UNSUPPORTED),
        EmitterCapabilities("frem", frem_emitter.SUPPORTS, frem_emitter.EXPLICITLY_UNSUPPORTED),
    )


def registered_emitters() -> tuple[EmitterCapabilities, ...]:
    """Return the capability declarations for every registered DSL emitter."""
    return _registry()


_STATUS_SUPPORTED = "supported"
_STATUS_EXPLICITLY_UNSUPPORTED = "explicitly_unsupported"
_STATUS_UNKNOWN_GAP = "unknown_gap"
_STATUS_EXPERIMENTAL_NO_STABLE_BACKEND = "experimental_no_stable_backend"

# Tags for AST variants with no working backend anywhere (Phase 0 P0.8):
# every registered emitter raises NotImplementedError for these, so a plain
# "explicitly_unsupported" (this emitter's deliberate choice not to support
# a construct another emitter handles) undersells the real status -- there
# is no stable backend at all. Gated in the AST by
# ``DSLSpec.experimental.node`` / ``FrmCode.LANE_NODE_EXPERIMENTAL_GATE``.
_NODE_EXPERIMENTAL_TAGS: frozenset[CapabilityTag] = frozenset(
    {CapabilityTag.ABSORPTION_NODE, CapabilityTag.ELIMINATION_NODE}
)


def report(spec_or_tags: DSLSpec | Iterable[CapabilityTag]) -> dict[str, dict[str, str]]:
    """Report per-emitter support status for a spec or an explicit tag set.

    Args:
        spec_or_tags: Either a compiled :class:`DSLSpec` (resolved via
            :func:`tags_for_spec`) or any iterable of :class:`CapabilityTag`.

    Returns:
        ``{emitter_name: {tag_value: status}}`` where ``status`` is one of
        ``"supported"``, ``"experimental_no_stable_backend"`` (a NODE-tagged
        capability -- no emitter has a working code path, gated by
        ``DSLSpec.experimental.node``), ``"explicitly_unsupported"``, or
        ``"unknown_gap"`` (a tag classified in neither of the emitter's
        declared sets -- a coverage bug, since :func:`registered_emitters`
        should never have one; see ``scripts/verify_capability_coverage.py``).
    """
    tags = (
        tags_for_spec(spec_or_tags)
        if isinstance(spec_or_tags, DSLSpec)
        else frozenset(spec_or_tags)
    )

    result: dict[str, dict[str, str]] = {}
    for emitter in _registry():
        status_by_tag: dict[str, str] = {}
        for tag in sorted(tags, key=lambda t: t.value):
            if tag in emitter.supports:
                status_by_tag[tag.value] = _STATUS_SUPPORTED
            elif tag in _NODE_EXPERIMENTAL_TAGS:
                status_by_tag[tag.value] = _STATUS_EXPERIMENTAL_NO_STABLE_BACKEND
            elif tag in emitter.explicitly_unsupported:
                status_by_tag[tag.value] = _STATUS_EXPLICITLY_UNSUPPORTED
            else:
                status_by_tag[tag.value] = _STATUS_UNKNOWN_GAP
        result[emitter.name] = status_by_tag
    return result
