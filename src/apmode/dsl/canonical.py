# SPDX-License-Identifier: GPL-2.0-or-later
"""Canonical serialization and content-based fingerprints for :class:`DSLSpec`.

``DSLSpec.model_id`` (see ``apmode.ids``) is a sparkid-generated,
time-sortable, non-seedable handle — two runs that produce *structurally
identical* specs get different ``model_id`` values, so ID equality is not a
usable content-comparison signal (``apmode.ids`` documents this explicitly:
"compare bundles via content hashes ... instead of ID strings"). This module
is that content-hash layer.

Four fingerprints are exposed, each scoped to a different reproducibility
question:

* :func:`structure_fingerprint` — module topology only (variant types,
  structural/integer parameters, variability *shape*, prior
  target+family presence). Two specs with the same structure but different
  calibrated values collide here — that is the point: it lets tooling ask
  "is this the same *model*, just re-fit?"
* :func:`spec_fingerprint` — structure plus every calibrated numeric value
  (the ``DSLSpec.initial`` block, plus sigma values and prior
  hyperparameters that stay inline). Two specs collide here only if they
  are the same model with the same parameterization end to end.
* :func:`initial_fingerprint` — the ``DSLSpec.initial`` block alone (see
  Phase 1 note below).
* :func:`justification_hash` — free-text prior justification + DOI
  provenance, deliberately split out because it is prose/citation content,
  not modeling content, and changes to it (e.g. fixing a typo in a
  justification) should not be conflated with a change to the model.

All four return ``{"schema": CANONICAL_SCHEMA_VERSION, "digest": <hex>}`` —
see ``docs/FINGERPRINT_MIGRATION.md`` for the versioning contract. Digests
computed under different schema versions are never comparable.

Phase 1 migration (initial estimates)
--------------------------------------
As of v0.7, ``DSLSpec.initial: dict[str, float]`` is the single source of
truth for calibration values (``ka``, ``CL``, ``V``, ...) — structural
modules carry only topology fields (e.g. ``Transit.n``, ``SumIG.k``, NODE
``dim``/``constraint_template``, ``TimeVaryingElim.decay_fn``). Before this
migration (schema 1.0.0), calibrated values lived inline on the structural
modules and :func:`initial_fingerprint` hashed those inline fields instead;
``CANONICAL_SCHEMA_VERSION`` bumped to ``2.0.0`` with this change since the
canonicalized shape of both :func:`spec_fingerprint` (now includes the
``initial`` block explicitly) and :func:`initial_fingerprint` changed.

Phase 1 migration (covariates, P1.6)
--------------------------------------
``CovariateLink`` moved out of the ``variability`` list (formerly a sibling
of ``IIV``/``IOV``) into its own top-level ``DSLSpec.covariates`` list, and
gained per-form reference-value fields (``theta``/``ref``/``reference``/
``tm50``/``hill``). ``CANONICAL_SCHEMA_VERSION`` bumped to ``2.1.0``: both
:func:`_structure_dict` and :func:`_spec_dict` gained a new top-level
``"covariates"`` key (projected the same way ``"variability"``/``"priors"``
already were — sorted-by-canonical-json list of per-item dicts), and the
``"variability"`` key's item shape changed (no longer ever a
``CovariateLink`` dict). Field classification within a covariate item
mirrors the sigma-value precedent: ``param``/``covariate``/``form`` are
structural; ``theta``/``ref``/``tm50``/``hill`` are calibration-like
(spec-only); ``reference`` (the categorical baseline *level name*, a
string) is treated as structural since it is a discrete design choice, not
a re-estimable numeric value — see ``CovariateLink`` docstring.

Phase 1 migration (multi-analyte observations, P1.7)
--------------------------------------
``DSLSpec.observations`` (the optional multi-analyte ``observations:``
block) is a new top-level field; ``CANONICAL_SCHEMA_VERSION`` bumped to
``2.2.0`` and both :func:`_structure_dict` and :func:`_spec_dict` gained a
new top-level ``"observations"`` key (empty list when unset), projected the
same sorted-by-canonical-json way ``"variability"``/``"priors"``/
``"covariates"`` already are. This is additive, not a reclassification: the
existing ``"observation"`` key (singular) is unchanged and always reflects
``DSLSpec.observation`` (the first ``observations:`` entry when that block
was used — see ``DSLSpec.observations`` docstring). Without this addition
two multi-analyte specs differing only in their ``observations:`` content
(different dvids/predictions/second-endpoint error models) would
incorrectly collide on both fingerprints, since only the single proxy
``observation`` field would be visible to the hash.

Phase 2 addition (macro provenance, P2.1) — no schema bump
------------------------------------------------------------
``DSLSpec.macros_used`` (the audit trail for ``use <macro>`` statement
expansion) is deliberately *excluded* from both :func:`structure_fingerprint`
and :func:`spec_fingerprint`, and this required no
``CANONICAL_SCHEMA_VERSION`` bump. Unlike a hypothetical implementation that
hashes ``spec.model_dump(mode="json")`` wholesale, :func:`_structure_dict`
and :func:`_spec_dict` below both hand-build their projection dict from an
explicit, named field list — adding a new top-level field to ``DSLSpec``
does not automatically appear in either dict's output; only fields this
module's authors deliberately add to those two functions do. So
``macros_used`` needed no code change here at all to stay excluded, and two
specs that are byte-identical after macro expansion fingerprint identically
whether or not either used a ``use`` shortcut to get there — exactly the
"sugar, not semantics" property macro expansion is supposed to have. A
future non-provenance field added to ``DSLSpec`` still requires the author
to explicitly decide whether it belongs in ``_structure_dict``/``_spec_dict``
(and bump the schema version if the answer is "structure_fingerprint should
now see it") — this exclusion-by-omission property does not remove that
obligation, it just means an *omitted* field defaults to excluded rather
than silently included.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, cast

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    CovariateLink,
    Erlang,
    IVBolus,
    NODEAbsorption,
    NODEElimination,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
    SumIG,
    TimeVaryingElim,
    Transit,
)

if TYPE_CHECKING:
    from apmode.dsl.ast_models import (
        AbsorptionModule,
        DistributionModule,
        DSLSpec,
        EliminationModule,
        ObservationEndpoint,
        ObservationModule,
        OccasionSpec,
        VariabilityItem,
    )
    from apmode.dsl.priors import PriorSpec

CANONICAL_SCHEMA_VERSION = "2.2.0"

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

FingerprintResult = dict[str, str]


def _canonical_json_bytes(value: JSONValue) -> bytes:
    """Serialize ``value`` with sorted keys and a stable separator set."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: JSONValue) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sorted_by_canonical_json(items: list[JSONValue]) -> list[JSONValue]:
    """Sort a list of JSON-serializable values by their canonical JSON string.

    This is a stable, deterministic total order for any list of
    dicts/scalars regardless of construction/insertion order — used for
    ``variability`` and ``priors``, both of which are semantically
    unordered sets from a fingerprinting point of view.
    """
    return sorted(items, key=lambda v: _canonical_json_bytes(v))


def _result(digest: str) -> FingerprintResult:
    return {"schema": CANONICAL_SCHEMA_VERSION, "digest": digest}


# ---------------------------------------------------------------------------
# Structural (topology-only) module projections
# ---------------------------------------------------------------------------
#
# Every function below returns only "module identity" fields: variant type,
# structural integer/categorical parameters (compartment/transit counts,
# NODE dims, decay-function choice, IIV/IOV/covariate shape). Calibrated
# numeric values (rate constants, volumes, sigmas, prior hyperparameters)
# are never included here — that is exactly the structure/spec split.


def _absorption_structure(mod: AbsorptionModule) -> dict[str, JSONValue]:
    if isinstance(mod, Transit):
        return {"type": mod.type, "n": mod.n}
    if isinstance(mod, Erlang):
        return {"type": mod.type, "n": mod.n}
    if isinstance(mod, SumIG):
        return {"type": mod.type, "k": mod.k}
    if isinstance(mod, NODEAbsorption):
        return {"type": mod.type, "dim": mod.dim, "constraint_template": mod.constraint_template}
    if isinstance(mod, IVBolus):
        return {"type": mod.type}
    # FirstOrder, ZeroOrder, LaggedFirstOrder, MixedFirstZero, ParallelFirstOrder:
    # every field is a calibrated value; only the variant type is structural.
    return {"type": mod.type}


def _distribution_structure(mod: DistributionModule) -> dict[str, JSONValue]:
    # OneCmt/TwoCmt/ThreeCmt/TMDDCore/TMDDQSS: every field is a calibrated
    # value (volumes, Q, R0, kon/koff/kint, KD) — only the variant type is
    # structural.
    return {"type": mod.type}


def _elimination_structure(mod: EliminationModule) -> dict[str, JSONValue]:
    if isinstance(mod, TimeVaryingElim):
        return {"type": mod.type, "decay_fn": mod.decay_fn}
    if isinstance(mod, NODEElimination):
        return {"type": mod.type, "dim": mod.dim, "constraint_template": mod.constraint_template}
    # LinearElim, MichaelisMenten, ParallelLinearMM: values only.
    return {"type": mod.type}


def _occasion_structure(spec: OccasionSpec) -> dict[str, JSONValue]:
    if isinstance(spec, OccasionByStudy):
        return {"type": spec.type}
    if isinstance(spec, OccasionByVisit | OccasionByDoseEpoch | OccasionCustom):
        return {"type": spec.type, "column": spec.column}
    return {"type": spec.type}


def _variability_item_structure(item: VariabilityItem) -> dict[str, JSONValue]:
    if isinstance(item, IIV):
        return {
            "type": item.type,
            "params": cast("JSONValue", sorted(item.params)),
            "structure": item.structure,
        }
    if isinstance(item, IOV):
        return {
            "type": item.type,
            "params": cast("JSONValue", sorted(item.params)),
            "occasions": _occasion_structure(item.occasions),
        }
    msg = f"unhandled VariabilityItem variant: {item!r}"
    raise TypeError(msg)


def _covariate_structure(item: CovariateLink) -> dict[str, JSONValue]:
    """Structural projection of a top-level ``covariates:`` entry (P1.6).

    ``param``/``covariate``/``form`` are structural. ``reference`` (the
    categorical baseline level name) is also included here — it is a
    discrete design choice, not a re-estimable numeric value — but
    ``theta``/``ref``/``tm50``/``hill`` are calibration-like and excluded
    (present only in :func:`_covariate_full`, hence in ``spec_fingerprint``).
    """
    out: dict[str, JSONValue] = {
        "type": item.type,
        "param": item.param,
        "covariate": item.covariate,
        "form": item.form,
    }
    if item.reference is not None:
        out["reference"] = item.reference
    return out


def _covariate_full(item: CovariateLink) -> dict[str, JSONValue]:
    return item.model_dump(mode="json")


def _observation_structure(mod: ObservationModule) -> dict[str, JSONValue]:
    if isinstance(mod, BLQM3 | BLQM4):
        return {"type": mod.type, "error_model": mod.error_model}
    # Proportional/Additive/Combined: sigma_* values only.
    return {"type": mod.type}


def _observation_endpoint_structure(item: ObservationEndpoint) -> dict[str, JSONValue]:
    """Structural projection of a multi-analyte ``observations:`` entry (P1.7).

    ``name``/``dvid``/``prediction`` are structural (they identify *which*
    analyte this is and how it is routed/observed); the nested ``error``
    module is projected the same way the singular ``observation`` field is.
    """
    return {
        "name": item.name,
        "dvid": item.dvid,
        "prediction": item.prediction,
        "error": _observation_structure(item.error),
    }


def _observation_endpoint_full(item: ObservationEndpoint) -> dict[str, JSONValue]:
    return item.model_dump(mode="json")


def _prior_structure(prior: PriorSpec) -> dict[str, JSONValue]:
    """Prior target + family *presence* only — no hyperparameters, source, or provenance."""
    return {"target": prior.target, "family_type": prior.family.type}


def _structure_dict(spec: DSLSpec) -> dict[str, JSONValue]:
    return {
        "absorption": _absorption_structure(spec.absorption),
        "distribution": _distribution_structure(spec.distribution),
        "elimination": _elimination_structure(spec.elimination),
        "variability": _sorted_by_canonical_json(
            [_variability_item_structure(item) for item in spec.variability]
        ),
        "covariates": _sorted_by_canonical_json(
            [_covariate_structure(item) for item in spec.covariates]
        ),
        "observation": _observation_structure(spec.observation),
        "observations": _sorted_by_canonical_json(
            [_observation_endpoint_structure(ep) for ep in (spec.observations or {}).values()]
        ),
        "priors": _sorted_by_canonical_json([_prior_structure(p) for p in spec.priors]),
    }


# ---------------------------------------------------------------------------
# Full (structure + calibrated values) module projections
# ---------------------------------------------------------------------------
#
# These reuse Pydantic's own ``model_dump(mode="json")`` for each module —
# it already includes every field (structural and calibrated) — so the only
# canonicalization work left is dropping the non-deterministic / provenance
# fields (justification, doi) and imposing a stable list order.


def _prior_full(prior: PriorSpec) -> dict[str, JSONValue]:
    dumped = prior.model_dump(mode="json")
    dumped.pop("justification", None)
    dumped.pop("doi", None)
    return dumped


def _variability_item_full(item: VariabilityItem) -> dict[str, JSONValue]:
    """Full (structure + calibrated) dump of a variability item.

    ``IIV``/``IOV.params`` is semantically an unordered parameter set (the
    same block declared as ``["CL", "V"]`` or ``["V", "CL"]`` is the same
    variability declaration), so it is sorted here exactly as it is for
    :func:`_variability_item_structure` — otherwise a spec that differs
    only in declaration order would spuriously fail to collide on
    :func:`spec_fingerprint`.
    """
    dumped = item.model_dump(mode="json")
    params = dumped.get("params")
    if isinstance(params, list):
        dumped["params"] = sorted(params)
    return dumped


def _spec_dict(spec: DSLSpec) -> dict[str, JSONValue]:
    return {
        "absorption": spec.absorption.model_dump(mode="json"),
        "distribution": spec.distribution.model_dump(mode="json"),
        "elimination": spec.elimination.model_dump(mode="json"),
        "variability": _sorted_by_canonical_json(
            [_variability_item_full(item) for item in spec.variability]
        ),
        "covariates": _sorted_by_canonical_json(
            [_covariate_full(item) for item in spec.covariates]
        ),
        "observation": spec.observation.model_dump(mode="json"),
        "observations": _sorted_by_canonical_json(
            [_observation_endpoint_full(ep) for ep in (spec.observations or {}).values()]
        ),
        "priors": _sorted_by_canonical_json([_prior_full(p) for p in spec.priors]),
        "initial": cast("JSONValue", dict(spec.initial)),
    }


# ---------------------------------------------------------------------------
# Public fingerprint API
# ---------------------------------------------------------------------------


def canonicalize(spec: DSLSpec) -> dict[str, JSONValue]:
    """Return the full (structure + calibrated-value) canonical dict for ``spec``.

    Deterministic across dict-insertion order and list order: keys are
    sorted at serialization time (via :func:`_canonical_json_bytes`) and
    list-valued fields (``variability``, ``priors``) are sorted by their
    own canonical JSON representation. Excludes ``model_id`` and
    ``source_meta`` (non-deterministic / parse-position metadata) and, per
    prior, ``justification``/``doi`` (see :func:`justification_hash`).
    """
    return _spec_dict(spec)


def structure_fingerprint(spec: DSLSpec) -> FingerprintResult:
    """Sha256 over module topology only — excludes every calibrated numeric value.

    Excludes: ``model_id``, ``source_meta``, all initial-estimate-like
    numeric fields (ka, CL, V, sigma_*, loq_value, prior hyperparameters,
    ...), prior ``source``/``justification``/``doi``/``historical_refs``.
    Two specs that differ only in calibrated values (e.g. re-estimated
    THETA) produce the *same* structure_fingerprint.
    """
    return _result(_digest(_structure_dict(spec)))


def spec_fingerprint(spec: DSLSpec) -> FingerprintResult:
    """Sha256 over structure plus every calibrated numeric value.

    Excludes only: ``model_id``, ``source_meta``, prior ``justification``,
    prior ``doi``. Two specs collide here only if they are identical in
    both topology and calibrated parameterization.
    """
    return _result(_digest(_spec_dict(spec)))


def initial_fingerprint(spec: DSLSpec) -> FingerprintResult:
    """Sha256 over the ``DSLSpec.initial`` calibration-value block alone (no topology).

    Two specs with identical structure but re-fit/re-estimated ``initial``
    values produce different digests here; two specs with different
    structure but the same calibration values (e.g. same ``CL``/``V``
    happen to be reused after a structural swap) still collide here since
    topology is intentionally excluded — pair with
    :func:`structure_fingerprint` when both signals are needed.
    """
    return _result(_digest(cast("JSONValue", dict(spec.initial))))


def justification_hash(spec: DSLSpec) -> FingerprintResult:
    """Sha256 over sorted justification text + DOI strings across ``spec.priors``.

    Deliberately isolated from :func:`spec_fingerprint` — editing a prior's
    prose justification or fixing a DOI typo is a provenance/documentation
    change, not a modeling change, and should not perturb the modeling
    fingerprints.
    """
    entries = sorted(f"{p.justification} {p.doi or ''}" for p in spec.priors)
    return _result(_digest(cast("JSONValue", entries)))


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "FingerprintResult",
    "canonicalize",
    "initial_fingerprint",
    "justification_hash",
    "spec_fingerprint",
    "structure_fingerprint",
]
