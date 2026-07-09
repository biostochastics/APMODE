# DSLSpec Fingerprint Schema — Versioning Policy

`apmode.dsl.canonical` computes four sha256 fingerprints over a compiled
`DSLSpec`: `structure_fingerprint`, `spec_fingerprint`, `initial_fingerprint`,
and `justification_hash`. Every one of them returns
`{"schema": CANONICAL_SCHEMA_VERSION, "digest": "<hex>"}` rather than a bare
digest string, so the schema version travels with every digest that is
persisted or compared.

## What `CANONICAL_SCHEMA_VERSION` means

The version pins the *canonicalization contract*: exactly which `DSLSpec`
fields are included, which are excluded, how nested modules are projected to
a JSON-serializable dict, and how list-valued fields (`variability`,
`priors`) are ordered before hashing. Two digests are only meaningfully
comparable if they were produced under the same schema version — the digest
is a hash of a *shape*, and the shape itself is versioned separately from
`DSLSpec`'s own Pydantic field set.

## When to bump the version

Bump `CANONICAL_SCHEMA_VERSION` (and only then) whenever any of the
following change:

- The field-inclusion/exclusion set for any of the four fingerprints (e.g.
  a field currently treated as "structural" is reclassified as a
  "calibrated value", or vice versa).
- A `DSLSpec` field is renamed, reordered, restructured, or split into a
  new sub-model in a way that changes the shape of `model_dump(mode="json")`
  output being canonicalized — most notably, the Phase 1 migration that
  moves initial estimates out of the structural modules and into their own
  block (see the module docstring in `src/apmode/dsl/canonical.py`).
- A new `DSLSpec` module variant is added whose structural vs. calibrated
  field split does not fit the existing per-variant projection functions
  and requires a new categorization decision.
- The sort key or serialization parameters used to make list ordering
  deterministic change (key function, JSON separators, key-sorting rule).

Routine additions that do not change what is included/excluded or how it is
serialized (e.g. adding a new prior family whose hyperparameters are handled
by the existing generic `model_dump` path) do **not** require a bump.

## The explicit contract: cross-version digests are never equal

Fingerprints computed under different `CANONICAL_SCHEMA_VERSION` values are
**never treated as equal**, even if the underlying `DSLSpec` content would
canonicalize to the same bytes under a hypothetical unified schema. Any
consumer that compares two fingerprint dicts (bundle diffing, candidate
deduplication, lineage tracking) must first compare the `schema` field and
short-circuit to "not comparable" / "different" on a mismatch — it must
never fall through to comparing `digest` alone across schema versions. This
keeps a schema bump from silently producing false "unchanged" or false
"changed" verdicts across an upgrade boundary.

## Where fingerprints live in the bundle

`BundleEmitter.write_compiled_spec` writes
`compiled_specs/{model_id}_fingerprints.json` alongside the existing
`compiled_specs/{model_id}.json` (+ `.R`) artifacts, containing all four
fingerprint dicts keyed by name. This file is deterministic content derived
from the sealed spec, so it participates in the normal bundle digest
computation (`_compute_bundle_digest`) like any other artifact — it is
**not** added to `_DIGEST_EXCLUDED_RELATIVE_PATHS`.
