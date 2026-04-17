# SPDX-License-Identifier: GPL-2.0-or-later
"""JSON-LD ``@context`` construction and profile URI constants.

The context is a three-element list:

1. The official RO-Crate 1.1 context (brings in schema.org, bioschemas,
   and ro-crate / pcdm terms).
2. The Workflow Run RO-Crate terms context (brings in ``wfrun:``,
   ``ControlAction``, ``OrganizeAction`` typing).
3. An inline dict with the ``apmode:`` / ``prov:`` namespaces plus
   short aliases for the most frequently used extension terms — this
   makes the serialized graph compact and human-readable while keeping
   the canonical URI accessible.

``include_provagent=True`` adds the PROV-AGENT namespace (v0.9 scope,
kept opt-in so v0.6 crates remain stable for importers that do not
know it yet).
"""

from __future__ import annotations

from apmode.bundle.rocrate.vocab import APMODE_TERMS_BASE, PROVAGENT_TERMS_BASE

# --- Profile URIs (normative) ---

ROCRATE_1_1 = "https://w3id.org/ro/crate/1.1"
ROCRATE_1_1_CONTEXT = "https://w3id.org/ro/crate/1.1/context"

WRROC_PROCESS_0_5 = "https://w3id.org/ro/wfrun/process/0.5"
WRROC_WORKFLOW_0_5 = "https://w3id.org/ro/wfrun/workflow/0.5"
WRROC_PROVENANCE_0_5 = "https://w3id.org/ro/wfrun/provenance/0.5"

WRROC_TERMS_CONTEXT = "https://w3id.org/ro/terms/workflow-run/context"
"""JSON-LD terms context for Workflow Run RO-Crate (imports wfrun,
schema.org extensions, prov aliases)."""

BIOSCHEMAS_COMPUTATIONAL_WORKFLOW = "https://bioschemas.org/profiles/ComputationalWorkflow/"

# --- License ---

GPL_2_OR_LATER = "https://spdx.org/licenses/GPL-2.0-or-later.html"
"""SPDX URI for the project license. Emitted on the root Dataset and
on created ComputationalWorkflow entities because ro-crate-1.1 REQUIRED
validation demands a root-level license property."""

# --- PROV / schema.org identifiers we embed as @id values ---

PROV_WAS_DERIVED_FROM = "prov:wasDerivedFrom"
PROV_WAS_INFORMED_BY = "prov:wasInformedBy"

SCHEMA_COMPLETED_ACTION_STATUS = "http://schema.org/CompletedActionStatus"
SCHEMA_FAILED_ACTION_STATUS = "http://schema.org/FailedActionStatus"


def build_rocrate_context(*, include_provagent: bool = False) -> list[str | dict[str, str]]:
    """Build the JSON-LD ``@context`` for an APMODE RO-Crate.

    Args:
        include_provagent: When True, the inline namespace dict also
            binds ``provagent:`` to the PROV-AGENT namespace. Default
            False so v0.6 crates remain compatible with importers that
            don't yet understand PROV-AGENT (it is pre-1.0).

    Returns:
        A list suitable for direct use as the value of ``@context`` in
        ``ro-crate-metadata.json``. Callers should not mutate the
        returned object — its third element is a shared dict, so copy
        it first if you need to extend.
    """
    inline: dict[str, str] = {
        "apmode": APMODE_TERMS_BASE,
        "prov": "http://www.w3.org/ns/prov#",
        # Explicit full-prefix PROV term entries — the projector emits
        # ``prov:wasDerivedFrom`` / ``prov:wasInformedBy`` keys, and
        # roc-validator's compaction check wants each to resolve via
        # the context directly rather than via the namespace prefix.
        "prov:wasDerivedFrom": "http://www.w3.org/ns/prov#wasDerivedFrom",
        "prov:wasInformedBy": "http://www.w3.org/ns/prov#wasInformedBy",
        # Short aliases for commonly-used extension terms — prefer the
        # full prefix form in the graph (``"apmode:lane"``) so the
        # short form is only a display-side aid.
        "lane": "apmode:lane",
        "lanePolicy": "apmode:lanePolicy",
        "gate": "apmode:gate",
        "gateRationale": "apmode:gateRationale",
        "candidateLineageEdge": "apmode:candidateLineageEdge",
        "searchGraph": "apmode:searchGraph",
        "regulatoryContext": "apmode:regulatoryContext",
        "modificationDescription": "apmode:modificationDescription",
        "modificationProtocol": "apmode:modificationProtocol",
        "impactAssessment": "apmode:impactAssessment",
        "traceabilityTable": "apmode:traceabilityTable",
        "dslSpec": "apmode:dslSpec",
        "dslTransform": "apmode:dslTransform",
        "llmInvocation": "apmode:llmInvocation",
        "credibilityReport": "apmode:credibilityReport",
        "loroCV": "apmode:loroCV",
        "scoringContract": "apmode:scoringContract",
        "nlpdComparabilityProtocol": "apmode:nlpdComparabilityProtocol",
        # Explicit full-prefix mappings for each extension term — the
        # projector emits keys in the ``apmode:X`` form, and
        # roc-validator's SHACL compaction check wants each exact key
        # to resolve via the context. Entries below are equivalent to
        # the short aliases above via the ``apmode:`` prefix, but
        # declaring them explicitly satisfies the validator.
        "apmode:lane": f"{APMODE_TERMS_BASE}lane",
        "apmode:lanePolicy": f"{APMODE_TERMS_BASE}lanePolicy",
        "apmode:gate": f"{APMODE_TERMS_BASE}gate",
        "apmode:gateRationale": f"{APMODE_TERMS_BASE}gateRationale",
        "apmode:candidateLineageEdge": f"{APMODE_TERMS_BASE}candidateLineageEdge",
        "apmode:searchGraph": f"{APMODE_TERMS_BASE}searchGraph",
        "apmode:regulatoryContext": f"{APMODE_TERMS_BASE}regulatoryContext",
        "apmode:modificationDescription": f"{APMODE_TERMS_BASE}modificationDescription",
        "apmode:modificationProtocol": f"{APMODE_TERMS_BASE}modificationProtocol",
        "apmode:impactAssessment": f"{APMODE_TERMS_BASE}impactAssessment",
        "apmode:traceabilityTable": f"{APMODE_TERMS_BASE}traceabilityTable",
        "apmode:dslSpec": f"{APMODE_TERMS_BASE}dslSpec",
        "apmode:dslTransform": f"{APMODE_TERMS_BASE}dslTransform",
        "apmode:llmInvocation": f"{APMODE_TERMS_BASE}llmInvocation",
        "apmode:credibilityReport": f"{APMODE_TERMS_BASE}credibilityReport",
        "apmode:loroCV": f"{APMODE_TERMS_BASE}loroCV",
        "apmode:scoringContract": f"{APMODE_TERMS_BASE}scoringContract",
        "apmode:nlpdComparabilityProtocol": (f"{APMODE_TERMS_BASE}nlpdComparabilityProtocol"),
    }
    if include_provagent:
        inline["provagent"] = PROVAGENT_TERMS_BASE
    return [ROCRATE_1_1_CONTEXT, WRROC_TERMS_CONTEXT, inline]
