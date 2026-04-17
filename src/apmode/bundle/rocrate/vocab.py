# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE extension vocabulary for RO-Crate projection.

The ``apmode:`` namespace carries policy/gate/regulatory crosswalk fields
that standard WRROC vocabulary does not cover (FDA PCCP elements, EU AI
Act Article 12 record-keeping, cross-paradigm NLPD scoring-contract
provenance). See ``_research/ROCRATE_INTEGRATION_PLAN.md`` §D for the
full term table and cardinality constraints.

URI layout: ``https://w3id.org/apmode/terms#<TermName>``. The w3id
redirect is tracked as a separate ops task — during development the
constant is still the canonical identifier and importers that do not
understand the namespace ignore it per JSON-LD semantics.
"""

from __future__ import annotations

from enum import StrEnum

APMODE_TERMS_BASE = "https://w3id.org/apmode/terms#"
"""Base URI for the APMODE extension vocabulary."""

PROVAGENT_TERMS_BASE = "https://w3id.org/provagent#"
"""Base URI for PROV-AGENT (Souza et al., arXiv:2508.02866). Opt-in via
:func:`apmode.bundle.rocrate.context.build_rocrate_context(include_provagent=True)`;
used only when the agentic backend run is being projected (v0.9 scope)."""

# --- Lane / policy ---

LANE = "apmode:lane"
LANE_POLICY = "apmode:lanePolicy"

# --- Governance gates ---

GATE = "apmode:gate"
GATE_RATIONALE = "apmode:gateRationale"

# --- Search provenance ---

CANDIDATE_LINEAGE_EDGE = "apmode:candidateLineageEdge"
SEARCH_GRAPH = "apmode:searchGraph"

# --- Regulatory ---

MODIFICATION_DESCRIPTION = "apmode:modificationDescription"
MODIFICATION_PROTOCOL = "apmode:modificationProtocol"
IMPACT_ASSESSMENT = "apmode:impactAssessment"
TRACEABILITY_TABLE = "apmode:traceabilityTable"
REGULATORY_CONTEXT = "apmode:regulatoryContext"

# --- DSL / backends ---

DSL_SPEC = "apmode:dslSpec"
DSL_TRANSFORM = "apmode:dslTransform"
LLM_INVOCATION = "apmode:llmInvocation"
CREDIBILITY_REPORT = "apmode:credibilityReport"
LORO_CV = "apmode:loroCV"
SCORING_CONTRACT = "apmode:scoringContract"
NLPD_COMPARABILITY_PROTOCOL = "apmode:nlpdComparabilityProtocol"

# --- Integrity ---

COMPLETE_SENTINEL_TYPE = "apmode:completeSentinel"
"""``additionalType`` value for the ``_COMPLETE`` bundle-integrity
sentinel File entity. Consumers that know this type can cross-verify
the SHA-256 bundle digest regardless of the RO-Crate packaging."""

SBOM_TYPE = "apmode:sbom"
"""``additionalType`` value for the producer-side CycloneDX SBOM
(``bom.cdx.json``). Carried in the crate so consumers can audit the
Python dependency graph that produced the bundle without fetching
release-asset sidecars. Excluded from the sealed-bundle digest so
regenerating the SBOM never invalidates ``_COMPLETE``."""


class RegulatoryContext(StrEnum):
    """Allowed values of ``apmode:regulatoryContext``.

    Declared on the root ``Dataset`` to advertise which regulatory regime
    (if any) the bundle is being prepared under. ``research-only`` is the
    default for an unflagged run; other values indicate that the
    downstream PCCP / MDR / Article-12 artifacts are expected to be
    present in the ``regulatory/`` subdirectory (v0.8+).
    """

    RESEARCH_ONLY = "research-only"
    PCCP_AI_DSF = "pccp-ai-dsf"
    MDR = "mdr"
    AI_ACT_ARTICLE_12 = "ai-act-article-12"
