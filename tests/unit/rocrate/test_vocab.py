# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the ``apmode:`` vocabulary constants."""

from __future__ import annotations

from apmode.bundle.rocrate import vocab


class TestApmodeTermsBase:
    def test_base_uri_is_w3id_apmode(self) -> None:
        assert vocab.APMODE_TERMS_BASE == "https://w3id.org/apmode/terms#"

    def test_base_ends_with_hash_for_fragment_appending(self) -> None:
        assert vocab.APMODE_TERMS_BASE.endswith("#")


class TestTermConstants:
    def test_lane_term_is_under_apmode_namespace(self) -> None:
        assert vocab.LANE == "apmode:lane"

    def test_lane_policy_term(self) -> None:
        assert vocab.LANE_POLICY == "apmode:lanePolicy"

    def test_gate_terms(self) -> None:
        assert vocab.GATE == "apmode:gate"
        assert vocab.GATE_RATIONALE == "apmode:gateRationale"

    def test_regulatory_and_pccp_terms(self) -> None:
        assert vocab.REGULATORY_CONTEXT == "apmode:regulatoryContext"
        assert vocab.MODIFICATION_DESCRIPTION == "apmode:modificationDescription"
        assert vocab.MODIFICATION_PROTOCOL == "apmode:modificationProtocol"
        assert vocab.IMPACT_ASSESSMENT == "apmode:impactAssessment"
        assert vocab.TRACEABILITY_TABLE == "apmode:traceabilityTable"

    def test_dsl_terms(self) -> None:
        assert vocab.DSL_SPEC == "apmode:dslSpec"
        assert vocab.DSL_TRANSFORM == "apmode:dslTransform"

    def test_complete_sentinel_type(self) -> None:
        assert vocab.COMPLETE_SENTINEL_TYPE == "apmode:completeSentinel"

    def test_llm_invocation_term(self) -> None:
        assert vocab.LLM_INVOCATION == "apmode:llmInvocation"

    def test_credibility_report_term(self) -> None:
        assert vocab.CREDIBILITY_REPORT == "apmode:credibilityReport"

    def test_search_graph_term(self) -> None:
        assert vocab.SEARCH_GRAPH == "apmode:searchGraph"

    def test_loro_cv_term(self) -> None:
        assert vocab.LORO_CV == "apmode:loroCV"

    def test_scoring_contract_term(self) -> None:
        assert vocab.SCORING_CONTRACT == "apmode:scoringContract"

    def test_nlpd_comparability_protocol_term(self) -> None:
        assert vocab.NLPD_COMPARABILITY_PROTOCOL == "apmode:nlpdComparabilityProtocol"


class TestRegulatoryContextEnum:
    def test_enum_values(self) -> None:
        assert vocab.RegulatoryContext.RESEARCH_ONLY == "research-only"
        assert vocab.RegulatoryContext.PCCP_AI_DSF == "pccp-ai-dsf"
        assert vocab.RegulatoryContext.MDR == "mdr"
        assert vocab.RegulatoryContext.AI_ACT_ARTICLE_12 == "ai-act-article-12"
