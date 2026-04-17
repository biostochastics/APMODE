# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the JSON-LD @context builder and profile URI constants."""

from __future__ import annotations

import json

from apmode.bundle.rocrate import context


class TestProfileUris:
    def test_rocrate_1_1_context(self) -> None:
        assert context.ROCRATE_1_1_CONTEXT == "https://w3id.org/ro/crate/1.1/context"

    def test_rocrate_1_1_profile(self) -> None:
        assert context.ROCRATE_1_1 == "https://w3id.org/ro/crate/1.1"

    def test_wfrun_workflow_run_0_5(self) -> None:
        assert context.WRROC_WORKFLOW_0_5 == "https://w3id.org/ro/wfrun/workflow/0.5"

    def test_wfrun_process_run_0_5(self) -> None:
        assert context.WRROC_PROCESS_0_5 == "https://w3id.org/ro/wfrun/process/0.5"

    def test_wfrun_provenance_run_0_5(self) -> None:
        assert context.WRROC_PROVENANCE_0_5 == "https://w3id.org/ro/wfrun/provenance/0.5"

    def test_workflow_run_terms_context(self) -> None:
        assert context.WRROC_TERMS_CONTEXT == "https://w3id.org/ro/terms/workflow-run/context"

    def test_bioschemas_computational_workflow(self) -> None:
        assert context.BIOSCHEMAS_COMPUTATIONAL_WORKFLOW == (
            "https://bioschemas.org/profiles/ComputationalWorkflow/"
        )


class TestBuildRocrateContext:
    def test_returns_list_of_contexts(self) -> None:
        ctx = context.build_rocrate_context()
        assert isinstance(ctx, list)
        assert len(ctx) == 3
        assert ctx[0] == context.ROCRATE_1_1_CONTEXT
        assert ctx[1] == context.WRROC_TERMS_CONTEXT
        assert isinstance(ctx[2], dict)

    def test_inline_context_includes_apmode_namespace(self) -> None:
        ctx = context.build_rocrate_context()
        inline = ctx[2]
        assert inline["apmode"] == "https://w3id.org/apmode/terms#"

    def test_inline_context_includes_prov_namespace(self) -> None:
        ctx = context.build_rocrate_context()
        inline = ctx[2]
        assert inline["prov"] == "http://www.w3.org/ns/prov#"

    def test_inline_context_has_key_term_shortcuts(self) -> None:
        ctx = context.build_rocrate_context()
        inline = ctx[2]
        for key in ("lane", "lanePolicy", "gate", "gateRationale", "dslSpec"):
            assert key in inline, f"expected {key!r} in inline context"
            assert inline[key].startswith("apmode:")

    def test_provagent_excluded_by_default(self) -> None:
        ctx = context.build_rocrate_context()
        inline = ctx[2]
        assert "provagent" not in inline

    def test_provagent_included_when_opted_in(self) -> None:
        ctx = context.build_rocrate_context(include_provagent=True)
        inline = ctx[2]
        assert "provagent" in inline
        assert inline["provagent"] == "https://w3id.org/provagent#"

    def test_context_is_json_serializable(self) -> None:
        ctx = context.build_rocrate_context()
        serialized = json.dumps(ctx)
        reparsed = json.loads(serialized)
        assert reparsed[0] == context.ROCRATE_1_1_CONTEXT
