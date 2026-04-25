# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for candidate / CreateAction projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities import backend as ent_backend

from ._fixtures import build_submission_bundle


class TestCandidateSoftwareApp:
    def test_mints_candidate_app_id(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        app_id = ent_backend.add_candidate_software_app(graph, bundle, "c001")

        assert app_id == "#candidate-c001"
        app = next(e for e in graph if e["@id"] == app_id)
        assert app["@type"] == "SoftwareApplication"
        assert app["apmode:dslSpec"] == {"@id": "compiled_specs/c001.json"}

    def test_links_r_lowering_hasPart(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        ent_backend.add_candidate_software_app(graph, bundle, "c001")

        app = next(e for e in graph if e["@id"] == "#candidate-c001")
        assert app["hasPart"] == [{"@id": "compiled_specs/c001.R"}]

    def test_idempotent(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        ent_backend.add_candidate_software_app(graph, bundle, "c001")
        ent_backend.add_candidate_software_app(graph, bundle, "c001")

        apps = [e for e in graph if e["@id"] == "#candidate-c001"]
        assert len(apps) == 1


class TestBackendCreateAction:
    def test_projects_completed_status(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [{"@id": "./", "@type": "Dataset"}]

        action_id = ent_backend.add_backend_create_action(
            graph,
            bundle,
            "c001",
            data_manifest_id="data_manifest.json",
            organize_action_id="#run-organize-action",
        )

        assert action_id == "#backend-create-c001"
        action = next(e for e in graph if e["@id"] == action_id)
        assert action["@type"] == "CreateAction"
        assert action["actionStatus"] == {"@id": "http://schema.org/CompletedActionStatus"}
        # instrument points at the engine SoftwareApplication (WRROC
        # semantics: the tool being invoked is the backend engine, not
        # the candidate DSL spec).
        assert action["instrument"] == {"@id": "#engine-nlmixr2"}
        # object carries the inputs: the data manifest plus the
        # candidate's DSL SoftwareApplication.
        assert {"@id": "data_manifest.json"} in action["object"]
        assert {"@id": "#candidate-c001"} in action["object"]
        # Engine entity is registered with name + version.
        engine = next(e for e in graph if e["@id"] == "#engine-nlmixr2")
        assert engine["@type"] == "SoftwareApplication"
        assert engine["softwareVersion"] == "3.0.0"
        # Backend HowToStep is present with workExample → engine.
        step = next(e for e in graph if e["@id"] == "#step-backend-nlmixr2")
        assert step["@type"] == "HowToStep"
        assert step["workExample"] == {"@id": "#engine-nlmixr2"}

    def test_organize_action_captures_create_action(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("c001",))
        graph: list[dict[str, Any]] = [
            {"@id": "./", "@type": "Dataset"},
            {"@id": "#run-organize-action", "@type": "OrganizeAction"},
        ]

        ent_backend.add_backend_create_action(
            graph,
            bundle,
            "c001",
            data_manifest_id="data_manifest.json",
            organize_action_id="#run-organize-action",
        )

        organize = next(e for e in graph if e["@id"] == "#run-organize-action")
        assert {"@id": "#backend-create-c001"} in organize["object"]


class TestCollectHelpers:
    def test_collect_result_ids(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("a", "b"))
        assert ent_backend.collect_result_ids(bundle) == ["a", "b"]

    def test_collect_candidate_ids(self, tmp_path: Path) -> None:
        bundle = build_submission_bundle(tmp_path, candidate_ids=("a", "b"))
        assert ent_backend.collect_candidate_ids(bundle) == ["a", "b"]
