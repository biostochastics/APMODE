# SPDX-License-Identifier: GPL-2.0-or-later
"""Projector that turns a sealed APMODE bundle into an RO-Crate.

The projector is a deterministic, read-only transformation: it never
mutates the source bundle, and for the same sealed bundle it produces
byte-identical ``ro-crate-metadata.json`` output (ignoring the
optionally-injected ``datePublished`` timestamp).

Two output forms are supported:

- **Directory form** — ``out`` is a directory path; every bundle file is
  copied into it and ``ro-crate-metadata.json`` is written at the root.
- **ZIP form** — ``out`` ends with ``.zip``; the same tree is produced
  inside a ZIP archive. ZIP entries are written in sorted order and
  with a fixed UTC timestamp so the archive is reproducible.
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from apmode.bundle.rocrate import context as ctx
from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities import (
    agentic as ent_agentic,
)
from apmode.bundle.rocrate.entities import (
    backend as ent_backend,
)
from apmode.bundle.rocrate.entities import (
    bayesian as ent_bayesian,
)
from apmode.bundle.rocrate.entities import (
    credibility as ent_credibility,
)
from apmode.bundle.rocrate.entities import (
    data as ent_data,
)
from apmode.bundle.rocrate.entities import (
    gate as ent_gate,
)
from apmode.bundle.rocrate.entities import (
    lineage as ent_lineage,
)
from apmode.bundle.rocrate.entities import (
    pccp as ent_pccp,
)
from apmode.bundle.rocrate.entities import (
    policy as ent_policy,
)
from apmode.bundle.rocrate.entities import (
    sbom as ent_sbom,
)
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)

# Sentinel filename duplicated here to avoid importing ``BundleEmitter``
# (which would pull in Lark + the whole DSL stack). Keep this in sync
# with ``apmode.bundle.emitter._COMPLETE_SENTINEL``.
_COMPLETE_SENTINEL = "_COMPLETE"


def _is_iso8601(value: str) -> bool:
    """Return True when ``value`` parses as an ISO-8601 date or datetime.

    ``datetime.fromisoformat`` accepts the modern (Python 3.11+) ISO-8601
    spectrum including the ``Z`` suffix. We treat anything it rejects as
    invalid so a malformed ``_COMPLETE.sealed_at`` cannot leak into
    ``datePublished`` and silently fail downstream validation.
    """
    try:
        # ``fromisoformat`` rejects bare dates without time on older
        # Python; we accept either by trying date too.
        _dt.datetime.fromisoformat(value)
        return True
    except ValueError:
        try:
            _dt.date.fromisoformat(value)
            return True
        except ValueError:
            return False


# Fixed epoch for ZIP timestamps to produce reproducible archives
# regardless of wall-clock time during export.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

_METADATA_FILENAME = "ro-crate-metadata.json"


class RoCrateProfile(StrEnum):
    """The three WRROC v0.5 profile URIs.

    Members are the profile identifiers recognised by ``roc-validator``
    (the ``profile_identifier`` setting). ``PROVENANCE`` is the default
    and declaratively conforms to ``provenance-run-crate-0.5``; the
    other two are inherited via WRROC's profile hierarchy.
    """

    PROVENANCE = "provenance-run-crate-0.5"
    WORKFLOW = "workflow-run-crate-0.5"
    PROCESS = "process-run-crate-0.5"


class ReportableSelection(StrEnum):
    """Policy for which candidates get a full per-candidate CreateAction.

    v0.6 always uses :attr:`ALL` because the Submission lane has a
    small candidate count. :attr:`GATE2_PASS` and :attr:`TOP_K` are
    recognised by the CLI for forward-compatibility with v0.7's
    Discovery-lane tiering — they currently behave like :attr:`ALL`.
    """

    ALL = "all-evaluated"
    GATE2_PASS = "gate2-pass"
    TOP_K = "top-k:50"


@dataclass
class RoCrateExportOptions:
    """Tunable knobs for :meth:`RoCrateEmitter.export_from_sealed_bundle`."""

    profile: RoCrateProfile = RoCrateProfile.PROVENANCE
    reportable: ReportableSelection = ReportableSelection.ALL
    include_provagent: bool = False
    regulatory_context: str | None = None
    date_published: str | None = None
    """ISO-8601 timestamp injected into the root Dataset. When ``None``
    the projector reads ``_COMPLETE.sealed_at`` (if present) so the
    stamp travels with the bundle across hosts. The ``_COMPLETE``
    mtime is only used as a legacy fallback for bundles sealed before
    ``sealed_at`` landed; callers that copy bundles between machines
    SHOULD pin this option or re-seal to avoid host-dependent output."""
    extra_conforms_to: list[str] = field(default_factory=list)


class BundleNotSealedError(RuntimeError):
    """Raised when exporting from a bundle that lacks ``_COMPLETE``."""


class RoCrateEmitter:
    """Project a sealed APMODE bundle into an RO-Crate.

    The emitter is stateless — every call to
    :meth:`export_from_sealed_bundle` performs a full projection from
    disk. Instances are cheap to create, so it is fine to construct one
    per export. Tests can override the default options by passing a
    fully-populated :class:`RoCrateExportOptions`.
    """

    def export_from_sealed_bundle(
        self,
        bundle_dir: Path,
        out: Path,
        options: RoCrateExportOptions | None = None,
    ) -> Path:
        """Project ``bundle_dir`` into an RO-Crate at ``out``.

        Args:
            bundle_dir: Source sealed bundle. Must exist, be a
                directory, and contain a ``_COMPLETE`` sentinel.
            out: Destination. Directory path for directory form, or a
                path ending in ``.zip`` for zip form.
            options: Export knobs. When ``None`` the defaults are used
                (provenance profile, all candidates, no PROV-AGENT).

        Returns:
            The ``Path`` of the output crate (the directory or the
            ``.zip`` file).

        Raises:
            BundleNotSealedError: if ``bundle_dir`` lacks ``_COMPLETE``.
            FileNotFoundError: if ``bundle_dir`` does not exist.
            NotADirectoryError: if ``bundle_dir`` is not a directory.
            FileExistsError: if ``out`` exists and is not empty
                (directory form) or already a file (zip form).
        """
        opts = options or RoCrateExportOptions()
        self._check_source(bundle_dir)
        metadata = self.build_metadata_json(bundle_dir, opts)

        if out.suffix.lower() == ".zip":
            return self._write_zip(bundle_dir, out, metadata)
        return self._write_directory(bundle_dir, out, metadata)

    # ----- metadata construction -----

    def build_metadata_json(
        self,
        bundle_dir: Path,
        options: RoCrateExportOptions,
    ) -> dict[str, Any]:
        """Construct the ``ro-crate-metadata.json`` payload.

        Pure function of the bundle + options. Does not perform any
        disk writes; callers use it when they want to inspect or
        round-trip the metadata without materialising a crate.
        """
        graph: list[dict[str, Any]] = []
        root_id = "./"

        # 1. Metadata descriptor + root Dataset
        graph.append(
            {
                "@id": _METADATA_FILENAME,
                "@type": "CreativeWork",
                "about": {"@id": root_id},
                "conformsTo": {"@id": ctx.ROCRATE_1_1},
            }
        )
        root = self._build_root_dataset(bundle_dir, options)
        graph.append(root)

        # 2. Always-present context entities
        lane = self._detect_lane(bundle_dir)
        workflow_id = self._workflow_id_for_lane(lane)
        organize_id = "#run-organize-action"
        run_timestamp = self._resolve_date_published(bundle_dir, options)
        self._add_core_entities(
            graph,
            bundle_dir,
            lane=lane,
            workflow_id=workflow_id,
            organize_id=organize_id,
            run_timestamp=run_timestamp,
        )
        root["mainEntity"] = {"@id": workflow_id}
        # Workflow file must be in root.hasPart so the inverse-hasPart
        # rule on ComputationalWorkflow finds a parent Dataset.
        merge_list_property(root, "hasPart", {"@id": workflow_id})
        root[vocab.LANE] = lane

        # 3. Project bundle artifacts
        data_id = ent_data.add_data_manifest(graph, bundle_dir, root_id)
        ent_data.add_split_manifest(graph, bundle_dir, root_id)
        ent_data.add_evidence_manifest(graph, bundle_dir, root_id)
        ent_data.add_seed_registry(graph, bundle_dir, root_id)
        ent_data.add_backend_versions(graph, bundle_dir, root_id)
        ent_data.add_initial_estimates(graph, bundle_dir, root_id)

        _, policy_lane = ent_policy.add_policy_file(graph, bundle_dir, root_id)
        if policy_lane:
            root[vocab.LANE] = policy_lane
        ent_policy.add_missing_data_directive(graph, bundle_dir, root_id)

        # 4. Candidate-level projection. Backend engines
        # (nlmixr2 / Stan / NODE) are the "orchestrated tools" that
        # satisfy the provenance-run-crate MUST
        # "ComputationalWorkflow MUST refer to orchestrated tools via
        # hasPart" — each engine becomes a SoftwareApplication in the
        # graph and is added to the workflow's ``hasPart`` by
        # ``ent_backend.add_backend_create_action`` when the first
        # candidate using that engine is projected. Candidate DSL
        # SoftwareApplications are inputs to these tools, not tools
        # themselves, so they are carried as ``CreateAction.object``
        # entries rather than registered on the workflow.
        #
        # The candidate CreateActions are NOT added to the
        # OrganizeAction.object list — ControlActions own that slot.
        for candidate_id in self._select_candidates(bundle_dir, options):
            ent_backend.add_backend_create_action(
                graph,
                bundle_dir,
                candidate_id,
                data_manifest_id=data_id,
                organize_action_id=None,  # keep OrganizeAction.object ControlAction-only
                workflow_id=workflow_id,
            )
            # Attach the backend engine tool to the workflow's hasPart
            # so ``ProvRCToolRequired`` (every instrument tool must be
            # hasPart of the workflow) is satisfied for every engine
            # that actually produced a result.
            self._register_engine_on_workflow(graph, workflow_id, bundle_dir, candidate_id)

        # 5. Gates (per-candidate ControlActions). Every ControlAction
        # is attached to the OrganizeAction.object list so the
        # "must reference ControlAction via object" rule is satisfied.
        for gate_path in ent_gate.iter_gate_decisions(bundle_dir):
            ent_gate.add_gate_control_action(
                graph,
                bundle_dir,
                gate_path,
                workflow_id=workflow_id,
                organize_action_id=organize_id,
            )

        # 5a. HowToStep workExample — each step MUST refer to its tool.
        # We use the orchestrator as the canonical tool for gate steps;
        # this is consistent with how APMODE evaluates gates (the
        # orchestrator dispatches rather than a per-gate binary).
        self._attach_howtostep_workexamples(graph)

        # 6. Lineage and search artifacts
        ent_backend.add_candidate_lineage(graph, bundle_dir)
        ent_lineage.add_candidate_lineage_derivations(graph, bundle_dir)
        ent_lineage.add_run_lineage(graph, bundle_dir, root_id)
        # Search CreateAction is NOT registered on OrganizeAction.object —
        # that slot is reserved for ControlActions per
        # provenance-run-crate MUST. The search artefacts remain linked
        # via the root Dataset hasPart + the CreateAction's own result list.
        ent_backend.add_search_artifacts(graph, bundle_dir, organize_action_id=None)
        ent_backend.add_ranking(graph, bundle_dir)

        # 7. Credibility / Bayesian / Agentic
        ent_credibility.add_credibility_reports(graph, bundle_dir)
        ent_credibility.add_report_provenance(graph, bundle_dir, root_id)
        ent_bayesian.add_bayesian_artifacts(graph, bundle_dir)
        # Agentic iterations are *orchestrator-side* actions (part of
        # the LLM search loop), not per-gate step executions, so they
        # are NOT added to OrganizeAction.object — that slot is
        # ControlAction-only per provenance-run-crate MUST.
        ent_agentic.add_agentic_trace(
            graph,
            bundle_dir,
            organize_action_id=None,
            include_provagent=options.include_provagent,
        )

        # 8. Regulatory / PCCP
        any_regulatory = ent_pccp.add_regulatory_artifacts(
            graph,
            bundle_dir,
            root_id=root_id,
            regulatory_context=options.regulatory_context,
        )
        if not any_regulatory and vocab.REGULATORY_CONTEXT not in root:
            root[vocab.REGULATORY_CONTEXT] = (
                options.regulatory_context or vocab.RegulatoryContext.RESEARCH_ONLY.value
            )

        # 9. _COMPLETE sentinel — integrity anchor
        self._add_complete_sentinel(graph, bundle_dir, root_id)

        # 9a. CycloneDX SBOM sidecar (optional; present when the bundle
        # was post-processed with ``apmode bundle sbom`` or dropped in
        # by CI). Excluded from the sealed digest so this is purely a
        # projector-side addition — no integrity implications.
        ent_sbom.add_sbom(graph, bundle_dir, root_id)

        # 10. Finalise — sort graph deterministically
        ordered_graph = self._order_graph(graph)
        return {
            "@context": ctx.build_rocrate_context(include_provagent=options.include_provagent),
            "@graph": ordered_graph,
        }

    # ----- root dataset / core entities -----

    def _build_root_dataset(
        self,
        bundle_dir: Path,
        options: RoCrateExportOptions,
    ) -> dict[str, Any]:
        run_id = self._detect_run_id(bundle_dir)
        name = f"APMODE run {run_id}"
        conforms_to: list[dict[str, str]] = [
            {"@id": ctx.WRROC_PROVENANCE_0_5},
            {"@id": ctx.WRROC_WORKFLOW_0_5},
            {"@id": ctx.WRROC_PROCESS_0_5},
            {"@id": ctx.WORKFLOW_RO_CRATE_1_0},
            {"@id": ctx.ROCRATE_1_1},
        ]
        for extra in options.extra_conforms_to:
            conforms_to.append({"@id": extra})
        root: dict[str, Any] = {
            "@id": "./",
            "@type": "Dataset",
            "name": name,
            "description": (
                f"APMODE reproducibility bundle projected as RO-Crate "
                f"(Workflow Run / Provenance Run Crate v0.5); run_id={run_id}"
            ),
            "datePublished": self._resolve_date_published(bundle_dir, options),
            "license": {"@id": ctx.GPL_2_OR_LATER},
            "conformsTo": conforms_to,
            "hasPart": [],
        }
        return root

    def _add_core_entities(
        self,
        graph: list[dict[str, Any]],
        bundle_dir: Path,
        *,
        lane: str,
        workflow_id: str,
        organize_id: str,
        run_timestamp: str,
    ) -> None:
        # DSL language entity
        upsert(
            graph,
            {
                "@id": "#apmode-dsl",
                "@type": ["ComputerLanguage", "SoftwareApplication"],
                "name": "APMODE PK DSL",
                "url": {"@id": "https://github.com/biostochastics/APMODE"},
                "version": self._apmode_version(bundle_dir),
            },
        )
        # Orchestrator SoftwareApplication
        upsert(
            graph,
            {
                "@id": "#apmode-orchestrator",
                "@type": "SoftwareApplication",
                "name": "APMODE orchestrator",
                "version": self._apmode_version(bundle_dir),
            },
        )
        # Workflow entity (virtual file; materialised during export).
        # ``hasPart`` must contain at least one SoftwareApplication per
        # provenance-run-crate MUST constraint "ComputationalWorkflow
        # MUST refer to orchestrated tools via hasPart". The
        # orchestrator is the workflow engine (OrganizeAction
        # instrument), not an orchestrated tool, so it is deliberately
        # excluded from this list — each orchestrated tool in
        # ``hasPart`` must also be referenced by a CreateAction /
        # ActivateAction / UpdateAction via ``instrument``
        # (ProvRCToolRequired). Candidate SoftwareApplications are
        # appended to ``hasPart`` as they are projected.
        workflow_entity: dict[str, Any] = {
            "@id": workflow_id,
            "@type": [
                "File",
                "SoftwareSourceCode",
                "ComputationalWorkflow",
                "HowTo",
            ],
            "name": f"APMODE {lane} Lane",
            "description": (
                f"Virtual workflow definition for the APMODE {lane} lane — "
                "declarative list of gate + backend steps."
            ),
            "programmingLanguage": {"@id": "#apmode-dsl"},
            "version": self._apmode_version(bundle_dir),
            "license": {"@id": ctx.GPL_2_OR_LATER},
            "encodingFormat": "application/json",
            "step": [],
            "hasPart": [],
        }
        upsert(graph, workflow_entity)

        # OrganizeAction wrapping the lane execution. Per
        # provenance-run-crate MUST ProvRCOrganizeActionRequired, its
        # ``object`` property must contain ControlAction instances (and
        # nothing else). Non-ControlAction children of the run (search
        # CreateAction, candidate CreateActions) are linked to
        # ControlActions via ``schema:object`` on the ControlAction.
        #
        # ``startTime`` / ``endTime`` satisfy the Process Run Crate
        # MUST-have temporal properties on the wrapping action + the
        # EU AI Act Article 12(a) "automatic recording of events over
        # the lifetime of the system" expectation. We don't currently
        # separate the run start from its seal time in the bundle, so
        # both slots share the sentinel timestamp — a conservative
        # lower bound on when the run must have completed.
        organize_entity: dict[str, Any] = {
            "@id": organize_id,
            "@type": "OrganizeAction",
            "name": f"Run of APMODE {lane} Lane",
            "instrument": {"@id": "#apmode-orchestrator"},
            "object": [],
            "result": {"@id": "#run-workflow-action"},
            "startTime": run_timestamp,
            "endTime": run_timestamp,
            "actionStatus": {"@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS},
        }
        upsert(graph, organize_entity)

        # CreateAction for the lane execution itself. ``instrument``
        # points at the workflow — orchestrator is already the
        # instrument of the wrapping OrganizeAction. Mirrors the
        # OrganizeAction's temporal extent so the retrospective
        # provenance graph carries consistent timings.
        upsert(
            graph,
            {
                "@id": "#run-workflow-action",
                "@type": "CreateAction",
                "name": f"{lane} lane workflow execution",
                "instrument": {"@id": workflow_id},
                "actionStatus": {"@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS},
                "startTime": run_timestamp,
                "endTime": run_timestamp,
            },
        )

        # Profile CreativeWork entities — required by
        # provenance-run-crate MUST Root Data Entity conformsTo rule
        # (must be a schema:CreativeWork whose @id matches the versioned
        # permalink). We add the three WRROC profiles + base RO-Crate 1.1.
        for profile_uri, profile_name, profile_version in (
            (ctx.WRROC_PROVENANCE_0_5, "Workflow Run Provenance RO-Crate Profile v0.5", "0.5"),
            (ctx.WRROC_WORKFLOW_0_5, "Workflow Run RO-Crate Profile v0.5", "0.5"),
            (ctx.WRROC_PROCESS_0_5, "Process Run RO-Crate Profile v0.5", "0.5"),
            (ctx.WORKFLOW_RO_CRATE_1_0, "Workflow RO-Crate Profile", "1.0"),
            (ctx.ROCRATE_1_1, "RO-Crate 1.1 Specification", "1.1"),
        ):
            upsert(
                graph,
                {
                    "@id": profile_uri,
                    "@type": "CreativeWork",
                    "name": profile_name,
                    "version": profile_version,
                },
            )

    # ----- candidate selection / lane detection -----

    def _select_candidates(
        self,
        bundle_dir: Path,
        options: RoCrateExportOptions,
    ) -> list[str]:
        """Return the list of candidate ids to project as CreateActions.

        v0.6 always returns every candidate with a result file. The
        other selections are recognised for forward-compat — v0.7 will
        cap Discovery-lane projections to a tiered subset.
        """
        ids = ent_backend.collect_result_ids(bundle_dir)
        if options.reportable == ReportableSelection.GATE2_PASS:
            return [cid for cid in ids if self._gate_passed(bundle_dir, cid, "2")]
        if options.reportable == ReportableSelection.TOP_K:
            return ids[:50]
        return ids

    def _register_engine_on_workflow(
        self,
        graph: list[dict[str, Any]],
        workflow_id: str,
        bundle_dir: Path,
        candidate_id: str,
    ) -> None:
        """Attach the backend engine SoftwareApplication to the workflow's hasPart.

        Each engine is added at most once per run (``merge_list_property``
        dedupes by ``@id``). The engine entity is upserted earlier by
        :func:`apmode.bundle.rocrate.entities.backend.add_backend_create_action`
        via ``_ensure_backend_engine``; this method just wires the
        ``hasPart`` edge, which satisfies the provenance-run-crate
        ``ProvRCToolRequired`` rule that every instrument tool must be
        orchestrated by the workflow.
        """
        result_path = bundle_dir / "results" / f"{candidate_id}_result.json"
        result_payload = load_json_optional(result_path) or {}
        backend_name = str(result_payload.get("backend", "unknown"))
        engine_id = ent_backend.engine_id_for_backend(backend_name)
        workflow = next((e for e in graph if e.get("@id") == workflow_id), None)
        if workflow is None:
            return
        merge_list_property(workflow, "hasPart", {"@id": engine_id})

    def _attach_howtostep_workexamples(self, graph: list[dict[str, Any]]) -> None:
        """Ensure every HowToStep has a ``workExample`` to a tool.

        provenance-run-crate MUST ProvRCHowToStepRequired demands this.
        Gate steps are executed by the orchestrator, so we point each
        HowToStep at ``#apmode-orchestrator`` — the SoftwareApplication
        that actually runs the gate evaluation.
        """
        for entity in graph:
            t = entity.get("@type")
            is_step = t == "HowToStep" or (isinstance(t, list) and "HowToStep" in t)
            if not is_step:
                continue
            if "workExample" not in entity:
                entity["workExample"] = {"@id": "#apmode-orchestrator"}

    def _gate_passed(self, bundle_dir: Path, candidate_id: str, gate_n: str) -> bool:
        path = bundle_dir / "gate_decisions" / f"gate{gate_n}_{candidate_id}.json"
        payload = load_json_optional(path) or {}
        return bool(payload.get("passed", False))

    def _detect_lane(self, bundle_dir: Path) -> str:
        policy_path = bundle_dir / "policy_file.json"
        payload = load_json_optional(policy_path) or {}
        raw_lane = payload.get("lane")
        if isinstance(raw_lane, str) and raw_lane.strip():
            lane = raw_lane.strip()
            return ent_policy._normalise_lane(lane)
        return "Submission"

    def _workflow_id_for_lane(self, lane: str) -> str:
        slug = lane.lower().replace(" ", "-")
        return f"workflows/{slug}-lane.apmode"

    def _detect_run_id(self, bundle_dir: Path) -> str:
        sentinel = load_json_optional(bundle_dir / _COMPLETE_SENTINEL) or {}
        run_id = sentinel.get("run_id")
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
        return bundle_dir.name

    def _resolve_date_published(
        self,
        bundle_dir: Path,
        options: RoCrateExportOptions,
    ) -> str:
        """Pick a deterministic ISO-8601 ``datePublished`` for the crate.

        Precedence (most deterministic first):

        1. Explicit ``options.date_published`` — always wins.
        2. ``_COMPLETE.sealed_at`` — an ISO-8601 timestamp stored inside
           the sentinel at seal time. Travels with the bundle, so it is
           identical across hosts and filesystems.
        3. ``_COMPLETE`` mtime (legacy fallback for bundles sealed before
           the ``sealed_at`` field landed). Not cross-host-stable, so
           callers that copy bundles between machines SHOULD pin
           ``date_published`` or re-seal. A comment on
           :class:`RoCrateExportOptions.date_published` records this
           caveat.
        4. The epoch (``1970-01-01T00:00:00Z``) as a last resort so we
           never emit a wall-clock timestamp that depends on when the
           export ran.
        """
        if options.date_published:
            return options.date_published
        sentinel_path = bundle_dir / _COMPLETE_SENTINEL
        payload = load_json_optional(sentinel_path) or {}
        sealed_at = payload.get("sealed_at")
        if isinstance(sealed_at, str) and sealed_at.strip():
            stripped = sealed_at.strip()
            # Validate the sealed_at value parses as ISO-8601 before
            # passing it through. A garbage string would propagate into
            # ``datePublished`` and only fail at validator time;
            # falling back to mtime here surfaces the bad sentinel
            # earlier and keeps the crate well-formed.
            if _is_iso8601(stripped):
                return stripped
        try:
            mtime = sentinel_path.stat().st_mtime
            return (
                _dt.datetime.fromtimestamp(mtime, tz=_dt.UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        except OSError:
            return "1970-01-01T00:00:00Z"

    def _apmode_version(self, bundle_dir: Path) -> str:
        versions = load_json_optional(bundle_dir / "backend_versions.json") or {}
        v = versions.get("apmode_version")
        if isinstance(v, str) and v:
            return v
        return "0.6.0"

    def _add_complete_sentinel(
        self,
        graph: list[dict[str, Any]],
        bundle_dir: Path,
        root_id: str,
    ) -> None:
        """Project ``_COMPLETE`` as a File entity with the bundle digest.

        Refuses to project a sentinel that is present-but-unparseable —
        such a file is either a partial seal (crashed mid-write) or
        deliberate tampering; in either case the crate would carry a
        File entity without an ``identifier`` and consumers could not
        verify the sealed digest externally. Better to fail fast.
        """
        path = bundle_dir / _COMPLETE_SENTINEL
        if not path.is_file():  # pragma: no cover — defended by _check_source
            msg = f"_COMPLETE sentinel missing at {path}"
            raise BundleNotSealedError(msg)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # ``UnicodeDecodeError`` is a ``ValueError``, not an
            # ``OSError`` — it would otherwise escape the handler and
            # surface as an opaque traceback. Catch it explicitly so
            # binary-corrupted sentinels produce the same diagnostic
            # path as JSON-parse failures.
            msg = (
                f"_COMPLETE sentinel at {path} is unparseable: {exc}. "
                "Bundle is not sealed; re-run the pipeline or restore the original "
                "bundle before exporting."
            )
            raise BundleNotSealedError(msg) from exc
        if not isinstance(payload, dict):
            kind = type(payload).__name__
            msg = f"_COMPLETE sentinel at {path} must be a JSON object; got {kind}"
            raise BundleNotSealedError(msg)

        digest = payload.get("sha256")
        description = "Bundle integrity sentinel (SHA-256 over all non-sentinel bundle files)"
        entity = file_entity(
            bundle_dir,
            path,
            name="Bundle integrity sentinel",
            extra={
                "additionalType": vocab.COMPLETE_SENTINEL_TYPE,
                "description": description,
            },
        )
        if isinstance(digest, str) and digest:
            entity["identifier"] = f"sha256:{digest}"
        upsert(graph, entity)
        root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})

    # ----- graph ordering -----

    _TYPE_ORDER: ClassVar[dict[str, int]] = {
        "CreativeWork": 0,
        "Dataset": 1,
        "ComputerLanguage": 2,
        "SoftwareApplication": 3,
        "ComputationalWorkflow": 4,
        "HowToStep": 5,
        "OrganizeAction": 6,
        "ControlAction": 7,
        "CreateAction": 8,
        "File": 9,
    }

    def _primary_type(self, entity: dict[str, Any]) -> str:
        t = entity.get("@type")
        if isinstance(t, list):
            for candidate in self._TYPE_ORDER:
                if candidate in t:
                    return candidate
            return str(t[0]) if t else ""
        return str(t or "")

    def _order_graph(self, graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def key(entity: dict[str, Any]) -> tuple[int, str]:
            # Keep metadata descriptor and root Dataset pinned to the top.
            eid = str(entity.get("@id", ""))
            if eid == _METADATA_FILENAME:
                return (-2, eid)
            if eid == "./":
                return (-1, eid)
            primary = self._primary_type(entity)
            return (self._TYPE_ORDER.get(primary, 100), eid)

        ordered = sorted(graph, key=key)
        # Also sort list properties by @id for determinism
        for entity in ordered:
            self._sort_nested_ref_lists(entity)
        return ordered

    def _sort_nested_ref_lists(self, entity: dict[str, Any]) -> None:
        for k, v in list(entity.items()):
            if not isinstance(v, list):
                continue
            if all(isinstance(item, dict) and "@id" in item for item in v):
                entity[k] = sorted(v, key=lambda item: str(item["@id"]))

    # ----- writers -----

    def _check_source(self, bundle_dir: Path) -> None:
        if not bundle_dir.exists():
            msg = f"bundle_dir not found: {bundle_dir}"
            raise FileNotFoundError(msg)
        if not bundle_dir.is_dir():
            msg = f"bundle_dir is not a directory: {bundle_dir}"
            raise NotADirectoryError(msg)
        if not (bundle_dir / _COMPLETE_SENTINEL).exists():
            msg = (
                f"bundle at {bundle_dir} is not sealed "
                f"(missing {_COMPLETE_SENTINEL!r}); refuse to export"
            )
            raise BundleNotSealedError(msg)

    def _write_directory(
        self,
        bundle_dir: Path,
        out: Path,
        metadata: dict[str, Any],
    ) -> Path:
        if out.exists() and out.is_file():
            msg = f"out path is a file: {out}"
            raise FileExistsError(msg)
        if out.is_dir() and any(out.iterdir()):
            msg = f"out directory already exists and is not empty: {out}"
            raise FileExistsError(msg)
        out.mkdir(parents=True, exist_ok=True)

        # Copy bundle contents without touching the source
        for src in sorted(bundle_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(bundle_dir)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Materialise the virtual workflow file referenced by mainEntity
        self._materialise_virtual_workflow(out, metadata)

        (out / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return out

    def _write_zip(
        self,
        bundle_dir: Path,
        out: Path,
        metadata: dict[str, Any],
    ) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.is_dir():
            msg = f"out zip path refers to an existing directory: {out}"
            raise FileExistsError(msg)

        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "crate"
            self._write_directory(bundle_dir, staging, metadata)
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(staging.rglob("*")):
                    if not p.is_file():
                        continue
                    rel = p.relative_to(staging).as_posix()
                    info = zipfile.ZipInfo(filename=rel, date_time=_ZIP_EPOCH)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with p.open("rb") as f:
                        zf.writestr(info, f.read())
        return out

    def _materialise_virtual_workflow(
        self,
        out_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Write a tiny workflow-definition JSON so the crate contains
        the ``File`` entity referenced by ``mainEntity``.

        RO-Crate's ``ComputationalWorkflow`` is a ``File`` subtype, so
        the ``@id`` must resolve to an actual artifact inside the
        crate. We materialise a declarative JSON listing the lane's
        ``HowToStep`` order — sufficient for validators, and cheap to
        reproduce.

        The stub is ALWAYS (re)written from the graph; the previous
        behaviour of short-circuiting on ``dst.exists()`` meant that a
        pre-existing lookalike file at the same path (carried over from
        the source bundle) would leave the ``ComputationalWorkflow``
        entity without ``sha256``/``contentSize``. Overwriting guarantees
        the graph hashes match what's on disk.
        """
        main_id: str | None = None
        for entity in metadata["@graph"]:
            if entity.get("@id") == "./":
                main = entity.get("mainEntity")
                if isinstance(main, dict):
                    main_id = main.get("@id")
                break
        if not isinstance(main_id, str):
            return

        dst = out_dir / main_id
        dst.parent.mkdir(parents=True, exist_ok=True)

        steps: list[str] = []
        workflow: dict[str, Any] | None = None
        for entity in metadata["@graph"]:
            if entity.get("@id") == main_id:
                workflow = entity
                break
        if workflow is not None:
            step = workflow.get("step")
            if isinstance(step, list):
                for s in step:
                    if isinstance(s, dict) and "@id" in s:
                        steps.append(str(s["@id"]))
        body = {
            "workflow_id": main_id,
            "description": (workflow.get("description") if workflow else "APMODE lane workflow"),
            "steps": steps,
        }
        dst.write_text(json.dumps(body, indent=2) + "\n")

        # Always recompute sha256 + contentSize so the graph stays in
        # lock-step with disk bytes, regardless of prior state.
        from apmode.bundle.rocrate.entities._common import _sha256_hex

        for entity in metadata["@graph"]:
            if entity.get("@id") == main_id:
                entity["contentSize"] = str(dst.stat().st_size)
                entity["sha256"] = _sha256_hex(dst)
                break

        # Rewrite ro-crate-metadata.json with the updated hashes so
        # validators see a consistent SHA-256 / size pair.
        (out_dir / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )


__all__ = [
    "BundleNotSealedError",
    "ReportableSelection",
    "RoCrateEmitter",
    "RoCrateExportOptions",
    "RoCrateProfile",
]
