# SPDX-License-Identifier: GPL-2.0-or-later
"""Project candidate specs + backend results as SoftwareApplication + CreateAction.

Each candidate contributes three related entities:

1. A ``File`` (``SoftwareSourceCode``) for ``compiled_specs/<id>.json``.
2. A virtual ``SoftwareApplication`` (``#candidate-<id>``) that carries
   the ``apmode:dslSpec`` pointer to the File.
3. A ``CreateAction`` (``#backend-create-<id>``) describing the backend
   fit — ``instrument`` = candidate app; ``object`` = input data
   manifest; ``result`` = result JSON file.

The optional ``.R`` companion (nlmixr2 lowering) is emitted as a
separate ``File`` (``SoftwareSourceCode``) that the candidate's
``SoftwareApplication`` also references via ``hasPart``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmode.bundle.rocrate import context as ctx
from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    posix_id,
    upsert,
)


def _action_status_id(converged: bool | None) -> dict[str, str]:
    if converged is False:
        return {"@id": ctx.SCHEMA_FAILED_ACTION_STATUS}
    return {"@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS}


# Mapping from the ``backend`` field on ``<cid>_result.json`` to a stable
# SoftwareApplication ``@id`` inside the crate. New backends are added
# here so the engine entity appears once regardless of how many
# candidates were fit with it.
_BACKEND_ENGINE_IDS: dict[str, str] = {
    "nlmixr2": "#engine-nlmixr2",
    "bayesian_stan": "#engine-bayesian-stan",
    "node": "#engine-node",
    "agentic": "#engine-agentic",
}


def engine_id_for_backend(backend: str) -> str:
    """Return the canonical engine SoftwareApplication ``@id`` for ``backend``.

    Falls back to a slug-form id for unknown backends so forward-compat
    backends still appear in the graph rather than disappearing into the
    ``#apmode-orchestrator`` catch-all.
    """
    if backend in _BACKEND_ENGINE_IDS:
        return _BACKEND_ENGINE_IDS[backend]
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in backend.lower())
    return f"#engine-{slug}" if slug else "#engine-unknown"


def _ensure_backend_engine(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    backend: str,
) -> str:
    """Upsert a SoftwareApplication for the backend engine and return its @id.

    Pulls the engine version from ``backend_versions.json`` when
    available (e.g. ``nlmixr2_version``, ``stan_version``); leaves the
    ``softwareVersion`` off the entity when the bundle has no version
    record rather than fabricating an ``"unknown"`` string.
    """
    engine_id = engine_id_for_backend(backend)
    versions = load_json_optional(bundle_dir / "backend_versions.json") or {}
    version_keys = {
        "nlmixr2": ("nlmixr2_version", "rxode2_version"),
        "bayesian_stan": ("stan_version", "cmdstan_version", "cmdstanpy_version"),
        "node": ("diffrax_version", "jax_version"),
        "agentic": ("apmode_version",),
    }
    software_version: str | None = None
    for key in version_keys.get(backend, ()):
        v = versions.get(key)
        if isinstance(v, str) and v.strip():
            software_version = v.strip()
            break
    entity: dict[str, Any] = {
        "@id": engine_id,
        "@type": "SoftwareApplication",
        "name": _human_engine_name(backend),
    }
    if software_version is not None:
        entity["softwareVersion"] = software_version
    upsert(graph, entity)
    return engine_id


def _human_engine_name(backend: str) -> str:
    """Human-friendly display name for the backend engine entity."""
    return {
        "nlmixr2": "nlmixr2 (R)",
        "bayesian_stan": "Stan / CmdStan (Bayesian backend)",
        "node": "Diffrax / JAX (Neural ODE backend)",
        "agentic": "APMODE agentic LLM backend",
    }.get(backend, backend)


def backend_step_id(backend: str) -> str:
    """Stable ``@id`` of the ``HowToStep`` representing a ``backend`` fit."""
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in backend.lower())
    return f"#step-backend-{slug}" if slug else "#step-backend-unknown"


def add_backend_howto_step(
    graph: list[dict[str, Any]],
    backend: str,
    *,
    workflow_id: str | None = None,
    engine_id: str | None = None,
) -> str:
    """Ensure a HowToStep exists for ``backend`` and register it on the workflow.

    The HowToStep's ``workExample`` points at the backend engine
    SoftwareApplication (e.g. nlmixr2), not at the orchestrator, so the
    provenance-run-crate semantic that "steps point to the tools that
    execute them" is satisfied with a real engine rather than the
    dispatch wrapper.
    """
    step_id = backend_step_id(backend)
    step: dict[str, Any] = {
        "@id": step_id,
        "@type": "HowToStep",
        "name": f"{_human_engine_name(backend)} fit",
    }
    if engine_id:
        step["workExample"] = {"@id": engine_id}
    upsert(graph, step)

    if workflow_id:
        workflow = upsert(
            graph,
            {
                "@id": workflow_id,
                "@type": [
                    "File",
                    "SoftwareSourceCode",
                    "ComputationalWorkflow",
                    "HowTo",
                ],
            },
        )
        merge_list_property(workflow, "step", {"@id": step_id})
    return step_id


def add_candidate_software_app(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    candidate_id: str,
) -> str:
    """Project the candidate's compiled spec as File + SoftwareApplication.

    Returns the ``@id`` of the SoftwareApplication entity
    (``#candidate-<id>``). Safe to call repeatedly for the same
    candidate — it dedups on ``@id``.
    """
    spec_path = bundle_dir / "compiled_specs" / f"{candidate_id}.json"
    spec_id: str | None = None
    name = f"Candidate {candidate_id}"
    if spec_path.is_file():
        spec_entity = file_entity(
            bundle_dir,
            spec_path,
            name=f"Candidate {candidate_id} DSL spec",
            extra={
                "@type": ["File", "SoftwareSourceCode"],
                "description": "Typed APMODE DSL specification (JSON)",
                "programmingLanguage": {"@id": "#apmode-dsl"},
            },
        )
        # ``file_entity`` defaults ``@type`` to ``"File"``; our extra
        # overrides it to the list form — keep it that way.
        upsert(graph, spec_entity)
        spec_id = str(spec_entity["@id"])

        spec_payload = load_json_optional(spec_path) or {}
        route = spec_payload.get("absorption", {})
        if isinstance(route, dict) and "type" in route:
            name = f"Candidate {candidate_id} ({route['type']})"

    r_path = bundle_dir / "compiled_specs" / f"{candidate_id}.R"
    r_id: str | None = None
    if r_path.is_file():
        r_entity = file_entity(
            bundle_dir,
            r_path,
            name=f"Candidate {candidate_id} nlmixr2 R code",
            encoding_format="text/x-r-source",
            extra={
                "@type": ["File", "SoftwareSourceCode"],
                "description": "nlmixr2 lowering of the DSL spec",
                "programmingLanguage": "R",
            },
        )
        upsert(graph, r_entity)
        r_id = str(r_entity["@id"])
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": r_id})

    if spec_id is not None:
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": spec_id})

    app_id = f"#candidate-{candidate_id}"
    app_entity: dict[str, Any] = {
        "@id": app_id,
        "@type": "SoftwareApplication",
        "name": name,
        "version": candidate_id,
    }
    if spec_id is not None:
        app_entity[vocab.DSL_SPEC] = {"@id": spec_id}
    if r_id is not None:
        app_entity["hasPart"] = [{"@id": r_id}]
    upsert(graph, app_entity)
    return app_id


def add_backend_create_action(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    candidate_id: str,
    *,
    data_manifest_id: str | None,
    organize_action_id: str | None = None,
    workflow_id: str | None = None,
) -> str | None:
    """Project ``results/<id>_result.json`` as a CreateAction + result File.

    Returns the ``@id`` of the CreateAction, or ``None`` when the
    result JSON is absent. Registers the candidate's
    :class:`SoftwareApplication` (via :func:`add_candidate_software_app`)
    and the backend engine's :class:`SoftwareApplication` if they do
    not exist yet, so callers may invoke this projector directly
    per-candidate without pre-seeding the graph.

    Per provenance-run-crate v0.5, ``CreateAction.instrument`` names
    the tool being invoked. For a PK fit that is the backend engine
    (nlmixr2 / Stan / NODE), not the candidate DSL spec. The candidate
    SoftwareApplication is carried as one of the ``object`` inputs
    alongside the data manifest, which matches the semantics that the
    engine fits a model to data. ``workflow_id`` (when provided) is
    used to register the matching HowToStep on the workflow so the
    step/action chain is complete.
    """
    result_path = bundle_dir / "results" / f"{candidate_id}_result.json"
    if not result_path.is_file():
        return None
    result_payload = load_json_optional(result_path) or {}

    app_id = add_candidate_software_app(graph, bundle_dir, candidate_id)

    result_file = file_entity(
        bundle_dir,
        result_path,
        name=f"Backend result for {candidate_id}",
        extra={"description": f"{result_payload.get('backend', 'backend')} fit result"},
    )
    upsert(graph, result_file)
    root = upsert(graph, {"@id": "./", "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": result_file["@id"]})

    backend_name = str(result_payload.get("backend", "unknown"))
    engine_id = _ensure_backend_engine(graph, bundle_dir, backend_name)
    add_backend_howto_step(graph, backend_name, workflow_id=workflow_id, engine_id=engine_id)

    action_id = f"#backend-create-{candidate_id}"
    converged = bool(result_payload.get("converged", False))
    action: dict[str, Any] = {
        "@id": action_id,
        "@type": "CreateAction",
        "name": f"{backend_name} fit of {candidate_id}",
        "instrument": {"@id": engine_id},
        "result": [{"@id": result_file["@id"]}],
        "actionStatus": _action_status_id(converged),
    }
    # Objects are the "inputs" to the action: the data manifest plus
    # the candidate model specification (projected as a
    # SoftwareApplication carrying the DSL spec File via
    # ``apmode:dslSpec``).
    action_objects: list[dict[str, str]] = []
    if data_manifest_id:
        action_objects.append({"@id": data_manifest_id})
    action_objects.append({"@id": app_id})
    action["object"] = action_objects
    upsert(graph, action)

    # Register the CreateAction as an object of the OrganizeAction
    if organize_action_id:
        organize = upsert(graph, {"@id": organize_action_id, "@type": "OrganizeAction"})
        merge_list_property(organize, "object", {"@id": action_id})

    return action_id


def add_ranking(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> str | None:
    path = bundle_dir / "ranking.json"
    if not path.is_file():
        return None
    ranking_payload = load_json_optional(path)
    best = ranking_payload.get("best_candidate_id") if ranking_payload else None
    description = f"Gate 3 ranking (best = {best})" if best else "Gate 3 ranking"
    entity = file_entity(
        bundle_dir,
        path,
        name="Gate 3 ranking",
        extra={
            "@type": ["File", "CreativeWork"],
            "description": description,
        },
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": "./", "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_search_artifacts(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    *,
    organize_action_id: str | None = None,
) -> dict[str, str]:
    """Project search_trajectory.jsonl, failed_candidates.jsonl, search_graph.json.

    When ``organize_action_id`` is provided, the aggregate search ``CreateAction``
    (``#search-create``) is registered as one of its ``object`` entries.

    Returns a dict ``{kind: file_id}`` for the artifacts that were present.
    """
    out: dict[str, str] = {}
    paths = {
        "search_trajectory": bundle_dir / "search_trajectory.jsonl",
        "failed_candidates": bundle_dir / "failed_candidates.jsonl",
        "search_graph": bundle_dir / "search_graph.json",
    }
    result_refs: list[dict[str, str]] = []
    for kind, p in paths.items():
        if not p.is_file():
            continue
        enc = "application/x-ndjson" if p.suffix == ".jsonl" else "application/json"
        entity = file_entity(
            bundle_dir,
            p,
            name=kind.replace("_", " ").title(),
            encoding_format=enc,
            extra={"description": f"APMODE {kind.replace('_', ' ')} artifact"},
        )
        upsert(graph, entity)
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})
        out[kind] = str(entity["@id"])
        result_refs.append({"@id": str(entity["@id"])})

    if result_refs:
        search_action: dict[str, Any] = {
            "@id": "#search-create",
            "@type": "CreateAction",
            "name": "APMODE structural search (aggregate)",
            "instrument": {"@id": "#apmode-orchestrator"},
            "result": result_refs,
            "actionStatus": {"@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS},
        }
        if "search_graph" in out:
            search_action[vocab.SEARCH_GRAPH] = {"@id": out["search_graph"]}
        upsert(graph, search_action)

        if organize_action_id:
            organize = upsert(graph, {"@id": organize_action_id, "@type": "OrganizeAction"})
            merge_list_property(organize, "object", {"@id": "#search-create"})

    # Expose search_graph pointer on root dataset for tool discovery.
    if "search_graph" in out:
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        root[vocab.SEARCH_GRAPH] = {"@id": out["search_graph"]}

    return out


def add_candidate_lineage(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> str | None:
    path = bundle_dir / "candidate_lineage.json"
    if not path.is_file():
        return None
    entity = file_entity(
        bundle_dir,
        path,
        name="Candidate lineage (DAG)",
        extra={"description": "APMODE candidate derivation DAG"},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": "./", "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def collect_candidate_ids(bundle_dir: Path) -> list[str]:
    """Discover candidate ids from the compiled_specs/ directory."""
    specs_dir = bundle_dir / "compiled_specs"
    if not specs_dir.is_dir():
        return []
    ids = sorted(p.stem for p in specs_dir.glob("*.json"))
    return ids


def collect_result_ids(bundle_dir: Path) -> list[str]:
    """Discover candidate ids that actually have a backend result on disk."""
    results_dir = bundle_dir / "results"
    if not results_dir.is_dir():
        return []
    ids: list[str] = []
    for p in sorted(results_dir.glob("*_result.json")):
        stem = p.stem
        if stem.endswith("_result"):
            ids.append(stem[: -len("_result")])
    return ids


def candidate_id_from_gate_path(bundle_dir: Path, gate_path: Path) -> str:
    """Extract the candidate id portion of a ``gate{n}_<id>.json`` path."""
    rel = posix_id(bundle_dir, gate_path)
    stem = Path(rel).stem  # gate1_<id>
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem
