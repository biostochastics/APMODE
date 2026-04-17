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
) -> str | None:
    """Project ``results/<id>_result.json`` as a CreateAction + result File.

    Returns the ``@id`` of the CreateAction, or ``None`` when the
    result JSON is absent. Registers the candidate's
    :class:`SoftwareApplication` (via :func:`add_candidate_software_app`)
    if it does not exist yet, so callers may invoke this projector
    directly per-candidate without pre-seeding the graph.
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

    action_id = f"#backend-create-{candidate_id}"
    backend_name = result_payload.get("backend", "unknown")
    converged = bool(result_payload.get("converged", False))
    action: dict[str, Any] = {
        "@id": action_id,
        "@type": "CreateAction",
        "name": f"{backend_name} fit of {candidate_id}",
        "instrument": {"@id": app_id},
        "result": [{"@id": result_file["@id"]}],
        "actionStatus": _action_status_id(converged),
    }
    if data_manifest_id:
        action["object"] = [{"@id": data_manifest_id}]
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
