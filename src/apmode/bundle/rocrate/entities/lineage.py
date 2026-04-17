# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``candidate_lineage.json`` + ``run_lineage.json`` as PROV derivation.

Lineage in APMODE is *search derivation* (an agent transform produces a
child spec from a parent spec) — not dataflow parameter wiring. We
therefore encode it with ``prov:wasDerivedFrom`` rather than
``wfrun:ParameterConnection`` (plan decision #7).
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate import context as ctx
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)


def _candidate_id(app_id_or_name: str) -> str:
    """Normalise either a ``#candidate-<id>`` app id or a bare id to app id."""
    if app_id_or_name.startswith("#candidate-"):
        return app_id_or_name
    return f"#candidate-{app_id_or_name}"


def add_candidate_lineage_derivations(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> int:
    """Project ``candidate_lineage.json`` as ``prov:wasDerivedFrom`` edges.

    Each entry ``(candidate_id, parent_id, transform)`` becomes a
    derivation edge on the child SoftwareApplication. Returns the
    number of edges added.

    The backing ``candidate_lineage.json`` File is also registered (via
    ``backend.add_candidate_lineage``) — we assume the orchestrator
    called that first so the file is already in the graph.
    """
    path = bundle_dir / "candidate_lineage.json"
    if not path.is_file():
        return 0
    payload = load_json_optional(path) or {}
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return 0
    n = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate_id = entry.get("candidate_id")
        parent_id = entry.get("parent_id")
        if not isinstance(candidate_id, str) or not isinstance(parent_id, str):
            continue
        child_app = upsert(
            graph,
            {"@id": _candidate_id(candidate_id), "@type": "SoftwareApplication"},
        )
        parent_ref = {"@id": _candidate_id(parent_id)}
        merge_list_property(child_app, ctx.PROV_WAS_DERIVED_FROM, parent_ref)
        transform = entry.get("transform")
        if isinstance(transform, str) and transform:
            merge_list_property(child_app, "apmode:dslTransform", transform)
        n += 1
    return n


def add_run_lineage(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    """Project ``run_lineage.json`` as File + prov link on root Dataset."""
    path = bundle_dir / "run_lineage.json"
    if not path.is_file():
        return None
    payload = load_json_optional(path) or {}
    entity = file_entity(
        bundle_dir,
        path,
        name="Run lineage",
        extra={"description": "Parent-run linkage for multi-run provenance"},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    parents = payload.get("parent_run_ids")
    if isinstance(parents, list):
        for parent in parents:
            if not isinstance(parent, str) or not parent:
                continue
            # Parents live in sibling directories; we encode them as
            # arcp-style blank references so consumers can correlate
            # without assuming a specific storage layout.
            ref = {"@id": f"#run-{parent}"}
            merge_list_property(root, ctx.PROV_WAS_DERIVED_FROM, ref)
    return str(entity["@id"])
