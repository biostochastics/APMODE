# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``agentic_trace/`` iterations as CreateAction chains.

Each iteration has up to three files:

- ``<iteration_id>_input.json``  (redacted LLM input)
- ``<iteration_id>_output.json`` (verbatim LLM output)
- ``<iteration_id>_meta.json``   (cost + model metadata)

Each iteration becomes one ``CreateAction`` with ``@id =
#agentic-<iteration_id>``. Iterations are linked by
``prov:wasInformedBy`` in discovery order. When
``include_provagent=True`` the metadata File is additionally typed
as ``provagent:ModelInvocation`` — the v0.9 PROV-AGENT alignment
(Souza et al., arXiv:2508.02866).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apmode.bundle.rocrate import context as ctx
from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)

_TRACE_NAME_RE = re.compile(r"^(?P<iter>.+)_(?P<kind>input|output|meta)\.json$")


def add_agentic_trace(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    *,
    organize_action_id: str | None = None,
    include_provagent: bool = False,
) -> list[str]:
    """Project ``agentic_trace/*.json`` as a linked chain of CreateActions.

    Returns the list of CreateAction ``@id`` values added in iteration
    order.
    """
    d = bundle_dir / "agentic_trace"
    if not d.is_dir():
        return []

    by_iter: dict[str, dict[str, Path]] = {}
    for p in sorted(d.glob("*.json")):
        m = _TRACE_NAME_RE.match(p.name)
        if not m:
            continue
        by_iter.setdefault(m.group("iter"), {})[m.group("kind")] = p

    action_ids: list[str] = []
    previous_id: str | None = None
    for iteration_id in sorted(by_iter.keys()):
        files = by_iter[iteration_id]
        file_refs: list[dict[str, str]] = []
        output_payload = load_json_optional(files.get("output", Path("/dev/null"))) or {}
        meta_payload = load_json_optional(files.get("meta", Path("/dev/null"))) or {}

        for kind, p in sorted(files.items()):
            extra: dict[str, Any] = {
                "description": f"Agentic trace ({kind}) for iteration {iteration_id}",
            }
            if kind == "meta" and include_provagent:
                extra["additionalType"] = "provagent:ModelInvocation"
            entity = file_entity(
                bundle_dir,
                p,
                name=f"Agentic trace {kind} ({iteration_id})",
                extra=extra,
            )
            upsert(graph, entity)
            root = upsert(graph, {"@id": "./", "@type": "Dataset"})
            merge_list_property(root, "hasPart", {"@id": entity["@id"]})
            file_refs.append({"@id": str(entity["@id"])})

        action_id = f"#agentic-{iteration_id}"
        action: dict[str, Any] = {
            "@id": action_id,
            "@type": "CreateAction",
            "name": f"Agentic iteration {iteration_id}",
            "instrument": {"@id": "#apmode-orchestrator"},
            "result": file_refs,
            "actionStatus": {"@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS},
        }
        parsed = output_payload.get("parsed_transforms")
        if isinstance(parsed, list):
            # Normalise to plain strings for stability
            transforms = [str(t) for t in parsed if t]
            if transforms:
                action[vocab.DSL_TRANSFORM] = transforms
        # LLM agent — register as a separate SoftwareApplication entity
        # so the reference on the action is a bare ``@id`` node.
        # RO-Crate's flattened form rejects inline referenceable nodes
        # with extra properties ("entity is not a valid node object
        # reference: it MUST have only @id"), so splitting into a full
        # entity + ``@id``-only reference is required.
        if isinstance(meta_payload.get("model_id"), str):
            agent_id = f"#llm-{meta_payload['model_id']}"
            upsert(
                graph,
                {
                    "@id": agent_id,
                    "@type": "SoftwareApplication",
                    "name": meta_payload.get("model_id"),
                    "version": meta_payload.get("model_version", "unknown"),
                },
            )
            action["agent"] = {"@id": agent_id}
        if previous_id is not None:
            merge_list_property(action, ctx.PROV_WAS_INFORMED_BY, {"@id": previous_id})
        if include_provagent and files.get("meta") is not None:
            # `apmode:llmInvocation` → meta file
            meta_id = bundle_dir / "agentic_trace" / f"{iteration_id}_meta.json"
            action[vocab.LLM_INVOCATION] = {"@id": meta_id.relative_to(bundle_dir).as_posix()}
        upsert(graph, action)
        if organize_action_id:
            organize = upsert(graph, {"@id": organize_action_id, "@type": "OrganizeAction"})
            merge_list_property(organize, "object", {"@id": action_id})

        action_ids.append(action_id)
        previous_id = action_id

    return action_ids
