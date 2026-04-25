# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``agentic_trace/`` iterations as CreateAction chains.

Each iteration has up to three files:

- ``<iteration_id>_input.json``  (redacted LLM input)
- ``<iteration_id>_output.json`` (verbatim LLM output)
- ``<iteration_id>_meta.json``   (cost + model metadata)

Each iteration becomes one ``CreateAction`` with ``@id =
#agentic-<iteration_id>``. Iterations are linked by
``prov:wasInformedBy`` in **trace order** — sort priority is:

1. ``meta.sequence_number`` (integer; written by the agentic runner)
2. ``meta.started_at`` (ISO-8601 timestamp)
3. lexicographic iteration id (fallback for older bundles)

This matters when iteration ids do not sort lexicographically in
trace order (e.g. ``iter2`` vs ``iter10``, or retry ids like
``iter_retry_001`` interleaved with the main chain).

When ``include_provagent=True`` the metadata File is additionally
typed as ``provagent:AIModelInvocation`` — the canonical class name
from PROV-AGENT (Souza et al., eScience 2025, arXiv:2508.02866v3).
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


def _extract_sequence_number(meta_payload: dict[str, Any]) -> int | None:
    raw = meta_payload.get("sequence_number")
    if isinstance(raw, bool):  # bool is an int subclass; skip
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _extract_timestamp(meta_payload: dict[str, Any]) -> str | None:
    for key in ("started_at", "timestamp"):
        v = meta_payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _resolve_iteration_order(
    payloads: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Return iterations in trace order, using a single consistent signal.

    The sort key is chosen *globally* per bundle so we never mix two
    ordering signals in the same chain (which would interleave
    sequence-numbered iterations with timestamp-only ones in
    counter-intuitive ways):

    1. If **every** iteration carries a numeric ``sequence_number``,
       sort by it (with iteration id as deterministic tiebreaker).
    2. Else if **every** iteration carries a non-empty ``started_at`` /
       ``timestamp``, sort by that string with iteration id tiebreak.
    3. Else fall back to lexicographic iteration id — accepting that
       ``iter10`` will follow ``iter2`` only if the writer pads ids.

    Choosing one signal across the whole chain means a future bundle
    that adds ``sequence_number`` to *some* iterations cannot quietly
    rearrange the chain; the writer must populate every iteration to
    benefit from the stronger signal.
    """
    sequences = [_extract_sequence_number(meta) for _, meta in payloads]
    if all(s is not None for s in sequences):
        # mypy: every entry is int after the all() check
        return sorted(
            payloads,
            key=lambda pair: (_extract_sequence_number(pair[1]) or 0, pair[0]),
        )
    timestamps = [_extract_timestamp(meta) for _, meta in payloads]
    if all(t is not None for t in timestamps):
        return sorted(
            payloads,
            key=lambda pair: (_extract_timestamp(pair[1]) or "", pair[0]),
        )
    return sorted(payloads, key=lambda pair: pair[0])


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

    # Resolve ordering once so we can reuse meta payloads below.
    payloads: list[tuple[str, dict[str, Any]]] = []
    for iteration_id, files in by_iter.items():
        meta_payload = load_json_optional(files.get("meta", Path("/dev/null"))) or {}
        payloads.append((iteration_id, meta_payload))
    ordered = _resolve_iteration_order(payloads)

    action_ids: list[str] = []
    previous_id: str | None = None
    for iteration_id, meta_payload in ordered:
        files = by_iter[iteration_id]
        file_refs: list[dict[str, str]] = []
        output_payload = load_json_optional(files.get("output", Path("/dev/null"))) or {}

        for kind, p in sorted(files.items()):
            extra: dict[str, Any] = {
                "description": f"Agentic trace ({kind}) for iteration {iteration_id}",
            }
            if kind == "meta" and include_provagent:
                # PROV-AGENT v3 (eScience 2025) canonicalizes this
                # class name as ``AIModelInvocation``; earlier preprint
                # drafts used ``ModelInvocation``. Keep the typed value
                # in lock-step with the published paper.
                extra["additionalType"] = "provagent:AIModelInvocation"
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
