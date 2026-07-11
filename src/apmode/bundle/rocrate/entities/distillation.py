# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``distillation/<id>.json`` files as RO-Crate ``File`` entities.

Each per-candidate functional-distillation report becomes a ``File`` referenced
from the candidate's ``CreateAction.result`` via
:func:`apmode.bundle.rocrate.entities.backend.add_backend_create_action` — this
module only adds the File entities and the cross-reference. A distillation report
is only projected when its ``#backend-create-<id>`` CreateAction is already in the
graph, so the crate never carries an orphan File (which fails roc-validator at
REQUIRED). The projector must run this after
:mod:`apmode.bundle.rocrate.entities.backend`.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)


def add_distillation_reports(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> list[str]:
    """Project ``distillation/<id>.json`` files as ``File`` entities.

    Returns the list of File ``@id``s added (may be empty).
    """
    d = bundle_dir / "distillation"
    if not d.is_dir():
        return []
    added: list[str] = []
    for p in sorted(d.glob("*.json")):
        candidate_id = p.stem

        action_id = f"#backend-create-{candidate_id}"
        candidate_action = next((e for e in graph if e.get("@id") == action_id), None)
        if candidate_action is None:
            # No backend result for this candidate — skip to avoid an orphan
            # File that would fail roc-validator at REQUIRED.
            continue

        payload = load_json_optional(p) or {}
        surrogate = payload.get("surrogate")
        surrogate_type = surrogate.get("surrogate_type") if isinstance(surrogate, dict) else None
        description = f"NODE functional-distillation surrogate for {candidate_id}" + (
            f" ({surrogate_type})" if isinstance(surrogate_type, str) else ""
        )
        entity = file_entity(
            bundle_dir,
            p,
            name=f"Distillation report ({candidate_id})",
            extra={
                "description": description,
                "additionalType": vocab.DISTILLATION_REPORT,
                "encodingFormat": "application/json",
            },
        )
        upsert(graph, entity)
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})
        merge_list_property(candidate_action, "result", {"@id": entity["@id"]})
        added.append(str(entity["@id"]))
    return added
