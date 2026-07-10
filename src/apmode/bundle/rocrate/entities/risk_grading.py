# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``risk_grading/<id>.json`` (V&V40-style Gate 2.5 companion).

Mirrors :mod:`apmode.bundle.rocrate.entities.credibility` — each
per-candidate risk-grading report becomes a ``File`` referenced from
the candidate's ``CreateAction.result``.
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


def add_risk_grading_reports(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> list[str]:
    """Project ``risk_grading/<id>.json`` files as ``File`` entities.

    Same orphan-avoidance rule as :func:`add_credibility_reports`: only
    project when the candidate's ``#backend-create-<id>`` CreateAction
    already exists in the graph. Caller must project
    :mod:`apmode.bundle.rocrate.entities.backend` first.

    Returns the list of File ``@id``s added (may be empty).
    """
    d = bundle_dir / "risk_grading"
    if not d.is_dir():
        return []
    added: list[str] = []
    for p in sorted(d.glob("*.json")):
        candidate_id = p.stem
        action_id = f"#backend-create-{candidate_id}"
        candidate_action = next((e for e in graph if e.get("@id") == action_id), None)
        if candidate_action is None:
            continue

        payload = load_json_optional(p) or {}
        risk_tier = payload.get("risk_tier")
        description = f"Risk grading report for {candidate_id}" + (
            f" — tier={risk_tier}" if isinstance(risk_tier, str) else ""
        )
        entity = file_entity(
            bundle_dir,
            p,
            name=f"Risk grading report ({candidate_id})",
            extra={
                "description": description,
                "additionalType": vocab.RISK_GRADING_REPORT,
            },
        )
        upsert(graph, entity)
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})
        merge_list_property(candidate_action, "result", {"@id": entity["@id"]})
        added.append(str(entity["@id"]))
    return added
