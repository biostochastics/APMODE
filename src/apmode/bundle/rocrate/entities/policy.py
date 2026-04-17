# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``policy_file.json`` and ``missing_data_directive.json``."""

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


def add_policy_file(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> tuple[str | None, str | None]:
    """Project the lane's policy file.

    Returns ``(file_id, lane)`` — file_id is the ``@id`` of the File
    entity; lane is the human-readable lane name extracted from the
    policy payload (``"Submission"`` / ``"Discovery"`` /
    ``"Optimization"``) or ``None`` when the policy cannot be parsed.
    """
    path = bundle_dir / "policy_file.json"
    if not path.is_file():
        return None, None
    policy = load_json_optional(path)
    raw_lane = policy.get("lane") if policy else None
    lane = _normalise_lane(raw_lane) if isinstance(raw_lane, str) else None
    policy_version = policy.get("policy_version") if policy else None
    description_parts = ["APMODE gate policy"]
    if lane is not None:
        description_parts.append(lane)
    if policy_version:
        description_parts.append(f"v{policy_version}")
    entity = file_entity(
        bundle_dir,
        path,
        name="APMODE gate policy",
        extra={"description": " — ".join(description_parts)},
    )
    upsert(graph, entity)

    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    root[vocab.LANE_POLICY] = {"@id": entity["@id"]}
    if lane:
        root[vocab.LANE] = lane
    return str(entity["@id"]), lane


def add_missing_data_directive(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    path = bundle_dir / "missing_data_directive.json"
    if not path.is_file():
        return None
    directive = load_json_optional(path)
    cov_method = directive.get("covariate_method") if directive else None
    description = (
        f"Missing-data directive (covariate_method={cov_method})"
        if cov_method
        else "Missing-data directive"
    )
    entity = file_entity(
        bundle_dir,
        path,
        name="Missing-data directive",
        extra={"description": description},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def _normalise_lane(lane: str) -> str:
    """Map backend-canonical lane strings to display-cased names.

    Backends use ``"submission"``; the RO-Crate surface uses
    capitalised forms (``"Submission"``) to match the examples in plan
    §K and to align with WorkflowHub registration naming.
    """
    canon = lane.strip().lower()
    if canon == "submission":
        return "Submission"
    if canon == "discovery":
        return "Discovery"
    if canon == "optimization":
        return "Optimization"
    return lane
