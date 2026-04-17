# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``gate_decisions/gate{n}_{id}.json`` as WRROC ControlActions.

Gates are control steps — instrumented by ``HowToStep`` entities on
the lane's ``ComputationalWorkflow`` / ``HowTo``. Each per-candidate
decision becomes a ``ControlAction`` whose:

- ``instrument`` → the gate's ``HowToStep``
- ``object`` → the backend-fit ``CreateAction`` being gated
- ``result`` → the gate-decision ``File`` entity (JSON)
- ``apmode:gate`` → the canonical gate name (``gate1`` / ``gate2`` /
  ``gate2.5`` / ``gate3``)
- ``apmode:gateRationale`` → same File ``@id`` as ``result``
"""

from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate import context as ctx
from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)

# gate{n}_<candidate_id>.json  where n ∈ {1, 2, 2.5, 3}
# Regex permits the ``2.5`` fractional identifier while still matching
# integer gates cleanly.
_GATE_FILENAME_RE = re.compile(r"^gate(?P<n>2\.5|\d+)_(?P<candidate_id>.+)\.json$")

_GATE_LABELS: dict[str, str] = {
    "1": "Gate 1: Technical Validity",
    "2": "Gate 2: Lane Admissibility",
    "2.5": "Gate 2.5: Credibility",
    "3": "Gate 3: Lane Ranking",
}


def gate_step_id(gate_n: str) -> str:
    """Stable ``@id`` for the ``HowToStep`` representing gate n."""
    safe = gate_n.replace(".", "-")
    return f"#step-gate{safe}"


def _gate_action_id(gate_n: str, candidate_id: str) -> str:
    safe = gate_n.replace(".", "-")
    return f"#gate{safe}-control-{candidate_id}"


def parse_gate_filename(filename: str) -> tuple[str, str] | None:
    """Return ``(gate_n, candidate_id)`` or None if the filename does not match."""
    m = _GATE_FILENAME_RE.match(filename)
    if not m:
        return None
    return m.group("n"), m.group("candidate_id")


def add_gate_howto_step(
    graph: list[dict[str, Any]],
    gate_n: str,
    *,
    workflow_id: str | None = None,
) -> str:
    """Ensure the ``HowToStep`` for ``gate_n`` exists in the graph."""
    step_id = gate_step_id(gate_n)
    label = _GATE_LABELS.get(gate_n, f"Gate {gate_n}")
    step: dict[str, Any] = {
        "@id": step_id,
        "@type": "HowToStep",
        "name": label,
    }
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


def add_gate_control_action(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    gate_path: Path,
    *,
    workflow_id: str | None = None,
    organize_action_id: str | None = None,
) -> str | None:
    """Project a single gate-decision JSON onto ``ControlAction`` + File.

    Returns the ``@id`` of the ControlAction or ``None`` when the
    filename is malformed.
    """
    parsed = parse_gate_filename(gate_path.name)
    if parsed is None:
        return None
    gate_n, candidate_id = parsed

    decision = load_json_optional(gate_path) or {}
    passed = bool(decision.get("passed", False))
    summary = decision.get("summary_reason") or decision.get("gate_name")

    gate_file = file_entity(
        bundle_dir,
        gate_path,
        name=f"Gate {gate_n} decision for {candidate_id}",
        extra={
            "description": summary
            if isinstance(summary, str)
            else f"Gate {gate_n} decision record",
        },
    )
    upsert(graph, gate_file)
    root = upsert(graph, {"@id": "./", "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": gate_file["@id"]})

    step_id = add_gate_howto_step(graph, gate_n, workflow_id=workflow_id)

    action_id = _gate_action_id(gate_n, candidate_id)
    action: dict[str, Any] = {
        "@id": action_id,
        "@type": "ControlAction",
        "name": f"Gate {gate_n} for {candidate_id}",
        "instrument": {"@id": step_id},
        "object": {"@id": f"#backend-create-{candidate_id}"},
        "result": {"@id": gate_file["@id"]},
        "actionStatus": {
            "@id": ctx.SCHEMA_COMPLETED_ACTION_STATUS
            if passed
            else ctx.SCHEMA_FAILED_ACTION_STATUS
        },
        vocab.GATE: _canonical_gate_name(gate_n),
        vocab.GATE_RATIONALE: {"@id": gate_file["@id"]},
    }
    upsert(graph, action)

    if organize_action_id:
        organize = upsert(graph, {"@id": organize_action_id, "@type": "OrganizeAction"})
        merge_list_property(organize, "object", {"@id": action_id})

    return action_id


def _canonical_gate_name(gate_n: str) -> str:
    if gate_n == "2.5":
        return "gate2.5"
    return f"gate{gate_n}"


def iter_gate_decisions(bundle_dir: Path) -> list[Path]:
    """Return sorted list of gate-decision JSON paths under ``gate_decisions/``."""
    d = bundle_dir / "gate_decisions"
    if not d.is_dir():
        return []
    return sorted(d.glob("gate*_*.json"))
