# SPDX-License-Identifier: GPL-2.0-or-later
"""Project the ``regulatory/`` subdirectory (FDA PCCP + EU AI Act).

In v0.6, ``regulatory/`` is an optional convention: operators who are
preparing a submission for FDA PCCP or EU AI Act Article 12 populate
the subdirectory with four files before sealing the bundle:

- ``regulatory/md.json``         (Modification Description)
- ``regulatory/mp.json``         (Modification Protocol)
- ``regulatory/ia.json``         (Impact Assessment)
- ``regulatory/traceability.csv`` (Traceability Table — CSV or JSON)

This projector:

1. Adds each file as a ``File`` entity under the root ``Dataset.hasPart``.
2. Sets the corresponding ``apmode:*`` properties on the root Dataset.
3. Sets ``apmode:regulatoryContext`` to ``"pccp-ai-dsf"`` when *any*
   regulatory file is present; otherwise leaves the root value alone
   (the orchestrator applies ``"research-only"`` as the default).

When none of the files exist, the projector returns early without
touching the graph.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate import vocab
from apmode.bundle.rocrate.entities._common import (
    file_entity,
    merge_list_property,
    upsert,
)

_REGULATORY_FILES: dict[str, tuple[str, str, str]] = {
    # key -> (filename, apmode_term, display_name)
    "md": ("md.json", vocab.MODIFICATION_DESCRIPTION, "Modification Description"),
    "mp": ("mp.json", vocab.MODIFICATION_PROTOCOL, "Modification Protocol"),
    "ia": ("ia.json", vocab.IMPACT_ASSESSMENT, "Impact Assessment"),
}


def add_regulatory_artifacts(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    *,
    root_id: str = "./",
    regulatory_context: str | None = None,
) -> bool:
    """Project ``regulatory/`` artifacts onto the root Dataset.

    Args:
        regulatory_context: Optional explicit context string — MUST be
            one of :class:`apmode.bundle.rocrate.vocab.RegulatoryContext`
            when regulatory files are present. Earlier versions defaulted
            to ``pccp-ai-dsf`` whenever any file existed, which silently
            mislabeled AI-Act-only or MDR-only bundles. We now refuse
            the implicit default: operators must pass
            ``--regulatory-context`` explicitly via
            :mod:`apmode.bundle.rocrate.cli_hooks`, or the projector
            leaves the context slot untouched and the orchestrator will
            fill in ``research-only`` (non-regulated default).

    Returns True when at least one regulatory file was projected.
    """
    reg_dir = bundle_dir / "regulatory"
    if not reg_dir.is_dir():
        return False

    any_added = False
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})

    for key, (filename, term, display_name) in _REGULATORY_FILES.items():
        p = reg_dir / filename
        if not p.is_file():
            continue
        entity = file_entity(
            bundle_dir,
            p,
            name=f"{display_name} ({key.upper()})",
            extra={"description": f"Regulatory artifact ({display_name})"},
        )
        upsert(graph, entity)
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})
        root[term] = {"@id": entity["@id"]}
        any_added = True

    # Traceability table: prefer CSV, fall back to JSON.
    for traceability_name in ("traceability.csv", "traceability.json"):
        p = reg_dir / traceability_name
        if not p.is_file():
            continue
        encoding = "text/csv" if traceability_name.endswith(".csv") else "application/json"
        entity = file_entity(
            bundle_dir,
            p,
            name="Traceability Table",
            encoding_format=encoding,
            extra={"description": "Regulatory traceability crosswalk"},
        )
        upsert(graph, entity)
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})
        root[vocab.TRACEABILITY_TABLE] = {"@id": entity["@id"]}
        any_added = True
        break

    if any_added and regulatory_context is not None:
        # Only set the context when the operator supplied one —
        # otherwise the projector leaves the slot alone so the caller
        # (orchestrator) can apply the ``research-only`` default.
        root[vocab.REGULATORY_CONTEXT] = regulatory_context
    return any_added
