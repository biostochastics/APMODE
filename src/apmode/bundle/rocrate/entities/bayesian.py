# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``bayesian/`` artifacts as File entities.

Covers prior manifests, simulation protocols, MCMC diagnostics, and
posterior draws (parquet). Each is attached to the corresponding
candidate's ``CreateAction`` via ``result`` when the candidate's action
is in the graph.
"""

from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate.entities._common import (
    file_entity,
    merge_list_property,
    upsert,
)

# ``<candidate_id>_<suffix>.(json|parquet)``
#
# Covers the sealed-bundle Bayesian artifacts:
#   *_prior_manifest.json       — plan Task 15
#   *_simulation_protocol.json  — plan Task 23
#   *_mcmc_diagnostics.json     — plan Task 12
#   *_sampler_config.json       — plan Task 12
#   *_posterior_summary.parquet — plan Task 11
#   *_posterior_draws.parquet   — plan Task 10
#   *_draws.parquet             — legacy ``copy_posterior_draws`` sidecar
_BAYESIAN_NAME_RE = re.compile(
    r"^(?P<candidate>.+)_(?P<kind>"
    r"prior_manifest|simulation_protocol|mcmc_diagnostics|sampler_config|"
    r"posterior_summary|posterior_draws|draws"
    r")\.(?P<ext>json|parquet)$"
)


def add_bayesian_artifacts(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
) -> list[str]:
    """Project every ``bayesian/*.json|parquet`` file.

    Returns the list of File ``@id`` values added.
    """
    d = bundle_dir / "bayesian"
    if not d.is_dir():
        return []
    added: list[str] = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        m = _BAYESIAN_NAME_RE.match(p.name)
        if not m:
            continue
        candidate_id = m.group("candidate")
        kind = m.group("kind")
        ext = m.group("ext")
        label = kind.replace("_", " ").title()
        # Parquet does not have an IANA-registered media type yet
        # (Apache submission is pending). ``application/x-parquet`` is
        # the de-facto value used across Arrow tooling, Dremio, DuckDB
        # and the CNCF ParquetFormat project. Keep this until IANA
        # approves an official vendor tree assignment.
        encoding = "application/x-parquet" if ext == "parquet" else "application/json"
        entity = file_entity(
            bundle_dir,
            p,
            name=f"{label} — {candidate_id}",
            encoding_format=encoding,
            extra={"description": f"Bayesian artifact ({kind}) for {candidate_id}"},
        )
        upsert(graph, entity)
        root = upsert(graph, {"@id": "./", "@type": "Dataset"})
        merge_list_property(root, "hasPart", {"@id": entity["@id"]})

        # Attach to candidate CreateAction if present.
        action_id = f"#backend-create-{candidate_id}"
        for ent in graph:
            if ent.get("@id") == action_id:
                merge_list_property(ent, "result", {"@id": entity["@id"]})
                break
        added.append(str(entity["@id"]))
    return added
