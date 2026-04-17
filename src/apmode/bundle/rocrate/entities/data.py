# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``data_manifest.json`` / ``split_manifest.json`` onto File entities.

The input dataset file itself is not copied into the bundle by the
APMODE emitter — it remains at whatever path the user ran against, and
its SHA-256 is captured in ``data_manifest.data_sha256``. Consumers that
need the raw data must look it up externally by hash. That is
consistent with plan §E ("the RO-Crate graph carries the connective
tissue... detailed facts remain in referenced JSON/JSONL files").
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any

from apmode.bundle.rocrate.entities._common import (
    file_entity,
    load_json_optional,
    merge_list_property,
    upsert,
)


def add_data_manifest(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    """Project ``data_manifest.json`` as a File entity + hasPart link.

    Returns the ``@id`` of the File entity, or ``None`` if the artifact
    is absent.
    """
    path = bundle_dir / "data_manifest.json"
    if not path.is_file():
        return None
    manifest = load_json_optional(path)
    n_subjects = manifest.get("n_subjects") if manifest else None
    n_observations = manifest.get("n_observations") if manifest else None
    description_parts = ["APMODE data manifest"]
    if n_subjects is not None:
        description_parts.append(f"{n_subjects} subjects")
    if n_observations is not None:
        description_parts.append(f"{n_observations} observations")
    description = "; ".join(description_parts)

    data_sha = None
    if manifest and isinstance(manifest.get("data_sha256"), str):
        data_sha = manifest["data_sha256"]

    extra: dict[str, Any] = {"description": description}
    # Carry the source-CSV SHA so downstream consumers can match against
    # their local copy without opening the manifest.
    if data_sha:
        extra["identifier"] = f"sha256:{data_sha}"
    entity = file_entity(
        bundle_dir,
        path,
        name="Data manifest (APMODE)",
        extra=extra,
    )
    upsert(graph, entity)

    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_split_manifest(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    """Project ``split_manifest.json`` as a File entity + hasPart link."""
    path = bundle_dir / "split_manifest.json"
    if not path.is_file():
        return None
    manifest = load_json_optional(path)
    strategy = None
    if manifest and isinstance(manifest.get("split_strategy"), str):
        strategy = manifest["split_strategy"]
    description = f"APMODE split manifest ({strategy})" if strategy else "APMODE split manifest"
    entity = file_entity(
        bundle_dir,
        path,
        name="Split manifest (APMODE)",
        extra={"description": description, "additionalType": "apmode:split"},
    )
    upsert(graph, entity)

    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_evidence_manifest(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    """Project ``evidence_manifest.json`` as a File entity + hasPart link."""
    path = bundle_dir / "evidence_manifest.json"
    if not path.is_file():
        return None
    manifest = load_json_optional(path)
    richness = manifest.get("richness_category") if manifest else None
    description = (
        f"APMODE evidence manifest (richness={richness})"
        if richness
        else "APMODE evidence manifest"
    )
    entity = file_entity(
        bundle_dir,
        path,
        name="Evidence manifest (APMODE profiler)",
        extra={"description": description, "additionalType": "apmode:evidenceClass"},
    )
    upsert(graph, entity)

    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_seed_registry(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    path = bundle_dir / "seed_registry.json"
    if not path.is_file():
        return None
    entity = file_entity(
        bundle_dir,
        path,
        name="Seed registry",
        extra={"description": "Random seeds used across backends for reproducibility"},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_backend_versions(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    path = bundle_dir / "backend_versions.json"
    if not path.is_file():
        return None
    entity = file_entity(
        bundle_dir,
        path,
        name="Backend versions",
        extra={"description": "Software versions and container digests"},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


def add_initial_estimates(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    path = bundle_dir / "initial_estimates.json"
    if not path.is_file():
        return None
    entity = file_entity(
        bundle_dir,
        path,
        name="Initial estimates",
        extra={"description": "Per-candidate initial parameter estimates with provenance"},
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])
