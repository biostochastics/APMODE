# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared helpers for entity projectors.

The helpers here are intentionally tiny and side-effect-free so they
can be unit-tested in isolation and composed from any projector. They
keep the projector modules terse and ensure that the ``@id`` scheme,
graph-dedup semantics, and File-entity construction remain consistent
across all of them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path  # noqa: TC003 — runtime type in function signatures
from typing import Any


def _sha256_hex(path: Path) -> str:
    """Return hex SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def posix_id(bundle_dir: Path, file_path: Path) -> str:
    """Return the canonical ``@id`` for a file entity.

    RO-Crate uses bundle-relative POSIX paths as ``@id`` for data entities
    contained in the crate. This helper is the only place we perform
    that conversion so the ``@id`` scheme stays consistent across
    projectors. ``file_path`` must live under ``bundle_dir``.
    """
    return file_path.relative_to(bundle_dir).as_posix()


def upsert(graph: list[dict[str, Any]], entity: dict[str, Any]) -> dict[str, Any]:
    """Append ``entity`` to ``graph`` (dedup by ``@id``).

    If an entity with the same ``@id`` already exists, its fields are
    *merged* into the existing record — new keys are added, existing
    keys from the caller take precedence on conflict. This lets
    successive projectors enrich the same entity (e.g., the root
    Dataset) without clobbering earlier contributions.
    """
    eid = entity["@id"]
    for existing in graph:
        if existing.get("@id") == eid:
            existing.update(entity)
            return existing
    graph.append(entity)
    return entity


def get_entity(graph: list[dict[str, Any]], entity_id: str) -> dict[str, Any] | None:
    """Return the entity with ``@id == entity_id``, or ``None``."""
    for entity in graph:
        if entity.get("@id") == entity_id:
            return entity
    return None


def merge_list_property(entity: dict[str, Any], key: str, value: Any) -> None:
    """Append ``value`` to ``entity[key]``, initialising it as a list.

    ``entity[key]`` may be absent (→ ``[value]``), a scalar (→ ``[old, value]``),
    or a list (→ append). Duplicate ``@id`` references are de-duped.
    """
    current = entity.get(key)
    if current is None:
        entity[key] = [value] if not isinstance(value, list) else list(value)
        return
    as_list: list[Any] = current if isinstance(current, list) else [current]
    to_add = value if isinstance(value, list) else [value]
    for item in to_add:
        if isinstance(item, dict) and "@id" in item:
            if any(isinstance(x, dict) and x.get("@id") == item["@id"] for x in as_list):
                continue
        elif item in as_list:
            continue
        as_list.append(item)
    entity[key] = as_list


def file_entity(
    bundle_dir: Path,
    file_path: Path,
    *,
    name: str,
    encoding_format: str = "application/json",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``File`` entity for a file inside the bundle.

    Computes SHA-256 lazily only if the path is a regular file — if the
    caller is referencing a placeholder that does not yet exist on disk
    (e.g., a virtual workflow entity), pass a path that does not exist
    and SHA-256 is omitted.

    ``contentSize`` is emitted as a string of decimal bytes to match
    the schema.org Text range for the property
    (https://schema.org/contentSize). RO-Crate examples in the wild use
    both integer and string forms; string is the schema.org-canonical
    serialisation and keeps JSON-LD processors from coercing the value
    through an integer type-promotion path.
    """
    entity: dict[str, Any] = {
        "@id": posix_id(bundle_dir, file_path),
        "@type": "File",
        "name": name,
        "encodingFormat": encoding_format,
    }
    if file_path.is_file():
        try:
            entity["contentSize"] = str(file_path.stat().st_size)
            entity["sha256"] = _sha256_hex(file_path)
        except OSError:  # pragma: no cover - defensive
            pass
    if extra:
        entity.update(extra)
    return entity


def load_json_optional(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from ``path`` or return ``None`` if absent/unparseable.

    ``UnicodeDecodeError`` is caught alongside ``OSError`` /
    ``json.JSONDecodeError`` so binary-corrupted files do not surface
    as opaque tracebacks. Callers that need stricter behaviour (e.g.
    fail fast on a corrupted sentinel) should use ``json.loads`` directly.
    """
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_jsonl_optional(path: Path) -> list[dict[str, Any]]:
    """Load JSONL from ``path``, returning a (possibly empty) list of dict rows."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        return []
    return rows
