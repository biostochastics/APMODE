# SPDX-License-Identifier: GPL-2.0-or-later
"""Project ``attestation.json`` (reviewer sign-off) onto a File entity.

The attestation is a *post-seal sidecar*, mirroring the SBOM projector
(``entities/sbom.py``): it is written by ``apmode attest`` strictly
after ``_COMPLETE`` exists, and
:func:`apmode.bundle.emitter._DIGEST_EXCLUDED_RELATIVE_PATHS` excludes
``attestation.json`` from the sealed-bundle digest so attesting (or
re-attesting with ``--force``) never invalidates the sentinel it is
reviewing.

When present the projector adds a single ``File`` entity tagged with
``apmode:attestation`` and links it into the root Dataset's
``hasPart``, so exports run either before or after attestation both
project — and validate — successfully.
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

_ATTESTATION_FILENAME = "attestation.json"
_JSON_MEDIA_TYPE = "application/json"


def add_attestation(
    graph: list[dict[str, Any]],
    bundle_dir: Path,
    root_id: str = "./",
) -> str | None:
    """Project ``attestation.json`` as a File entity + root ``hasPart`` link.

    Returns the ``@id`` of the File entity, or ``None`` if no
    attestation is present (attestation is optional — this plan
    deliberately does not make it a Gate-blocking requirement). This
    is the only path that projects the attestation — callers should
    not hand-build File entities for it.
    """
    path = bundle_dir / _ATTESTATION_FILENAME
    if not path.is_file():
        return None
    entity = file_entity(
        bundle_dir,
        path,
        name="Reviewer Attestation",
        encoding_format=_JSON_MEDIA_TYPE,
        extra={
            "additionalType": vocab.ATTESTATION_TYPE,
            "description": (
                "Human-in-the-loop reviewer sign-off (approval decision, "
                "rationale, and any gate-check overrides) recorded after "
                "the bundle was sealed."
            ),
        },
    )
    upsert(graph, entity)
    root = upsert(graph, {"@id": root_id, "@type": "Dataset"})
    merge_list_property(root, "hasPart", {"@id": entity["@id"]})
    return str(entity["@id"])


__all__ = ["add_attestation"]
