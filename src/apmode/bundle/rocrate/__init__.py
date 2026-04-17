# SPDX-License-Identifier: GPL-2.0-or-later
"""RO-Crate projection of APMODE reproducibility bundles.

Projects a sealed APMODE bundle onto Workflow Run RO-Crate — Provenance
Run Crate v0.5 (``https://w3id.org/ro/wfrun/provenance/0.5``).

The bundle emitted by :class:`apmode.bundle.emitter.BundleEmitter`
remains the producer-side source of truth; the RO-Crate is a *read-only
external projection* for FAIR packaging, interoperability with
WorkflowHub / Zenodo, and machine-readable regulatory crosswalk (FDA
PCCP, EU AI Act Article 12). See ``_research/ROCRATE_INTEGRATION_PLAN.md``
for the authoritative design (sections A-H).

Public API:
- :class:`RoCrateEmitter` — orchestrator; projects a sealed bundle to
  either a directory-form crate or a ZIP archive.
- :class:`RoCrateExportOptions` — export knobs (profile, severity,
  reportable-candidate selection, PROV-AGENT opt-in).
- :class:`RoCrateProfile` — the three WRROC v0.5 profile URIs as an
  enum; ``PROVENANCE`` is the default (and the only one v0.6 conforms
  to declaratively — others are inherited per the profile hierarchy).
- :class:`ReportableSelection` — forward-compat enum for v0.7's
  Discovery-lane tiering. v0.6 always behaves like ``ALL``.
- :class:`BundleNotSealedError` — raised when a caller tries to export
  from a bundle that lacks the ``_COMPLETE`` sentinel.
- :func:`import_crate` — round-trip a crate back to a bundle directory
  with digest verification (see :mod:`.importer`).
- :class:`RoCrateImportError` — raised by :func:`import_crate` on
  integrity failures (ZIP-slip, digest mismatch, symlink).
"""

from __future__ import annotations

from apmode.bundle.rocrate.importer import RoCrateImportError, import_crate
from apmode.bundle.rocrate.projector import (
    BundleNotSealedError,
    ReportableSelection,
    RoCrateEmitter,
    RoCrateExportOptions,
    RoCrateProfile,
)

__all__ = [
    "BundleNotSealedError",
    "ReportableSelection",
    "RoCrateEmitter",
    "RoCrateExportOptions",
    "RoCrateImportError",
    "RoCrateProfile",
    "import_crate",
]
