# SPDX-License-Identifier: GPL-2.0-or-later
"""Publishing stubs for WorkflowHub / Zenodo.

The CLI surface exists, but upload implementations are not wired yet. The
stubs raise :class:`NotImplementedError` with guidance to export a crate ZIP
locally and track the publishing work separately.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures

_DEFERRED_MSG = (
    "APMODE bundle publishing is not implemented yet. "
    "Use `apmode bundle rocrate export` to produce the crate now; "
    "the future publisher will upload the zip to WorkflowHub / Zenodo. "
    "Tracking: _research/ROCRATE_INTEGRATION_PLAN.md §H."
)


def publish_to_workflowhub(
    crate_zip: Path,
    *,
    sandbox: bool = True,
    token_env: str = "WORKFLOWHUB_TOKEN",
) -> None:
    """Upload a crate ZIP to WorkflowHub. Not implemented."""
    del crate_zip, sandbox, token_env
    raise NotImplementedError(_DEFERRED_MSG)


def publish_to_zenodo(
    crate_zip: Path,
    *,
    sandbox: bool = True,
    token_env: str = "ZENODO_TOKEN",
) -> None:
    """Upload a crate ZIP to Zenodo. Not implemented."""
    del crate_zip, sandbox, token_env
    raise NotImplementedError(_DEFERRED_MSG)
