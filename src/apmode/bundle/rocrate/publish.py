# SPDX-License-Identifier: GPL-2.0-or-later
"""Publishing stubs for WorkflowHub / Zenodo (Phase v0.8 target).

v0.6 ships the CLI surface only — the implementations raise
:class:`NotImplementedError` so operators discover the flow and see a
helpful message pointing at the roadmap. The signatures match plan
§F so that enabling the feature in v0.8 is a drop-in replacement rather
than a CLI migration.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — runtime type in function signatures

_DEFERRED_MSG = (
    "APMODE bundle publishing is scheduled for v0.8. "
    "Use `apmode bundle rocrate export` to produce the crate now; "
    "once v0.8 lands it will upload the zip to WorkflowHub / Zenodo. "
    "Tracking: _research/ROCRATE_INTEGRATION_PLAN.md §H (v0.8)."
)


def publish_to_workflowhub(
    crate_zip: Path,
    *,
    sandbox: bool = True,
    token_env: str = "WORKFLOWHUB_TOKEN",
) -> None:
    """Upload a crate ZIP to WorkflowHub. Not implemented in v0.6."""
    del crate_zip, sandbox, token_env
    raise NotImplementedError(_DEFERRED_MSG)


def publish_to_zenodo(
    crate_zip: Path,
    *,
    sandbox: bool = True,
    token_env: str = "ZENODO_TOKEN",
) -> None:
    """Upload a crate ZIP to Zenodo. Not implemented in v0.6."""
    del crate_zip, sandbox, token_env
    raise NotImplementedError(_DEFERRED_MSG)
