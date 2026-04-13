# SPDX-License-Identifier: GPL-2.0-or-later
"""BackendRunner protocol and Lane enum (ARCHITECTURE.md §4.1)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 — used at runtime in Protocol signature
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest
    from apmode.dsl.ast_models import DSLSpec


class Lane(StrEnum):
    """Operating lanes per PRD §3."""

    SUBMISSION = "submission"
    DISCOVERY = "discovery"
    OPTIMIZATION = "optimization"


@runtime_checkable
class BackendRunner(Protocol):
    """Interface contract for all backends.

    Phase 1: subprocess implementation.
    Phase 2: Flyte @task or Temporal Activity wrapper.
    """

    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
    ) -> BackendResult: ...
