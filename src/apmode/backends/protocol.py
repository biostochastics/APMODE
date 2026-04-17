# SPDX-License-Identifier: GPL-2.0-or-later
"""BackendRunner protocol and Lane enum (ARCHITECTURE.md §4.1)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path  # noqa: TC003 — used at runtime in Protocol signature
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config


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
        split_manifest: dict[str, object] | None = None,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
        fixed_parameter: bool = False,
    ) -> BackendResult:
        """Run the backend.

        ``fixed_parameter``: when True, the backend MUST evaluate the
        observation likelihood / predictions at the supplied
        ``initial_estimates`` without re-estimating any structural or
        variance-component parameter. This is the contract LORO-CV relies
        on to prevent full-data posteriors from leaking into held-out
        folds via warm-start refits. Backends that cannot honour
        fixed-parameter mode must raise ``NotImplementedError``.
        """
        ...
