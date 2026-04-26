# SPDX-License-Identifier: GPL-2.0-or-later
"""Orchestrator dispatch for the ``bayesian_stan`` backend (plan Task 8).

Verifies that the lane-backend map admits ``bayesian_stan`` in the
Discovery and Optimization lanes only (Submission lane stays classical
NLME only in v0.6), and that ``Orchestrator`` exposes a ``bayesian_runner``
slot alongside ``node_runner`` and ``agentic_runner``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from apmode.orchestrator import Orchestrator, RunConfig
from apmode.routing import _LANE_BACKENDS

# Both orchestrator-instantiation tests below construct an
# ``Nlmixr2Runner``, whose ``__init__`` resolves ``Rscript`` via
# ``shutil.which`` and raises ``FileNotFoundError`` when R is not on
# PATH (defence in depth from issue #22). The CI image does not ship
# R; skip those two tests when the binary is missing rather than
# leaking the platform-dependent failure into the suite.
_RSCRIPT_AVAILABLE = shutil.which("Rscript") is not None
_RSCRIPT_REASON = "Rscript not on PATH — Nlmixr2Runner.__init__ requires it (issue #22)"


def test_lane_backends_admits_bayesian_in_discovery_and_optimization() -> None:
    """Bayesian is eligible in discovery/optimization lanes."""
    assert "bayesian_stan" in _LANE_BACKENDS["discovery"]
    assert "bayesian_stan" in _LANE_BACKENDS["optimization"]


def test_lane_backends_excludes_bayesian_from_submission() -> None:
    """Bayesian is not dispatched in the Submission lane (v0.6 scope)."""
    assert "bayesian_stan" not in _LANE_BACKENDS["submission"]


@pytest.mark.skipif(not _RSCRIPT_AVAILABLE, reason=_RSCRIPT_REASON)
@pytest.mark.asyncio
async def test_orchestrator_accepts_bayesian_runner(tmp_path: Path) -> None:
    """Orchestrator exposes a ``bayesian_runner`` kwarg that round-trips."""
    from apmode.backends.nlmixr2_runner import Nlmixr2Runner

    bay = AsyncMock()
    bay.run = AsyncMock()
    orch = Orchestrator(
        runner=Nlmixr2Runner(work_dir=tmp_path),
        bundle_base_dir=tmp_path,
        config=RunConfig(lane="discovery", seed=42),
        bayesian_runner=bay,
    )
    assert orch._bayesian_runner is bay


@pytest.mark.skipif(not _RSCRIPT_AVAILABLE, reason=_RSCRIPT_REASON)
@pytest.mark.asyncio
async def test_orchestrator_bayesian_runner_defaults_to_none(tmp_path: Path) -> None:
    """When no bayesian runner is supplied, the slot is None (back-compat)."""
    from apmode.backends.nlmixr2_runner import Nlmixr2Runner

    orch = Orchestrator(
        runner=Nlmixr2Runner(work_dir=tmp_path),
        bundle_base_dir=tmp_path,
        config=RunConfig(lane="discovery", seed=42),
    )
    assert orch._bayesian_runner is None
