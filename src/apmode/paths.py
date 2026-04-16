# SPDX-License-Identifier: GPL-2.0-or-later
"""Filesystem-path helpers shared by the CLI and orchestrator.

Centralizes the repo-root / policies lookup so the CLI and orchestrator
cannot drift on parent-count heuristics. Resolution strategy, in order:

1. If ``APMODE_POLICIES_DIR`` is set in the environment, use it verbatim.
2. Walk up from this module's ``__file__`` until a directory with
   ``pyproject.toml`` is found; use ``<root>/policies``. Handles the
   editable-install / repo-root workflow that APMODE currently targets.
3. Fall back to ``importlib.resources.files("apmode").joinpath("policies")``
   for site-packages installs once ``policies/`` is shipped as package data
   (see hatch ``include_package_data`` / TODO in ``pyproject.toml``).

The helper returns ``None`` if nothing resolves; callers decide whether
that's fatal (orchestrator treats as "no policy" → default).
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def _walk_up_for_pyproject(start: Path) -> Path | None:
    """Walk up parents from ``start`` looking for ``pyproject.toml``.

    Returns the directory containing ``pyproject.toml`` or ``None`` if the
    filesystem root is reached without finding one. This handles both the
    editable-install layout (``src/apmode/...``) and flat src-less layouts.
    """
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def policies_dir() -> Path | None:
    """Locate the versioned gate-policy directory.

    See module docstring for resolution order. Returns ``None`` if no
    strategy finds a usable directory.
    """
    override = os.environ.get("APMODE_POLICIES_DIR")
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            return path

    here = Path(__file__).resolve().parent
    repo_root = _walk_up_for_pyproject(here)
    if repo_root is not None:
        candidate = repo_root / "policies"
        if candidate.is_dir():
            return candidate

    try:
        pkg_policies = resources.files("apmode").joinpath("policies")
        # ``as_file`` + ``is_dir()`` is the portable way to query; for
        # editable installs the MultiplexedPath yields a real Path.
        as_path = Path(str(pkg_policies))
        if as_path.is_dir():
            return as_path
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    return None


def policy_path_for_lane(lane: str) -> Path | None:
    """Return ``policies/<lane>.json`` if both directory and file exist."""
    base = policies_dir()
    if base is None:
        return None
    p = base / f"{lane}.json"
    return p if p.is_file() else None
