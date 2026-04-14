# SPDX-License-Identifier: GPL-2.0-or-later
"""sparkid integration for time-sortable, monotonic ID generation.

Reproducibility caveat
----------------------
``sparkid.generate_id()`` is **not seedable** — it uses wall-clock time plus a
random component, so candidate / run / gate IDs will differ across two runs
that set the same ``RunConfig.seed``. Bundle artifacts remain internally
consistent within a single run (IDs form a DAG via ``candidate_lineage.json``),
but cross-run comparison by ID is not meaningful.

A fully reproducible bundle would need a seeded deterministic generator, for
example a hash of ``(root_seed, role, monotonic_counter)``. Introducing one is
non-trivial because most callsites use these IDs as globally-unique handles;
the change must preserve that invariant across concurrent search, agentic
refine, and LORO-CV fold runs. Tracked for a future PR — for now, treat
cross-run ID equality as a non-goal and compare bundles via content hashes
(e.g. ``compiled_specs/<id>.json`` contents) instead of ID strings.
"""

from sparkid import generate_id


def generate_run_id() -> str:
    """Generate a unique run ID (21-char, time-sortable, Base58)."""
    return generate_id()


def generate_candidate_id() -> str:
    """Generate a unique candidate model ID."""
    return generate_id()


def generate_gate_id() -> str:
    """Generate a unique gate decision ID."""
    return generate_id()
