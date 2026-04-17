# SPDX-License-Identifier: GPL-2.0-or-later
"""Entity projectors.

Each submodule here is responsible for projecting one family of APMODE
bundle artifacts onto the RO-Crate graph. Projectors are pure: they
accept a mutable ``graph`` list, a ``bundle_dir`` path, and whatever
context the caller has (already-minted root ``@id`` strings, typed
Pydantic models, etc.). They do not perform disk I/O outside the
provided ``bundle_dir`` and they do not mutate the bundle.

All projectors should be importable without pulling in Pydantic models
that might trigger schema drift errors — the typed validation happens
inside each function, guarded by try/except, so a missing or
slightly-evolved artifact simply falls back to a minimal File-only
projection rather than aborting the whole export.
"""

from __future__ import annotations
