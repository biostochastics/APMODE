# SPDX-License-Identifier: GPL-2.0-or-later
"""Typed operating-lane taxonomy (PRD §3).

Canonical home for the ``Lane`` enum: Submission, Discovery, and
Optimization are the three operating lanes with distinct admissible
backends, stopping rules, and evidence thresholds (PRD §3,
ARCHITECTURE.md §3). ``validate_dsl`` (:mod:`apmode.dsl.validator`) and
the governance gates/ranker (:mod:`apmode.governance`) both dispatch on
lane, so the enum lives in the DSL package -- the moat that every other
paradigm is built on top of -- rather than in ``backends``.

``apmode.backends.protocol`` re-exports this symbol (``from
apmode.dsl.lane import Lane as Lane``) for backward compatibility with
existing call sites and tests that import ``Lane`` from there; both names
refer to the same class object, so ``isinstance``/equality checks are
unaffected by which module a caller imports from. New code should prefer
importing from here.
"""

from __future__ import annotations

from enum import StrEnum

# Bump this when lane *membership* or *semantics* change -- e.g. a fourth
# lane is added, an existing lane's admissible-backend set changes in a
# way that alters downstream dispatch, or a lane is renamed. Governance-
# relevant: a future policy-schema check may cross-reference this against
# ``policies/<lane>.json`` to detect stale policy files after a taxonomy
# change. Not yet consumed by any validator as of v0.6.1-rc1.
LANE_TAXONOMY_VERSION = "1.0.0"


class Lane(StrEnum):
    """Operating lanes per PRD §3.

    Values match the plain strings persisted in ``policies/*.json``,
    ``GatePolicy.lane``, ``RunConfig.lane``, and bundle/API ``lane``
    fields, so string-keyed policy lookups and JSON (de)serialization
    continue to work unchanged when a raw string is coerced via
    ``Lane(the_string)`` at a system boundary (CLI arg parsing, API
    request bodies, policy-file loading).
    """

    SUBMISSION = "submission"
    DISCOVERY = "discovery"
    OPTIMIZATION = "optimization"


__all__ = ["LANE_TAXONOMY_VERSION", "Lane"]
