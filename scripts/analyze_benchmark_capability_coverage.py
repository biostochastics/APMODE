#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Read-only coverage check: which DSL v0.7 capability tags do the Suite A/C
benchmark fixtures actually exercise?

Companion to ``scripts/verify_capability_coverage.py`` (which checks emitter
code paths, not fixtures). This script never mutates ``benchmarks/suite_c/*.dsl.json``
or Suite A scenario definitions — it only reports the gap so fixtures can be
extended deliberately once DSL v0.7 lands on ``main``, not against a moving
grammar target.

Usage:
    uv run python scripts/analyze_benchmark_capability_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from apmode.benchmarks.suite_a import ALL_SCENARIOS
from apmode.dsl.ast_models import DSLSpec
from apmode.dsl.capabilities import CapabilityTag, tags_for_spec

_SUITE_C_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "suite_c"

# NODE tags have no stable backend anywhere yet (Phase 0 P0.8) — excluded
# from "uncovered" since there is nothing a benchmark fixture could exercise
# that would change their status.
_EXPERIMENTAL_EXEMPT: frozenset[CapabilityTag] = frozenset(
    {CapabilityTag.ABSORPTION_NODE, CapabilityTag.ELIMINATION_NODE}
)


def _suite_c_specs() -> list[DSLSpec]:
    return [
        DSLSpec.model_validate_json(p.read_text()) for p in sorted(_SUITE_C_DIR.glob("*.dsl.json"))
    ]


def exercised_tags() -> frozenset[CapabilityTag]:
    """Union of capability tags exercised across every Suite A + Suite C fixture."""
    tags: set[CapabilityTag] = set()
    for _scenario_id, factory in ALL_SCENARIOS:
        tags |= tags_for_spec(factory())
    for spec in _suite_c_specs():
        tags |= tags_for_spec(spec)
    return frozenset(tags)


def uncovered_tags() -> frozenset[CapabilityTag]:
    """Capability tags with zero benchmark-fixture coverage (excluding NODE)."""
    all_tags = frozenset(CapabilityTag) - _EXPERIMENTAL_EXEMPT
    return all_tags - exercised_tags()


def main() -> int:
    exercised = exercised_tags()
    uncovered = uncovered_tags()

    print(f"{'tag':<45}{'covered'}")
    print("-" * 55)
    for tag in sorted(CapabilityTag, key=lambda t: t.value):
        status = (
            "covered"
            if tag in exercised
            else ("n/a (experimental)" if tag in _EXPERIMENTAL_EXEMPT else "** UNCOVERED **")
        )
        print(f"{tag.value:<45}{status}")

    print()
    if uncovered:
        print(f"{len(uncovered)} capability tag(s) with zero benchmark-fixture coverage:")
        for tag in sorted(uncovered, key=lambda t: t.value):
            print(f"  {tag.value}")
        print()
        print(
            "This is a report, not a failure — Suite C fixtures should be "
            "extended to cover these once DSL v0.7 lands on main. Do not "
            "edit suite_c/*.dsl.json against the current uncommitted grammar."
        )
    else:
        print("OK: all non-experimental capability tags have benchmark-fixture coverage.")
    return 0  # informational — never fails CI, mirrors "report" framing above


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
