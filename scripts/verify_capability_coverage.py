#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""CI-runnable coverage check for the DSL emitter capability matrix (P0.7).

Enumerates every ``apmode.dsl.capabilities.CapabilityTag`` and asserts, for
every registered emitter, the tag appears in either that emitter's
``SUPPORTS`` or ``EXPLICITLY_UNSUPPORTED`` frozenset. A tag missing from
both is a silent gap — the emitter would either crash unpredictably on that
capability or, worse, silently emit a model that does not honour it. Prints
a diagnostic table and exits non-zero on any gap.

Usage:
    uv run python scripts/verify_capability_coverage.py
"""

from __future__ import annotations

import sys

from apmode.dsl.capabilities import CapabilityTag, registered_emitters

_STATUS_SUPPORTED = "supported"
_STATUS_UNSUPPORTED = "unsupported"
_STATUS_GAP = "** GAP **"


def find_gaps() -> dict[str, list[CapabilityTag]]:
    """Return ``{emitter_name: [unclassified tags]}`` for any coverage gap."""
    gaps: dict[str, list[CapabilityTag]] = {}
    for emitter in registered_emitters():
        classified = emitter.supports | emitter.explicitly_unsupported
        missing = sorted(
            (tag for tag in CapabilityTag if tag not in classified),
            key=lambda t: t.value,
        )
        if missing:
            gaps[emitter.name] = missing
    return gaps


def print_table() -> None:
    """Print a tag x emitter support-status table to stdout."""
    emitters = registered_emitters()
    col_width = max(len(tag.value) for tag in CapabilityTag) + 2

    header = f"{'tag':<{col_width}}" + "".join(f"{e.name:<14}" for e in emitters)
    print(header)
    print("-" * len(header))

    for tag in sorted(CapabilityTag, key=lambda t: t.value):
        row = f"{tag.value:<{col_width}}"
        for emitter in emitters:
            if tag in emitter.supports:
                status = _STATUS_SUPPORTED
            elif tag in emitter.explicitly_unsupported:
                status = _STATUS_UNSUPPORTED
            else:
                status = _STATUS_GAP
            row += f"{status:<14}"
        print(row)


def main() -> int:
    print_table()
    gaps = find_gaps()

    if gaps:
        print()
        print(
            "COVERAGE GAP: the following tags are classified in neither "
            "SUPPORTS nor EXPLICITLY_UNSUPPORTED:"
        )
        for emitter_name, tags in gaps.items():
            for tag in tags:
                print(f"  {emitter_name}: {tag.value}")
        print()
        print(
            "Fix by adding each tag to the appropriate emitter's SUPPORTS or "
            "EXPLICITLY_UNSUPPORTED frozenset in src/apmode/dsl/*_emitter.py, "
            "based on the emitter's actual behaviour."
        )
        return 1

    print()
    print(
        f"OK: all {len(list(CapabilityTag))} capability tags classified for "
        f"all {len(registered_emitters())} registered emitters."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
