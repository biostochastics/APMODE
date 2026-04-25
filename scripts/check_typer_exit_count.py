#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""CI ratchet: monotonic-decrease guard on ``raise typer.Exit`` in cli.py.

PR2 introduces a typed-error hierarchy at ``apmode.cli_errors`` and an
entrypoint catcher at ``apmode.__main__``. Existing ``raise typer.Exit(N)``
call sites are *preserved* on purpose — migration happens command-by-command
in PR6+ to keep each diff small. This script enforces the migration
direction:

* The current count of ``raise typer.Exit`` lines in ``src/apmode/cli.py``
  is compared against the locked baseline at
  ``scripts/typer_exit_baseline.json``.
* When the count *equals* the baseline → exit ``0``.
* When the count is *below* the baseline → print a one-line note
  prompting the maintainer to lower the baseline file (this is the
  "you migrated some call sites; lock in the win" path) and exit ``0``.
* When the count *exceeds* the baseline → exit ``1`` with a hard-fail
  message naming the surplus.

The intent is to make regressions visible without forcing every PR to
rewrite the baseline. Run from CI as:

    python scripts/check_typer_exit_count.py

Update the baseline (after a deliberate PR that moves call sites onto
the typed-error infrastructure) by passing ``--update``:

    python scripts/check_typer_exit_count.py --update
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_FILE = ROOT / "src/apmode/cli.py"
BASELINE_FILE = ROOT / "scripts/typer_exit_baseline.json"

# Match ``raise typer.Exit(...)`` on its own line, with any leading
# indentation. Accepts the parenthesised forms ``raise (typer.Exit(...))``
# and ``raise(typer.Exit(...))`` that line-continuation refactors
# sometimes produce — without the alternation the ratchet would silently
# undercount and a regression could slip past CI.
#
#   ``raise typer.Exit``   matches via the ``\s+`` branch.
#   ``raise (typer.Exit``  matches via the parenthesised branch.
#   ``raise(typer.Exit``   matches via the parenthesised branch (no space).
#   ``raisetyper.Exit``    does NOT match (invalid Python anyway).
_RAISE_TYPER_EXIT_RE = re.compile(
    r"^\s*raise(?:\s+|\s*\(\s*)typer\.Exit\b",
    re.M,
)


def count_typer_exits(source: str) -> int:
    """Count ``raise typer.Exit`` statements in a source string."""
    return len(_RAISE_TYPER_EXIT_RE.findall(source))


def read_baseline(path: Path = BASELINE_FILE) -> int:
    """Read the locked baseline count. Returns ``-1`` if the file is absent.

    A missing baseline is treated as "PR2 has not landed yet" and the
    script silently no-ops with exit ``0``. That keeps the script safe
    to drop in before the baseline file is committed.
    """
    if not path.exists():
        return -1
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["raise_typer_exit_count"])


def write_baseline(count: int, path: Path = BASELINE_FILE) -> None:
    """Persist the current count to the baseline file."""
    payload = {
        "raise_typer_exit_count": count,
        "rationale": (
            "Locked by scripts/check_typer_exit_count.py. PR2 introduced "
            "typed-error infrastructure at apmode.cli_errors + an entrypoint "
            "catcher at apmode.__main__. Subsequent PRs migrate individual "
            "call sites and lower this count. Run `python "
            "scripts/check_typer_exit_count.py --update` after a migration."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the baseline file to the current count (do this on the migration PR).",
    )
    args = parser.parse_args()

    if not CLI_FILE.exists():
        print(f"error: {CLI_FILE} not found", file=sys.stderr)
        return 1

    current = count_typer_exits(CLI_FILE.read_text(encoding="utf-8"))

    if args.update:
        write_baseline(current)
        print(f"baseline updated → {current} ``raise typer.Exit`` call sites in cli.py")
        return 0

    baseline = read_baseline()
    if baseline < 0:
        # No baseline yet — be silent so the script can land in CI ahead
        # of the baseline file.
        print(
            f"note: {BASELINE_FILE.name} not present yet; "
            f"current count is {current}. Run with --update to lock it in."
        )
        return 0

    if current > baseline:
        print(
            "error: ``raise typer.Exit`` count regressed in src/apmode/cli.py.\n"
            f"  baseline: {baseline}\n"
            f"  current:  {current}\n"
            f"  surplus:  +{current - baseline}\n"
            "Either remove the new ``raise typer.Exit`` (use the typed errors at "
            "apmode.cli_errors instead) or, if the new sites are intentional, "
            "raise the baseline via --update with a justification.",
            file=sys.stderr,
        )
        return 1

    if current < baseline:
        print(
            f"good: ``raise typer.Exit`` count dropped from {baseline} to {current}. "
            f"Run with --update to lock the win in.",
        )
        return 0

    print(f"ok: {current} ``raise typer.Exit`` call sites (matches baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
