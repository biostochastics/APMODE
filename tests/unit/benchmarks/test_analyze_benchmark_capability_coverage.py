# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for scripts/analyze_benchmark_capability_coverage.py.

``scripts/`` is not an importable package (no ``scripts/__init__.py``), so
this script is exercised via subprocess invocation rather than import,
mirroring the convention used for its sibling ``verify_capability_coverage.py``
(which currently has zero direct test coverage of its own).
"""

from __future__ import annotations

import subprocess
import sys

from tests._helpers.fixtures_data import FIXTURES_DIR

_REPO_ROOT = FIXTURES_DIR.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "analyze_benchmark_capability_coverage.py"


def _run_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_exits_zero() -> None:
    """The analyzer is a read-only report and must never fail CI."""
    result = _run_script()
    assert result.returncode == 0, result.stderr


def test_report_includes_exercised_tag() -> None:
    """A1 (1cmt oral linear) exercises first-order absorption; the report
    table must show it as covered."""
    result = _run_script()
    assert "absorption.first_order" in result.stdout


def test_report_marks_node_tags_as_experimental_not_uncovered() -> None:
    """NODE capability tags are explicitly exempted from the "uncovered"
    bucket (no stable backend exists yet -- Phase 0 P0.8), so they must
    never appear in the "zero benchmark-fixture coverage" summary list."""
    result = _run_script()
    assert "n/a (experimental)" in result.stdout
    # The summary list (if present) enumerates uncovered tags one per line
    # with a two-space indent; NODE tags must not appear there.
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped in {"absorption.node", "elimination.node"}:
            raise AssertionError(
                f"NODE tag {stripped!r} should be experimental-exempt, not listed as uncovered"
            )


def test_report_does_not_write_any_files() -> None:
    """This is a read-only report script -- running it must not create or
    modify any file under the benchmarks/ directory."""
    suite_c_dir = _REPO_ROOT / "benchmarks" / "suite_c"
    before = {p: p.stat().st_mtime for p in suite_c_dir.glob("*.dsl.json")}
    _run_script()
    after = {p: p.stat().st_mtime for p in suite_c_dir.glob("*.dsl.json")}
    assert before == after
