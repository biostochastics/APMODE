# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the ``raise typer.Exit`` CI ratchet.

The ratchet is the migration safety net for PR6+: it locks in the
current count of legacy ``raise typer.Exit(N)`` call sites so a future
PR cannot accidentally add new ones without raising the baseline. As
call sites migrate onto ``apmode.cli_errors.APModeCLIError``, the
baseline file is updated downward via ``--update``.

These tests pin three behaviours of the script:

* ``count_typer_exits`` only matches well-formed ``raise typer.Exit``
  statements — not docstrings, comments, or string literals.
* The ``--update`` path writes a baseline file with the current count.
* Running with ``current > baseline`` exits non-zero; ``current ==
  baseline`` exits zero; ``current < baseline`` exits zero with a hint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Import the script's helpers directly. The script lives at
# ``scripts/check_typer_exit_count.py`` and is not on ``sys.path``, so
# add the parent directory before importing.
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from check_typer_exit_count import (  # noqa: E402
    count_typer_exits,
    read_baseline,
    write_baseline,
)

# ---------------------------------------------------------------------------
# count_typer_exits — regex precision
# ---------------------------------------------------------------------------


def test_counts_simple_raise() -> None:
    src = "    raise typer.Exit(1)\n"
    assert count_typer_exits(src) == 1


def test_counts_multiple_raises() -> None:
    src = (
        "def a():\n"
        "    raise typer.Exit(1)\n"
        "def b():\n"
        "    raise typer.Exit(code=2)\n"
        "def c():\n"
        "    raise typer.Exit()\n"
    )
    assert count_typer_exits(src) == 3


def test_ignores_comments() -> None:
    src = "# raise typer.Exit(1) — explanatory comment\n"
    assert count_typer_exits(src) == 0


def test_ignores_docstring_mentions() -> None:
    """A real docstring is a string literal; the regex only matches the
    statement form, so even mentions in triple quotes are skipped."""
    src = '"""On error this function will raise typer.Exit(1)."""\n'
    assert count_typer_exits(src) == 0


def test_handles_no_indentation() -> None:
    """Module-level raises (rare but legal) must still be counted."""
    src = "raise typer.Exit(1)\n"
    assert count_typer_exits(src) == 1


def test_does_not_match_typer_exit_assignment() -> None:
    """Variable assignment to a ``typer.Exit`` instance is not a raise."""
    src = "    e = typer.Exit(1)\n"
    assert count_typer_exits(src) == 0


def test_counts_parenthesised_raise() -> None:
    """``raise (typer.Exit(...))`` (line-continuation refactor form) must be
    counted; otherwise the ratchet silently undercounts and a regression
    could slip past CI."""
    src = "    raise (typer.Exit(1))\n    raise(  typer.Exit(2))\n"
    assert count_typer_exits(src) == 2


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def test_write_then_read_baseline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    write_baseline(42, path=path)
    assert read_baseline(path) == 42
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["raise_typer_exit_count"] == 42
    # Rationale string is committed alongside so reviewers know why the
    # number changed when they see a baseline diff.
    assert "PR2" in payload["rationale"]


def test_read_baseline_returns_negative_when_missing(tmp_path: Path) -> None:
    """Missing baseline → script no-ops with exit 0 (drop-in safe)."""
    assert read_baseline(tmp_path / "nope.json") == -1


# ---------------------------------------------------------------------------
# End-to-end script invocation
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script(*extra: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO_ROOT / "scripts/check_typer_exit_count.py"), *extra]
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_passes_against_locked_baseline() -> None:
    """In a clean repo state the count matches the locked baseline."""
    result = _run_script()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches baseline" in result.stdout


def test_script_fails_when_count_exceeds_baseline(tmp_path: Path) -> None:
    """Synthetic repo: forge a baseline of 0 against a cli.py with raises."""
    repo = tmp_path / "repo"
    (repo / "src/apmode").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/apmode/cli.py").write_text(
        "def x():\n    raise typer.Exit(1)\n", encoding="utf-8"
    )
    (repo / "scripts/typer_exit_baseline.json").write_text(
        json.dumps({"raise_typer_exit_count": 0, "rationale": "test"}),
        encoding="utf-8",
    )

    # Copy the script so it resolves paths against the synthetic repo
    # (the production script uses ``parents[1]`` from its own location).
    script_src = (REPO_ROOT / "scripts/check_typer_exit_count.py").read_text(encoding="utf-8")
    (repo / "scripts/check_typer_exit_count.py").write_text(script_src, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/check_typer_exit_count.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, (result.stdout, result.stderr)
    # Hard-fail message names the surplus and the migration target.
    assert "regressed" in result.stderr
    assert "apmode.cli_errors" in result.stderr


def test_script_passes_when_count_dropped_below_baseline(tmp_path: Path) -> None:
    """Synthetic repo: forge a baseline of 5 against a cli.py with 1 raise."""
    repo = tmp_path / "repo"
    (repo / "src/apmode").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/apmode/cli.py").write_text(
        "def x():\n    raise typer.Exit(1)\n", encoding="utf-8"
    )
    (repo / "scripts/typer_exit_baseline.json").write_text(
        json.dumps({"raise_typer_exit_count": 5, "rationale": "test"}),
        encoding="utf-8",
    )
    script_src = (REPO_ROOT / "scripts/check_typer_exit_count.py").read_text(encoding="utf-8")
    (repo / "scripts/check_typer_exit_count.py").write_text(script_src, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/check_typer_exit_count.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "dropped from 5 to 1" in result.stdout
    assert "--update" in result.stdout


def test_script_update_flag_writes_current_count(tmp_path: Path) -> None:
    """``--update`` rewrites the baseline file to whatever the current count is."""
    repo = tmp_path / "repo"
    (repo / "src/apmode").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "src/apmode/cli.py").write_text(
        "raise typer.Exit(1)\nraise typer.Exit(2)\nraise typer.Exit()\n",
        encoding="utf-8",
    )
    script_src = (REPO_ROOT / "scripts/check_typer_exit_count.py").read_text(encoding="utf-8")
    (repo / "scripts/check_typer_exit_count.py").write_text(script_src, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/check_typer_exit_count.py", "--update"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    payload = json.loads((repo / "scripts/typer_exit_baseline.json").read_text(encoding="utf-8"))
    assert payload["raise_typer_exit_count"] == 3
