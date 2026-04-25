# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for ``apmode._version_drift`` and the ``--version`` CLI line.

The drift helper has two shapes worth pinning:

* **Pure functions** (``is_drifted`` + ``format_version_line``) — exhaustive
  parametrised coverage of the PEP-440 corner cases that motivated this
  module: hyphen-vs-no-hyphen rc forms, ``.devN`` segments, ``+local``
  provenance suffixes, and the ``runtime == "unknown"`` branch.
* **I/O-touching collectors** (``collect_runtime_version`` +
  ``collect_declared_version``) — exercised through monkeypatching so the
  tests do not depend on whatever VCS state the repo happens to be in.

The CLI surface (``apmode --version``) is tested last with ``CliRunner`` to
confirm aligned and drifted outputs are stable end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from apmode._version_drift import (
    collect_declared_version,
    collect_runtime_version,
    format_version_line,
    is_drifted,
)
from apmode.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# is_drifted — PEP-440 canonicalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declared", "runtime"),
    [
        # Identical
        ("0.6.0", "0.6.0"),
        # Hyphen-vs-no-hyphen rc form (the original motivating case)
        ("0.6.0-rc1", "0.6.0rc1"),
        ("0.6.0rc1", "0.6.0-rc1"),
        # +local segment is provenance, not identity — must be ignored
        ("0.6.0-rc1", "0.6.0rc1+gabc1234.d20260424"),
        ("0.6.0", "0.6.0+local.build"),
        # Other PEP-440 separators that canonicalise to the same release
        ("1.0.0-alpha1", "1.0.0a1"),
        ("1.0.0-beta2", "1.0.0b2"),
    ],
)
def test_aligned_versions_are_not_drifted(declared: str, runtime: str) -> None:
    assert is_drifted(declared, runtime) is False


@pytest.mark.parametrize(
    ("declared", "runtime"),
    [
        # Different release identity
        ("0.6.0-rc1", "0.5.0-rc2"),
        ("0.6.0-rc1", "0.6.0-rc2"),
        ("0.6.0", "0.6.1"),
        # devN downstream of the most recent tag is drift relative to a
        # tagged-release declaration: the bytes installed are NOT what
        # 0.6.0-rc1 will be when it is cut.
        ("0.6.0-rc1", "0.3.0rc4.dev77+gfcf87e16e.d20260425"),
        ("0.6.0-rc1", "0.6.0rc1.dev5"),
        # runtime == "unknown" is drift by convention (declared known, runtime not)
        ("0.6.0-rc1", "unknown"),
    ],
)
def test_mismatched_versions_are_drifted(declared: str, runtime: str) -> None:
    assert is_drifted(declared, runtime) is True


def test_drift_is_false_when_declared_unknown() -> None:
    """When CHANGELOG.md is missing (typical wheel install), drift is N/A."""
    assert is_drifted(None, "0.3.0rc4.dev77+gabc") is False
    assert is_drifted(None, "unknown") is False


def test_invalid_pep440_falls_back_to_lexical_compare() -> None:
    """Garbage inputs do not crash; they degrade to string compare."""
    assert is_drifted("not-a-version", "also-not") is True
    assert is_drifted("not-a-version", "not-a-version") is False


# ---------------------------------------------------------------------------
# format_version_line
# ---------------------------------------------------------------------------


def test_format_aligned_shows_single_version() -> None:
    assert format_version_line(declared="0.6.0-rc1", runtime="0.6.0rc1") == "apmode 0.6.0-rc1"


def test_format_drifted_surfaces_both() -> None:
    line = format_version_line(
        declared="0.6.0-rc1",
        runtime="0.3.0rc4.dev77+gfcf87e16e.d20260425",
    )
    assert line == "apmode 0.6.0-rc1 (runtime 0.3.0rc4.dev77+gfcf87e16e.d20260425)"


def test_format_falls_back_to_runtime_when_declared_unknown() -> None:
    """Wheel install path — show whatever bits are installed."""
    assert format_version_line(declared=None, runtime="0.6.0rc1") == "apmode 0.6.0rc1"


def test_format_preserves_local_segment_in_runtime_display() -> None:
    """Drift display must not strip ``+gHASH.dDATE`` — it is the provenance."""
    line = format_version_line(
        declared="0.6.0-rc1",
        runtime="0.6.0rc1.dev5+gabc.d20260424",
    )
    assert "+gabc.d20260424" in line


# ---------------------------------------------------------------------------
# collect_runtime_version
# ---------------------------------------------------------------------------


def test_collect_runtime_uses_importlib_metadata_when_present() -> None:
    """First-choice resolver is ``importlib.metadata.version``."""
    with patch("apmode._version_drift._metadata_version", return_value="9.9.9"):
        assert collect_runtime_version() == "9.9.9"


def test_collect_runtime_falls_back_to_version_module() -> None:
    """When importlib.metadata raises, ``apmode._version`` is consulted."""
    from importlib.metadata import PackageNotFoundError

    with patch(
        "apmode._version_drift._metadata_version",
        side_effect=PackageNotFoundError("apmode"),
    ):
        # ``apmode._version`` is generated by hatch-vcs in this repo, so the
        # call returns a real value rather than ``"unknown"``. We don't
        # hard-code the value (it varies per checkout) but assert it is a
        # non-empty string that is not the sentinel.
        result = collect_runtime_version()
        assert isinstance(result, str)
        assert result != ""
        assert result != "unknown"


def test_collect_runtime_returns_unknown_when_both_resolvers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    with patch(
        "apmode._version_drift._metadata_version",
        side_effect=PackageNotFoundError("apmode"),
    ):
        # Block the secondary import path. Removing the cached module forces
        # re-import; patching ``builtins.__import__`` rejects it.
        import builtins
        import sys

        monkeypatch.delitem(sys.modules, "apmode._version", raising=False)
        real_import = builtins.__import__

        def _block_version(name: str, *a: object, **kw: object) -> object:
            if name == "apmode._version":
                raise ModuleNotFoundError("blocked for test")
            return real_import(name, *a, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _block_version)
        assert collect_runtime_version() == "unknown"


# ---------------------------------------------------------------------------
# collect_declared_version
# ---------------------------------------------------------------------------


def test_collect_declared_reads_most_recent_changelog_header(tmp_path: Path) -> None:
    """The first ``## [X.Y.Z]`` header (top-down) wins."""
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n- something\n\n"
        "## [0.7.0-rc1] — 2026-05-01\n\n- newer\n\n"
        "## [0.6.0-rc1] — 2026-04-24\n\n- older\n",
        encoding="utf-8",
    )
    assert collect_declared_version(repo_root=tmp_path) == "0.7.0-rc1"


def test_collect_declared_returns_none_when_changelog_missing(tmp_path: Path) -> None:
    """Wheel install path — CHANGELOG.md is not shipped, return None."""
    assert collect_declared_version(repo_root=tmp_path) is None


def test_collect_declared_returns_none_on_unparseable_changelog(tmp_path: Path) -> None:
    """A malformed file is not fatal; the CLI degrades to runtime-only."""
    (tmp_path / "CHANGELOG.md").write_text("nothing useful here\n", encoding="utf-8")
    assert collect_declared_version(repo_root=tmp_path) is None


def test_collect_declared_skips_unreleased_header() -> None:
    """``## [Unreleased]`` is the ``--check`` working-set marker, never the
    declared version: it does not match the X.Y.Z regex so the next header
    wins, which is what users want."""
    assert collect_declared_version() is not None  # repo has a real version


def test_collect_declared_accepts_multi_segment_pre_release(tmp_path: Path) -> None:
    """The widened regex accepts ``-rc1.2`` and ``.dev5`` PEP-440 forms."""
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.7.0-rc1.2] — 2026-05-01\n",
        encoding="utf-8",
    )
    assert collect_declared_version(repo_root=tmp_path) == "0.7.0-rc1.2"


def test_collect_declared_accepts_dot_separated_dev_segment(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.0.0.dev1] — 2026-06-01\n",
        encoding="utf-8",
    )
    assert collect_declared_version(repo_root=tmp_path) == "1.0.0.dev1"


def test_drift_strips_local_segment_from_declared_too() -> None:
    """Symmetric ``+local`` strip — declared bearing a local segment should
    not falsely flag drift against an aligned runtime."""
    # Same release identity, both sides carry their own local stamp.
    assert is_drifted("0.6.0-rc1+a.b.c", "0.6.0rc1+gabc.d20260424") is False


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestVersionCli:
    """End-to-end: ``apmode --version`` exit code + output shape."""

    def test_version_flag_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_version_flag_prints_apmode_token(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert "apmode" in result.output

    def test_version_aligned_format(self) -> None:
        """Force aligned state via patches; first line shows single value."""
        with (
            patch(
                "apmode._version_drift.collect_runtime_version",
                return_value="0.6.0rc1",
            ),
            patch(
                "apmode._version_drift.collect_declared_version",
                return_value="0.6.0-rc1",
            ),
        ):
            result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0]
        assert first_line == "apmode 0.6.0-rc1"

    def test_version_drifted_format(self) -> None:
        """Force drifted state via patches; first line shows both."""
        with (
            patch(
                "apmode._version_drift.collect_runtime_version",
                return_value="0.3.0rc4.dev77+gabc.d20260424",
            ),
            patch(
                "apmode._version_drift.collect_declared_version",
                return_value="0.6.0-rc1",
            ),
        ):
            result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        first_line = result.output.splitlines()[0]
        assert first_line == "apmode 0.6.0-rc1 (runtime 0.3.0rc4.dev77+gabc.d20260424)"

    def test_version_includes_copyright_and_citation(self) -> None:
        """Provenance lines are preserved across the drift refactor."""
        result = runner.invoke(app, ["--version"])
        assert "Biostochastics" in result.output
        assert "Cite" in result.output
