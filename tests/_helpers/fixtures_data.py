# SPDX-License-Identifier: GPL-2.0-or-later
"""Rootdir-anchored access to the shared test fixture corpus.

``FIXTURES_DIR`` resolves to ``tests/fixtures`` from this file's location
(``tests/_helpers/../fixtures``), so it is STABLE no matter how deep in the
``tests/unit`` subpackage tree the importing test module lives. Test modules
must use this instead of ``Path(__file__).parent.parent / "fixtures"`` (which
breaks the moment the module is moved into a subpackage) or CWD-relative
``Path("tests/fixtures/...")`` (which breaks when pytest runs from another cwd).
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def pk_fixture(*parts: str) -> Path:
    """Return a path under ``tests/fixtures``.

    Example: ``pk_fixture("pk_data", "simple_1cmt.csv")``.
    """
    return FIXTURES_DIR.joinpath(*parts)
