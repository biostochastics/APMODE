# SPDX-License-Identifier: GPL-2.0-or-later
"""The optional-dependency probe must not eagerly import ArviZ."""

from __future__ import annotations

import subprocess
import sys


def test_harness_import_does_not_import_arviz() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import apmode.bayes.harness; "
            "raise SystemExit(1 if 'arviz' in sys.modules else 0)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
