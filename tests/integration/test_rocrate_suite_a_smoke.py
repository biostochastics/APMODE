# SPDX-License-Identifier: GPL-2.0-or-later
"""Opt-in smoke tests that export + validate RO-Crates from real Suite-A runs.

The synthetic fixtures in :mod:`tests.unit.rocrate._fixtures` cover every
projector code path, but they deliberately short-circuit the R /
nlmixr2 dependency. Plan §H v0.6 acceptance criterion 1 requires that
*real* Suite-A Submission bundles validate at REQUIRED severity
against ``provenance-run-crate-0.5``. This file closes that gap:

- Tests are gated by ``APMODE_SUITE_A_SMOKE=1`` so CI can opt in when
  R / rxode2 / nlmixr2 are provisioned, while local runs and PR
  validation without R stay fast.
- The fixture drives ``apmode run`` for one Suite-A CSV, then runs the
  full projector + roc-validator pipeline over the resulting bundle.
- Failure here means a real-bundle shape slipped past the synthetic
  coverage — exactly the gap the synthetic fixtures could not catch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rocrate_validator")

from rocrate_validator import models as _rcv_models
from rocrate_validator import services as _rcv_services

from apmode.bundle.rocrate import RoCrateEmitter, RoCrateExportOptions

_SUITE_A_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "suite_a"
_SKIP_REASON = (
    "Suite-A smoke is opt-in — set APMODE_SUITE_A_SMOKE=1 and provision R / "
    "nlmixr2 (see docs/PRD_APMODE_v0.3.md) to enable."
)


def _suite_a_enabled() -> bool:
    if os.environ.get("APMODE_SUITE_A_SMOKE") != "1":
        return False
    if shutil.which("R") is None:
        return False
    return _SUITE_A_ROOT.is_dir()


@pytest.mark.slow
@pytest.mark.skipif(not _suite_a_enabled(), reason=_SKIP_REASON)
@pytest.mark.parametrize(
    "dataset",
    [
        "a1_1cmt_oral_linear.csv",
        "a3_transit_1cmt_linear.csv",
    ],
)
def test_suite_a_bundle_validates(dataset: str, tmp_path: Path) -> None:
    """Run ``apmode run`` on a Suite-A CSV, export, and validate REQUIRED."""
    csv_path = _SUITE_A_ROOT / dataset
    assert csv_path.is_file(), f"Suite-A fixture missing: {csv_path}"

    runs_dir = tmp_path / "runs"
    run_cmd = [
        sys.executable,
        "-m",
        "apmode.cli",
        "run",
        str(csv_path),
        "--lane",
        "submission",
        "--output-dir",
        str(runs_dir),
        "--timeout",
        "300",
    ]
    # ``apmode run`` itself performs the full pipeline; bubble up its
    # stderr on failure so the CI log explains what broke.
    result = subprocess.run(run_cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"apmode run failed ({result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    bundles = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    assert bundles, f"apmode run produced no bundle under {runs_dir}"
    bundle = bundles[-1]

    crate_dir = tmp_path / "crate"
    RoCrateEmitter().export_from_sealed_bundle(
        bundle,
        crate_dir,
        RoCrateExportOptions(),
    )

    settings = _rcv_models.ValidationSettings(
        rocrate_uri=str(crate_dir),
        profile_identifier="provenance-run-crate-0.5",
        requirement_severity=_rcv_models.Severity.REQUIRED,
    )
    validation = _rcv_services.validate(settings)
    messages = [f"{issue.severity.name}: {issue.message}" for issue in validation.get_issues()]
    assert validation.passed(), (
        "Real Suite-A bundle failed provenance-run-crate-0.5 REQUIRED:\n  " + "\n  ".join(messages)
    )
