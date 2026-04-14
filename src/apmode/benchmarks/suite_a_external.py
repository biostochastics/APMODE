# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A-External: Schoemaker 2019 standard grid (PRD §5).

Integrates the 12 ACOP 2016 simulated datasets from the nlmixr2data R
package. These are the pharmacometrics field's canonical algorithm
comparison grid, used by Schoemaker et al. (2019) to compare nlmixr2
SAEM/FOCEI vs NONMEM FOCE vs Monolix SAEM.

Each dataset is a full-factorial crossing of:
  Structure: {1-CMT, 2-CMT}
  Route:     {IV bolus, IV infusion, Oral}
  Clearance: {Linear, Michaelis-Menten}

120 subjects per dataset (30 x 4 dose levels), rich sampling.

Reference:
  Schoemaker R et al. (2019). CPT: Pharmacometrics & Systems Pharmacology
  8(12):923-930. doi:10.1002/psp4.12471

Usage:
  Cases are used for breadth + convergence stress-testing + cross-tool
  comparability. They supplement (not replace) the custom A1-A7 scenarios.
  Default cadence: weekly (too large for per-PR CI).
"""

from __future__ import annotations

from apmode.benchmarks.models import (
    BenchmarkCase,
    ExpectedStructure,
)

# ---------------------------------------------------------------------------
# Dataset grid: 12 datasets x structural properties
# ---------------------------------------------------------------------------

_GRID: list[dict[str, str | int]] = [
    {"id": "bolus_1cpt", "route": "iv_bolus", "cmt": 1, "elim": "Linear", "abs": "none"},
    {
        "id": "bolus_1cptmm",
        "route": "iv_bolus",
        "cmt": 1,
        "elim": "MichaelisMenten",
        "abs": "none",
    },
    {"id": "bolus_2cpt", "route": "iv_bolus", "cmt": 2, "elim": "Linear", "abs": "none"},
    {
        "id": "bolus_2cptmm",
        "route": "iv_bolus",
        "cmt": 2,
        "elim": "MichaelisMenten",
        "abs": "none",
    },
    {"id": "infusion_1cpt", "route": "iv_infusion", "cmt": 1, "elim": "Linear", "abs": "none"},
    {
        "id": "infusion_1cptmm",
        "route": "iv_infusion",
        "cmt": 1,
        "elim": "MichaelisMenten",
        "abs": "none",
    },
    {"id": "infusion_2cpt", "route": "iv_infusion", "cmt": 2, "elim": "Linear", "abs": "none"},
    {
        "id": "infusion_2cptmm",
        "route": "iv_infusion",
        "cmt": 2,
        "elim": "MichaelisMenten",
        "abs": "none",
    },
    {"id": "oral_1cpt", "route": "oral", "cmt": 1, "elim": "Linear", "abs": "FirstOrder"},
    {
        "id": "oral_1cptmm",
        "route": "oral",
        "cmt": 1,
        "elim": "MichaelisMenten",
        "abs": "FirstOrder",
    },
    {"id": "oral_2cpt", "route": "oral", "cmt": 2, "elim": "Linear", "abs": "FirstOrder"},
    {
        "id": "oral_2cptmm",
        "route": "oral",
        "cmt": 2,
        "elim": "MichaelisMenten",
        "abs": "FirstOrder",
    },
]


def _dist_type(cmt: int) -> str:
    return "OneCmt" if cmt == 1 else "TwoCmt"


def _make_case(entry: dict[str, str | int]) -> BenchmarkCase:
    """Build a BenchmarkCase from a grid entry."""
    ds_id = str(entry["id"])
    cmt = int(entry["cmt"])
    elim = str(entry["elim"])
    abs_type = str(entry["abs"])

    return BenchmarkCase(
        case_id=f"a_ext_{ds_id}",
        suite="A_external",
        dataset_id="nlmixr2data_schoemaker",
        description=(
            f"Schoemaker grid: {ds_id} — {cmt}-CMT, {entry['route']}, {elim} elimination"
        ),
        lane="submission",
        policy_file="submission.json",
        expected_structure=ExpectedStructure(
            absorption=abs_type if abs_type != "none" else None,
            distribution=_dist_type(cmt),
            elimination=elim,
            n_compartments=cmt,
        ),
        ci_cadence="weekly",
    )


# All 12 cases
ALL_CASES: list[BenchmarkCase] = [_make_case(e) for e in _GRID]

# Subset for nightly runs (one per route)
NIGHTLY_CASES: list[BenchmarkCase] = [
    c
    for c in ALL_CASES
    if c.case_id
    in (
        "a_ext_bolus_1cpt",
        "a_ext_infusion_2cptmm",
        "a_ext_oral_2cpt",
    )
]

# Quick subset for CI smoke test
CI_SMOKE_CASES: list[BenchmarkCase] = [c for c in ALL_CASES if c.case_id == "a_ext_oral_1cpt"]

__all__ = ["ALL_CASES", "CI_SMOKE_CASES", "NIGHTLY_CASES"]
