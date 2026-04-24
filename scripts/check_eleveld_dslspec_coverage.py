#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check whether DSLSpec primitives can express the Eleveld 2018 propofol model.

Plan Task 42 — read this report before adding ``opentci_propofol`` to
Phase-1 Bayesian fixtures. Exits non-zero when any *blocking* gap is
detected so CI / preflight scripts can short-circuit.

Usage:
    uv run python scripts/check_eleveld_dslspec_coverage.py [--verbose]

The full discussion lives in ``docs/discovery/eleveld_propofol_coverage.md``;
this script is the auditable counterpart.
"""

from __future__ import annotations

import argparse
import sys
import typing
from dataclasses import dataclass

from apmode.dsl.ast_models import CovariateLink

# Eleveld 2018 covariate effects on the typical-value parameters.
# (parameter, covariate, dsl_form_required, blocking, note)
_ELEVELD_COVARIATE_REQUIREMENTS: list[tuple[str, str, str, bool, str]] = [
    ("V1", "FFM", "power", True, "Fat-free mass is a derived covariate (weight + sex + height)"),
    ("V2", "FFM", "power", True, "Same FFM derivation"),
    ("V2", "AGE", "linear", False, "Linear age effect on top of allometric scaling"),
    ("V3", "FFM", "power", True, "Same FFM derivation"),
    ("V3", "SEX", "categorical", False, "Sex categorical multiplier"),
    ("CL", "FFM", "power", True, "Same FFM derivation"),
    ("CL", "PMA", "maturation", True, "Hill function — Stan emitter raises NotImplementedError"),
    ("CL", "AGE", "decay", True, "Piecewise age decay above adulthood — no DSL primitive"),
    ("CL", "OPIOID", "categorical", False, "Categorical opioid multiplier"),
    ("Q2", "FFM", "power", True, "Same FFM derivation"),
    ("Q3", "FFM", "power", True, "Same FFM derivation"),
    (
        "KE0",
        "WT",
        "power",
        False,
        "Allometric on weight (PD path — also needs effect-site compartment)",
    ),
]

_DSL_AVAILABLE_FORMS: frozenset[str] = frozenset(
    typing.get_args(CovariateLink.model_fields["form"].annotation)
)


@dataclass(frozen=True)
class CoverageGap:
    parameter: str
    covariate: str
    required_form: str
    blocking: bool
    note: str


def find_gaps() -> list[CoverageGap]:
    gaps: list[CoverageGap] = []
    for param, cov, form, blocking, note in _ELEVELD_COVARIATE_REQUIREMENTS:
        # Two failure modes: (a) the form isn't a valid DSL form at all,
        # or (b) the form is valid but a backend won't emit it. The
        # nlmixr2 emitter handles every form; the Stan emitter raises
        # NotImplementedError for ``maturation``. Per-backend coverage
        # is reflected in ``blocking``.
        if form not in _DSL_AVAILABLE_FORMS or blocking:
            gaps.append(
                CoverageGap(
                    parameter=param,
                    covariate=cov,
                    required_form=form,
                    blocking=blocking,
                    note=note,
                )
            )
    return gaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true", help="Print non-blocking gaps too")
    args = ap.parse_args()

    gaps = find_gaps()
    blocking_gaps = [g for g in gaps if g.blocking]
    nonblocking_gaps = [g for g in gaps if not g.blocking]

    print("Eleveld 2018 propofol — DSLSpec coverage check")
    print("=" * 60)
    print(f"DSL available CovariateLink.form values: {sorted(_DSL_AVAILABLE_FORMS)}")
    print(f"Total Eleveld covariate effects checked:  {len(_ELEVELD_COVARIATE_REQUIREMENTS)}")
    print(f"Blocking gaps (cannot proceed):           {len(blocking_gaps)}")
    print(f"Non-blocking gaps (workaround exists):    {len(nonblocking_gaps)}")
    print()

    if blocking_gaps:
        print("BLOCKING GAPS:")
        for g in blocking_gaps:
            print(f"  - {g.parameter} <- {g.covariate} ({g.required_form}): {g.note}")
        print()
    if args.verbose and nonblocking_gaps:
        print("NON-BLOCKING GAPS (informational):")
        for g in nonblocking_gaps:
            print(f"  - {g.parameter} <- {g.covariate} ({g.required_form}): {g.note}")
        print()

    if blocking_gaps:
        print("RECOMMENDATION: NO-GO for v0.6 Phase-1 Bayesian fixtures.")
        print("Defer Eleveld to discovery work; ship vancomycin only.")
        print("See docs/discovery/eleveld_propofol_coverage.md for full discussion.")
        return 1

    print("RECOMMENDATION: GO — every required DSL primitive is available.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
