#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Replay Gate 1/2 over sealed-bundle results and emit a calibration report.

Thin entrypoint over ``apmode.governance.calibration`` (see
docs/plans/2026-07-09-qaqc-remediation.md, "Empirically calibrate
Gate1/2/3 thresholds against Suite A/B false-pass/fail rates", Task 5).

This script does *not* orchestrate new R/nlmixr2 fits — it consumes
already-sealed bundles' ``results/{candidate_id}_result.json`` artifacts,
labels each fit against ``benchmarks/suite_a/reference_params.json``
ground truth (via ``apmode.governance.calibration.label_fit``), replays
it through the real Gate 1/2 evaluators, and writes a versioned
``GateCalibrationReport`` under ``benchmarks/calibration/reports/`` by
default.

A candidate is matched to a Suite A scenario by a leading ``A<digits>``
(or ``B<digits>``) token in its ``model_id`` (e.g. ``"A1_candidate_007"``
-> scenario ``"A1"``); candidates that don't match a known scenario id
are skipped, not errored, since a bundle directory may contain candidates
outside the calibration harness's scope (report an accurate coverage
count rather than crash on unrelated bundles).

This report is a *reference* for policy authors setting
``policies/*.json`` thresholds — it does not auto-tune them.

Usage:
    uv run python scripts/run_gate_calibration.py 'runs/*' \\
        --policy submission --gate 1 --out-dir benchmarks/calibration/reports
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from apmode.bundle.models import BackendResult  # noqa: E402
from apmode.governance.calibration import (  # noqa: E402
    LabeledFit,
    aggregate_calibration,
    label_fit,
    replay_gate1,
    replay_gate2,
    write_calibration_report,
)
from apmode.governance.policy import GatePolicy  # noqa: E402

_SCENARIO_ID_RE = re.compile(r"^(?P<scenario>[AB]\d+)")

# Wilson/Clopper-Pearson CIs are deliberately out of scope for the per-check
# rates in this report (Task 3 note): at Suite A/B's expected small n
# (~21 scenarios), a per-check CI is wide enough to be easy to over-read as
# tighter evidence than it is. Flagged here, not computed.
_CI_DEFERRED_NOTE = (
    "Wilson/Clopper-Pearson confidence intervals are deliberately not "
    "computed for false_pass_rate/false_fail_rate: at Suite A/B's small "
    "per-scenario n, per-check CIs are wide and easy to over-read as "
    "tighter evidence than they are. Treat rates as directional signal."
)


def _load_reference_params(path: Path) -> dict[str, dict[str, object]]:
    data: dict[str, object] = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def _extract_recovered_params(result: BackendResult) -> dict[str, float]:
    return {name: est.estimate for name, est in result.parameter_estimates.items()}


def _iter_bundle_result_files(bundles_glob: str) -> list[Path]:
    paths: list[Path] = []
    for bundle_dir in sorted(glob.glob(bundles_glob)):
        bundle_path = Path(bundle_dir)
        results_dir = bundle_path / "results"
        if results_dir.is_dir():
            paths.extend(sorted(results_dir.glob("*_result.json")))
    return paths


def build_labeled_fits(
    bundles_glob: str,
    reference_params: dict[str, dict[str, object]],
) -> list[LabeledFit]:
    """Extract, label, and collect fits from every matching bundle's results/.

    Candidates whose ``model_id`` doesn't carry a recognizable Suite A/B
    scenario prefix, or whose scenario id has no reference-params entry,
    are silently skipped (not a calibration harness error — the bundle may
    legitimately contain non-benchmark candidates).
    """
    fits: list[LabeledFit] = []
    for result_path in _iter_bundle_result_files(bundles_glob):
        result = BackendResult.model_validate_json(result_path.read_text())
        match = _SCENARIO_ID_RE.match(result.model_id)
        if match is None:
            continue
        scenario_id = match.group("scenario")
        reference = reference_params.get(scenario_id)
        if reference is None:
            continue
        recovered = _extract_recovered_params(result)
        label = label_fit(scenario_id=scenario_id, recovered_params=recovered, reference=reference)
        fits.append(LabeledFit(scenario_id=scenario_id, result=result, label=label))
    return fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "bundles_glob",
        help=(
            "Glob matching sealed bundle directories, e.g. 'runs/*' "
            "(quote to avoid shell expansion)"
        ),
    )
    parser.add_argument(
        "--policy",
        default="submission",
        help="Lane policy name under policies/ (default: submission)",
    )
    parser.add_argument(
        "--gate",
        choices=["1", "2"],
        default="1",
        help="Which gate to replay (default: 1)",
    )
    parser.add_argument(
        "--reference-params",
        default=str(_REPO_ROOT / "benchmarks" / "suite_a" / "reference_params.json"),
        help="Path to Suite A reference_params.json",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "benchmarks" / "calibration" / "reports"),
        help="Output directory for the calibration report",
    )
    args = parser.parse_args(argv)

    policy_path = _REPO_ROOT / "policies" / f"{args.policy}.json"
    policy = GatePolicy.model_validate(json.loads(policy_path.read_text()))

    reference_params = _load_reference_params(Path(args.reference_params))
    fits = build_labeled_fits(args.bundles_glob, reference_params)

    if not fits:
        print(
            f"No labelable candidates found under glob {args.bundles_glob!r} "
            "(no results/*_result.json with a recognizable Suite A/B scenario "
            "prefix in model_id).",
            file=sys.stderr,
        )
        return 1

    if args.gate == "1":
        verdicts = replay_gate1(fits, policy)
        gate_id = "gate1"
    else:
        verdicts = replay_gate2(fits, policy, args.policy)
        gate_id = "gate2"

    stats = aggregate_calibration(verdicts)
    n_good = len({fit.scenario_id for fit in fits if fit.label == "known_good"})
    n_bad = len({fit.scenario_id for fit in fits if fit.label == "known_bad"})

    out_path = write_calibration_report(
        stats,
        policy_version=policy.policy_version,
        out_dir=Path(args.out_dir),
        gate_id=gate_id,
        n_scenarios_known_good=n_good,
        n_scenarios_known_bad=n_bad,
        notes=[_CI_DEFERRED_NOTE],
    )
    print(f"Wrote calibration report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
