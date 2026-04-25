# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite B scorer + CLI — reads ``suite_b_results.json``, produces a scorecard.

Mirrors :mod:`apmode.benchmarks.suite_c_phase1_cli`: a CI-friendly,
non-live entry point that consumes whatever the live runner emitted
and produces a pass/fail gate plus a markdown summary suitable for
appending to the GitHub Actions step summary.

Gate rules (data-driven defaults; override via CLI flags):

* ``min_convergence_rate``: per-case convergence ≥ this fraction across
  the seeds the runner attempted. 0.8 by default.
* ``max_cross_seed_cv``: per-case max-parameter cross-seed coefficient
  of variation ≤ this fraction (PRD §5 / R8 monitor). 0.5 by default —
  50% CV is the "wide but not absurd" floor; tightening this requires
  more seeds per case.

Skipped cases (e.g. NODE-backed B1-B3) are reported but excluded from
the gate maths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

_EXIT_OK: int = 0
_EXIT_VALIDATION: int = 3
_EXIT_GATE_MISS: int = 1


# The runner emits ``Any``-shaped JSON; we narrow at the boundary
# (``_score``) so the scorecard structures are typed concretely.
_RawPayload = dict[str, Any]
_ScoreRow = dict[str, Any]
_Scorecard = dict[str, Any]


def _load_results(path: Path) -> dict[str, _RawPayload]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        msg = f"{path} did not parse as a top-level JSON object"
        raise ValueError(msg)
    return cast("dict[str, _RawPayload]", raw)


def _score(
    results: dict[str, _RawPayload],
    *,
    min_convergence_rate: float,
    max_cross_seed_cv: float,
) -> _Scorecard:
    per_case: list[_ScoreRow] = []
    pass_count = 0
    eligible = 0
    skipped = 0
    for case_id, payload in sorted(results.items()):
        if bool(payload.get("skipped")):
            skipped += 1
            per_case.append(
                {
                    "case_id": case_id,
                    "status": "skipped",
                    "reason": payload.get("skip_reason"),
                }
            )
            continue

        eligible += 1
        raw_conv = payload.get("convergence_rate") or 0.0
        conv = float(cast("float", raw_conv))
        raw_cv = payload.get("cross_seed_cv_max")
        cv_value: float | None = (
            float(cast("float", raw_cv)) if isinstance(raw_cv, (int, float)) else None
        )
        cv_pass = cv_value is None or cv_value <= max_cross_seed_cv
        conv_pass = conv >= min_convergence_rate
        passed = conv_pass and cv_pass
        if passed:
            pass_count += 1
        raw_seeds = payload.get("n_seeds") or 0
        per_case.append(
            {
                "case_id": case_id,
                "status": "pass" if passed else "fail",
                "convergence_rate": conv,
                "convergence_pass": conv_pass,
                "cross_seed_cv_max": cv_value,
                "cross_seed_pass": cv_pass,
                "n_seeds": int(cast("int", raw_seeds)),
            }
        )

    pass_fraction = pass_count / eligible if eligible else 0.0
    return {
        "passes_gate": eligible > 0 and pass_count == eligible,
        "n_eligible": eligible,
        "n_passed": pass_count,
        "n_skipped": skipped,
        "pass_fraction": pass_fraction,
        "min_convergence_rate": min_convergence_rate,
        "max_cross_seed_cv": max_cross_seed_cv,
        "per_case": per_case,
    }


def _markdown(scorecard: _Scorecard) -> str:
    lines = ["# Suite B scorecard", ""]
    lines.append(
        f"- Eligible cases: **{scorecard['n_eligible']}**, passed: **{scorecard['n_passed']}**, "
        f"skipped: **{scorecard['n_skipped']}**"
    )
    lines.append(
        f"- Gate: convergence >= {scorecard['min_convergence_rate']:.0%} AND "
        f"cross-seed CV <= {scorecard['max_cross_seed_cv']:.0%}"
    )
    verdict = "PASS" if scorecard["passes_gate"] else "MISS"
    lines.append(f"- Verdict: **{verdict}**")
    lines.append("")
    lines.append("| case | status | convergence | cross-seed CV max | seeds |")
    lines.append("|---|---|---|---|---|")
    rows: list[_ScoreRow] = scorecard["per_case"]
    for row in rows:
        if row["status"] == "skipped":
            lines.append(f"| {row['case_id']} | skipped | - | - | - |")
            continue
        cv = row.get("cross_seed_cv_max")
        cv_str = f"{cv:.3f}" if isinstance(cv, (int, float)) else "NA"
        lines.append(
            f"| {row['case_id']} | {row['status']} | "
            f"{row['convergence_rate']:.2f} | {cv_str} | {row['n_seeds']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apmode.benchmarks.suite_b_cli",
        description="Score the Suite B results JSON written by suite_b_runner.py.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("benchmarks/suite_b/suite_b_results.json"),
        help="Path to the suite_b_results.json the runner emitted.",
    )
    parser.add_argument("--out", type=Path, default=Path("scorecard.json"))
    parser.add_argument("--markdown-summary", type=Path, default=Path("scorecard.md"))
    parser.add_argument("--min-convergence-rate", type=float, default=0.8)
    parser.add_argument("--max-cross-seed-cv", type=float, default=0.5)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.inputs.is_file():
        sys.stderr.write(f"error: inputs file {args.inputs} does not exist\n")
        return _EXIT_VALIDATION

    try:
        raw = _load_results(args.inputs)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: failed to parse {args.inputs}: {exc}\n")
        return _EXIT_VALIDATION

    scorecard = _score(
        raw,
        min_convergence_rate=args.min_convergence_rate,
        max_cross_seed_cv=args.max_cross_seed_cv,
    )

    args.out.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    args.markdown_summary.write_text(_markdown(scorecard))

    if not scorecard["passes_gate"]:
        sys.stderr.write("info: Suite B gate missed — see scorecard.md\n")
        return _EXIT_GATE_MISS

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
