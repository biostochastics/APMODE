# SPDX-License-Identifier: GPL-2.0-or-later
"""``python -m apmode.benchmarks.suite_c_phase1_cli`` — score the Phase-1 roster.

Plan Task 41 — driver invoked by ``.github/workflows/suite_c_phase1.yml``.
The CLI reads a JSON file mapping ``fixture_id`` → ``{npe_apmode,
npe_literature}``, computes the per-fixture
:class:`~apmode.benchmarks.suite_c_phase1_scoring.FixtureScore`,
aggregates them into a
:class:`~apmode.benchmarks.suite_c_phase1_scoring.SuiteCPhase1Scorecard`,
writes a machine-readable JSON scorecard, and (optionally) renders a
human-readable Markdown summary for the GitHub Actions
``$GITHUB_STEP_SUMMARY`` and the failure-issue body.

Why a separate CLI module rather than a Typer subcommand on
``apmode.cli``: the weekly workflow runs on a vanilla ``uv sync --extra
dev`` (no R, no cmdstan); routing through ``apmode.cli`` would pull in
``Nlmixr2Runner`` import-side imports and surface a less obvious
"R not found" failure when ``Rscript`` happens to be missing on the
runner. A standalone ``python -m`` entry point keeps the dependency
surface minimal.

Exit codes:
  * ``0`` — scorecard written successfully (gate result is in the JSON
    + Markdown; the workflow reads ``passes_gate`` to decide whether
    to open an issue).
  * ``2`` — usage error (bad CLI arguments, missing inputs file,
    malformed JSON).
  * ``3`` — at least one fixture's NPE values failed validation
    (negative or non-finite). Exit non-zero so the workflow surfaces
    a hard error rather than silently falsifying the scorecard.
  * ``4`` — only emitted when ``--fail-on-missed-gate`` is supplied
    AND ``passes_gate`` is False. Lets the same CLI back per-PR jobs
    that want a hard failure on regression instead of deferring to the
    weekly workflow's open-issue path. Without the flag, a missed gate
    still exits 0 (the gate is reported in JSON for downstream consumers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.benchmarks.suite_c_phase1_scoring import (
    FixtureScore,
    SuiteCPhase1Scorecard,
    aggregate_phase1_scorecard,
    score_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Inputs file helpers
# ---------------------------------------------------------------------------


def _load_inputs(path: Path) -> dict[str, dict[str, object]]:
    """Read the inputs JSON and return ``{fixture_id: payload}`` per fixture.

    The expected payload shape:

    .. code-block:: json

        {
          "theophylline_boeckmann_1992": {
            "npe_apmode": 0.95,
            "npe_literature": 1.00,
            "npe_apmode_per_fold": [0.93, 0.96, 0.94, 0.97, 0.95]
          },
          "warfarin_funaki_2018": {"npe_apmode": 0.99, "npe_literature": 1.00}
        }

    ``npe_apmode_per_fold`` is optional — supplied by the live-fit
    runner (plan Task 44) to carry the raw per-fold NPE values whose
    median is ``npe_apmode``. Inputs JSON files written before the
    runner landed are still accepted unchanged. Extra unrelated fields
    per entry are tolerated for forward compat.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        msg = f"inputs JSON root must be an object, got {type(raw).__name__}"
        raise TypeError(msg)
    out: dict[str, dict[str, object]] = {}
    for fid, payload in raw.items():
        if not isinstance(payload, dict):
            msg = f"inputs entry for fixture {fid!r} must be an object"
            raise TypeError(msg)
        if "npe_apmode" not in payload or "npe_literature" not in payload:
            msg = (
                f"inputs entry for fixture {fid!r} must include 'npe_apmode' and "
                "'npe_literature' keys"
            )
            raise KeyError(msg)
        entry: dict[str, object] = {
            "npe_apmode": float(payload["npe_apmode"]),
            "npe_literature": float(payload["npe_literature"]),
        }
        per_fold = payload.get("npe_apmode_per_fold")
        if per_fold is not None:
            if not isinstance(per_fold, list):
                msg = (
                    f"inputs entry for fixture {fid!r}: "
                    "'npe_apmode_per_fold' must be a list when present"
                )
                raise TypeError(msg)
            entry["npe_apmode_per_fold"] = tuple(float(x) for x in per_fold)
        out[fid] = entry
    return out


def _score_all(inputs: dict[str, dict[str, object]]) -> list[FixtureScore]:
    """Score every fixture in the inputs map. Caller-stable order."""
    scores: list[FixtureScore] = []
    for fid, payload in inputs.items():
        per_fold_raw = payload.get("npe_apmode_per_fold")
        per_fold: tuple[float, ...] | None = (
            per_fold_raw if isinstance(per_fold_raw, tuple) else None
        )
        scores.append(
            score_fixture(
                fixture_id=fid,
                npe_apmode=float(payload["npe_apmode"]),  # type: ignore[arg-type]
                npe_literature=float(payload["npe_literature"]),  # type: ignore[arg-type]
                npe_apmode_per_fold=per_fold,
            )
        )
    return scores


# ---------------------------------------------------------------------------
# Markdown summary rendering
# ---------------------------------------------------------------------------


def render_markdown_summary(card: SuiteCPhase1Scorecard) -> str:
    """Render the scorecard as a Markdown table + headline.

    Used by the workflow to populate ``$GITHUB_STEP_SUMMARY`` and the
    body of the auto-opened GitHub issue when the gate misses. Kept
    deterministic (no timestamps) so the artifact diff is meaningful
    week-on-week.
    """
    lines: list[str] = ["# Suite C Phase-1 scorecard", ""]
    if card.fraction_beats_literature_median is None:
        lines.append(
            f"**Fraction beating literature**: not computed "
            f"(< {len(card.scores)} fixtures < min {card.target * 100:.0f}%)"
        )
    else:
        emoji = ":white_check_mark:" if card.passes_gate else ":x:"
        lines.append(
            f"**Fraction beating literature**: "
            f"{card.fraction_beats_literature_median:.0%} "
            f"({card.n_beats}/{card.n_datasets}) — target "
            f"{card.target:.0%} {emoji}"
        )
    lines.extend(
        [
            "",
            "| Fixture | NPE APMODE | NPE Literature | Beats? |",
            "| --- | ---: | ---: | :---: |",
        ]
    )
    for s in card.scores:
        beats = ":white_check_mark:" if s.beats_literature else ":x:"
        lines.append(
            f"| `{s.fixture_id}` | {s.npe_apmode:.4f} | {s.npe_literature:.4f} | {beats} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m apmode.benchmarks.suite_c_phase1_cli",
        description=(
            "Score the Phase-1 Suite C roster from a JSON inputs file "
            "and emit a SuiteCPhase1Scorecard."
        ),
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON file with {fixture_id: {npe_apmode, npe_literature}} entries.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination path for the JSON scorecard.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        default=None,
        help=(
            "Optional path for a Markdown rendering of the scorecard "
            "(used by the GitHub Actions step summary)."
        ),
    )
    parser.add_argument(
        "--fail-on-missed-gate",
        action="store_true",
        default=False,
        help=(
            "Exit non-zero (code 4) when passes_gate=false. "
            "Useful for per-PR jobs that want a hard failure instead of "
            "deferring to the weekly workflow's open-issue path."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.inputs.is_file():
        sys.stderr.write(f"error: inputs file not found: {args.inputs}\n")
        return 2

    try:
        inputs = _load_inputs(args.inputs)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"error: failed to parse inputs file: {exc}\n")
        return 2

    try:
        scores = _score_all(inputs)
    except ValueError as exc:
        sys.stderr.write(f"error: fixture score validation failed: {exc}\n")
        return 3

    card = aggregate_phase1_scorecard(scores)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.out, card.model_dump_json(indent=2) + "\n")

    if args.markdown_summary is not None:
        args.markdown_summary.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(args.markdown_summary, render_markdown_summary(card))

    if args.fail_on_missed_gate and not card.passes_gate:
        sys.stderr.write("error: gate missed (passes_gate=false)\n")
        return 4

    return 0


def _atomic_write(target: Path, content: str) -> None:
    """Write ``content`` to ``target`` via tmp-file + rename.

    A SIGKILL (or OOM) mid-write leaves either the previous version
    intact or the tmp file orphaned next to the target — never a
    half-written scorecard the workflow's `gh issue create` step
    would mis-render. The PID-suffixed tmp name makes concurrent
    invocations unlikely to collide; the rename is atomic on the
    same filesystem (the only mode the CLI is exercised in — both
    the local invocation and the runner write under the same tmp /
    repo workspace).
    """
    tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(content)
    tmp.replace(target)


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in CI
    raise SystemExit(main())
