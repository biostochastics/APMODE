# SPDX-License-Identifier: GPL-2.0-or-later
"""``python -m apmode.benchmarks.suite_c_phase1_cli`` — score the Suite C roster.

Run manually/on-demand by an operator after regenerating
``benchmarks/suite_c/phase1_npe_inputs.json`` via
``python -m apmode.benchmarks.suite_c_phase1_runner`` (see
``benchmarks/suite_c/README.md``). There is no scheduled CI job for
this — an earlier weekly cron workflow
(``.github/workflows/suite_c_phase1.yml``) was removed because it only
re-scored the static committed JSON (pure arithmetic, no R, no
``nlmixr2``, no live fit), which created a false impression of ongoing
live validation without actually re-running anything. The CLI reads a
JSON file mapping ``fixture_id`` → ``{npe_apmode,
npe_literature}``, computes the per-fixture
:class:`~apmode.benchmarks.suite_c_phase1_scoring.FixtureScore`,
aggregates them into a
:class:`~apmode.benchmarks.suite_c_phase1_scoring.SuiteCPhase1Scorecard`,
writes a machine-readable JSON scorecard, and (optionally) renders a
human-readable Markdown summary suitable for a PR description, an
issue body, or ad hoc CI use.

Why a separate CLI module rather than a Typer subcommand on
``apmode.cli``: an on-demand/CI run of just the scoring math needs a
vanilla ``uv sync --extra dev`` (no R, no cmdstan); routing through
``apmode.cli`` would pull in ``Nlmixr2Runner`` import-side imports and
surface a less obvious "R not found" failure when ``Rscript`` happens
to be missing. A standalone ``python -m`` entry point keeps the
dependency surface minimal.

Exit codes:
  * ``0`` — scorecard written successfully (gate result is in the JSON
    + Markdown).
  * ``2`` — usage error (bad CLI arguments, missing inputs file,
    malformed JSON).
  * ``3`` — at least one fixture's NPE values failed validation
    (negative or non-finite). Exit non-zero rather than silently
    falsifying the scorecard.
  * ``4`` — only emitted when ``--fail-on-missed-gate`` is supplied
    AND ``passes_gate`` is False. Lets a caller (e.g. an ad hoc CI
    job) request a hard failure on regression. Without the flag, a
    missed gate still exits 0 (the gate is reported in JSON for
    downstream consumers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from apmode.benchmarks.suite_c_phase1_scoring import (
    DEFAULT_STALE_WARN_DAYS,
    PROVENANCE_KEY,
    FixtureScore,
    SuiteCPhase1Provenance,
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
    runner to carry the raw per-fold NPE values whose
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
        if fid == PROVENANCE_KEY:
            # Reserved top-level provenance block — lifted out by
            # _extract_provenance, not a fixture. Skip so it is never
            # mis-parsed as a fixture entry.
            continue
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
        # PIT/NPDE-lite calibration is informational only — pass through
        # verbatim (no gate reads it) so render_markdown_summary can
        # display it. Optional: absent in inputs files written before
        # this field existed.
        for pit_key in ("pit_calibration_apmode", "pit_calibration_literature"):
            pit_val = payload.get(pit_key)
            if pit_val is not None:
                if not isinstance(pit_val, dict):
                    msg = f"inputs entry for fixture {fid!r}: {pit_key!r} must be an object"
                    raise TypeError(msg)
                entry[pit_key] = {str(k): float(v) for k, v in pit_val.items()}
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
# Provenance
# ---------------------------------------------------------------------------


def _extract_provenance(path: Path) -> SuiteCPhase1Provenance:
    """Lift the reserved ``_provenance`` block out of the inputs file.

    Fails loud (raises) when the block is absent: Suite C scoring runs over
    a *static committed snapshot*, so a Markdown summary that omitted the
    provenance signal could be mistaken for live validation. ``snapshot_source``
    defaults to the inputs path when the block does not carry its own.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        msg = f"inputs JSON root must be an object, got {type(raw).__name__}"
        raise TypeError(msg)
    block = raw.get(PROVENANCE_KEY)
    if block is None:
        msg = (
            f"inputs file {path} has no {PROVENANCE_KEY!r} block; refusing to render a "
            "Markdown summary that could be mistaken for live validation. Add a "
            f"{PROVENANCE_KEY!r} object with generated_at / snapshot_source / live_runner "
            "(see benchmarks/suite_c/README.md)."
        )
        raise ValueError(msg)
    if not isinstance(block, dict):
        msg = f"{PROVENANCE_KEY!r} must be an object, got {type(block).__name__}"
        raise TypeError(msg)
    data: dict[str, object] = {str(k): v for k, v in block.items()}
    data.setdefault("snapshot_source", str(path))
    return SuiteCPhase1Provenance.model_validate(data)


# ---------------------------------------------------------------------------
# Markdown summary rendering
# ---------------------------------------------------------------------------


def _format_pit_cell(calibration: object) -> str:
    """Render a PIT calibration dict as ``p5=.05 p50=.49 p95=.94``, or a dash if absent."""
    if not isinstance(calibration, dict) or not calibration:
        return "—"
    return " ".join(f"{k}={v:.2f}" for k, v in sorted(calibration.items()))


def render_markdown_summary(
    card: SuiteCPhase1Scorecard,
    inputs: dict[str, dict[str, object]] | None = None,
    *,
    provenance: SuiteCPhase1Provenance | None = None,
    stale_warn_days: float = DEFAULT_STALE_WARN_DAYS,
    now: datetime | None = None,
) -> str:
    """Render the scorecard as a Markdown table + headline.

    Suitable for a PR description, an issue body, or a
    ``$GITHUB_STEP_SUMMARY`` in an ad hoc CI invocation. The table body
    is kept deterministic (no wall-clock timestamps) so the artifact
    diff is meaningful run-to-run; ``provenance`` timestamps come from
    data, not :func:`datetime.now`.

    ``provenance`` is **required** — passing ``None`` raises. Suite C
    scoring runs over a static committed snapshot, so a summary rendered
    without a provenance signal could be mistaken for live validation. A
    prominent ``⚠ STALE / NON-LIVE SNAPSHOT`` banner is emitted whenever
    the snapshot is non-live (``live_runner=False``) OR older than
    ``stale_warn_days``; a fresh, live snapshot omits it. ``now`` is
    injectable for deterministic tests (defaults to the current UTC time,
    used only to decide whether to warn — never printed).

    ``inputs`` (the raw ``_load_inputs`` map) is optional and, when
    supplied, adds PIT/NPDE-lite calibration columns per fixture — this
    is informational only (not part of ``FixtureScore``/the win/loss
    gate) so it lives here rather than on the scorecard model itself.
    Absent for inputs files written before this field existed, or when
    the caller doesn't have the raw inputs map on hand.
    """
    if provenance is None:
        msg = (
            "render_markdown_summary requires provenance; refusing to render a Suite C "
            "scorecard as if it were live validation. Pass a SuiteCPhase1Provenance "
            "describing the snapshot (generated_at, snapshot_source, live_runner)."
        )
        raise ValueError(msg)

    ref_now = now if now is not None else datetime.now(UTC)
    stale_reason = provenance.staleness_reason(now=ref_now, stale_warn_days=stale_warn_days)

    lines: list[str] = ["# Suite C Phase-1 scorecard", ""]
    if stale_reason is not None:
        lines.extend(
            [
                "> ⚠ **STALE / NON-LIVE SNAPSHOT** — this scorecard was scored from a "
                "static committed snapshot, not a live fit. Do NOT read it as live "
                "validation.",
                ">",
                f"> Generated at `{provenance.generated_at}` · source "
                f"`{provenance.snapshot_source}` · git `{provenance.git_sha or 'unknown'}` "
                f"· live_runner `{provenance.live_runner}`.",
                f"> Reason: {stale_reason}",
                "",
            ]
        )
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
    has_pit = inputs is not None
    header = "| Fixture | NPE APMODE | NPE Literature | Beats? |"
    divider = "| --- | ---: | ---: | :---: |"
    if has_pit:
        header += " PIT APMODE | PIT Literature |"
        divider += " --- | --- |"
    lines.extend(["", header, divider])
    for s in card.scores:
        beats = ":white_check_mark:" if s.beats_literature else ":x:"
        row = f"| `{s.fixture_id}` | {s.npe_apmode:.4f} | {s.npe_literature:.4f} | {beats} |"
        if has_pit:
            entry = (inputs or {}).get(s.fixture_id, {})
            row += (
                f" {_format_pit_cell(entry.get('pit_calibration_apmode'))} "
                f"| {_format_pit_cell(entry.get('pit_calibration_literature'))} |"
            )
        lines.append(row)
    if has_pit:
        lines.extend(
            [
                "",
                "_PIT/NPDE-lite calibration (`p{level}=observed coverage`; should be "
                "close to the nominal level, e.g. p50≈0.50) is a calibration diagnostic, "
                "distinct from the NPE point-accuracy comparison above — see "
                "`benchmarks/suite_c/README.md`._",
            ]
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
        "--stale-warn-days",
        type=float,
        default=DEFAULT_STALE_WARN_DAYS,
        help=(
            "Age in days beyond which even a live snapshot is flagged as stale in "
            f"the Markdown banner (default {DEFAULT_STALE_WARN_DAYS:g}). A non-live "
            "snapshot (live_runner=false) always gets the banner regardless of age."
        ),
    )
    parser.add_argument(
        "--fail-on-missed-gate",
        action="store_true",
        default=False,
        help=(
            "Exit non-zero (code 4) when passes_gate=false. "
            "Useful for callers that want a hard failure on regression "
            "rather than just reporting the gate result in the JSON output."
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
        try:
            provenance = _extract_provenance(args.inputs)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            sys.stderr.write(f"error: cannot render Markdown summary: {exc}\n")
            return 2
        args.markdown_summary.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            args.markdown_summary,
            render_markdown_summary(
                card,
                inputs,
                provenance=provenance,
                stale_warn_days=args.stale_warn_days,
            ),
        )

    if args.fail_on_missed_gate and not card.passes_gate:
        sys.stderr.write("error: gate missed (passes_gate=false)\n")
        return 4

    return 0


def _atomic_write(target: Path, content: str) -> None:
    """Write ``content`` to ``target`` via tmp-file + rename.

    A SIGKILL (or OOM) mid-write leaves either the previous version
    intact or the tmp file orphaned next to the target — never a
    half-written scorecard that a downstream consumer (e.g. an issue
    body or step summary) would mis-render. The PID-suffixed tmp name
    makes concurrent
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
