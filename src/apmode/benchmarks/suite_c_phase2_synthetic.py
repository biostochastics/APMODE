# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite C Phase 2 — synthetic methodology-validation panel.

PRD §5 reserves the headline metric ``fraction-beats-median-expert >= 60%``
for a blinded human-expert head-to-head benchmark. That panel requires
external collaborator coordination and is **out of v0.6 scope**. To keep
the metric exercised end-to-end before real experts arrive, this module
synthesises an "expert" panel from log-normal perturbations of the
fixture's literature anchor and applies the same
:func:`apmode.benchmarks.scoring.evaluate_expert_comparison` /
:func:`compute_fraction_beats_expert` plumbing the real panel will use.

NEVER present the resulting numbers as a real-expert benchmark. The
scorecard schema and the markdown summary explicitly tag every output
with ``synthetic=True`` and the ``methodology_validation_only`` banner so
a reviewer cannot mistake the scaffold for a credibility claim.

The module ships:

* :func:`synthesize_expert_npes` — given a literature anchor NPE,
  returns ``panel_size`` jittered values that imitate an expert panel's
  spread. Pure, deterministic given ``seed``.
* :func:`score_synthetic_phase2` — consumes the same
  ``phase1_npe_inputs.json`` the live Phase-1 runner emits, synthesises
  per-fixture panels, and returns a
  :class:`SyntheticPhase2Scorecard`.
* CLI entry (``python -m apmode.benchmarks.suite_c_phase2_synthetic``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apmode.benchmarks.scoring import (
    compute_fraction_beats_expert,
    evaluate_expert_comparison,
)
from apmode.benchmarks.suite_c import WIN_MARGIN_DELTA

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PANEL_SIZE: int = 5
"""Synthetic experts per fixture. Three is the floor below which the
median is statistically meaningless (matches
``MIN_LITERATURE_COUNT`` in :mod:`apmode.benchmarks.suite_c`); five
matches the PRD §5 ``2-3`` expert guidance with margin so the
median-vs-best decomposition is informative."""

DEFAULT_JITTER_SIGMA: float = 0.20
"""Log-normal sigma applied to the literature NPE to simulate an
expert panel's spread. 0.20 corresponds to a geo-SD of ~22% — wide
enough that the synthetic panel does not trivially lose, narrow
enough that a real APMODE methodology improvement is detectable.
Pin this in policy before changing — drifting sigma per release would
invalidate cross-release comparisons of the synthetic gate."""

DEFAULT_GATE_TARGET: float = 0.60
"""Mirrors PRD §5 primary metric (``>=60%``). Gating against this
target with synthetic experts is purely a methodology check."""

SYNTHETIC_BANNER: str = (
    "**SYNTHETIC METHODOLOGY VALIDATION ONLY** — this panel is a "
    "log-normal perturbation of the published literature anchor; it is "
    "NOT a real-expert benchmark and must not be cited as one. The PRD "
    "§5 fraction-beats-median-expert metric remains gated on a blinded "
    "human-expert panel that is out of v0.6 scope."
)

# Exit codes mirror the Phase-1 CLI for muscle-memory.
_EXIT_OK: int = 0
_EXIT_USAGE: int = 2
_EXIT_VALIDATION: int = 3
_EXIT_GATE_MISS: int = 1


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SyntheticFixtureScore(BaseModel):
    """Per-fixture record for the synthetic Phase 2 scorecard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str = Field(min_length=1)
    npe_apmode: float = Field(gt=0.0)
    npe_literature: float = Field(gt=0.0)
    synthetic_expert_npes: tuple[float, ...] = Field(min_length=1)
    expert_median_npe: float = Field(gt=0.0)
    npe_gap_vs_median: float
    beats_synthetic_median: bool
    panel_size: int = Field(ge=1)
    jitter_sigma: float = Field(gt=0.0)


class SyntheticPhase2Scorecard(BaseModel):
    """Aggregate scorecard for the synthetic Phase 2 panel.

    ``synthetic`` is always ``True`` — keeps any downstream consumer
    that filters by this flag from accidentally treating the scorecard
    as a real-expert benchmark.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthetic: bool = True
    banner: str = SYNTHETIC_BANNER
    scores: list[SyntheticFixtureScore]
    n_datasets: int = Field(ge=0)
    n_beats: int = Field(ge=0)
    fraction_beats_synthetic_median: float | None = Field(default=None, ge=0.0, le=1.0)
    target: float = Field(default=DEFAULT_GATE_TARGET, ge=0.0, le=1.0)
    passes_gate: bool
    panel_size: int = Field(ge=1)
    jitter_sigma: float = Field(gt=0.0)
    win_margin_delta: float = Field(default=WIN_MARGIN_DELTA, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Synthesis math
# ---------------------------------------------------------------------------


def synthesize_expert_npes(
    npe_literature: float,
    *,
    panel_size: int = DEFAULT_PANEL_SIZE,
    jitter_sigma: float = DEFAULT_JITTER_SIGMA,
    seed: int = 20260425,
) -> tuple[float, ...]:
    """Return ``panel_size`` log-normal-jittered NPE values around ``npe_literature``.

    Each "expert" NPE is::

        expert_npe = npe_literature * exp(N(0, jitter_sigma))

    with the jitter draws coming from a numpy RNG seeded by ``seed`` so
    the synthetic panel is reproducible across runs (the gate cannot be
    coin-flipped by re-running). The first sample is intentionally the
    literature anchor itself (jitter = 0) so the panel always contains
    the canonical reference; the remaining ``panel_size - 1`` samples
    are jittered.
    """
    if not math.isfinite(npe_literature) or npe_literature <= 0:
        msg = f"npe_literature must be a positive finite number, got {npe_literature!r}"
        raise ValueError(msg)
    if panel_size < 1:
        msg = f"panel_size must be >= 1, got {panel_size}"
        raise ValueError(msg)
    if jitter_sigma <= 0:
        msg = f"jitter_sigma must be > 0, got {jitter_sigma}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    extra = panel_size - 1
    if extra <= 0:
        return (float(npe_literature),)
    log_jitters = rng.normal(loc=0.0, scale=jitter_sigma, size=extra)
    jittered = (np.exp(log_jitters) * npe_literature).tolist()
    return (float(npe_literature), *(float(v) for v in jittered))


def score_fixture_synthetic(
    *,
    fixture_id: str,
    npe_apmode: float,
    npe_literature: float,
    panel_size: int = DEFAULT_PANEL_SIZE,
    jitter_sigma: float = DEFAULT_JITTER_SIGMA,
    win_margin_delta: float = WIN_MARGIN_DELTA,
    seed: int | None = None,
) -> SyntheticFixtureScore:
    """Score one fixture against a synthesized expert panel.

    ``seed`` defaults to a deterministic hash of ``fixture_id`` so two
    fixtures don't share their jitter draws (same seed across all
    fixtures would correlate the panel idiosyncrasies and make a
    cross-fixture median misleading).
    """
    if seed is None:
        # Seed is a 32-bit hash of the fixture id so every fixture gets
        # an independent jitter draw while staying reproducible.
        seed = int(abs(hash(fixture_id)) % (2**31 - 1))

    panel = synthesize_expert_npes(
        npe_literature,
        panel_size=panel_size,
        jitter_sigma=jitter_sigma,
        seed=seed,
    )
    beats, median_npe, gap = evaluate_expert_comparison(
        apmode_npe=npe_apmode,
        expert_npes=list(panel),
        win_margin=win_margin_delta,
    )
    if beats is None or median_npe is None or gap is None:
        # evaluate_expert_comparison only returns None when the panel
        # is empty; we just synthesised a non-empty panel above so this
        # branch indicates an internal contract violation.
        msg = "synthetic panel was empty after synthesis — internal error"
        raise RuntimeError(msg)
    return SyntheticFixtureScore(
        fixture_id=fixture_id,
        npe_apmode=npe_apmode,
        npe_literature=npe_literature,
        synthetic_expert_npes=panel,
        expert_median_npe=float(median_npe),
        npe_gap_vs_median=float(gap),
        beats_synthetic_median=bool(beats),
        panel_size=panel_size,
        jitter_sigma=jitter_sigma,
    )


def score_synthetic_phase2(
    inputs: dict[str, dict[str, float]],
    *,
    panel_size: int = DEFAULT_PANEL_SIZE,
    jitter_sigma: float = DEFAULT_JITTER_SIGMA,
    win_margin_delta: float = WIN_MARGIN_DELTA,
    target: float = DEFAULT_GATE_TARGET,
    seed_offset: int = 0,
) -> SyntheticPhase2Scorecard:
    """Build a synthetic-panel scorecard from a Phase-1 inputs dict.

    ``inputs`` mirrors the shape of ``phase1_npe_inputs.json`` written
    by the Suite C Phase-1 runner: ``{fixture_id: {npe_apmode,
    npe_literature, ...}}``. Extra fields are ignored.

    ``seed_offset`` is added to the per-fixture deterministic seed so
    callers can sweep alternative jitter realisations without editing
    fixture ids — used in tests to confirm the gate is robust.
    """
    scores: list[SyntheticFixtureScore] = []
    for fixture_id in sorted(inputs):
        payload = inputs[fixture_id]
        npe_apmode = float(payload["npe_apmode"])
        npe_literature = float(payload["npe_literature"])
        seed = int((abs(hash(fixture_id)) % (2**31 - 1)) + seed_offset)
        scores.append(
            score_fixture_synthetic(
                fixture_id=fixture_id,
                npe_apmode=npe_apmode,
                npe_literature=npe_literature,
                panel_size=panel_size,
                jitter_sigma=jitter_sigma,
                win_margin_delta=win_margin_delta,
                seed=seed,
            )
        )

    n_datasets = len(scores)
    n_beats = sum(1 for s in scores if s.beats_synthetic_median)

    case_results = [(s.npe_apmode, list(s.synthetic_expert_npes)) for s in scores]
    fraction = (
        compute_fraction_beats_expert(case_results, win_margin=win_margin_delta)
        if n_datasets > 0
        else None
    )

    return SyntheticPhase2Scorecard(
        scores=scores,
        n_datasets=n_datasets,
        n_beats=n_beats,
        fraction_beats_synthetic_median=fraction,
        target=target,
        passes_gate=fraction is not None and fraction >= target,
        panel_size=panel_size,
        jitter_sigma=jitter_sigma,
        win_margin_delta=win_margin_delta,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_inputs(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        msg = f"{path} did not parse as a top-level JSON object"
        raise ValueError(msg)
    out: dict[str, dict[str, float]] = {}
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            msg = f"inputs[{key!r}] must be an object, got {type(payload).__name__}"
            raise ValueError(msg)
        if "npe_apmode" not in payload or "npe_literature" not in payload:
            msg = (
                f"inputs[{key!r}] missing one of 'npe_apmode' / "
                f"'npe_literature'; got keys {sorted(payload)}"
            )
            raise ValueError(msg)
        out[key] = {
            "npe_apmode": float(payload["npe_apmode"]),
            "npe_literature": float(payload["npe_literature"]),
        }
    return out


def _markdown_summary(scorecard: SyntheticPhase2Scorecard) -> str:
    lines = ["# Suite C Phase 2 — synthetic-panel scorecard", "", scorecard.banner, ""]
    lines.append(
        f"- Panel size: **{scorecard.panel_size}**, "
        f"jitter sigma: **{scorecard.jitter_sigma:.2f}** "
        f"(log-normal multiplier on the literature NPE)."
    )
    fraction = scorecard.fraction_beats_synthetic_median
    if fraction is None:
        lines.append("- Fraction beats synthetic median: **N/A** (no fixtures scored).")
    else:
        lines.append(
            f"- Fraction beats synthetic median: **{fraction:.0%}** "
            f"(target ≥ {scorecard.target:.0%})."
        )
    verdict = "PASS" if scorecard.passes_gate else "MISS"
    lines.append(f"- Verdict: **{verdict}**")
    lines.append("")
    lines.append("| fixture | NPE APMODE | NPE literature | synthetic median | gap | beats? |")
    lines.append("|---|---|---|---|---|---|")
    for s in scorecard.scores:
        lines.append(
            f"| {s.fixture_id} | {s.npe_apmode:.4f} | {s.npe_literature:.4f} | "
            f"{s.expert_median_npe:.4f} | {s.npe_gap_vs_median:+.4f} | "
            f"{'✓' if s.beats_synthetic_median else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apmode.benchmarks.suite_c_phase2_synthetic",
        description=(
            "Score APMODE against a SYNTHETIC expert panel "
            "(log-normal perturbation of the literature anchor). "
            "Methodology validation only — never cite as a real-expert benchmark."
        ),
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("benchmarks/suite_c/phase1_npe_inputs.json"),
        help="Same file the Phase-1 runner / scorer reads.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("synthetic_phase2_scorecard.json"),
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        default=Path("synthetic_phase2_scorecard.md"),
    )
    parser.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    parser.add_argument("--jitter-sigma", type=float, default=DEFAULT_JITTER_SIGMA)
    parser.add_argument(
        "--target",
        type=float,
        default=DEFAULT_GATE_TARGET,
        help="Pass threshold for fraction-beats-synthetic-median.",
    )
    parser.add_argument(
        "--win-margin-delta",
        type=float,
        default=WIN_MARGIN_DELTA,
        help="APMODE NPE must beat the synthetic median by at least this fraction.",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.panel_size < 1:
        sys.stderr.write("error: --panel-size must be >= 1\n")
        return _EXIT_USAGE
    if args.jitter_sigma <= 0:
        sys.stderr.write("error: --jitter-sigma must be > 0\n")
        return _EXIT_USAGE

    if not args.inputs.is_file():
        sys.stderr.write(
            f"error: inputs file {args.inputs} does not exist — "
            "run the Phase-1 runner first or supply --inputs\n"
        )
        return _EXIT_VALIDATION

    try:
        inputs = _load_inputs(args.inputs)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: failed to parse {args.inputs}: {exc}\n")
        return _EXIT_VALIDATION

    scorecard = score_synthetic_phase2(
        inputs,
        panel_size=args.panel_size,
        jitter_sigma=args.jitter_sigma,
        win_margin_delta=args.win_margin_delta,
        target=args.target,
    )

    args.out.write_text(scorecard.model_dump_json(indent=2) + "\n")
    args.markdown_summary.write_text(_markdown_summary(scorecard))

    if not scorecard.passes_gate:
        sys.stderr.write(
            f"info: synthetic Phase 2 gate missed "
            f"(fraction={scorecard.fraction_beats_synthetic_median} "
            f"target={scorecard.target}). See {args.markdown_summary}.\n"
        )
        return _EXIT_GATE_MISS
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_GATE_TARGET",
    "DEFAULT_JITTER_SIGMA",
    "DEFAULT_PANEL_SIZE",
    "SYNTHETIC_BANNER",
    "SyntheticFixtureScore",
    "SyntheticPhase2Scorecard",
    "main",
    "score_fixture_synthetic",
    "score_synthetic_phase2",
    "synthesize_expert_npes",
]
