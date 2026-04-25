# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite C Phase-1 scoring helper (plan Task 41).

The Phase-1 Suite C contract per PRD §4 / plan Task 41:

* Score APMODE against each Phase-1 MLE fixture using subject-level
  5-fold cross-validation (the same split strategy
  ``apmode.benchmarks.suite_c.DEFAULT_SPLIT`` codifies for the legacy
  ``BenchmarkCase`` system).
* Report the per-dataset ``NPE_median`` for both the APMODE candidate
  and the literature anchor.
* Aggregate via

      fraction_beats_literature_median = Σ(NPE_APMODE ≤ NPE_lit·(1-δ))/|D|

  with ``δ = 0.02`` (2% margin); the v0.6 CI gate is ``>= 0.60``
  (3 of 5 fixtures must beat the literature NPE by at least 2%).

This module ships **only the scoring math + scorecard schema**. The
actual NPE production loop (load fixture → drive the orchestrator
through the dataset → record per-fold NPE) is deferred to plan Task 44
(full regression sweep) and the weekly CI workflow shipped alongside
this file. Decoupling lets the unit tests cover the disqualifying
arithmetic without spinning up R + cmdstan, while the CI workflow
reuses the helper as soon as a credentialed run is available.

Wins vs the legacy :mod:`apmode.benchmarks.suite_c`
----------------------------------------------------

The legacy module defines ``BenchmarkCase`` + ``WIN_MARGIN_DELTA``
under the older "expert benchmark" framing (plan Task 38 renamed the
suite). This module is the *Phase-1 specific* surface that operates on
:class:`apmode.benchmarks.models.LiteratureFixture` (the YAML/DSL pair
landed in Task 40) — so the scorecard schema can carry the fixture id
+ DOI + reference parameter set rather than the legacy
``BenchmarkCase.expected_structure``. Both layers agree on
``WIN_MARGIN_DELTA = 0.02`` and the same 5-fold CV cadence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apmode.benchmarks.literature_loader import (
    PHASE1_MLE_FIXTURE_IDS,
    load_fixture_by_id,
)
from apmode.benchmarks.suite_c import DEFAULT_SPLIT, WIN_MARGIN_DELTA

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Constants — keep the gate threshold + win margin in one place
# ---------------------------------------------------------------------------

PHASE1_FRACTION_BEATS_TARGET: float = 0.60
"""CI target per plan Task 41. ``fraction_beats_literature_median`` must
hit at least this number for the weekly Suite C run to pass. A miss
opens a GitHub issue — it is *not* a release block (the suite measures
methodology drift, not a release-critical contract)."""

PHASE1_MIN_FIXTURES_FOR_AGGREGATE: int = 3
"""Minimum number of fixture scores required before
:func:`aggregate_phase1_scorecard` returns a non-``None``
``fraction_beats_literature_median``. With < 3 datasets the aggregate
is statistically meaningless — the legacy
:data:`apmode.benchmarks.suite_c.MIN_LITERATURE_COUNT` uses the same
floor for the same reason."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FixtureScore(BaseModel):
    """Per-fixture Phase-1 score record.

    ``npe_apmode`` is the APMODE candidate's median NPE on the
    held-out folds (subject-level 5-fold CV). ``npe_literature`` is
    the published reference fit's NPE, computed under the *same*
    observation model so the win/loss decision is on commensurate
    metrics (PRD §4.3.1 / plan §10 Q2). When the literature paper does
    not report NPE directly, the value is computed by re-fitting the
    published parameter set under the orchestrator's harness — that
    workflow is the integration concern of plan Task 44, not this
    helper.

    ``npe_apmode_per_fold`` carries the raw per-fold NPEs that produce
    ``npe_apmode`` (their median). Optional and ``None`` for legacy
    inputs that only ship the median; when present, it MUST contain
    exactly ``DEFAULT_SPLIT.n_folds`` (5) finite positive values, so
    downstream variance/CI tooling does not need to special-case
    partial-fold runs. Adding it as an optional field rather than a
    required one preserves the v0.5 inputs JSON shape.

    ``beats_literature`` is the canonical disqualifier:
    ``True`` iff ``npe_apmode <= npe_literature * (1 - WIN_MARGIN_DELTA)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str = Field(min_length=1)
    npe_apmode: float = Field(gt=0.0)
    npe_literature: float = Field(gt=0.0)
    npe_apmode_per_fold: tuple[float, ...] | None = None
    beats_literature: bool
    win_margin_delta: float = Field(default=WIN_MARGIN_DELTA, ge=0.0, le=1.0)

    @field_validator("npe_apmode", "npe_literature")
    @classmethod
    def _check_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            msg = "NPE values must be finite (no inf/NaN)"
            raise ValueError(msg)
        return v

    @field_validator("npe_apmode_per_fold")
    @classmethod
    def _check_per_fold_values(cls, v: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if v is None:
            return v
        if not v:
            msg = "npe_apmode_per_fold must be non-empty when provided"
            raise ValueError(msg)
        expected = DEFAULT_SPLIT.n_folds
        if len(v) != expected:
            msg = (
                f"npe_apmode_per_fold must have exactly {expected} entries "
                f"(matching DEFAULT_SPLIT.n_folds), got {len(v)}"
            )
            raise ValueError(msg)
        for i, fold_npe in enumerate(v):
            if not math.isfinite(fold_npe) or fold_npe <= 0:
                msg = (
                    f"npe_apmode_per_fold[{i}] must be a positive finite number, got {fold_npe!r}"
                )
                raise ValueError(msg)
        return v


class SuiteCPhase1Scorecard(BaseModel):
    """Aggregate Phase-1 scorecard returned by :func:`aggregate_phase1_scorecard`.

    ``fraction_beats_literature_median`` is the headline number; ``None``
    when the underlying score list has fewer than
    :data:`PHASE1_MIN_FIXTURES_FOR_AGGREGATE` entries (avoids reporting
    a 1-of-1 = 100% as a meaningful win-rate).

    ``passes_gate`` is the CI gate: ``True`` iff
    ``fraction_beats_literature_median >= PHASE1_FRACTION_BEATS_TARGET``
    (and the aggregate is computable). The weekly workflow uses this
    field to decide whether to open a GitHub issue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: list[FixtureScore]
    n_datasets: int = Field(ge=0)
    n_beats: int = Field(ge=0)
    fraction_beats_literature_median: float | None = Field(default=None, ge=0.0, le=1.0)
    target: float = Field(default=PHASE1_FRACTION_BEATS_TARGET, ge=0.0, le=1.0)
    passes_gate: bool


# ---------------------------------------------------------------------------
# Scoring math
# ---------------------------------------------------------------------------


def score_fixture(
    *,
    fixture_id: str,
    npe_apmode: float,
    npe_literature: float,
    win_margin_delta: float = WIN_MARGIN_DELTA,
    npe_apmode_per_fold: tuple[float, ...] | None = None,
) -> FixtureScore:
    """Decide whether APMODE beats the literature NPE for one fixture.

    ``win_margin_delta`` defaults to the project-wide
    :data:`apmode.benchmarks.suite_c.WIN_MARGIN_DELTA` constant
    (``0.02``); callers should not override it without a versioned
    policy reason — drifting the margin per dataset would invalidate
    cross-release comparisons.

    ``npe_apmode_per_fold`` is optional — when supplied by the
    Task 44 live-fit runner it carries the raw per-fold NPEs whose
    median is ``npe_apmode``. Validation is delegated to
    :class:`FixtureScore` so the validator runs once at construction.
    """
    if not math.isfinite(npe_apmode) or npe_apmode <= 0:
        msg = f"npe_apmode must be a positive finite number, got {npe_apmode!r}"
        raise ValueError(msg)
    if not math.isfinite(npe_literature) or npe_literature <= 0:
        msg = f"npe_literature must be a positive finite number, got {npe_literature!r}"
        raise ValueError(msg)

    threshold = npe_literature * (1.0 - win_margin_delta)
    beats = npe_apmode <= threshold
    return FixtureScore(
        fixture_id=fixture_id,
        npe_apmode=npe_apmode,
        npe_literature=npe_literature,
        npe_apmode_per_fold=npe_apmode_per_fold,
        beats_literature=beats,
        win_margin_delta=win_margin_delta,
    )


def aggregate_phase1_scorecard(
    scores: Iterable[FixtureScore],
    *,
    target: float = PHASE1_FRACTION_BEATS_TARGET,
    min_fixtures: int = PHASE1_MIN_FIXTURES_FOR_AGGREGATE,
) -> SuiteCPhase1Scorecard:
    """Aggregate per-fixture scores into the Phase-1 scorecard.

    ``min_fixtures`` is the floor below which the aggregate fraction
    is reported as ``None`` (and ``passes_gate=False`` regardless of
    the partial wins) — this prevents a 1-of-1 = 100% from looking
    like a green Phase-1 run. ``target`` defaults to the v0.6 CI gate
    (``0.60``); the weekly workflow reads this off
    ``SuiteCPhase1Scorecard.passes_gate``.

    The returned scorecard preserves the input order of ``scores`` for
    determinism — callers that need a stable diff across CI runs
    should pass ``scores`` sorted by ``fixture_id`` (the helper does
    not re-sort to avoid surprising callers who pass a meaningful
    ordering).
    """
    score_list = list(scores)
    n_datasets = len(score_list)
    n_beats = sum(1 for s in score_list if s.beats_literature)

    if n_datasets < min_fixtures:
        return SuiteCPhase1Scorecard(
            scores=score_list,
            n_datasets=n_datasets,
            n_beats=n_beats,
            fraction_beats_literature_median=None,
            target=target,
            passes_gate=False,
        )

    fraction = n_beats / n_datasets
    return SuiteCPhase1Scorecard(
        scores=score_list,
        n_datasets=n_datasets,
        n_beats=n_beats,
        fraction_beats_literature_median=fraction,
        target=target,
        passes_gate=fraction >= target,
    )


# ---------------------------------------------------------------------------
# Roster bookkeeping
# ---------------------------------------------------------------------------


def phase1_roster_dois() -> dict[str, str]:
    """Return ``{fixture_id: DOI}`` for the canonical Phase-1 MLE roster.

    Used by the weekly workflow to render a table in the GitHub issue
    that fires on a missed gate, so a reviewer can map each fixture
    back to its literature anchor without grepping the YAMLs.
    """
    out: dict[str, str] = {}
    for fid in PHASE1_MLE_FIXTURE_IDS:
        fix = load_fixture_by_id(fid)
        out[fid] = fix.reference.doi
    return out


__all__ = [
    "PHASE1_FRACTION_BEATS_TARGET",
    "PHASE1_MIN_FIXTURES_FOR_AGGREGATE",
    "FixtureScore",
    "SuiteCPhase1Scorecard",
    "aggregate_phase1_scorecard",
    "phase1_roster_dois",
    "score_fixture",
]
