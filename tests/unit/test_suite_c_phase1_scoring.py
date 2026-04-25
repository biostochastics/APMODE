# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for plan Task 41 — Phase-1 Suite C scoring helper.

Pure-Python coverage of the win-rate math + scorecard schema.
Orchestrator-driven NPE production is plan Task 44; the helper itself
must not require R / cmdstan to test.
"""

from __future__ import annotations

import pytest

from apmode.benchmarks.literature_loader import PHASE1_MLE_FIXTURE_IDS
from apmode.benchmarks.suite_c import WIN_MARGIN_DELTA
from apmode.benchmarks.suite_c_phase1_scoring import (
    PHASE1_FRACTION_BEATS_TARGET,
    PHASE1_MIN_FIXTURES_FOR_AGGREGATE,
    FixtureScore,
    aggregate_phase1_scorecard,
    phase1_roster_dois,
    score_fixture,
)

# ---------------------------------------------------------------------------
# score_fixture — single-fixture decision math
# ---------------------------------------------------------------------------


def test_score_fixture_apmode_beats_with_sufficient_margin() -> None:
    """APMODE NPE strictly below ``lit * (1 - delta)`` → beats=True."""
    score = score_fixture(
        fixture_id="theophylline_boeckmann_1992",
        npe_apmode=0.95,
        npe_literature=1.00,  # threshold = 0.98
    )
    assert score.beats_literature is True
    assert score.npe_apmode == pytest.approx(0.95)
    assert score.npe_literature == pytest.approx(1.00)
    assert score.win_margin_delta == pytest.approx(WIN_MARGIN_DELTA)


def test_score_fixture_apmode_loses_when_inside_margin() -> None:
    """APMODE NPE within ``lit * (1 - delta)`` margin → beats=False."""
    score = score_fixture(
        fixture_id="warfarin_funaki_2018",
        npe_apmode=0.99,
        npe_literature=1.00,  # threshold = 0.98
    )
    assert score.beats_literature is False


def test_score_fixture_apmode_exactly_at_threshold_beats() -> None:
    """``<=`` boundary: NPE exactly at threshold counts as a win.

    Documents the inclusive-bound choice — a tie-breaking convention
    the legacy ``compute_fraction_beats_expert`` already uses.
    """
    score = score_fixture(
        fixture_id="schoemaker_nlmixr2_tutorial",
        npe_apmode=0.98,
        npe_literature=1.00,  # threshold = 0.98
    )
    assert score.beats_literature is True


def test_score_fixture_per_fold_must_match_default_split_n_folds() -> None:
    """The per-fold validator enforces the ``DEFAULT_SPLIT.n_folds`` invariant.

    A wrong-length per-fold tuple can sneak through the inputs JSON
    layer, so the FixtureScore model is the last guard before
    downstream variance / CI tooling consumes the values.
    """
    with pytest.raises(ValueError, match="must have exactly 5 entries"):
        score_fixture(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=0.95,
            npe_literature=1.0,
            npe_apmode_per_fold=(0.93, 0.96, 0.94, 0.97),  # 4 entries
        )
    s = score_fixture(
        fixture_id="theophylline_boeckmann_1992",
        npe_apmode=0.95,
        npe_literature=1.0,
        npe_apmode_per_fold=(0.93, 0.96, 0.94, 0.97, 0.95),
    )
    assert s.npe_apmode_per_fold == (0.93, 0.96, 0.94, 0.97, 0.95)


def test_score_fixture_per_fold_rejects_non_finite_value() -> None:
    """Inf/NaN sneaking through one fold short-circuits the median honesty."""
    with pytest.raises(ValueError, match=r"npe_apmode_per_fold\[2\]"):
        score_fixture(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=0.95,
            npe_literature=1.0,
            npe_apmode_per_fold=(0.93, 0.96, float("inf"), 0.97, 0.95),
        )


def test_score_fixture_rejects_non_positive_npe() -> None:
    with pytest.raises(ValueError, match="npe_apmode must be a positive"):
        score_fixture(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=0.0,
            npe_literature=1.0,
        )
    with pytest.raises(ValueError, match="npe_literature must be a positive"):
        score_fixture(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=1.0,
            npe_literature=-0.5,
        )


def test_score_fixture_rejects_non_finite_npe() -> None:
    with pytest.raises(ValueError, match="npe_apmode must be a positive finite"):
        score_fixture(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=float("inf"),
            npe_literature=1.0,
        )


def test_score_fixture_custom_delta_overrides_default() -> None:
    """A versioned policy may pass a tighter or looser margin."""
    score = score_fixture(
        fixture_id="gentamicin_germovsek_2017",
        npe_apmode=0.95,
        npe_literature=1.00,
        win_margin_delta=0.10,  # threshold 0.90 — stricter
    )
    assert score.beats_literature is False
    assert score.win_margin_delta == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# aggregate_phase1_scorecard — roster-level decision math
# ---------------------------------------------------------------------------


def _make_score(fixture_id: str, *, beats: bool) -> FixtureScore:
    """Test helper — pin the NPE pair so ``beats`` is the only knob."""
    return score_fixture(
        fixture_id=fixture_id,
        npe_apmode=0.95 if beats else 0.99,
        npe_literature=1.00,
    )


def test_aggregate_returns_none_fraction_when_below_min_fixtures() -> None:
    """< 3 fixtures → fraction is None and gate is False (no false positive)."""
    scores = [
        _make_score("a", beats=True),
        _make_score("b", beats=True),
    ]
    card = aggregate_phase1_scorecard(scores)
    assert card.n_datasets == 2
    assert card.n_beats == 2
    assert card.fraction_beats_literature_median is None
    assert card.passes_gate is False


def test_aggregate_passes_gate_at_60_percent() -> None:
    """3-of-5 = 0.60 hits the target boundary → passes_gate=True."""
    scores = [
        _make_score("a", beats=True),
        _make_score("b", beats=True),
        _make_score("c", beats=True),
        _make_score("d", beats=False),
        _make_score("e", beats=False),
    ]
    card = aggregate_phase1_scorecard(scores)
    assert card.n_datasets == 5
    assert card.n_beats == 3
    assert card.fraction_beats_literature_median == pytest.approx(0.60)
    assert card.passes_gate is True
    assert card.target == pytest.approx(PHASE1_FRACTION_BEATS_TARGET)


def test_aggregate_fails_gate_at_2_of_5() -> None:
    """2-of-5 = 0.40 misses the 0.60 target."""
    scores = [
        _make_score("a", beats=True),
        _make_score("b", beats=True),
        _make_score("c", beats=False),
        _make_score("d", beats=False),
        _make_score("e", beats=False),
    ]
    card = aggregate_phase1_scorecard(scores)
    assert card.fraction_beats_literature_median == pytest.approx(0.40)
    assert card.passes_gate is False


def test_aggregate_preserves_input_order_for_deterministic_ci_diff() -> None:
    """Determinism contract: helper does not re-sort scores."""
    scores = [
        _make_score("z_last", beats=True),
        _make_score("a_first", beats=True),
        _make_score("m_middle", beats=False),
    ]
    card = aggregate_phase1_scorecard(scores)
    assert [s.fixture_id for s in card.scores] == ["z_last", "a_first", "m_middle"]


def test_aggregate_custom_target_overrides_default() -> None:
    """A future policy may bump the gate threshold without code changes."""
    scores = [
        _make_score("a", beats=True),
        _make_score("b", beats=True),
        _make_score("c", beats=True),
        _make_score("d", beats=True),
        _make_score("e", beats=False),
    ]
    # 4/5 = 0.80; default target 0.60 → pass; custom 0.90 → fail
    card_default = aggregate_phase1_scorecard(scores)
    card_strict = aggregate_phase1_scorecard(scores, target=0.90)
    assert card_default.passes_gate is True
    assert card_strict.passes_gate is False
    assert card_strict.target == pytest.approx(0.90)


def test_aggregate_min_fixtures_override() -> None:
    """Lowering ``min_fixtures`` enables the aggregate on smaller rosters.

    Useful for an ad-hoc CI sub-run that scores only the 3 most-stable
    fixtures while a slow new fixture is debugging — the aggregate
    still has math behind it.
    """
    scores = [
        _make_score("a", beats=True),
        _make_score("b", beats=True),
    ]
    card = aggregate_phase1_scorecard(scores, min_fixtures=2)
    assert card.fraction_beats_literature_median == pytest.approx(1.0)
    assert card.passes_gate is True


# ---------------------------------------------------------------------------
# Roster bookkeeping
# ---------------------------------------------------------------------------


def test_phase1_roster_dois_returns_one_doi_per_fixture() -> None:
    """The roster helper resolves every Phase-1 MLE fixture's DOI."""
    dois = phase1_roster_dois()
    assert set(dois.keys()) == set(PHASE1_MLE_FIXTURE_IDS)
    for fid, doi in dois.items():
        assert doi, f"fixture {fid} returned an empty DOI"
        assert doi.startswith("10."), (
            f"fixture {fid} DOI {doi!r} is not Crossref-canonical (must start with '10.')"
        )


def test_min_fixtures_default_matches_legacy_min_literature_count() -> None:
    """Phase-1 floor agrees with the legacy ``MIN_LITERATURE_COUNT``.

    Cross-link guard — if either constant drifts the other should
    follow, since they encode the same statistical intuition (a
    single-fixture median is not meaningful).
    """
    from apmode.benchmarks.suite_c import MIN_LITERATURE_COUNT

    assert PHASE1_MIN_FIXTURES_FOR_AGGREGATE == MIN_LITERATURE_COUNT
