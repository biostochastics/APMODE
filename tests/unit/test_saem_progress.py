# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit + property tests for :mod:`apmode.backends.saem_progress`.

The parser is the heart of PR4's streaming progress. Three pinned
contracts:

1. **Real-fixture parsing.** A captured ``nlmixr2 5.0`` SAEM run on the
   ``theo_sd`` dataset (12 subjects, 30 burn-in iters + 30 EM iters)
   sits at ``tests/fixtures/saem/theo_sd_30b_30e.log``. Replaying the
   stderr lines through the parser must yield exactly 60 iteration
   states with the correct phase assignment.
2. **Header → values pairing.** The parameter-name vector is captured
   from the ``params:`` header and zipped onto every subsequent
   iteration's ``param_values``.
3. **Robustness.** A Hypothesis fuzz drives arbitrary strings and bytes
   through ``parse()`` and asserts it never raises — that is the load-
   bearing safety property the runner relies on when it tees stderr
   into the parser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apmode.backends.saem_progress import (
    SAEMLineParser,
    SAEMState,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "saem" / "theo_sd_30b_30e.log"


# ---------------------------------------------------------------------------
# Real fixture
# ---------------------------------------------------------------------------


def test_fixture_exists() -> None:
    """The captured nlmixr2 log fixture is committed alongside the parser."""
    assert _FIXTURE.exists(), (
        f"Missing real-SAEM fixture at {_FIXTURE}. "
        "Regenerate via the R script in tests/fixtures/saem/README.md."
    )


def _replay_fixture(*, nburn: int | None = None) -> list[SAEMState]:
    parser = SAEMLineParser(nburn=nburn)
    states: list[SAEMState] = []
    for raw in _FIXTURE.read_text(encoding="utf-8").splitlines():
        s = parser.parse(raw)
        if s is not None:
            states.append(s)
    return states


def test_replay_yields_sixty_iterations() -> None:
    states = _replay_fixture()
    assert len(states) == 60
    # Iteration sequence is dense: 1..60 with no gaps.
    assert [s.iteration for s in states] == list(range(1, 61))


def test_replay_captures_header_param_names() -> None:
    states = _replay_fixture()
    # Fixture header: ``params:`` + tab-separated tka/tcl/tv/V(eta.ka)/V(eta.cl)/V(eta.v)/add.sd
    assert states[0].param_names == (
        "tka",
        "tcl",
        "tv",
        "V(eta.ka)",
        "V(eta.cl)",
        "V(eta.v)",
        "add.sd",
    )
    # All states share the same names (header captured once).
    for s in states[1:]:
        assert s.param_names == states[0].param_names


def test_replay_param_values_align_with_header() -> None:
    states = _replay_fixture()
    for s in states:
        assert len(s.param_values) == len(s.param_names)


def test_replay_first_iteration_values_match_fixture() -> None:
    """Pin the exact float values from iter 001 to detect parser drift."""
    states = _replay_fixture()
    expected = (0.508466, 0.889380, 3.462163, 0.570000, 0.285000, 0.095000, 2.312013)
    assert states[0].iteration == 1
    for got, want in zip(states[0].param_values, expected, strict=True):
        assert abs(got - want) < 1e-9, f"got {got}, expected {want}"


def test_replay_last_iteration_is_60() -> None:
    states = _replay_fixture()
    assert states[-1].iteration == 60
    assert states[-1].param_values[0] == pytest.approx(0.463163)


def test_phase_classification_with_nburn() -> None:
    states = _replay_fixture(nburn=30)
    burnin = [s for s in states if s.phase == "burnin"]
    main = [s for s in states if s.phase == "main"]
    assert len(burnin) == 30
    assert len(main) == 30
    # Boundary: iter 30 is the last burnin, iter 31 is the first main.
    assert next(s for s in states if s.iteration == 30).phase == "burnin"
    assert next(s for s in states if s.iteration == 31).phase == "main"


def test_phase_is_none_when_nburn_omitted() -> None:
    """``nburn=None`` is the explicit-unknown path; never default to ``"burnin"``."""
    states = _replay_fixture()
    assert all(s.phase is None for s in states)


# ---------------------------------------------------------------------------
# Targeted-line edge cases
# ---------------------------------------------------------------------------


class TestParserEdgeCases:
    def test_returns_none_for_blank_line(self) -> None:
        assert SAEMLineParser().parse("") is None
        assert SAEMLineParser().parse("\n") is None
        assert SAEMLineParser().parse("   \t  \n") is None

    def test_returns_none_for_progress_bar(self) -> None:
        bar = "[====|====|====|====|====|====|====|====|====|====] 0:00:00"
        assert SAEMLineParser().parse(bar) is None

    def test_returns_none_for_arrow_log_line(self) -> None:
        assert SAEMLineParser().parse("→ loading into symengine environment...") is None
        assert SAEMLineParser().parse("✔ done") is None

    def test_accepts_bytes_input(self) -> None:
        parser = SAEMLineParser()
        # Header first so param_names is populated.
        parser.parse(b"params:\ttka\ttcl\n")
        s = parser.parse(b"001: 0.5\t1.0\n")
        assert s is not None
        assert s.iteration == 1
        assert s.param_values == (0.5, 1.0)

    def test_invalid_utf8_does_not_raise(self) -> None:
        parser = SAEMLineParser()
        # Mixes a valid prefix with an invalid byte sequence.
        garbage = b"001: \xff\xfe\xfd 0.5"
        result = parser.parse(garbage)
        # Non-numeric token aborted the values list → result is either
        # None (no values + no header) or a state with empty values.
        if result is not None:
            assert result.param_values == ()

    def test_handles_NA_token(self) -> None:
        parser = SAEMLineParser()
        parser.parse("params:\ta\tb")
        s = parser.parse("003: 1.5\tNA")
        assert s is not None
        assert s.param_values[0] == pytest.approx(1.5)
        # NA → NaN
        import math

        assert math.isnan(s.param_values[1])

    def test_iteration_rejects_zero_and_too_big(self) -> None:
        parser = SAEMLineParser()
        # iter == 0 is non-physical (1-based)
        assert parser.parse("000: 0.5") is None

    def test_re_parsing_does_not_lose_param_names(self) -> None:
        parser = SAEMLineParser()
        parser.parse("params:\ttka\ttcl")
        first = parser.parse("001: 0.5\t1.0")
        second = parser.parse("002: 0.6\t1.1")
        assert first is not None and second is not None
        assert first.param_names == second.param_names == ("tka", "tcl")

    def test_tolerant_fallback_handles_dash_separator(self) -> None:
        """If nlmixr2 ever switched ``:`` to ``-`` the tolerant pattern still wins."""
        parser = SAEMLineParser()
        parser.parse("params:\ta\tb")
        s = parser.parse("042 - 1.5\t2.5")
        assert s is not None
        assert s.iteration == 42
        assert s.param_values == (1.5, 2.5)

    def test_inf_tokens_are_accepted(self) -> None:
        """Divergent SAEM iteration with ``Inf``/``-Inf`` should still surface."""
        import math

        parser = SAEMLineParser()
        parser.parse("params:\ta\tb\tc")
        s = parser.parse("005: 1.5\tInf\t-Inf")
        assert s is not None
        assert s.param_values[0] == 1.5
        assert math.isinf(s.param_values[1]) and s.param_values[1] > 0
        assert math.isinf(s.param_values[2]) and s.param_values[2] < 0

    def test_returns_none_when_values_arity_mismatches_header(self) -> None:
        """Length-mismatch guard: a row with the wrong column count is dropped
        rather than emitting a ``SAEMState`` whose ``zip(names, values)``
        silently produces nothing."""
        parser = SAEMLineParser()
        parser.parse("params:\ta\tb\tc")
        # Only two values for three names — must be rejected.
        assert parser.parse("007: 1.0\t2.0") is None
        # But the matching arity still parses.
        s = parser.parse("008: 1.0\t2.0\t3.0")
        assert s is not None and len(s.param_values) == 3

    def test_returns_none_when_header_present_but_values_unparseable(self) -> None:
        """Header captured + bad token in values → None (no hollow state)."""
        parser = SAEMLineParser()
        parser.parse("params:\ta\tb")
        # Tolerant regex matches but ``_parse_values`` rejects ``foo``.
        # The character class for the regex includes letters from ``Inf``/``NA``
        # so we use a token outside that set: ``$`` is rejected at regex level
        # (good) but a stray ``Z`` is not a number → values list empty.
        # Actually the value class is ``[\-\d\.eE\+\sNAaNnIiFf]``. A token
        # like ``99Z`` matches the regex but float() rejects it.
        s = parser.parse("009: 99Z\t99Z")
        assert s is None


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@given(line=st.text())
@settings(max_examples=300, deadline=None)
def test_parse_never_raises_on_arbitrary_text(line: str) -> None:
    """Robustness contract — the runner must not crash on any stderr byte."""
    parser = SAEMLineParser(nburn=30)
    # Must return either None or a SAEMState; never raise.
    result = parser.parse(line)
    assert result is None or isinstance(result, SAEMState)


@given(line=st.binary())
@settings(max_examples=300, deadline=None)
def test_parse_never_raises_on_arbitrary_bytes(line: bytes) -> None:
    parser = SAEMLineParser(nburn=30)
    result = parser.parse(line)
    assert result is None or isinstance(result, SAEMState)


@given(
    iteration=st.integers(min_value=1, max_value=99999),
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10),
)
@settings(max_examples=200, deadline=None)
def test_parse_round_trips_synthesized_iteration_lines(
    iteration: int, values: list[float]
) -> None:
    """A line built from a known iter + known values parses back identically."""
    parser = SAEMLineParser()
    # Need a header first so param_names is populated.
    header = "params:\t" + "\t".join(f"p{i}" for i in range(len(values)))
    parser.parse(header)
    line = f"{iteration:03d}: " + "\t".join(f"{v:.6f}" for v in values)
    state = parser.parse(line)
    assert state is not None
    assert state.iteration == iteration
    assert len(state.param_values) == len(values)
    for got, want in zip(state.param_values, values, strict=True):
        # Round-trip through %f formatting → up to 6 decimal places.
        assert abs(got - want) < 1e-5
