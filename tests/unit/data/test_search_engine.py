# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for SearchEngine: candidate dispatch, scoring, Pareto frontier."""

from __future__ import annotations

from apmode.search.engine import SearchResult, _pareto_frontier


class TestParetoFrontier:
    """Pareto frontier: minimize n_params AND BIC."""

    def test_single_candidate(self) -> None:
        sr = SearchResult(
            candidate_id="a",
            spec=None,
            result=None,
            converged=True,
            bic=100.0,
            n_params=3,  # type: ignore[arg-type]
        )
        front = _pareto_frontier([sr])
        assert len(front) == 1
        assert front[0].candidate_id == "a"

    def test_dominated_excluded(self) -> None:
        sr1 = SearchResult(
            candidate_id="a",
            spec=None,
            result=None,
            converged=True,
            bic=100.0,
            n_params=3,  # type: ignore[arg-type]
        )
        sr2 = SearchResult(
            candidate_id="b",
            spec=None,
            result=None,
            converged=True,
            bic=200.0,
            n_params=5,  # type: ignore[arg-type]
        )
        front = _pareto_frontier([sr1, sr2])
        assert len(front) == 1
        assert front[0].candidate_id == "a"

    def test_pareto_tradeoff(self) -> None:
        # a: fewer params, higher BIC; b: more params, lower BIC
        sr1 = SearchResult(
            candidate_id="a",
            spec=None,
            result=None,
            converged=True,
            bic=150.0,
            n_params=2,  # type: ignore[arg-type]
        )
        sr2 = SearchResult(
            candidate_id="b",
            spec=None,
            result=None,
            converged=True,
            bic=100.0,
            n_params=5,  # type: ignore[arg-type]
        )
        front = _pareto_frontier([sr1, sr2])
        # Both are Pareto-optimal (neither dominates the other)
        assert len(front) == 2

    def test_nonconverged_excluded(self) -> None:
        sr1 = SearchResult(
            candidate_id="a",
            spec=None,
            result=None,
            converged=False,
            bic=None,
            n_params=3,  # type: ignore[arg-type]
        )
        sr2 = SearchResult(
            candidate_id="b",
            spec=None,
            result=None,
            converged=True,
            bic=100.0,
            n_params=3,  # type: ignore[arg-type]
        )
        front = _pareto_frontier([sr1, sr2])
        assert len(front) == 1
        assert front[0].candidate_id == "b"

    def test_empty_input(self) -> None:
        assert _pareto_frontier([]) == []
