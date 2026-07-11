# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the Suite B scorer CLI."""

from __future__ import annotations

import json
from pathlib import Path

from apmode.benchmarks.suite_b_cli import _markdown, _score, main


def _payload(
    *,
    case_id: str,
    convergence_rate: float = 1.0,
    cross_seed_cv_max: float | None = 0.05,
    n_seeds: int = 3,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "suite": "B",
        "dataset_id": "x",
        "skipped": skipped,
        "skip_reason": skip_reason,
        "n_seeds": n_seeds,
        "convergence_rate": convergence_rate,
        "cross_seed_cv_max": cross_seed_cv_max,
        "cross_seed_cv_per_param": {},
        "perturbation_manifests": [],
        "seed_results": [],
    }


class TestScore:
    def test_passes_when_all_eligible_meet_gate(self) -> None:
        results = {
            "a": _payload(case_id="a", convergence_rate=1.0, cross_seed_cv_max=0.1),
            "b": _payload(case_id="b", convergence_rate=0.9, cross_seed_cv_max=0.4),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["passes_gate"] is True
        assert scorecard["n_eligible"] == 2
        assert scorecard["n_passed"] == 2

    def test_fails_when_any_case_below_convergence(self) -> None:
        results = {
            "a": _payload(case_id="a", convergence_rate=1.0),
            "b": _payload(case_id="b", convergence_rate=0.5),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["passes_gate"] is False
        assert scorecard["n_passed"] == 1

    def test_fails_when_cross_seed_cv_exceeds_threshold(self) -> None:
        results = {
            "a": _payload(case_id="a", convergence_rate=1.0, cross_seed_cv_max=0.6),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["passes_gate"] is False

    def test_skipped_cases_excluded_from_gate_math(self) -> None:
        results = {
            "a": _payload(case_id="a", convergence_rate=1.0),
            "b": _payload(
                case_id="b",
                skipped=True,
                skip_reason="NODE",
                convergence_rate=0.0,
                cross_seed_cv_max=None,
                n_seeds=0,
            ),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["n_eligible"] == 1
        assert scorecard["n_skipped"] == 1
        assert scorecard["passes_gate"] is True

    def test_no_eligible_cases_means_no_pass(self) -> None:
        results = {
            "a": _payload(case_id="a", skipped=True, skip_reason="x", convergence_rate=0.0),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["passes_gate"] is False
        assert scorecard["n_eligible"] == 0

    def test_cross_seed_cv_none_is_treated_as_pass(self) -> None:
        # When fewer than 2 seeds converged, cv_max is None — the gate
        # should not penalise that (no signal yet).
        results = {
            "a": _payload(case_id="a", convergence_rate=1.0, cross_seed_cv_max=None),
        }
        scorecard = _score(results, min_convergence_rate=0.8, max_cross_seed_cv=0.5)
        assert scorecard["passes_gate"] is True


class TestMarkdown:
    def test_renders_pass_fail_skipped_rows(self) -> None:
        scorecard = _score(
            {
                "a": _payload(case_id="a", convergence_rate=1.0, cross_seed_cv_max=0.1),
                "b": _payload(case_id="b", convergence_rate=0.5),
                "c": _payload(case_id="c", skipped=True, skip_reason="NODE"),
            },
            min_convergence_rate=0.8,
            max_cross_seed_cv=0.5,
        )
        md = _markdown(scorecard)
        assert "# Suite B scorecard" in md
        assert "| a | pass" in md
        assert "| b | fail" in md
        assert "| c | skipped" in md
        assert "Verdict: **MISS**" in md


class TestMainCLI:
    def test_pass_returns_zero(self, tmp_path: Path) -> None:
        inputs = tmp_path / "results.json"
        inputs.write_text(
            json.dumps(
                {
                    "a": _payload(case_id="a", convergence_rate=1.0, cross_seed_cv_max=0.1),
                }
            )
        )
        out = tmp_path / "scorecard.json"
        md = tmp_path / "scorecard.md"
        rc = main(
            [
                "--inputs",
                str(inputs),
                "--out",
                str(out),
                "--markdown-summary",
                str(md),
                "--min-convergence-rate",
                "0.8",
                "--max-cross-seed-cv",
                "0.5",
            ]
        )
        assert rc == 0
        assert json.loads(out.read_text())["passes_gate"] is True
        assert "Verdict: **PASS**" in md.read_text()

    def test_miss_returns_one(self, tmp_path: Path) -> None:
        inputs = tmp_path / "results.json"
        inputs.write_text(
            json.dumps(
                {
                    "a": _payload(case_id="a", convergence_rate=0.0, cross_seed_cv_max=0.1),
                }
            )
        )
        out = tmp_path / "scorecard.json"
        md = tmp_path / "scorecard.md"
        rc = main(
            [
                "--inputs",
                str(inputs),
                "--out",
                str(out),
                "--markdown-summary",
                str(md),
            ]
        )
        assert rc == 1

    def test_missing_inputs_returns_three(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--inputs",
                str(tmp_path / "ghost.json"),
                "--out",
                str(tmp_path / "scorecard.json"),
                "--markdown-summary",
                str(tmp_path / "scorecard.md"),
            ]
        )
        assert rc == 3

    def test_malformed_inputs_returns_three(self, tmp_path: Path) -> None:
        inputs = tmp_path / "results.json"
        inputs.write_text("[]")  # JSON list, not the expected dict
        rc = main(
            [
                "--inputs",
                str(inputs),
                "--out",
                str(tmp_path / "scorecard.json"),
                "--markdown-summary",
                str(tmp_path / "scorecard.md"),
            ]
        )
        assert rc == 3
