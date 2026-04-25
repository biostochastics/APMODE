# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the ``python -m apmode.benchmarks.suite_c_phase1_cli`` driver.

Plan Task 41 — exercise the JSON inputs loader, scorecard JSON
emission, Markdown rendering, and the documented exit codes (2 for
usage errors, 3 for fixture validation failures).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.benchmarks.suite_c_phase1_cli import (
    main,
    render_markdown_summary,
)
from apmode.benchmarks.suite_c_phase1_scoring import (
    SuiteCPhase1Scorecard,
    aggregate_phase1_scorecard,
    score_fixture,
)

# ---------------------------------------------------------------------------
# CLI happy-path
# ---------------------------------------------------------------------------


def _five_fixture_inputs() -> dict[str, dict[str, float]]:
    return {
        "theophylline_boeckmann_1992": {"npe_apmode": 0.95, "npe_literature": 1.00},
        "warfarin_funaki_2018": {"npe_apmode": 0.93, "npe_literature": 1.00},
        "mavoglurant_wendling_2015": {"npe_apmode": 0.97, "npe_literature": 1.00},
        "gentamicin_germovsek_2017": {"npe_apmode": 0.99, "npe_literature": 1.00},
        "schoemaker_nlmixr2_tutorial": {"npe_apmode": 0.95, "npe_literature": 1.00},
    }


def test_cli_writes_json_scorecard(tmp_path: Path) -> None:
    """JSON scorecard parses back into ``SuiteCPhase1Scorecard``."""
    inputs_file = tmp_path / "in.json"
    inputs_file.write_text(json.dumps(_five_fixture_inputs()))
    out_file = tmp_path / "scorecard.json"

    rc = main(["--inputs", str(inputs_file), "--out", str(out_file)])
    assert rc == 0

    payload = json.loads(out_file.read_text())
    card = SuiteCPhase1Scorecard.model_validate(payload)
    assert card.n_datasets == 5
    # 3-of-5 below threshold (theo, warfarin, schoemaker beat;
    # mavoglurant 0.97 vs threshold 0.98 beats; gentamicin 0.99 loses)
    assert card.n_beats == 4
    assert card.fraction_beats_literature_median == pytest.approx(0.80)
    assert card.passes_gate is True


def test_cli_writes_markdown_summary_when_requested(tmp_path: Path) -> None:
    inputs_file = tmp_path / "in.json"
    inputs_file.write_text(json.dumps(_five_fixture_inputs()))
    md_file = tmp_path / "scorecard.md"
    out_file = tmp_path / "scorecard.json"

    rc = main(
        [
            "--inputs",
            str(inputs_file),
            "--out",
            str(out_file),
            "--markdown-summary",
            str(md_file),
        ]
    )
    assert rc == 0

    md = md_file.read_text()
    assert "# Suite C Phase-1 scorecard" in md
    assert "theophylline_boeckmann_1992" in md
    assert "Fraction beating literature" in md
    assert "80%" in md  # 4/5 beats


# ---------------------------------------------------------------------------
# CLI error paths
# ---------------------------------------------------------------------------


def test_cli_returns_2_for_missing_inputs(tmp_path: Path) -> None:
    rc = main(
        [
            "--inputs",
            str(tmp_path / "does_not_exist.json"),
            "--out",
            str(tmp_path / "scorecard.json"),
        ]
    )
    assert rc == 2


def test_cli_returns_2_for_malformed_inputs_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = main(["--inputs", str(bad), "--out", str(tmp_path / "out.json")])
    assert rc == 2


def test_cli_returns_2_for_inputs_missing_required_keys(tmp_path: Path) -> None:
    bad = tmp_path / "missing_keys.json"
    bad.write_text(json.dumps({"fix1": {"npe_apmode": 1.0}}))
    rc = main(["--inputs", str(bad), "--out", str(tmp_path / "out.json")])
    assert rc == 2


def test_cli_returns_3_for_negative_npe_value(tmp_path: Path) -> None:
    bad = tmp_path / "neg_npe.json"
    bad.write_text(
        json.dumps(
            {"fix1": {"npe_apmode": -0.5, "npe_literature": 1.0}},
        )
    )
    rc = main(["--inputs", str(bad), "--out", str(tmp_path / "out.json")])
    assert rc == 3


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_markdown_summary_marks_failed_gate_with_red_x() -> None:
    scores = [
        score_fixture(fixture_id="a", npe_apmode=0.99, npe_literature=1.0),
        score_fixture(fixture_id="b", npe_apmode=0.99, npe_literature=1.0),
        score_fixture(fixture_id="c", npe_apmode=0.99, npe_literature=1.0),
    ]
    card = aggregate_phase1_scorecard(scores)
    md = render_markdown_summary(card)
    assert ":x:" in md
    assert ":white_check_mark:" not in md.split("|", 1)[0]  # not in header


def test_markdown_summary_omits_fraction_for_small_roster() -> None:
    """< 3 fixtures → fraction is None → headline says 'not computed'."""
    scores = [score_fixture(fixture_id="a", npe_apmode=0.95, npe_literature=1.0)]
    card = aggregate_phase1_scorecard(scores)
    md = render_markdown_summary(card)
    assert "not computed" in md
