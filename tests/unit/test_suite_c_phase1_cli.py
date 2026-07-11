# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the ``python -m apmode.benchmarks.suite_c_phase1_cli`` driver.

Exercise the JSON inputs loader, scorecard JSON emission, Markdown rendering,
and the documented exit codes (2 for
usage errors, 3 for fixture validation failures).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apmode.benchmarks.suite_c_phase1_cli import (
    main,
    render_markdown_summary,
)
from apmode.benchmarks.suite_c_phase1_scoring import (
    SuiteCPhase1Provenance,
    SuiteCPhase1Scorecard,
    aggregate_phase1_scorecard,
    score_fixture,
)

# A fixed, live, fresh provenance for deterministic Markdown-rendering tests
# (no wall-clock timestamps leak into golden-ish output paths).
_FRESH_NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _fresh_live_provenance() -> SuiteCPhase1Provenance:
    return SuiteCPhase1Provenance(
        generated_at="2026-07-10T00:00:00+00:00",
        snapshot_source="benchmarks/suite_c/phase1_npe_inputs.json",
        git_sha="deadbeef",
        live_runner=True,
    )


def _provenance_block(*, live_runner: bool, generated_at: str) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "snapshot_source": "benchmarks/suite_c/phase1_npe_inputs.json",
        "git_sha": "deadbeef",
        "live_runner": live_runner,
    }


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
    payload: dict[str, object] = dict(_five_fixture_inputs())
    payload["_provenance"] = _provenance_block(
        live_runner=True, generated_at="2026-07-10T00:00:00+00:00"
    )
    inputs_file.write_text(json.dumps(payload))
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
    # inputs without pit_calibration_* still render (dash placeholder,
    # not a KeyError) — forward/backward compat with older inputs files.
    assert "—" in md


def test_cli_markdown_summary_renders_pit_calibration_when_present(tmp_path: Path) -> None:
    """PIT/NPDE-lite calibration — previously computed and discarded — now
    shows up as its own column pair in the Markdown summary."""
    payload = _five_fixture_inputs()
    payload["theophylline_boeckmann_1992"]["pit_calibration_apmode"] = {  # type: ignore[index]
        "p5": 0.06,
        "p50": 0.49,
        "p95": 0.94,
    }
    payload["theophylline_boeckmann_1992"]["pit_calibration_literature"] = {  # type: ignore[index]
        "p5": 0.04,
        "p50": 0.52,
        "p95": 0.97,
    }
    payload["_provenance"] = _provenance_block(  # type: ignore[assignment]
        live_runner=True, generated_at="2026-07-10T00:00:00+00:00"
    )
    inputs_file = tmp_path / "in.json"
    inputs_file.write_text(json.dumps(payload))
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
    assert "PIT APMODE" in md
    assert "PIT Literature" in md
    assert "p50=0.49" in md
    assert "p50=0.52" in md
    assert "PIT/NPDE-lite calibration" in md  # explanatory footnote


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
    md = render_markdown_summary(card, provenance=_fresh_live_provenance(), now=_FRESH_NOW)
    assert ":x:" in md
    assert ":white_check_mark:" not in md.split("|", 1)[0]  # not in header


def test_markdown_summary_omits_fraction_for_small_roster() -> None:
    """< 3 fixtures → fraction is None → headline says 'not computed'."""
    scores = [score_fixture(fixture_id="a", npe_apmode=0.95, npe_literature=1.0)]
    card = aggregate_phase1_scorecard(scores)
    md = render_markdown_summary(card, provenance=_fresh_live_provenance(), now=_FRESH_NOW)
    assert "not computed" in md


# ---------------------------------------------------------------------------
# Provenance + staleness banner
# ---------------------------------------------------------------------------


def _one_score_card() -> SuiteCPhase1Scorecard:
    scores = [
        score_fixture(fixture_id="a", npe_apmode=0.95, npe_literature=1.0),
        score_fixture(fixture_id="b", npe_apmode=0.95, npe_literature=1.0),
        score_fixture(fixture_id="c", npe_apmode=0.95, npe_literature=1.0),
    ]
    return aggregate_phase1_scorecard(scores)


def test_missing_provenance_raises() -> None:
    """Rendering with no provenance must fail loud rather than silently
    render a static snapshot as if it were live validation."""
    card = _one_score_card()
    with pytest.raises(ValueError, match="provenance"):
        render_markdown_summary(card)


def test_markdown_banner_appears_for_non_live_snapshot() -> None:
    """live_runner=False always triggers the STALE / NON-LIVE banner,
    regardless of how recent the snapshot is."""
    card = _one_score_card()
    prov = SuiteCPhase1Provenance(
        generated_at="2026-07-10T00:00:00+00:00",
        snapshot_source="benchmarks/suite_c/phase1_npe_inputs.json",
        live_runner=False,
    )
    md = render_markdown_summary(card, provenance=prov, now=_FRESH_NOW)
    assert "STALE / NON-LIVE SNAPSHOT" in md
    assert "live_runner is false" in md


def test_markdown_banner_appears_for_stale_live_snapshot() -> None:
    """A live snapshot older than --stale-warn-days still gets the banner."""
    card = _one_score_card()
    prov = SuiteCPhase1Provenance(
        generated_at="2020-01-01T00:00:00+00:00",
        snapshot_source="x",
        live_runner=True,
    )
    md = render_markdown_summary(card, provenance=prov, stale_warn_days=30.0, now=_FRESH_NOW)
    assert "STALE / NON-LIVE SNAPSHOT" in md
    assert "staleness threshold" in md


def test_markdown_banner_omitted_for_fresh_live_snapshot() -> None:
    """A fresh, live snapshot omits the banner entirely."""
    card = _one_score_card()
    prov = SuiteCPhase1Provenance(
        generated_at="2026-07-10T00:00:00+00:00",
        snapshot_source="x",
        live_runner=True,
    )
    md = render_markdown_summary(card, provenance=prov, stale_warn_days=30.0, now=_FRESH_NOW)
    assert "STALE / NON-LIVE SNAPSHOT" not in md


def test_cli_markdown_missing_provenance_returns_2(tmp_path: Path) -> None:
    """The CLI refuses to render a Markdown summary when the inputs file
    carries no _provenance block (exit 2, loud stderr)."""
    inputs_file = tmp_path / "in.json"
    inputs_file.write_text(json.dumps(_five_fixture_inputs()))
    rc = main(
        [
            "--inputs",
            str(inputs_file),
            "--out",
            str(tmp_path / "scorecard.json"),
            "--markdown-summary",
            str(tmp_path / "scorecard.md"),
        ]
    )
    assert rc == 2


def test_cli_markdown_stale_warn_days_flag_triggers_banner(tmp_path: Path) -> None:
    """--stale-warn-days=0 forces every non-zero-age snapshot to read stale."""
    payload: dict[str, object] = dict(_five_fixture_inputs())
    payload["_provenance"] = _provenance_block(
        live_runner=True, generated_at="2000-01-01T00:00:00+00:00"
    )
    inputs_file = tmp_path / "in.json"
    inputs_file.write_text(json.dumps(payload))
    md_file = tmp_path / "scorecard.md"
    rc = main(
        [
            "--inputs",
            str(inputs_file),
            "--out",
            str(tmp_path / "scorecard.json"),
            "--markdown-summary",
            str(md_file),
            "--stale-warn-days",
            "1",
        ]
    )
    assert rc == 0
    assert "STALE / NON-LIVE SNAPSHOT" in md_file.read_text()


def test_provenance_rejects_non_iso_generated_at() -> None:
    with pytest.raises(ValueError, match="ISO-8601"):
        SuiteCPhase1Provenance(generated_at="not-a-date", snapshot_source="x", live_runner=True)
