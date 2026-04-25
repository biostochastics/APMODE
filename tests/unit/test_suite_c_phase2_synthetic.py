# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the synthetic Phase 2 scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.benchmarks.suite_c_phase2_synthetic import (
    DEFAULT_JITTER_SIGMA,
    SYNTHETIC_BANNER,
    SyntheticPhase2Scorecard,
    main,
    score_fixture_synthetic,
    score_synthetic_phase2,
    synthesize_expert_npes,
)


class TestSynthesizeExpertNPEs:
    def test_first_sample_is_literature_anchor(self) -> None:
        out = synthesize_expert_npes(10.0, panel_size=5, jitter_sigma=0.2, seed=1)
        assert out[0] == pytest.approx(10.0)
        assert len(out) == 5

    def test_panel_size_one_returns_literature_only(self) -> None:
        out = synthesize_expert_npes(10.0, panel_size=1, jitter_sigma=0.2, seed=1)
        assert out == (10.0,)

    def test_jitter_is_deterministic_under_seed(self) -> None:
        a = synthesize_expert_npes(10.0, panel_size=5, jitter_sigma=0.2, seed=1)
        b = synthesize_expert_npes(10.0, panel_size=5, jitter_sigma=0.2, seed=1)
        assert a == b

    def test_jitter_changes_with_seed(self) -> None:
        a = synthesize_expert_npes(10.0, panel_size=5, jitter_sigma=0.2, seed=1)
        b = synthesize_expert_npes(10.0, panel_size=5, jitter_sigma=0.2, seed=2)
        assert a != b

    def test_log_normal_is_strictly_positive(self) -> None:
        out = synthesize_expert_npes(10.0, panel_size=20, jitter_sigma=2.0, seed=1)
        # Even with large sigma, log-normal samples are > 0.
        assert all(v > 0 for v in out)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            synthesize_expert_npes(0.0, panel_size=3, jitter_sigma=0.2, seed=1)
        with pytest.raises(ValueError):
            synthesize_expert_npes(10.0, panel_size=0, jitter_sigma=0.2, seed=1)
        with pytest.raises(ValueError):
            synthesize_expert_npes(10.0, panel_size=3, jitter_sigma=0.0, seed=1)
        with pytest.raises(ValueError):
            synthesize_expert_npes(float("inf"), panel_size=3, jitter_sigma=0.2, seed=1)


class TestScoreFixtureSynthetic:
    def test_apmode_well_below_anchor_beats_panel(self) -> None:
        score = score_fixture_synthetic(
            fixture_id="theo",
            npe_apmode=5.0,
            npe_literature=10.0,
            panel_size=5,
            jitter_sigma=0.1,
            seed=42,
        )
        assert score.beats_synthetic_median is True
        assert score.expert_median_npe == pytest.approx(10.0, rel=0.2)

    def test_apmode_above_anchor_loses_to_panel(self) -> None:
        score = score_fixture_synthetic(
            fixture_id="theo",
            npe_apmode=20.0,
            npe_literature=10.0,
            panel_size=5,
            jitter_sigma=0.1,
            seed=42,
        )
        assert score.beats_synthetic_median is False

    def test_default_seed_uses_fixture_id_hash(self) -> None:
        # Two different fixtures with the same anchor get *different*
        # synthetic panels because the seed depends on the fixture id.
        a = score_fixture_synthetic(fixture_id="alpha", npe_apmode=10.0, npe_literature=10.0)
        b = score_fixture_synthetic(fixture_id="beta", npe_apmode=10.0, npe_literature=10.0)
        assert a.synthetic_expert_npes != b.synthetic_expert_npes


class TestScoreSyntheticPhase2:
    def test_full_pass_when_apmode_dominates(self) -> None:
        inputs = {
            "alpha": {"npe_apmode": 5.0, "npe_literature": 10.0},
            "beta": {"npe_apmode": 4.5, "npe_literature": 9.0},
            "gamma": {"npe_apmode": 6.0, "npe_literature": 12.0},
        }
        sc = score_synthetic_phase2(inputs, panel_size=5, jitter_sigma=0.1)
        assert sc.synthetic is True
        assert sc.banner == SYNTHETIC_BANNER
        assert sc.n_datasets == 3
        assert sc.passes_gate is True

    def test_full_miss_when_apmode_loses_everywhere(self) -> None:
        inputs = {
            "alpha": {"npe_apmode": 20.0, "npe_literature": 10.0},
            "beta": {"npe_apmode": 18.0, "npe_literature": 9.0},
            "gamma": {"npe_apmode": 24.0, "npe_literature": 12.0},
        }
        sc = score_synthetic_phase2(inputs, panel_size=5, jitter_sigma=0.1)
        assert sc.passes_gate is False
        assert sc.fraction_beats_synthetic_median == 0.0

    def test_target_threshold_is_inclusive_at_60_percent(self) -> None:
        # 3 of 5 = 60% — exactly the target — must pass.
        inputs = {
            "good_a": {"npe_apmode": 5.0, "npe_literature": 10.0},
            "good_b": {"npe_apmode": 5.0, "npe_literature": 10.0},
            "good_c": {"npe_apmode": 5.0, "npe_literature": 10.0},
            "bad_a": {"npe_apmode": 50.0, "npe_literature": 10.0},
            "bad_b": {"npe_apmode": 50.0, "npe_literature": 10.0},
        }
        sc = score_synthetic_phase2(inputs, panel_size=3, jitter_sigma=0.1, target=0.60)
        assert sc.fraction_beats_synthetic_median == pytest.approx(0.6)
        assert sc.passes_gate is True

    def test_empty_inputs_does_not_pass(self) -> None:
        sc = score_synthetic_phase2({})
        assert sc.passes_gate is False
        assert sc.n_datasets == 0
        assert sc.fraction_beats_synthetic_median is None

    def test_scorecard_round_trips_through_pydantic(self) -> None:
        inputs = {"alpha": {"npe_apmode": 5.0, "npe_literature": 10.0}}
        sc = score_synthetic_phase2(inputs)
        round_tripped = SyntheticPhase2Scorecard.model_validate_json(sc.model_dump_json())
        assert round_tripped == sc


class TestMainCLI:
    def _write_inputs(self, tmp_path: Path, payload: dict[str, dict[str, float]]) -> Path:
        inputs = tmp_path / "phase1_npe_inputs.json"
        inputs.write_text(json.dumps(payload))
        return inputs

    def test_pass_returns_zero_and_writes_outputs(self, tmp_path: Path) -> None:
        inputs = self._write_inputs(
            tmp_path,
            {
                "alpha": {"npe_apmode": 5.0, "npe_literature": 10.0},
                "beta": {"npe_apmode": 4.5, "npe_literature": 9.0},
                "gamma": {"npe_apmode": 6.0, "npe_literature": 12.0},
            },
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
                "--jitter-sigma",
                str(DEFAULT_JITTER_SIGMA),
            ]
        )
        assert rc == 0
        scorecard = json.loads(out.read_text())
        assert scorecard["synthetic"] is True
        assert scorecard["passes_gate"] is True
        assert "SYNTHETIC METHODOLOGY VALIDATION ONLY" in md.read_text()

    def test_miss_returns_one(self, tmp_path: Path) -> None:
        inputs = self._write_inputs(
            tmp_path,
            {
                "alpha": {"npe_apmode": 20.0, "npe_literature": 10.0},
                "beta": {"npe_apmode": 18.0, "npe_literature": 9.0},
                "gamma": {"npe_apmode": 24.0, "npe_literature": 12.0},
            },
        )
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

    def test_inputs_missing_npe_field_returns_three(self, tmp_path: Path) -> None:
        inputs = self._write_inputs(tmp_path, {"alpha": {"npe_apmode": 5.0}})  # type: ignore[dict-item]
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

    def test_negative_panel_size_returns_two(self, tmp_path: Path) -> None:
        inputs = self._write_inputs(
            tmp_path,
            {"alpha": {"npe_apmode": 5.0, "npe_literature": 10.0}},
        )
        rc = main(
            [
                "--inputs",
                str(inputs),
                "--out",
                str(tmp_path / "scorecard.json"),
                "--markdown-summary",
                str(tmp_path / "scorecard.md"),
                "--panel-size",
                "0",
            ]
        )
        assert rc == 2
