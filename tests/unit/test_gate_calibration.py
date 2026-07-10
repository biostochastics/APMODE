# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Gate 1/2 empirical calibration harness.

Covers the bite-sized TDD tasks in
docs/plans/2026-07-09-qaqc-remediation.md ("Empirically calibrate
Gate1/2/3 thresholds against Suite A/B false-pass/fail rates"):

  Task 1 — ground-truth correctness labeller (``label_fit``)
  Task 2 — gate replay over a labeled fit set (``replay_gate1``/``replay_gate2``)
  Task 3 — per-check false-pass/false-fail rate aggregation
  Task 4 — versioned calibration report artifact
  Task 5 — thin CLI/script entrypoint over sealed-bundle fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.models import BackendResult
from apmode.governance.calibration import (
    CalibrationStat,
    CheckVerdict,
    GateCalibrationReport,
    LabeledFit,
    aggregate_calibration,
    label_fit,
    replay_gate1,
    replay_gate2,
    write_calibration_report,
)
from tests.unit.test_gates import _load_policy, _make_backend_result

_REFERENCE_PATH = (
    Path(__file__).parent.parent.parent / "benchmarks" / "suite_a" / "reference_params.json"
)


def _reference_a1() -> dict[str, object]:
    return json.loads(_REFERENCE_PATH.read_text())["A1"]  # type: ignore[no-any-return]


def _bad_result(*, model_id: str | None = None, **overrides: object) -> BackendResult:
    """A `_make_backend_result()` fit with CL perturbed >2x off truth."""
    base = _make_backend_result(**overrides)  # type: ignore[arg-type]
    bad_cl = base.parameter_estimates["CL"].model_copy(update={"estimate": 11.5})
    update: dict[str, object] = {
        "parameter_estimates": {**base.parameter_estimates, "CL": bad_cl},
    }
    if model_id is not None:
        update["model_id"] = model_id
    return base.model_copy(update=update)


def _good_result(*, model_id: str | None = None, **overrides: object) -> BackendResult:
    base = _make_backend_result(**overrides)  # type: ignore[arg-type]
    if model_id is None:
        return base
    return base.model_copy(update={"model_id": model_id})


# ---------------------------------------------------------------------------
# Task 1 — Ground-truth correctness labeller
# ---------------------------------------------------------------------------


class TestGroundTruthLabel:
    def test_close_to_truth_is_known_good(self) -> None:
        reference = _reference_a1()
        recovered = {"ka": 1.48, "V": 71.0, "CL": 5.1}
        label = label_fit(scenario_id="A1", recovered_params=recovered, reference=reference)
        assert label == "known_good"

    def test_cl_off_by_more_than_2x_is_known_bad(self) -> None:
        reference = _reference_a1()
        recovered = {"ka": 1.5, "V": 70.0, "CL": 11.5}  # true CL=5 -> >2x off
        label = label_fit(scenario_id="A1", recovered_params=recovered, reference=reference)
        assert label == "known_bad"

    def test_missing_structural_parameter_is_known_bad(self) -> None:
        reference = _reference_a1()
        recovered = {"ka": 1.5, "V": 70.0}  # CL missing entirely
        label = label_fit(scenario_id="A1", recovered_params=recovered, reference=reference)
        assert label == "known_bad"

    def test_within_documented_tolerance_boundary(self) -> None:
        reference = _reference_a1()
        # CL=5 true; 19% off is within the 20% default tolerance.
        recovered = {"ka": 1.5, "V": 70.0, "CL": 5.95}
        label = label_fit(scenario_id="A1", recovered_params=recovered, reference=reference)
        assert label == "known_good"

    def test_custom_tolerance_is_respected(self) -> None:
        reference = _reference_a1()
        recovered = {"ka": 1.5, "V": 70.0, "CL": 5.15}  # 3% off
        assert (
            label_fit(
                scenario_id="A1", recovered_params=recovered, reference=reference, tolerance=0.01
            )
            == "known_bad"
        )
        assert (
            label_fit(
                scenario_id="A1", recovered_params=recovered, reference=reference, tolerance=0.05
            )
            == "known_good"
        )

    def test_reference_with_no_structural_params_raises(self) -> None:
        with pytest.raises(ValueError, match="No structural parameters"):
            label_fit(
                scenario_id="A_empty",
                recovered_params={"CL": 5.0},
                reference={"note": "no numeric structural params here"},
            )


# ---------------------------------------------------------------------------
# Task 2 — Gate replay over a labeled fit set
# ---------------------------------------------------------------------------


class TestGateReplay:
    def test_replay_gate1_flattens_one_row_per_check(self) -> None:
        policy = _load_policy("submission")
        good = LabeledFit(scenario_id="A1", result=_make_backend_result(), label="known_good")
        bad = LabeledFit(
            scenario_id="A1_bad", result=_bad_result(converged=False), label="known_bad"
        )

        verdicts = replay_gate1([good, bad], policy)

        assert all(isinstance(v, CheckVerdict) for v in verdicts)
        good_check_ids = {v.check_id for v in verdicts if v.scenario_id == "A1"}
        bad_check_ids = {v.check_id for v in verdicts if v.scenario_id == "A1_bad"}
        assert good_check_ids == bad_check_ids  # same gate, same checks per candidate
        assert "convergence" in good_check_ids

        # The known-good fit passes convergence; the known-bad (non-converged)
        # fit fails it — this is the check we deliberately stressed.
        good_convergence = next(
            v for v in verdicts if v.scenario_id == "A1" and v.check_id == "convergence"
        )
        bad_convergence = next(
            v for v in verdicts if v.scenario_id == "A1_bad" and v.check_id == "convergence"
        )
        assert good_convergence.passed is True
        assert good_convergence.label == "known_good"
        assert bad_convergence.passed is False
        assert bad_convergence.label == "known_bad"

    def test_replay_gate2_flattens_one_row_per_check(self) -> None:
        policy = _load_policy("submission")
        good = LabeledFit(scenario_id="A1", result=_make_backend_result(), label="known_good")

        verdicts = replay_gate2([good], policy, "submission")

        assert verdicts
        assert all(v.scenario_id == "A1" and v.label == "known_good" for v in verdicts)
        assert {"interpretable_parameterization", "shrinkage", "identifiability"}.issubset(
            {v.check_id for v in verdicts}
        )


# ---------------------------------------------------------------------------
# Task 3 — Per-check false-pass / false-fail rate aggregation
# ---------------------------------------------------------------------------


class TestAggregateCalibration:
    def test_exact_rates_on_hand_built_table(self) -> None:
        # 4 known-good rows for "convergence": 3 pass, 1 fails -> false_fail_rate = 0.25
        # 4 known-bad rows for "convergence": 1 passes, 3 fail -> false_pass_rate = 0.25
        verdicts = [
            CheckVerdict(
                scenario_id="g1", check_id="convergence", passed=True, label="known_good"
            ),
            CheckVerdict(
                scenario_id="g2", check_id="convergence", passed=True, label="known_good"
            ),
            CheckVerdict(
                scenario_id="g3", check_id="convergence", passed=True, label="known_good"
            ),
            CheckVerdict(
                scenario_id="g4", check_id="convergence", passed=False, label="known_good"
            ),
            CheckVerdict(
                scenario_id="b1", check_id="convergence", passed=False, label="known_bad"
            ),
            CheckVerdict(
                scenario_id="b2", check_id="convergence", passed=False, label="known_bad"
            ),
            CheckVerdict(
                scenario_id="b3", check_id="convergence", passed=False, label="known_bad"
            ),
            CheckVerdict(scenario_id="b4", check_id="convergence", passed=True, label="known_bad"),
        ]

        stats = aggregate_calibration(verdicts)

        assert set(stats) == {"convergence"}
        stat = stats["convergence"]
        assert isinstance(stat, CalibrationStat)
        assert stat.n_known_good == 4
        assert stat.n_known_bad == 4
        assert stat.false_fail_rate == pytest.approx(0.25)
        assert stat.false_pass_rate == pytest.approx(0.25)

    def test_empty_label_population_yields_none_not_zero(self) -> None:
        verdicts = [
            CheckVerdict(
                scenario_id="g1", check_id="convergence", passed=True, label="known_good"
            ),
        ]
        stats = aggregate_calibration(verdicts)
        stat = stats["convergence"]
        assert stat.n_known_good == 1
        assert stat.n_known_bad == 0
        assert stat.false_fail_rate == pytest.approx(0.0)
        assert stat.false_pass_rate is None

    def test_multiple_check_ids_are_independent(self) -> None:
        verdicts = [
            CheckVerdict(
                scenario_id="g1", check_id="convergence", passed=True, label="known_good"
            ),
            CheckVerdict(
                scenario_id="g1", check_id="cwres_mean", passed=False, label="known_good"
            ),
        ]
        stats = aggregate_calibration(verdicts)
        assert set(stats) == {"convergence", "cwres_mean"}
        assert stats["convergence"].false_fail_rate == pytest.approx(0.0)
        assert stats["cwres_mean"].false_fail_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task 4 — Versioned calibration report artifact
# ---------------------------------------------------------------------------


class TestCalibrationReport:
    def test_write_and_round_trip(self, tmp_path: Path) -> None:
        stats = {
            "convergence": CalibrationStat(
                check_id="convergence",
                n_known_good=4,
                n_known_bad=4,
                false_fail_rate=0.25,
                false_pass_rate=0.25,
            )
        }

        out_path = write_calibration_report(
            stats,
            policy_version="0.6.0",
            out_dir=tmp_path,
            gate_id="gate1",
            n_scenarios_known_good=4,
            n_scenarios_known_bad=4,
            notes=["Wilson/Clopper-Pearson CIs deliberately out of scope; n is small."],
        )

        assert out_path == tmp_path / "gate1_calibration_0.6.0.json"
        assert out_path.exists()

        report = GateCalibrationReport.model_validate_json(out_path.read_text())
        assert report.gate_id == "gate1"
        assert report.policy_version == "0.6.0"
        assert report.n_scenarios_known_good == 4
        assert report.n_scenarios_known_bad == 4
        assert report.per_check["convergence"].false_fail_rate == pytest.approx(0.25)
        assert report.notes

    def test_report_path_is_not_a_bundle_artifact(self, tmp_path: Path) -> None:
        """Report path lives under benchmarks/, never inside a sealed bundle dir.

        No entry is needed in `_DIGEST_EXCLUDED_RELATIVE_PATHS`
        (`apmode.bundle.emitter`) because this file is never written
        inside a `BundleEmitter`-managed `run_dir` in the first place —
        confirmed here so a future reviewer doesn't have to re-derive it.
        """
        out_path = write_calibration_report(
            {},
            policy_version="0.6.0",
            out_dir=tmp_path,
            gate_id="gate1",
            n_scenarios_known_good=0,
            n_scenarios_known_bad=0,
        )
        assert "runs" not in out_path.parts
        assert "bundle" not in out_path.parts


# ---------------------------------------------------------------------------
# Task 5 — Thin CLI/script entrypoint wired to (fixture) sealed bundles
# ---------------------------------------------------------------------------


def _write_result_json(results_dir: Path, model_id: str, result: BackendResult) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{model_id}_result.json").write_text(result.model_dump_json(indent=2))


class TestCalibrationEntrypoint:
    """End-to-end: fixture 'sealed bundles' -> labeled fits -> report.

    Builds a small set of bundle-shaped directories under ``tmp_path``
    (``<bundle>/results/<candidate_id>_result.json``) representing one
    Suite-A-like known-good fit and one deliberately perturbed
    (Suite-B-like) known-bad fit, then drives the full pipeline through
    the ``scripts/run_gate_calibration.py`` entrypoint.
    """

    def test_end_to_end_produces_valid_report(self, tmp_path: Path) -> None:
        import sys

        scripts_dir = Path(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        try:
            import run_gate_calibration as rgc
        finally:
            sys.path.remove(str(scripts_dir))

        good_bundle = tmp_path / "bundle_good"
        bad_bundle = tmp_path / "bundle_bad"
        _write_result_json(
            good_bundle / "results",
            "A1_candidate_good",
            _good_result(model_id="A1_candidate_good"),
        )
        _write_result_json(
            bad_bundle / "results",
            "A1_candidate_bad",
            _bad_result(model_id="A1_candidate_bad"),
        )

        out_dir = tmp_path / "reports"
        exit_code = rgc.main(
            [
                str(tmp_path / "bundle_*"),
                "--policy",
                "submission",
                "--gate",
                "1",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0

        report_paths = list(out_dir.glob("gate1_calibration_*.json"))
        assert len(report_paths) == 1
        report = GateCalibrationReport.model_validate_json(report_paths[0].read_text())
        assert report.n_scenarios_known_good >= 1
        assert report.n_scenarios_known_bad >= 1
        assert report.per_check  # at least one check aggregated
