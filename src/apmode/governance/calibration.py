# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate threshold empirical calibration harness.

Cross-tabulates Gate 1/2 per-check verdicts against a ground-truth
correctness label so that a policy author has an *empirical* false-pass /
false-fail rate to look at before hand-tuning a `policies/*.json`
threshold — rather than tuning blind.

This module is deliberately a *measurement* harness, not a governance
path: it never mutates `policies/*.json`, never runs new R/nlmixr2 fits,
and never feeds back into `evaluate_gate1`/`evaluate_gate2` at runtime.
It replays already-computed `BackendResult`s (either hand-built test
fixtures or `results/{candidate_id}_result.json` extracted from sealed
bundles) through the existing gate evaluators and aggregates the
per-check pass/fail against a label derived from Suite A/B ground truth.

Scope decision (docs/plans/2026-07-09-qaqc-remediation.md, "Empirically
calibrate Gate1/2/3 thresholds..."): Suite B fixtures do not exist yet
(``benchmarks/suite_b/`` is currently just a README stub), so this
module only builds the measurement/aggregation/report machinery. Once
Suite B fixtures and real bundle runs exist, calibration numbers can be
produced without further engineering.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from apmode.governance.gates import evaluate_gate1, evaluate_gate2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from apmode.bundle.models import BackendResult, ImputationStabilityEntry, MissingDataDirective
    from apmode.governance.policy import GatePolicy

__all__ = [
    "CHECK_ID_GATE1",
    "CHECK_ID_GATE2",
    "DEFAULT_RECOVERY_TOLERANCE",
    "CalibrationStat",
    "CheckVerdict",
    "GateCalibrationReport",
    "Label",
    "LabeledFit",
    "aggregate_calibration",
    "label_fit",
    "replay_gate1",
    "replay_gate2",
    "write_calibration_report",
]

# Names retained purely for readability at call sites (not gate identifiers
# used elsewhere in the codebase — see the `gate_id` field on
# `GateCalibrationReport`, which is a free-text discriminator for the
# report filename, distinct from `GateResult.gate_id` which is a UUID).
CHECK_ID_GATE1 = "gate1"
CHECK_ID_GATE2 = "gate2"

Label = Literal["known_good", "known_bad"]

# Fixed scientific parameter-recovery tolerance used *only* to derive a
# ground-truth correctness label for the calibration harness (Suite A
# reference_params.json vs. a fitted BackendResult.parameter_estimates).
# This is deliberately NOT a `policies/*.json` value: it does not gate
# any governance decision (Gate 1/2/3 evaluation is untouched by this
# module) and is analogous to the existing repo convention of citing a
# fixed scientific constant inline rather than policy-versioning it (cf.
# `apmode.benchmarks.scoring._BE_GMR_LOWER`/`_BE_GMR_UPPER`, the Smith
# (2000) FDA bioequivalence goalposts). The value matches the existing
# `BenchmarkCase.param_bias_tolerance` default of 0.20 (20% relative
# error) already used by Suite A scoring
# (`apmode.benchmarks.models.BenchmarkCase`), so calibration ground truth
# and benchmark bias scoring agree on what "close to truth" means.
DEFAULT_RECOVERY_TOLERANCE: float = 0.20

# Reference-params.json keys that are never themselves structural point
# estimates to compare against a fit's `parameter_estimates` (nested
# IIV/residual-error blocks, dosing metadata, free-text notes, and assay
# LLOQ are not "recovered parameters" in the `ParameterEstimate` sense).
_REFERENCE_NON_STRUCTURAL_KEYS = frozenset(
    {"omega", "sigma", "dose_arms_mg", "note", "lloq", "generated_at", "sigma_convention"}
)


def _reference_structural_params(reference: Mapping[str, Any]) -> dict[str, float]:
    """Extract scalar structural parameters from a reference_params.json entry."""
    return {
        k: float(v)
        for k, v in reference.items()
        if k not in _REFERENCE_NON_STRUCTURAL_KEYS
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }


def label_fit(
    *,
    scenario_id: str,
    recovered_params: Mapping[str, float],
    reference: Mapping[str, Any],
    tolerance: float = DEFAULT_RECOVERY_TOLERANCE,
) -> Label:
    """Label a fit ``known_good`` or ``known_bad`` against Suite A ground truth.

    Compares every scalar structural parameter in ``reference`` (a single
    scenario entry from ``benchmarks/suite_a/reference_params.json``,
    e.g. ``reference_params["A1"]``) against ``recovered_params`` (e.g.
    ``{name: est.estimate for name, est in result.parameter_estimates.items()}``).

    A fit is ``known_bad`` when:
      - a structural parameter present in the reference is *missing* from
        ``recovered_params`` (signals a wrong structural family — the fit
        did not even attempt to estimate a parameter the DGP has), or
      - any recovered structural parameter's relative error exceeds
        ``tolerance``.

    Otherwise the fit is ``known_good``.

    Args:
        scenario_id: Suite A scenario id (e.g. ``"A1"``); used only in the
            raised error message for a reference with no structural keys.
        recovered_params: Fitted point estimates, keyed by parameter name.
        reference: One scenario's entry from ``reference_params.json``.
        tolerance: Relative-error recovery tolerance (fraction of truth).
            Defaults to :data:`DEFAULT_RECOVERY_TOLERANCE`.
    """
    truth = _reference_structural_params(reference)
    if not truth:
        msg = f"No structural parameters found in reference for scenario {scenario_id!r}"
        raise ValueError(msg)

    for name, true_value in truth.items():
        if name not in recovered_params:
            return "known_bad"
        recovered_value = float(recovered_params[name])
        if true_value == 0:
            if abs(recovered_value) > tolerance:
                return "known_bad"
            continue
        rel_error = abs(recovered_value - true_value) / abs(true_value)
        if rel_error > tolerance:
            return "known_bad"

    return "known_good"


@dataclass(frozen=True, slots=True)
class LabeledFit:
    """A single `BackendResult` tagged with its ground-truth correctness label."""

    scenario_id: str
    result: BackendResult
    label: Label
    seed_results: list[BackendResult] | None = None


@dataclass(frozen=True, slots=True)
class CheckVerdict:
    """One `(scenario_id, check_id)` row: did the check pass, and what's the true label?"""

    scenario_id: str
    check_id: str
    passed: bool
    label: Label


def replay_gate1(
    fits: Sequence[LabeledFit],
    policy: GatePolicy,
    *,
    stability: Mapping[str, ImputationStabilityEntry] | None = None,
    directive: MissingDataDirective | None = None,
) -> list[CheckVerdict]:
    """Replay Gate 1 over a labeled fit set and flatten per-check verdicts.

    Thin harness call-site over :func:`apmode.governance.gates.evaluate_gate1`
    — no new gate logic. ``stability`` is an optional per-scenario-id map
    (mirrors ``evaluate_gate1``'s per-candidate ``stability`` kwarg).
    """
    verdicts: list[CheckVerdict] = []
    for fit in fits:
        gate_result = evaluate_gate1(
            fit.result,
            policy,
            seed_results=fit.seed_results,
            stability=stability.get(fit.scenario_id) if stability else None,
            directive=directive,
        )
        verdicts.extend(
            CheckVerdict(
                scenario_id=fit.scenario_id,
                check_id=check.check_id,
                passed=check.passed,
                label=fit.label,
            )
            for check in gate_result.checks
        )
    return verdicts


def replay_gate2(
    fits: Sequence[LabeledFit],
    policy: GatePolicy,
    lane: str,
) -> list[CheckVerdict]:
    """Replay Gate 2 over a labeled fit set and flatten per-check verdicts.

    Thin harness call-site over :func:`apmode.governance.gates.evaluate_gate2`
    — no new gate logic.
    """
    verdicts: list[CheckVerdict] = []
    for fit in fits:
        gate_result = evaluate_gate2(fit.result, policy, lane)
        verdicts.extend(
            CheckVerdict(
                scenario_id=fit.scenario_id,
                check_id=check.check_id,
                passed=check.passed,
                label=fit.label,
            )
            for check in gate_result.checks
        )
    return verdicts


class CalibrationStat(BaseModel):
    """Per-check false-pass / false-fail rate against ground-truth labels.

    ``false_fail_rate`` and ``false_pass_rate`` are ``None`` (rather than
    ``0.0``) when the corresponding label population is empty — a rate of
    exactly ``0.0`` computed from zero rows is misleading (a policy author
    would read it as "verified zero false fails" when in fact nothing was
    ever measured).

    Deliberately does *not* compute a Wilson/Clopper-Pearson confidence
    interval: at Suite A/B's expected small n (~21 scenarios), per-check
    CIs are wide and easy to over-read as tighter evidence than they are.
    Flagged as a follow-up rather than in scope here — see
    ``GateCalibrationReport.notes``.
    """

    model_config = ConfigDict(frozen=True)

    check_id: str
    n_known_good: int = Field(ge=0)
    n_known_bad: int = Field(ge=0)
    false_fail_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)


def aggregate_calibration(verdicts: Sequence[CheckVerdict]) -> dict[str, CalibrationStat]:
    """Cross-tabulate per-check verdicts into false-pass / false-fail rates.

    ``false_fail_rate`` = fraction of ``known_good`` rows where the check
    did *not* pass (the check is too strict — rejects correct fits).
    ``false_pass_rate`` = fraction of ``known_bad`` rows where the check
    *did* pass (the check is too lenient — admits incorrect fits).
    """
    by_check: dict[str, list[CheckVerdict]] = defaultdict(list)
    for v in verdicts:
        by_check[v.check_id].append(v)

    stats: dict[str, CalibrationStat] = {}
    for check_id, rows in by_check.items():
        good = [r for r in rows if r.label == "known_good"]
        bad = [r for r in rows if r.label == "known_bad"]
        false_fail_rate = sum(1 for r in good if not r.passed) / len(good) if good else None
        false_pass_rate = sum(1 for r in bad if r.passed) / len(bad) if bad else None
        stats[check_id] = CalibrationStat(
            check_id=check_id,
            n_known_good=len(good),
            n_known_bad=len(bad),
            false_fail_rate=false_fail_rate,
            false_pass_rate=false_pass_rate,
        )
    return stats


class GateCalibrationReport(BaseModel):
    """A versioned, on-disk snapshot of gate-check calibration statistics.

    This report is a *reference* for policy authors setting
    `policies/*.json` thresholds — it does not auto-tune them, preserving
    the repo convention that policy is a versioned JSON artifact, never
    inferred/hard-coded.

    Lives under ``benchmarks/calibration/reports/``, not a bundle
    directory — it is not subject to `_compute_bundle_digest` or the
    `_DIGEST_EXCLUDED_RELATIVE_PATHS` lockstep contract in
    `apmode.bundle.emitter` (that contract only applies to files written
    *inside* a sealed reproducibility bundle; this report is a standalone
    benchmarking artifact and requires no digest-exclusion-set change).
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0.0"
    gate_id: str
    policy_version: str
    generated_at: str
    per_check: dict[str, CalibrationStat]
    n_scenarios_known_good: int = Field(ge=0)
    n_scenarios_known_bad: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)


def write_calibration_report(
    stats: Mapping[str, CalibrationStat],
    *,
    policy_version: str,
    out_dir: Path,
    gate_id: str = CHECK_ID_GATE1,
    n_scenarios_known_good: int,
    n_scenarios_known_bad: int,
    notes: Sequence[str] = (),
) -> Path:
    """Write a `GateCalibrationReport` to
    ``<out_dir>/<gate_id>_calibration_<policy_version>.json``.

    Uses a tmp-file + ``Path.replace`` atomic write, mirroring
    `apmode.benchmarks.suite_a_runner.write_results_atomic`.
    """
    report = GateCalibrationReport(
        gate_id=gate_id,
        policy_version=policy_version,
        generated_at=datetime.now(tz=UTC).isoformat(),
        per_check=dict(stats),
        n_scenarios_known_good=n_scenarios_known_good,
        n_scenarios_known_bad=n_scenarios_known_bad,
        notes=list(notes),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{gate_id}_calibration_{policy_version}.json"
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(report.model_dump_json(indent=2) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
