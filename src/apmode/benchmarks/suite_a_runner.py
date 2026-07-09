# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite A live-fit eta-recovery runner (PRD §5 Suite A, per-subject ETAs).

For each Suite A scenario (A1-A8 in :mod:`apmode.benchmarks.suite_a`,
minus A7) this module composes the existing pieces into a single
producer of ``benchmarks/suite_a/eta_recovery_report.json``:

  1. Resolve the scenario's data CSV via
     :func:`apmode.benchmarks.suite_a.scenario_dataset_paths` (first
     replicate only) and its matching ``<stem>_eta.csv`` ground-truth
     file.
  2. Ingest the CSV into a ``DataManifest``
     (:func:`apmode.data.ingest.ingest_nonmem_csv`).
  3. Build ``initial_estimates`` from
     :data:`apmode.benchmarks.suite_a.REFERENCE_PARAMS`, filtered down
     to :meth:`~apmode.dsl.ast_models.DSLSpec.calibration_param_names`
     — the DSL's own documented distinction between "structural
     parameter names" (used for typing/validation) and "calibration
     parameter names" (the subset that needs an ``initial:`` value and
     is meaningful as an nlmixr2 THETA). This matters for A3: its
     ``REFERENCE_PARAMS["A3"]`` carries ``"n": 3.0``, the Transit
     absorption chain-length count, not a THETA. Empirically
     ``emit_nlmixr2`` does *not* raise on an extra ``"n"`` key (the
     Transit branch of ``_emit_structural_ini`` happens to consume it
     via ``ov.get("n", abs_mod.n)``), but filtering through
     ``calibration_param_names()`` is the documented, forward-compatible
     contract rather than relying on that per-module lookup shape —
     future structural-only keys (Erlang, NODE weights, SumIG's ``k``)
     are not guaranteed to be similarly harmless if passed through.
  4. Run a single ``Nlmixr2Runner.run`` fit per scenario. Any exception
     is caught and recorded as ``status="fit_error"`` — one scenario's
     failure must never abort the rest of the suite.
  5. Score per-subject ETA recovery via
     :func:`apmode.benchmarks.scoring.score_eta_recovery` against the
     ``<stem>_eta.csv`` ground truth
     (:func:`apmode.benchmarks.suite_a.load_reference_eta`).
  6. Atomic write of the aggregate report JSON (tmp + rename) so a
     SIGKILL mid-write never leaves the CI-consumed report half-written.

A7 is always skipped (``status="skipped"``): its ground truth uses
``NODEAbsorption`` gated by ``ExperimentalFlags(node=True)``, and no
nlmixr2 backend exists for NODE capability tags
(``apmode.dsl.capabilities._NODE_EXPERIMENTAL_TAGS`` —
``emit_nlmixr2`` raises ``NotImplementedError`` for
``spec.has_node_modules()``), mirroring
``suite_b_runner._NODE_BACKED_CASES``.

A8 nuance
---------
A8's ground truth has monotonic CL autoinduction
(``CL(t, CRCL) = CL0 * (CRCL/90)^theta * exp(-delta * t / 24)``) with no
DSL primitive (see ``scenario_a8()``'s own docstring in
``apmode.benchmarks.suite_a``). APMODE fits a static ``CL``, which
biases the recovered *THETA* toward the time-averaged truth — that bias
does not necessarily show up as a per-subject ETA bias directly, but a
consequence is that ``eta_rmse`` for A8 may run higher than the other
scenarios as a downstream symptom of the THETA misspecification (the
individual ETAs partly compensate for the model's inability to track
the time-varying covariate effect). This is expected and documented,
not a defect in this runner to "fix".

CLI
---
``python -m apmode.benchmarks.suite_a_runner --out
benchmarks/suite_a/eta_recovery_report.json --suite-dir benchmarks/suite_a
[--scenarios A1,A2,...] [--seed 20260708] [--timeout-seconds 600]
[--rscript Rscript] [--estimation focei]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.benchmarks.scoring import score_eta_recovery
from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
    load_reference_eta,
    scenario_dataset_paths,
)
from apmode.data.ingest import ingest_nonmem_csv

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from apmode.dsl.ast_models import DSLSpec

logger = logging.getLogger(__name__)

# Per-scenario outcome status.
ScenarioStatus = Literal["ok", "no_data", "no_eta_ground_truth", "fit_error", "skipped"]

# Scenarios skipped at runtime — no stable nlmixr2 backend exists for
# NODE-capability-tagged modules (apmode.dsl.capabilities._NODE_EXPERIMENTAL_TAGS).
# Mirrors suite_b_runner._NODE_BACKED_CASES.
_NODE_SKIPPED_SCENARIOS: dict[str, str] = {
    "A7": (
        "A7 uses NODEAbsorption with ExperimentalFlags(node=True); no stable "
        "nlmixr2 backend exists for NODE capability tags "
        "(apmode.dsl.capabilities._NODE_EXPERIMENTAL_TAGS) — emit_nlmixr2 "
        "raises NotImplementedError for spec.has_node_modules()."
    ),
}

_EXIT_OK: int = 0
_EXIT_USAGE: int = 2
_EXIT_VALIDATION: int = 3
_EXIT_FIT_FAILURE: int = 5


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioEtaResult:
    """Per-scenario Suite A eta-recovery output."""

    scenario_id: str
    status: ScenarioStatus
    dataset_csv: str | None = None
    eta_csv: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    converged: bool | None = None
    eta_rmse: dict[str, float] = field(default_factory=dict)
    n_subjects_true: int = 0
    n_subjects_fitted: int = 0
    wall_time_seconds: float = 0.0
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Initial estimates: filter REFERENCE_PARAMS to calibratable keys
# ---------------------------------------------------------------------------


def _calibration_initial_estimates(
    spec: DSLSpec, reference_params: dict[str, float]
) -> dict[str, float]:
    """Filter ``REFERENCE_PARAMS[scenario_id]`` to nlmixr2-calibratable keys.

    ``DSLSpec.calibration_param_names()`` is the subset of
    ``structural_param_names()`` that requires an ``initial:`` value —
    it excludes Transit/Erlang ``n`` (chain-length topology, not a
    THETA) and NODE weight names. See the module docstring for why this
    is the documented filter to use rather than relying on
    ``emit_nlmixr2`` silently tolerating extra keys.
    """
    calibratable = set(spec.calibration_param_names())
    return {name: value for name, value in reference_params.items() if name in calibratable}


# ---------------------------------------------------------------------------
# Per-scenario driver
# ---------------------------------------------------------------------------


async def run_scenario(
    scenario_id: str,
    factory: Callable[[], DSLSpec],
    *,
    runner: Nlmixr2Runner,
    suite_dir: Path,
    seed: int,
    timeout_seconds: int | None,
) -> ScenarioEtaResult:
    """Drive one Suite A scenario through resolve -> fit -> eta-score.

    Skips A7 (NODE-backed, no stable nlmixr2 path) with a clear
    ``skipped=True`` signal. Never raises on a fit failure — one
    scenario's exception is recorded as ``status="fit_error"`` so the
    remaining scenarios still run.
    """
    if scenario_id in _NODE_SKIPPED_SCENARIOS:
        return ScenarioEtaResult(
            scenario_id=scenario_id,
            status="skipped",
            skipped=True,
            skip_reason=_NODE_SKIPPED_SCENARIOS[scenario_id],
        )

    data_paths = scenario_dataset_paths(suite_dir, scenario_id)
    if not data_paths:
        return ScenarioEtaResult(scenario_id=scenario_id, status="no_data")
    csv_path = data_paths[0]

    eta_csv_path = csv_path.with_name(csv_path.stem + "_eta.csv")
    if not eta_csv_path.exists():
        return ScenarioEtaResult(
            scenario_id=scenario_id,
            status="no_eta_ground_truth",
            dataset_csv=str(csv_path),
        )

    spec = factory()
    manifest, _df = ingest_nonmem_csv(csv_path)
    reference_params = REFERENCE_PARAMS.get(scenario_id, {})
    initial_estimates = _calibration_initial_estimates(spec, reference_params)

    try:
        result = await runner.run(
            spec,
            manifest,
            initial_estimates,
            seed,
            data_path=csv_path,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.warning("scenario %s: fit raised %s: %s", scenario_id, type(exc).__name__, exc)
        return ScenarioEtaResult(
            scenario_id=scenario_id,
            status="fit_error",
            dataset_csv=str(csv_path),
            eta_csv=str(eta_csv_path),
            error_message=f"{type(exc).__name__}: {exc}",
        )

    true_eta = load_reference_eta(eta_csv_path)
    fitted_eta = result.per_subject_eta
    eta_rmse = score_eta_recovery(true_eta, fitted_eta)

    return ScenarioEtaResult(
        scenario_id=scenario_id,
        status="ok",
        dataset_csv=str(csv_path),
        eta_csv=str(eta_csv_path),
        converged=bool(result.converged),
        eta_rmse=eta_rmse,
        n_subjects_true=len(true_eta),
        n_subjects_fitted=len(fitted_eta),
        wall_time_seconds=float(result.wall_time_seconds or 0.0),
    )


async def run_all(
    scenarios: Sequence[tuple[str, Callable[[], DSLSpec]]],
    *,
    runner: Nlmixr2Runner,
    suite_dir: Path,
    seed: int,
    timeout_seconds: int | None,
) -> dict[str, ScenarioEtaResult]:
    """Drive multiple Suite A scenarios sequentially.

    Sequential for the same reason as Suite B/C — one R subprocess at a
    time keeps CI resource contention predictable.
    """
    out: dict[str, ScenarioEtaResult] = {}
    for scenario_id, factory in scenarios:
        out[scenario_id] = await run_scenario(
            scenario_id,
            factory,
            runner=runner,
            suite_dir=suite_dir,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
    return out


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _serialize_scenario(result: ScenarioEtaResult) -> dict[str, object]:
    return {
        "scenario_id": result.scenario_id,
        "status": result.status,
        "dataset_csv": result.dataset_csv,
        "eta_csv": result.eta_csv,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "converged": result.converged,
        "eta_rmse": result.eta_rmse,
        "n_subjects_true": result.n_subjects_true,
        "n_subjects_fitted": result.n_subjects_fitted,
        "wall_time_seconds": result.wall_time_seconds,
        "error_message": result.error_message,
    }


def write_results_atomic(path: Path, results: dict[str, ScenarioEtaResult]) -> None:
    """Write the Suite A eta-recovery report JSON via tmp-file + ``Path.replace``."""
    payload: dict[str, dict[str, object]] = {
        scenario_id: _serialize_scenario(r) for scenario_id, r in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

_DEFAULT_SCENARIO_IDS: list[str] = [
    scenario_id for scenario_id, _ in ALL_SCENARIOS if scenario_id not in _NODE_SKIPPED_SCENARIOS
]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m apmode.benchmarks.suite_a_runner",
        description="Run the Suite A synthetic-recovery eta-scoring loop.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/suite_a/eta_recovery_report.json"),
        help="Destination path for the eta_recovery_report.json the CI workflow reads.",
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path("benchmarks/suite_a"),
        help="Directory containing the Suite A simulator output CSVs.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(_DEFAULT_SCENARIO_IDS),
        help="Comma-separated scenario ids (defaults to ALL_SCENARIOS minus A7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260708,
        help="RNG seed passed to every scenario's fit (default 20260708).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help=(
            "Per-fit timeout. Default 600 (10 min) — Suite A models are small "
            "synthetic 1-2cmt fits, not Suite B's real-data cases."
        ),
    )
    parser.add_argument("--rscript", type=str, default="Rscript")
    parser.add_argument(
        "--estimation",
        type=str,
        default=None,
        help="Comma-separated nlmixr2 estimation methods (e.g. 'focei' or 'saem,focei').",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scenario_ids = [sid.strip() for sid in args.scenarios.split(",") if sid.strip()]
    if not scenario_ids:
        sys.stderr.write("error: --scenarios resolved to an empty list\n")
        return _EXIT_USAGE

    by_id = dict(ALL_SCENARIOS)
    unknown = sorted(set(scenario_ids) - set(by_id))
    if unknown:
        sys.stderr.write(f"error: unknown scenario(s) {unknown} — valid ids: {sorted(by_id)}\n")
        return _EXIT_USAGE
    selected = [(sid, by_id[sid]) for sid in scenario_ids]

    estimation_methods: list[str] | None = None
    if args.estimation:
        estimation_methods = [m.strip() for m in args.estimation.split(",") if m.strip()]
        if not estimation_methods:
            sys.stderr.write("error: --estimation resolved to an empty method list\n")
            return _EXIT_USAGE

    with tempfile.TemporaryDirectory(prefix="apmode_suite_a_") as tmp_root:
        work_dir = Path(tmp_root) / "work"

        try:
            runner = Nlmixr2Runner(
                work_dir=work_dir,
                r_executable=args.rscript,
                estimation=estimation_methods,
            )
        except FileNotFoundError as exc:
            sys.stderr.write(f"error: cannot start Nlmixr2Runner: {exc}\n")
            return _EXIT_USAGE

        try:
            results = asyncio.run(
                run_all(
                    selected,
                    runner=runner,
                    suite_dir=args.suite_dir,
                    seed=args.seed,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        except (RuntimeError, ValueError) as exc:
            sys.stderr.write(f"error: fit pipeline failed: {exc}\n")
            return _EXIT_FIT_FAILURE

        try:
            write_results_atomic(args.out, results)
        except OSError as exc:
            sys.stderr.write(f"error: failed to write {args.out}: {exc}\n")
            return _EXIT_VALIDATION

        for scenario_id, r in results.items():
            if r.skipped:
                sys.stderr.write(f"info: {scenario_id}: skipped — {r.skip_reason}\n")
                continue
            sys.stderr.write(
                f"info: {scenario_id}: status={r.status} converged={r.converged} "
                f"eta_rmse={r.eta_rmse}\n"
            )

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ScenarioEtaResult",
    "main",
    "run_all",
    "run_scenario",
    "write_results_atomic",
]
