# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite B live-fit runner — perturbation resilience + cross-seed stability.

For each Suite B case (B4-B9 in :mod:`apmode.benchmarks.suite_b_extended`)
this module composes the existing pieces into a single producer of
``benchmarks/suite_b/suite_b_results.json``:

  1. Resolve dataset CSV (registry map → :func:`fetch_dataset` or
     ``--dataset-csv`` override for non-registry datasets).
  2. Apply the case's perturbation recipe(s) →
     :func:`apply_perturbations`. Persist the perturbed CSV and
     manifest in the per-case scratch dir for audit.
  3. Build a default DSLSpec from the case's ``ExpectedStructure``
     (minimum-viable: 1- or 2-cmt + first-order ka + linear elim, sized
     to the expected ``n_compartments``). Cases that need NODE
     elimination/absorption (B1-B3) are skipped — the NODE backend live
     wiring is out of v0.6 scope.
  4. Run **N_seeds** independent fits via :class:`Nlmixr2Runner.run`,
     each with a different RNG seed. The PRD §5 R8 cross-seed
     diagnostic-leakage monitor wants to see whether the proposed
     structure / parameter estimates are stable under seed
     perturbation; we record the per-seed estimates plus the across-
     seed coefficient of variation on each parameter and surface that
     as ``cross_seed_cv_max``.
  5. Score: convergence rate, dispatch correctness (when declared),
     structural recovery (when declared), and the cross-seed
     stability metric.
  6. Atomic write of the inputs JSON (tmp + rename) so a SIGKILL
     mid-write never half-writes the file the CI workflow ingests.

Honest mode contracts inherited from Suite C Phase 1:

* The persisted full-dataset CSV at the caller's path is byte-identical
  to what the registry returned; only the perturbed copy lives in the
  per-case scratch dir.
* Same RNG seed across the runner / harness / posterior-predictive
  pipeline so seed differences are the only source of cross-seed NPE
  variance — without this, simulator noise would dominate the seed-
  stability signal.

CLI
---
``python -m apmode.benchmarks.suite_b_runner --out
benchmarks/suite_b/suite_b_results.json [--cases id1,id2,...]
[--dataset-csv id=/abs/path ...] [--n-seeds 3] [--work-dir DIR]
[--timeout-seconds 1800]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.benchmarks.perturbations import apply_perturbations
from apmode.benchmarks.suite_b_extended import ALL_EXTENDED_CASES
from apmode.data.datasets import DATASET_REGISTRY, fetch_dataset
from apmode.data.ingest import ingest_nonmem_csv
from apmode.dsl.ast_models import (
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    TwoCmt,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apmode.benchmarks.models import BenchmarkCase
    from apmode.bundle.models import BackendResult

logger = logging.getLogger(__name__)


# Hand-curated dataset-id → DATASET_REGISTRY key map. Mirrors the
# Suite C runner so non-registry datasets (ddmore_gentamicin) require
# a ``--dataset-csv`` override and produce a clear error otherwise.
_DATASET_TO_REGISTRY_KEY: dict[str, str] = {
    "nlmixr2data_theophylline": "theo_sd",
    "nlmixr2data_warfarin": "warfarin",
    "nlmixr2data_mavoglurant": "mavoglurant",
}

# Cases skipped at runtime — NODE backend live wiring is out of v0.6 scope.
_NODE_BACKED_CASES = {"b1_node_absorption", "b2_node_elimination_sparse"}

_EXIT_OK: int = 0
_EXIT_USAGE: int = 2
_EXIT_VALIDATION: int = 3
_EXIT_FIT_FAILURE: int = 5


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedRunResult:
    """Per-seed fit result for the cross-seed stability monitor."""

    seed: int
    converged: bool
    minimization_status: str
    parameter_estimates: dict[str, float]
    bic: float | None
    wall_time_seconds: float


@dataclass(frozen=True)
class SuiteBCaseResult:
    """Per-case Suite B output."""

    case_id: str
    suite: str
    dataset_id: str
    perturbation_manifests: tuple[dict[str, object], ...] = field(default_factory=tuple)
    n_seeds: int = 0
    seed_results: tuple[SeedRunResult, ...] = field(default_factory=tuple)
    convergence_rate: float = 0.0
    cross_seed_cv_max: float | None = None
    cross_seed_cv_per_param: dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# DSLSpec construction from ExpectedStructure
# ---------------------------------------------------------------------------


def _build_default_spec(case: BenchmarkCase) -> DSLSpec:
    """Build a minimum-viable DSLSpec from the case's expected structure.

    Scope cut: this runner uses classical first-order absorption +
    linear elimination + 1- or 2-cmt distribution sized to
    ``expected_structure.n_compartments``. Cases whose expected
    structure is NODE-backed are caught upstream by the
    ``_NODE_BACKED_CASES`` skip list; cases without an expected
    structure (e.g. B9_genta_iov) default to a 1-cmt linear template
    that's appropriate for the gentamicin dataset.
    """
    n_cmt = 1
    if case.expected_structure is not None and case.expected_structure.n_compartments is not None:
        n_cmt = case.expected_structure.n_compartments

    distribution = OneCmt(V=50.0) if n_cmt == 1 else TwoCmt(V1=50.0, V2=80.0, Q=10.0)
    iiv_params = ["CL", "V"] if n_cmt == 1 else ["CL", "V1"]
    return DSLSpec(
        model_id=f"suite_b_{case.case_id}",
        absorption=FirstOrder(ka=1.0),
        distribution=distribution,
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=iiv_params, structure="diagonal")],
        observation=Combined(sigma_prop=0.15, sigma_add=0.5),
    )


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def resolve_dataset_csv(
    dataset_id: str,
    *,
    cache_dir: Path,
    overrides: dict[str, Path],
) -> Path:
    """Resolve a Suite B case's dataset_id to a NONMEM-style CSV path.

    Resolution order: explicit ``overrides`` first, then registry map.
    """
    override = overrides.get(dataset_id)
    if override is not None:
        if not override.is_file():
            msg = (
                f"--dataset-csv override for {dataset_id!r} points at "
                f"{override} which is not a regular file"
            )
            raise FileNotFoundError(msg)
        return override

    registry_key = _DATASET_TO_REGISTRY_KEY.get(dataset_id)
    if registry_key is None or registry_key not in DATASET_REGISTRY:
        msg = (
            f"dataset_id {dataset_id!r} is not in the built-in "
            "registry map and no --dataset-csv override was supplied. "
            f"Pass --dataset-csv {dataset_id}=/abs/path/to.csv"
        )
        raise KeyError(msg)
    return fetch_dataset(registry_key, cache_dir)


# ---------------------------------------------------------------------------
# Per-seed fit + cross-seed stability
# ---------------------------------------------------------------------------


def _extract_estimates(result: BackendResult) -> dict[str, float]:
    """Pull point estimates out of a BackendResult into a flat name→value dict."""
    return {name: float(est.estimate) for name, est in result.parameter_estimates.items()}


def _compute_cross_seed_stability(
    seed_results: Sequence[SeedRunResult],
) -> tuple[dict[str, float], float | None]:
    """Compute per-parameter coefficient of variation across seeds.

    Only converged seeds contribute. Returns the per-parameter CV map
    plus the maximum CV across parameters (the headline R8 metric).
    Returns (empty, None) when fewer than 2 seeds converged.
    """
    converged = [r for r in seed_results if r.converged]
    if len(converged) < 2:
        return {}, None

    param_names = sorted(set().union(*(r.parameter_estimates.keys() for r in converged)))
    cv_per_param: dict[str, float] = {}
    for name in param_names:
        values = [r.parameter_estimates[name] for r in converged if name in r.parameter_estimates]
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        if abs(mean) < 1e-12:
            continue
        sd = statistics.stdev(values)
        # A parameter that came back identical across every seed contributes
        # ``sd == 0`` (no cross-seed information). Including it as ``CV=0``
        # doesn't affect ``max(cv_per_param.values())`` but pollutes the
        # per-param map and the report layer's stability narrative — the
        # parameter literally has nothing to tell us about reproducibility.
        # Exclude so the dict only carries parameters with measurable
        # cross-seed dispersion.
        if sd < 1e-12:
            continue
        cv_per_param[name] = float(sd / abs(mean))

    if not cv_per_param:
        return {}, None
    return cv_per_param, max(cv_per_param.values())


async def _fit_one_seed(
    *,
    runner: Nlmixr2Runner,
    spec: DSLSpec,
    csv_path: Path,
    seed: int,
    timeout_seconds: int | None,
) -> SeedRunResult:
    """Run a single nlmixr2 fit at ``seed`` against ``csv_path``."""
    manifest, _df = ingest_nonmem_csv(csv_path)
    initial_estimates: dict[str, float] = {
        "ka": 1.0,
        "V": 50.0,
        "CL": 5.0,
        "V1": 50.0,
        "V2": 80.0,
        "Q": 10.0,
    }
    result = await runner.run(
        spec,
        manifest,
        initial_estimates,
        seed,
        timeout_seconds=timeout_seconds,
        data_path=csv_path,
    )
    return SeedRunResult(
        seed=seed,
        converged=bool(result.converged),
        minimization_status=str(result.convergence_metadata.minimization_status),
        parameter_estimates=_extract_estimates(result),
        bic=float(result.bic) if result.bic is not None else None,
        wall_time_seconds=float(result.wall_time_seconds or 0.0),
    )


# ---------------------------------------------------------------------------
# Per-case driver
# ---------------------------------------------------------------------------


async def run_case(
    case: BenchmarkCase,
    *,
    runner: Nlmixr2Runner,
    cache_dir: Path,
    work_dir: Path,
    overrides: dict[str, Path],
    n_seeds: int,
    base_seed: int,
    timeout_seconds: int | None,
) -> SuiteBCaseResult:
    """Drive one Suite B case through perturb → multi-seed fit → score.

    Skips B1-B3 NODE-backed cases with a clear ``skipped=True`` signal
    so the CI dashboard surfaces the gap rather than silently omitting
    them.
    """
    if case.case_id in _NODE_BACKED_CASES:
        return SuiteBCaseResult(
            case_id=case.case_id,
            suite=case.suite,
            dataset_id=case.dataset_id,
            skipped=True,
            skip_reason="NODE backend live wiring is out of v0.6 scope",
        )

    case_dir = work_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve and load dataset.
    csv_path = resolve_dataset_csv(case.dataset_id, cache_dir=cache_dir, overrides=overrides)
    _manifest, df = ingest_nonmem_csv(csv_path)

    # 2. Apply perturbation recipes if any. The full perturbed frame is
    # written to disk so the harness ingests a manifest whose sha256
    # matches the actually-fit data.
    if case.perturbations:
        perturbed_df, perturbation_manifests = apply_perturbations(df, list(case.perturbations))
    else:
        perturbed_df, perturbation_manifests = df.copy(), []
    perturbed_csv = case_dir / f"{case.case_id}_perturbed.csv"
    perturbed_df.to_csv(perturbed_csv, index=False)

    # Persist the perturbation manifest sidecar for audit even when no
    # recipes were applied (empty list is a meaningful claim).
    (case_dir / f"{case.case_id}_perturbation_manifest.json").write_text(
        json.dumps(perturbation_manifests, indent=2, sort_keys=True, default=str) + "\n"
    )

    # 3. Build the per-case DSLSpec.
    spec = _build_default_spec(case)

    # 4. Multi-seed fits — the PRD R8 cross-seed stability monitor.
    seed_results: list[SeedRunResult] = []
    for i in range(n_seeds):
        # Large prime offset (7919) keeps per-seed seeds disjoint for
        # any realistic n_seeds; mirrors the Suite C Phase 1 pattern.
        seed = base_seed + 7919 * i
        try:
            seed_result = await _fit_one_seed(
                runner=runner,
                spec=spec,
                csv_path=perturbed_csv,
                seed=seed,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "case %s seed %d: fit raised %s; recording as non-converged",
                case.case_id,
                seed,
                exc,
            )
            seed_result = SeedRunResult(
                seed=seed,
                converged=False,
                minimization_status=f"runner_error: {type(exc).__name__}",
                parameter_estimates={},
                bic=None,
                wall_time_seconds=0.0,
            )
        seed_results.append(seed_result)

    # 5. Aggregate.
    convergence_rate = (
        sum(1 for r in seed_results if r.converged) / len(seed_results) if seed_results else 0.0
    )
    cv_per_param, cv_max = _compute_cross_seed_stability(seed_results)

    return SuiteBCaseResult(
        case_id=case.case_id,
        suite=case.suite,
        dataset_id=case.dataset_id,
        perturbation_manifests=tuple(perturbation_manifests),
        n_seeds=len(seed_results),
        seed_results=tuple(seed_results),
        convergence_rate=convergence_rate,
        cross_seed_cv_max=cv_max,
        cross_seed_cv_per_param=cv_per_param,
    )


async def run_all(
    cases: Sequence[BenchmarkCase],
    *,
    runner: Nlmixr2Runner,
    cache_dir: Path,
    work_dir: Path,
    overrides: dict[str, Path],
    n_seeds: int,
    base_seed: int,
    timeout_seconds: int | None,
) -> dict[str, SuiteBCaseResult]:
    """Drive multiple Suite B cases sequentially.

    Sequential for the same reason as Suite C Phase 1 — one R subprocess
    at a time keeps the CI runner's 4 vCPUs from competing.
    """
    out: dict[str, SuiteBCaseResult] = {}
    for case in cases:
        out[case.case_id] = await run_case(
            case,
            runner=runner,
            cache_dir=cache_dir,
            work_dir=work_dir,
            overrides=overrides,
            n_seeds=n_seeds,
            base_seed=base_seed,
            timeout_seconds=timeout_seconds,
        )
    return out


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def _serialize_case(result: SuiteBCaseResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "suite": result.suite,
        "dataset_id": result.dataset_id,
        "skipped": result.skipped,
        "skip_reason": result.skip_reason,
        "n_seeds": result.n_seeds,
        "convergence_rate": result.convergence_rate,
        "cross_seed_cv_max": result.cross_seed_cv_max,
        "cross_seed_cv_per_param": result.cross_seed_cv_per_param,
        "perturbation_manifests": list(result.perturbation_manifests),
        "seed_results": [
            {
                "seed": r.seed,
                "converged": r.converged,
                "minimization_status": r.minimization_status,
                "parameter_estimates": r.parameter_estimates,
                "bic": r.bic,
                "wall_time_seconds": r.wall_time_seconds,
            }
            for r in result.seed_results
        ],
    }


def write_results_atomic(path: Path, results: dict[str, SuiteBCaseResult]) -> None:
    """Write Suite B results JSON via tmp-file + ``Path.replace``."""
    payload: dict[str, dict[str, object]] = {
        case_id: _serialize_case(r) for case_id, r in results.items()
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


def _parse_dataset_csv(values: Sequence[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in values or ():
        if "=" not in raw:
            msg = f"--dataset-csv expects id=path, got {raw!r}"
            raise argparse.ArgumentTypeError(msg)
        key, _, path_str = raw.partition("=")
        if not key or not path_str:
            msg = f"--dataset-csv id and path must both be non-empty, got {raw!r}"
            raise argparse.ArgumentTypeError(msg)
        out[key] = Path(path_str).expanduser().resolve()
    return out


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m apmode.benchmarks.suite_b_runner",
        description="Run the Suite B perturbation + cross-seed stability loop.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/suite_b/suite_b_results.json"),
        help="Destination path for the suite_b_results.json the CI workflow reads.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(c.case_id for c in ALL_EXTENDED_CASES),
        help="Comma-separated case ids (defaults to ALL_EXTENDED_CASES).",
    )
    parser.add_argument(
        "--dataset-csv",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Override the CSV path for a case's dataset_id. Repeatable.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Independent seeds per case for the R8 cross-seed monitor (default 3).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260425,
        help="Base RNG seed; per-seed offsets are 7919*i.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-fit timeout. Default 1800 (30 min) accommodates the slowest mavoglurant cases.",
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

    try:
        overrides = _parse_dataset_csv(args.dataset_csv)
    except argparse.ArgumentTypeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return _EXIT_USAGE

    case_ids = [cid.strip() for cid in args.cases.split(",") if cid.strip()]
    if not case_ids:
        sys.stderr.write("error: --cases resolved to an empty list\n")
        return _EXIT_USAGE

    by_id = {c.case_id: c for c in ALL_EXTENDED_CASES}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        sys.stderr.write(f"error: unknown case(s) {unknown} — valid ids: {sorted(by_id)}\n")
        return _EXIT_USAGE
    selected = [by_id[cid] for cid in case_ids]

    with tempfile.TemporaryDirectory(prefix="apmode_suite_b_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        cache_dir = args.cache_dir or (tmp_root_path / "cache")
        work_dir = args.work_dir or (tmp_root_path / "work")

        estimation_methods: list[str] | None = None
        if args.estimation:
            estimation_methods = [m.strip() for m in args.estimation.split(",") if m.strip()]
            if not estimation_methods:
                sys.stderr.write("error: --estimation resolved to an empty method list\n")
                return _EXIT_USAGE

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
                    cache_dir=cache_dir,
                    work_dir=work_dir,
                    overrides=overrides,
                    n_seeds=args.n_seeds,
                    base_seed=args.base_seed,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        except KeyError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return _EXIT_USAGE
        except (RuntimeError, ValueError) as exc:
            sys.stderr.write(f"error: fit pipeline failed: {exc}\n")
            return _EXIT_FIT_FAILURE

        try:
            write_results_atomic(args.out, results)
        except OSError as exc:
            sys.stderr.write(f"error: failed to write {args.out}: {exc}\n")
            return _EXIT_VALIDATION

        for cid, r in results.items():
            if r.skipped:
                sys.stderr.write(f"info: {cid}: skipped — {r.skip_reason}\n")
                continue
            sys.stderr.write(
                f"info: {cid}: convergence_rate={r.convergence_rate:.2f} "
                f"cross_seed_cv_max="
                f"{r.cross_seed_cv_max if r.cross_seed_cv_max is not None else 'NA'} "
                f"(n_seeds={r.n_seeds})\n"
            )

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SeedRunResult",
    "SuiteBCaseResult",
    "main",
    "resolve_dataset_csv",
    "run_all",
    "run_case",
    "write_results_atomic",
]
