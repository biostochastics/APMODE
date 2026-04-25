# SPDX-License-Identifier: GPL-2.0-or-later
"""Plan Task 44 — Phase-1 Suite C live-fit runner.

For each Phase-1 MLE fixture this module composes the existing pieces
into a single producer of ``benchmarks/suite_c/phase1_npe_inputs.json``:

  1. ``literature_loader.load_fixture_by_id`` → :class:`LiteratureFixture`
  2. ``data.datasets.fetch_dataset`` (or a ``--dataset-csv`` override)
     → NONMEM-style CSV path.
  3. ``data.ingest.ingest_nonmem_csv`` → :class:`DataManifest` + DataFrame.
  4. ``data.splitter.k_fold_split`` → 5 :class:`SplitManifest` folds
     using :data:`apmode.benchmarks.suite_c.DEFAULT_SPLIT` (subject-level,
     seed ``20260414``).
  5. Per fold, two nlmixr2 fits:
        * APMODE-default initial estimates derived from
          :class:`NCAEstimator.estimate_population_level` (or NCA per-subject
          when subjects support it).
        * Warm-started at ``fixture.reference_params`` (translated through
          ``parameterization_mapping``).
     Both runs request ``gate3_policy.n_posterior_predictive_sims`` so the
     runner returns ``BackendResult.diagnostics.npe_score`` directly via
     the canonical :func:`build_predictive_diagnostics` helper — the
     comparability anchor (PRD §10 Q2): both NPEs flow through the same
     observation model and the same aggregation.
  6. Median NPE across folds → entry in ``phase1_npe_inputs.json``;
     per-fold values flow through to ``FixtureScore.npe_apmode_per_fold``
     for downstream variance bars.
  7. Atomic write of the inputs JSON (tmp + rename) so a SIGKILL
     mid-write never half-writes the file the Task 41 scorer ingests.

Honest mode (v0.6.1)
--------------------

* **Held-out NPE per fold.** Each fold writes both a ``train.csv`` and
  a disjoint ``test.csv``. The APMODE-side fit uses the train CSV via
  ``Nlmixr2Runner.run(..., data_path=train_csv,
  test_data_path=test_csv)``; the harness fits on the train CSV and
  routes ``rxode2::rxSolve(events=test_df)`` so the posterior-predictive
  matrix is generated on subjects the model never saw. The reported
  NPE is therefore a true held-out generalisation metric, not a
  goodness-of-fit metric. Subject-level k-fold guarantees the train/test
  IDs are disjoint — required because rxode2 partitions sims by ID
  and a colliding ID would silently recycle the train subject's
  posthoc ETA instead of drawing from Omega.

* **Literature side is fixed-THETA evaluation** (true methodology-drift
  detector). The literature-side fit calls
  ``Nlmixr2Runner.run(..., fixed_parameter=True,
  initial_estimates=reference_params)``: the harness runs
  ``est='posthoc'`` exactly once, freezing THETA/OMEGA/SIGMA at the
  published values and only estimating ETAs. Posterior-predictive sims
  then run on the held-out fold. APMODE wins iff its free-fit NPE is
  better than the published parameter set's NPE on the same held-out
  subjects — which is the definition of methodology drift (or its
  absence). For fixtures whose published parameters ARE the data-driven
  optimum, the two NPEs should be statistically indistinguishable; only
  drift away from that optimum produces a measurable APMODE win.

* **Same seed within a fold.** Both fits in a fold use the *same* RNG
  seed so the only NPE difference comes from differing parameter
  vectors, not from differing posterior-predictive ETA draws. With
  200 sims, sim-noise-only NPE std-error is ~5-10% of the value, which
  would otherwise dominate the 2% gate margin and turn the gate into
  a coin flip.

CLI
---
``python -m apmode.benchmarks.suite_c_phase1_runner --out
benchmarks/suite_c/phase1_npe_inputs.json [--fixtures id1,id2,...]
[--dataset-csv id=/abs/path ...] [--work-dir DIR] [--n-sims 200]
[--n-folds 5]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from apmode.backends.nlmixr2_runner import Nlmixr2Runner
from apmode.benchmarks.literature_loader import (
    PHASE1_MLE_FIXTURE_IDS,
    load_dsl_spec,
    load_fixture_by_id,
)
from apmode.benchmarks.suite_c import DEFAULT_SPLIT
from apmode.data.datasets import DATASET_REGISTRY, fetch_dataset
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.splitter import k_fold_split
from apmode.governance.policy import Gate3Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apmode.benchmarks.models import LiteratureFixture
    from apmode.bundle.models import BackendResult, DataManifest
    from apmode.dsl.ast_models import DSLSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hand-curated fixture-id → DATASET_REGISTRY key mapping for the three
# Phase-1 fixtures whose dataset is shipped via ``data.datasets.fetch_dataset``
# (the nlmixr2data ones). The two non-registry fixtures
# (``ddmore_gentamicin``, ``mimic_vancomycin``) require a ``--dataset-csv``
# override; the runner surfaces a clear error when that override is missing
# rather than silently skipping the fixture.
_FIXTURE_TO_REGISTRY_KEY: dict[str, str] = {
    "nlmixr2data_theophylline": "theo_sd",
    "nlmixr2data_warfarin": "warfarin",
    "nlmixr2data_mavoglurant": "mavoglurant",
}

# Exit codes mirror the CLI scorer (suite_c_phase1_cli.py) for operator
# muscle-memory: 0 happy path, 2 usage, 3 input validation, 5 R-harness
# / fit failure (distinct from 3 so the workflow can route differently).
_EXIT_OK: int = 0
_EXIT_USAGE: int = 2
_EXIT_VALIDATION: int = 3
_EXIT_FIT_FAILURE: int = 5


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixturePhase1Inputs:
    """Per-fixture inputs for the Task 41 scorer.

    Plain dataclass (not Pydantic) because the runner serialises directly
    to the inputs JSON shape the scorer ingests; an extra Pydantic wrap
    would just round-trip the same dict.
    """

    fixture_id: str
    npe_apmode: float
    npe_literature: float
    npe_apmode_per_fold: tuple[float, ...]
    npe_literature_per_fold: tuple[float, ...]
    n_subjects: int
    n_folds: int


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def resolve_dataset_csv(
    fixture: LiteratureFixture,
    *,
    cache_dir: Path,
    overrides: dict[str, Path],
) -> Path:
    """Resolve a fixture's dataset_id to a NONMEM-style CSV path.

    Resolution order:
      1. ``overrides[fixture.dataset_id]`` if present — supports the
         ``--dataset-csv id=path`` CLI flag for the non-registry fixtures.
      2. The hand-curated ``_FIXTURE_TO_REGISTRY_KEY`` map → ``fetch_dataset``.
         ``fetch_dataset`` is cache-aware (returns the cached CSV when the
         file already exists) so repeated runs do not re-extract from R.

    Raises ``KeyError`` with an actionable message when neither path
    resolves — the message names the missing fixture and the
    ``--dataset-csv`` flag the operator should supply.
    """
    dataset_id = fixture.dataset_id
    override = overrides.get(dataset_id)
    if override is not None:
        if not override.is_file():
            msg = (
                f"--dataset-csv override for {dataset_id!r} points at "
                f"{override} which is not a regular file"
            )
            raise FileNotFoundError(msg)
        return override

    registry_key = _FIXTURE_TO_REGISTRY_KEY.get(dataset_id)
    if registry_key is None or registry_key not in DATASET_REGISTRY:
        msg = (
            f"fixture dataset_id {dataset_id!r} is not in the built-in "
            "registry map and no --dataset-csv override was supplied. "
            f"Pass --dataset-csv {dataset_id}=/abs/path/to.csv"
        )
        raise KeyError(msg)
    return fetch_dataset(registry_key, cache_dir)


# ---------------------------------------------------------------------------
# Per-fold orchestration
# ---------------------------------------------------------------------------


def _translate_reference_params(fixture: LiteratureFixture) -> dict[str, float]:
    """Apply ``parameterization_mapping`` so reference_params keys are DSL-canonical.

    Fixture YAMLs declare ``reference_params`` under DSL-canonical names
    already (validated at load time by
    :meth:`LiteratureFixture.mapping_values_resolve_to_known_params`), so
    the function is identity in the common case. The pass-through stays
    explicit so a future fixture that ships only the published-symbol
    names can wire the mapping here without changing the runner core.
    """
    return dict(fixture.reference_params)


async def _fit_one(
    *,
    runner: Nlmixr2Runner,
    spec: DSLSpec,
    manifest: DataManifest,
    data_path: Path,
    initial_estimates: dict[str, float],
    seed: int,
    gate3_policy: Gate3Config,
    timeout_seconds: int | None,
    test_data_path: Path | None = None,
    fixed_parameter: bool = False,
) -> BackendResult:
    """Single nlmixr2 fit with NPE-producing posterior-predictive sims requested.

    ``test_data_path`` and ``fixed_parameter`` switch the inner runner
    call into honest-mode: held-out NPE for the APMODE side, fixed-THETA
    held-out NPE for the literature side. Defaults preserve the legacy
    in-sample / warm-start behaviour for callers that need it.
    """
    return await runner.run(
        spec,
        manifest,
        initial_estimates,
        seed,
        timeout_seconds=timeout_seconds,
        data_path=data_path,
        gate3_policy=gate3_policy,
        test_data_path=test_data_path,
        fixed_parameter=fixed_parameter,
    )


def _extract_npe(result: BackendResult, *, label: str, fold_idx: int) -> float:
    """Pull ``diagnostics.npe_score`` out of a BackendResult or fail loudly.

    A missing ``npe_score`` means ``build_predictive_diagnostics`` short-
    circuited (no per-subject sims) — silently substituting NaN would
    falsify the inputs JSON. Surfaced as ``RuntimeError`` so the CLI
    main loop can map it to the fit-failure exit code.
    """
    npe = result.diagnostics.npe_score
    if npe is None:
        msg = (
            f"fold {fold_idx} ({label}): BackendResult.diagnostics.npe_score "
            "is None — posterior-predictive simulation did not produce a "
            "value (see fit logs in the runner work dir). The inputs JSON "
            "intentionally rejects None to keep the scorer's gate honest."
        )
        raise RuntimeError(msg)
    return float(npe)


# ---------------------------------------------------------------------------
# Top-level fixture driver
# ---------------------------------------------------------------------------


async def run_fixture(
    fixture_id: str,
    *,
    runner: Nlmixr2Runner,
    dataset_overrides: dict[str, Path] | None = None,
    cache_dir: Path,
    work_dir: Path,
    n_folds: int = DEFAULT_SPLIT.n_folds,
    fold_seed: int = DEFAULT_SPLIT.seed,
    n_sims: int = 200,
    timeout_seconds: int | None = None,
) -> FixturePhase1Inputs:
    """Drive one fixture through the per-fold APMODE + literature loop.

    ``cache_dir`` holds dataset CSVs from :func:`fetch_dataset`;
    ``work_dir`` is the per-fold temp scratch (CSVs + R subprocess
    output). Both directories are created if missing.

    ``n_sims`` matches Bergstrand 2011 VPC convention at 500 by default
    in :class:`Gate3Config`; the runner default of 200 trades a touch of
    posterior-predictive precision for ~2.5x faster fold turnaround so
    the weekly workflow finishes inside the 30-minute job budget.
    Override via the ``--n-sims`` CLI flag for higher fidelity.
    """
    overrides = dataset_overrides or {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    fixture = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fixture)
    csv_path = resolve_dataset_csv(fixture, cache_dir=cache_dir, overrides=overrides)
    # The full-dataset DataFrame is only used to compute the subject-fold
    # split; per-fold ingestion is re-run on the train CSV so the
    # backend gets a manifest whose sha256 / n_subjects match its data
    # file. Passing the full-dataset manifest with a subset CSV would
    # break the harness's manifest-vs-data consistency check.
    _full_manifest, df = ingest_nonmem_csv(csv_path)
    folds = k_fold_split(df, seed=fold_seed, k=n_folds)
    if not folds:
        msg = f"fixture {fixture_id}: k_fold_split returned no folds for k={n_folds}"
        raise RuntimeError(msg)

    gate3_policy = Gate3Config(n_posterior_predictive_sims=n_sims)
    literature_estimates = _translate_reference_params(fixture)

    apmode_per_fold: list[float] = []
    literature_per_fold: list[float] = []

    # NCAEstimator is imported lazily inside the loop to avoid a hard
    # dependency at module-import time on the rest of the data-stack
    # (initial_estimates pulls in pandera-validated schemas).
    from apmode.data.initial_estimates import NCAEstimator

    for fold_idx, fold in enumerate(folds):
        fold_dir = work_dir / fixture_id / f"fold{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Derive train + test subset DataFrames ONCE and reuse for CSV
        # emission, the NCA estimator, and the held-out simulation. A
        # single source of truth eliminates the risk of a future filter
        # divergence silently desynchronising the fitted data and the
        # held-out events.
        train_subjects = {a.subject_id for a in fold.assignments if a.fold == "train"}
        test_subjects = {a.subject_id for a in fold.assignments if a.fold == "test"}
        if not train_subjects.isdisjoint(test_subjects):
            # Defence in depth: subject-level k-fold MUST emit disjoint
            # subsets. A future split bug would otherwise feed colliding
            # IDs to rxode2's per-ID partition and silently recycle the
            # train subject's posthoc ETA in place of a fresh draw.
            msg = (
                f"fold {fold_idx}: train/test subject IDs overlap "
                f"({sorted(train_subjects & test_subjects)[:5]}); "
                "rxode2 would silently recycle posthoc ETAs"
            )
            raise ValueError(msg)
        train_df = df[df["NMID"].astype(str).isin(train_subjects)].copy()
        test_df = df[df["NMID"].astype(str).isin(test_subjects)].copy()
        if train_df.empty:
            msg = f"fold {fold_idx}: train subset is empty after subject filter"
            raise ValueError(msg)
        if test_df.empty:
            msg = f"fold {fold_idx}: test subset is empty after subject filter"
            raise ValueError(msg)
        train_csv = fold_dir / f"fold{fold_idx:02d}_train.csv"
        test_csv = fold_dir / f"fold{fold_idx:02d}_test.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        # Re-ingest the train CSV so the DataManifest the backend
        # receives reflects the actual fitted data (sha256, n_subjects,
        # n_observations).
        train_manifest, _ = ingest_nonmem_csv(train_csv)

        # APMODE-default initial estimates: NCA on the train subset only,
        # so the literature side does not get an unfair NCA hint from the
        # held-out subjects. ``estimate_population_level`` returns sensible
        # CL/V/ka defaults when the subset is too sparse.
        nca_train = NCAEstimator(train_df, train_manifest)
        apmode_initial = nca_train.estimate_per_subject()
        # NCA returns metadata under leading-underscore keys (e.g.
        # ``_unit_scale``); the runner contract is bare params only.
        apmode_initial = {k: v for k, v in apmode_initial.items() if not k.startswith("_")}
        apmode_initial.setdefault("ka", 1.0)
        apmode_initial.setdefault("V", 50.0)
        apmode_initial.setdefault("CL", 5.0)

        # Same seed within a fold for both fits — different seeds would
        # introduce posterior-predictive Monte Carlo noise that exceeds
        # the 2% scorer gate margin and turn it into a coin flip. The
        # multiplicative offset (large prime 7919) keeps fold seeds
        # disjoint for any realistic n_folds.
        fold_rng_seed = fold_seed + 7919 * fold_idx

        try:
            # APMODE side: free fit on train, posterior-predictive on
            # the held-out test fold → true cross-validation NPE.
            apmode_result = await _fit_one(
                runner=runner,
                spec=spec,
                manifest=train_manifest,
                data_path=train_csv,
                initial_estimates=apmode_initial,
                seed=fold_rng_seed,
                gate3_policy=gate3_policy,
                timeout_seconds=timeout_seconds,
                test_data_path=test_csv,
            )
            # Literature side: fixed-THETA evaluation. The harness runs
            # est='posthoc' once with reference_params loaded into the
            # ini() block, freezing THETA/OMEGA/SIGMA, then simulates on
            # the held-out fold. The resulting NPE is the true target
            # APMODE must beat to demonstrate methodology improvement
            # (not just optimisation noise around the published optimum).
            literature_result = await _fit_one(
                runner=runner,
                spec=spec,
                manifest=train_manifest,
                data_path=train_csv,
                initial_estimates=literature_estimates,
                seed=fold_rng_seed,
                gate3_policy=gate3_policy,
                timeout_seconds=timeout_seconds,
                test_data_path=test_csv,
                fixed_parameter=True,
            )
        except Exception:
            logger.exception(
                "fixture %s fold %d: nlmixr2 fit raised; aborting fixture",
                fixture_id,
                fold_idx,
            )
            raise

        apmode_per_fold.append(_extract_npe(apmode_result, label="apmode", fold_idx=fold_idx))
        literature_per_fold.append(
            _extract_npe(literature_result, label="literature", fold_idx=fold_idx)
        )

    return FixturePhase1Inputs(
        fixture_id=fixture_id,
        npe_apmode=statistics.median(apmode_per_fold),
        npe_literature=statistics.median(literature_per_fold),
        npe_apmode_per_fold=tuple(apmode_per_fold),
        npe_literature_per_fold=tuple(literature_per_fold),
        n_subjects=int(df["NMID"].nunique()),
        n_folds=len(folds),
    )


async def run_all(
    fixture_ids: Sequence[str],
    *,
    runner: Nlmixr2Runner,
    cache_dir: Path,
    work_dir: Path,
    dataset_overrides: dict[str, Path] | None = None,
    n_folds: int = DEFAULT_SPLIT.n_folds,
    fold_seed: int = DEFAULT_SPLIT.seed,
    n_sims: int = 200,
    timeout_seconds: int | None = None,
) -> dict[str, FixturePhase1Inputs]:
    """Drive multiple fixtures sequentially.

    Sequential (not concurrent) because each fixture spawns one R
    subprocess at a time and the GitHub Actions ubuntu-latest runner
    has 4 vCPUs — concurrent fits compete for cores and slow the wall
    clock without freeing up the budget. Local operators with more
    cores can wrap the call in ``asyncio.gather`` over fixture-scoped
    runners if needed.
    """
    out: dict[str, FixturePhase1Inputs] = {}
    for fid in fixture_ids:
        out[fid] = await run_fixture(
            fid,
            runner=runner,
            dataset_overrides=dataset_overrides,
            cache_dir=cache_dir,
            work_dir=work_dir,
            n_folds=n_folds,
            fold_seed=fold_seed,
            n_sims=n_sims,
            timeout_seconds=timeout_seconds,
        )
    return out


# ---------------------------------------------------------------------------
# Inputs JSON writer
# ---------------------------------------------------------------------------


def write_inputs_atomic(path: Path, inputs: dict[str, FixturePhase1Inputs]) -> None:
    """Write ``phase1_npe_inputs.json`` via tmp-file + ``Path.replace``.

    The shape mirrors what :func:`apmode.benchmarks.suite_c_phase1_cli._load_inputs`
    expects: ``{fixture_id: {npe_apmode, npe_literature, npe_apmode_per_fold}}``
    with ``npe_apmode_per_fold`` plumbed through to ``FixtureScore``.
    Per-fold literature values are emitted under
    ``npe_literature_per_fold`` for downstream tooling — the scorer
    tolerates extra fields (forward-compat per its docstring).
    """
    payload: dict[str, dict[str, object]] = {}
    for fid, item in inputs.items():
        payload[fid] = {
            "npe_apmode": item.npe_apmode,
            "npe_literature": item.npe_literature,
            "npe_apmode_per_fold": list(item.npe_apmode_per_fold),
            "npe_literature_per_fold": list(item.npe_literature_per_fold),
            "n_subjects": item.n_subjects,
            "n_folds": item.n_folds,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    except BaseException:
        # ``BaseException`` covers KeyboardInterrupt as well so a CTRL-C
        # mid-write does not orphan the .tmp on the operator's disk.
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_dataset_csv(values: Sequence[str] | None) -> dict[str, Path]:
    """Parse repeated ``--dataset-csv id=path`` flags into a dict."""
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
        prog="python -m apmode.benchmarks.suite_c_phase1_runner",
        description=(
            "Run the Phase-1 Suite C live-fit loop and write "
            "phase1_npe_inputs.json for the Task 41 scorer."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/suite_c/phase1_npe_inputs.json"),
        help=(
            "Destination path for phase1_npe_inputs.json. Default points "
            "at the location the weekly workflow's scorer reads."
        ),
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default=",".join(PHASE1_MLE_FIXTURE_IDS),
        help=(
            "Comma-separated fixture ids to run (defaults to the full "
            "PHASE1_MLE_FIXTURE_IDS roster)."
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        action="append",
        default=[],
        metavar="ID=PATH",
        help=(
            "Override the CSV path for a fixture's dataset_id. Required "
            "for fixtures whose dataset is not in DATASET_REGISTRY "
            "(ddmore_gentamicin, mimic_vancomycin). Repeatable."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=("Where fetch_dataset caches CSVs from R. Defaults to a tempdir scoped to this run."),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Per-fold scratch (R subprocess work + train CSV temps). "
            "Defaults to a tempdir scoped to this run."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_SPLIT.n_folds,
        help="Number of CV folds (default 5 per DEFAULT_SPLIT).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=200,
        help=(
            "Posterior-predictive simulation count per fit. 200 trades "
            "fidelity for runner-budget speed; raise to 500 to match "
            "Gate3Config default."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Per-fit timeout passed to Nlmixr2Runner.run.",
    )
    parser.add_argument(
        "--rscript",
        type=str,
        default="Rscript",
        help="R executable name or absolute path (default: Rscript on PATH).",
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

    fixture_ids = [fid.strip() for fid in args.fixtures.split(",") if fid.strip()]
    if not fixture_ids:
        sys.stderr.write("error: --fixtures resolved to an empty list\n")
        return _EXIT_USAGE

    unknown = set(fixture_ids) - set(PHASE1_MLE_FIXTURE_IDS)
    if unknown:
        sys.stderr.write(
            f"error: unknown fixture(s) {sorted(unknown)} — "
            f"valid ids: {list(PHASE1_MLE_FIXTURE_IDS)}\n"
        )
        return _EXIT_USAGE

    # Tempdirs cleaned up on exit so a long-running operator does not
    # accumulate per-run scratch under /tmp. Cache-dir is reused across
    # runs only when the operator passes --cache-dir explicitly.
    with tempfile.TemporaryDirectory(prefix="apmode_suite_c_p1_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        cache_dir = args.cache_dir or (tmp_root_path / "cache")
        work_dir = args.work_dir or (tmp_root_path / "work")

        try:
            runner = Nlmixr2Runner(work_dir=work_dir, r_executable=args.rscript)
        except FileNotFoundError as exc:
            sys.stderr.write(f"error: cannot start Nlmixr2Runner: {exc}\n")
            return _EXIT_USAGE

        try:
            inputs = asyncio.run(
                run_all(
                    fixture_ids,
                    runner=runner,
                    cache_dir=cache_dir,
                    work_dir=work_dir,
                    dataset_overrides=overrides,
                    n_folds=args.n_folds,
                    n_sims=args.n_sims,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        except KeyError as exc:
            # Missing dataset override — actionable at the CLI surface.
            sys.stderr.write(f"error: {exc}\n")
            return _EXIT_USAGE
        except (RuntimeError, ValueError) as exc:
            sys.stderr.write(f"error: fit pipeline failed: {exc}\n")
            return _EXIT_FIT_FAILURE

        try:
            write_inputs_atomic(args.out, inputs)
        except OSError as exc:
            sys.stderr.write(f"error: failed to write {args.out}: {exc}\n")
            return _EXIT_VALIDATION

        # Echo a one-line summary so the workflow log captures progress
        # without needing to cat the JSON.
        for fid, item in inputs.items():
            sys.stderr.write(
                f"info: {fid}: npe_apmode={item.npe_apmode:.4f} "
                f"npe_literature={item.npe_literature:.4f} "
                f"(n_subjects={item.n_subjects}, folds={item.n_folds})\n"
            )

        # The cache_dir under tmp_root would be cleaned by the context
        # exit — copy fetched CSVs out only when the caller passed an
        # explicit --cache-dir (already a persistent location).
        if args.cache_dir is None:
            # Nothing to preserve; tempdir cleans itself up.
            pass
        else:  # pragma: no cover — exercised only with --cache-dir
            shutil.rmtree(work_dir, ignore_errors=True)

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in CI
    raise SystemExit(main())


__all__ = [
    "FixturePhase1Inputs",
    "main",
    "resolve_dataset_csv",
    "run_all",
    "run_fixture",
    "write_inputs_atomic",
]
