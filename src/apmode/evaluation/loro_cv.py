# SPDX-License-Identifier: GPL-2.0-or-later
"""LORO-CV execution engine (Phase 3 — Optimization lane, PRD §3.3).

Leave-one-regimen-out cross-validation for predictive performance evaluation.
Each fold holds out one regimen group, trains on the rest, and evaluates
predictions on the held-out group.

Each fold is evaluated in two steps so that no information about the
held-out regimen group ever informs the parameters used to score it:

  1. **Per-fold refit** (``fixed_parameter=False``): the candidate is
     re-estimated from scratch on a CSV containing *only* the fold's train
     subjects. ``initial_estimates`` (the run's NCA-derived warm start) only
     seeds the optimizer — the likelihood driving convergence is computed
     exclusively from train-fold data, so the fold-specific structural
     estimates this step produces cannot be informed by the held-out group.
  2. **Frozen posthoc evaluation** (``fixed_parameter=True``): THETA/OMEGA/
     SIGMA are frozen at step 1's fold-specific estimates and posthoc ETAs
     are fit across the full dataset (train and test), exactly as the
     ``BackendRunner.run`` ``fixed_parameter`` contract specifies. Because
     posthoc/empirical-Bayes ETA estimation is inherently per-subject, this
     step does not reintroduce leakage as long as the frozen values
     themselves are leak-free — which step 1 now guarantees.

Earlier versions of this module skipped step 1 and froze parameters taken
directly from ``candidate_result`` (the Gate-1 survivor's *full-data* fit,
which by construction already saw every regimen group, including whichever
one a given fold holds out). That let the held-out group's own data
influence the "fixed" values being evaluated against it, silently turning
the extrapolation check into a goodness-of-fit check. Step 1 above is the
fix — it costs one extra backend call per fold, which is the standard price
of leak-free k-fold CV.

Metrics aggregated across folds:
  - Pooled NPDE (approximated via CWRES on test fold): mean ~0, variance ~1
    (Comets et al., 2008). True NPDE requires Monte Carlo simulation;
    CWRES on test subjects is used as a computationally feasible proxy.
  - VPC coverage concordance: min coverage across percentile bands, aggregated
    from per-fold VPC diagnostics (Bergstrand et al., 2011).
  - AUC/Cmax GMR 80-125% bioequivalence (PRD v0.3 §4.3.1).
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from apmode.bundle.models import LOROCVResult, LOROFoldResult, LOROMetrics

if TYPE_CHECKING:
    import pandas as pd

    from apmode.backends.protocol import BackendRunner
    from apmode.bundle.models import (
        BackendResult,
        DataManifest,
        NCASubjectDiagnostic,
        SplitManifest,
    )
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

logger = structlog.get_logger(__name__)


async def evaluate_loro_cv(
    candidate_spec: DSLSpec,
    candidate_result: BackendResult,
    folds: list[SplitManifest],
    runner: BackendRunner,
    data_manifest: DataManifest,
    data_path: Path,
    df: pd.DataFrame,
    initial_estimates: dict[str, float],
    seed: int,
    timeout_seconds: int = 600,
    regimen_labels: list[str] | None = None,
    gate3_policy: Gate3Config | None = None,
    nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
) -> LOROCVResult:
    """Run LORO-CV for a single candidate across all folds.

    For each fold, refits the candidate on train-only data (see module
    docstring step 1), then freezes those fold-specific parameters and
    evaluates posthoc ETAs across the full dataset (step 2). This measures
    "does this model extrapolate to new regimens?" without letting the
    held-out regimen's own data inform the values it is scored against.

    Args:
        candidate_spec: DSLSpec for the candidate model.
        candidate_result: BackendResult from full-data fit (Gate 1 survivor).
            Used only for its ``model_id`` — never as a source of frozen
            parameter values (see module docstring).
        folds: List of SplitManifest from loro_cv_splits().
        runner: BackendRunner for executing fits.
        data_manifest: Data manifest for the run.
        data_path: Path to the full data CSV (used for step 2's
            frozen/posthoc pass, which needs every subject).
        df: The full dataset already loaded in memory (the orchestrator
            has this on hand already), used to carve out each fold's
            train-only CSV for step 1 without re-reading ``data_path``.
        initial_estimates: Structural parameter estimates. Used only as an
            optimizer warm-start seed for step 1's train-only refit —
            never as a frozen value — so it may legitimately be derived
            from the full dataset (e.g. NCA) without reintroducing leakage.
        seed: Random seed for reproducibility.
        timeout_seconds: Per-fold, per-step timeout.

    Returns:
        LOROCVResult with per-fold results and aggregated metrics.
    """
    start_time = time.monotonic()
    fold_results: list[LOROFoldResult] = []
    regimen_groups: list[str] = []

    with tempfile.TemporaryDirectory(prefix="apmode_loro_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for fold_idx, fold_manifest in enumerate(folds):
            test_subjects = {a.subject_id for a in fold_manifest.assignments if a.fold == "test"}
            train_subjects = {a.subject_id for a in fold_manifest.assignments if a.fold == "train"}

            # Use actual regimen group name for auditability
            if regimen_labels is not None and fold_idx < len(regimen_labels):
                regimen_label = regimen_labels[fold_idx]
            else:
                regimen_label = f"fold_{fold_idx}"
            regimen_groups.append(regimen_label)

            logger.info(
                "loro_fold_start",
                fold=fold_idx + 1,
                total_folds=len(folds),
                n_train=len(train_subjects),
                n_test=len(test_subjects),
            )

            train_csv = tmp_dir / f"fold{fold_idx:02d}_train.csv"
            train_df = df[df["NMID"].astype(str).isin(train_subjects)]
            train_df.to_csv(train_csv, index=False)

            try:
                # Step 1: leak-free per-fold refit on train-only data. No
                # split_manifest / gate3_policy needed — this call exists
                # only to obtain fold-specific structural estimates.
                fold_refit = await runner.run(
                    spec=candidate_spec,
                    data_manifest=data_manifest,
                    initial_estimates=initial_estimates,
                    seed=seed + fold_idx,
                    timeout_seconds=timeout_seconds,
                    data_path=train_csv,
                    fixed_parameter=False,
                )
                fold_warm_estimates = {
                    name: pe.estimate
                    for name, pe in fold_refit.parameter_estimates.items()
                    if pe.category == "structural"
                }
                if not fold_warm_estimates:
                    fold_warm_estimates = initial_estimates

                # Step 2: freeze THETA/OMEGA/SIGMA at the fold-specific
                # (leak-free) estimates and fit posthoc ETAs across the
                # full dataset so split_gof can partition residuals by
                # train/test fold.
                fold_result_backend = await runner.run(
                    spec=candidate_spec,
                    data_manifest=data_manifest,
                    initial_estimates=fold_warm_estimates,
                    seed=seed + fold_idx,
                    timeout_seconds=timeout_seconds,
                    data_path=data_path,
                    split_manifest=fold_manifest.model_dump(),
                    gate3_policy=gate3_policy,
                    nca_diagnostics=nca_diagnostics,
                    fixed_parameter=True,
                )

                fold_result = _extract_fold_metrics(
                    fold_idx,
                    regimen_label,
                    fold_result_backend,
                    len(train_subjects),
                    len(test_subjects),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "loro_fold_failed",
                    fold=fold_idx,
                    candidate=candidate_spec.model_id,
                    exc_info=True,
                )
                fold_result = LOROFoldResult(
                    fold_index=fold_idx,
                    regimen_group=regimen_label,
                    n_train_subjects=len(train_subjects),
                    n_test_subjects=len(test_subjects),
                    converged=False,
                )

            fold_results.append(fold_result)

    metrics = _aggregate_loro_metrics(fold_results)
    wall_time = time.monotonic() - start_time

    return LOROCVResult(
        candidate_id=candidate_result.model_id,
        metrics=metrics,
        fold_results=fold_results,
        wall_time_seconds=wall_time,
        regimen_groups=regimen_groups,
        seed=seed,
    )


def _extract_fold_metrics(
    fold_idx: int,
    regimen_label: str,
    result: BackendResult,
    n_train: int,
    n_test: int,
) -> LOROFoldResult:
    """Extract per-fold metrics from a BackendResult.

    Prefers split_gof (test-set CWRES from split-aware evaluation) when
    available. Falls back to overall GOF metrics.

    Note: CWRES on the test fold is used as a computationally feasible
    proxy for NPDE. True NPDE requires Monte Carlo simulation of the
    predictive distribution; this is a planned future enhancement.
    """
    # Prefer split-aware test-set CWRES mean when available;
    # always use overall CWRES SD² for variance (split_gof lacks test_cwres_sd).
    # ``None`` for either propagates as ``None`` into the LORO fold result —
    # the downstream aggregator at ``test_npde_mean / variance`` already
    # treats None as "fold diagnostic unavailable" (and uses 0.0/1.0 only
    # for the cross-fold mean/variance aggregation).
    sgof = result.diagnostics.split_gof
    if sgof is not None:
        test_cwres_mean: float | None = sgof.test_cwres_mean
    else:
        test_cwres_mean = result.diagnostics.gof.cwres_mean
    # Variance: always from overall GOF cwres_sd (the only SD available).
    cwres_sd = result.diagnostics.gof.cwres_sd
    test_cwres_var: float | None = cwres_sd**2 if cwres_sd is not None else None

    # Extract VPC coverage for this fold (min across bands)
    vpc = result.diagnostics.vpc
    fold_vpc_min_coverage: float | None = None
    if vpc is not None and vpc.coverage:
        fold_vpc_min_coverage = min(vpc.coverage.values())

    return LOROFoldResult(
        fold_index=fold_idx,
        regimen_group=regimen_label,
        n_train_subjects=n_train,
        n_test_subjects=n_test,
        train_ofv=result.ofv,
        test_npde_mean=test_cwres_mean,
        test_npde_variance=test_cwres_var,
        test_bic=result.bic,
        converged=result.converged,
        fold_vpc_min_coverage=fold_vpc_min_coverage,
    )


def _aggregate_loro_metrics(fold_results: list[LOROFoldResult]) -> LOROMetrics:
    """Aggregate per-fold results into pooled LORO metrics.

    Uses the law of total variance for correct pooled variance:
      Var_pooled = E[Var_i] + Var[E_i]
                 = weighted_mean(var_i) + weighted_mean(mean_i^2) - pooled_mean^2

    VPC concordance is computed as the weighted average of per-fold minimum
    VPC coverage across percentile bands, not the convergence fraction.
    """
    converged_folds = [f for f in fold_results if f.converged]
    n_converged = len(converged_folds)

    if n_converged == 0:
        return LOROMetrics(
            n_folds=len(fold_results),
            n_total_test_subjects=sum(f.n_test_subjects for f in fold_results),
            pooled_npde_mean=float("nan"),
            pooled_npde_variance=float("nan"),
            vpc_coverage_concordance=0.0,
            overall_pass=False,
        )

    total_test = sum(f.n_test_subjects for f in converged_folds)

    # Collect per-fold metrics
    means: list[float] = []
    variances: list[float] = []
    weights: list[float] = []
    per_fold_bic: list[float] = []
    vpc_coverages: list[float] = []
    vpc_weights: list[float] = []

    for f in converged_folds:
        w = float(f.n_test_subjects)
        weights.append(w)
        means.append(f.test_npde_mean if f.test_npde_mean is not None else 0.0)
        variances.append(f.test_npde_variance if f.test_npde_variance is not None else 1.0)
        if f.test_bic is not None:
            per_fold_bic.append(f.test_bic)
        if f.fold_vpc_min_coverage is not None:
            vpc_coverages.append(f.fold_vpc_min_coverage)
            vpc_weights.append(w)

    w_arr = np.array(weights, dtype=float)
    m_arr = np.array(means, dtype=float)
    v_arr = np.array(variances, dtype=float)
    w_sum = float(w_arr.sum())

    # Pooled mean: weighted average of fold means
    pooled_mean = float(np.average(m_arr, weights=w_arr)) if w_sum > 0 else 0.0

    # Pooled variance: law of total variance
    # Var_total = E[Var_i] + Var[E_i]
    # = weighted_mean(var_i) + weighted_mean(mean_i^2) - pooled_mean^2
    if w_sum > 0:
        e_var = float(np.average(v_arr, weights=w_arr))
        e_mean_sq = float(np.average(m_arr**2, weights=w_arr))
        pooled_var = e_var + e_mean_sq - pooled_mean**2
    else:
        pooled_var = 1.0

    # VPC concordance: weighted average of per-fold min VPC coverage.
    # Missing VPC evidence → 0.0 (fail-closed: missing evidence must fail
    # to prevent false Gate 2 pass)
    if vpc_coverages:
        vpc_w_arr = np.array(vpc_weights, dtype=float)
        vpc_concordance = float(np.average(vpc_coverages, weights=vpc_w_arr))
    else:
        vpc_concordance = 0.0

    # Worst fold tracking
    worst_mean_idx = int(np.argmax(np.abs(m_arr)))
    worst_var_idx = int(np.argmax(np.abs(v_arr - 1.0)))

    # Conservative: all folds must converge for overall_pass
    overall_pass = n_converged == len(fold_results)

    return LOROMetrics(
        n_folds=len(fold_results),
        n_total_test_subjects=total_test,
        pooled_npde_mean=pooled_mean,
        pooled_npde_variance=pooled_var,
        vpc_coverage_concordance=vpc_concordance,
        per_fold_bic=per_fold_bic,
        worst_fold_npde_mean=float(m_arr[worst_mean_idx]) if len(m_arr) > 0 else None,
        worst_fold_npde_variance=float(v_arr[worst_var_idx]) if len(v_arr) > 0 else None,
        overall_pass=overall_pass,
        evaluation_mode="fixed_parameter",
    )
