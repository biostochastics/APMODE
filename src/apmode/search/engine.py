# SPDX-License-Identifier: GPL-2.0-or-later
"""Automated search engine (PRD §4.2.3, ARCHITECTURE.md §6 Phase 2).

Generates candidate models from the search space, dispatches them to the
appropriate backend runner (nlmixr2 for classical specs, jax_node for NODE
specs), collects results, and identifies Pareto-optimal candidates
(parsimony vs. fit).

Search flow:
  1. Generate root candidates from SearchSpace x NCA initial estimates
  2. Dispatch each to the appropriate BackendRunner based on spec type
  3. Score: BIC (primary), AIC (secondary)
  4. Log search trajectory entry for each candidate
  5. Identify Pareto frontier: n_params vs. BIC
  6. Track lineage via SearchDAG
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from apmode.bundle.models import SearchTrajectoryEntry
from apmode.search.candidates import SearchDAG, SearchSpace, generate_root_candidates

if TYPE_CHECKING:
    from pathlib import Path

    from apmode.backends.protocol import BackendRunner
    from apmode.bundle.emitter import BundleEmitter
    from apmode.bundle.models import (
        BackendResult,
        DataManifest,
        EvidenceManifest,
        NCASubjectDiagnostic,
    )
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """Result of a single candidate evaluation."""

    candidate_id: str
    spec: DSLSpec
    result: BackendResult | None
    error: str | None = None
    converged: bool = False
    bic: float | None = None
    aic: float | None = None
    n_params: int = 0


@dataclass
class SearchOutcome:
    """Outcome of the full automated search."""

    results: list[SearchResult] = field(default_factory=list)
    pareto_front: list[SearchResult] = field(default_factory=list)
    dag: SearchDAG = field(default_factory=SearchDAG)
    best_bic: SearchResult | None = None


class SearchEngine:
    """Automated structural search engine.

    Generates candidates from the search space, dispatches to the
    appropriate backend runner, scores, and identifies the Pareto frontier.

    Phase 2: Accepts multiple backend runners keyed by name. Dispatches
    NODE specs to ``jax_node`` runner and classical specs to ``nlmixr2``.
    """

    def __init__(
        self,
        runner: BackendRunner,
        data_manifest: DataManifest,
        data_path: Path,
        seed: int,
        timeout_seconds: int | None = None,
        allowed_backends: list[str] | None = None,
        split_manifest: dict[str, object] | None = None,
        runners: dict[str, BackendRunner] | None = None,
        max_concurrency: int = 1,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
    ) -> None:
        self._data_manifest = data_manifest
        self._data_path = data_path
        self._seed = seed
        self._timeout = timeout_seconds
        self._allowed_backends = allowed_backends or ["nlmixr2"]
        self._split_manifest = split_manifest
        self._max_concurrency = max(1, max_concurrency)
        self._semaphore: asyncio.Semaphore | None = None
        # Forwarded verbatim to the selected backend runner on each candidate
        # dispatch so the rc8 posterior-predictive pipeline populates
        # DiagnosticBundle.{vpc,npe_score,auc_cmax_be_score} atomically.
        # When ``gate3_policy`` is None the runners skip simulation entirely
        # and Gate 3 falls back to the CWRES NPE proxy (see Nlmixr2Runner).
        self._gate3_policy = gate3_policy
        self._nca_diagnostics = nca_diagnostics

        # Phase 2: multiple runners keyed by backend name.
        # ``runner`` is the default (nlmixr2); ``runners`` overrides.
        if runners is not None:
            self._runners = runners
        else:
            self._runners = {"nlmixr2": runner}
        # Keep a default runner reference for seed-stability re-runs
        self._default_runner = runner

    async def run(
        self,
        evidence_manifest: EvidenceManifest,
        initial_estimates: dict[str, float],
        emitter: BundleEmitter | None = None,
        covariate_names: list[str] | None = None,
    ) -> SearchOutcome:
        """Execute the automated search.

        Args:
            evidence_manifest: Constrains the search space.
            initial_estimates: NCA-derived initial estimates for root candidates.
            emitter: Optional BundleEmitter for writing trajectory + specs.
            covariate_names: Covariate columns for SCM search.
        """
        # Build bounded search space from evidence manifest
        space = SearchSpace.from_manifest(evidence_manifest, covariate_names)

        # Generate root candidates
        candidates = generate_root_candidates(space, initial_estimates)
        if not candidates:
            return SearchOutcome()

        dag = SearchDAG()
        all_results: list[SearchResult] = []

        # Phase 1: evaluate all root candidates
        # Register all roots in the DAG and write compiled specs (cheap, sequential)
        for spec in candidates:
            dag.add_root(spec)
            if emitter:
                clean_estimates = {
                    k: v for k, v in initial_estimates.items() if not k.startswith("_")
                }
                emitter.write_compiled_spec(spec, initial_estimates=clean_estimates)

        # Dispatch evaluations (parallel when max_concurrency > 1)
        root_results = await self._gather_evaluations(
            [(spec, initial_estimates) for spec in candidates]
        )
        for sr in root_results:
            all_results.append(sr)
            if sr.converged and sr.bic is not None:
                dag.update_score(sr.candidate_id, sr.bic, True)
            else:
                dag.update_score(sr.candidate_id, float("inf"), False)
            if emitter:
                self._write_trajectory_entry(emitter, sr, parent_id=None)

        # Phase 2: warm-start children from best converged models
        converged = [r for r in all_results if r.converged and r.result is not None]
        if converged:
            # Top 3 by BIC for warm-starting
            top_k = sorted(converged, key=lambda r: r.bic or float("inf"))[:3]

            # Collect all child tasks upfront so they can run concurrently
            child_tasks: list[tuple[DSLSpec, dict[str, float], str, str]] = []
            seen_ids = {r.candidate_id for r in all_results}

            for parent_sr in top_k:
                if parent_sr.result is None:
                    continue
                parent_params = {
                    name: pe.estimate
                    for name, pe in parent_sr.result.parameter_estimates.items()
                    if pe.category == "structural"
                }
                if not parent_params:
                    continue

                for err_type in ["proportional", "additive", "combined"]:
                    child_specs = generate_root_candidates(
                        SearchSpace(
                            structural_cmt=space.structural_cmt,
                            absorption_types=space.absorption_types,
                            elimination_types=space.elimination_types,
                            error_types=[err_type],
                            iiv_structures=["block"],
                            force_blq_method=space.force_blq_method,
                            force_iov=space.force_iov,
                            lloq_value=space.lloq_value,
                        ),
                        parent_params,
                    )
                    for child_spec in child_specs[:2]:
                        if child_spec.model_id in seen_ids:
                            continue
                        seen_ids.add(child_spec.model_id)
                        dag.add_child(parent_sr.candidate_id, child_spec, f"warm_start_{err_type}")
                        child_tasks.append(
                            (
                                child_spec,
                                parent_params.copy(),
                                parent_sr.candidate_id,
                                err_type,
                            )
                        )

            if child_tasks:
                child_results = await self._gather_evaluations(
                    [(spec, est) for spec, est, _, _ in child_tasks]
                )
                for (child_spec, _, parent_id, _), child_sr in zip(
                    child_tasks, child_results, strict=True
                ):
                    all_results.append(child_sr)
                    if child_sr.converged and child_sr.bic is not None:
                        dag.update_score(child_spec.model_id, child_sr.bic, True)
                    if emitter:
                        self._write_trajectory_entry(emitter, child_sr, parent_id=parent_id)

        # Compute Pareto frontier
        pareto = _pareto_frontier(all_results)

        # Best BIC
        converged_final = [r for r in all_results if r.converged and r.bic is not None]
        best = (
            min(converged_final, key=lambda r: r.bic or float("inf")) if converged_final else None
        )

        return SearchOutcome(
            results=all_results,
            pareto_front=pareto,
            dag=dag,
            best_bic=best,
        )

    async def _gather_evaluations(
        self,
        tasks: list[tuple[DSLSpec, dict[str, float]]],
    ) -> list[SearchResult]:
        """Evaluate candidates, respecting max_concurrency via semaphore.

        When max_concurrency == 1, evaluates sequentially (no gather overhead).
        """
        if not tasks:
            return []

        if self._max_concurrency == 1:
            # Fast path: sequential, no gather overhead
            return [await self._evaluate_candidate(spec, est) for spec, est in tasks]

        # Lazy semaphore creation — binds to the running event loop
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        sem = self._semaphore

        async def _bounded(spec: DSLSpec, est: dict[str, float]) -> SearchResult:
            async with sem:
                return await self._evaluate_candidate(spec, est)

        # Structured concurrency: if any task raises, the group cancels the
        # rest and surfaces errors as an ExceptionGroup. _evaluate_candidate
        # catches its own failures, so this is a defensive boundary.
        async with asyncio.TaskGroup() as tg:
            created = [tg.create_task(_bounded(s, e)) for s, e in tasks]
        return [t.result() for t in created]

    def _select_runner(self, spec: DSLSpec) -> tuple[BackendRunner, str]:
        """Select the appropriate backend runner for a spec.

        NODE specs → jax_node runner (if available and allowed).
        Classical specs → nlmixr2 runner.

        Returns:
            (runner, backend_name) tuple.
        """
        if spec.has_node_modules():
            if "jax_node" in self._allowed_backends and "jax_node" in self._runners:
                return self._runners["jax_node"], "jax_node"
            # NODE spec but no NODE runner available — skip
            msg = (
                f"Spec {spec.model_id} has NODE modules but jax_node runner "
                f"not available (allowed={self._allowed_backends})"
            )
            raise _BackendNotAvailable(msg)

        # Classical spec → default runner
        return self._default_runner, "nlmixr2"

    async def _evaluate_candidate(
        self,
        spec: DSLSpec,
        initial_estimates: dict[str, float],
    ) -> SearchResult:
        """Dispatch a single candidate to the appropriate backend."""
        from apmode.errors import BackendError

        # Filter metadata keys from initial estimates
        clean_estimates = {k: v for k, v in initial_estimates.items() if not k.startswith("_")}

        n_params = len(spec.structural_param_names())
        try:
            runner, backend_name = self._select_runner(spec)
        except _BackendNotAvailable as e:
            logger.info("Skipping %s: %s", spec.model_id, e)
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=None,
                error=str(e),
                n_params=n_params,
            )

        try:
            result = await runner.run(
                spec=spec,
                data_manifest=self._data_manifest,
                initial_estimates=clean_estimates,
                seed=self._seed,
                timeout_seconds=self._timeout,
                data_path=self._data_path,
                split_manifest=self._split_manifest,
                gate3_policy=self._gate3_policy,
                nca_diagnostics=self._nca_diagnostics,
            )
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=result,
                converged=result.converged,
                bic=result.bic,
                aic=result.aic,
                n_params=len(result.parameter_estimates),
            )
        except BackendError as e:
            logger.warning("Candidate %s (%s) backend error: %s", spec.model_id, backend_name, e)
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=None,
                error=str(e),
                n_params=n_params,
            )
        except Exception as e:
            logger.error(
                "Candidate %s (%s) unexpected error: %s",
                spec.model_id,
                backend_name,
                e,
                exc_info=True,
            )
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=None,
                error=f"unexpected: {e}",
                n_params=n_params,
            )

    @staticmethod
    def _write_trajectory_entry(
        emitter: BundleEmitter,
        sr: SearchResult,
        parent_id: str | None,
    ) -> None:
        """Write a search trajectory JSONL entry."""
        backend = sr.result.backend if sr.result else "unknown"
        entry = SearchTrajectoryEntry(
            candidate_id=sr.candidate_id,
            parent_id=parent_id,
            backend=backend,
            converged=sr.converged,
            ofv=sr.result.ofv if sr.result else None,
            aic=sr.aic,
            bic=sr.bic,
            wall_time_seconds=sr.result.wall_time_seconds if sr.result else None,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_search_trajectory(entry)


class _BackendNotAvailable(Exception):
    """Raised when a spec requires a backend that is not configured."""


def _pareto_frontier(results: list[SearchResult]) -> list[SearchResult]:
    """Identify Pareto-optimal candidates: minimize n_params AND BIC.

    A candidate is Pareto-optimal if no other candidate dominates it on both
    dimensions (fewer parameters AND lower BIC).
    """
    converged = [r for r in results if r.converged and r.bic is not None]
    if not converged:
        return []

    # Sort by n_params ascending, then BIC ascending
    sorted_results = sorted(
        converged,
        key=lambda r: (r.n_params, r.bic if r.bic is not None else float("inf")),
    )

    pareto: list[SearchResult] = []
    best_bic = float("inf")

    for r in sorted_results:
        bic = r.bic if r.bic is not None else float("inf")
        # Strict improvement: more complex model must have strictly lower BIC
        # to be Pareto-optimal (parsimony preference on ties)
        if bic < best_bic or not pareto:
            pareto.append(r)
            best_bic = bic

    return pareto
