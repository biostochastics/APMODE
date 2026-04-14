# SPDX-License-Identifier: GPL-2.0-or-later
"""Automated search engine (PRD §4.2.3, ARCHITECTURE.md §6 Phase 1 Month 3-4).

Generates candidate models from the search space, dispatches them to the
backend runner, collects results, and identifies Pareto-optimal candidates
(parsimony vs. fit).

Search flow:
  1. Generate root candidates from SearchSpace x NCA initial estimates
  2. Dispatch each to Nlmixr2Runner (or mock)
  3. Score: BIC (primary), AIC (secondary)
  4. Log search trajectory entry for each candidate
  5. Identify Pareto frontier: n_params vs. BIC
  6. Track lineage via SearchDAG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from apmode.bundle.models import SearchTrajectoryEntry
from apmode.search.candidates import SearchDAG, SearchSpace, generate_root_candidates

if TYPE_CHECKING:
    from pathlib import Path

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.bundle.emitter import BundleEmitter
    from apmode.bundle.models import BackendResult, DataManifest, EvidenceManifest
    from apmode.dsl.ast_models import DSLSpec

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

    Generates candidates from the search space, dispatches to backend,
    scores, and identifies the Pareto frontier.
    """

    def __init__(
        self,
        runner: Nlmixr2Runner,
        data_manifest: DataManifest,
        data_path: Path,
        seed: int,
        timeout_seconds: int | None = None,
        allowed_backends: list[str] | None = None,
        split_manifest: dict[str, object] | None = None,
    ) -> None:
        self._runner = runner
        self._data_manifest = data_manifest
        self._data_path = data_path
        self._seed = seed
        self._timeout = timeout_seconds
        # From Lane Router dispatch — constrains which backends are used.
        # Phase 1: only nlmixr2 is implemented, so this is informational.
        # Phase 2: SearchEngine will dispatch to multiple BackendRunners
        # based on this list.
        self._allowed_backends = allowed_backends or ["nlmixr2"]
        self._split_manifest = split_manifest

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
        for spec in candidates:
            dag.add_root(spec)

            if emitter:
                # Filter out metadata keys (prefixed with _) from initial estimates
                clean_estimates = {
                    k: v for k, v in initial_estimates.items() if not k.startswith("_")
                }
                emitter.write_compiled_spec(spec, initial_estimates=clean_estimates)

            sr = await self._evaluate_candidate(spec, initial_estimates)
            all_results.append(sr)

            if sr.converged and sr.bic is not None:
                dag.update_score(spec.model_id, sr.bic, True)
            else:
                dag.update_score(spec.model_id, float("inf"), False)

            if emitter:
                self._write_trajectory_entry(emitter, sr, parent_id=None)

        # Phase 2: warm-start children from best converged models
        converged = [r for r in all_results if r.converged and r.result is not None]
        if converged:
            # Top 3 by BIC for warm-starting
            top_k = sorted(converged, key=lambda r: r.bic or float("inf"))[:3]
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

                # Generate child candidates with different error models
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
                    for child_spec in child_specs[:2]:  # limit children per parent
                        if any(r.spec.model_id == child_spec.model_id for r in all_results):
                            continue

                        dag.add_child(parent_sr.candidate_id, child_spec, f"warm_start_{err_type}")
                        warm_est = parent_params.copy()
                        child_sr = await self._evaluate_candidate(child_spec, warm_est)
                        all_results.append(child_sr)

                        if child_sr.converged and child_sr.bic is not None:
                            dag.update_score(child_spec.model_id, child_sr.bic, True)

                        if emitter:
                            self._write_trajectory_entry(
                                emitter, child_sr, parent_id=parent_sr.candidate_id
                            )

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

    async def _evaluate_candidate(
        self,
        spec: DSLSpec,
        initial_estimates: dict[str, float],
    ) -> SearchResult:
        """Dispatch a single candidate to the backend and collect result."""
        from apmode.errors import BackendError

        # Filter metadata keys from initial estimates
        clean_estimates = {k: v for k, v in initial_estimates.items() if not k.startswith("_")}

        n_params = len(spec.structural_param_names())
        try:
            result = await self._runner.run(
                spec=spec,
                data_manifest=self._data_manifest,
                initial_estimates=clean_estimates,
                seed=self._seed,
                timeout_seconds=self._timeout,
                data_path=self._data_path,
                split_manifest=self._split_manifest,
            )
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=result,
                converged=result.converged,
                bic=result.bic,
                aic=result.aic,
                n_params=n_params,
            )
        except BackendError as e:
            logger.warning("Candidate %s backend error: %s", spec.model_id, e)
            return SearchResult(
                candidate_id=spec.model_id,
                spec=spec,
                result=None,
                error=str(e),
                n_params=n_params,
            )
        except Exception as e:
            logger.error("Candidate %s unexpected error: %s", spec.model_id, e, exc_info=True)
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
        entry = SearchTrajectoryEntry(
            candidate_id=sr.candidate_id,
            parent_id=parent_id,
            backend="nlmixr2",
            converged=sr.converged,
            ofv=sr.result.ofv if sr.result else None,
            aic=sr.aic,
            bic=sr.bic,
            wall_time_seconds=sr.result.wall_time_seconds if sr.result else None,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        emitter.append_search_trajectory(entry)


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
