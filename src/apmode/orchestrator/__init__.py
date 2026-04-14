# SPDX-License-Identifier: GPL-2.0-or-later
"""Orchestrator: asyncio event loop, dispatch, pipeline control.

Full pipeline per ARCHITECTURE.md §3:
  ingest → profile_data() → NCA estimates → data split → search dispatch →
  Gate 1 (technical validity) → Gate 2 (lane admissibility) → bundle emission.

Dispatch constraints from Evidence Manifest (PRD §4.2.1):
  - richness=sparse + absorption_coverage=inadequate → NODE not dispatched
  - nonlinear_clearance_signature=True → automated search includes MM candidates
  - blq_burden > 0.20 → all backends must use BLQ-aware likelihood (M3/M4)
  - protocol_heterogeneity=pooled-heterogeneous → IOV must be tested
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    BackendVersions,
    CandidateLineage,
    CandidateLineageEntry,
    FailedCandidate,
    LOROMetrics,
    SeedRegistry,
)
from apmode.data.initial_estimates import NCAEstimator, build_initial_estimates_bundle
from apmode.data.profiler import profile_data
from apmode.data.splitter import loro_cv_splits, split_subjects
from apmode.evaluation.loro_cv import evaluate_loro_cv
from apmode.governance.gates import (
    evaluate_gate1,
    evaluate_gate2,
    evaluate_gate2_5,
    evaluate_gate3,
)
from apmode.governance.policy import GatePolicy
from apmode.routing import route

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.backends.protocol import BackendRunner
    from apmode.bundle.models import BackendResult, DataManifest, EvidenceManifest
    from apmode.dsl.ast_models import DSLSpec
    from apmode.routing import DispatchDecision
    from apmode.search.engine import SearchOutcome, SearchResult

logger = structlog.get_logger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single APMODE run."""

    lane: Literal["submission", "discovery", "optimization"] = "submission"
    seed: int = 42
    timeout_seconds: int = 600
    policy_path: Path | None = None
    covariate_names: list[str] = field(default_factory=list)
    execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic"


@dataclass
class RunOutcome:
    """Complete outcome of an APMODE run."""

    run_id: str
    bundle_dir: Path
    evidence_manifest: EvidenceManifest | None = None
    dispatch_decision: DispatchDecision | None = None
    search_outcome: SearchOutcome | None = None
    gate1_results: list[tuple[str, bool]] = field(default_factory=list)
    gate2_results: list[tuple[str, bool]] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)
    ranked: list[str] = field(default_factory=list)


class Orchestrator:
    """Central pipeline controller.

    Wires together: ingestion → profiling → NCA → splitting → search →
    governance gates → bundle emission.
    """

    def __init__(
        self,
        runner: Nlmixr2Runner,
        bundle_base_dir: Path,
        config: RunConfig | None = None,
        node_runner: BackendRunner | None = None,
        agentic_runner: BackendRunner | None = None,
    ) -> None:
        self._runner = runner
        self._bundle_base = bundle_base_dir
        self._config = config or RunConfig()
        self._node_runner = node_runner
        self._agentic_runner = agentic_runner
        self._spec_map: dict[str, DSLSpec] = {}  # candidate_id → DSLSpec

    async def run(
        self,
        manifest: DataManifest,
        df: pd.DataFrame,
        data_path: Path,
    ) -> RunOutcome:
        """Execute the full APMODE pipeline.

        Args:
            manifest: DataManifest from ingestion step.
            df: Validated DataFrame (post-CanonicalPKSchema).
            data_path: Path to the nlmixr2-ready CSV file.
        """
        from datetime import datetime

        from apmode.search.engine import SearchEngine

        # Initialize bundle
        emitter = BundleEmitter(self._bundle_base)
        run_dir = emitter.initialize()
        outcome = RunOutcome(run_id=emitter.run_id, bundle_dir=run_dir)

        # Write data manifest
        emitter.write_data_manifest(manifest)

        # Write seed registry
        emitter.write_seed_registry(
            SeedRegistry(
                root_seed=self._config.seed,
                r_seed=self._config.seed,
                r_rng_kind="L'Ecuyer-CMRG",
                np_seed=self._config.seed,
            )
        )

        # Write backend versions
        import sys

        emitter.write_backend_versions(
            BackendVersions(
                apmode_version="0.2.0-dev",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            )
        )

        # --- Stage 1: Data Profiling ---
        evidence = profile_data(df, manifest)
        emitter.write_evidence_manifest(evidence)
        outcome.evidence_manifest = evidence
        logger.info(
            "Profiled: richness=%s, nonlinear_CL=%s, blq=%.2f",
            evidence.richness_category,
            evidence.nonlinear_clearance_signature,
            evidence.blq_burden,
        )

        # --- Stage 1b: Lane Router Dispatch ---
        dispatch = route(self._config.lane, evidence)
        outcome.dispatch_decision = dispatch
        for constraint in dispatch.constraints:
            logger.info("Dispatch constraint: %s", constraint)
        logger.info(
            "Lane=%s, backends=%s, node_eligible=%s",
            dispatch.lane,
            dispatch.backends,
            dispatch.node_eligible,
        )

        # --- Stage 2: NCA Initial Estimates ---
        estimator = NCAEstimator(df, manifest)
        nca_estimates = estimator.estimate_per_subject()
        nca_entry = estimator.build_entry("root", source="nca")
        ie_bundle = build_initial_estimates_bundle([nca_entry])
        emitter.write_initial_estimates(ie_bundle)

        # --- Stage 3: Data Splitting ---
        split = split_subjects(df, seed=self._config.seed)
        emitter.write_split_manifest(split)

        # --- Stage 4: Load Policy ---
        policy = self._load_policy()
        if policy:
            emitter.write_policy_file(policy.model_dump())

        # --- Stage 5: Automated Search ---
        split_manifest_dict = split.model_dump()
        # Build multi-backend runner map for SearchEngine
        runners: dict[str, BackendRunner] = {"nlmixr2": self._runner}
        if self._node_runner is not None and "jax_node" in dispatch.backends:
            runners["jax_node"] = self._node_runner
        if self._agentic_runner is not None and "agentic_llm" in dispatch.backends:
            runners["agentic_llm"] = self._agentic_runner
        search_engine = SearchEngine(
            runner=self._runner,
            data_manifest=manifest,
            data_path=data_path,
            seed=self._config.seed,
            timeout_seconds=self._config.timeout_seconds,
            allowed_backends=dispatch.backends,
            split_manifest=split_manifest_dict,
            runners=runners,
        )
        search_outcome = await search_engine.run(
            evidence_manifest=evidence,
            initial_estimates=nca_estimates,
            emitter=emitter,
            covariate_names=self._config.covariate_names,
        )
        outcome.search_outcome = search_outcome
        # Build candidate→spec mapping for LORO-CV spec lookup
        self._spec_map = {sr.candidate_id: sr.spec for sr in search_outcome.results}

        # Write candidate lineage
        lineage_entries = [
            CandidateLineageEntry(
                candidate_id=e["candidate_id"],
                parent_id=e["parent_id"],
                transform=e["transform"],
            )
            for e in search_outcome.dag.to_lineage_entries()
        ]
        emitter.write_candidate_lineage(CandidateLineage(entries=lineage_entries))

        # --- Stage 6: Governance Gates ---
        from apmode.bundle.models import BackendResult as BRModel

        gate12_survivors: list[BRModel] = []

        if policy:
            # Stage 6a: Pre-screen Gate 1 (without seed stability) to
            # identify top candidates for seed runs (PRD §4.3.1:
            # "Top model consistent across ≥ N random seeds")
            seed_results_map: dict[str, list[BRModel]] = {}
            seed_n = policy.gate1.seed_stability_n
            if seed_n > 1:
                # Rank converged candidates by BIC, seed top-k
                converged_srs = sorted(
                    [
                        sr
                        for sr in search_outcome.results
                        if sr.converged and sr.result is not None
                    ],
                    key=lambda r: r.bic if r.bic is not None else float("inf"),
                )
                # Seed top-k: enough to survive other Gate 1/2 attrition
                top_k = min(len(converged_srs), max(3, seed_n))
                for sr in converged_srs[:top_k]:
                    cid = sr.candidate_id
                    # Use the candidate's own fitted estimates for seed
                    # stability (not NCA root estimates), so warm-start
                    # children are re-evaluated from their discovered minimum
                    if sr.result is not None:
                        cand_estimates = {
                            name: pe.estimate
                            for name, pe in sr.result.parameter_estimates.items()
                            if pe.category == "structural"
                        }
                    else:
                        cand_estimates = {
                            k: v for k, v in nca_estimates.items() if not k.startswith("_")
                        }
                    # Select runner based on spec type
                    seed_runner: BackendRunner = self._runner
                    if sr.spec.has_node_modules() and self._node_runner is not None:
                        seed_runner = self._node_runner
                    seed_runs: list[BRModel] = []
                    for seed_offset in range(1, seed_n):
                        try:
                            seed_result = await seed_runner.run(
                                spec=sr.spec,
                                data_manifest=manifest,
                                initial_estimates=cand_estimates,
                                seed=self._config.seed + seed_offset,
                                timeout_seconds=self._config.timeout_seconds,
                                data_path=data_path,
                                split_manifest=split_manifest_dict,
                            )
                            seed_runs.append(seed_result)
                            emitter.write_seed_result(seed_result, cid, seed_offset)
                        except Exception:
                            logger.warning("Seed run %d failed for %s", seed_offset, cid)
                    if seed_runs:
                        seed_results_map[cid] = seed_runs

            # Stage 6b: Gate 1 evaluation (collect survivors)
            gate1_survivors: list[tuple[SearchResult, BRModel]] = []
            for sr in search_outcome.results:
                if sr.result is None:
                    continue

                emitter.write_backend_result(sr.result)

                seed_results = seed_results_map.get(sr.candidate_id)
                g1 = evaluate_gate1(sr.result, policy, seed_results=seed_results)
                emitter.write_gate_decision(g1, gate_number=1)
                outcome.gate1_results.append((sr.candidate_id, g1.passed))

                if not g1.passed:
                    emitter.append_failed_candidate(
                        FailedCandidate(
                            candidate_id=sr.candidate_id,
                            backend=sr.result.backend,
                            gate_failed="gate1",
                            failed_checks=[c.check_id for c in g1.checks if not c.passed],
                            summary_reason=g1.summary_reason,
                            timestamp=datetime.now(tz=UTC).isoformat(),
                        )
                    )
                    continue

                gate1_survivors.append((sr, sr.result))

            # Stage 6b.5: LORO-CV for optimization lane (after Gate 1, before Gate 2)
            loro_metrics_map: dict[str, LOROMetrics] = {}
            if self._config.lane == "optimization" and gate1_survivors:
                loro_metrics_map = await self._run_loro_cv(
                    gate1_survivors=[r for _, r in gate1_survivors],
                    df=df,
                    manifest=manifest,
                    data_path=data_path,
                    nca_estimates=nca_estimates,
                    emitter=emitter,
                    policy=policy,
                )

            # Stage 6c: Gate 2 + Gate 2.5 evaluation (with LORO metrics)
            for _sr, sr_result in gate1_survivors:
                loro_m = loro_metrics_map.get(sr_result.model_id)

                g2 = evaluate_gate2(sr_result, policy, self._config.lane, loro_metrics=loro_m)
                emitter.write_gate_decision(g2, gate_number=2)
                outcome.gate2_results.append((sr_result.model_id, g2.passed))

                if not g2.passed:
                    emitter.append_failed_candidate(
                        FailedCandidate(
                            candidate_id=sr_result.model_id,
                            backend=sr_result.backend,
                            gate_failed="gate2",
                            failed_checks=[c.check_id for c in g2.checks if not c.passed],
                            summary_reason=g2.summary_reason,
                            timestamp=datetime.now(tz=UTC).isoformat(),
                        )
                    )
                    continue

                # Gate 2.5: Credibility Qualification
                from apmode.bundle.models import CredibilityContext

                cred_ctx = CredibilityContext(
                    context_of_use=f"{self._config.lane} lane analysis",
                    risk_level="medium",
                    n_observations=manifest.n_observations,
                    n_parameters=len(sr_result.parameter_estimates),
                    ml_transparency_statement=(
                        f"Backend: {sr_result.backend}"
                        if sr_result.backend in ("jax_node", "agentic_llm")
                        else None
                    ),
                )
                g25 = evaluate_gate2_5(sr_result, policy, credibility_context=cred_ctx)
                emitter.write_gate_decision(g25, gate_number=25)

                if not g25.passed:
                    emitter.append_failed_candidate(
                        FailedCandidate(
                            candidate_id=sr_result.model_id,
                            backend=sr_result.backend,
                            gate_failed="gate2_5",
                            failed_checks=[c.check_id for c in g25.checks if not c.passed],
                            summary_reason=g25.summary_reason,
                            timestamp=datetime.now(tz=UTC).isoformat(),
                        )
                    )
                    continue

                # Passed gates 1, 2, 2.5 — emit credibility report
                from apmode.report.credibility import generate_credibility_report

                cred_report = generate_credibility_report(
                    sr_result,
                    lane=self._config.lane,
                    n_observations=manifest.n_observations,
                )
                emitter.write_credibility_report(cred_report)

                outcome.recommended.append(sr_result.model_id)
                gate12_survivors.append(sr_result)

            # Stage 6c: Gate 3 — Ranking (within- or cross-paradigm)
            if gate12_survivors:
                g3_result, ranked = evaluate_gate3(gate12_survivors, policy)
                emitter.write_gate_decision(g3_result, gate_number=3)
                outcome.ranked = [rc.candidate_id for rc in ranked]

                # Write ranking.json with full ordered candidate list
                from apmode.bundle.models import RankedCandidateEntry, Ranking

                ranking = Ranking(
                    ranked_candidates=[
                        RankedCandidateEntry(
                            candidate_id=rc.candidate_id,
                            rank=rc.rank,
                            bic=rc.bic,
                            aic=rc.aic,
                            n_params=rc.n_params,
                            backend=rc.backend,
                        )
                        for rc in ranked
                    ],
                    best_candidate_id=ranked[0].candidate_id if ranked else None,
                    ranking_metric="bic",
                    n_survivors=len(ranked),
                )
                emitter.write_ranking(ranking)

        # --- Stage 7: Render human-readable report ---
        if gate12_survivors:
            from apmode.report.renderer import render_run_report

            cred_reports = [
                generate_credibility_report(r, self._config.lane, manifest.n_observations)
                for r in gate12_survivors
            ]
            report_md = render_run_report(
                run_id=emitter.run_id,
                lane=self._config.lane,
                manifest=manifest,
                evidence=evidence,
                ranked=gate12_survivors,
                ranking=ranking if "ranking" in dir() else None,
                credibility_reports=cred_reports,
                failed_count=len(search_outcome.results) - len(gate12_survivors),
                total_candidates=len(search_outcome.results),
            )
            report_path = emitter.run_dir / "report.md"
            report_path.write_text(report_md)

        # --- Stage 7b: Report Provenance ---
        from apmode.bundle.models import ReportProvenance

        emitter.write_report_provenance(
            ReportProvenance(
                generated_at=datetime.now(tz=UTC).isoformat(),
                apmode_version="0.2.0-dev",
                generator="apmode.orchestrator",
                component_versions={
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "pipeline": "Phase3",
                },
            )
        )

        logger.info(
            "Run %s complete: %d candidates, %d recommended, %d ranked",
            emitter.run_id,
            len(search_outcome.results),
            len(outcome.recommended),
            len(outcome.ranked),
        )

        return outcome

    async def _run_loro_cv(
        self,
        gate1_survivors: list[BackendResult],
        df: pd.DataFrame,
        manifest: DataManifest,
        data_path: Path,
        nca_estimates: dict[str, float],
        emitter: BundleEmitter,
        policy: GatePolicy,
    ) -> dict[str, LOROMetrics]:
        """Run LORO-CV for optimization lane Gate 1 survivors.

        Returns a map of candidate_id → LOROMetrics for use by Gate 2.
        Respects policy.gate2.loro_budget_top_n to limit computation.
        """
        import numpy as np

        # Generate LORO folds from the data
        try:
            folds = loro_cv_splits(
                df,
                seed=self._config.seed,
                min_folds=policy.gate2.loro_min_folds,
            )
        except ValueError:
            logger.warning("loro_cv_insufficient_regimens", min_folds=policy.gate2.loro_min_folds)
            return {}

        # Budget control: only evaluate top-N by BIC
        budget = policy.gate2.loro_budget_top_n
        candidates = sorted(
            gate1_survivors,
            key=lambda r: r.bic if r.bic is not None and np.isfinite(r.bic) else float("inf"),
        )
        if budget is not None:
            candidates = candidates[:budget]

        logger.info(
            "loro_cv_start",
            n_candidates=len(candidates),
            n_folds=len(folds),
            lane=self._config.lane,
        )

        loro_map: dict[str, LOROMetrics] = {}
        for cand_result in candidates:
            # Select the appropriate runner for this candidate's backend
            runner: BackendRunner = self._runner
            if cand_result.backend == "jax_node" and self._node_runner is not None:
                runner = self._node_runner
            elif cand_result.backend == "agentic_llm" and self._agentic_runner is not None:
                runner = self._agentic_runner

            # Use candidate's fitted estimates for warm-start
            warm_estimates = {
                name: pe.estimate
                for name, pe in cand_result.parameter_estimates.items()
                if pe.category == "structural"
            }
            if not warm_estimates:
                warm_estimates = {k: v for k, v in nca_estimates.items() if not k.startswith("_")}

            # Need the candidate's DSLSpec — find it from search results
            spec = self._spec_map.get(cand_result.model_id)
            if spec is None:
                logger.warning("loro_cv_no_spec", candidate=cand_result.model_id)
                continue

            try:
                loro_result = await evaluate_loro_cv(
                    candidate_spec=spec,
                    candidate_result=cand_result,
                    folds=folds,
                    runner=runner,
                    data_manifest=manifest,
                    data_path=data_path,
                    initial_estimates=warm_estimates,
                    seed=self._config.seed,
                    timeout_seconds=self._config.timeout_seconds,
                )
                emitter.write_loro_cv_result(loro_result)
                loro_map[cand_result.model_id] = loro_result.metrics
                logger.info(
                    "loro_cv_complete",
                    candidate=cand_result.model_id,
                    npde_mean=loro_result.metrics.pooled_npde_mean,
                    vpc_concordance=loro_result.metrics.vpc_coverage_concordance,
                )
            except Exception:
                logger.warning(
                    "loro_cv_candidate_failed",
                    candidate=cand_result.model_id,
                    exc_info=True,
                )

        return loro_map

    def _load_policy(self) -> GatePolicy | None:
        """Load gate policy from file or default."""
        if self._config.policy_path and self._config.policy_path.exists():
            data = json.loads(self._config.policy_path.read_text())
            return GatePolicy.model_validate(data)

        # Try default policy directory
        default_path = Path("policies") / f"{self._config.lane}.json"
        if default_path.exists():
            data = json.loads(default_path.read_text())
            return GatePolicy.model_validate(data)

        return None
