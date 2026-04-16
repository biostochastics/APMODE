# SPDX-License-Identifier: GPL-2.0-or-later
"""Orchestrator: asyncio event loop, dispatch, pipeline control.

Full pipeline per ARCHITECTURE.md §3:
  ingest → profile_data() → NCA estimates → data split → search dispatch →
  Gate 1 (technical validity) → Gate 2 (lane admissibility) → bundle emission.

Dispatch constraints from Evidence Manifest (PRD §4.2.1):
  - richness=sparse + absorption_coverage=inadequate → NODE not dispatched
  - nonlinear_clearance_evidence_strength="strong" → automated search includes MM candidates
  - blq_burden > 0.20 → all backends must use BLQ-aware likelihood (M3/M4)
  - protocol_heterogeneity=pooled-heterogeneous → IOV must be tested
"""

from __future__ import annotations

import asyncio
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import ValidationError

from apmode import __version__ as _apmode_version
from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    BackendVersions,
    CandidateLineage,
    CandidateLineageEntry,
    FailedCandidate,
    LOROCVResult,
    LOROMetrics,
    SeedRegistry,
)
from apmode.data.initial_estimates import NCAEstimator, build_initial_estimates_bundle
from apmode.data.profiler import profile_data
from apmode.data.splitter import loro_cv_splits, split_subjects
from apmode.errors import AgenticExhaustionError, BackendError
from apmode.evaluation.loro_cv import evaluate_loro_cv
from apmode.governance.gates import (
    evaluate_gate1,
    evaluate_gate2,
    evaluate_gate2_5,
    evaluate_gate3,
)
from apmode.governance.policy import GatePolicy
from apmode.report.credibility import generate_credibility_report
from apmode.routing import route

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.backends.protocol import BackendRunner
    from apmode.bundle.models import (
        BackendResult,
        DataManifest,
        EvidenceManifest,
        ImputationStabilityManifest,
        MissingDataDirective,
        Ranking,
    )
    from apmode.dsl.ast_models import DSLSpec
    from apmode.routing import DispatchDecision
    from apmode.search.engine import SearchOutcome, SearchResult
    from apmode.search.stability import ImputationProvider

logger = structlog.get_logger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single APMODE run."""

    lane: Literal["submission", "discovery", "optimization"] = "submission"
    seed: int = 753849
    timeout_seconds: int = 900
    policy_path: Path | None = None
    covariate_names: list[str] = field(default_factory=list)
    execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic"
    max_concurrency: int = 1
    # Literature priors (e.g., DatasetCard.published_model.key_estimates).
    # When provided, NCAEstimator uses these as root initial estimates after
    # per-subject NCA exclusion exceeds 50% — prevents SAEM from seeding on
    # degenerate defaults when a reliable prior is known.
    fallback_estimates: dict[str, float] | None = None
    # Per-column binary encoding overrides for categorical covariates whose
    # auto-detected polarity is wrong for the analysis (e.g., overriding
    # the alphabetic-default mapping of two unknown string levels). Maps
    # column name → {raw_value: 0 or 1}. Threaded through to
    # ``summarize_covariates`` and ``prepare_frem_data`` for FREM and MI
    # paths. See ``apmode.data.categorical_encoding.EXPECTED_BINARY_FORMAT``
    # for the convention APMODE auto-detects.
    binary_encode_overrides: dict[str, dict[object, int]] | None = None

    def __post_init__(self) -> None:
        self.max_concurrency = max(1, self.max_concurrency)


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
        frem_runner: Nlmixr2Runner | None = None,
        mi_provider: ImputationProvider | None = None,
    ) -> None:
        """Construct the orchestrator.

        Args:
            frem_runner: Optional FOCE-I-configured ``Nlmixr2Runner`` used
                for the FREM execution stage. When omitted but the
                missing-data directive resolves to ``FREM``, the orchestrator
                instantiates a default ``Nlmixr2Runner(estimation=["focei"])``
                that shares the main runner's work directory. SAEM cannot
                fit FREM endpoints, so the default explicitly forces FOCE-I.
            mi_provider: Optional ``ImputationProvider`` used for the MI
                execution stage. When omitted but the directive resolves to
                ``MI-PMM`` / ``MI-missRanger``, the orchestrator instantiates
                ``R_MiceImputer`` or ``R_MissRangerImputer`` automatically.
        """
        self._runner = runner
        self._bundle_base = bundle_base_dir
        self._config = config or RunConfig()
        self._node_runner = node_runner
        self._agentic_runner = agentic_runner
        self._frem_runner = frem_runner
        self._mi_provider = mi_provider
        self._spec_map: dict[str, DSLSpec] = {}  # candidate_id → DSLSpec

    async def run(
        self,
        manifest: DataManifest,
        df: pd.DataFrame,
        data_path: Path,
        *,
        skip_classical: bool = False,
    ) -> RunOutcome:
        """Execute the full APMODE pipeline.

        Args:
            manifest: DataManifest from ingestion step.
            df: Validated DataFrame (post-CanonicalPKSchema).
            data_path: Path to the nlmixr2-ready CSV file.
            skip_classical: When True, look for an existing
                ``classical_checkpoint.json`` in the bundle base directory and
                skip Stage 5 (automated search), jumping straight to the
                agentic refinement loop.  Use ``apmode run --resume-agentic``
                to restart a run after an LLM API failure without re-running
                the expensive classical SAEM search.
        """
        from apmode.search.engine import SearchEngine

        # Initialize bundle — when resuming, reuse the existing run dir so
        # all artifacts land in the same bundle as the original classical run.
        existing_run_id: str | None = None
        if skip_classical:
            existing_run_id = self._find_existing_run_id()
            if existing_run_id is None:
                # Fail-fast: silently falling back would burn 55+ min SAEM.
                # The user explicitly asked to skip classical search; if we
                # cannot locate exactly one bundle to resume, tell them why.
                n_dirs = sum(
                    1
                    for d in (self._bundle_base.iterdir() if self._bundle_base.is_dir() else [])
                    if d.is_dir() and not d.name.startswith("_")
                )
                if n_dirs == 0:
                    raise RuntimeError(
                        f"--resume-agentic requested but no existing bundle found in "
                        f"{self._bundle_base}. Run without --resume-agentic first."
                    )
                raise RuntimeError(
                    f"--resume-agentic requested but {n_dirs} bundle directories found in "
                    f"{self._bundle_base} — cannot auto-select. "
                    f"Remove all but the target bundle directory and retry."
                )

        emitter = BundleEmitter(self._bundle_base, run_id=existing_run_id)
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
        emitter.write_backend_versions(
            BackendVersions(
                apmode_version=_apmode_version,
                python_version=platform.python_version(),
            )
        )

        # --- Stage 1: Data Profiling ---
        evidence = profile_data(df, manifest)
        emitter.write_evidence_manifest(evidence)
        outcome.evidence_manifest = evidence
        logger.info(
            "Profiled: richness=%s, nonlinear_CL=%s, blq=%.2f",
            evidence.richness_category,
            evidence.nonlinear_clearance_evidence_strength,
            evidence.blq_burden,
        )

        # --- Stage 1b: Load Policy (before routing so the missing-data
        # directive can be resolved and attached to DispatchDecision) ---
        policy = self._load_policy()
        if policy:
            emitter.write_policy_file(policy.model_dump())

        # --- Stage 1c: Lane Router Dispatch ---
        missing_data_policy = policy.missing_data if policy is not None else None
        dispatch = route(self._config.lane, evidence, missing_data_policy)
        outcome.dispatch_decision = dispatch
        for constraint in dispatch.constraints:
            logger.info("Dispatch constraint: %s", constraint)
        logger.info(
            "Lane=%s, backends=%s, node_eligible=%s",
            dispatch.lane,
            dispatch.backends,
            dispatch.node_eligible,
        )
        if dispatch.missing_data_directive is not None:
            directive = dispatch.missing_data_directive
            emitter.write_missing_data_directive(directive)
            logger.info(
                "Missing-data directive: covariate=%s, m=%s, BLQ=%s, pooled_only=%s",
                directive.covariate_method,
                directive.m_imputations,
                directive.blq_method,
                directive.llm_pooled_only,
            )
            for reason in directive.rationale:
                logger.info("Missing-data rationale: %s", reason)
            if directive.covariate_method.startswith("MI-"):
                logger.warning(
                    "Multiple-imputation execution path is staged but not yet "
                    "driven by the orchestrator loop. Directive is recorded; use "
                    "apmode.search.stability.run_with_imputations directly for "
                    "end-to-end execution in this release."
                )
            elif directive.covariate_method == "FREM":
                logger.info(
                    "FREM emitter available via apmode.dsl.frem_emitter. "
                    "Directive recorded; orchestrator FREM refit-on-best is a "
                    "follow-up PR."
                )

        # --- Stage 2: NCA Initial Estimates ---
        estimator = NCAEstimator(
            df,
            manifest,
            fallback_estimates=self._config.fallback_estimates,
        )
        nca_estimates = estimator.estimate_per_subject()
        nca_entry = estimator.build_entry("root", source="nca", estimates=nca_estimates)
        ie_bundle = build_initial_estimates_bundle([nca_entry])
        emitter.write_initial_estimates(ie_bundle)
        if estimator.diagnostics:
            emitter.write_nca_diagnostics(estimator.diagnostics)
            estimator.emit_plots(emitter.nca_plots_dir())
        logger.info(
            "NCA initial estimates: source=%s, n_subjects=%d, n_excluded=%d",
            estimator.fallback_source,
            len(estimator.diagnostics),
            sum(1 for d in estimator.diagnostics if d.excluded),
        )

        # --- Stage 3: Data Splitting ---
        split = split_subjects(df, seed=self._config.seed)
        emitter.write_split_manifest(split)

        # --- Stage 5: Automated Search ---
        split_manifest_dict = split.model_dump()

        # Resume path: load classical results from checkpoint, skip R search.
        _checkpoint_loaded = False
        if skip_classical:
            _ckpt = self._load_classical_checkpoint(run_dir)
            if _ckpt is not None:
                search_outcome, nca_estimates = _ckpt
                _checkpoint_loaded = True
                logger.info(
                    "Resuming from classical checkpoint: skipping Stage 5 (%d candidates loaded)",
                    len(search_outcome.results),
                )

        if not _checkpoint_loaded:
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
                max_concurrency=self._config.max_concurrency,
            )
            search_outcome = await search_engine.run(
                evidence_manifest=evidence,
                initial_estimates=nca_estimates,
                emitter=emitter,
                covariate_names=self._config.covariate_names,
            )
            # Write checkpoint immediately so --resume-agentic can restart
            # the agentic loop without re-running the classical search.
            self._write_classical_checkpoint(run_dir, search_outcome, nca_estimates)

        outcome.search_outcome = search_outcome
        # Build candidate→spec mapping for LORO-CV spec lookup
        self._spec_map = {sr.candidate_id: sr.spec for sr in search_outcome.results}

        # --- Stage 5b: Agentic LLM Refinement (Phase 3, PRD §4.2.6) ---
        # The agentic runner operates as a post-search refinement loop:
        #   Mode 1 (refine): take best classical candidate, improve via transforms
        #   Mode 2 (independent): start from a base spec, LLM builds from scratch
        # In discovery/optimization lanes, the LLM can also propose NODE transforms.
        if self._agentic_runner is not None and "agentic_llm" in dispatch.backends:
            agentic_results = await self._run_agentic_stage(
                search_outcome=search_outcome,
                manifest=manifest,
                data_path=data_path,
                nca_estimates=nca_estimates,
                split_manifest_dict=split_manifest_dict,
            )
            # Merge agentic results into search outcome
            for ar in agentic_results:
                search_outcome.results.append(ar)
                if ar.spec is not None:
                    self._spec_map[ar.candidate_id] = ar.spec

        # --- Stage 5c: Missing-Data Execution (FREM single-fit or MI loop) ---
        # When the policy resolves to FREM or MI-*, drive the corresponding
        # execution helper with the best classical candidate (or the spec
        # template) as the starting point. Bundle artifacts:
        #   - frem_<id>_result.json     (FREM path, written via emitter)
        #   - imputation_stability.json (MI path)
        # Failures here are logged but do not abort the run — the classical
        # search results remain available so governance can still rank them.
        stability_manifest: ImputationStabilityManifest | None = None
        if dispatch.missing_data_directive is not None:
            directive = dispatch.missing_data_directive
            covariate_names = self._config.covariate_names
            if directive.covariate_method == "FREM" and covariate_names:
                try:
                    frem_result = await self._run_frem_stage(
                        search_outcome=search_outcome,
                        df=df,
                        data_path=data_path,
                        manifest=manifest,
                        covariate_names=covariate_names,
                        run_dir=run_dir,
                        nca_estimates=nca_estimates,
                    )
                    if frem_result is not None:
                        search_outcome.results.append(frem_result)
                        if frem_result.spec is not None:
                            self._spec_map[frem_result.candidate_id] = frem_result.spec
                        if frem_result.result is not None:
                            emitter.write_backend_result(frem_result.result)
                except (
                    BackendError,
                    RuntimeError,
                    ValueError,
                    NotImplementedError,
                ) as e:
                    logger.warning("FREM execution failed: %s", e, exc_info=True)
            elif directive.covariate_method.startswith("MI-") and covariate_names:
                try:
                    stability_manifest = await self._run_mi_stage(
                        directive=directive,
                        search_outcome=search_outcome,
                        data_path=data_path,
                        manifest=manifest,
                        covariate_names=covariate_names,
                        run_dir=run_dir,
                        nca_estimates=nca_estimates,
                    )
                    if stability_manifest is not None:
                        emitter.write_imputation_stability(stability_manifest)
                except (
                    BackendError,
                    RuntimeError,
                    ValueError,
                    NotImplementedError,
                ) as e:
                    logger.warning("MI execution failed: %s", e, exc_info=True)

        # Write candidate lineage — classical entries come from the search DAG,
        # agentic entries from the _run_agentic_stage results (the DAG's
        # SearchNode requires a full DSLSpec that matches the final candidate
        # id, which we don't have post-agentic; recording them here keeps the
        # reproducibility bundle consistent without polluting the DAG type).
        # On checkpoint resume the SearchOutcome.dag is empty (not serialized).
        # Read the existing candidate_lineage.json to preserve classical entries;
        # only append net-new agentic entries produced in this session.
        if _checkpoint_loaded:
            _existing_lineage_path = run_dir / "candidate_lineage.json"
            if _existing_lineage_path.exists():
                try:
                    _existing = CandidateLineage.model_validate_json(
                        _existing_lineage_path.read_text()
                    )
                    lineage_entries = list(_existing.entries)
                except (json.JSONDecodeError, ValidationError, OSError) as exc:
                    logger.warning(
                        "candidate_lineage_load_failed",
                        path=str(_existing_lineage_path),
                        error=str(exc),
                    )
                    lineage_entries = []
            else:
                lineage_entries = []
        else:
            lineage_entries = [
                CandidateLineageEntry(
                    candidate_id=e["candidate_id"],
                    parent_id=e["parent_id"],
                    transform=e["transform"],
                )
                for e in search_outcome.dag.to_lineage_entries()
            ]
        dag_ids = {e.candidate_id for e in lineage_entries}
        for ar in search_outcome.results:
            if ar.result and ar.result.backend == "agentic_llm" and ar.candidate_id not in dag_ids:
                lineage_entries.append(
                    CandidateLineageEntry(
                        candidate_id=ar.candidate_id,
                        # Full chain is in agentic_trace/<mode>/agentic_lineage.json
                        parent_id=None,
                        transform="agentic_llm",
                    )
                )
                dag_ids.add(ar.candidate_id)
        emitter.write_candidate_lineage(CandidateLineage(entries=lineage_entries))

        # --- Stage 6: Governance Gates ---
        # On checkpoint resume, clear failed_candidates.jsonl before re-running
        # gate evaluation to prevent duplicate entries from the original run.
        if _checkpoint_loaded:
            _fc_path = run_dir / "failed_candidates.jsonl"
            if _fc_path.exists():
                _fc_path.unlink()
        from apmode.bundle.models import BackendResult as BRModel

        gate12_survivors: list[BRModel] = []
        ranking: Ranking | None = None

        if policy:
            # Stage 6a: Pre-screen Gate 1 (without seed stability) to
            # identify top candidates for seed runs (PRD §4.3.1:
            # "Top model consistent across ≥ N random seeds")
            seed_results_map: dict[str, list[BRModel]] = {}
            seed_n = policy.gate1.seed_stability_n
            if seed_n > 1:
                converged_srs = sorted(
                    [
                        sr
                        for sr in search_outcome.results
                        if sr.converged and sr.result is not None
                    ],
                    key=lambda r: r.bic if r.bic is not None else float("inf"),
                )
                top_k = min(len(converged_srs), max(3, seed_n))
                sem = asyncio.Semaphore(self._config.max_concurrency)

                async def _seed_run(
                    sr: SearchResult,
                    seed_offset: int,
                ) -> tuple[str, int, BRModel | None]:
                    cid = sr.candidate_id
                    if sr.result is not None:
                        cand_est = {
                            name: pe.estimate
                            for name, pe in sr.result.parameter_estimates.items()
                            if pe.category == "structural"
                        }
                    else:
                        cand_est = {
                            k: v for k, v in nca_estimates.items() if not k.startswith("_")
                        }
                    seed_runner: BackendRunner = self._runner
                    if sr.spec.has_node_modules() and self._node_runner is not None:
                        seed_runner = self._node_runner
                    async with sem:
                        try:
                            result = await seed_runner.run(
                                spec=sr.spec,
                                data_manifest=manifest,
                                initial_estimates=cand_est,
                                seed=self._config.seed + seed_offset,
                                timeout_seconds=self._config.timeout_seconds,
                                data_path=data_path,
                                split_manifest=split_manifest_dict,
                            )
                            return cid, seed_offset, result
                        except BackendError as e:
                            logger.warning("Seed run %d failed for %s: %s", seed_offset, cid, e)
                            return cid, seed_offset, None

                # Build all seed tasks for all top-k candidates.
                # TaskGroup provides structured cancellation if any inner
                # coroutine raises unexpectedly (_seed_run catches its own
                # failures, so this is a defensive boundary).
                async with asyncio.TaskGroup() as tg:
                    seed_task_handles = [
                        tg.create_task(_seed_run(sr, offset))
                        for sr in converged_srs[:top_k]
                        for offset in range(1, seed_n)
                    ]
                seed_outcomes = [t.result() for t in seed_task_handles]

                # Collect results per candidate
                for cid, seed_offset, result in seed_outcomes:
                    if result is not None:
                        seed_results_map.setdefault(cid, []).append(result)
                        emitter.write_seed_result(result, cid, seed_offset)

            # Stage 6b: Gate 1 evaluation (collect survivors)
            gate1_survivors: list[tuple[SearchResult, BRModel]] = []
            for sr in search_outcome.results:
                if sr.result is None:
                    continue

                emitter.write_backend_result(sr.result)

                seed_results = seed_results_map.get(sr.candidate_id)
                # When MI is active, look up this candidate's per-imputation
                # stability entry and pass it to Gate 1 — the imputation-
                # stability check uses ``convergence_rate`` and
                # ``rank_stability`` to disqualify candidates that flip
                # across imputations. Non-MI runs leave both at None and
                # the check marks itself ``not_applicable``.
                stability_entry = None
                if stability_manifest is not None:
                    stability_entry = next(
                        (
                            e
                            for e in stability_manifest.entries
                            if e.candidate_id == sr.candidate_id
                        ),
                        None,
                    )
                g1 = evaluate_gate1(
                    sr.result,
                    policy,
                    seed_results=seed_results,
                    directive=dispatch.missing_data_directive,
                    stability=stability_entry,
                )
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
            for _, sr_result in gate1_survivors:
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

                # Passed gates 1, 2, 2.5 — emit credibility report.
                # The directive is forwarded so that MI Ω-pooling caveats
                # are appended to the limitations block automatically.
                cred_report = generate_credibility_report(
                    sr_result,
                    lane=self._config.lane,
                    n_observations=manifest.n_observations,
                    directive=dispatch.missing_data_directive,
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
                generate_credibility_report(
                    r,
                    self._config.lane,
                    manifest.n_observations,
                    directive=dispatch.missing_data_directive,
                )
                for r in gate12_survivors
            ]
            report_md = render_run_report(
                run_id=emitter.run_id,
                lane=self._config.lane,
                manifest=manifest,
                evidence=evidence,
                ranked=gate12_survivors,
                ranking=ranking,
                credibility_reports=cred_reports,
                failed_count=len(search_outcome.results) - len(gate12_survivors),
                total_candidates=len(search_outcome.results),
            )
            report_path = emitter.run_dir / "report.md"
            report_path.write_text(report_md)

            # HTML export (rc1: rich-rendered terminal-style HTML).
            from apmode.report.renderer import render_markdown_to_html

            report_html = render_markdown_to_html(report_md, title=f"APMODE Run {emitter.run_id}")
            (emitter.run_dir / "report.html").write_text(report_html)

        # --- Stage 7b: Report Provenance ---
        from apmode.bundle.models import ReportProvenance

        emitter.write_report_provenance(
            ReportProvenance(
                generated_at=datetime.now(tz=UTC).isoformat(),
                apmode_version=_apmode_version,
                generator="apmode.orchestrator",
                component_versions={
                    "python": platform.python_version(),
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

    async def _run_agentic_stage(
        self,
        search_outcome: SearchOutcome,
        manifest: DataManifest,
        data_path: Path,
        nca_estimates: dict[str, float],
        split_manifest_dict: dict[str, object],
    ) -> list[SearchResult]:
        """Run the agentic LLM refinement stage (PRD §4.2.6).

        Two modes:
          1. Refine: take the best converged classical candidate and improve it
          2. Independent: start from a base spec and let the LLM build from scratch

        In discovery/optimization lanes the LLM may also propose NODE transforms.

        Returns new SearchResult entries to merge into search_outcome.
        """
        assert self._agentic_runner is not None

        from apmode.backends.agentic_runner import AgenticRunner
        from apmode.search.engine import SearchResult as SR

        # Concrete-type check — orchestrator stores as BackendRunner protocol
        # but needs AgenticRunner-specific trace-dir scoping here. Using an
        # explicit TypeError (not `assert`) keeps the guard effective under
        # ``python -O``, where assertions are stripped.
        if not isinstance(self._agentic_runner, AgenticRunner):
            msg = (
                f"_run_agentic_stage requires AgenticRunner, got "
                f"{type(self._agentic_runner).__name__}"
            )
            raise TypeError(msg)
        agentic = self._agentic_runner
        base_trace_dir = agentic.trace_dir

        results: list[SR] = []

        # NOTE: refine and independent modes run serially. Parallelization
        # via TaskGroup is tempting — distinct trace subdirs, distinct
        # seeds — but the underlying ``inner_runner`` (Nlmixr2Runner) shares
        # a single ``work_dir`` across both calls, so R subprocess writes
        # would race on intermediate files. Any future parallelization
        # must (1) give each mode its own Nlmixr2Runner with a distinct
        # ``work_dir``, and (2) verify the LLM client has no shared
        # mutable state (connection pool, rate limiter).

        # --- Mode 1: Refine best classical candidate ---
        converged_srs = sorted(
            [sr for sr in search_outcome.results if sr.converged and sr.result is not None],
            key=lambda r: r.bic if r.bic is not None else float("inf"),
        )
        if converged_srs:
            best_sr = converged_srs[0]
            assert best_sr.result is not None
            best_estimates = {
                name: pe.estimate
                for name, pe in best_sr.result.parameter_estimates.items()
                if pe.category == "structural"
            }
            logger.info(
                "Agentic refine: starting from %s (BIC=%.1f)",
                best_sr.candidate_id,
                best_sr.bic or 0,
            )
            try:
                with agentic.with_trace_dir(base_trace_dir / "refine"):
                    agentic_result = await agentic.run(
                        spec=best_sr.spec,
                        data_manifest=manifest,
                        initial_estimates=best_estimates,
                        seed=self._config.seed,
                        timeout_seconds=self._config.timeout_seconds,
                        data_path=data_path,
                        split_manifest=split_manifest_dict,
                    )
                results.append(
                    SR(
                        candidate_id=agentic_result.model_id,
                        spec=best_sr.spec,
                        result=agentic_result,
                        converged=agentic_result.converged,
                        bic=agentic_result.bic,
                        aic=agentic_result.aic,
                    )
                )
                logger.info(
                    "Agentic refine complete: %s BIC=%.1f (was %.1f)",
                    agentic_result.model_id,
                    agentic_result.bic or 0,
                    best_sr.bic or 0,
                )
            except AgenticExhaustionError as e:
                logger.warning(
                    "Agentic refine exhausted: no converged result after %s iterations",
                    e.iterations,
                )
            except BackendError as e:
                logger.warning("Agentic refine failed (backend error): %s", e)

        # --- Mode 2: Independent — start from a minimal base spec ---
        from apmode.dsl.ast_models import (
            IIV,
            DSLSpec,
            FirstOrder,
            LinearElim,
            OneCmt,
            Proportional,
        )
        from apmode.ids import generate_candidate_id

        base_spec = DSLSpec(
            model_id=generate_candidate_id(),
            absorption=FirstOrder(ka=nca_estimates.get("ka", 1.0)),
            distribution=OneCmt(V=nca_estimates.get("V", 50.0)),
            elimination=LinearElim(CL=nca_estimates.get("CL", 3.0)),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.15),
        )
        base_estimates = {
            "ka": nca_estimates.get("ka", 1.0),
            "V": nca_estimates.get("V", 50.0),
            "CL": nca_estimates.get("CL", 3.0),
        }
        logger.info("Agentic independent: starting from base 1-cmt oral spec")
        try:
            with agentic.with_trace_dir(base_trace_dir / "independent"):
                independent_result = await agentic.run(
                    spec=base_spec,
                    data_manifest=manifest,
                    initial_estimates=base_estimates,
                    seed=self._config.seed + 1000,
                    timeout_seconds=self._config.timeout_seconds,
                    data_path=data_path,
                    split_manifest=split_manifest_dict,
                )
            results.append(
                SR(
                    candidate_id=independent_result.model_id,
                    spec=base_spec,
                    result=independent_result,
                    converged=independent_result.converged,
                    bic=independent_result.bic,
                    aic=independent_result.aic,
                )
            )
            logger.info(
                "Agentic independent complete: %s BIC=%.1f",
                independent_result.model_id,
                independent_result.bic or 0,
            )
        except AgenticExhaustionError as e:
            logger.warning(
                "Agentic independent exhausted: no converged result after %s iterations",
                e.iterations,
            )
        except BackendError as e:
            logger.warning("Agentic independent failed (backend error): %s", e)

        return results

    async def _run_frem_stage(
        self,
        *,
        search_outcome: SearchOutcome,
        df: pd.DataFrame,
        data_path: Path,
        manifest: DataManifest,
        covariate_names: list[str],
        run_dir: Path,
        nca_estimates: dict[str, float],
    ) -> SearchResult | None:
        """Run a single FREM fit on the best classical candidate.

        Builds a FOCE-I-only ``Nlmixr2Runner`` (or uses the injected
        ``self._frem_runner``), composes the FREM emitter pipeline via
        :func:`run_frem_fit`, and returns a ``SearchResult`` whose
        ``candidate_id`` is prefixed with ``frem_`` so downstream
        bookkeeping and ranking can identify the FREM fit distinctly.

        Returns ``None`` when no converged classical candidate exists to
        seed the FREM template — without a converged spec we have no
        principled basis for the joint Ω structure.
        """
        from apmode.backends.frem_runner import run_frem_fit
        from apmode.backends.nlmixr2_runner import Nlmixr2Runner
        from apmode.search.engine import SearchResult as _SR

        # Filter to healthy CLASSICAL candidates only: NODE/agentic specs
        # cannot be fit via the FREM emitter, and ill-conditioned or
        # unidentifiable warm-starts destabilize the joint Ω fit. We
        # require converged + nlmixr2 backend + finite BIC + not
        # ill-conditioned.
        converged = [
            sr
            for sr in search_outcome.results
            if (
                sr.converged
                and sr.result is not None
                and sr.spec is not None
                and sr.result.backend == "nlmixr2"
                and sr.bic is not None
                and sr.bic != float("inf")
                and not sr.result.diagnostics.identifiability.ill_conditioned
            )
        ]
        if not converged:
            logger.info(
                "FREM stage skipped: no healthy classical candidate "
                "(converged + identifiable + finite BIC) to seed the refit"
            )
            return None
        best = min(converged, key=lambda r: r.bic if r.bic is not None else float("inf"))

        frem_work_dir = run_dir / "frem"
        frem_work_dir.mkdir(parents=True, exist_ok=True)
        runner = self._frem_runner or Nlmixr2Runner(work_dir=frem_work_dir, estimation=["focei"])
        # Use the best candidate's structural parameter estimates as
        # initial values for the FREM refit (warm-started FREM fit).
        if best.result is not None:
            init = {
                name: pe.estimate
                for name, pe in best.result.parameter_estimates.items()
                if pe.category == "structural"
            }
        else:
            init = {k: v for k, v in nca_estimates.items() if not k.startswith("_")}

        # Reuse the best spec but tag the FREM-fit model_id distinctly so
        # bundle artifacts and lineage can identify it.
        frem_spec = best.spec.model_copy(update={"model_id": f"frem_{best.spec.model_id}"})
        result = await run_frem_fit(
            spec_template=frem_spec,
            df=df,
            data_path=data_path,
            data_manifest=manifest,
            covariate_names=covariate_names,
            runner=runner,
            work_dir=frem_work_dir,
            seed=self._config.seed,
            timeout_seconds=self._config.timeout_seconds,
            initial_estimates=init,
            binary_encode_overrides=self._config.binary_encode_overrides,
        )
        return _SR(
            candidate_id=frem_spec.model_id,
            spec=frem_spec,
            result=result,
            converged=result.converged,
            bic=result.bic,
            aic=result.aic,
            n_params=len(result.parameter_estimates),
        )

    async def _run_mi_stage(
        self,
        *,
        directive: MissingDataDirective,
        search_outcome: SearchOutcome,
        data_path: Path,
        manifest: DataManifest,
        covariate_names: list[str],
        run_dir: Path,
        nca_estimates: dict[str, float],
    ) -> ImputationStabilityManifest | None:
        """Drive the MI loop: produce m imputed CSVs, refit each, aggregate.

        Refits the frozen classical candidate set on each of m imputed
        CSVs produced by the configured ``ImputationProvider`` (mice
        PMM or missRanger). ``aggregate_stability`` then applies Rubin's
        rules to the per-imputation (estimate, SE) tuples when the
        backend supplies them, and Gate 1 receives per-candidate
        stability entries to drive the imputation-stability check.

        ``directive.m_imputations`` must be set (the resolver guarantees
        this for MI-* methods). Returns ``None`` and logs a warning when
        no covariate specs can be refit or the provider cannot run.
        """
        from apmode.data.imputers import R_MiceImputer, R_MissRangerImputer
        from apmode.errors import BackendError
        from apmode.search.stability import (
            PerImputationFit,
            run_with_imputations,
        )

        if directive.m_imputations is None:
            logger.warning("MI stage: directive.m_imputations is None; skipping")
            return None

        provider = self._mi_provider
        if provider is None:
            imputer_cls = (
                R_MissRangerImputer
                if directive.covariate_method == "MI-missRanger"
                else R_MiceImputer
            )
            provider = imputer_cls(
                work_dir=run_dir / "mi",
                covariates=covariate_names,
            )

        # Freeze the classical candidate set and refit only those specs per
        # imputation. Running a fresh SearchEngine per imputation would
        # generate different warm-started
        # children per draw; stability metrics (convergence_rate,
        # rank_stability) would then operate on misaligned candidate_ids and
        # become meaningless for the primary search candidates. Restrict to
        # classical nlmixr2 results because NODE specs cannot be fit through
        # Nlmixr2Runner.
        ref_specs = [
            sr.spec
            for sr in search_outcome.results
            if sr.spec is not None and sr.result is not None and sr.result.backend == "nlmixr2"
        ]
        if not ref_specs:
            logger.info("MI stage skipped: no classical candidate specs to refit per imputation")
            return None

        base_init = {k: v for k, v in nca_estimates.items() if not k.startswith("_")}

        async def _fit_one_imputation(
            imputed_csv: Path,
            seed: int,
            imputation_idx: int,
        ) -> list[PerImputationFit]:
            """Refit each classical candidate on a single imputed dataset.

            Calls the main runner directly so candidate_ids align with the
            classical search exactly — aggregation can pool per-candidate
            across imputations without alignment issues.

            ``imputation_idx`` is stamped on each ``PerImputationFit``
            explicitly so there is no implicit overwrite in
            ``run_with_imputations``.
            """
            fits: list[PerImputationFit] = []
            for spec in ref_specs:
                try:
                    result = await self._runner.run(
                        spec=spec,
                        data_manifest=manifest,
                        initial_estimates=base_init,
                        seed=seed,
                        timeout_seconds=self._config.timeout_seconds,
                        data_path=imputed_csv,
                    )
                    fits.append(
                        PerImputationFit(
                            imputation_idx=imputation_idx,
                            candidate_id=spec.model_id,
                            converged=result.converged,
                            ofv=result.ofv,
                            aic=result.aic,
                            bic=result.bic,
                            parameter_estimates={
                                name: (pe.estimate, pe.se)
                                for name, pe in result.parameter_estimates.items()
                                if pe.category == "structural"
                            },
                        )
                    )
                except BackendError as e:
                    logger.warning("MI per-imputation fit failed for %s: %s", spec.model_id, e)
                    fits.append(
                        PerImputationFit(
                            imputation_idx=imputation_idx,
                            candidate_id=spec.model_id,
                            converged=False,
                        )
                    )
            return fits

        return await run_with_imputations(
            directive=directive,
            provider=provider,
            source_csv=data_path,
            search=_fit_one_imputation,
            seed=self._config.seed,
        )

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

        sem = asyncio.Semaphore(max(1, self._config.max_concurrency))

        async def _eval_one(
            cand_result: BackendResult,
        ) -> tuple[str, LOROCVResult | None]:
            cand_runner: BackendRunner = self._runner
            if cand_result.backend == "jax_node" and self._node_runner is not None:
                cand_runner = self._node_runner
            elif cand_result.backend == "agentic_llm" and self._agentic_runner is not None:
                cand_runner = self._agentic_runner

            warm_estimates = {
                name: pe.estimate
                for name, pe in cand_result.parameter_estimates.items()
                if pe.category == "structural"
            }
            if not warm_estimates:
                warm_estimates = {k: v for k, v in nca_estimates.items() if not k.startswith("_")}

            spec = self._spec_map.get(cand_result.model_id)
            if spec is None:
                logger.warning("loro_cv_no_spec", candidate=cand_result.model_id)
                return cand_result.model_id, None

            async with sem:
                try:
                    return cand_result.model_id, await evaluate_loro_cv(
                        candidate_spec=spec,
                        candidate_result=cand_result,
                        folds=folds,
                        runner=cand_runner,
                        data_manifest=manifest,
                        data_path=data_path,
                        initial_estimates=warm_estimates,
                        seed=self._config.seed,
                        timeout_seconds=self._config.timeout_seconds,
                    )
                except BackendError:
                    logger.warning(
                        "loro_cv_candidate_failed",
                        candidate=cand_result.model_id,
                        exc_info=True,
                    )
                    return cand_result.model_id, None

        # Structured concurrency: _eval_one catches its own failures;
        # TaskGroup surfaces any unexpected leaks as an ExceptionGroup.
        async with asyncio.TaskGroup() as tg:
            loro_task_handles = [tg.create_task(_eval_one(c)) for c in candidates]
        gathered = [t.result() for t in loro_task_handles]

        # Post-gather: write artifacts and collect metrics (sequential, no races)
        loro_map: dict[str, LOROMetrics] = {}
        for model_id, loro_result in gathered:
            if loro_result is not None:
                emitter.write_loro_cv_result(loro_result)
                loro_map[model_id] = loro_result.metrics
                logger.info(
                    "loro_cv_complete",
                    candidate=model_id,
                    npde_mean=loro_result.metrics.pooled_npde_mean,
                    vpc_concordance=loro_result.metrics.vpc_coverage_concordance,
                )
        return loro_map

    def _find_existing_run_id(self) -> str | None:
        """Return the run_id of the single existing bundle in ``_bundle_base``.

        Looks for exactly one non-``_`` prefixed subdirectory.  Returns
        ``None`` if zero or more than one are found (ambiguous resume).
        """
        if not self._bundle_base.is_dir():
            return None
        candidates = [
            d.name
            for d in self._bundle_base.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Multiple run dirs in %s — cannot auto-select for resume: %s",
                self._bundle_base,
                candidates,
            )
        return None

    def _write_classical_checkpoint(
        self,
        run_dir: Path,
        search_outcome: SearchOutcome,
        nca_estimates: dict[str, float],
    ) -> None:
        """Persist classical search results so the agentic step can be resumed.

        Written to ``run_dir/classical_checkpoint.json`` immediately after
        Stage 5 completes.  Contains all converged-candidate specs and
        parameter estimates needed to reconstruct a ``SearchOutcome`` for
        ``_run_agentic_stage`` without re-running SAEM.
        """
        results_data = []
        for sr in search_outcome.results:
            if sr.result is None:
                continue
            # Use model_dump(mode="json") so specs/results appear as native
            # dicts inside the checkpoint, making the artifact directly
            # inspectable with jq and avoiding the double-parse on load.
            results_data.append(
                {
                    "candidate_id": sr.candidate_id,
                    "spec": sr.spec.model_dump(mode="json"),
                    "result": sr.result.model_dump(mode="json"),
                    "converged": sr.converged,
                    "bic": sr.bic,
                    "aic": sr.aic,
                    "n_params": sr.n_params,
                }
            )

        best_bic_val = search_outcome.best_bic.bic if search_outcome.best_bic is not None else None
        checkpoint: dict[str, Any] = {
            # Schema bumped to 1.1 because "spec"/"result" replaced the
            # legacy "spec_json"/"result_json" nested-string fields.
            # ``_load_classical_checkpoint`` still understands both.
            "schema_version": "1.1",
            "checkpoint_type": "classical_search",
            "results": results_data,
            "nca_estimates": nca_estimates,
            "best_candidate_id": (
                search_outcome.best_bic.candidate_id
                if search_outcome.best_bic is not None
                else None
            ),
            "best_bic": best_bic_val,
        }
        path = run_dir / "classical_checkpoint.json"
        path.write_text(json.dumps(checkpoint, indent=2))
        logger.info(
            "Classical checkpoint written: %d results, best_bic=%s",
            len(results_data),
            f"{best_bic_val:.1f}" if best_bic_val is not None else "n/a",
        )

    def _load_classical_checkpoint(
        self,
        run_dir: Path,
    ) -> tuple[SearchOutcome, dict[str, float]] | None:
        """Load ``classical_checkpoint.json`` and reconstruct a ``SearchOutcome``.

        Returns ``(SearchOutcome, nca_estimates)`` or ``None`` when no valid
        checkpoint is found.
        """
        from apmode.bundle.models import BackendResult as _BRModel
        from apmode.dsl.ast_models import DSLSpec as _DSLSpec
        from apmode.search.engine import SearchOutcome as _SO
        from apmode.search.engine import SearchResult as _SR

        path = run_dir / "classical_checkpoint.json"
        if not path.exists():
            logger.info("No classical_checkpoint.json in %s", run_dir)
            return None

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read classical checkpoint: %s", exc)
            return None

        if data.get("schema_version") not in {"1.0", "1.1"} or (
            data.get("checkpoint_type") != "classical_search"
        ):
            logger.warning(
                "classical_checkpoint.json has unexpected schema (version=%s type=%s); ignoring",
                data.get("schema_version"),
                data.get("checkpoint_type"),
            )
            return None

        results: list[SearchResult] = []
        for rd in data.get("results", []):
            try:
                # 1.1 stores nested dicts; 1.0 stored nested JSON strings.
                if "spec" in rd and "result" in rd:
                    spec = _DSLSpec.model_validate(rd["spec"])
                    result = _BRModel.model_validate(rd["result"])
                else:
                    spec = _DSLSpec.model_validate_json(rd["spec_json"])
                    result = _BRModel.model_validate_json(rd["result_json"])
            except (ValidationError, json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Skipping malformed checkpoint entry %s: %s",
                    rd.get("candidate_id"),
                    exc,
                )
                continue
            results.append(
                _SR(
                    candidate_id=rd["candidate_id"],
                    spec=spec,
                    result=result,
                    converged=rd.get("converged", False),
                    bic=rd.get("bic"),
                    aic=rd.get("aic"),
                    n_params=rd.get("n_params", 0),
                )
            )

        outcome = _SO(results=results)
        converged = [r for r in results if r.converged and r.bic is not None]
        if converged:
            outcome.best_bic = min(converged, key=lambda r: r.bic or float("inf"))

        nca_estimates: dict[str, float] = {
            k: float(v)
            for k, v in data.get("nca_estimates", {}).items()
            if isinstance(v, (int, float))
        }
        logger.info(
            "Loaded classical checkpoint: %d results (%d converged), best_bic=%s",
            len(results),
            len(converged),
            data.get("best_bic"),
        )
        return outcome, nca_estimates

    def _load_policy(self) -> GatePolicy | None:
        """Load gate policy from file or default."""
        if self._config.policy_path and self._config.policy_path.exists():
            data = json.loads(self._config.policy_path.read_text())
            return GatePolicy.model_validate(data)

        # Fallback to the versioned policy shipped in the repo/package. The
        # lookup is delegated to ``apmode.paths`` so CLI and orchestrator
        # cannot drift on parent-count heuristics.
        from apmode.paths import policy_path_for_lane

        default_path = policy_path_for_lane(self._config.lane)
        if default_path is not None:
            data = json.loads(default_path.read_text())
            return GatePolicy.model_validate(data)

        return None
