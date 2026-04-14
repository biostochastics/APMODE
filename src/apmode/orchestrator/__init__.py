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
    SeedRegistry,
)
from apmode.data.initial_estimates import NCAEstimator, build_initial_estimates_bundle
from apmode.data.profiler import profile_data
from apmode.data.splitter import split_subjects
from apmode.governance.gates import evaluate_gate1, evaluate_gate2, evaluate_gate3
from apmode.governance.policy import GatePolicy
from apmode.routing import route

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.bundle.models import DataManifest, EvidenceManifest
    from apmode.routing import DispatchDecision
    from apmode.search.engine import SearchOutcome

logger = structlog.get_logger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single APMODE run."""

    lane: Literal["submission", "discovery", "optimization"] = "submission"
    seed: int = 42
    timeout_seconds: int = 600
    policy_path: Path | None = None
    covariate_names: list[str] = field(default_factory=list)


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
    ) -> None:
        self._runner = runner
        self._bundle_base = bundle_base_dir
        self._config = config or RunConfig()

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
                apmode_version="0.1.0-dev",
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
        search_engine = SearchEngine(
            runner=self._runner,
            data_manifest=manifest,
            data_path=data_path,
            seed=self._config.seed,
            timeout_seconds=self._config.timeout_seconds,
            allowed_backends=dispatch.backends,
        )
        search_outcome = await search_engine.run(
            evidence_manifest=evidence,
            initial_estimates=nca_estimates,
            emitter=emitter,
            covariate_names=self._config.covariate_names,
        )
        outcome.search_outcome = search_outcome

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
                    seed_runs: list[BRModel] = []
                    for seed_offset in range(1, seed_n):
                        try:
                            seed_result = await self._runner.run(
                                spec=sr.spec,
                                data_manifest=manifest,
                                initial_estimates=cand_estimates,
                                seed=self._config.seed + seed_offset,
                                timeout_seconds=self._config.timeout_seconds,
                                data_path=data_path,
                            )
                            seed_runs.append(seed_result)
                            emitter.write_seed_result(seed_result, cid, seed_offset)
                        except Exception:
                            logger.warning("Seed run %d failed for %s", seed_offset, cid)
                    if seed_runs:
                        seed_results_map[cid] = seed_runs

            # Stage 6b: Gate 1 + Gate 2 evaluation
            for sr in search_outcome.results:
                if sr.result is None:
                    continue

                emitter.write_backend_result(sr.result)

                # Gate 1: Technical Validity (with seed results if available)
                seed_results = seed_results_map.get(sr.candidate_id)
                g1 = evaluate_gate1(sr.result, policy, seed_results=seed_results)
                emitter.write_gate_decision(g1, gate_number=1)
                outcome.gate1_results.append((sr.candidate_id, g1.passed))

                if not g1.passed:
                    emitter.append_failed_candidate(
                        FailedCandidate(
                            candidate_id=sr.candidate_id,
                            backend="nlmixr2",
                            gate_failed="gate1",
                            failed_checks=[c.check_id for c in g1.checks if not c.passed],
                            summary_reason=g1.summary_reason,
                            timestamp=datetime.now(tz=UTC).isoformat(),
                        )
                    )
                    continue

                # Gate 2: Lane-Specific Admissibility
                g2 = evaluate_gate2(sr.result, policy, self._config.lane)
                emitter.write_gate_decision(g2, gate_number=2)
                outcome.gate2_results.append((sr.candidate_id, g2.passed))

                if not g2.passed:
                    emitter.append_failed_candidate(
                        FailedCandidate(
                            candidate_id=sr.candidate_id,
                            backend="nlmixr2",
                            gate_failed="gate2",
                            failed_checks=[c.check_id for c in g2.checks if not c.passed],
                            summary_reason=g2.summary_reason,
                            timestamp=datetime.now(tz=UTC).isoformat(),
                        )
                    )
                    continue

                # Passed both gates
                outcome.recommended.append(sr.candidate_id)
                gate12_survivors.append(sr.result)

            # Stage 6c: Gate 3 — Within-paradigm ranking
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

        # --- Stage 7: Report Provenance ---
        from apmode.bundle.models import ReportProvenance

        emitter.write_report_provenance(
            ReportProvenance(
                generated_at=datetime.now(tz=UTC).isoformat(),
                apmode_version="0.1.0-dev",
                generator="apmode.orchestrator",
                component_versions={
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "pipeline": "Phase1",
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
