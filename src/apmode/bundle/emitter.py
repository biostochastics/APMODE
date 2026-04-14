# SPDX-License-Identifier: GPL-2.0-or-later
"""Bundle emitter (ARCHITECTURE.md §5, PRD §4.3.2).

Writes all reproducibility bundle artifacts to disk. Each JSON/JSONL artifact
is Pydantic-validated before writing (enforced by typed method signatures).

Artifacts:
  data_manifest.json, seed_registry.json, backend_versions.json, policy_file.json,
  evidence_manifest.json, initial_estimates.json, split_manifest.json,
  search_trajectory.jsonl, failed_candidates.jsonl, candidate_lineage.json,
  compiled_specs/{id}.json + .R, results/{id}_result.json,
  gate_decisions/gate{n}_{id}.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path  # noqa: TC003 — used at runtime in __init__

from apmode.bundle.models import (  # noqa: TC001 — used at runtime in method signatures
    AgenticTraceInput,
    AgenticTraceMeta,
    AgenticTraceOutput,
    BackendResult,
    BackendVersions,
    CandidateLineage,
    CredibilityReport,
    DataManifest,
    EvidenceManifest,
    FailedCandidate,
    GateResult,
    InitialEstimates,
    LOROCVResult,
    PosteriorDiagnostics,
    PriorManifest,
    Ranking,
    ReportProvenance,
    RunLineage,
    SearchGraph,
    SearchTrajectoryEntry,
    SeedRegistry,
    SimulationProtocol,
    SplitManifest,
)
from apmode.dsl.ast_models import DSLSpec  # noqa: TC001 — used at runtime
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.ids import generate_run_id

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_path_component(value: str, label: str) -> None:
    """Reject IDs containing path traversal characters."""
    if not _SAFE_ID_RE.match(value):
        msg = f"{label} contains unsafe characters: {value!r}"
        raise ValueError(msg)


class BundleEmitter:
    """Writes reproducibility bundle artifacts to disk.

    All artifacts per ARCHITECTURE.md §5 / PRD §4.3.2:
    - data_manifest.json, seed_registry.json, backend_versions.json
    - evidence_manifest.json, initial_estimates.json, split_manifest.json
    - policy_file.json
    - compiled_specs/{candidate_id}.json + .R
    - results/{candidate_id}_result.json
    - gate_decisions/gate{n}_{candidate_id}.json
    - search_trajectory.jsonl, failed_candidates.jsonl, candidate_lineage.json
    """

    def __init__(self, base_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or generate_run_id()
        self.run_dir = base_dir / self.run_id
        self._compiled_specs_dir = self.run_dir / "compiled_specs"
        self._gate_decisions_dir = self.run_dir / "gate_decisions"
        self._results_dir = self.run_dir / "results"

    def initialize(self) -> Path:
        """Create the bundle directory structure. Returns the run directory."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._compiled_specs_dir.mkdir(exist_ok=True)
        self._gate_decisions_dir.mkdir(exist_ok=True)
        self._results_dir.mkdir(exist_ok=True)
        return self.run_dir

    # --- Core manifests ---

    def write_data_manifest(self, manifest: DataManifest) -> Path:
        """Write data_manifest.json."""
        path = self.run_dir / "data_manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def write_seed_registry(self, registry: SeedRegistry) -> Path:
        """Write seed_registry.json."""
        path = self.run_dir / "seed_registry.json"
        path.write_text(registry.model_dump_json(indent=2))
        return path

    def write_backend_versions(self, versions: BackendVersions) -> Path:
        """Write backend_versions.json."""
        path = self.run_dir / "backend_versions.json"
        path.write_text(versions.model_dump_json(indent=2))
        return path

    def write_evidence_manifest(self, manifest: EvidenceManifest) -> Path:
        """Write evidence_manifest.json (Data Profiler output)."""
        path = self.run_dir / "evidence_manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def write_initial_estimates(self, estimates: InitialEstimates) -> Path:
        """Write initial_estimates.json (keyed by candidate_id)."""
        path = self.run_dir / "initial_estimates.json"
        path.write_text(estimates.model_dump_json(indent=2))
        return path

    def write_split_manifest(self, split: SplitManifest) -> Path:
        """Write split_manifest.json."""
        path = self.run_dir / "split_manifest.json"
        path.write_text(split.model_dump_json(indent=2))
        return path

    def write_policy_file(self, policy_data: dict[str, object]) -> Path:
        """Write policy_file.json (copy of the gate thresholds used for this run)."""
        path = self.run_dir / "policy_file.json"
        path.write_text(json.dumps(policy_data, indent=2, default=str))
        return path

    # --- Compiled specs ---

    def write_compiled_spec(
        self,
        spec: DSLSpec,
        initial_estimates: dict[str, float] | None = None,
    ) -> tuple[Path, Path | None]:
        """Write compiled_specs/{candidate_id}.json and .R.

        NODE specs are written as JSON only (no R lowering in Phase 1).

        Returns (json_path, r_path). r_path is None for NODE specs.
        """
        _validate_path_component(spec.model_id, "model_id")
        json_path = self._compiled_specs_dir / f"{spec.model_id}.json"
        json_path.write_text(spec.model_dump_json(indent=2))

        if spec.has_node_modules():
            return json_path, None

        r_path = self._compiled_specs_dir / f"{spec.model_id}.R"
        r_code = emit_nlmixr2(spec, initial_estimates=initial_estimates)
        r_path.write_text(r_code)

        return json_path, r_path

    # --- Results ---

    def write_backend_result(self, result: BackendResult) -> Path:
        """Write results/{candidate_id}_result.json."""
        _validate_path_component(result.model_id, "model_id")
        path = self._results_dir / f"{result.model_id}_result.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    def write_seed_result(self, result: BackendResult, candidate_id: str, seed_index: int) -> Path:
        """Write results/{candidate_id}_seed_{n}_result.json."""
        _validate_path_component(candidate_id, "candidate_id")
        path = self._results_dir / f"{candidate_id}_seed_{seed_index}_result.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    # --- Gate decisions ---

    def write_gate_decision(self, gate_result: GateResult, gate_number: int) -> Path:
        """Write gate_decisions/gate{n}_{candidate_id}.json."""
        _validate_path_component(gate_result.candidate_id, "candidate_id")
        filename = f"gate{gate_number}_{gate_result.candidate_id}.json"
        path = self._gate_decisions_dir / filename
        path.write_text(gate_result.model_dump_json(indent=2))
        return path

    # --- Search artifacts ---

    def append_search_trajectory(self, entry: SearchTrajectoryEntry) -> Path:
        """Append one line to search_trajectory.jsonl."""
        path = self.run_dir / "search_trajectory.jsonl"
        with path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")
        return path

    def append_failed_candidate(self, entry: FailedCandidate) -> Path:
        """Append one line to failed_candidates.jsonl."""
        path = self.run_dir / "failed_candidates.jsonl"
        with path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")
        return path

    def write_candidate_lineage(self, lineage: CandidateLineage) -> Path:
        """Write candidate_lineage.json (DAG of candidate parentage)."""
        path = self.run_dir / "candidate_lineage.json"
        path.write_text(lineage.model_dump_json(indent=2))
        return path

    def write_search_graph(self, graph: SearchGraph) -> Path:
        """Write search_graph.json (enriched DAG for deep inspection)."""
        path = self.run_dir / "search_graph.json"
        path.write_text(graph.model_dump_json(indent=2))
        return path

    def write_ranking(self, ranking: Ranking) -> Path:
        """Write ranking.json (full ordered candidate list from Gate 3)."""
        path = self.run_dir / "ranking.json"
        path.write_text(ranking.model_dump_json(indent=2))
        return path

    def write_report_provenance(self, provenance: ReportProvenance) -> Path:
        """Write report_provenance.json (who/what generated each section)."""
        path = self.run_dir / "report_provenance.json"
        path.write_text(provenance.model_dump_json(indent=2))
        return path

    # --- Credibility reports (per-candidate, Phase 2+) ---

    def _credibility_dir(self) -> Path:
        """Ensure and return the credibility/ subdirectory."""
        cred_dir = self.run_dir / "credibility"
        cred_dir.mkdir(exist_ok=True)
        return cred_dir

    def write_credibility_report(self, report: CredibilityReport) -> Path:
        """Write credibility/{candidate_id}.json."""
        _validate_path_component(report.candidate_id, "candidate_id")
        path = self._credibility_dir() / f"{report.candidate_id}.json"
        path.write_text(report.model_dump_json(indent=2))
        return path

    # --- LORO-CV results (Phase 3, Optimization lane) ---

    def _loro_cv_dir(self) -> Path:
        """Ensure and return the loro_cv/ subdirectory."""
        loro_dir = self.run_dir / "loro_cv"
        loro_dir.mkdir(exist_ok=True)
        return loro_dir

    def write_loro_cv_result(self, result: LOROCVResult) -> Path:
        """Write loro_cv/{candidate_id}.json."""
        _validate_path_component(result.candidate_id, "candidate_id")
        path = self._loro_cv_dir() / f"{result.candidate_id}.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    # --- Agentic trace (Phase 3, PRD §4.2.6) ---

    def _agentic_trace_dir(self) -> Path:
        """Ensure and return the agentic_trace/ subdirectory."""
        trace_dir = self.run_dir / "agentic_trace"
        trace_dir.mkdir(exist_ok=True)
        return trace_dir

    def write_agentic_trace_input(self, inp: AgenticTraceInput) -> Path:
        """Write agentic_trace/{iteration_id}_input.json."""
        _validate_path_component(inp.iteration_id, "iteration_id")
        path = self._agentic_trace_dir() / f"{inp.iteration_id}_input.json"
        path.write_text(inp.model_dump_json(indent=2))
        return path

    def write_agentic_trace_output(self, out: AgenticTraceOutput) -> Path:
        """Write agentic_trace/{iteration_id}_output.json."""
        _validate_path_component(out.iteration_id, "iteration_id")
        path = self._agentic_trace_dir() / f"{out.iteration_id}_output.json"
        path.write_text(out.model_dump_json(indent=2))
        return path

    def write_agentic_trace_meta(self, meta: AgenticTraceMeta) -> Path:
        """Write agentic_trace/{iteration_id}_meta.json."""
        _validate_path_component(meta.iteration_id, "iteration_id")
        path = self._agentic_trace_dir() / f"{meta.iteration_id}_meta.json"
        path.write_text(meta.model_dump_json(indent=2))
        return path

    def write_run_lineage(self, lineage: RunLineage) -> Path:
        """Write run_lineage.json (multi-run provenance, PRD §4.2.6)."""
        path = self.run_dir / "run_lineage.json"
        path.write_text(lineage.model_dump_json(indent=2))
        return path

    # --- Bayesian artifacts (Phase 2+, plan 2026-04-14 §3.5) ---

    def _bayesian_dir(self) -> Path:
        """Ensure and return the bayesian/ subdirectory."""
        bayes_dir = self.run_dir / "bayesian"
        bayes_dir.mkdir(exist_ok=True)
        return bayes_dir

    def write_prior_manifest(self, manifest: PriorManifest, candidate_id: str) -> Path:
        """Write bayesian/{candidate_id}_prior_manifest.json.

        FDA 2026 draft guidance requires the prior justification artifact to be
        persisted with the run (kimi review Issue 2).
        """
        _validate_path_component(candidate_id, "candidate_id")
        path = self._bayesian_dir() / f"{candidate_id}_prior_manifest.json"
        path.write_text(manifest.model_dump_json(indent=2))
        return path

    def write_simulation_protocol(self, protocol: SimulationProtocol, candidate_id: str) -> Path:
        """Write bayesian/{candidate_id}_simulation_protocol.json.

        Prospective simulation protocol — locked before Gate 3 to prevent
        metric shopping (per gpt-5.2 review and FDA 2026 operating-characteristics
        requirement).
        """
        _validate_path_component(candidate_id, "candidate_id")
        path = self._bayesian_dir() / f"{candidate_id}_simulation_protocol.json"
        path.write_text(protocol.model_dump_json(indent=2))
        return path

    def write_mcmc_diagnostics(self, diagnostics: PosteriorDiagnostics, candidate_id: str) -> Path:
        """Write bayesian/{candidate_id}_mcmc_diagnostics.json (kimi review Issue 4).

        Deep-inspection tools expect diagnostics as a standalone artifact in
        addition to the nested ``BackendResult.posterior_diagnostics`` field.
        """
        _validate_path_component(candidate_id, "candidate_id")
        path = self._bayesian_dir() / f"{candidate_id}_mcmc_diagnostics.json"
        path.write_text(diagnostics.model_dump_json(indent=2))
        return path

    def copy_posterior_draws(self, source: Path, candidate_id: str) -> Path:
        """Copy posterior_draws.parquet from the harness work_dir into the bundle.

        Returns the bundle-relative path that should be stored on
        ``BackendResult.posterior_draws_path`` so the reference is durable.
        (kimi review Issue 2.)
        """
        import shutil

        _validate_path_component(candidate_id, "candidate_id")
        dest = self._bayesian_dir() / f"{candidate_id}_draws.parquet"
        shutil.copy2(source, dest)
        return dest
