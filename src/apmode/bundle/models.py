# SPDX-License-Identifier: GPL-2.0-or-later
"""Pydantic models for reproducibility bundle artifacts (ARCHITECTURE.md §5, PRD §4.3.2).

Every JSON/JSONL artifact in the bundle has a corresponding Pydantic model here.
Models are validated before writing to disk.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# --- Sub-models for BackendResult (ARCHITECTURE.md §4.1) ---

HexSHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]


class ParameterEstimate(BaseModel):
    """A single parameter estimate with uncertainty.

    Fields are a superset of what any single backend emits:
    - MLE backends populate estimate/se/rse/ci95_{lower,upper}.
    - Bayesian backend populates estimate (posterior mean), posterior_sd,
      q05/q50/q95, and leaves se/rse/ci95 as None. ci95 from Bayesian is
      the 95% credible interval; MLE-side ci95 is the Wald/profile CI.
    """

    name: str
    estimate: float
    se: float | None = None
    rse: float | None = None
    ci95_lower: float | None = None
    ci95_upper: float | None = None
    posterior_sd: float | None = None
    q05: float | None = None
    q50: float | None = None
    q95: float | None = None
    fixed: bool = False
    category: Literal["structural", "iiv", "iov", "residual"]


class ConvergenceMetadata(BaseModel):
    """Convergence details from an estimation run."""

    method: str
    converged: bool
    iterations: int = Field(ge=0)
    gradient_norm: float | None = None
    minimization_status: Literal[
        "successful", "terminated", "boundary", "rounding_errors", "max_evaluations"
    ]
    wall_time_seconds: float = Field(ge=0.0)


class GOFMetrics(BaseModel):
    """Goodness-of-fit metrics."""

    cwres_mean: float
    cwres_sd: float
    outlier_fraction: float = Field(ge=0.0, le=1.0)
    obs_vs_pred_r2: float | None = None


class VPCSummary(BaseModel):
    """Visual Predictive Check summary statistics."""

    percentiles: list[float] = Field(default_factory=lambda: [5.0, 50.0, 95.0])
    coverage: dict[str, float]
    n_bins: int = Field(gt=0)
    prediction_corrected: bool

    @model_validator(mode="after")
    def coverage_keys_match_percentiles(self) -> VPCSummary:
        expected = {f"p{int(p)}" for p in self.percentiles}
        actual = set(self.coverage.keys())
        if expected != actual:
            msg = f"coverage keys {actual} must match percentiles {expected}"
            raise ValueError(msg)
        return self


class IdentifiabilityFlags(BaseModel):
    """Practical identifiability assessment."""

    condition_number: float | None = None
    profile_likelihood_ci: dict[str, bool]
    ill_conditioned: bool


class BLQHandling(BaseModel):
    """Below-limit-of-quantification handling method."""

    method: Literal["none", "m3", "m4"]
    lloq: float | None = None
    n_blq: int = Field(ge=0)
    blq_fraction: float = Field(ge=0.0, le=1.0)


class SplitGOFMetrics(BaseModel):
    """Per-fold GOF from split-aware backend evaluation.

    Computed by partitioning per-subject CWRES by train/test assignment
    from the SplitManifest — no second backend run needed.
    """

    train_cwres_mean: float
    train_outlier_fraction: float = Field(ge=0.0, le=1.0)
    test_cwres_mean: float
    test_outlier_fraction: float = Field(ge=0.0, le=1.0)
    n_train: int = Field(gt=0)
    n_test: int = Field(gt=0)


class DiagnosticBundle(BaseModel):
    """All diagnostic outputs for a candidate model."""

    gof: GOFMetrics
    split_gof: SplitGOFMetrics | None = None
    vpc: VPCSummary | None = None
    identifiability: IdentifiabilityFlags
    blq: BLQHandling
    diagnostic_plots: dict[str, str] = Field(default_factory=dict)


# --- BackendResult ---


class PosteriorDiagnostics(BaseModel):
    """MCMC convergence and reliability diagnostics (plan 2026-04-14 §3.4).

    Populated by the Bayesian backend; None otherwise.

    Thresholds for disqualification (policy-configurable, defaults per
    Vehtari et al. 2021, rank-normalized R̂ < 1.01):
      - rhat_max            ≤ 1.01
      - ess_bulk_min        ≥ 400
      - ess_tail_min        ≥ 400
      - n_divergent         == 0
      - n_max_treedepth     ≤ 0.01 * total_samples
      - ebfmi_min           ≥ 0.3
      - pareto_k_max        ≤ 0.7 (LOO reliability)
    """

    model_config = ConfigDict(frozen=True)

    rhat_max: float = Field(ge=0.0)
    ess_bulk_min: float = Field(ge=0.0)
    ess_tail_min: float = Field(ge=0.0)
    n_divergent: int = Field(ge=0)
    n_max_treedepth: int = Field(ge=0)
    ebfmi_min: float
    pareto_k_max: float | None = None
    pareto_k_counts: dict[str, int] = Field(default_factory=dict)  # "good/ok/bad/very_bad"
    mcse_by_param: dict[str, float] = Field(default_factory=dict)  # headline params only
    per_chain_rhat: dict[str, list[float]] = Field(default_factory=dict)


class SamplerConfig(BaseModel):
    """NUTS sampler configuration and environment — captured for reproducibility.

    Fields like cmdstan_version/torsten_version/stan_version/compiler_id are
    filled in by the harness post-compile. This record is persisted in the
    bundle (backend_versions.json) and included inside BackendResult.
    """

    model_config = ConfigDict(frozen=True)

    chains: int = Field(default=4, ge=1)
    warmup: int = Field(default=1000, ge=100)
    sampling: int = Field(default=1000, ge=100)
    adapt_delta: float = Field(default=0.95, gt=0.0, lt=1.0)
    max_treedepth: int = Field(default=12, ge=4, le=20)
    parallel_chains: int | None = None
    threads_per_chain: int | None = None
    seed: int = 0
    cmdstan_version: str = ""
    torsten_version: str = ""
    stan_version: str = ""
    compiler_id: str = ""


class BackendResult(BaseModel):
    """Standardized result from any backend (ARCHITECTURE.md §4.1)."""

    model_id: str
    backend: Literal["nlmixr2", "jax_node", "agentic_llm", "bayesian_stan"]
    converged: bool
    ofv: float | None = None
    aic: float | None = None
    bic: float | None = None
    parameter_estimates: dict[str, ParameterEstimate]
    eta_shrinkage: dict[str, float]
    convergence_metadata: ConvergenceMetadata
    diagnostics: DiagnosticBundle
    wall_time_seconds: float = Field(ge=0.0)
    backend_versions: dict[str, str]
    initial_estimate_source: Literal["nca", "warm_start", "fallback"]
    # Bayesian-only: populated by BayesianRunner, None otherwise.
    posterior_diagnostics: PosteriorDiagnostics | None = None
    sampler_config: SamplerConfig | None = None
    posterior_draws_path: str | None = None  # relative to bundle root


# --- Data Manifest ---


class ColumnMapping(BaseModel):
    """Maps source columns to canonical schema fields.

    Each field name is a semantic key (snake_case), and each value is
    the canonical NONMEM column name (UPPERCASE).  Use ``to_canonical()``
    and ``to_semantic()`` for bidirectional lookup.
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str
    time: str
    dv: str
    evid: str
    amt: str
    mdv: str | None = None
    cmt: str | None = None
    rate: str | None = None
    dur: str | None = None
    addl: str | None = None
    ii: str | None = None
    ss: str | None = None
    blq_flag: str | None = None
    lloq: str | None = None
    occasion: str | None = None
    study_id: str | None = None

    def to_canonical(self) -> dict[str, str]:
        """Return semantic -> canonical mapping (only non-None entries)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def to_semantic(self) -> dict[str, str]:
        """Return canonical -> semantic mapping (only non-None entries)."""
        return {v: k for k, v in self.model_dump().items() if v is not None}


class CovariateMetadata(BaseModel):
    """Metadata about a covariate column."""

    name: str
    type: Literal["continuous", "categorical"]
    time_varying: bool = False


class DataManifest(BaseModel):
    """data_manifest.json — SHA-256, column mapping, BLQ coding, ingestion format."""

    model_config = ConfigDict(frozen=True)

    data_sha256: HexSHA256
    ingestion_format: Literal["nonmem_csv", "nlmixr2_event_table", "cdisc_adam"]
    column_mapping: ColumnMapping
    n_subjects: int = Field(gt=0)
    n_observations: int = Field(gt=0)
    n_doses: int = Field(gt=0)
    has_multidose: bool = False
    has_steady_state: bool = False
    blq_coding: str | None = None
    covariates: list[CovariateMetadata] = Field(default_factory=list)


# --- Split Manifest ---


class SubjectAssignment(BaseModel):
    """Per-subject fold assignment."""

    subject_id: str
    fold: Literal["train", "test", "validation"]


class SplitManifest(BaseModel):
    """split_manifest.json — subject-level assignments and fold seed."""

    model_config = ConfigDict(frozen=True)

    split_seed: int
    split_strategy: Literal["subject_level", "time_based", "stratified", "regimen_level"]
    assignments: list[SubjectAssignment]


# --- Seed Registry ---


class SeedRegistry(BaseModel):
    """seed_registry.json — all random seeds."""

    model_config = ConfigDict(frozen=True)

    root_seed: int
    r_seed: int
    r_rng_kind: Literal["L'Ecuyer-CMRG", "Mersenne-Twister"]
    np_seed: int
    jax_key: int | None = None
    backend_seeds: dict[str, int] = Field(default_factory=dict)


# --- Evidence Manifest ---


class CovariateSpec(BaseModel):
    """Covariate missingness specification."""

    pattern: Literal["MCAR", "MAR", "informative-suspected"]
    fraction_incomplete: float = Field(ge=0.0, le=1.0)
    strategy: Literal["impute-median", "impute-LOCF", "full-information", "exclude"]


class EvidenceManifest(BaseModel):
    """evidence_manifest.json — Data Profiler output (PRD §4.2.1)."""

    model_config = ConfigDict(frozen=True)

    data_sha256: HexSHA256 | None = None
    route_certainty: Literal["confirmed", "inferred", "ambiguous"]
    absorption_complexity: Literal["simple", "multi-phase", "lag-signature", "unknown"]
    nonlinear_clearance_signature: bool
    nonlinear_clearance_confidence: float | None = None
    richness_category: Literal["sparse", "moderate", "rich"]
    identifiability_ceiling: Literal["low", "medium", "high"]
    covariate_burden: int = Field(ge=0)
    covariate_correlated: bool
    covariate_missingness: CovariateSpec | None = None
    blq_burden: float = Field(ge=0.0, le=1.0)
    protocol_heterogeneity: Literal["single-study", "pooled-similar", "pooled-heterogeneous"]
    absorption_phase_coverage: Literal["adequate", "inadequate"]
    elimination_phase_coverage: Literal["adequate", "inadequate"]


# --- Initial Estimates ---


class InitialEstimateEntry(BaseModel):
    """Per-candidate initial estimates with provenance."""

    candidate_id: str
    source: Literal["nca", "warm_start", "fallback"]
    estimates: dict[str, float]
    inputs_used: list[str] = Field(default_factory=list)


class InitialEstimates(BaseModel):
    """initial_estimates.json — keyed by candidate_id."""

    entries: dict[str, InitialEstimateEntry]


# --- Search Trajectory ---


class SearchTrajectoryEntry(BaseModel):
    """One line in search_trajectory.jsonl."""

    candidate_id: str
    parent_id: str | None = None
    backend: str
    converged: bool
    ofv: float | None = None
    aic: float | None = None
    bic: float | None = None
    gate1_passed: bool | None = None
    gate2_passed: bool | None = None
    wall_time_seconds: float | None = None
    timestamp: str


# --- Failed Candidates ---


class FailedCandidate(BaseModel):
    """One line in failed_candidates.jsonl."""

    candidate_id: str
    backend: str
    gate_failed: str
    failed_checks: list[str]
    summary_reason: str
    timestamp: str


# --- Candidate Lineage ---


class CandidateLineageEntry(BaseModel):
    """One node in the candidate derivation DAG."""

    candidate_id: str
    parent_id: str | None = None
    transform: str | None = None


class CandidateLineage(BaseModel):
    """candidate_lineage.json — DAG of candidate parentage."""

    entries: list[CandidateLineageEntry]


# --- Search Graph (Deep Inspection) ---


class SearchGraphNode(BaseModel):
    """A node in the enriched search graph (candidate + gate status)."""

    candidate_id: str
    parent_id: str | None = None
    backend: Literal["nlmixr2", "jax_node", "agentic_llm"] = "nlmixr2"
    converged: bool = False
    bic: float | None = None
    aic: float | None = None
    n_params: int = 0
    gate1_passed: bool | None = None
    gate2_passed: bool | None = None
    gate2_5_passed: bool | None = None
    rank: int | None = None


class SearchGraphEdge(BaseModel):
    """An edge in the search graph (parent -> child via transform)."""

    parent_id: str
    child_id: str
    transform: str


class SearchGraph(BaseModel):
    """search_graph.json — enriched DAG of the full search space."""

    nodes: list[SearchGraphNode]
    edges: list[SearchGraphEdge] = Field(default_factory=list)


# --- Agentic Iteration Entry (Deep Inspection) ---


class AgenticIterationEntry(BaseModel):
    """One line in agentic_iterations.jsonl — typed audit trail."""

    iteration: int = Field(ge=1)
    spec_before: str
    spec_after: str | None = None
    transforms_proposed: list[str] = Field(default_factory=list)
    transforms_rejected: list[str] = Field(default_factory=list)
    reasoning: str = ""
    converged: bool = False
    bic: float | None = None
    error: str | None = None
    validation_feedback: list[str] = Field(default_factory=list)


# --- Backend Versions ---


class BackendVersions(BaseModel):
    """backend_versions.json — software versions and container digests."""

    model_config = ConfigDict(frozen=True)

    apmode_version: str
    python_version: str
    r_version: str | None = None
    nlmixr2_version: str | None = None
    jax_version: str | None = None
    container_image_digest: str | None = None
    git_sha: str | None = None


# --- Policy File (bundle copy) ---


class PolicyFile(BaseModel):
    """policy_file.json — versioned gate thresholds (PRD §4.3.1).

    This is the bundle-copy model. For typed validation, use
    apmode.governance.policy.GatePolicy.
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str
    lane: Literal["submission", "discovery", "optimization"]
    gate1_thresholds: dict[str, float | int | bool]
    gate2_thresholds: dict[str, float | int | bool]
    gate2_5_thresholds: dict[str, float | int | bool] | None = None
    gate3_config: dict[str, float | int | bool | str] | None = None


# --- Gate Results (ARCHITECTURE.md §4.3) ---


class GateCheckResult(BaseModel):
    """Result of a single gate check."""

    check_id: str
    passed: bool
    observed: float | bool | str
    threshold: float | str | None = None
    units: str | None = None
    evidence_ref: str | None = None


class GateResult(BaseModel):
    """Full result of a gate evaluation."""

    gate_id: str
    gate_name: str
    candidate_id: str
    passed: bool
    checks: list[GateCheckResult]
    summary_reason: str
    policy_version: str
    timestamp: str


# --- Report Provenance ---


class RankedCandidateEntry(BaseModel):
    """One entry in the Gate 3 ranking."""

    candidate_id: str
    rank: int = Field(ge=1)
    bic: float
    aic: float | None = None
    n_params: int = Field(ge=0)
    backend: str


class Ranking(BaseModel):
    """ranking.json — full ordered candidate list from Gate 3."""

    ranked_candidates: list[RankedCandidateEntry]
    best_candidate_id: str | None = None
    ranking_metric: str = "bic"
    n_survivors: int = Field(ge=0)

    @model_validator(mode="after")
    def n_survivors_matches_list(self) -> Ranking:
        if self.n_survivors != len(self.ranked_candidates):
            msg = (
                f"n_survivors ({self.n_survivors}) must equal "
                f"len(ranked_candidates) ({len(self.ranked_candidates)})"
            )
            raise ValueError(msg)
        return self


# --- Gate 2.5 Credibility Context (Phase 2) ---


class CredibilityContext(BaseModel):
    """Input context for Gate 2.5 credibility checks.

    Carries the information needed to evaluate ICH M15 credibility
    qualification: context-of-use, data adequacy metrics, sensitivity
    analysis availability, and AI/ML transparency.
    """

    context_of_use: str | None = None
    risk_level: Literal["low", "medium", "high"] | None = None
    n_observations: int = Field(ge=0, default=0)
    n_parameters: int = Field(ge=0, default=0)
    sensitivity_available: bool = False
    limitations: list[str] = Field(default_factory=list)
    ml_transparency_statement: str | None = None


# --- Phase 2+ Prep Models ---


class CredibilityReport(BaseModel):
    """Per-candidate credibility assessment (ARCHITECTURE.md §4.4, PRD §4.3.3).

    Phase 2: each recommended model's report includes context-of-use,
    credibility evidence, data adequacy, limitations, and ML transparency.
    """

    candidate_id: str
    context_of_use: str
    model_credibility: dict[str, str | float | bool]
    data_adequacy: str
    limitations: list[str] = Field(default_factory=list)
    ml_transparency: str | None = None
    sensitivity_results: dict[str, str | float] = Field(default_factory=dict)
    evidence_refs: dict[str, str] = Field(default_factory=dict)


class AgenticTraceInput(BaseModel):
    """agentic_trace/{iteration_id}_input.json — redacted LLM input (Phase 3)."""

    iteration_id: str
    run_id: str
    candidate_id: str
    prompt_hash: str
    prompt_template: str
    dsl_spec_json: str
    diagnostics_summary: dict[str, str | float] = Field(default_factory=dict)


class AgenticTraceOutput(BaseModel):
    """agentic_trace/{iteration_id}_output.json — verbatim LLM output (Phase 3)."""

    iteration_id: str
    raw_output: str
    parsed_transforms: list[str] = Field(default_factory=list)
    transforms_rejected: list[str] = Field(default_factory=list)
    validation_passed: bool = False
    validation_errors: list[str] = Field(default_factory=list)


class AgenticTraceMeta(BaseModel):
    """agentic_trace/{iteration_id}_meta.json — model/cost metadata (Phase 3)."""

    iteration_id: str
    model_id: str
    model_version: str
    prompt_hash: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    temperature: float = 0.0
    wall_time_seconds: float = Field(ge=0.0)
    request_payload_hash: str = ""
    agentic_reproducibility: Literal["full", "best-effort"] = "full"


class RunLineage(BaseModel):
    """run_lineage.json — links to prior run IDs for multi-run provenance (Phase 3)."""

    current_run_id: str
    parent_run_ids: list[str] = Field(default_factory=list)
    lineage_type: Literal["independent", "continuation", "refinement"] = "independent"
    notes: str | None = None


# --- LORO-CV Results (Phase 3 — Optimization lane) ---


class LOROFoldResult(BaseModel):
    """Per-fold result from leave-one-regimen-out cross-validation.

    Each fold holds out one regimen group and trains on the rest.
    CWRES on the test fold is used as a proxy for NPDE (true NPDE
    requires Monte Carlo simulation of the predictive distribution).
    """

    fold_index: int = Field(ge=0)
    regimen_group: str
    n_train_subjects: int = Field(gt=0)
    n_test_subjects: int = Field(gt=0)
    train_ofv: float | None = None
    test_npde_mean: float | None = None
    test_npde_variance: float | None = None
    test_bic: float | None = None
    converged: bool = False
    fold_vpc_min_coverage: float | None = None


class LOROMetrics(BaseModel):
    """Aggregated LORO-CV predictive performance metrics.

    Gate 2 in the Optimization lane evaluates these against policy thresholds.
    Pooled variance uses the law of total variance (E[Var] + Var[E]) per
    review.
    """

    n_folds: int = Field(ge=1)
    n_total_test_subjects: int = Field(ge=1)
    pooled_npde_mean: float
    pooled_npde_variance: float
    vpc_coverage_concordance: float = Field(ge=0.0, le=1.0)
    auc_gmr: float | None = None
    cmax_gmr: float | None = None
    bioequivalence_pass: bool | None = None
    per_fold_bic: list[float] = Field(default_factory=list)
    worst_fold_npde_mean: float | None = None
    worst_fold_npde_variance: float | None = None
    overall_pass: bool = False
    evaluation_mode: Literal["fixed_parameter", "refit"] = "fixed_parameter"


class LOROCVResult(BaseModel):
    """loro_cv/{candidate_id}.json — complete LORO-CV output for one candidate.

    Stored as a separate bundle artifact under loro_cv/ directory.
    """

    candidate_id: str
    metrics: LOROMetrics
    fold_results: list[LOROFoldResult]
    wall_time_seconds: float = Field(ge=0.0)
    regimen_groups: list[str] = Field(default_factory=list)
    seed: int = 0


class ReportSummary(BaseModel):
    """report/summary.json — structured summary of the run (Phase 2+)."""

    run_id: str
    lane: str
    n_candidates_evaluated: int = Field(ge=0)
    n_gate1_passed: int = Field(ge=0)
    n_gate2_passed: int = Field(ge=0)
    recommended_candidate_id: str | None = None
    recommended_bic: float | None = None
    ranking_metric: str = "bic"
    data_summary: dict[str, str | int | float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ReportProvenance(BaseModel):
    """report_provenance.json — who/what generated each section."""

    generated_at: str
    apmode_version: str
    generator: str
    component_versions: dict[str, str] = Field(default_factory=dict)


# --- Bayesian artifacts (Phase 2+) ---


class PriorManifestEntry(BaseModel):
    """One declared prior with full provenance for FDA Gate 2 justification."""

    model_config = ConfigDict(frozen=True)

    target: str
    family: str
    source: Literal[
        "uninformative",
        "weakly_informative",
        "historical_data",
        "expert_elicitation",
        "meta_analysis",
    ]
    hyperparams: dict[str, float | list[float] | str | list[str]]
    justification: str
    historical_refs: list[str] = Field(default_factory=list)


class PriorManifest(BaseModel):
    """prior_manifest.json — versioned record of all declared priors (plan §3.5).

    This is the FDA-required prior justification artifact. Every prior on
    non-uninformative/weakly-informative source must carry a non-empty
    justification; historical_data must also carry historical_refs.
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str
    entries: list[PriorManifestEntry]
    default_prior_policy: Literal["weakly_informative", "custom"] = "weakly_informative"


class SimulationScenario(BaseModel):
    """One prospective-simulation scenario for Gate 3 operating characteristics."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_subjects: int = Field(gt=0)
    n_replicates: int = Field(gt=0)
    dropout_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    assay_cv: float = Field(ge=0.0, default=0.0)
    blq_mechanism: Literal["none", "m3", "m4"] = "none"
    lloq: float | None = None


class SimulationProtocol(BaseModel):
    """simulation_protocol.json — prospective-simulation specification (plan §3.5).

    Required for FDA 2026 operating-characteristics evaluation. Locked before
    Gate 3 execution to prevent metric shopping (per gpt-5.2 review note).
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str
    scenarios: list[SimulationScenario] = Field(min_length=1)
    metrics: list[
        Literal[
            "vpc_coverage",
            "auc_bioequivalence",
            "cmax_bioequivalence",
            "npe",
            "posterior_probability_target",
        ]
    ] = Field(min_length=1)
    seed: int = 0
