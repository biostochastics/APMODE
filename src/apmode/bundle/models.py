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
    """A single parameter estimate with uncertainty."""

    name: str
    estimate: float
    se: float | None = None
    rse: float | None = None
    ci95_lower: float | None = None
    ci95_upper: float | None = None
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


class DiagnosticBundle(BaseModel):
    """All diagnostic outputs for a candidate model."""

    gof: GOFMetrics
    vpc: VPCSummary | None = None
    identifiability: IdentifiabilityFlags
    blq: BLQHandling
    diagnostic_plots: dict[str, str] = Field(default_factory=dict)


# --- BackendResult ---


class BackendResult(BaseModel):
    """Standardized result from any backend (ARCHITECTURE.md §4.1)."""

    model_id: str
    backend: Literal["nlmixr2", "jax_node", "agentic_llm"]
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


# --- Data Manifest ---


class ColumnMapping(BaseModel):
    """Maps source columns to canonical schema fields."""

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
    blq_flag: str | None = None
    lloq: str | None = None
    occasion: str | None = None
    study_id: str | None = None


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


class ReportProvenance(BaseModel):
    """report_provenance.json — who/what generated each section."""

    generated_at: str
    apmode_version: str
    generator: str
    component_versions: dict[str, str] = Field(default_factory=dict)
