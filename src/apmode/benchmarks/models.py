# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark infrastructure models (PRD §5, §9).

Declarative case specs, dataset cards, perturbation recipes, and scoring
models for the three-tier benchmark system:

  Suite A:  Synthetic recovery (known ground truth)
  Suite A-External: Schoemaker 2019 standard grid (nlmixr2data)
  Suite B:  Real-data anchors with controlled perturbations
  Suite C:  Expert head-to-head comparison

Architecture:
  - Each benchmark *case* = (dataset, preprocessing, perturbation, lane, policy, split, scoring)
  - Dataset cards describe provenance, access tier, and canonicalization
  - Scoring harness is backend-agnostic
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Literal type alias for suites (shared across models + scoring)
# ---------------------------------------------------------------------------

SuiteName = Literal["A", "A_external", "B", "C"]

# ---------------------------------------------------------------------------
# Access tier: controls CI eligibility
# ---------------------------------------------------------------------------


class AccessTier(StrEnum):
    """Dataset access level — determines CI/nightly eligibility."""

    OPEN = "open"  # Freely redistributable or CRAN-installable
    SCRIPTED = "scripted"  # Requires automated download (OpenTCI, DDMoRe)
    CREDENTIALED = "credentialed"  # Requires login (MIMIC-IV, PhysioNet)


# ---------------------------------------------------------------------------
# Published Model (defined before DatasetCard to avoid forward-ref issues)
# ---------------------------------------------------------------------------


class PublishedModel(BaseModel):
    """Reference to a published expert model for comparison."""

    model_config = ConfigDict(frozen=True)

    citation: str
    structure: str  # e.g., "2-CMT oral, first-order absorption, linear elimination"
    n_parameters: int = Field(gt=0)
    estimation_method: str  # e.g., "FOCE-I", "SAEM"
    tool: str  # e.g., "NONMEM 7.5", "Monolix 2024R1"
    key_estimates: dict[str, float] = Field(default_factory=dict)
    ofv: float | None = None


# ---------------------------------------------------------------------------
# Dataset Card
# ---------------------------------------------------------------------------


class DatasetCard(BaseModel):
    """Metadata for a benchmark dataset (provenance, license, schema).

    One card per dataset. Stored as YAML in benchmarks/datasets/<id>/.
    The prepare script canonicalizes raw data into NONMEM-style CSV.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    name: str
    description: str
    source: str  # URL or package::dataset
    citation: str  # Short citation (Author YYYY)
    license: str  # SPDX identifier or description
    access_tier: AccessTier

    # Data characteristics
    n_subjects: int = Field(gt=0)
    n_observations: int = Field(gt=0)
    route: Literal["oral", "iv_bolus", "iv_infusion", "sc", "mixed"]
    n_compartments: int = Field(ge=1, le=5)
    has_covariates: bool = False
    covariate_names: list[str] = Field(default_factory=list)
    has_iov: bool = False
    has_blq: bool = False
    has_multidose: bool = False
    has_steady_state: bool = False

    # Files
    prepare_script: str  # Relative path to prepare.R or prepare.py
    canonical_csv: str  # Expected output filename
    sha256: str | None = None  # SHA-256 of canonical CSV (set after generation)

    # Known published model (for Suite C expert baseline)
    published_model: PublishedModel | None = None


# ---------------------------------------------------------------------------
# Perturbation Recipe
# ---------------------------------------------------------------------------


class PerturbationType(StrEnum):
    """Types of controlled data perturbation for Suite B/C.

    #26: the extra members below extend Suite C stress coverage to the
    PRD §10 risk surfaces (BSV scaling, saturation, TMDD, flip-flop).
    The concrete transforms live in benchmarks.perturbations; each one
    raises NotImplementedError until the implementation lands so
    Suite C never silently runs without the requested stressor.
    """

    INJECT_BLQ = "inject_blq"
    REMOVE_ABSORPTION_SAMPLES = "remove_absorption_samples"
    ADD_NULL_COVARIATES = "add_null_covariates"
    INJECT_OUTLIERS = "inject_outliers"
    SPARSIFY = "sparsify"
    ADD_PROTOCOL_POOLING = "add_protocol_pooling"
    ADD_OCCASION_LABELS = "add_occasion_labels"
    SCALE_BSV_VARIANCES = "scale_bsv_variances"
    SATURATE_CLEARANCE = "saturate_clearance"
    TMDD = "tmdd"
    FLIP_FLOP = "flip_flop"


class PerturbationRecipe(BaseModel):
    """Declarative perturbation to apply to a dataset.

    Perturbations are pure functions: (canonical_csv, recipe) -> perturbed_csv.
    Each produces a perturbation_manifest.json describing what changed.
    """

    model_config = ConfigDict(frozen=True)

    perturbation_type: PerturbationType
    seed: int = 753849

    # BLQ injection
    blq_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    lloq: float | None = Field(default=None, gt=0.0)

    # Absorption sample removal
    absorption_time_cutoff: float | None = Field(default=None, gt=0.0)

    # Null covariate injection
    null_covariate_names: list[str] = Field(default_factory=list)
    null_covariate_n: int = Field(default=0, ge=0)

    # Outlier injection
    outlier_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    outlier_magnitude: float = Field(default=5.0, gt=1.0)

    # Sparsification
    target_obs_per_subject: int | None = Field(default=None, ge=1)

    # Protocol pooling
    n_protocols: int = Field(default=2, ge=2, le=10)
    vary_sampling: bool = False  # Vary sampling schedules per protocol
    vary_lloq: bool = False  # Vary LLOQ per protocol

    # #26: stress-surface parameters. None when the corresponding
    # perturbation is not selected; the model_validator below enforces
    # required combinations.
    bsv_scale_factor: float | None = Field(default=None, gt=0.0)
    saturation_km: float | None = Field(default=None, gt=0.0)
    saturation_vmax: float | None = Field(default=None, gt=0.0)
    tmdd_kss: float | None = Field(default=None, gt=0.0)
    tmdd_r0: float | None = Field(default=None, gt=0.0)
    flip_flop_ka: float | None = Field(default=None, gt=0.0)
    flip_flop_ke_ratio: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _validate_required_fields(self) -> PerturbationRecipe:
        """Ensure perturbation-specific fields are set for the given type."""
        t = self.perturbation_type
        if t == PerturbationType.INJECT_BLQ and self.blq_fraction is None:
            msg = "blq_fraction required for INJECT_BLQ"
            raise ValueError(msg)
        if t == PerturbationType.REMOVE_ABSORPTION_SAMPLES and self.absorption_time_cutoff is None:
            msg = "absorption_time_cutoff required for REMOVE_ABSORPTION_SAMPLES"
            raise ValueError(msg)
        if t == PerturbationType.INJECT_OUTLIERS and self.outlier_fraction is None:
            msg = "outlier_fraction required for INJECT_OUTLIERS"
            raise ValueError(msg)
        if t == PerturbationType.SPARSIFY and self.target_obs_per_subject is None:
            msg = "target_obs_per_subject required for SPARSIFY"
            raise ValueError(msg)
        if (
            t == PerturbationType.ADD_NULL_COVARIATES
            and self.null_covariate_n == 0
            and not self.null_covariate_names
        ):
            msg = "null_covariate_n or null_covariate_names required"
            raise ValueError(msg)
        # #26 stress-surface parameter presence checks
        if t == PerturbationType.SCALE_BSV_VARIANCES and self.bsv_scale_factor is None:
            msg = "bsv_scale_factor required for SCALE_BSV_VARIANCES"
            raise ValueError(msg)
        if t == PerturbationType.SATURATE_CLEARANCE and (
            self.saturation_km is None or self.saturation_vmax is None
        ):
            msg = "saturation_km and saturation_vmax required for SATURATE_CLEARANCE"
            raise ValueError(msg)
        if t == PerturbationType.TMDD and (self.tmdd_kss is None or self.tmdd_r0 is None):
            msg = "tmdd_kss and tmdd_r0 required for TMDD"
            raise ValueError(msg)
        if t == PerturbationType.FLIP_FLOP and (
            self.flip_flop_ka is None or self.flip_flop_ke_ratio is None
        ):
            msg = "flip_flop_ka and flip_flop_ke_ratio required for FLIP_FLOP"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Benchmark Case Spec
# ---------------------------------------------------------------------------


class SplitStrategy(BaseModel):
    """Train/test split configuration for predictive evaluation."""

    model_config = ConfigDict(frozen=True)

    method: Literal["subject_level_kfold", "time_based", "stratified", "regimen_level"]
    n_folds: int = Field(default=5, ge=2, le=20)
    seed: int = 753849


class ExpectedStructure(BaseModel):
    """Expected model structure for recovery testing."""

    model_config = ConfigDict(frozen=True)

    absorption: str | None = None  # e.g., "FirstOrder"
    distribution: str | None = None  # e.g., "TwoCmt"
    elimination: str | None = None  # e.g., "Linear"
    n_compartments: int | None = None


class BenchmarkCase(BaseModel):
    """A single benchmark evaluation case.

    The atomic unit of the benchmark system: one dataset, one configuration,
    one set of assertions or scoring criteria. Multiple cases can share a
    dataset with different perturbations/lanes/policies.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    suite: SuiteName
    dataset_id: str
    description: str

    # Configuration
    lane: Literal["submission", "discovery", "optimization"]
    policy_file: str = "submission.json"
    perturbations: list[PerturbationRecipe] = Field(default_factory=list)
    split_strategy: SplitStrategy | None = None

    # Suite A/A-ext: ground-truth assertions
    expected_structure: ExpectedStructure | None = None
    reference_params: dict[str, float] = Field(default_factory=dict)
    param_bias_tolerance: float = 0.20  # 20% default

    # Suite B: dispatch/perturbation assertions
    expected_dispatch_includes: list[str] = Field(default_factory=list)
    expected_dispatch_excludes: list[str] = Field(default_factory=list)

    # Suite C: literature-benchmark comparators (plan Task 38 rename from
    # ``expert_models``). Phase 1 of Suite C compares APMODE against
    # published, peer-reviewed reference parameterizations. Phase 2
    # (head-to-head vs a blinded human-expert panel) is gated on
    # external collaborator coordination and out of v0.6 scope.
    literature_models: list[PublishedModel] = Field(default_factory=list)
    win_margin: float = 0.02  # delta for NPE win rule

    # CI cadence
    ci_cadence: Literal["per_pr", "nightly", "weekly", "quarterly"] = "nightly"


# ---------------------------------------------------------------------------
# Benchmark Score (output of scoring harness)
# ---------------------------------------------------------------------------


class MetricValue(BaseModel):
    """A single scored metric with optional uncertainty."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    lower_ci: float | None = None
    upper_ci: float | None = None
    unit: str = ""
    passed: bool | None = None  # None if no threshold


class BenchmarkScore(BaseModel):
    """Scoring output for a single benchmark case.

    Produced by the scoring harness; one per (case, run) pair.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    run_id: str
    suite: SuiteName

    # Primary metrics
    metrics: list[MetricValue] = Field(default_factory=list)

    # Structure recovery (Suite A)
    structure_recovered: bool | None = None

    # Parameter recovery (Suite A)
    param_bias: dict[str, float] = Field(default_factory=dict)
    param_coverage: dict[str, bool | None] = Field(
        default_factory=dict
    )  # None = CI unavailable (unscorable)

    # Predictive performance (Suite B/C)
    npe: float | None = None  # Nonparametric prediction error
    prediction_interval_calibration: dict[str, float] = Field(
        default_factory=dict
    )  # {"50": 0.52, "80": 0.79, ...}

    # Expert comparison (Suite C)
    beats_median_expert: bool | None = None
    expert_npe_median: float | None = None
    npe_gap: float | None = None  # APMODE NPE - expert median NPE

    # Dispatch assertions (Suite B)
    dispatch_correct: bool | None = None

    # Efficiency
    wall_time_seconds: float | None = None
    candidates_evaluated: int | None = None

    # Governance
    gate1_passed: bool | None = None
    gate2_passed: bool | None = None
    gate3_rank: int | None = None

    # Convergence
    convergence_rate: float | None = None
    failure_classes: dict[str, int] = Field(
        default_factory=dict
    )  # {"non_convergence": 2, "crash": 0, ...}

    # Overall
    overall_passed: bool = False


class SuiteSummary(BaseModel):
    """Summary statistics for a suite run."""

    model_config = ConfigDict(frozen=True)

    n_cases: int = Field(ge=0)
    n_passed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    mean_wall_time_seconds: float | None = None
    # Suite A specific
    structural_recovery_rate: float | None = None
    mean_param_bias: float | None = None
    # Suite C specific
    fraction_beats_expert: float | None = None

    @model_validator(mode="after")
    def _validate_passed_leq_cases(self) -> SuiteSummary:
        if self.n_passed > self.n_cases:
            msg = f"n_passed ({self.n_passed}) > n_cases ({self.n_cases})"
            raise ValueError(msg)
        return self


class SuiteReport(BaseModel):
    """Aggregate report for an entire benchmark suite run."""

    model_config = ConfigDict(frozen=True)

    suite: SuiteName
    scores: list[BenchmarkScore]
    summary: SuiteSummary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AccessTier",
    "BenchmarkCase",
    "BenchmarkScore",
    "DatasetCard",
    "ExpectedStructure",
    "MetricValue",
    "PerturbationRecipe",
    "PerturbationType",
    "PublishedModel",
    "SplitStrategy",
    "SuiteName",
    "SuiteReport",
    "SuiteSummary",
]
