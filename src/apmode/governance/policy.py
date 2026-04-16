# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate policy file schema (PRD §4.3.1, ARCHITECTURE.md §4.3).

Gate thresholds are versioned policy artifacts, not hard-coded constants.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class BayesianThresholds(BaseModel):
    """Gate-1 Bayesian MCMC disqualification thresholds (PRD §4.3.1).

    Applied only when ``BackendResult.backend == "bayesian_stan"`` — MLE-based
    backends skip these checks. Defaults follow Vehtari et al. (2021)
    rank-normalized R-hat recommendations and arviz default nominal ESS.
    """

    rhat_max: float = Field(default=1.01, gt=1.0, le=2.0)
    ess_bulk_min: float = Field(default=400.0, gt=0.0)
    ess_tail_min: float = Field(default=400.0, gt=0.0)
    n_divergent_max: int = Field(default=0, ge=0)
    max_treedepth_fraction_max: float = Field(default=0.01, ge=0.0, le=1.0)
    ebfmi_min: float = Field(default=0.3, gt=0.0, le=1.0)
    pareto_k_max: float = Field(default=0.7, gt=0.0, le=1.0)


class Gate1Config(BaseModel):
    """Gate 1: Technical Validity thresholds (PRD §4.3.1)."""

    convergence_required: bool = True
    parameter_plausibility_required: bool = True
    state_trajectory_validity_required: bool = True
    split_integrity_required: bool = True
    # When False, candidates lacking a VPC pass the vpc_coverage check with
    # observed="vpc_not_configured" instead of failing. Defaults True so
    # production policies are conservative; set False in lane policies
    # until the backend VPC pipeline is wired end-to-end.
    vpc_required: bool = True
    cwres_mean_max: float
    outlier_fraction_max: float = Field(ge=0.0, le=1.0)
    vpc_coverage_lower: float
    vpc_coverage_upper: float
    seed_stability_n: int = Field(ge=1)
    seed_stability_cv_max: float = Field(default=0.10, gt=0.0, le=1.0)
    # State trajectory validity thresholds
    obs_vs_pred_r2_min: float = Field(default=0.30, ge=0.0, le=1.0)
    cwres_sd_min: float = Field(default=0.50, gt=0.0)
    cwres_sd_max: float = Field(default=2.0, gt=0.0)
    gradient_norm_max: float = Field(default=100.0, gt=0.0)
    # Bayesian-only thresholds (applied when backend == "bayesian_stan")
    bayesian: BayesianThresholds = Field(default_factory=BayesianThresholds)

    @model_validator(mode="after")
    def vpc_bounds_ordered(self) -> Gate1Config:
        if self.vpc_coverage_lower >= self.vpc_coverage_upper:
            msg = (
                f"vpc_coverage_lower ({self.vpc_coverage_lower}) must be < "
                f"vpc_coverage_upper ({self.vpc_coverage_upper})"
            )
            raise ValueError(msg)
        return self


class Gate2Config(BaseModel):
    """Gate 2: Lane-Specific Admissibility thresholds (PRD §4.3.1)."""

    interpretable_parameterization: Literal["required", "preferred", "not_required"]
    reproducible_estimation: Literal["required", "not_required"]
    shrinkage_max: float | None = None
    identifiability_required: bool
    node_eligible: bool
    loro_required: bool
    # LORO-CV thresholds (Phase 3 — Optimization lane)
    loro_npde_mean_max: float = Field(default=0.3, gt=0.0)
    loro_npde_variance_min: float = Field(default=0.5, gt=0.0)
    loro_npde_variance_max: float = Field(default=1.5, gt=0.0)
    loro_vpc_coverage_min: float = Field(default=0.80, ge=0.0, le=1.0)
    loro_min_folds: int = Field(default=3, ge=2)
    loro_budget_top_n: int | None = None


class Gate3Config(BaseModel):
    """Gate 3: Cross-paradigm ranking composite configuration (PRD §4.3.1).

    Cross-paradigm ranking must use simulation-based metrics because
    likelihoods (NLPD, BIC) are not comparable across different observation
    models. This config externalizes the composite-score machinery so
    deployments can choose between two aggregation strategies without
    touching source code:

      * ``composite_method="weighted_sum"`` — the legacy weighted sum of
        ``(1 - vpc_concordance)``, normalized NPE-proxy, and normalized
        BIC. Sensitive to metric scaling; ``npe_cap`` and
        ``bic_norm_scale`` tame the range.
      * ``composite_method="borda"`` — rank each candidate on each
        enabled metric (weight > 0), sum the ranks, lower total wins.
        Scale-invariant by construction; recommended for cross-paradigm
        comparisons per multi-model consensus.

    Metric inclusion is keyed on weight: setting ``bic_weight=0.0``
    drops BIC from the composite under both methods. This is the PRD
    v0.3-aligned default for cross-paradigm ranking (likelihood
    incomparability). Within-paradigm BIC ranking is unaffected — this
    config only governs the cross-paradigm / mixed-BLQ path.
    """

    # PRD §10 Q2: BIC / NLPD are incomparable across observation models, so
    # the cross-paradigm default is BIC off. Deployments wanting BIC in
    # cross-paradigm must opt in by setting ``bic_weight > 0`` explicitly.
    # Within-paradigm BIC ranking lives in a separate code path and is
    # unaffected.
    composite_method: Literal["weighted_sum", "borda"] = "weighted_sum"
    vpc_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    npe_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bic_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    # Weighted-sum normalization caps. Ignored under ``borda``.
    npe_cap: float = Field(default=5.0, gt=0.0)
    bic_norm_scale: float = Field(default=1000.0, gt=0.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> Self:
        total = self.vpc_weight + self.npe_weight + self.bic_weight
        # Tolerate small rounding error in policy JSON.
        if abs(total - 1.0) > 1e-6:
            msg = (
                f"gate3 composite weights must sum to 1.0 (got "
                f"vpc={self.vpc_weight} + npe={self.npe_weight} + "
                f"bic={self.bic_weight} = {total})"
            )
            raise ValueError(msg)
        # sum_to_one with non-negative components implies at least one weight
        # is strictly positive; no separate invariant needed.
        return self


class Gate25Config(BaseModel):
    """Gate 2.5: Credibility Qualification (Phase 2, PRD §4.3.1 / ICH M15).

    ICH M15 requires context-of-use specification and risk-based model
    assessment. This is a qualifying gate, not documentation.
    """

    context_of_use_required: bool = True
    limitation_to_risk_mapping_required: bool = False
    data_adequacy_required: bool = True
    data_adequacy_ratio_min: float = Field(default=5.0, gt=0)
    sensitivity_analysis_required: bool = False
    ai_ml_transparency_required: bool = False


class MissingDataPolicy(BaseModel):
    """Lane-specific missing-data handling policy.

    Drives ``MissingDataDirective`` resolution (see
    ``apmode.data.missing_data.resolve_directive``). Defaults correspond to
    the Discovery lane; Submission lane tightens thresholds, Optimization
    matches Discovery.

    References:
      - Nyberg 2024, Jonsson 2024 — FREM > FFEM+mean-imputation at high missingness
      - Wijk 2025 (DiVA) — M7+ ≈ M3 precision with better stability
      - Beal 2001 — M3/M7 BLQ method family
    """

    # Covariate-side: imputation method selection thresholds.
    # Below ``mi_pmm_max_missingness`` → MI-PMM; above → FREM (if data supports).
    mi_pmm_max_missingness: float = Field(default=0.30, ge=0.0, le=1.0)
    # If time-varying covariates detected, prefer FREM regardless of %-missing.
    frem_for_time_varying: bool = True
    # Minimum %-missing to trigger FREM preference over MI-PMM for static covariates.
    frem_preferred_above: float = Field(default=0.30, ge=0.0, le=1.0)
    # Fallback when missRanger (ranger-backed RF + PMM) is preferred over
    # plain PMM for nonlinear covariate relations. Legacy field name
    # preserved for policy-file backwards compatibility.
    missforest_fallback: bool = True

    # Number of imputations (m) for MI methods.
    m_imputations: int = Field(default=5, ge=1)
    # Adaptive escalation: if between-imputation variance exceeds threshold,
    # bump m up to m_max.
    adaptive_m: bool = False
    m_max: int = Field(default=20, ge=1)
    adaptive_variance_threshold: float = Field(default=0.10, gt=0.0, le=1.0)

    # BLQ (observation-side) method selection by BLQ% threshold.
    # Below threshold → M7+ (stable, simple); above → M3 (likelihood-based).
    blq_m3_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    # Override: M3 always (Submission-lane conservative option).
    blq_force_m3: bool = False

    # Agentic backend protections.
    llm_pooled_only: bool = True
    imputation_stability_penalty: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def m_ordering(self) -> Self:
        if self.adaptive_m and self.m_max < self.m_imputations:
            msg = (
                f"m_max ({self.m_max}) must be >= m_imputations "
                f"({self.m_imputations}) when adaptive_m is True"
            )
            raise ValueError(msg)
        if self.frem_preferred_above < self.mi_pmm_max_missingness:
            msg = (
                f"frem_preferred_above ({self.frem_preferred_above}) must be "
                f">= mi_pmm_max_missingness ({self.mi_pmm_max_missingness})"
            )
            raise ValueError(msg)
        return self


class GatePolicy(BaseModel):
    """Top-level policy file: versioned gate configuration per lane.

    Validates PRD §3 hard rule: Submission lane excludes NODE.
    """

    policy_version: str
    lane: Literal["submission", "discovery", "optimization"]
    gate1: Gate1Config
    gate2: Gate2Config
    gate2_5: Gate25Config | None = None
    gate3: Gate3Config = Field(default_factory=Gate3Config)
    missing_data: MissingDataPolicy = Field(default_factory=MissingDataPolicy)
    vpc_concordance_target: float = Field(default=0.90, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def submission_excludes_node(self) -> Self:
        """PRD §3 hard rule: NODE models not eligible in Submission lane."""
        if self.lane == "submission" and self.gate2.node_eligible:
            msg = (
                "NODE models are not eligible in the Submission lane "
                "(PRD §3 hard rule). Set gate2.node_eligible=false."
            )
            raise ValueError(msg)
        return self
