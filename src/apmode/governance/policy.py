# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate policy file schema (PRD §4.3.1, ARCHITECTURE.md §4.3).

Gate thresholds are versioned policy artifacts, not hard-coded constants.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class PolicyError(ValueError):
    """Raised when a candidate violates a versioned lane policy invariant.

    Scoped narrowly to disambiguate policy-driven rejections (metric
    shopping, unsupported backends, unsupported metric names) from the
    pydantic ``ValueError`` raised at schema-validation time. Inherits
    from :class:`ValueError` so existing except-handlers stay correct.
    """


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
    # When True, a ``BackendResult`` missing ``split_gof`` fails Gate 1;
    # missing required evidence must never silently pass (PRD §4.3.1
    # disqualifying-funnel invariant). Default is False because
    # benchmark scenarios and single-fold workflows legitimately run
    # without a split manifest; policies that require held-out GOF
    # agreement must set this to True explicitly.
    split_integrity_required: bool = False
    # When False, candidates lacking a VPC pass the vpc_coverage check with
    # observed="vpc_not_configured" instead of failing. Defaults True so
    # production policies are conservative; set False in lane policies
    # until the backend VPC pipeline is wired end-to-end.
    vpc_required: bool = True
    cwres_mean_max: float
    outlier_fraction_max: float = Field(ge=0.0, le=1.0)
    # Legacy VPC knobs — retained for backward compatibility with policy
    # JSONs that still set them, but Gate 1 no longer consumes these. The
    # bin-level VPC coverage metric produced brittle 1/n_bins-quantized
    # coverage values that false-rejected textbook-correct models on
    # sparse real datasets (warfarin: 0/33 pass in rc9). Replaced by the
    # PIT/NPDE-lite calibration knobs below (policy 0.4.2).
    vpc_coverage_target: float = Field(default=0.90, ge=0.0, le=1.0)
    vpc_coverage_tolerance: float = Field(default=0.15, gt=0.0, le=1.0)
    # PIT / NPDE-lite predictive calibration — 0.4.2 Gate 1 gated metric.
    # For each probability level p in {0.05, 0.50, 0.95} the check asks
    # ``|c_p - p| <= tol(p, n_subjects)`` where ``c_p`` is the subject-
    # robust fraction of observations at or below the simulated
    # p-quantile at that observation (see ``PITCalibrationSummary``
    # docstring for the formula). The tolerance is **n-scaled** to keep
    # the SE-coverage constant across dataset sizes:
    #
    #   tol(p, n) = max(floor_{tail|med}, z_alpha · sqrt(p(1-p)/n_subjects))
    #
    # - ``z_alpha``: lane-specific admissibility (SE multiplier). Submission
    #   1.5 (strictest, ≈87% CI per-band), optimization 2.0, discovery
    #   2.5. Higher z_alpha → wider window → more permissive.
    # - ``floor_*``: absolute lower bounds on the tolerance. They bind
    #   at large ``n`` where the z·SE expression would otherwise make
    #   the gate asymptotically vacuous-strict (n=1e4 → SE ≈ 0.002 →
    #   tol ≈ 0.003 which would reject perfectly-calibrated models
    #   due to sampling noise alone).
    # - Effective sample size is ``n_subjects`` (outer-mean denominator)
    #   rather than ``n_observations`` — governance-conservative since
    #   within-subject observations are strongly correlated in
    #   population-PK. Using n_observations would understate the
    #   sampling variance and produce over-strict gates on dense
    #   regimens.
    #
    # Tail-vs-median split: tail miscalibration is the most diagnostic
    # signal of residual-error / IIV misspecification, so we allocate a
    # smaller floor (tighter absolute gate) to tails. The n-scaled term
    # also gives tails a smaller SE contribution (sqrt(p(1-p)) is ≈0.22
    # at tails vs ≈0.50 at the median) which aligns with the intent.
    #
    # Lane calibration: submission strictest (regulatory), discovery
    # widest (NODE/agentic variance expected), optimization in between.
    # Defaults below mirror the ``policies/*.json`` submission lane —
    # per-lane JSONs override.
    pit_required: bool = True
    pit_z_alpha: float = Field(default=1.5, gt=0.0, le=5.0)
    pit_tol_tail_floor: float = Field(default=0.03, gt=0.0, le=1.0)
    pit_tol_median_floor: float = Field(default=0.05, gt=0.0, le=1.0)
    seed_stability_n: int = Field(ge=1)
    seed_stability_cv_max: float = Field(default=0.10, gt=0.0, le=1.0)
    # State trajectory validity thresholds
    obs_vs_pred_r2_min: float = Field(default=0.30, ge=0.0, le=1.0)
    cwres_sd_min: float = Field(default=0.50, gt=0.0)
    cwres_sd_max: float = Field(default=2.0, gt=0.0)
    gradient_norm_max: float = Field(default=100.0, gt=0.0)
    # Split-integrity thresholds (previously hard-coded in gates.py).
    # ``split_cwres_drift_max`` caps |abs(test_cwres_mean) - abs(train_cwres_mean)|.
    # ``split_outlier_ratio_slope``/``split_outlier_ratio_intercept`` define
    # the admissible ceiling on test outlier fraction as
    # ``slope * train_outlier_fraction + intercept`` (legacy: 2.0, 0.05).
    split_cwres_drift_max: float = Field(default=0.5, gt=0.0)
    split_outlier_ratio_slope: float = Field(default=2.0, gt=0.0)
    split_outlier_ratio_intercept: float = Field(default=0.05, ge=0.0, le=1.0)
    # Parameter-plausibility sanity bounds (previously hard-coded in
    # gates.py::_check_parameter_plausibility). The back-transformed
    # structural estimate (or the raw estimate for non-log-space params)
    # must lie in (param_value_min, param_value_max). An RSE above
    # ``param_rse_max`` flags the estimate as under-identified.
    param_value_min: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "Lower bound (exclusive) on structural parameter estimates. "
            "Units follow the parameter: rates in 1/time, volumes in L, "
            "clearances in L/time. #31 audit metadata."
        ),
    )
    param_value_max: float = Field(
        default=1e5,
        gt=0.0,
        description=(
            "Upper bound (exclusive) on structural parameter estimates. "
            "Same units as :attr:`param_value_min`."
        ),
    )
    param_rse_max: float = Field(
        default=200.0,
        gt=0.0,
        description=(
            "Maximum allowed relative standard error (percent). "
            "Estimates with RSE above this flag under-identification."
        ),
    )
    # Seed-stability short-circuit: platform/BLAS float-accumulation can
    # move OFV by ~1e-3 OFV units between "identical" fits. When the
    # absolute peak-to-peak OFV spread is below this floor, the CV check
    # is skipped (it would still report instability even though the
    # spread is below any scientifically meaningful ΔAIC threshold).
    seed_stability_ofv_abs_spread_floor: float = Field(default=0.1, gt=0.0)
    # Bayesian-only thresholds (applied when backend == "bayesian_stan")
    bayesian: BayesianThresholds = Field(default_factory=BayesianThresholds)


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

    # Bayesian prior justification (plan Task 19, FDA 2026 draft). When
    # True, every informative prior on the candidate's prior manifest
    # must carry a justification of at least
    # ``bayesian_prior_justification_min_length`` characters AND a
    # Crossref-canonical DOI. The Submission lane defaults this to True;
    # Discovery/Optimization leave it False so model-search workflows
    # aren't held hostage by provenance paperwork.
    bayesian_prior_justification_required: bool = False
    # Length floor for the justification when
    # ``bayesian_prior_justification_required`` is True. Defaults to
    # the FDA Gate 2 floor (50 chars); Submission-lane policies tighten
    # to 500 so reviewers see substantive provenance, not boilerplate.
    bayesian_prior_justification_min_length: int = Field(default=50, ge=10)


_GATE3_DEFAULT_METRIC_STACK: list[Literal["vpc_concordance", "auc_cmax_gmr", "npe"]] = [
    "vpc_concordance",
    "auc_cmax_gmr",
    "npe",
]


def _default_gate3_metric_stack() -> list[Literal["vpc_concordance", "auc_cmax_gmr", "npe"]]:
    return list(_GATE3_DEFAULT_METRIC_STACK)


class Gate3Config(BaseModel):
    """Gate 3: Cross-paradigm ranking composite configuration (PRD §4.3.1).

    Cross-paradigm ranking must use simulation-based metrics because
    likelihoods (NLPD, BIC) are not comparable across different observation
    models. This config externalizes the composite-score machinery so
    deployments can choose between two aggregation strategies without
    touching source code:

      * ``composite_method="weighted_sum"`` — the legacy weighted sum of
        ``(1 - vpc_concordance)``, normalized NPE-proxy, normalized BIC,
        and ``(1 - auc_cmax_be_score)``. Sensitive to metric scaling;
        ``npe_cap`` and ``bic_norm_scale`` tame the range.
      * ``composite_method="borda"`` — rank each candidate on each
        enabled metric (weight > 0), sum the ranks, lower total wins.
        Scale-invariant by construction; recommended for cross-paradigm
        comparisons because likelihood scales are incomparable (PRD §10 Q2).

    Metric inclusion is keyed on weight: setting ``bic_weight=0.0`` or
    ``auc_cmax_weight=0.0`` drops that component from the composite. When a
    candidate legitimately lacks a weighted component (e.g. NCA reference is
    ineligible → ``auc_cmax_be_score is None``) the rule is uniform-drop:
    the component is removed from the composite for *every* candidate in
    that ranking and the remaining weights renormalized. Per-candidate
    renormalization is rejected because it lets data-poor candidates dodge
    metrics their siblings are scored on — defeats the comparability goal
    of Gate 3 (PRD §4.3.1).
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
    # AUC/Cmax bioequivalence component (Smith 2000 style GMR in [0.80, 1.25]).
    # Default off: computing it requires both a candidate posterior-predictive
    # simulation matrix AND an NCA-admissible observed dataset. Enable it in
    # policy (e.g. optimization lane) once the backend VPC pipeline is wired.
    auc_cmax_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    # Weighted-sum normalization caps. Ignored under ``borda``.
    npe_cap: float = Field(default=5.0, gt=0.0)
    bic_norm_scale: float = Field(default=1000.0, gt=0.0)
    # Eligibility threshold for deriving an observed-NCA reference for
    # auc_cmax_be. Above this BLQ fraction, per-subject AUC is biased by
    # censoring (Thway 2018) and the candidate score falls back to ``None``
    # rather than a candidate-derived reference (avoids circular herd bias).
    auc_cmax_nca_max_blq_burden: float = Field(default=0.20, ge=0.0, le=1.0)
    # Number of posterior-predictive simulation replicates backends draw
    # per candidate. 500 matches Bergstrand 2011 VPC convention; lower
    # bound 100 preserves variance estimates, upper 5000 caps ODE cost.
    # Policy-versioned so deployments tuning this see the change in the
    # bundle's policy_file.json.
    n_posterior_predictive_sims: int = Field(default=500, ge=100, le=5000)
    # Time-bin count for post-hoc VPC coverage aggregation (no pre-
    # declared simulation grid — binning happens on observation times
    # after backends simulate at observed records only). Default 10
    # preserves the bin count used by existing rc7 tests.
    vpc_n_bins: int = Field(default=10, ge=3, le=100)
    # Per-subject NCA eligibility floors for emitting auc_cmax_be_score.
    # Below either floor the score drops to ``None`` rather than being
    # computed on an unreliably small reference set. ``min_eligible`` is
    # the absolute subject count; ``min_eligible_fraction`` is the
    # fraction of the ranked survivor's cohort that must be NCA-admissible.
    # Both must hold: floors are AND-combined.
    auc_cmax_nca_min_eligible: int = Field(default=8, ge=1)
    auc_cmax_nca_min_eligible_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    # Warn when the effective VPC bin count (after ``np.unique`` on quantile
    # edges collapses ties) falls below this fraction of ``vpc_n_bins``.
    # 0.5 = half the requested bins; below that the post-hoc binning is
    # producing a noisy VPC the audit log must flag.
    vpc_n_bin_collapse_warn_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    # NPE aggregation across subjects. ``"flatten"`` (default, rc8 behavior)
    # pools all observed (obs, sim-median) pairs before the median-absolute-
    # error; well-sampled subjects contribute more. ``"per_subject_median"``
    # computes NPE per subject first, then medians across subjects — reduces
    # bias toward dense-sampled subjects in unbalanced designs.
    npe_aggregation: Literal["flatten", "per_subject_median"] = "flatten"
    # AUC/Cmax per-subject summary across simulation draws.
    # ``"median_trajectory"`` (default, rc8 behavior) collapses sims to a
    # per-sim median trajectory then trapezoids once — a point-estimate
    # summary. ``"median_of_aucs"`` trapezoidates each sim separately per
    # subject and takes the median of those scalar AUC/Cmax values —
    # preserves distributional uncertainty for nonlinear profiles.
    auc_cmax_aggregation: Literal["median_trajectory", "median_of_aucs"] = "median_trajectory"
    # When True, backends emit a second simulation pass on a pooled time
    # grid (uniform spacing across observed range) in addition to the
    # observed-time sims. The resulting VPCSummary carries
    # ``prediction_corrected=True`` and is preferred by the ranker for
    # regulatory-facing runs where a smoothed VPC curve is expected.
    # Default False: rc8 ships the observed-time VPC only. The R harness
    # side of the second pass is tracked as a follow-up commit.
    vpc_include_prediction_corrected: bool = False

    # Gate 3 anti-metric-shopping whitelist (plan Task 24). Candidates
    # that propose a Gate-3 metric outside this list are rejected by
    # the ranker with a ``PolicyError``. The default covers the three
    # metrics the PRD §4.3.1 ranker computes today. Deployments that
    # add a metric must extend the Literal type *and* re-version the
    # policy so downstream bundles can detect the change.
    metric_stack: list[Literal["vpc_concordance", "auc_cmax_gmr", "npe"]] = Field(
        default_factory=_default_gate3_metric_stack,
    )
    # Number of Laplace draws for the MLE posterior-predictive helper
    # (plan Task 22). Submission lane defaults to 2000 (tight intervals
    # for regulatory review); Discovery to 500 (exploration budget).
    laplace_draws: int = Field(default=500, ge=100, le=10000)

    def validate_metric(self, metric: str) -> None:
        """Raise :class:`PolicyError` if ``metric`` is outside ``metric_stack``.

        Gate 3 ranker calls this before consuming any candidate-proposed
        metric to prevent *metric shopping*: a candidate declaring a
        favourable metric not on the lane's whitelist would otherwise
        dominate the ranking by exclusion. The stack is policy-versioned
        so an addition/removal shows up in ``policy_file.json`` and
        downstream bundles.
        """
        if metric not in self.metric_stack:
            msg = (
                f"Gate 3 metric {metric!r} is not on this lane's "
                f"metric_stack {sorted(self.metric_stack)}; refusing to "
                "rank with it (policy anti-metric-shopping invariant)."
            )
            raise PolicyError(msg)

    @model_validator(mode="after")
    def metric_stack_non_empty(self) -> Self:
        if not self.metric_stack:
            msg = "gate3.metric_stack must list at least one metric"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> Self:
        total = self.vpc_weight + self.npe_weight + self.bic_weight + self.auc_cmax_weight
        # Tolerate small rounding error in policy JSON.
        if abs(total - 1.0) > 1e-6:
            msg = (
                f"gate3 composite weights must sum to 1.0 (got "
                f"vpc={self.vpc_weight} + npe={self.npe_weight} + "
                f"bic={self.bic_weight} + auc_cmax={self.auc_cmax_weight} = {total})"
            )
            raise ValueError(msg)
        # sum_to_one with non-negative components implies at least one weight
        # is strictly positive; no separate invariant needed.
        return self


class Gate1BayesianRhat(BaseModel):
    """Per-parameter-class R-hat floors for Gate 1 Bayesian checks.

    Defaults follow Vehtari et al. (2021) rank-normalized split-R-hat
    recommendations: 1.01 for well-identified fixed effects / residual
    error, 1.02 for IIV SDs (slightly looser because partial pooling
    broadens the posterior), and 1.05 for correlation entries (notoriously
    slow-mixing under LKJ).
    """

    fixed_effects: float = Field(default=1.01, gt=1.0, le=2.0)
    iiv: float = Field(default=1.02, gt=1.0, le=2.0)
    residual: float = Field(default=1.01, gt=1.0, le=2.0)
    correlations: float = Field(default=1.05, gt=1.0, le=2.0)


class Gate1BayesianESS(BaseModel):
    """Per-parameter-class effective-sample-size floors for Gate 1 Bayesian."""

    fixed_effects: int = Field(default=400, gt=0)
    iiv: int = Field(default=400, gt=0)
    residual: int = Field(default=400, gt=0)
    correlations: int = Field(default=100, gt=0)


_GATE1_BAYESIAN_DEFAULT_SEVERITY: dict[str, Literal["warn", "fail"]] = {
    "rhat": "fail",
    "ess": "fail",
    "divergences": "fail",
    "pareto_k": "warn",
}


def _default_gate1_bayesian_severity() -> dict[str, Literal["warn", "fail"]]:
    return dict(_GATE1_BAYESIAN_DEFAULT_SEVERITY)


class Gate1BayesianConfig(BaseModel):
    """Gate 1 Bayesian convergence floor, by parameter class (PRD §4.3.1).

    Refines the scalar :class:`BayesianThresholds` (kept for existing
    consumers) with per-parameter-class R-hat / ESS floors and a
    severity map that lets lanes treat specific violations as ``warn``
    instead of ``fail``. Defaults are strict (submission-lane style);
    Discovery / Optimization lane policies relax selectively.
    """

    rhat_max: Gate1BayesianRhat = Field(default_factory=Gate1BayesianRhat)
    ess_bulk_min: Gate1BayesianESS = Field(default_factory=Gate1BayesianESS)
    ess_tail_min: Gate1BayesianESS = Field(default_factory=Gate1BayesianESS)
    divergence_tolerance: int = Field(default=0, ge=0)
    max_treedepth_fraction: float = Field(default=0.01, ge=0.0, le=1.0)
    e_bfmi_min: float = Field(default=0.3, gt=0.0, le=1.0)
    pareto_k_max: float = Field(default=0.7, gt=0.0, le=1.0)
    severity: dict[str, Literal["warn", "fail"]] = Field(
        default_factory=_default_gate1_bayesian_severity
    )

    @model_validator(mode="after")
    def severity_covers_four_axes(self) -> Self:
        expected = {"rhat", "ess", "divergences", "pareto_k"}
        missing = expected - set(self.severity.keys())
        if missing:
            msg = (
                f"gate1_bayesian.severity must cover {sorted(expected)}; "
                f"missing: {sorted(missing)}"
            )
            raise ValueError(msg)
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

    # Per-lane floor on the imputation convergence rate. Candidates whose
    # MI fits converge in fewer than this fraction of imputations are
    # hard-rejected at Gate 1 regardless of penalty weight. ICH E9(R1)
    # requires defensible missing-data handling; a low convergence rate
    # means the candidate's evidence base is unreliable. Policies should
    # tighten for Submission and Optimization lanes (see policies/*.json).
    imputation_convergence_rate_min: float = Field(default=0.5, ge=0.0, le=1.0)

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
    gate1_bayesian: Gate1BayesianConfig = Field(default_factory=Gate1BayesianConfig)
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
