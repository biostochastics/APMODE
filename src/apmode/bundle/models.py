# SPDX-License-Identifier: GPL-2.0-or-later
"""Pydantic models for reproducibility bundle artifacts (ARCHITECTURE.md §5, PRD §4.3.2).

Every JSON/JSONL artifact in the bundle has a corresponding Pydantic model here.
Models are validated before writing to disk.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class SignalId(StrEnum):
    """Stable identifiers for nonlinear-clearance voting signals.

    These strings are the de-facto public API for downstream consumers
    (Lane Router, report renderer, external audit tooling). New signals
    append new members; existing values MUST NOT change.
    """

    CURVATURE_RATIO = "curvature_ratio"
    TERMINAL_MONOEXP = "terminal_monoexp"
    DOSE_PROPORTIONALITY_SMITH = "dose_proportionality_smith"


class NonlinearClearanceSignal(BaseModel):
    """One voting signal for nonlinear-clearance classification.

    Each signal carries enough provenance to reproduce the dispatch
    decision offline: the algorithm family, a bibliographic citation,
    a JSON-pointer into ``policies/profiler.json`` identifying the
    threshold used, the observed value + 90% CI where available, and
    the eligibility rationale when the signal abstained.
    """

    model_config = ConfigDict(frozen=True)

    signal_id: SignalId
    algorithm: str
    citation: str
    policy_key: str
    threshold_value: float | None = None
    observed_value: float | None = None
    ci90_low: float | None = None
    ci90_high: float | None = None
    eligible: bool
    eligibility_reason: str | None = None
    voted: bool
    n_subjects: int = Field(default=0, ge=0)
    extras: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


# --- Sub-models for BackendResult (ARCHITECTURE.md §4.1) ---

HexSHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]


class ParameterEstimate(BaseModel):
    """A single parameter estimate with uncertainty.

    Fields are a superset of what any single backend emits:
    - MLE backends populate estimate/se/rse/ci95_{lower,upper}.
    - Bayesian backend populates estimate (primary point estimate —
      posterior mean by default), posterior_mean (redundantly, when reports
      need to disambiguate from posterior median), posterior_sd,
      q05/q50/q95, and leaves se/rse/ci95 as None. ci95 from Bayesian is
      the 95% credible interval; MLE-side ci95 is the Wald/profile CI.
    """

    name: str
    estimate: float
    se: float | None = None
    rse: float | None = None
    ci95_lower: float | None = None
    ci95_upper: float | None = None
    posterior_mean: float | None = None
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
    """Visual Predictive Check summary statistics.

    Descriptive only as of policy 0.4.2 — retained for reports/plots and
    within-lane Gate 3 concordance ranking, but no longer a Gate 1 pass/
    fail gate. The bin-level percentile-curve-containment metric was
    brittle on sparse real data (warfarin false-rejected all 33 candidates
    due to 1/8-discrete tail coverage). Gate 1 now uses
    :class:`PITCalibrationSummary` instead — see CHANGELOG rc9 follow-up
    "PIT/NPDE-lite Gate 1 calibration".
    """

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


def _pit_key(p: float) -> str:
    """Canonical dict-key for a PIT probability level.

    All sites that round-trip PIT keys — the calibration computer, the
    Gate 1 check, and ``PITCalibrationSummary``'s Pydantic validator —
    route through this helper so changes to the naming convention are a
    one-place edit rather than a three-way drift waiting to happen.
    """
    return f"p{round(100 * p)}"


class PITCalibrationSummary(BaseModel):
    """PIT / NPDE-lite predictive calibration (policy 0.4.2 Gate 1 metric).

    Replaces :class:`VPCSummary` as the Gate 1 gated calibration check.
    For each observation ``j`` and each probability level ``p`` in
    ``probability_levels`` (default 0.05 / 0.50 / 0.95):

    1. Compute the simulated predictive ``p``-quantile at that observation
       from the per-subject ``(n_sims, n_obs_i)`` matrix:
       ``q_p(j) = quantile_p({y_sim[s, j] for s in n_sims})``.
    2. Evaluate the CDF indicator:
       ``I_p(j) = 1[y_obs[j] <= q_p(j)]``.
    3. Aggregate subject-robustly: per-subject mean of ``I_p``, then mean
       across subjects. ``calibration[f"p{int(p*100)}"] ≈ p`` when the
       predictive distribution is well-calibrated at level ``p``.

    The Gate 1 check asks ``|calibration[f"p{int(p*100)}"] - p| <= tol``
    with lane-specific tail vs. median tolerances
    (``Gate1Config.pit_tol_tail`` / ``pit_tol_median``). Unlike the
    prior bin-level VPC check, the denominator is ``n_subjects`` (inner
    mean) then averaged across subjects — no dependency on a bin grid,
    no 1/n_bins quantization artifact on sparse real-data designs.

    Rationale and design choice log lives in the rc9 follow-up CHANGELOG
    entry "PIT/NPDE-lite Gate 1 calibration".
    """

    probability_levels: list[float] = Field(
        default_factory=lambda: [0.05, 0.50, 0.95],
        description="p values at which the predictive CDF is evaluated; "
        "each must be in the open unit interval.",
    )
    calibration: dict[str, float] = Field(
        description="Empirical CDF hit-rate at each probability level. "
        "Keys ``pNN`` where ``NN = int(100 * p)``. Values in ``[0, 1]``.",
    )
    # ``n_observations`` / ``n_subjects`` report the actual counts that
    # contributed to calibration — including zero when the PIT path ran
    # but no subject produced a finite (obs, sim) pair. Use ge=0 so the
    # audit record preserves that signal instead of forcing a phantom 1.
    n_observations: int = Field(ge=0)
    n_subjects: int = Field(ge=0)
    aggregation: Literal["subject_robust", "pooled"] = "subject_robust"

    @model_validator(mode="after")
    def _keys_match_levels(self) -> PITCalibrationSummary:
        expected = {_pit_key(p) for p in self.probability_levels}
        actual = set(self.calibration.keys())
        if expected != actual:
            msg = (
                f"calibration keys {actual} must match probability_levels "
                f"{self.probability_levels} (expected {expected})"
            )
            raise ValueError(msg)
        for p in self.probability_levels:
            if not 0.0 < p < 1.0:
                msg = f"probability_levels must be in (0, 1), got {p}"
                raise ValueError(msg)
        for key, val in self.calibration.items():
            if not 0.0 <= val <= 1.0:
                msg = f"calibration[{key}]={val} not in [0, 1]"
                raise ValueError(msg)
        return self


class IdentifiabilityFlags(BaseModel):
    """Practical identifiability assessment."""

    condition_number: float | None = None
    profile_likelihood_ci: dict[str, bool]
    ill_conditioned: bool


class BLQHandling(BaseModel):
    """Below-limit-of-quantification handling method.

    Method values correspond to Beal (2001) nomenclature, preserved across
    the policy→bundle boundary via
    :func:`apmode.data.missing_data.normalize_blq_method_for_bundle`:

    - ``"none"`` — no BLQ handling (data has no BLQ records).
    - ``"m1"`` — discard BLQ records (biased; retained for benchmark/legacy).
    - ``"m3"`` — likelihood-based censoring (Beal 2001 Method 3).
    - ``"m4"`` — likelihood-based censoring + positivity constraint.
    - ``"m6_plus"`` — substitute BLQ with LLOQ/2 for first BLQ per subject,
      drop rest (Beal 2001 Method 6+).
    - ``"m7_plus"`` — substitute BLQ with 0 + inflated additive error
      (Wijk 2025; preferred when BLQ burden is low).

    ``method`` must be consistent across survivors entering the same Gate 3
    ranking — mixing (e.g.) ``"m3"`` and ``"m7_plus"`` forces simulation-based
    ranking because likelihood scales are not comparable (PRD §10 Q2, routed
    by :func:`apmode.governance.ranking.ranking_requires_simulation_metrics`).
    """

    method: Literal["none", "m1", "m3", "m4", "m6_plus", "m7_plus"]
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


class ScoringContract(BaseModel):
    """Per-candidate NLPD scoring contract (plan §3, PRD §4.3.1, §10 Q2).

    Gate 3 must not compose scores across contracts that are not
    mathematically comparable (conditional vs marginal NLPD,
    integrated-Laplace vs HMC, float32 vs float64 accumulation).
    The contract is emitted into :class:`DiagnosticBundle` by every
    backend runner; :mod:`apmode.governance.ranking` groups survivors
    by exact-equality on this object and emits one leaderboard per
    contract class — never a mixed composite.

    The Submission lane additionally restricts ``recommended``
    candidates to ``re_treatment == "integrated"`` and
    ``nlpd_kind == "marginal"`` (see ranking.submission_dominance_rule).

    ``contract_version`` is bumped when fields are added or redefined so
    that older bundles remain readable through explicit migration.

    Migration path:

    * **v0.5.0 → v0.5.1**: Laplace-based NODE RE lands in M3. If M3
      introduces *new* ``nlpd_integrator`` values (e.g., ``"laplace_elbo"``
      from a variational-inference follow-on) or a new ``re_treatment``
      value, bump ``contract_version`` to 2 and extend the relevant
      Literals. The DiagnosticBundle default factory stays on contract 1
      so old bundles continue to deserialize; a `migrate_contract_v1_to_v2`
      helper in this module is the right place to route such changes.
    * **Field removal** is a breaking change — never remove a Literal
      value even if a backend stops using it; mark it deprecated in the
      docstring and let it live.
    """

    model_config = ConfigDict(frozen=True)

    contract_version: Literal[1] = 1
    nlpd_kind: Literal["conditional", "marginal"]
    re_treatment: Literal["integrated", "conditional_ebe", "pooled"]
    nlpd_integrator: Literal[
        "nlmixr2_focei",
        "laplace_blockdiag",
        "laplace_diag",
        "hmc_nuts",
        "none",
    ]
    blq_method: Literal["none", "m1", "m3", "m4", "m6_plus", "m7_plus"]
    observation_model: Literal["additive", "proportional", "combined"]
    float_precision: Literal["float32", "float64"]


class DiagnosticBundle(BaseModel):
    """All diagnostic outputs for a candidate model.

    **Intentionally mutable** (no ``frozen=True``). The entire backend-
    result family — :class:`GOFMetrics`, :class:`IdentifiabilityFlags`,
    :class:`BLQHandling`, :class:`SplitGOFMetrics`, :class:`BackendResult`,
    and this class — is mutable because results get populated in stages
    (convergence, diagnostics, then later post-hoc fields like
    ``npe_score`` after posterior-predictive simulations land). Bundle
    classes representing fixed inputs (``EvidenceManifest``,
    ``NonlinearClearanceSignal``, ``MissingDataDirective``) are
    ``frozen=True`` — that asymmetry is by design, not an oversight.

    ``npe_score`` is the canonical simulation-based Nonparametric
    Prediction Error (median absolute prediction error from posterior
    predictive simulations; see :func:`apmode.benchmarks.scoring.compute_npe`).
    Backends that generate VPC / posterior-predictive simulations must
    populate this field using the benchmarks helper so every site in
    APMODE consumes the *same* NPE definition. When ``npe_score`` is
    ``None`` the ranking layer falls back to a documented CWRES proxy
    (:func:`apmode.governance.ranking.compute_cwres_npe_proxy`) —
    never silently redefining NPE.

    ``auc_cmax_be_score`` is the per-subject AUC/Cmax bioequivalence rate:
    fraction of subjects whose geometric mean ratios (candidate sim ÷ NCA
    reference) fall in [0.80, 1.25] for *both* AUC and Cmax (see
    :func:`apmode.benchmarks.scoring.compute_auc_cmax_be_score`). Populated
    from the same posterior-predictive simulation matrix as ``npe_score``
    when NCA eligibility holds (adequate absorption/elimination coverage,
    BLQ burden below policy threshold). ``None`` otherwise — the ranking
    layer drops the ``auc_cmax`` component uniformly across candidates
    rather than using a candidate-derived fallback (avoids circular bias).
    ``auc_cmax_source`` records provenance so reports can surface *why* the
    score was computed or skipped.
    """

    gof: GOFMetrics
    split_gof: SplitGOFMetrics | None = None
    vpc: VPCSummary | None = None
    pit_calibration: PITCalibrationSummary | None = None
    identifiability: IdentifiabilityFlags
    blq: BLQHandling
    # Non-negative by construction: NPE is an absolute-error summary.
    # Backends that can't compute it must leave this None and let the
    # ranking layer fall back to the documented CWRES proxy.
    npe_score: float | None = Field(default=None, ge=0.0)
    # AUC/Cmax bioequivalence fraction in [0, 1]. ``None`` when the NCA
    # reference is ineligible (policy-gated) or when the backend has not
    # yet produced posterior-predictive simulations. See class docstring.
    auc_cmax_be_score: float | None = Field(default=None, ge=0.0, le=1.0)
    auc_cmax_source: Literal["observed_trapezoid"] | None = None
    diagnostic_plots: dict[str, str] = Field(default_factory=dict)
    scoring_contract: ScoringContract = Field(
        default_factory=lambda: ScoringContract(
            nlpd_kind="marginal",
            re_treatment="integrated",
            nlpd_integrator="nlmixr2_focei",
            blq_method="none",
            observation_model="combined",
            float_precision="float64",
        )
    )


# --- BackendResult ---


class PosteriorDiagnostics(BaseModel):
    """MCMC convergence and reliability diagnostics.

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

    # Per-parameter-class diagnostics (plan Task 17). Keys are
    # ``{"fixed_effects", "iiv", "residual", "correlations"}``; values
    # are the worst-case R-hat / minimum bulk / tail ESS within each
    # class. Defaults to ``{}`` so older bundles remain loadable; the
    # Gate 1 Bayesian evaluator treats an empty dict as "no per-class
    # information, fall back to the scalar fields".
    rhat_max_by_class: dict[str, float] = Field(default_factory=dict)
    ess_bulk_min_by_class: dict[str, float] = Field(default_factory=dict)
    ess_tail_min_by_class: dict[str, float] = Field(default_factory=dict)


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
    prior_manifest_path: str | None = None  # bundle-relative path to prior_manifest.json
    simulation_protocol_path: str | None = None  # bundle-relative path to simulation_protocol.json
    # Sidecar artefacts produced by the Bayesian harness alongside the
    # main result. Embedded inline rather than persisted as separate
    # files in the harness output dir so the orchestrator owns the
    # bundle-write step (single emitter call site, single audit trail).
    # ``loo_summary`` is None when the model does not declare ``log_lik``
    # or arviz could not compute LOO; ``reparameterization_recommendation``
    # is None when the sampler ran cleanly.
    loo_summary: LOOSummary | None = None
    reparameterization_recommendation: ReparameterizationRecommendation | None = None
    # Gate 2 prior diagnostics (plan Tasks 20 + 21). Both default to
    # ``None`` so MLE backends and Bayesian runs without informative
    # priors carry no extra payload; the orchestrator decides whether
    # to emit ``status="not_computed"`` artefacts when the lane policy
    # demands a recorded skip vs an actual computation.
    prior_data_conflict: PriorDataConflict | None = None
    prior_sensitivity: PriorSensitivity | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_predicted_simulations_field(cls, data: object) -> object:
        """Guard against schema drift with the out-of-band `predicted_simulations` carrier.

        Backends that produce posterior-predictive draws ship them as an
        out-of-band ``predicted_simulations`` key on the raw harness
        response dict (see ``Nlmixr2Runner._parse_response`` and
        ``r/harness.R::.extract_backend_result``). The runner pops that
        key *before* validating into :class:`BackendResult`. If a future
        schema migration accidentally adds ``predicted_simulations`` as a
        real field here, the out-of-band pattern would silently route
        sims into a Pydantic-owned slot instead of the helper path.

        Raise loudly here so that drift is a loud test failure rather
        than a silent audit-trail inconsistency. No-op for every
        well-formed payload today.
        """
        if isinstance(data, dict) and "predicted_simulations" in data:
            msg = (
                "BackendResult payload contains 'predicted_simulations' — this key "
                "must be stripped by the runner before validation (out-of-band "
                "carrier pattern; see apmode.backends.nlmixr2_runner._parse_response). "
                "If this field is now schema-owned, remove this validator."
            )
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def validate_backend_scoring_contract_consistency(self) -> BackendResult:
        """Enforce backend ↔ scoring_contract.nlpd_integrator consistency.

        The default_factory on :class:`DiagnosticBundle.scoring_contract`
        emits the nlmixr2-FOCEI contract. If a Stan or NODE runner forgets
        to call ``attach_scoring_contract``, the bundle silently carries
        the wrong contract and Gate 3 groups it with classical candidates.

        This validator rejects contract/backend pairs that can never be
        correct. Allowed combinations:

        * nlmixr2       → nlpd_integrator == "nlmixr2_focei"
        * bayesian_stan → nlpd_integrator == "hmc_nuts"
        * jax_node      → nlpd_integrator in {"none", "laplace_diag",
          "laplace_blockdiag"}
        * agentic_llm   → any integrator (inherits from inner runner)

        The nlmixr2-FOCEI default is allowed for ``backend == "nlmixr2"``;
        for other backends the default is an active error, which is
        exactly what we want.
        """
        integrator = self.diagnostics.scoring_contract.nlpd_integrator
        allowed: dict[str, set[str]] = {
            "nlmixr2": {"nlmixr2_focei"},
            "bayesian_stan": {"hmc_nuts"},
            "jax_node": {"none", "laplace_diag", "laplace_blockdiag"},
            "agentic_llm": {
                "nlmixr2_focei",
                "hmc_nuts",
                "none",
                "laplace_diag",
                "laplace_blockdiag",
            },
        }
        valid = allowed.get(self.backend, set())
        if integrator not in valid:
            msg = (
                f"ScoringContract.nlpd_integrator={integrator!r} is not valid "
                f"for backend={self.backend!r}. Allowed: {sorted(valid)}. "
                f"Runner likely forgot to call "
                f"apmode.bundle.scoring_contract.attach_scoring_contract "
                f"before emitting the result. See plan §3 "
                f"(.plans/v0.5.0_limitations_closure.md)."
            )
            raise ValueError(msg)
        return self


# --- Data Manifest ---


class ColumnMapping(BaseModel):
    """Maps source columns to canonical schema fields.

    Each field name is a semantic key (snake_case), and each value is
    the canonical NONMEM column name (UPPERCASE).  Use ``to_canonical``
    and ``to_semantic`` for bidirectional lookup.
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


CovariateStrategy = Literal[
    "impute-median",
    "impute-LOCF",
    "full-information",
    "exclude",
    "MI-PMM",
    "MI-missRanger",
    "FREM",
    "FFEM",
]
"""Valid covariate-missingness handling strategies.

The first four are legacy/simple strategies; the last four correspond to the
policy-driven missing-data directive (see MissingDataDirective and
``apmode.data.missing_data.resolve_directive``).
"""


class CovariateSpec(BaseModel):
    """Covariate missingness specification (profile-time signal).

    ``strategy`` is an advisory hint from the profiler. The binding dispatch
    decision is ``MissingDataDirective`` emitted by the router after policy
    resolution.
    """

    pattern: Literal["MCAR", "MAR", "informative-suspected", "MNAR"]
    fraction_incomplete: float = Field(ge=0.0, le=1.0)
    strategy: CovariateStrategy


ErrorModelPrimary = Literal["proportional", "additive", "combined", "blq_m3", "blq_m4"]


class ErrorModelPreference(BaseModel):
    """Profiler-stage signal for admissible residual-error models.

    Consumed by ``SearchSpace.from_manifest`` to prune candidate generation
    before the search runs. Narrows the error_types cross-product when the
    data strongly suggests a particular structure (e.g., BLQ ≥ 10% → M3
    with proportional/combined underlying, never additive-only).

    References:
      - Beal (2001) "Ways to Fit a PK Model with Some Data Below the
        Quantification Limit" J Pharmacokin Pharmacodyn 28:481.
      - Ahn et al. (2008) "Likelihood based approaches to handling data
        below the quantification limit" J Pharmacokin Pharmacodyn 35:401.
    """

    model_config = ConfigDict(frozen=True)

    primary: ErrorModelPrimary
    # Allowed underlying error families in candidate generation. For BLQ
    # primaries this is the set composed with the BLQ observation model;
    # for non-BLQ primaries this is the set of error_types used directly.
    allowed: list[Literal["proportional", "additive", "combined"]] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"] = "medium"
    rationale: str = ""


class EvidenceManifest(BaseModel):
    """evidence_manifest.json — Data Profiler output (PRD §4.2.1)."""

    model_config = ConfigDict(frozen=True)

    # Schema version. Bumped when fields are added/removed/changed
    # semantically. Lane Router fails fast on an unknown version.
    # v3 = structured nonlinear-clearance signal provenance: the scattered
    # scalar fields (curvature_ratio_median/_ci90_*, terminal_fit_adj_r2_
    # median, dose_proportionality_*, signal_eligibility_mask, signal_votes)
    # collapse into ``nonlinear_clearance_signals: dict[SignalId, NonlinearClearanceSignal]``
    # carrying per-signal algorithm / citation / policy_key / threshold /
    # value+CI / eligibility reason / vote. Enables ICH M15 traceability.
    manifest_schema_version: int = 3
    data_sha256: HexSHA256 | None = None
    route_certainty: Literal["confirmed", "inferred", "ambiguous"]
    absorption_complexity: Literal["simple", "multi-phase", "lag-signature", "unknown"]
    # Graded evidence strength for nonlinear clearance (PRD §10 Q2 follow-up).
    # Multi-signal voting per the warfarin false-positive analysis :
    # strong   = all 3 independent signals fire (curvature ratio,
    # terminal-monoexp failure, dose-AUC nonproportionality)
    # moderate = 2 signals fire — search engine adds MM as sentinel only
    # weak     = 1 signal fires
    # none     = no signal triggered
    # Replaces the legacy boolean ``nonlinear_clearance_signature`` field
    # which produced false positives on multi-cmt linear drugs (warfarin:
    # 0/24 candidate convergence).
    nonlinear_clearance_evidence_strength: Literal["none", "weak", "moderate", "strong"]
    nonlinear_clearance_confidence: float | None = None
    # True when a subject has ≥2 dose events (EVID==1, AMT>0). Drives the
    # dose-interval slicing path used by the terminal-monoexp R² and peak-
    # detection helpers — without slicing, multi-dose data destroys the
    # monoexponential terminal phase (warfarin: terminal R² = -0.49 across
    # full profile; ~0.95 within the last dosing interval).
    multi_dose_detected: bool = False
    # Compartmentality signal — derived from biexponential vs monoexponential
    # BIC on pooled post-Cmax data. ``insufficient`` when post-absorption
    # sample count is too low for reliable model discrimination.
    compartmentality: (
        Literal["one_compartment", "two_compartment", "multi_compartment_likely", "insufficient"]
        | None
    ) = None
    # Continuous diagnostic signals not tied to the NLC voting pipeline.
    # (Voting-related diagnostics — terminal monoexp R², curvature ratio,
    # Smith 2000 beta — now live inside ``nonlinear_clearance_signals`` with
    # full provenance; see SignalId enum.)
    auc_extrap_fraction_median: float | None = Field(default=None, ge=0.0, le=1.0)
    lambda_z_analyzable_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    peak_prominence_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    richness_category: Literal["sparse", "moderate", "rich"]
    identifiability_ceiling: Literal["low", "medium", "high"]
    covariate_burden: int = Field(ge=0)
    covariate_correlated: bool
    covariate_missingness: CovariateSpec | None = None
    # True when any covariate changes value within a subject over time.
    # Drives FREM preference in the missing-data directive (Nyberg 2024,
    # Jonsson 2024). Defaults to False for backwards compatibility.
    time_varying_covariates: bool = False
    blq_burden: float = Field(ge=0.0, le=1.0)
    lloq_value: float | None = Field(default=None, ge=0.0)
    protocol_heterogeneity: Literal["single-study", "pooled-similar", "pooled-heterogeneous"]
    absorption_phase_coverage: Literal["adequate", "inadequate"]
    elimination_phase_coverage: Literal["adequate", "inadequate"]
    # Profiler-stage signals informing the error-model heuristic. None for
    # manifests emitted before the heuristic was introduced.
    error_model_preference: ErrorModelPreference | None = None
    cmax_p95_p05_ratio: float | None = Field(default=None, ge=0.0)
    dv_cv_percent: float | None = Field(default=None, ge=0.0)
    terminal_log_residual_mad: float | None = Field(default=None, ge=0.0)

    # ---- Structured nonlinear-clearance signal provenance (v3) ----

    # One record per signal: algorithm, citation, policy_key (JSON pointer
    # into policies/profiler.json), threshold, observed value + 90% CI,
    # eligibility bool with human-readable reason when ineligible, and
    # whether it voted for nonlinearity. Keyed by SignalId for O(1) access
    # by Lane Router and deterministic JSON ordering.
    nonlinear_clearance_signals: dict[SignalId, NonlinearClearanceSignal] = Field(
        default_factory=dict
    )
    # R6 + R10: Huang 2025 best-lambda_z window stats for auditability.
    lambda_z_used_points_median: float | None = Field(default=None, ge=0.0)
    lambda_z_adj_r2_median: float | None = Field(default=None, ge=-1.0, le=1.0)
    # R14 (Richardson 2025): flip-flop kinetics risk for oral profiles.
    # Drives whether the structural search seeds both ka>ke and ka<ke.
    flip_flop_risk: Literal["none", "possible", "likely", "unknown"] = "unknown"
    # R15 (Wagner-Nelson 1963): population-median ka from Wagner-Nelson
    # method on single-dose oral subjects. Used as initial-estimate seed.
    wagner_nelson_ka_median: float | None = Field(default=None, ge=0.0)
    # R12 (Pharmpy AMD design feasibility): NODE input-dim budget per
    # PRD §4.2.4 R6 — 0 disables NODE backend; 4 = Optimization lane;
    # 8 = Discovery lane.
    node_dim_budget: int = Field(default=0, ge=0, le=8)
    # R19: TIME/TAD consistency for multi-dose datasets. ``contaminated``
    # = multi-dose subjects exist but TIME is not aligned to dose events;
    # downstream shape-based heuristics should be down-weighted.
    tad_consistency_flag: Literal["clean", "contaminated", "unknown"] = "unknown"
    # Filter audit (DVID non-PK row removal at profile_data entry).
    n_non_pk_rows_dropped: int = Field(default=0, ge=0)


# --- Missing-Data Directive (router output, policy-resolved) ---


class MissingDataDirective(BaseModel):
    """Policy-resolved missing-data handling plan for a run.

    Produced by ``apmode.data.missing_data.resolve_directive(policy, manifest)``
    and attached to ``DispatchDecision``. Backends MUST honor this directive;
    it is not advisory.
    """

    model_config = ConfigDict(frozen=True)

    # Covariate missingness handling
    covariate_method: CovariateStrategy
    # Number of imputations for MI methods; None for FREM/FFEM/single-imputation.
    m_imputations: int | None = Field(default=None, ge=1)
    # Adaptive-m: start at m_imputations, escalate if between-imputation variance
    # exceeds policy threshold.
    adaptive_m: bool = False
    m_max: int | None = Field(default=None, ge=1)
    # BLQ (observation-side) handling: method selected by BLQ% threshold in policy.
    blq_method: Literal["M1", "M3", "M4", "M6+", "M7+"]
    # Whether the LLM (agentic backend) must receive pooled/stability summaries
    # only, never per-imputation diagnostics (cherry-picking mitigation).
    llm_pooled_only: bool = True
    # Rank-stability penalty weight applied in Gate 1 for agentic candidates
    # that flip ranking across imputations. 0.0 disables the penalty.
    imputation_stability_penalty: float = Field(default=0.0, ge=0.0)
    # Human-readable rationale (populated by resolver for audit trail).
    rationale: list[str] = Field(default_factory=list)


class ImputationStabilityEntry(BaseModel):
    """Per-candidate stability summary across m imputed datasets."""

    candidate_id: str
    # Pooled (Rubin-combined) criteria
    pooled_ofv: float | None = None
    pooled_aic: float | None = None
    pooled_bic: float | None = None
    # Fraction of imputations where this candidate converged
    convergence_rate: float = Field(ge=0.0, le=1.0)
    # Within-imputation variance / between-imputation variance of OFV
    # Low values (<1) indicate between-imputation variance dominates (unstable).
    within_between_var_ratio: float | None = None
    # Fraction of m imputations where this candidate's rank stayed within top-K.
    # K is run-level (e.g., top-3). 1.0 = unanimously ranked in top-K.
    rank_stability: float = Field(ge=0.0, le=1.0)
    # Directional consistency of key covariate effects across imputations
    # (fraction of imputations agreeing on sign of each covariate effect).
    covariate_sign_consistency: dict[str, float] = Field(default_factory=dict)
    # Rubin-pooled per-parameter estimates (Rubin 1987). Keyed by parameter
    # name; each value carries the canonical 5-tuple of quantities:
    # ``pooled_estimate``: Q̄, mean of per-imputation estimates
    # ``within_var``: Ū, mean of per-imputation sampling variances (SE²)
    # ``between_var``: B, sample variance of per-imputation estimates
    # ``total_var``: T = Ū + (1 + 1/m) * B
    # ``dof``: Barnard-Rubin degrees of freedom for inference
    # Empty when the backend did not emit per-parameter (estimate, SE) on
    # converged imputations.
    pooled_parameters: dict[str, dict[str, float]] = Field(default_factory=dict)


class ImputationStabilityManifest(BaseModel):
    """imputation_stability.json — emitted per run when MI is active.

    This is the ONLY per-imputation artifact the agentic LLM backend is
    permitted to see (via the diagnostic summarizer). Raw per-imputation
    diagnostics are withheld to prevent cherry-picking.
    """

    model_config = ConfigDict(frozen=True)

    m: int = Field(ge=1)
    method: CovariateStrategy
    top_k: int = Field(default=3, ge=1)
    entries: list[ImputationStabilityEntry] = Field(default_factory=list)
    # Documented limitations for Gate 2.5 credibility ingestion.
    omega_pooling_caveats: list[str] = Field(default_factory=list)


# --- Categorical encoding provenance ---


class CategoricalEncodingEntry(BaseModel):
    """One column's encoding decision and remap, captured for audit."""

    column: str
    detected_encoding: Literal[
        "binary_zero_one",
        "binary_one_two",
        "binary_string_pair",
        "binary_boolean",
        "multi_level",
        "continuous",
        "constant",
        "all_missing",
    ]
    # Stringified for serialization stability — original values can be
    # mixed types (str, int, bool) and JSON does not distinguish them.
    unique_values: list[str] = Field(default_factory=list)
    # Maps stringified original → 0/1 target. Empty when no remap was
    # produced (already canonical, multi-level, continuous, etc.).
    applied_remap: dict[str, int] = Field(default_factory=dict)
    applied: bool = False
    # ``"auto"`` when the auto-detector chose the polarity, ``"override"``
    # when the caller supplied an explicit remap dict.
    source: Literal["auto", "override", "no_remap"] = "no_remap"
    rationale: str


class CategoricalEncodingProvenance(BaseModel):
    """categorical_encoding_provenance.json — bundle artifact.

    Records every column inspected by the auto-encoding pipeline + the
    polarity actually applied. Lets reviewers trace exactly how a raw
    "male"/"female" column became 0/1 in the FREM joint Ω entry — the
    safety guarantee that protects against silent NaN-on-incomplete-remap
    and inconsistent summarize-vs-prepare polarity bugs in the
    auto-detection pass (PRD §4.2.0).
    """

    model_config = ConfigDict(frozen=True)

    entries: list[CategoricalEncodingEntry] = Field(default_factory=list)


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


class NCASubjectDiagnostic(BaseModel):
    """Per-subject NCA QC record (one row in nca_diagnostics.jsonl).

    Captures the outcome of the PKNCA-style per-subject analysis: terminal
    curve-stripping metrics, AUC extrapolation, and whether the subject
    survives QC gates (adj_r²≥0.80, extrap≤20%, span≥1·t½, ≥3 λz points).

    ``excluded=True`` means the subject's NCA result did not enter the
    population median used as initial estimates; ``excluded_reason`` names
    the failing gate(s). Fields are Optional because subjects that fail
    very early (e.g., <3 observations) cannot populate downstream values.
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str
    tmax: float | None = None
    cmax: float | None = None
    cl: float | None = None
    v: float | None = None
    ka: float | None = None
    kel: float | None = None
    auc_last: float | None = None
    auc_inf: float | None = None
    auc_extrap_fraction: float | None = None
    lambda_z_adj_r2: float | None = None
    lambda_z_n_points: int | None = None
    span_ratio: float | None = None
    excluded: bool = False
    excluded_reason: str | None = None


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
    backend: Literal["nlmixr2", "jax_node", "agentic_llm", "bayesian_stan"] = "nlmixr2"
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
    # #19: direct pointer back to the BackendResult this credibility
    # report was synthesised from. Populated by the report generator so
    # an auditor can re-derive every field from the original bundle
    # entry. Both fields are optional for legacy bundles emitted before
    # the field existed; new code MUST populate them.
    source_result_path: str | None = None
    source_result_sha256: str | None = None


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
    Pooled variance uses the law of total variance (E[Var] + Var[E]).
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


class SBCPriorEntry(BaseModel):
    """One prior's Simulation-Based Calibration result (Talts et al. 2018).

    SBC tests whether posterior intervals from the inference algorithm are
    correctly calibrated to the prior: under exact inference, ranks of the
    simulated true parameter relative to the posterior draws are uniform.
    Departure from uniformity (low KS p-value) flags miscalibration in
    either the model or the sampler.

    The nightly runner (plan Task 27) fills this entry; the per-run
    orchestrator only emits a stub manifest (``priors: []``) so the
    artefact is always present in the bundle.
    """

    model_config = ConfigDict(frozen=True)

    target: str
    family: str
    n_simulations: int = Field(ge=1)
    rank_histogram_bins: int = Field(ge=4)
    rank_histogram_counts: list[int]
    ks_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool

    @model_validator(mode="after")
    def histogram_matches_bins(self) -> Self:
        if len(self.rank_histogram_counts) != self.rank_histogram_bins:
            msg = (
                f"rank_histogram_counts length {len(self.rank_histogram_counts)} "
                f"does not match rank_histogram_bins {self.rank_histogram_bins}"
            )
            raise ValueError(msg)
        if any(c < 0 for c in self.rank_histogram_counts):
            raise ValueError("rank_histogram_counts must be non-negative")
        return self


class SBCManifest(BaseModel):
    """sbc_manifest.json — Simulation-Based Calibration roll-up (plan Task 26).

    Always emitted on Bayesian runs as a stub with ``priors=[]`` so the
    artefact's *presence* signals the orchestrator did its job. The
    nightly SBC runner (Task 27) populates the entries from a canonical
    3-scenario set (1-cmt, 2-cmt, 2-cmt with IIV correlation).

    The file lives under ``artifacts/sbc/manifest.json`` — outside the
    sealed-digest scope (added to ``_DIGEST_EXCLUDED_NAMES``) so the
    nightly runner can rewrite it without invalidating ``_COMPLETE``.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    sbc_runner_commit: str = ""
    priors: list[SBCPriorEntry] = Field(default_factory=list)
    generated_at: str  # ISO-8601 UTC


class LOOSummary(BaseModel):
    """PSIS-LOO summary (plan Task 18, Vehtari et al. 2017).

    Captured under ``bayesian/{cid}_loo_summary.json``. ``status``
    distinguishes a real summary from a graceful skip when the Stan
    program does not declare a per-observation ``log_lik`` — small
    models legitimately ship without it, and Gate 1 Bayesian treats
    that case as a warning, not a failure.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    status: Literal["computed", "not_computed"]
    elpd_loo: float | None = None
    se_elpd_loo: float | None = None
    p_loo: float | None = None
    pareto_k_max: float | None = None
    n_observations: int | None = Field(default=None, ge=0)
    # Counts of observations whose Pareto-k landed in each band, per
    # ``arviz.loo`` documentation (Vehtari et al. 2017): good <= 0.5,
    # ok in (0.5, 0.7], bad in (0.7, 1.0], very_bad > 1.0. Used by
    # Gate 1 Bayesian and the report generator for at-a-glance reliability.
    k_counts: dict[str, int] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def k_counts_keys_are_known(self) -> Self:
        # The four Vehtari 2017 reliability bands are the only valid
        # keys; unknown labels (typos like "god" → silently lost in
        # downstream report rendering) must surface at construction time.
        expected = {"good", "ok", "bad", "very_bad"}
        unknown = sorted(set(self.k_counts) - expected)
        if unknown:
            msg = (
                f"LOOSummary.k_counts has unknown band labels {unknown}; "
                f"expected subset of {sorted(expected)}"
            )
            raise ValueError(msg)
        return self


class ReparameterizationRecommendation(BaseModel):
    """Diagnostic-driven reparameterization suggestion (advisory, not automatic).

    APMODE never switches a sampler's parameterization silently — per PRD
    §4.3.2 and consensus during multi-model review, auto-switching between
    centered and non-centered parameterizations masks model pathology and
    desynchronises the audit trail from the run. Instead the harness
    inspects the post-warmup divergence and tree-depth counts and emits
    this artifact when Gate 1 Bayesian would flag the fit.

    ``recommended_action`` is one of:

    * ``switch_to_non_centered`` — divergent-transition rate above the
      harness threshold (default 5% of post-warmup iterations). Non-
      centered parameterization typically clears funnel geometries
      (Betancourt & Girolami 2015).
    * ``refit_with_higher_adapt_delta`` — low-rate divergences or tree-
      depth saturation. Tightening the step-size adaptation is cheaper
      than restructuring the model.
    * ``add_jacobian`` — explicit Jacobian adjustment for non-linear
      reparameterization (reserved; emitted only by agentic-LLM
      transforms that know they've introduced a change-of-variable).

    The Gate 1 Bayesian check (plan Task 17) routes operators to this
    file when surfacing a divergence failure.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    divergence_count: int = Field(ge=0)
    divergence_fraction: float = Field(ge=0.0, le=1.0)
    max_treedepth_count: int = Field(ge=0, default=0)
    recommended_action: Literal[
        "switch_to_non_centered",
        "add_jacobian",
        "refit_with_higher_adapt_delta",
    ]
    rationale: str


_PRIOR_DOI_PATTERN = re.compile(
    r"^10\.\d{4,9}/[-._;()/:A-Z0-9<>\[\]]+$",
    re.IGNORECASE,
)


class PriorManifestEntry(BaseModel):
    """One declared prior with full provenance for FDA Gate 2 justification.

    The optional :attr:`doi` is enforced to match the Crossref canonical
    pattern when present — a free-form ``"n/a"`` would silently satisfy
    the Gate 2 prior-justification check that only tests truthiness.
    """

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
    doi: str | None = None

    @model_validator(mode="after")
    def doi_is_crossref_canonical(self) -> Self:
        if self.doi is not None and not _PRIOR_DOI_PATTERN.match(self.doi):
            msg = (
                f"PriorManifestEntry.doi {self.doi!r} does not match the "
                "Crossref-canonical pattern '10.<registrant>/<suffix>'. "
                "Use ``None`` rather than 'n/a' when no DOI is available."
            )
            raise ValueError(msg)
        return self


class PriorManifest(BaseModel):
    """prior_manifest.json — versioned record of all declared priors.

    FDA-required prior justification artifact. Every prior on a
    non-uninformative/weakly-informative source must carry a non-empty
    justification; historical_data must also carry historical_refs.
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str
    entries: list[PriorManifestEntry]
    default_prior_policy: Literal["weakly_informative", "custom"] = "weakly_informative"


class MetricTuple(BaseModel):
    """Commensurate mean + 90/95% interval carrier for Gate 3 ranking.

    Populated by the Bayesian backend directly from ``posterior_draws.parquet``
    or by the MLE path via :func:`apmode.governance.approximate_posterior.laplace_draws`
    (falling back to ``empirical_bootstrap`` when the asymptotic
    covariance is ill-conditioned). The ``method`` tag is carried onto
    disk so reviewers can tell whether the interval comes from real
    posterior draws or the Laplace approximation — the two have
    different credibility.

    Plan Task 23 / Task 22. Gate 3 ranker (``gate3_metric_stack``,
    Task 24) groups candidates by scoring contract and within each
    group consumes the interval to compute Borda rank or the weighted
    composite.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    ci_low: float
    ci_high: float
    method: Literal[
        # Real posterior samples from MCMC (Bayesian backend).
        "posterior_draws",
        # Generic Laplace tag — kept for back-compat with bundles that
        # don't distinguish faithful MVN from the diagonal fallback.
        "laplace_draws",
        # Faithful MVN draw from the asymptotic covariance (full
        # off-diagonal correlation structure preserved).
        "laplace_mvn",
        # Diagonal-only fallback when the asymptotic covariance is
        # ill-conditioned. Marginal SE is preserved but correlations
        # are lost — Gate 3 ranker should weight these intervals lower.
        "laplace_bootstrap_diagonal",
        # Legacy synonym for the diagonal fallback.
        "empirical_bootstrap",
    ]
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def ci_is_ordered(self) -> Self:
        if not (self.ci_low <= self.mean <= self.ci_high):
            msg = (
                f"MetricTuple must satisfy ci_low <= mean <= ci_high; "
                f"got ({self.ci_low}, {self.mean}, {self.ci_high})"
            )
            raise ValueError(msg)
        return self


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
    """simulation_protocol.json — prospective-simulation specification.

    Required for FDA 2026 operating-characteristics evaluation. Locked before
    Gate 3 execution to prevent metric shopping.
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


class PriorDataConflictEntry(BaseModel):
    """One T(y) statistic compared against the prior-predictive distribution.

    The prior 95% PI is the (2.5%, 97.5%) quantile of T(y_rep) where each
    y_rep is drawn from the *prior* predictive distribution (no data
    conditioning). ``in_pi`` is False when the observed value falls
    outside this interval — a marginal sign that the prior excludes the
    observed regime (Box 1980, Evans & Moshonov 2006).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    observed: float
    prior_pi_low: float
    prior_pi_high: float
    in_pi: bool
    prior_pred_mean: float | None = None
    prior_pred_sd: float | None = None

    @model_validator(mode="after")
    def in_pi_matches_bounds(self) -> Self:
        # Defensive: producers and consumers must agree on the band — if
        # ``in_pi`` is True but the observed value sits outside the
        # interval the artefact is internally inconsistent and the gate
        # would silently pass for the wrong reason.
        actually_in = self.prior_pi_low <= self.observed <= self.prior_pi_high
        if actually_in != self.in_pi:
            msg = (
                f"PriorDataConflictEntry({self.name!r}): in_pi={self.in_pi} "
                f"contradicts observed={self.observed} vs PI "
                f"[{self.prior_pi_low}, {self.prior_pi_high}]"
            )
            raise ValueError(msg)
        return self


class PriorDataConflict(BaseModel):
    """prior_data_conflict.json — Gate 2 prior-data conflict diagnostic (plan Task 20).

    Compares observed dataset summaries T(y) against the prior-predictive
    distribution induced by ``DSLSpec.priors`` via a fresh Stan
    ``generated_quantities`` pass with ``prior_only=true``. Gate 2 fails
    when ``conflict_fraction > prior_data_conflict_threshold`` so a
    candidate whose priors systematically exclude the observed data
    surface the violation in the audit trail (Evans & Moshonov 2006,
    Gabry et al. 2019 prior-predictive checking).

    ``status="not_computed"`` covers two legitimate cases:

    * Lane policy doesn't enable prior-data conflict (Discovery /
      Optimization defaults). The artefact is still emitted so the
      *absence of computation* is recorded, not the *absence of file*.
    * Bayesian extras (cmdstanpy) are unavailable in the runtime —
      typically a unit-test environment without Stan installed.

    The harness sets ``status="computed"`` only when it actually ran the
    prior-only sampling pass and computed the summaries.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    status: Literal["computed", "not_computed"]
    threshold: float = Field(ge=0.0, le=1.0)
    conflict_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    n_statistics_total: int = Field(default=0, ge=0)
    n_statistics_flagged: int = Field(default=0, ge=0)
    entries: list[PriorDataConflictEntry] = Field(default_factory=list)
    n_prior_predictive_draws: int | None = Field(default=None, ge=0)
    reason: str | None = None

    @model_validator(mode="after")
    def fields_consistent_with_status(self) -> Self:
        # When ``status="computed"`` the summary fields must be populated
        # so the gate can rule on them; when ``status="not_computed"`` we
        # require a reason so the audit trail explains the skip.
        if self.status == "computed":
            if self.conflict_fraction is None:
                msg = "PriorDataConflict(status='computed') requires conflict_fraction"
                raise ValueError(msg)
            if self.n_statistics_total != len(self.entries):
                msg = (
                    f"PriorDataConflict.n_statistics_total ({self.n_statistics_total}) "
                    f"must equal len(entries) ({len(self.entries)})"
                )
                raise ValueError(msg)
            actual_flagged = sum(1 for e in self.entries if not e.in_pi)
            if actual_flagged != self.n_statistics_flagged:
                msg = (
                    f"PriorDataConflict.n_statistics_flagged "
                    f"({self.n_statistics_flagged}) does not match the count of "
                    f"entries with in_pi=False ({actual_flagged})"
                )
                raise ValueError(msg)
        elif self.reason is None or not self.reason.strip():
            msg = "PriorDataConflict(status='not_computed') requires a non-empty reason"
            raise ValueError(msg)
        return self


class PriorSensitivityEntry(BaseModel):
    """One alternative-prior refit summary for a single structural parameter.

    ``alternative_id`` is a free-form tag identifying the perturbation
    (e.g. ``"sigma_x0.5"``, ``"sigma_x2.0"``). ``delta_normalized`` is
    ``|posterior_mean_alt - posterior_mean_baseline| / posterior_sd_baseline``
    — the unitless sensitivity score Gate 2 thresholds against
    ``sensitivity_max_delta`` (Roos et al. 2015, Kallioinen et al. 2024
    power-scaling sensitivity).
    """

    model_config = ConfigDict(frozen=True)

    parameter: str
    alternative_id: str
    posterior_mean_baseline: float
    posterior_mean_alternative: float
    posterior_sd_baseline: float = Field(gt=0.0)
    delta_normalized: float = Field(ge=0.0)

    @model_validator(mode="after")
    def delta_matches_components(self) -> Self:
        # Producers compute the score themselves so consumers don't have
        # to reproduce the denominator semantics — but cross-check the
        # arithmetic so a hand-edited artefact can't lie about the value.
        expected = (
            abs(self.posterior_mean_alternative - self.posterior_mean_baseline)
            / self.posterior_sd_baseline
        )
        if abs(expected - self.delta_normalized) > 1e-6:
            msg = (
                f"PriorSensitivityEntry({self.parameter!r}, "
                f"{self.alternative_id!r}): delta_normalized={self.delta_normalized} "
                f"does not match |Δmean|/sd_baseline={expected}"
            )
            raise ValueError(msg)
        return self


class PriorSensitivity(BaseModel):
    """prior_sensitivity.json — Gate 2 prior-sensitivity diagnostic (plan Task 21).

    For every structural parameter the harness refits the model under
    N≥2 alternative priors (typically scale perturbations of the
    informative components). Gate 2 fails when any
    ``delta_normalized > sensitivity_max_delta`` — informally, when the
    posterior moves by more than ``sensitivity_max_delta`` posterior SDs
    under a prior the regulator might equally have chosen.

    ``status="not_computed"`` records lane-policy disablement or the
    absence of cmdstanpy / arviz at runtime so reviewers see the skip
    explicitly. ``max_delta`` is the running maximum across entries —
    used by the gate so the per-entry list need not be re-scanned.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    status: Literal["computed", "not_computed"]
    threshold: float = Field(ge=0.0)
    max_delta: float | None = Field(default=None, ge=0.0)
    n_parameters: int = Field(default=0, ge=0)
    n_alternatives_per_parameter: int = Field(default=0, ge=0)
    entries: list[PriorSensitivityEntry] = Field(default_factory=list)
    flagged_parameters: list[str] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def fields_consistent_with_status(self) -> Self:
        if self.status == "computed":
            if self.max_delta is None:
                msg = "PriorSensitivity(status='computed') requires max_delta"
                raise ValueError(msg)
            if self.entries:
                actual_max = max(e.delta_normalized for e in self.entries)
                if abs(actual_max - self.max_delta) > 1e-6:
                    msg = (
                        f"PriorSensitivity.max_delta ({self.max_delta}) does not "
                        f"match max(entry.delta_normalized)={actual_max}"
                    )
                    raise ValueError(msg)
                actual_flagged = sorted(
                    {e.parameter for e in self.entries if e.delta_normalized > self.threshold}
                )
                if actual_flagged != sorted(self.flagged_parameters):
                    msg = (
                        f"PriorSensitivity.flagged_parameters "
                        f"({sorted(self.flagged_parameters)}) does not match the "
                        f"set of parameters with delta > threshold ({actual_flagged})"
                    )
                    raise ValueError(msg)
            elif self.max_delta != 0.0:
                msg = (
                    "PriorSensitivity(status='computed') with no entries must report "
                    f"max_delta=0.0 (got {self.max_delta})"
                )
                raise ValueError(msg)
        elif self.reason is None or not self.reason.strip():
            msg = "PriorSensitivity(status='not_computed') requires a non-empty reason"
            raise ValueError(msg)
        return self

    seed: int = 0
