# Phase 2+ — Bayesian backend (Stan/Torsten) integration

**Status:** Draft — 2026-04-14
**Authors:** Claude + gpt-5.2 second opinion (continuation_id c2981907-2af1-4d98-9eea-f1a0053851d8)
**PRD refs:** §3 (lanes), §4.2.5 (DSL), §4.3.1 (gates), §4.3.2 (bundle), §8 (phasing), §10 Q2 (NLPD comparability)
**License:** GPL-2-or-later. Stan/Torsten (BSD-3) compatible; Pumas excluded.

## 1. Motivation

- **FDA draft guidance, Jan 12 2026** (FDA-2025-D-3217) accepts Bayesian primary inference if prior justification is documented, operating characteristics are simulated, and success criteria are posterior-probability based.
- **Project Optimus final guidance** (Aug 2024) has shifted early-phase oncology trial design to 75% Bayesian (ASCO OA-25-00084).
- APMODE's Gate 3 already uses simulation-based metrics (VPC coverage, AUC/Cmax BE, NPE) — Bayesian produces these natively via posterior predictive draws, without the NLPD comparability problem (§10 Q2).
- `src/apmode/dsl/stan_emitter.py` is already substantial (ode_rk45, analytical superposition, BLQ M3/M4, log_lik). The DSL is Stan-ready; the missing piece is a runner and prior handling.

## 2. Scope

**In scope (this plan):**
- New backend `bayesian_stan` driven by `cmdstanpy` + Torsten.
- DSL extension: `PriorSpec` AST + `SetPrior` as the **7th** `FormularTransform`; priors as a new field on `DSLSpec`.
- `BackendResult` extensions for MCMC: `PosteriorDiagnostics`, `SamplerConfig`, quantile-extended `ParameterEstimate`.
- Bundle artifacts: `prior_manifest.json`, `posterior_draws.parquet`, `mcmc_diagnostics.json`, `ppc_summary.json`, `prior_sensitivity.json`, `simulation_protocol.json`.
- Gate 1 Bayesian technical checks; Gate 2 prior-justification artifact + prior-data conflict check; Gate 3 posterior-predictive simulation path.

**Out of scope (defer):**
- Bayesian NODE (HDCM) — separate Phase 2.5 plan.
- LORO-CV + precision dosing — Phase 3.
- Marginal NUTS (Pumas 2025 PAGE II-92 approach) — would require FOCE marginalization step; defer until core Bayesian path is stable.

## 3. Architectural decisions (with rationale)

### 3.1 New backend literal `bayesian_stan`

`BackendResult.backend` gains `"bayesian_stan"` alongside `"nlmixr2"`, `"jax_node"`, `"agentic_llm"`. Reasons:
- Different subprocess (cmdstanpy), different diagnostics vocabulary (R̂/ESS/divergences vs. gradient_norm), different Gate 1 checks.
- `ConvergenceMetadata.minimization_status` is MLE-shaped; conflating Bayesian into nlmixr2 forces `if backend=="nlmixr2" && method=="bayes"` branches everywhere.
- Keeps orchestration dispatch as a top-level Literal match.

### 3.2 Priors as a first-class DSL field, SetPrior as the 7th transform

`DSLSpec` gains:

```python
priors: list[PriorSpec] = Field(default_factory=list)
```

`PriorSpec` is a discriminated union (Normal, LogNormal, HalfNormal, HalfCauchy, Beta, Gamma, InvGamma, LKJ, Mixture, HistoricalBorrowing) with a `target` (param name or `omega_<param>`, `sigma_prop`, `sigma_add`, `corr_<axis>`) and `source` ∈ {uninformative, weakly_informative, historical_data, expert_elicitation, meta_analysis}.

`SetPrior` joins `FormularTransform` as the 7th variant. Reasons:
- Matches FDA's "prior is a distinct justification artifact" framing.
- Existing `nlmixr2_emitter` and NODE emitters are untouched — they ignore `priors` entirely.
- Prior-sensitivity sweeps become first-class policy artifacts (one DSLSpec, N priors, N fits).
- Agentic LLM can *elicit* priors through `SetPrior` under the DSL ceiling, inheriting the existing audit trail and escrow.

**Validation against a parameterization schema** (per gpt-5.2 compromise): each `PriorSpec` is validated against the parameter's required support. `omega_*` and `sigma_*` must take half-family priors (HalfCauchy/HalfNormal/Gamma/InvGamma); correlation matrices must take LKJ; structural log-scale params take Normal; covariate coefficients take Normal. Invalid pairs raise at compile time.

### 3.3 cmdstanpy direct, not R harness

- The emitter emits Stan, not rxode2-bayes DSL — routing through R adds translation with no benefit.
- Avoids `rstan`/`cmdstanr`/C++ toolchain coupling with the R container image.
- `reduce_sum` / MPI parallelization is first-class in cmdstanpy.
- `asyncio.create_subprocess_exec` pattern is identical to `Nlmixr2Runner` — we reuse `_spawn_r`-style plumbing.

**Concession to the R ecosystem:** draws are written to Arrow/Parquet so R-side post-processing (xpose, vpc, ggPMX) can consume them without reinvention.

### 3.4 BackendResult extensions

New sibling fields on `BackendResult` (all Optional; None for non-Bayesian backends):

```python
posterior_diagnostics: PosteriorDiagnostics | None = None
sampler_config: SamplerConfig | None = None
posterior_draws_path: str | None = None   # relative path to parquet in bundle
```

`PosteriorDiagnostics` carries: R̂_max, ESS_bulk_min, ESS_tail_min, n_divergent, n_max_treedepth, E_BFMI_min, MCSE for key parameters, Pareto-k_max and counts>{0.5,0.7,1.0} (for LOO).

`SamplerConfig` captures: chains, warmup, sampling, adapt_delta, max_treedepth, step_size_mean, mass_matrix_type, seed, cmdstan_version, torsten_version, stan_version, compiler_id, threads_per_chain.

`ParameterEstimate` extends with: posterior_mean, posterior_sd, q05, q50, q95 (Optional, populated only by Bayesian backend).

### 3.5 Bundle artifacts

| Artifact | Shape | Purpose |
|---|---|---|
| `prior_manifest.json` | `PriorManifest` — list of (target, family, hyperparams, source, justification, historical_refs) | FDA prior justification artifact |
| `posterior_draws.parquet` | long-form: chain, iter, param, value | first-class storage; Arrow interop |
| `mcmc_diagnostics.json` | `PosteriorDiagnostics` + per-chain summaries | Gate 1 evidence |
| `ppc_summary.json` | VPC coverage at time bins, BLQ-specific PPC if M3/M4 | Gate 2/3 evidence |
| `prior_sensitivity.json` | sweep: prior variant → posterior summary delta | sensitivity analysis |
| `simulation_protocol.json` | `SimulationProtocol` — scenarios, N, dropout, assay error, BLQ mechanism | prospective simulation (FDA requirement) |

### 3.6 Gate integration

**Gate 1 (technical validity) — Bayesian checks** (disqualifying):
- R̂ ≤ 1.01 for all parameters
- ESS_bulk ≥ 400 and ESS_tail ≥ 400 for all structural params
- n_divergent == 0
- n_max_treedepth ≤ 1% of post-warmup iterations
- E_BFMI_min ≥ 0.3
- Pareto-k_max ≤ 0.7 (LOO reliability)

**Gate 2 (Submission lane) — added requirements:**
- `prior_manifest.json` present with non-empty justification for every non-default prior
- Prior-data conflict check: prior predictive check against observed data summary statistics; if conflict p-value < 0.05 *and* source ∈ {historical_data, expert_elicitation, meta_analysis}, require explicit rationale
- Prior sensitivity sweep (≥2 alternative specifications) within lane policy tolerance

**Gate 3 (cross-paradigm ranking):**
- Bayesian: metrics computed as posterior expectations with credible intervals from draws
- MLE-based paradigms: same metrics via parametric bootstrap over MLE estimates + covariance
- Result: all paradigms emit `(metric_mean, metric_ci_low, metric_ci_high)` tuples — commensurate units
- **Predefine** the Gate 3 metric stack in lane policy to prevent metric shopping (per gpt-5.2 warning)

## 4. Implementation plan

### 4.1 Files to add

- `src/apmode/dsl/priors.py` — `PriorSpec` discriminated union, validator, parameterization schema
- `src/apmode/dsl/prior_transforms.py` — `SetPrior` transform, validation, application
- `src/apmode/backends/bayesian_runner.py` — `BayesianRunner` (cmdstanpy subprocess)
- `src/apmode/backends/bayesian_harness.py` — Python harness run as subprocess (mirrors R harness)
- `src/apmode/bundle/posterior_models.py` — `PosteriorDiagnostics`, `SamplerConfig`, `PriorManifest`, `SimulationProtocol`

### 4.2 Files to modify

- `src/apmode/dsl/ast_models.py` — add `priors: list[PriorSpec]` field on `DSLSpec` (default empty; backward compatible)
- `src/apmode/dsl/transforms.py` — extend `FormularTransform` union to include `SetPrior` from `prior_transforms`
- `src/apmode/dsl/stan_emitter.py` — inject priors from `spec.priors` into `_emit_model_block`, with defaults preserved for unset parameters
- `src/apmode/bundle/models.py` — extend `BackendResult`, extend `ParameterEstimate`, add `"bayesian_stan"` to backend Literal
- `src/apmode/backends/protocol.py` — no changes (runner protocol is already generic)
- `src/apmode/governance/policy.py` — add `gate1_bayesian_thresholds`, `gate2_bayesian_thresholds` blocks
- `src/apmode/cli.py` — add `--backend bayesian_stan` option and sampler-config flags

### 4.3 Phased rollout

| Phase | Weeks | Deliverable |
|---|---|---|
| 2A | 4 | `priors.py` + `prior_transforms.py` + emitter wiring + `BayesianRunner` skeleton + golden Stan snapshots |
| 2B | 4 | Full cmdstanpy integration, parquet draws, `PosteriorDiagnostics` parsing, Gate 1 Bayesian checks, first Suite A scenario under Bayesian |
| 2C | 4 | Prior-sensitivity sweep, prior-data conflict check, `prior_manifest.json`, Gate 2 prior justification flow, Submission lane admissibility |
| 2D | 4 | Gate 3 posterior-predictive simulation metrics (with MLE-side parametric bootstrap twin), `simulation_protocol.json`, Suite A/B Bayesian run |

### 4.4 Testing

- **Golden snapshots (syrupy):** Stan programs from fixed DSLSpecs with priors — regression guard for emitter changes.
- **Integration fixture:** Suite A `bolus_1cpt.csv` → Bayesian fit with `chains=2, warmup=200, sampling=200` (fast); assert R̂<1.05, ESS>100, parameter recovery within 30% of true.
- **Property-based (Hypothesis):** random DSLSpec + random prior configuration → validator terminates in bounded time + emits compilable Stan.
- **Mock cmdstanpy:** unit tests for `BayesianRunner._parse_response` with fabricated draws + diagnostics.

## 5. Risks and open questions

| Risk | Mitigation |
|---|---|
| MCMC wall-time (30 min–hours/fit) | Keep Bayesian off inner search loop; run only on Gate-2 survivors. Use `reduce_sum` + parallel chains. Revisit Marginal NUTS if bottleneck. |
| Prior-data conflict misses informative-suspected covariate structure | Require prior predictive + posterior predictive comparison, not only summary statistics |
| Torsten "prototype" status | Pin versions (`torsten_version` in SamplerConfig), snapshot-test event-handling edge cases |
| CmdStan/compiler drift breaks reproducibility | Pin cmdstan version + gcc/clang version in container digest; include in `backend_versions.json` |
| Funnels in hierarchical ODE models | Default to non-centered IIV parameterization; auto-reparam on divergence signal |
| BLQ + normal_lcdf numerical instability | Use log-sum-exp patterns; monitor divergences concentrated near BLQ regimes |
| "Metric shopping" under Gate 3 | Predefine metric stack in lane policy YAML (versioned policy artifact) |
| SBC (simulation-based calibration) too expensive for CI | Run nightly/weekly on a fixed scenario set; bundle results as `sbc_results.json` |

**Open questions — to decide before 2A ships:**
1. Is historical borrowing a single prior family (MAP) or a mixture (robust MAP)? Start with robust MAP mixture (Schmidli 2014), but expose single-MAP as a special case.
2. Does the prior-justification schema require per-historical-dataset provenance (SHA, access date), or only citation? Submission lane → full provenance; Discovery → citation suffices.
3. `SetPrior` on correlation structure (`corr_iiv`) vs. on individual omegas: start with LKJ on the full IIV correlation matrix; don't expose element-wise correlation priors in v1.

## 6. References

- FDA draft guidance (Jan 12 2026): "Use of Bayesian Methodology in Clinical Trials of Drug and Biological Products" (FDA-2025-D-3217)
- FDA final guidance (Aug 2024): "Optimizing the Dosage of Human Prescription Drugs and Biological Products for the Treatment of Oncologic Diseases"
- Margossian, Zhang, Gillespie (2022). "Flexible and efficient Bayesian pharmacometrics modeling using Stan and Torsten." CPT:PSP
- Davis & Vaddady (2025). "Within-chain parallelization—Giving Stan Jet Fuel for population modeling." CPT:PSP 14(1):52-67
- Säilynoja, Johnson, Martin, Vehtari (2025). "Recommendations for visual predictive checks in Bayesian workflow." arXiv:2503.01509
- Gelman et al. (2020). "Bayesian Workflow." arXiv:2011.01808
- Schmidli et al. (2014). "Robust meta-analytic-predictive priors in clinical trials with historical control information." Biometrics 70:1023-1032
- Tarek et al. (2025). "Marginal No-U-Turn Sampler for Bayesian Analysis in Pharmacometrics." PAGE 31 abstract II-92
