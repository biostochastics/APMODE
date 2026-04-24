# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — v0.6-rc1 Gate 2 prior-data-conflict + prior-sensitivity hard-gates (plan Tasks 20 + 21)

The Bayesian Gate 2 admissibility funnel now includes the two
remaining FDA-style prior diagnostics from PRD §4.3.1 / plan
§Block 1B. Both follow the same three-layer pattern as Task 19's
prior-justification check: a Pydantic artefact written to disk by
the harness/orchestrator, a pure-Python compute helper that the
harness can call once cmdstanpy lands the prior-only Stan pass, and
a Gate 2 check that reads the artefact back and rules pass/fail
against a versioned threshold.

- **`PriorDataConflict` + `PriorDataConflictEntry` schemas (Task 20).**
  New artefact `bayesian/{cid}_prior_data_conflict.json` records,
  per key dataset statistic, the observed value vs the central 95%
  prior-predictive PI computed from a fresh Stan
  `generated_quantities` pass with `prior_only=true`. The summary
  carries `conflict_fraction` (Gate threshold), per-entry
  `in_pi` flags with bound-consistency validation, and a
  `status="not_computed"` branch with a mandatory `reason` so a
  recorded skip is distinguishable from a missing artefact (Box
  1980; Evans & Moshonov 2006; Gabry et al. 2019).
- **`PriorSensitivity` + `PriorSensitivityEntry` schemas (Task 21).**
  New artefact `bayesian/{cid}_prior_sensitivity.json` records, per
  (parameter, alternative-prior) pair, the normalised posterior-mean
  shift `|Δmean|/posterior_sd_baseline`. The schema cross-checks the
  scoring arithmetic and the running `max_delta` so a hand-edited
  artefact cannot misreport (Roos et al. 2015; Kallioinen et al.
  2024 power-scaling sensitivity).
- **Pure-Python compute helpers.** `apmode.bayes.prior_data_conflict`
  exposes `compute_observed_summary`, `compute_prior_predictive_summaries`,
  and `compute_prior_data_conflict`; `apmode.bayes.prior_sensitivity`
  exposes `compute_prior_sensitivity`. They take already-computed
  numpy arrays / dict summaries so unit tests can exercise the gate
  end-to-end without cmdstanpy on the runner. The harness will plug
  them in once the prior-only Stan pass and N+1 alternative-prior
  refits land.
- **`Gate2Config.prior_data_conflict_required/_threshold` and
  `prior_sensitivity_required/sensitivity_max_delta`.** Submission
  defaults to `required=True` with `threshold=0.05` (5% of key
  statistics outside the prior 95% PI) and `max_delta=0.20` (≤ 1/5
  of a baseline posterior SD shift). Discovery / Optimization
  default both `*_required` knobs to `False` because the prior-only
  pass and N+1 refits are too expensive for the search-lane
  cost-benefit. Schema is fail-closed: when the lane requires the
  check but the artefact is missing or `status="not_computed"`,
  Gate 2 fails with a precise reason.
- **`evaluate_gate2` keyword-only `prior_data_conflict` and
  `prior_sensitivity`.** New `_check_prior_data_conflict` /
  `_check_prior_sensitivity` checks added to the Gate 2 funnel
  alongside the existing prior-justification check. Both trivially
  pass when the candidate is non-Bayesian or the lane policy
  doesn't require them.
- **`BackendResult.prior_data_conflict` /
  `prior_sensitivity` sidecar fields + emitter methods.** Mirror the
  `loo_summary` / `reparameterization_recommendation` pattern: the
  harness embeds the artefact inline on the result, the orchestrator
  emits it via `BundleEmitter.write_prior_data_conflict` /
  `write_prior_sensitivity`, and the gate reads the emitted JSON
  back through the new `evaluate_gate2` keywords so the
  reviewer-facing artefact is exactly what the gate ruled on.
- **Policy version bump 0.5.1 → 0.6.0.** All three lane policies
  carry the new fields with lane-appropriate defaults.

### Fixed — v0.6-rc1 multi-model review pass (correctness + wiring)

A second review pass turned up six high-confidence findings on the
just-merged Bayesian block, several of which were dead-code in
production. This change closes them.

- **`build_loo_summary` correctness.** `_coerce_float` returned
  `float("nan")` when arviz returned `None` or a non-coercible value,
  smuggling NaN past the `LOOSummary.<field>: float | None` contract
  and tripping JSON-serialization (NaN is not valid per RFC 8259).
  NaN now coerces to `None`. The `or` short-circuit on `getattr(loo,
  "elpd_loo", None) or getattr(loo, "elpd", None)` would have skipped
  legitimate `0.0` values and is replaced by an explicit
  `is not None` probe; `se_elpd_loo` also probes the
  `arviz_stats >= 1.0` `elpd_se` rename.
- **Per-class diagnostic dicts populated by the harness.** The Gate 1
  Bayesian evaluator already iterated
  `PosteriorDiagnostics.{rhat_max,ess_bulk_min,ess_tail_min}_by_class`
  but the harness left them as empty defaults — every per-class check
  silently skipped. `_compute_diagnostics` now buckets each variable
  via the new `apmode.governance.param_class.classify_param_class`
  (extracted to break the latent gates ↔ harness circular import).
- **Six-axis severity coverage on Gate 1 Bayesian.** `Gate1BayesianConfig`
  exposed `max_treedepth_fraction` and `e_bfmi_min` thresholds but the
  evaluator never read them and the severity validator listed only
  four axes (`rhat`, `ess`, `divergences`, `pareto_k`). Added explicit
  E-BFMI (Betancourt 2017 §6.1) and tree-depth-fraction checks; the
  severity dict now requires all six axes (`treedepth`, `ebfmi`
  added). Lane defaults: Submission/Optimization fail on `ebfmi`,
  warn on `treedepth`; Discovery warns on both. ESS warn-tier
  violations also surface in `warning_reasons` (previously dropped on
  the floor — only R-hat / divergences / pareto_k surfaced).
- **`pareto_k` non-finite sanitisation.** arviz can emit NaN/Inf
  Pareto-k entries when PSIS smoothing fails on individual
  observations. `_compute_diagnostics` now filters them before
  computing the max; `_bin_pareto_k` does the same so the bucket
  counts can't include phantom `very_bad` entries.
- **`PriorManifestEntry.doi` Crossref-canonical validator.** `doi`
  was `str | None` with no pattern check, so a free-form `"n/a"`
  satisfied Gate 2's truthiness probe. Added a Crossref regex
  validator matching the one on `LiteratureReference.doi`.
- **`LOOSummary.k_counts` key validator.** Accepts only the four
  Vehtari 2017 reliability bands (`good`/`ok`/`bad`/`very_bad`); a
  typo would otherwise corrupt downstream report rendering silently.
- **`approximate_posterior` polish.** `_empirical_bootstrap_draws`
  routed NaN diagonals through `np.where(np.isfinite(d) & (d > 0), …)`
  which raised RuntimeWarnings on NaN; now wrapped in
  `np.errstate(invalid="ignore")`. The `noise * std[np.newaxis, :]`
  reshape is removed in favour of natural broadcasting. The
  `cholesky-then-svd` comment that didn't match the actual `method="svd"`
  call is rewritten to describe what the code does. The variance
  floor `_BOOTSTRAP_VARIANCE_FLOOR = 1e-6` is now a named constant
  with a docstring describing when it activates.
- **`MetricTuple.method` distinguishes faithful Laplace from
  diagonal fallback.** Added `laplace_mvn` and
  `laplace_bootstrap_diagonal` literals so the Gate 3 ranker can tell
  a real MVN draw (full off-diagonal covariance preserved) from the
  degraded diagonal-only fallback. New
  `apmode.governance.approximate_posterior.laplace_draws_with_method`
  returns a `LaplaceDrawsResult` named tuple `(draws, method)` for
  callers that need the provenance; the array-only `laplace_draws`
  remains as a thin wrapper.

### Wired — v0.6-rc1 Bayesian artefacts into the orchestrator

The previous batch built `evaluate_gate1_bayesian`, `build_loo_summary`,
`build_reparameterization_recommendation`, and the prior-manifest
emitter — but none had a production caller, so they were dead code.
This change wires them into the orchestrator so v0.6-rc1 actually
governs Bayesian runs.

- **`evaluate_gate1_bayesian` runs alongside `evaluate_gate1`.**
  Bayesian candidates are now disqualified when per-class R-hat / ESS
  / divergences / treedepth / E-BFMI / Pareto-k thresholds fail with
  `severity="fail"`; `severity="warn"` violations surface in
  `summary_reason` without dropping the gate. Non-Bayesian backends
  pass trivially with `summary_reason="not_applicable — non-Bayesian
  backend"` so the audit trail is uniform.
- **Bayesian sidecar artefacts emitted on every run.**
  `Orchestrator._emit_bayesian_sidecars` writes
  `bayesian/{cid}_loo_summary.json`,
  `bayesian/{cid}_reparameterization_recommendation.json` (when the
  harness produces one), and `bayesian/{cid}_prior_manifest.json`
  before either gate evaluates. The harness now embeds `loo_summary`
  + `reparameterization_recommendation` inline on the result payload
  (new `BackendResult` fields) so the orchestrator owns the bundle-
  write step (single audit-trail point).
- **`evaluate_gate2` consumes the prior-manifest artefact.** The
  Submission-lane requirement (`bayesian_prior_justification_required:
  true`) is now actually enforced — the orchestrator loads
  `bayesian/{cid}_prior_manifest.json` from disk (the same artefact a
  reviewer would inspect) and threads it into `evaluate_gate2(...,
  prior_manifest=...)`.

Test counts: 2018 → <!-- apmode:AUTO:tests -->2024<!-- apmode:/AUTO:tests --> collected
(2001 → <!-- apmode:AUTO:tests_nonlive -->2007<!-- apmode:/AUTO:tests_nonlive --> non-live).
mypy `--strict` clean (108 source files); ruff `check` + `format` clean.

### Added — v0.6-rc1 Bayesian Block 1B/1C + Gate 1/2 Bayesian gates

Nine plan tasks (15, 16, 17, 18, 19, 22, 23, 24, 25) landed in a single
session, extending the Bayesian plumbing from "orchestrator wires through"
to "governance funnel actively consumes".

- **Plan Task 15 — `prior_manifest.json` emitter with validation.**
  `BundleEmitter.write_prior_manifest_from_specs(specs, cid, *, policy_version,
  justification_min_length)` validates every informative prior's
  justification + DOI before writing; aggregates failures so reviewers see
  every offending prior in one pass. `PriorManifestEntry` gains a `doi`
  field so Task 14's Crossref identifier survives the round-trip.
- **Plan Task 16 — per-lane `gate1_bayesian` block.**
  `Gate1BayesianConfig` splits R-hat and ESS floors across parameter
  classes (fixed_effects / iiv / residual / correlations) with a per-axis
  severity map (rhat / ess / divergences / pareto_k). Submission strictest,
  Discovery relaxes fixed-effects R-hat to 1.05, Optimization tightens
  Pareto-k to 0.5 with `fail` severity.
- **Plan Task 17 — `evaluate_gate1_bayesian()` warn/fail tiers.**
  `PosteriorDiagnostics` carries `rhat_max_by_class` / `ess_bulk_min_by_class`
  / `ess_tail_min_by_class`. The evaluator walks each axis against the
  matching threshold and applies the severity tier: warn-severity
  violations surface via `evidence_ref` without dropping the gate;
  fail-severity violations drop it with class-specific reasons.
- **Plan Task 18 — PSIS-LOO via `arviz.loo` + `loo_summary.json`.**
  `build_loo_summary(idata, cid)` is forward-compatible with arviz_stats
  1.0's renamed fields (`elpd` / `p`); bins Pareto-k into the four
  Vehtari 2017 reliability bands (`good` ≤ 0.5, `ok` ≤ 0.7, `bad` ≤ 1.0,
  `very_bad` > 1.0). Always emitted — `status="not_computed"` with reason
  is explicit, not an absent artifact.
- **Plan Task 19 — Gate 2 Bayesian prior-justification hard-gate.**
  `Gate2Config.bayesian_prior_justification_required` (Submission=True,
  Discovery/Optimization=False) + `bayesian_prior_justification_min_length`
  (Submission tightens to 500 chars). `evaluate_gate2` consumes an
  optional `prior_manifest` kwarg and aggregates per-prior errors. Fails
  closed when the lane demands the manifest but it isn't supplied.
- **Plan Task 22 — Laplace/MVN approximate posterior-predictive helper.**
  `apmode.governance.approximate_posterior.laplace_draws(theta, cov,
  n_draws, seed, *, fallback)` draws from the Laplace approximation for
  MLE backends so Gate 3 compares MLE and Bayesian candidates on
  commensurate intervals. Falls back to `empirical_bootstrap` (diagonal
  Normal around theta) when the covariance is ill-conditioned (cond > 1e12,
  non-finite entries, non-symmetric); `fallback="raise"` re-raises
  `LinAlgError` for callers that prefer hard-failing on degenerate MLE fits.
- **Plan Task 23 — commensurate `MetricTuple` carrier.**
  `MetricTuple(mean, ci_low, ci_high, method: Literal["posterior_draws",
  "laplace_draws", "empirical_bootstrap"], ci_level)` enforces
  `ci_low <= mean <= ci_high` at construction; the `method` tag records
  provenance so reviewers tell real posterior draws from Laplace-
  approximated ones. Dump shape is identical across methods so Gate 3
  iterates keys uniformly.
- **Plan Task 24 — Gate 3 metric stack + laplace_draws.**
  `Gate3Config.metric_stack: list[Literal[...]]` is the anti-metric-
  shopping whitelist; `Gate3Config.validate_metric(metric)` raises the
  new `PolicyError` (narrow `ValueError` subclass) when a candidate
  proposes an off-list metric. `Gate3Config.laplace_draws`: Submission
  2000, Discovery 500, Optimization 1000.
- **Plan Task 25 — drop silent auto-reparameterization.**
  `ReparameterizationRecommendation` advisory artifact emitted under
  `bayesian/{cid}_reparameterization_recommendation.json` when
  divergences or tree-depth saturations warrant intervention.
  `build_reparameterization_recommendation(diag, cfg, cid)` escalates
  above a 5% divergence-fraction floor to `switch_to_non_centered`
  (Betancourt & Girolami 2015) and falls back to
  `refit_with_higher_adapt_delta` for lower-rate divergences or tree-
  depth saturations. APMODE never switches parameterization silently —
  this is a consensus-review decision so the audit trail matches the run.

New unit tests: `test_bundle_prior_manifest.py` (10), `test_policy_gate1_bayesian.py` (12),
`test_gate1_bayesian.py` (24), `test_bayes_loo.py` (6),
`test_gate2_prior_justification.py` (8), `test_approximate_posterior.py` (12),
`test_metric_tuple.py` (9), `test_policy_gate3_metric_stack.py` (12),
`test_reparameterization_recommendation.py` (10). mypy `--strict` clean; ruff
`check` + `format` clean.

Continued in the same session — four more plan tasks (26, 38, 39, 42):

- **Plan Task 26 — `sbc_manifest.json` schema + stub emitter.**
  `SBCManifest` + `SBCPriorEntry` are the on-disk shape for the
  Talts 2018 Simulation-Based Calibration roll-up the nightly runner
  (Task 27) populates. Producer-side `BundleEmitter.write_sbc_manifest`
  emits a stub with `priors=[]` so the artefact's *presence* signals
  the Bayesian path executed end-to-end. Filename `sbc_manifest.json`
  joins the SBOM in `_DIGEST_EXCLUDED_NAMES` so a nightly rewrite does
  not invalidate `_COMPLETE`.
- **Plan Task 38 — Suite C rename + reframed claims.**
  `BenchmarkCase.expert_models` → `literature_models`. Suite C Phase 1
  is now framed as "methodology validation vs established literature
  models"; Phase 2 (head-to-head vs blinded human-expert panel) is
  marked out of v0.6 scope. `MIN_EXPERT_COUNT` retained as a
  deprecated alias for `MIN_LITERATURE_COUNT` (removal in v0.7).
- **Plan Task 39 — `LiteratureFixture` schema with parameterization mapping.**
  `LiteratureReference` (Crossref-canonical DOI + citation +
  population description) + `LiteratureFixture` (dataset_id + DSL
  spec path + reference_params + parameterization_mapping). The
  mapping is `{published_name: dsl_name}`; the validator enforces
  that mapping values index `reference_params` so cross-tool NPE
  comparisons (e.g. NONMEM `TVCL` vs DSL `CL`) cannot silently
  mis-align.
- **Plan Task 42 — Eleveld propofol DSLSpec coverage assessment.**
  `docs/discovery/eleveld_propofol_coverage.md` documents three
  blocking gaps preventing Eleveld 2018 from running on the
  Bayesian path: Stan emitter raises `NotImplementedError` for the
  `maturation` covariate form; no derived-covariate primitive
  (FFM = f(weight, sex, height)); no piecewise age-decay primitive.
  `scripts/check_eleveld_dslspec_coverage.py` is the auditable
  CI-friendly counterpart — exits non-zero on blocking gaps.
  Recommendation: **NO-GO** for v0.6 Phase-1 Bayesian fixtures.
  Vancomycin (Roberts 2011) is the only Phase-1 Bayesian fixture.

Plan tasks 26, 38, 39, 42 added unit tests:
`test_sbc_manifest.py` (9), `test_literature_fixture_schema.py` (11).
Total session test additions: 113 unit tests across thirteen plan
tasks. Test collected count: 1894 → 2018.

### Added — v0.6-rc1 in-progress (Bayesian Block 1 + Suite A8 + Gate 2.5 submission)

Work toward v0.6.0-rc1 has landed in pieces; the Bayesian orchestrator
plumbing, bundle-emitter Bayesian artefacts, Suite A8 scenario, and
Gate 2.5 block for the submission lane are now on main.

- **Gate 2.5 in submission lane.** `policies/submission.json` gains a
  `gate2_5` block (context_of_use, limitation_to_risk, data_adequacy
  with ratio floor 5.0, sensitivity, ai_ml_transparency=false). Policy
  version bumped to `0.5.1` in all three lane files (single-version
  invariant enforced by `scripts/sync_readme.py`). New
  `tests/unit/test_policy_schema.py` asserts the block shape.
- **Suite A8 — 1-cmt oral + time-varying CL + CRCL covariate.**
  `benchmarks/suite_a/simulate_all.R` simulates
  `CL(t, CRCL) = CL0 * (CRCL/90)^0.75 * exp(-0.15 * t / 24)` across
  60 subjects × 11 observations; `src/apmode/benchmarks/suite_a.py`
  adds `scenario_a8()` + `A8_COVARIATE_MODEL_NOTES` recording the
  diurnal rate that no DSL primitive expresses.
  `reference_params.json` carries an `_expected_misspecification_bias`
  block so benchmark tooling compares recovery to the time-averaged
  target rather than raw `CL0`.
- **LORO-CV lane gating tests.** Property tests
  (`tests/property/test_loro_fold_properties.py`) assert fold count =
  regimen count and permutation stability; integration tests
  (`tests/integration/test_loro_orchestrator.py`) pin that LORO fires
  in the optimization lane and is skipped in submission even on
  LORO-eligible data.
- **Platform-adaptive cmdstanpy kwargs.** New
  `src/apmode/bayes/platform.py::cmdstan_run_kwargs(*, uses_reduce_sum)`
  returns Windows-safe `force_one_process_per_chain=True` (cmdstanpy
  issue #895 — MinGW `thread_local` performance bug) and POSIX
  `cpp_options={"STAN_THREADS": True}` when `reduce_sum` is active
  (issue #780).
- **BayesianRunner wired into discovery/optimization.**
  `Orchestrator.__init__` gains `bayesian_runner`; `_LANE_BACKENDS`
  admits `bayesian_stan` in discovery and optimization lanes only
  (submission stays classical NLME per PRD §3 hard rule).
- **Provenance-recording sample helper.** New
  `sample_with_provenance` compiles + samples with
  `save_cmdstan_config=True` (cmdstanpy issue #848) and writes
  `backend_versions.json` with SHA-256 hashes of the Stan program and
  data, cmdstan version, host platform, and the resolved
  `one_process_per_chain` flag.
- **Bayesian bundle artefacts.**
  `BundleEmitter.write_posterior_draws` (long-form Parquet with
  optional thinning), `write_posterior_summary` (canonical 9-column
  schema), `write_mcmc_diagnostics`, `write_sampler_config` — each
  validated against `_SAFE_ID_RE` and written under
  `run_dir/bayesian/`. `ParameterEstimate` gains
  `posterior_mean` alongside `posterior_sd` / `q05` / `q50` / `q95`
  so reports can disambiguate posterior mean from primary estimate.
- **DSL prior-justification validator.**
  `apmode.dsl.priors.validate_prior_justification(spec, *, min_length)`
  checks the Crossref-canonical DOI pattern and a minimum
  justification length (default 50 chars; callers override via
  `min_length`). DOI regex accepts SICI-style suffixes with angle /
  square brackets. `PriorSpec` now carries a `doi: str | None` field
  and uses `_INFORMATIVE_SOURCES` frozenset as the single source of
  truth (previously duplicated as a tuple literal).

### Fixed — Multi-model review pass on unpushed v0.6 work

Multi-model review (gemini-pro / droid / crush / opencode) across the
14 unpushed v0.6-rc1 commits surfaced correctness, safety, and
documentation gaps. All resolved on main:

- **Bayesian harness hardcoded `converged=True`**. `_run` now derives
  the flag from a conservative floor — R-hat ≤ 1.05, bulk ESS ≥ 400,
  zero divergent transitions — via a new unit-testable
  `_is_converged` helper. Policy-driven warn/fail tiers land in
  plan Task 17 (Gate 1 Bayesian).
- **Silent DV≤0 dropping is now an explicit error**. `_build_stan_data`
  raises `ValueError` (mapped to `error_type="invalid_spec"`) when it
  encounters non-positive DV rows with `MDV=0`, since the lognormal /
  proportional likelihood cannot accommodate them. Callers must set
  `MDV=1` on pre-dose baselines or switch to the BLQ_M3 / BLQ_M4
  module with explicit censoring.
- **CSV path hardening**. `_build_stan_data` resolves
  `request["data_path"]`, rejects non-string / empty values, and
  requires the path to point to a regular file before `pd.read_csv`
  touches it (CWE-22 defence-in-depth).
- **DOI regex extended**. `_DOI_PATTERN` now accepts `<`, `>`, `[`,
  `]` so Wiley's legacy SICI-style DOIs (`10.1002/(SICI)…`) pass
  validation.
- **Bayesian RO-Crate projector coverage**. The
  `src/apmode/bundle/rocrate/entities/bayesian.py` regex now matches
  `sampler_config`, `posterior_summary`, and `posterior_draws`
  alongside the existing `prior_manifest` / `simulation_protocol` /
  `mcmc_diagnostics` / `draws` shapes.
- **Dead / duplicated code removed**. Unused
  `SCENARIOS: dict[str, type[None]]` deleted from `benchmarks/suite_a.py`;
  `_INFORMATIVE_SOURCES` frozenset is now the single source of truth
  for `PriorSpec`'s informative-source branch; `routing.py` had a
  dangling `Follow-up: (` comment, replaced with the active
  description of the `node_dim_budget` primary gate.
- **Trust-boundary documentation**. `sample_with_provenance` carries
  an explicit note that `stan_code` must originate from
  `apmode.dsl.stan_emitter.emit_stan` — user-supplied or agentic
  output must pass through the DSL allow-list before reaching
  cmdstanpy's C++ compile path.
- **Suite A8 docstring + reference_params.json now disclose
  misspecification bias**. Benchmark comparisons use the
  time-averaged `CL` target (`3.678`) rather than raw `CL0 = 4.482`.

Test coverage added: `test_bayes_harness_convergence.py`,
`test_bayes_harness_data.py`, `tests/unit/rocrate/test_entities_bayesian.py`,
plus DOI-bracket / `min_length` cases on
`test_prior_justification.py`.

### Changed — Deep-review hardening (41-finding audit)

A comprehensive cross-subsystem review identified 41 correctness, wiring,
and audit-trail issues across the DSL compiler, backends, governance
gates, reproducibility bundle, and benchmark suite. All 41 are now
resolved. Highlights:

- **Stan/nlmixr2 proportional-error likelihood unified** (#1).
  `stan_emitter` now emits `y ~ normal(f, sigma_prop * f)` to match the
  nlmixr2 `cp ~ prop(prop.sd)` semantics. Previously Stan used
  `lognormal(log(f), sigma)` for the non-BLQ path while the BLQ branch
  used the Normal form — cross-paradigm NLPD (PRD §4.3.1) was
  invalidated by the inconsistency.
- **Stan ODE solver O(N × N_subjects) collapse** (#3). The
  ``transformed data`` block precomputes ``obs_start[i]`` / ``obs_end[i]``
  per subject; the per-subject loop iterates the slice directly instead
  of scanning all observations with a subject filter.
- **Stan dose-event mismatch now raises** (#4). Previously
  `event_cmt > n_states` silently dropped the dose; emitter now emits
  a Stan ``reject(...)`` so the dataset/model CMT mismatch is visible.
- **Bundle seal is atomic** (#32, #34). `_COMPLETE` is written through
  a tmp + `os.replace` dance so a mid-write crash cannot leave a
  half-formed sentinel that `initialize()` would accept.
- **Post-seal bundle mutation refused** (#2). `append_search_trajectory`
  and `append_failed_candidate` raise `BundleAlreadySealedError` when
  `_COMPLETE` is present (would otherwise desync the stored digest).
- **LORO-CV `fixed_parameter` protocol** (#5). `BackendRunner.run` gains
  an explicit `fixed_parameter: bool = False` kwarg; `evaluate_loro_cv`
  passes `True`; every runner (`nlmixr2`, `bayesian`, `node`, `agentic`)
  raises `NotImplementedError` when requested rather than silently
  re-fitting and leaking train data through warm-start.
- **Gate 1 thresholds policy-driven** (#6). `Gate1Config` gains
  `param_value_min` / `param_value_max` / `param_rse_max` /
  `seed_stability_ofv_abs_spread_floor`. No more hardcoded `1e-4`,
  `1e5`, `200`, `0.1` in `gates.py`.
- **`adaptive_m` / `m_max` no longer phantom** (#7). `data/missing_data.py`
  raises `NotImplementedError` when a policy sets
  `adaptive_m=True` so the unimplemented escalation cannot silently
  degrade to fixed m.
- **Ranking tiebreak deterministic** (#8). `metrics.sort(key=...)`
  now includes `candidate_id` as tiebreaker so Borda ties resolve the
  same way across runs with identical bundles.
- **BLQ sigma validation gap + IVBolus validator branch** (#9, #10).
  `validator._validate_observation` now enforces sigma positivity on
  BLQ M3/M4 conditional on `error_model`; `_validate_absorption`
  gained an explicit no-op branch for IVBolus.
- **NODE modules contribute to `structural_param_names()`** (#11).
  Variability items that target NODE input-layer weights
  (`node_abs_wN` / `node_elim_wN`) now pass validation.
- **`_bt_logit` honours `cov.form`** (#12). Power / exponential /
  linear / maturation routing mirrors `_bt`; the rc8 path flattened
  all forms to linear-additive on the logit scale.
- **`nlmixr2` ZeroOrder + TMDDQSS routing fix** (#13). Emits
  `dur(Atot)` under TMDDQSS (central = total drug), `dur(centr)`
  elsewhere — via a new `_central_cmt_name` helper.
- **`MixedFirstZero` `frac == 1.0` no longer crashes** (#14). Clamp
  fraction to `1 - 1e-4` before emitting `log(frac / (1 - frac))`.
- **Stan TwoCmt flip-flop singularity detected** (#15). Emits a
  tolerance check; near-singular analytical denominators `reject(...)`
  so the caller can switch to the numerical ODE solver.
- **Stan TMDDQSS aliased as `Atot`** (#16). The central state is
  named both `Atot` (total drug) and `centr` (alias) so the RHS
  template stays uniform and the naming matches nlmixr2's
  `_emit_tmdd_qss_odes`.
- **Source positions flow into `DSLSpec.source_meta`** (#17).
  `parse_dsl` enables `propagate_positions=True`; `compile_dsl` walks
  the parse tree once and attaches `(line, column)` for absorption /
  distribution / elimination / observation / variability[i] so
  validator errors can point at source.
- **PIT calibration degenerate-sims safeguard** (#18, #33, #41).
  `build_predictive_diagnostics` catches the all-NaN `ValueError`
  from `_compute_pit_calibration` and returns a zero-subject sentinel;
  Gate 1 surfaces this as `pit_degenerate_no_finite_sims`; the
  `n_observations` denominator no longer inflates a true 0 to 1.
- **Credibility report provenance** (#19). `CredibilityReport` gains
  `source_result_path` + `source_result_sha256` so an auditor can
  walk back to the `BackendResult` JSON any score was derived from.
- **`SearchDAG` seals** (#20). `seal()` / `SearchDAGSealedError`; all
  mutators (`add_root`, `add_child`, `update_score`) refuse after
  seal so the in-memory DAG cannot desync from
  `candidate_lineage.json`.
- **Agentic LLM raw_text sanitized** (#21). Already guarded for
  `runner_error`; now applied on every `raw_text` append.
- **Subprocess executable validated up front** (#22).
  `Nlmixr2Runner` / `BayesianRunner` resolve the executable to an
  absolute path via `shutil.which` at `__init__`; `FileNotFoundError`
  surfaces immediately.
- **NODE runner rejects `timeout_seconds`** (#23). JAX training is
  non-interruptible; silently accepting a timeout was a lie. Now
  raises `NotImplementedError` so orchestrators can choose a
  watchdog-subprocess path.
- **MI per-imputation error isolation** (#24, #39). One failed
  imputation no longer discards every previously-completed fit;
  `m_imputations=0` refused upstream instead of emitted as a
  degenerate manifest.
- **`compute_npe` is likelihood-aware** (#25, #37). New
  `error_model` kwarg rescales residuals on proportional / combined
  models so cross-paradigm NPE comparison is no longer scale-biased;
  additive default preserves rc8 behaviour.
- **PRD §10 stress-surface perturbations declared** (#26).
  `PerturbationType` gains `SCALE_BSV_VARIANCES`, `SATURATE_CLEARANCE`,
  `TMDD`, `FLIP_FLOP` with required-parameter validators; dispatchers
  raise `NotImplementedError` until the transforms land, so Suite C
  never silently replaces a stress request with a no-op.
- **`ka` silent fallback surfaced** (#27). `_apply_fallback` logs a
  structured warning and sets `_ka_defaulted=1.0` metadata when the
  dataset card or defaults path supplies `ka`.
- **Silent emitter catch-alls removed** (#28). Unknown observation
  modules now raise `NotImplementedError` on both nlmixr2 and Stan
  paths instead of emitting a wrong proportional default.
- **InvGamma off `*_sd` targets** (#29). `_VALID_FAMILIES` no longer
  claims InvGamma is a valid SD-scale prior when the emitter draws
  on the SD scale; use `Gamma` / `HalfNormal` / `HalfCauchy`.
- **BLQ M3/M4 active-sigma helper** (#30). `active_sigmas()` on
  `BLQM3` / `BLQM4` returns only the fields the likelihood actually
  consumes given `error_model` — param-count helpers must call this
  instead of inspecting every field.
- **Gate 1 fields carry bounds + units** (#31). Added
  `description=` + explicit dimensions to the #6-added Gate1Config
  fields so audit readouts know the unit they're comparing against.
- **Agentic warm-start guards non-finite estimates** (#35). NaN / Inf
  parameters are dropped with a structured log entry instead of
  propagating into the next iteration.
- **transform_parser allow-list explicit** (#36). Extra LLM-supplied
  fields are rejected against `module_cls.model_fields` at parse time;
  Pydantic `extra="forbid"` becomes a redundant second line of
  defence instead of the only one.
- **Balanced protocol-pooling allocation** (#38). Replaces
  `rng.integers(...)` with block randomisation so the docstring's
  "balanced allocation" claim is now true.
- **`numpy.trapezoid` auto-select** (#40). Already honoured via
  `getattr(np, "trapezoid", None) or np.trapz` (NumPy 2.0+ forward,
  1.x backward).

Test baseline: 1825 passing, 1 skipped. `mypy --strict` clean (105
files); `ruff check` clean; 22 golden snapshots unchanged.

### Added — Supply-chain SBOM (CycloneDX)

APMODE now ships a CycloneDX Software Bill of Materials (SBOM) as a
producer-side sidecar and as a GitHub-release asset. The SBOM describes
the Python dependency graph that produced the bundle; it is generated
by `pip-audit --format cyclonedx-json` (pip-audit is already pinned in
the dev group, so no new top-level dependency is introduced).

- **`apmode bundle sbom <bundle_dir>`** (new CLI) — runs `pip-audit`
  against the active environment and writes
  `<bundle_dir>/bom.cdx.json`. Refuses to overwrite an existing SBOM
  without `--force`. Emits a JSON summary with `--json`.
- **Sealed-digest exclusion.** `bom.cdx.json` is excluded from
  `apmode.bundle.emitter._compute_bundle_digest` and the matching
  importer verifier, so adding or regenerating the SBOM on a sealed
  bundle never invalidates `_COMPLETE`. This preserves bundle
  immutability while still colocating the SBOM with the run.
- **RO-Crate projection.** A new `src/apmode/bundle/rocrate/entities/sbom.py`
  projector registers the SBOM as a `File` entity with
  `encodingFormat="application/vnd.cyclonedx+json"` and
  `additionalType="apmode:sbom"` (new term in `vocab.py`). The root
  Dataset's `hasPart` gains the SBOM file when present; no-op when
  absent (golden snapshot unchanged).
- **CI gate.** The `security` job in `.github/workflows/ci.yml` now
  emits `bom.cdx.json` on every push / PR and uploads it as a
  workflow artifact (`sbom-cyclonedx-json`, 90-day retention).
- **Release asset.** A new `.github/workflows/release.yml` triggers
  on `v*` tag pushes, regenerates the SBOM against the released
  environment, builds wheel + sdist with `uv build`, and attaches
  `bom.cdx.json` + distribution archives to the GitHub release via
  `gh release create`.
- **Test coverage.** `tests/unit/rocrate/test_entities_sbom.py`
  covers the no-op, successful projection (hash + media type +
  `apmode:sbom`), and the invariant that injecting `bom.cdx.json`
  after sealing does not trip `_verify_sentinel`.

### Added — RO-Crate v0.6 integration

APMODE reproducibility bundles can now be projected onto a **Workflow Run
RO-Crate — Provenance Run Crate v0.5** (`https://w3id.org/ro/wfrun/provenance/0.5`)
for FAIR packaging, WorkflowHub/Zenodo-ready archives, and machine-readable
regulatory crosswalk (FDA PCCP, EU AI Act Article 12). The Pydantic bundle
remains producer-side truth; the RO-Crate is a read-only external projection.
Design: `_research/ROCRATE_INTEGRATION_PLAN.md` §§A–H (accepted).

- **New module `src/apmode/bundle/rocrate/`** — hand-written projector
  with per-family entity modules (`data`, `policy`, `backend`, `gate`,
  `lineage`, `credibility`, `bayesian`, `agentic`, `pccp`). Reads a
  sealed bundle via JSON, emits a deterministic `ro-crate-metadata.json`
  plus a copy of every bundle artefact as a `File` entity. Never
  mutates the source bundle.
- **Two output forms**: directory (`apmode bundle rocrate export …
  --out /path/to/crate`) and ZIP (`… --out crate.zip`). ZIP entries
  use a fixed `1980-01-01` timestamp + sorted order for reproducible
  archives; `json.dumps(sort_keys=True)` gives byte-identical
  `ro-crate-metadata.json` across runs on the same sealed bundle.
- **`apmode:` vocabulary** at `https://w3id.org/apmode/terms#` —
  `lane`, `lanePolicy`, `gate`, `gateRationale`, `candidateLineageEdge`,
  `searchGraph`, `modificationDescription`, `modificationProtocol`,
  `impactAssessment`, `traceabilityTable`, `regulatoryContext`,
  `dslSpec`, `dslTransform`, `llmInvocation`, `credibilityReport`,
  `loroCV`, `scoringContract`, `nlpdComparabilityProtocol`,
  `completeSentinel`. Context declares each term with both prefix
  form and explicit full-prefix mapping to satisfy `roc-validator`'s
  compaction check.
- **Action triad per WRROC**: each candidate fit projects as a
  `CreateAction` instrumenting its `SoftwareApplication`; each gate
  decision projects as a `ControlAction` with `instrument=HowToStep`,
  `object=CreateAction`, and `apmode:gateRationale=File`; the lane
  run is wrapped in an `OrganizeAction` with
  `startTime=endTime=bundle_seal_ts`.
- **`_COMPLETE` sentinel** is embedded as a `File` entity with
  `additionalType="apmode:completeSentinel"`; the SHA-256 digest is
  surfaced via `schema:identifier="sha256:<hex>"`.
- **Round-trip import** (`apmode bundle import <crate> --out <bundle>`)
  extracts a directory- or ZIP-form crate back to a bundle directory
  and verifies the `_COMPLETE` digest against the extracted tree. ZIP
  extraction rejects path-traversal entries (ZIP-slip), symlink/hard
  link/socket entries, absolute paths, and Windows drive prefixes;
  directory imports reject symlinks inside the crate.
- **Validator integration**: `apmode validate <bundle> --rocrate`
  invokes `roc-validator` on the crate at a configurable severity
  (default REQUIRED) and profile (default `provenance-run-crate-0.5`);
  `apmode inspect <bundle> --rocrate-view` prints an RO-Crate
  summary (action-triad counts, lane, `mainEntity`, sentinel
  presence).
- **New dep**: `roc-validator>=0.8.1` in the `dev` group; main dep
  `rocrate>=0.13` for future ro-crate-py interop.
- **CI gate**: `tests/integration/test_rocrate_export_validate.py`
  parametrises 5 scenarios (minimal, multi-candidate+lineage,
  bayesian+credibility, agentic+PCCP, full-mixed) and validates both
  directory and ZIP forms at REQUIRED severity. REQUIRED failures
  block merge.
- **Security tests**: `tests/unit/rocrate/test_importer_security.py`
  covers ZIP-slip via parent-dir traversal, absolute paths, Windows
  drives, and symlink file-type mode bits.
- **Golden snapshot**: `tests/golden/rocrate/__snapshots__/` locks the
  canonical Submission-lane `ro-crate-metadata.json`.
- **Publish CLI surface** (`apmode bundle publish --workflowhub` /
  `--zenodo`) is wired as a CLI stub — sandbox/token handling is
  validated but the upload itself is deferred to v0.8 per
  `_research/ROCRATE_INTEGRATION_PLAN.md` §H.

### References

- Leo S. et al. (2024) *Recording provenance of workflow runs with RO-Crate*. PLOS ONE 19(9): e0309210. <https://doi.org/10.1371/journal.pone.0309210>
- Wilkinson S.R. et al. (2025) *Applying the FAIR Principles to computational workflows*. Scientific Data 12: 328. <https://doi.org/10.1038/s41597-025-04451-9>

## [0.5.0-rc2] — 2026-04-17

Known-Limitations Closure M0 landed: per-candidate `ScoringContract`
on every `DiagnosticBundle`, contract-grouped Gate-3 ranking, and the
Submission-lane dominance rule. Gate policy schema bumped to `0.5.0`;
bundle sentinel schema bumped to `2`. Plan: `.plans/v0.5.0_limitations_closure.md`.

### Added

- **`ScoringContract` Pydantic model** (`src/apmode/bundle/models.py`).
  Frozen, 7-field contract: `contract_version`, `nlpd_kind`
  (conditional / marginal), `re_treatment`
  (integrated / conditional_ebe / pooled), `nlpd_integrator`
  (nlmixr2_focei / laplace_blockdiag / laplace_diag / hmc_nuts / none),
  `blq_method`, `observation_model`, `float_precision`. Attached on
  `DiagnosticBundle.scoring_contract` (defaults to the classical
  nlmixr2 contract so existing call sites continue to type-check).
- **Per-backend contract derivation**
  (`src/apmode/bundle/scoring_contract.py`). `derive_scoring_contract`
  maps `BackendResult.backend` + `DSLSpec.observation` + BLQ method
  to the correct contract. Runners call `attach_scoring_contract` on
  their returned result: `nlmixr2_runner.run` and `bayesian_runner.run`
  override after `_parse_response`; `node_runner.run` sets the
  contract in-place on the constructed `DiagnosticBundle` (pooled
  path until M3 Laplace lands).
- **Contract-grouped Gate 3 ranking**
  (`src/apmode/governance/ranking.py`):
  - `group_by_scoring_contract(survivors)` — partitions survivors
    into same-contract buckets preserving order.
  - `ContractGroupedRanking` dataclass — one `CrossParadigmRankingResult`
    per contract, plus optional `recommended_candidate_id`.
  - `rank_by_scoring_contract(..., lane=...)` — groups, ranks each
    group via the existing `rank_cross_paradigm` primitive, applies
    Submission-lane dominance rule when `lane == "submission"`.
- **Submission-lane dominance rule.** Only candidates whose contract
  has `re_treatment == "integrated"` AND `nlpd_kind == "marginal"`
  are eligible for `recommended_candidate_id`. Otherwise the ranking
  returns `recommended_candidate_id = None` and a verbatim
  disclosure warning. Discovery and Optimization lanes leave the
  field `None` by design (separate leaderboards are the intended
  output).
- **Tests**: `tests/unit/test_scoring_contract.py` (10 tests) and
  `tests/integration/test_gate3_contract_enforcement.py` (8 tests).
- **`.plans/v0.5.0_limitations_closure.md`** — consolidated 10-item
  closure plan. Final DAG, M3 reduced-scope decision (Laplace-only,
  ≤16×16 block, L-BFGS + single ridge fallback), M1.5 MM-default
  gating, A1–A7 regression merge-gate mandate, six PR checkpoints.

### Changed

- **Bundle schema version** (`_COMPLETE_SCHEMA_VERSION`) bumped from
  1 → 2. Bundles produced before this version do not carry
  `scoring_contract`; at read time the field defaults to the
  classical nlmixr2 contract (no migration script required — Pydantic
  resolves it via the `default_factory`).
- **Gate policy schema version** bumped from `0.4.3` → `0.5.0`
  across all three lane JSONs. CLAUDE.md `apmode:AUTO:policy_gate`
  marker auto-synced.

### Deferred (explicitly surfaced, ADR-gated)

- **#2 NODE infusions (RATE>0)** — ADR: `docs/adr/0004-node-infusions.md`
  (pending); loud-reject landing in M2a.
- **Stan steady-state (SS!=0)** — ADR: `docs/adr/0003-stan-ss-scope.md`
  (pending); Stan and NODE backends hard-reject `SS!=0` at Gate 1
  (landing in M2a). README wording update pending:
  *"SS supported in nlmixr2 lane only; NODE is oral-only in v0.5.0."*

## [0.5.0-rc1] — 2026-04-17

First release candidate for 0.5. Gate policy schema `0.4.3`; profiler
policy `2.1.0`; bundle sentinel schema `1`.

### Added

- **Bundle `_COMPLETE` sentinel.** `BundleEmitter.seal()` writes a
  JSON sentinel with a SHA-256 digest of every other artifact in the
  bundle. The orchestrator calls it as the last step of a successful
  run. `apmode validate` refuses unsealed bundles and verifies the
  digest on every read. Required artifacts promoted:
  `evidence_manifest.json`, `candidate_lineage.json`.
- **`StanIdentifier` AST type alias** (regex
  `^[A-Za-z][A-Za-z0-9_]*$`) applied to `CovariateLink.param`,
  `CovariateLink.covariate`, `IIV.params`, `IOV.params`. Emitter-side
  `_sanitize_stan_name()` also rejects Stan reserved keywords and
  `__` suffixes.
- **`LLMConfig.timeout_seconds`** (default 120s) with a shared
  `_await_llm` helper wrapping both native SDK timeouts and an outer
  `asyncio.wait_for`. Applies to Anthropic / OpenAI / OpenRouter /
  Gemini / Ollama direct-SDK providers and the litellm fallback.
- **`MissingDataPolicy.imputation_convergence_rate_min`** policy
  field (submission 0.75, optimization 0.65, discovery 0.5). Replaces
  a hard-coded constant in `gates.py`.
- **BIC uniform-drop rule.** `_apply_uniform_bic_drop` in
  `governance/ranking.py` mirrors the AUC/Cmax drop; composite
  renormalization composes correctly when both components have
  missing candidates. Pathological fallback (BIC was the only
  positive weight) falls back to `vpc=0.5, npe=0.5`.
- **Transit `n` variability rejection** now applied to IOV blocks
  (previously only IIV).
- **`set_prior` transform** in the LLM-response parser
  (`backends/transform_parser.py`). Registration-map refactor plus a
  regression test that asserts every `FormularTransform` union
  variant has a parser entry.
- **Python 3.14** added to the CI test matrix.
- **`docs/adr/0001-review-deferrals.md`** registers intentional
  non-fixes (`from __future__ import annotations`, Pyright, profiler
  / orchestrator decomposition, FREM golden tests, `# type: ignore`
  audit, module-level Rich `Console`).

### Changed

- **Stan emitter IV-bolus support.** `IVBolus` no longer emits a
  phantom depot state or an undeclared `ka` term. `_needs_depot()`
  and `_centr_idx()` helpers thread through `_n_states`,
  `_emit_state_aliases`, `_emit_ode_dynamics`, and `_emit_ode_solve`.
- **Single `Lane` StrEnum** in `apmode.backends.protocol`. `cli.py`
  re-exports the enum; `routing.py` keeps the string `Literal` form
  for compatibility with policy JSONs.
- **`Orchestrator.__init__`** accepts the `BackendRunner` protocol
  directly (previously typed narrowly as `Nlmixr2Runner`, with a
  `cast` at the CLI callsite).
- **Anthropic `model_version` escrow** concatenates the response's
  `_request_id` / `id` so agentic reproducibility traces carry a
  per-call server-side fingerprint.
- **`_check_split_integrity`** fails on missing `split_gof` when
  `split_integrity_required=True`. The policy field default changed
  to `False` so benchmark and single-fold workflows do not need to
  opt out.
- **`nlmixr2` runner** spawns subprocesses with
  `start_new_session=True` instead of the thread-unsafe
  `preexec_fn=os.setsid` (matches the Bayesian runner).
- **`_within_between_ratio`** guards non-finite `pooled` against
  silently producing NaN composite scores.
- **Profiler helpers** (`_assess_protocol_heterogeneity`,
  `_assess_absorption_coverage`, `_assess_elimination_coverage`,
  `_assess_route_certainty`) return `Literal[...]` matching the
  `EvidenceManifest` fields.
- **`policy=None` on submission lane** now raises
  `APMODEConfigError` instead of silently skipping all gates.
- **Report renderer** uses `datetime.now(tz=UTC)`.
- **Seed-stability check** short-circuits to pass when the absolute
  OFV spread across seeds is below 0.1 units (platform-float floor,
  well below textbook model-selection tolerance).
- **`agentic_iterations.jsonl`** flushes and fsyncs per iteration so
  the audit trail survives mid-run crashes.
- **Assertion-based invariants** in the orchestrator's agentic stage
  replaced with explicit `RuntimeError` / `TypeError` guards so
  invariants survive `python -O`.
- **Shared `needs_ode`** helper in `apmode.dsl._emitter_utils`
  replaces duplicated definitions in both emitters.

### Fixed

- `np.trapezoid` compatibility shim — NumPy 2.0 removed
  `np.trapz`, older installs lack `np.trapezoid`; the project
  minimum `numpy>=1.25` tolerates either.
- Profiler policy-load failure wrapped in a typed
  `APMODEConfigError` with the resolved path.
- Ollama SDK drift — attribute access first, dict-subscript fallback
  for legacy installs.
- `transform_parser` handles `set_prior` (was previously missing
  from the if/elif chain).

### Deprecated

- Pre-0.5 reproducibility bundles without the `_COMPLETE` sentinel no
  longer validate. Re-run to upgrade.

### Security

- Stan emitter no longer interpolates unsanitized covariate /
  variability parameter names into generated code, closing an
  injection vector from LLM-proposed identifiers.

### Policy versions

- Gate policy schema: `0.4.2` → `0.4.3`
- Bundle sentinel schema: `1`
- Profiler policy: unchanged at `2.1.0`
