# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
