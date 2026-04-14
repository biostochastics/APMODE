# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Full multi-dose support across all backends (ADDL/II/SS)**
  - **Data schema**: Added optional ADDL (additional doses), II (inter-dose interval),
    SS (steady-state flag) columns to CanonicalPKSchema with cross-column validation
    (ADDL>0 requires II>0, SS in {1,2} requires dose EVID, ADDL only on dose rows)
  - **Data manifest**: Added `has_multidose` and `has_steady_state` flags to DataManifest
    for downstream backend dispatch decisions
  - **Dose expansion** (`apmode.data.dosing`): New module with `expand_addl()` to
    materialize ADDL/II into explicit dose rows, `expand_infusion_events()` for
    infusion stop event generation, `build_event_table()` for complete event table
    construction with deterministic EVID-priority sorting
  - **nlmixr2 backend**: Pass-through support — rxode2 handles ADDL/II/SS natively;
    columns are passed through in the data without code changes
  - **Stan emitter**: Event-based piecewise ODE integration with merged dose+observation
    timeline (chronological single-pass loop). Analytical solutions use superposition
    principle with reset-awareness (EVID=3/4 invalidate prior dose contributions)
  - **NODE/JAX backend**: Piecewise eager integration via `_solve_multidose_eager()` with
    merged dose+observation chronological timeline. Single-dose-at-t=0 uses legacy JIT
    path. Multi-dose/delayed-dose/reset subjects use event-driven Python control flow.
    Infusion data explicitly rejected with clear error until Phase 3
  - **Tests**: 19 new unit tests for dosing module covering expand_addl, infusion events,
    event table construction, subject extraction, and schema validation

### Fixed
- **NODE backend: 14 fixes from code review**
  - **[Critical]** 2-cmt y0 dimension: runner now builds 3-element state vector for
    2-compartment models (was always 2-element, causing shape crash)
  - **[Critical]** BIC/AIC parameter count: uses actual trainable param count from model
    tree (~25 for typical NODE) instead of `len(param_estimates)` (~5), which severely
    underpenalized NODE in Gate 3 cross-paradigm ranking
  - **[High]** NLL missing log(2π): added `0.5*n*log(2π)` constant for cross-backend
    comparability with nlmixr2 (was ~1.84/obs offset)
  - **[High]** Monotone constraint semantics: renamed/documented that MonotoneIncreasing
    guarantees positive output, not input-output monotonicity; MonotoneDecreasing now
    returns positive bounded output via `1/(1+sum(softplus))` (was returning negative)
  - **[High]** Multi-dose rejection: NODE runner raises `InvalidSpecError` for subjects
    with >1 dose event (was silently using only the first dose)
  - **[High]** ODE integration start: `solve()` now integrates from `t0=0.0` (dose time)
    instead of `times[0]` (first observation), preventing time-shift errors
  - **[High]** Saturable constraint: MM-like `Vmax*s/(Km+s)` form with learned log-space
    parameters replaces bounded sigmoid (which didn't model rate saturation)
  - **[High]** Convergence semantics: patience exhaustion no longer unconditionally marks
    `converged=True`; returns granular `minimization_status` (successful/plateau/nan/max)
  - **[High]** V/V2 volume scaling: trainer uses V2 for peripheral compartment observations
    instead of always dividing by V
  - **[Medium]** WeightLibrary thread safety: added `threading.Lock` for concurrent async runs
  - **[Medium]** Time sorting: CSV loader sorts observations by time before Diffrax solve
  - **[Low]** Numerical stability: replaced `log1p(exp(x))` with `jax.nn.softplus(x)`
  - **[Low]** Documented GMR direction in `FidelityResult` (surrogate=test, NODE=reference)

### Added
- Real-data NODE integration tests (12 tests):
  - theo_sd (12 subjects, 1-cmt): CSV load, train convergence, distillation, full E2E runner
  - Oral_2CPT: multi-dose rejection guard verified
  - 2-cmt mock mode: correct y0 shape, training without crash
  - All 5 constraint templates trained on real data without NaN
- GPU-not-required note in README prerequisites
- **Phase 3 P3.B: LORO-CV for Optimization lane** (PRD §3.3)
  - `LOROFoldResult`, `LOROMetrics`, `LOROCVResult` Pydantic models in bundle/models.py
  - `loro_cv_splits()` fold generation with regimen-signature grouping (modal dose
    per subject, not total AMT)
  - Gate 2 `_check_loro_requirement` now evaluates real LORO thresholds:
    pooled NPDE mean/variance, VPC coverage concordance, minimum fold count
  - LORO policy fields in `Gate2Config`: `loro_npde_mean_max`, `loro_npde_variance_min/max`,
    `loro_vpc_coverage_min`, `loro_min_folds`, `loro_budget_top_n`
  - `evaluate_loro_cv()` execution engine with per-fold fitting, metric aggregation,
    law of total variance for pooled variance (E[Var] + Var[E])
  - `write_loro_cv_result()` emitter writing to `loro_cv/{candidate_id}.json`
  - New `src/apmode/evaluation/` package for cross-validation and predictive performance
  - Orchestrator wiring: LORO-CV runs between Gate 1 and Gate 2 for optimization lane,
    budget-capped by `loro_budget_top_n`, regimen labels preserved for auditability
  - Evidence-grounded design: Khandokar (2025 Uppsala) CV for PK model selection,
    Comets et al. (2008) NPDE, Bergstrand et al. (2011) pcVPC
  - Multi-model consensus reviewed (MIMO-v2-Pro, GPT-5.2-Pro):
    - Fixed variance proxy formula (use cwres_sd² consistently, not outlier_fraction)
    - VPC missing-evidence → fail-closed (0.0, not convergence fraction fallback)
    - Gate 2 NaN safety: explicit `np.isfinite` checks prevent silent pass
    - Gate 2.5 no longer overwrites Gate 2 artifact (gate_number=25)
    - `_find_spec_for_candidate` replaced with typed `_spec_map` dict
  - 39 new tests: fold generation (13), Gate 2 LORO (14), execution engine (7),
    Hypothesis property tests (5)
- `TimeVaryingElim.kdecay` as first-class AST field (was phantom parameter
  with hardcoded initial estimate); grammar accepts optional `kdecay=` syntax
- BLQ grammar extended: `BLQ_M3`/`BLQ_M4` now accept optional `error_model`,
  `sigma_prop`, `sigma_add` parameters in DSL text (not just programmatically)
- Semantic validation: TMDD distribution requires LinearElim (prevents
  undefined `CL` in emitted code)
- Semantic validation: duplicate IIV parameters across blocks rejected
- Semantic validation: duplicate CovariateLinks (same param+covariate) rejected
- Semantic validation: Transit `n` cannot have IIV/IOV/CovariateLink (emitters
  don't apply eta to its back-transform)
- IOV pruning after module swap transforms (was only IIV + CovariateLink)
- Stan emitter: TMDD_QSS prediction now uses Cfree (algebraic QSS solve),
  not Ctot (was scientific correctness bug)
- Stan emitter: TMDD Core/QSS receptor ODE uses explicit kdeg derivation,
  consistent with nlmixr2 emitter
- Stan emitter guardrails: ZeroOrder/MixedFirstZero absorption and IOV
  raise NotImplementedError instead of emitting broken code
- Lark parser cached via `lru_cache` (was re-instantiated per parse call)
- Formular docs: semantic validation rules table (docs/FORMULAR.md)

### Previously added
- `apmode explore` interactive wizard: step-by-step dataset exploration
  (fetch -> ingest -> profile -> NCA -> search space -> optional pipeline launch)
  with `--non-interactive` flag for CI
- `apmode datasets` command: browse and download 14 public PK datasets from
  nlmixr2data (5 real clinical, 9 simulated ground-truth)
- `apmode log` command: query gate decisions (`--gate`), failed candidates
  (`--failed`), and top-N parameter estimates (`--top N`) from bundles
- `apmode diff` command: side-by-side comparison of evidence manifests,
  rankings, and gate pass rates between two bundles
- NODE initial estimate strategy (`node_init.py`): pre-trained weight library
  for 1-cmt/2-cmt reference dynamics + transfer learning from classical fits
- Stan codegen emitter (`stan_emitter.py`): DSL -> complete Stan program with
  ODE/analytical solvers, IIV, IOV, BLQ M3/M4, covariates, LOO-CV log_lik
- Per-backend lowering test suite: validates nlmixr2/Stan structural parity
- Suite A CSV fixtures A5-A7 generated via rxode2 (TMDD QSS, covariates,
  NODE saturable absorption)
- Independent `log_CL` field in `HybridPKODE` for proper absorption-mode
  elimination parameterization
- `WeightLibrary.reset()` and `reset_weight_library()` for test isolation
- Dataset local cache: skip re-fetch if CSV already exists
- Exit code documentation (0/1/2/130) in CLI help

### Changed
- VPC concordance target now configurable via `GatePolicy.vpc_concordance_target`
  (was hardcoded 0.90) and threaded through `rank_cross_paradigm()`
- Diffrax solver uses `dt0=None` for automatic step-size selection (was 0.1)
- Distillation Michaelis-Menten surrogate fit uses `scipy.optimize.curve_fit`
  (was Lineweaver-Burk linearization)
- BIC in `NodeBackendRunner` uses total observation count (was subject count)
- `np.polyfit` in distillation called once instead of twice
- Data profiler detects TMDD inverse curvature (ratio < 0.3) in addition to
  MM curvature (ratio > 1.8)

### Fixed
- NODE absorption mode no longer uses `ka` as CL proxy — `CL/V` is used for
  elimination, with CL as an independently trainable parameter
- Stan emitter: `maturation` covariate form now raises `NotImplementedError`
  instead of silently dropping the effect
- Profiler: `corr_matrix.values` read-only array error on pandas 2.x fixed
  via `to_numpy(copy=True)`
- R simulation script: rxode2 5.x compatibility (id column for iCov, EVID
  101->1 normalization, duplicate RENAL column merge)
- CLI: `rich.markup.escape()` for user-supplied paths, `--verbose`/`--quiet`
  mutual exclusion, `dataset.is_file()` check, `timeout min=1`

## [0.1.0] - 2026-04-13

### Added
- Phase 1 complete: DSL compiler + Lark grammar, nlmixr2 emitter, semantic
  validator, classical NLME backend (nlmixr2 R subprocess), data ingestion
  (NONMEM CSV + Pandera), data profiler (evidence manifest), NCA estimator,
  data splitter (k-fold, LORO-CV), automated search engine with Pareto
  frontier, governance gates (1/2/2.5/3), lane router, orchestrator,
  reproducibility bundle emitter, Typer CLI, structlog logging
- Phase 2 in progress: JAX/Diffrax/Equinox NODE backend (6 modules),
  Gate 2.5 ICH M15 credibility, Gate 3 cross-paradigm ranking,
  SearchEngine multi-backend dispatch, Discovery lane integration,
  Benchmark Suites A (7 scenarios) + B (3 NODE validation)
- 679 tests (Phase 1) + 270 new tests (Phase 2) = 949 total
- mypy --strict clean, ruff clean
- Seven rounds of code review

[Unreleased]: https://github.com/biostochastics/APMODE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/biostochastics/APMODE/releases/tag/v0.1.0
