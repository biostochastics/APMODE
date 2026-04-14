# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (code review sweep)
- **Orchestrator:** replaced `if "ranking" in dir()` sentinel with explicit
  `Ranking | None` initialization (C1). The prior pattern was fragile under
  refactors and masked scope-tracking errors.
- **Agentic runner:** a single `run_id` is now minted at the top of
  `AgenticRunner.run()` and reused across every `agentic_trace/*_input.json`
  and the run-level `RunLineage` artifact (C2). Previously each iteration
  received a freshly-generated candidate ID, breaking PRD §4.2.6 multi-run
  provenance.
- **Gate 3 cross-paradigm ranking:** orphan candidate IDs are now logged and
  skipped rather than raising `RuntimeError` mid-bundle-emission (C3).
- **Concurrency:** migrated three `asyncio.gather` sites (search engine,
  seed-stability runs, LORO-CV) to `asyncio.TaskGroup` for structured
  cancellation (H1, PEP 654).
- **Rich CLI formatters:** `isinstance(bic_val, float | int)` now correctly
  rejects booleans via `_is_real_number`; `str(True)` no longer renders as
  `"1"` in the LLM diagnostics summary (H2).
- **JAX platform:** the process-global `jax.config.update("jax_platform_name",
  "cpu")` is now routed through `configure_jax_platform()`, which warns on a
  conflicting request instead of silently drifting (H3).
- **Errors:** renamed `apmode.errors.TimeoutError` to `BackendTimeoutError`
  to stop shadowing `builtins.TimeoutError` at importers (M1).
- **Node subject records:** introduced `SubjectRecord` TypedDict; dropped
  `# type: ignore[dict-item]` on the event-driven subject dict (M2).
- **Profiler:** replaced the `[]`-sentinel + `isinstance(np.ndarray)` guard
  in lag detection with an explicit `(times > 0).any()` branch (M7).
- **Profiler thresholds:** the magic constants (0.7, 0.4, 1.8, 0.3, 0.5,
  0.15) are now named module-level constants with intent comments pointing
  toward a future `policies/profiler.json` artifact (M8).
- **Broad excepts:** narrowed agentic / seed / LORO catches from
  `except Exception` to `except BackendError` (plus `RuntimeError` where the
  agentic runner intentionally raises on exhaustion) (M10).
- **Ollama client:** the recorded `request_payload_hash` now covers the
  `options` dict (including `temperature`) so the hash matches the actual
  request (M11, PRD §4.2.6 model-version escrow).
- **CLI:** dropped the blanket `console.status(...)` spinner around the
  pipeline run so structlog stage messages are no longer suppressed (M14).

### Added (code review sweep)
- **DSL `IVBolus` absorption variant** (H7) — replaces the
  `FirstOrder(ka=100.0)` stiff-ODE hack previously used to approximate IV
  bolus dosing. Emitted explicitly by `nlmixr2_emitter` (no depot ODE;
  dose routes to the central compartment via CMT=1).
- **LLM diagnostics redaction layer** (H5) —
  `diagnostic_summarizer.redact_for_llm()` enforces an allow-list of
  aggregate fields before anything leaves the process to a third-party LLM
  provider (PRD §10, ARCHITECTURE.md §11). Unknown keys are dropped
  fail-closed.
- **Provider registry typing** (M6) — `_PROVIDER_REGISTRY` is now typed as
  `dict[str, Callable[[LLMConfig], LLMClientProtocol]]` and
  `register_provider` preserves concrete subclass types via a TypeVar.
- **CI security job** (H4) — new job runs `uv run bandit -r src/apmode/`
  and `uv run pip-audit` on every push/PR. `bandit` and `pip-audit` added
  to the `dev` extras group.
- **Test parallelism** (L6) — `pytest-xdist` added to the `test` extras;
  the CI `test` job now runs `pytest -n auto` without `-x`.
- **Python version matrix widened to 3.14** (M3) — `requires-python` is now
  `">=3.12,<3.15"` and the classifier block advertises 3.14. The CI matrix
  remains 3.12/3.13 until JAX/Diffrax publish 3.14 wheels.

### Changed (code review sweep)
- **Agentic backend default** (H6) — `apmode run --agentic` now defaults to
  `False`. The prior auto-on-when-API-key-present behavior would silently
  exfiltrate diagnostics as soon as an `ANTHROPIC_API_KEY` was in the shell
  env. Opt-in is explicit.
- **Ruff config** (M4) — removed the deprecated `UP038` entry from the
  ignore list.
- **Pre-existing pandas `# type: ignore[import-untyped]` pragmas** — removed
  throughout `data/` now that pandas is covered by the mypy overrides block.

### Added
- **Deep inspection CLI commands** (Phase 3 deep inspection, PRD §4.3.2)
  - `apmode trace <bundle>` — inspect agentic LLM iteration traces: summary table,
    per-iteration detail (`--iteration N`), token/cost aggregation (`--cost`),
    JSON export (`--json`)
  - `apmode lineage <bundle> <candidate_id>` — trace the DSL transform chain from
    root to any candidate, with gate status at each step; merges agentic lineage
    automatically; optional `--spec` for DSL snapshots, `--no-gate` to hide gates
  - `apmode graph <bundle>` — visualize the full search DAG as Rich tree (default),
    Graphviz DOT (`--format dot`), Mermaid (`--format mermaid`), or JSON
    (`--format json`); filters: `--converged`, `--backend`, `--depth`; file export
    via `--output`; cycle-safe rendering
  - `apmode inspect` now shows deep inspection hints when trace/graph/lineage
    artifacts are present in the bundle
  - New Pydantic models: `SearchGraphNode`, `SearchGraphEdge`, `SearchGraph`,
    `AgenticIterationEntry` in `bundle/models.py`
  - `BundleEmitter.write_search_graph()` for `search_graph.json` artifact
  - `SearchDAG.iter_nodes()` / `to_edges()` public API for graph building
  - `IterationRecord` now tracks `transforms_rejected` and `validation_feedback`
    for complete audit trail in `agentic_iterations.jsonl`
  - 25 new unit tests covering models, emitter, and all three CLI commands
- **Parallel model evaluation** (`--parallel-models N` / `-j N` CLI flag)
  - Concurrent candidate evaluation in SearchEngine via `asyncio.gather` + `Semaphore`
  - Parallelized seed-stability runs and LORO-CV evaluation in the orchestrator
  - Sequential fast-path when `max_concurrency=1` (default, zero overhead)
  - `RunConfig.max_concurrency` field with `__post_init__` clamping for API safety
  - Lazy semaphore creation to avoid event-loop binding issues
  - All shared-state mutations (DAG, trajectory writes, emitter) remain sequential post-gather
  - 6 new async unit tests covering semaphore bounds, result ordering, and error isolation
- **Comprehensive benchmark suite expansion** (Phase 3, PRD §5/§9)
  - **Benchmark infrastructure** (`benchmarks/models.py`): Pydantic v2 models for
    `DatasetCard`, `BenchmarkCase`, `BenchmarkScore`, `PerturbationRecipe`,
    `SuiteSummary`, `SuiteReport`; `SuiteName` Literal type alias; `AccessTier`
    enum (open/scripted/credentialed) for CI eligibility
  - **Suite A-External** (`suite_a_external.py`): Schoemaker 2019 / ACOP 2016 grid
    integration — 12 nlmixr2data datasets (bolus/infusion/oral x 1-cmt/2-cmt x
    linear/MM), weekly cadence with nightly and CI smoke subsets
  - **Suite B Extended** (`suite_b_extended.py`): 7 real-data benchmark cases:
    B4 theophylline NODE anchor (Bram 2023 continuity), B5 mavoglurant BLQ 25%/40%,
    B6 mavoglurant 5% outlier injection, B7 mavoglurant sparse absorption,
    B8 mavoglurant null covariate FPR (5 random covariates), B9 gentamicin IOV
  - **Suite C** (`suite_c.py`): Expert head-to-head framework — mavoglurant,
    gentamicin IOV (DDMoRe CC0), Eleveld propofol (OpenTCI); 5-fold subject-level
    CV, NPE primary metric, 2% win margin, pre-registered evaluation protocol;
    published models as reference anchors, blinded expert panel as primary baseline
  - **7 perturbation types** (`perturbations.py`): inject_blq (M3-style CENS/LLOQ),
    remove_absorption_samples, add_null_covariates, inject_outliers (additive
    fallback for near-zero DV), sparsify, add_protocol_pooling (STUDY_ID +
    optional sampling/LLOQ variation per protocol), add_occasion_labels (vectorized)
  - **Scoring harness** (`scoring.py`): structure recovery (DSLSpec-wired), parameter
    bias (NaN for missing estimates), parameter coverage (None for missing CI per
    GPT-5.2-pro: unscorable, not free pass), NPE, prediction interval calibration,
    expert comparison (empty-list guard), dispatch assertion checking, convergence
    taxonomy; `SuiteName`-typed `aggregate_suite()`
  - **Dataset cards and prepare scripts** for 7 public PK datasets:
    nlmixr2data Schoemaker grid (prepare.R), theophylline (prepare.R),
    warfarin PK/PD (prepare.R), mavoglurant (prepare.R), DDMoRe gentamicin IOV
    (prepare.py, CC0), Eleveld propofol (prepare.R, OpenTCI), MIMIC-IV vancomycin
    (README + access instructions, Tier 2 credentialed)
  - **Dataset registry** (`benchmarks/datasets/registry.yaml`): master index of all
    datasets with access tiers, suite assignments, and prepare script references
  - 77 new integration tests covering all three suites, all 7 perturbation types,
    scoring harness, expert comparison methodology, recipe validation, and
    protocol pooling; total test count: 1232 passing
- **PerturbationRecipe cross-field validators**: Pydantic model_validator enforces
  required fields per perturbation type (blq_fraction for BLQ, outlier_fraction
  for outliers, target_obs_per_subject for sparsify, etc.); Field constraints
  (blq_fraction in [0,1], outlier_magnitude > 1, target_obs_per_subject >= 1)
- **SuiteSummary n_passed <= n_cases constraint**: model_validator prevents
  impossible summary states
- `__all__` exports on all benchmark modules for explicit public API

### Changed
- **BLQ perturbation uses M3-style censoring**: DV set to LLOQ (not 0.0) with
  CENS=1 column per nlmixr2/NONMEM M3 convention; DV=0 was ambiguous and biased
  M3 likelihood (GPT-5.2-pro + Gemini 3.1 Pro consensus)
- **Observation identification uses (EVID==0) & (MDV==0)**: all perturbation
  functions now exclude EVID=0/MDV=1 rows (missing samples) per NONMEM convention;
  prevents corrupting missingness semantics in M3/M4 handling
- **score_parameter_coverage returns None (not True) for missing CI**: SAEM
  covariance failures are unscorable, not free passes; aggregators decide policy
- **score_structure_recovery wired to DSLSpec**: compares discovered spec's
  absorption/distribution/elimination types against ExpectedStructure; returns
  None when no spec available (unscorable) instead of always True
- **score_parameter_bias flags missing estimates with NaN**: reference params
  absent from BackendResult get bias=NaN, not silent skip
- **Occasion labeling vectorized**: replaced O(n*m) row loop with groupby+cumsum;
  sorts by (NMID, TIME, EVID) before labeling to handle unsorted input
- **PublishedModel defined before DatasetCard**: eliminates forward-reference
  risk in Pydantic v2 schema generation
- **aggregate_suite accepts SuiteName Literal** (was str): removes noqa/type-ignore
- Added `pandas.*` to mypy ignore_missing_imports (no stubs installed)

### Fixed
- Mock convergence test: `_mock_result(converged=False)` now sets
  `minimization_status="terminated"` (was "successful" — tautological assertion)
- Outlier injection handles near-zero DV: uses additive offset instead of
  multiplicative 0*magnitude=0

### Reviewed by
- 5-agent multi-model code review (droid, crush, gemini, codex, opencode)
- GPT-5.2-pro and Gemini 3.1 Pro consultation on PK-domain fixes
- All 15 identified issues addressed

### Added
- **Agentic LLM backend wired to CLI and orchestrator** (Phase 3, PRD §4.2.6)
  - `--agentic/--no-agentic` flag on `apmode run` (default: on)
  - `--provider` (anthropic/openai/gemini/ollama/openrouter), `--model`, `--max-iterations`
  - Auto-activates in discovery/optimization lanes when API key found; gracefully
    disables with warning when unavailable
  - Submission lane hard-blocks agentic (PRD §3 rule enforced in lane router)
  - **Post-search agentic stage** in orchestrator with two modes:
    - **Refine mode**: takes best classical candidate, LLM proposes targeted transforms
    - **Independent mode**: starts from minimal 1-cmt oral spec, LLM builds from scratch
  - Both modes feed results back into gate funnel alongside classical candidates
  - In discovery/optimization lanes, LLM can also propose NODE transforms
- **Multi-turn conversation history in agentic loop**: LLM now sees full history
  across iterations (prior diagnostics, its own proposals, validation outcomes)
  instead of stateless per-iteration prompts
- **Validation feedback to LLM**: transform validation failures, parse errors, and
  DSL semantic violations are fed back as structured user messages so the LLM can
  correct rejected proposals (was silently swallowed with `continue`)
- **Live LLM provider integration tests** (`test_llm_providers_live.py`):
  8 tests across Anthropic, OpenAI, OpenRouter, Ollama, and full agentic loop;
  `@_skip_on_billing` decorator for graceful skip on quota/auth errors;
  `live` pytest marker (`-m live` to run, excluded from default suite)
- **`llm` optional dependency group** in pyproject.toml: `uv sync --extra llm`
  installs anthropic, openai, google-genai, ollama, litellm

### Fixed
- **R harness compatibility with nlmixr2 5.0**: fit$time (data.frame not vector),
  fit$conditionNumberCov (renamed from conditionNumber), fit$shrink (data.frame
  replacing fit$shrinkage), CWRES fallback to IWRES (SAEM doesn't compute CWRES
  by default), OFV/AIC/BIC via Gaussian quadrature, empty-list-to-dict JSON
  serialization for profile_likelihood_ci and diagnostic_plots
- **Gemini client multi-turn**: replaced string concatenation with proper
  `types.Content` objects preserving multi-turn conversation structure
- **OpenRouter auth**: client now reads `OPENROUTER_API_KEY` directly instead
  of requiring users to alias it as `OPENAI_API_KEY`

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
