# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Seven rounds of multi-model code review

[Unreleased]: https://github.com/biostochastics/APMODE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/biostochastics/APMODE/releases/tag/v0.1.0
