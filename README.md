# APMODE

**Adaptive Pharmacokinetic Model Discovery Engine** — a governed meta-system that composes four PK modeling paradigms (classical NLME, automated structural search, agentic LLM model construction, and hybrid mechanistic-NODE) into a single workflow for population PK model discovery.

Licensed under GPL-2.0-or-later.

## Status

**Phase 1, Month 3-4 — IN PROGRESS.** 554 tests passing. `mypy --strict` clean. `ruff` clean.

Multi-model code reviews completed 2026-04-13 (5 models: codex, gemini, GLM-5, MiniMax, droid); all critical issues fixed.

### What exists

| Component | Path | Description |
|-----------|------|-------------|
| **PK DSL grammar** | `src/apmode/dsl/pk_grammar.lark` | Full Lark EBNF grammar for PK model specifications |
| **AST models** | `src/apmode/dsl/ast_models.py` | Typed Pydantic nodes: Absorption (6), Distribution (5+V), Elimination (5), Variability (3), Observation (5+BLQ composition) |
| **Semantic validator** | `src/apmode/dsl/validator.py` | Constraint table enforcement (PRD §4.2.5) |
| **nlmixr2 emitter** | `src/apmode/dsl/nlmixr2_emitter.py` | DSL AST → R code strings for nlmixr2/rxode2 |
| **Nlmixr2Runner** | `src/apmode/backends/nlmixr2_runner.py` | Async subprocess backend with file-based IPC |
| **R harness** | `src/apmode/r/harness.R` | R-side nlmixr2 SAEM/FOCEI estimation harness |
| **Data ingestion** | `src/apmode/data/ingest.py` | NONMEM CSV → Pandera validation → DataManifest |
| **Data profiler** | `src/apmode/data/profiler.py` | Evidence Manifest generation (PRD §4.2.1) |
| **NCA estimator** | `src/apmode/data/initial_estimates.py` | NCA-based initial estimate derivation (PRD §4.2.0.1) |
| **Data splitter** | `src/apmode/data/splitter.py` | Subject-level splitting, k-fold, LORO-CV |
| **Search candidates** | `src/apmode/search/candidates.py` | Automated search space, candidate generation, SearchDAG |
| **Benchmark Suite A** | `benchmarks/suite_a/` | rxode2 simulation for 4 recovery scenarios |
| **Bundle emitter** | `src/apmode/bundle/emitter.py` | Reproducibility bundle artifacts |
| **Bundle models** | `src/apmode/bundle/models.py` | All Pydantic schemas (BackendResult, EvidenceManifest, etc.) |
| **Gate policies** | `src/apmode/governance/policy.py` | Gate 1, 2, 2.5 policy file schemas |
| **Lane policies** | `policies/*.json` | Default gate thresholds per lane |

### Test suite

| Suite | Count | Description |
|-------|-------|-------------|
| Unit tests | ~350 | All modules: DSL, data, backends, search, governance, bundle |
| Integration tests | 3 | End-to-end mock R pipeline (success, convergence error, crash) |
| Golden masters | 21 | Syrupy snapshots for pharmacometrician-validated R output |
| R syntax validation | 168 | Balanced delimiters, eta/param consistency |
| Property-based | ~30 | Hypothesis: roundtrip, JSON, NODE constraints |

### DSL example

```
model {
    absorption: Transit(n=4, ktr=2.0, ka=1.0)
    distribution: TwoCmt(V1=30.0, V2=40.0, Q=5.0)
    elimination: ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)
    variability: {
        IIV(params=[CL, V1, ka], structure=block)
        CovariateLink(param=CL, covariate=WT, form=power)
    }
    observation: Combined(sigma_prop=0.1, sigma_add=0.5)
}
```

This compiles to a typed AST, validates against the constraint table (including param cross-references), and lowers to a complete nlmixr2 R model function.

## Setup

```bash
# Requires Python 3.12+, uv
uv sync --all-extras

# Optional: R 4.4+ with nlmixr2/rxode2 for real estimation
# (mock R subprocess tests work without R)
```

## Testing

```bash
uv run pytest tests/ -q                    # all 554 tests
uv run pytest tests/unit/ -q               # unit tests only
uv run pytest tests/integration/ -q        # end-to-end mock R pipeline
uv run pytest tests/property/ -q           # Hypothesis property-based
uv run pytest tests/golden/ -q             # syrupy golden master snapshots
uv run pytest tests/ --snapshot-update     # update snapshots after emitter changes
uv run mypy src/apmode/ --strict           # type checking (0 errors)
uv run ruff check src/apmode/ tests/       # linting (0 errors)
uv run ruff format src/apmode/ tests/      # formatting
```

## Benchmark Suite A

Simulated PK datasets with known ground truth for structure/parameter recovery (PRD §5).

```bash
# Requires R 4.4+ with rxode2, jsonlite, lotri
Rscript benchmarks/suite_a/simulate_all.R [output_dir]
```

| Scenario | Model | Key Test |
|----------|-------|----------|
| A1 | 1-cmt oral, first-order abs, linear elim | Structure identification |
| A2 | 2-cmt IV, parallel linear+MM elim | Compartment count + nonlinear CL |
| A3 | Transit (n=3), 1-cmt, linear elim | Transit chain detection |
| A4 | 1-cmt oral, MM elimination | Nonlinear clearance detection |

## Architecture

See `ARCHITECTURE.md` for the full technical architecture and `PRD_APMODE_v0.3.md` for the product requirements.

### System pipeline

```
NONMEM CSV ──→ ingest_nonmem_csv() ──→ CanonicalPKSchema (Pandera)
                                              │
                                    ┌─────────┴──────────┐
                                    ↓                    ↓
                            DataManifest          profile_data()
                            (SHA-256,             (Evidence Manifest:
                             covariates)           richness, route,
                                                   clearance, BLQ)
                                    │                    │
                                    ↓                    ↓
                            NCAEstimator         SearchSpace.from_manifest()
                            (CL, V, ka)          (dispatch constraints)
                                    │                    │
                                    ↓                    ↓
DSL text ──→ Lark parser ──→ DSLSpec ──→ emit_nlmixr2() ──→ R code
                                │                              │
                        Semantic validator            Nlmixr2Runner.run()
                                │                     (async subprocess)
                        split_subjects()                       │
                        (k-fold, LORO)               BackendResult (Pydantic)
                                                               │
                                                    BundleEmitter → JSON + .R
```

### Pharmacometric references

- **TMDD full binding**: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532
- **TMDD QSS**: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591
- **Transit compartments**: Savic et al. (2007), J Pharmacokinet Pharmacodyn 34:711-726
- **Allometric scaling**: Anderson & Holford (2008), Clin Pharmacokinet 47:455-467 (70 kg reference)
- **BLQ M3/M4**: nlmixr2 censoring via CENS/LIMIT data columns
- **NCA**: linear trapezoidal AUC, terminal log-linear kel, CL=Dose/AUC_inf

## Phasing

- **Phase 0** (complete): Schemas, protocols, grammar, error taxonomy, sparkid integration
- **Phase 1 Month 1-2** (complete): DSL compiler, nlmixr2 lowering, bundle scaffolding
- **Phase 1 Month 2-3** (complete): Classical NLME backend, R harness, data ingestion, benchmarks
- **Phase 1 Month 3-4** (in progress): Data profiler, NCA estimates, automated search, data splitting
- **Phase 1 Month 4-5**: Governance layer (Gate 1, Gate 2)
- **Phase 1 Month 5-6**: Orchestrator, CLI, full Benchmark Suite A

## Code Review Provenance

Two rounds of multi-model code review have been conducted:

| Date | Models | Focus | Key Fixes |
|------|--------|-------|-----------|
| 2026-04-13 (R1) | codex, gemini, droid, crush | DSL compiler, emitter, bundle models | IOV syntax, TMDD QSS KD/KSS, BLQ composition |
| 2026-04-13 (R2) | codex, gemini, GLM-5, MiniMax, droid | Data pipeline, NCA, search, profiler, security | A2 CMT bug, NCA terminal slope, NaN/Inf guards, route certainty, Spearman ties |

## Known Limitations (Phase 1)

- `TimeVaryingElim.decay_fn`: only `exponential` supported; `half_life` and `linear` rejected by validator
- TMDD QSS uses `KD` as approximation for `KSS = (koff + kint)/kon`; when `kint >> koff`, KD underestimates KSS and can overestimate complex formation
- `kdeg` in TMDD models is a heuristic initial estimate (`koff` for Core, `kint` for QSS), not a separately estimable parameter
- Zero-order absorption uses `dur(centr)` modeled-duration pattern; requires compatible dose event coding (CMT=central, modeled duration flag)
- NCA initial estimates use simple terminal-phase log-linear regression (last 3 post-Tmax points); not suitable for multi-dose or steady-state data without interval handling
- Nonlinear clearance detection uses Cmax as dose proxy (confounded); should use AMT when available
- Route certainty assessment is conservative: CMT=1 without RATE/DUR is classified as "inferred" not "confirmed"
- NODE modules raise `NotImplementedError` (Phase 2)
