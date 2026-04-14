# APMODE

**Adaptive Pharmacokinetic Model Discovery Engine** — a governed meta-system that composes four PK modeling paradigms (classical NLME, automated structural search, agentic LLM model construction, and hybrid mechanistic-NODE) into a single workflow for population PK model discovery.

Licensed under GPL-2.0-or-later.

## Status

**Phase 2 — IN PROGRESS.** 850 tests passing. `mypy --strict` clean (46 files). `ruff` clean.

Phase 1 complete (679 tests). Phase 2 NODE backend + governance + multi-backend dispatch + benchmarks implemented (171 new tests). Six rounds of multi-model code review completed; all critical and high-severity issues fixed.

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
| **Data profiler** | `src/apmode/data/profiler.py` | Evidence Manifest with AMT-based nonlinear CL detection |
| **NCA estimator** | `src/apmode/data/initial_estimates.py` | NCA with multi-dose AUC_tau, extrapolation fraction check |
| **Data splitter** | `src/apmode/data/splitter.py` | Subject-level splitting, k-fold, LORO-CV |
| **Search candidates** | `src/apmode/search/candidates.py` | Automated search space, candidate generation, SearchDAG |
| **Search engine** | `src/apmode/search/engine.py` | Multi-backend candidate dispatch (nlmixr2 + jax_node), BIC scoring, Pareto frontier |
| **Gate 1 evaluator** | `src/apmode/governance/gates.py` | Technical Validity: 7 checks (convergence, plausibility, CWRES, VPC, seed stability) |
| **Gate 2 evaluator** | `src/apmode/governance/gates.py` | Lane Admissibility: 6 checks (interpretability, shrinkage, identifiability, NODE exclusion) |
| **Gate 2.5 evaluator** | `src/apmode/governance/gates.py` | Credibility Qualification (ICH M15): 5 checks (COU, data adequacy, ML transparency) |
| **Gate 3 ranking** | `src/apmode/governance/gates.py` | Within-paradigm BIC + cross-paradigm simulation-based composite ranking |
| **Cross-paradigm ranking** | `src/apmode/governance/ranking.py` | VPC concordance, NPE, composite score for mixed-backend ranking |
| **NODE constraints** | `src/apmode/backends/node_constraints.py` | 5 enumerated constraint templates (monotone, bounded, saturable, smooth) |
| **NODE sub-model** | `src/apmode/backends/node_model.py` | Bram-style MLP with RE on input-layer weights |
| **Hybrid ODE** | `src/apmode/backends/node_ode.py` | Mechanistic PK skeleton + NODE sub-function, Diffrax Tsit5 solver |
| **NODE trainer** | `src/apmode/backends/node_trainer.py` | Optax Adam training loop with early stopping, log-space params |
| **NodeBackendRunner** | `src/apmode/backends/node_runner.py` | BackendRunner protocol impl for JAX/Diffrax NODE backend |
| **Functional distillation** | `src/apmode/backends/node_distillation.py` | Sub-function visualization, parametric surrogate fitting, AUC/Cmax BE fidelity |
| **Credibility report** | `src/apmode/report/credibility.py` | ICH M15-aligned credibility assessment per candidate |
| **Lane Router** | `src/apmode/routing.py` | Dispatch decisions by lane + evidence manifest constraints |
| **Orchestrator** | `src/apmode/orchestrator/__init__.py` | Full pipeline: ingest → profile → NCA → split → search (multi-backend) → gates 1/2/2.5/3 → bundle |
| **Bundle emitter** | `src/apmode/bundle/emitter.py` | All reproducibility bundle artifacts per §5 |
| **Bundle models** | `src/apmode/bundle/models.py` | All Pydantic schemas (BackendResult, EvidenceManifest, etc.) |
| **Gate policies** | `src/apmode/governance/policy.py` | Gate 1, 2, 2.5 policy file schemas |
| **Lane policies** | `policies/*.json` | Default gate thresholds per lane |
| **CLI** | `src/apmode/cli.py` | Typer CLI: `apmode run`, `apmode validate`, `apmode inspect` |
| **Structured logging** | `src/apmode/logging.py` | structlog JSON configuration, context-bound loggers |
| **Benchmark Suite A** | `benchmarks/suite_a/`, `src/apmode/benchmarks/suite_a.py` | 7 recovery scenarios (A1-A7): classical, TMDD, covariates, NODE absorption |
| **Benchmark Suite B** | `src/apmode/benchmarks/suite_b.py` | NODE-specific validation: absorption recovery (B1), sparse data dispatch (B2), cross-paradigm ranking (B3) |

### Test suite

| Suite | Count | Description |
|-------|-------|-------------|
| Unit tests | ~620 | All modules: DSL, data, backends (nlmixr2 + NODE), search, governance, gates, routing, bundle, benchmarks (A+B), distillation, credibility |
| NODE backend tests | 74 | Constraints (32), sub-model (16), ODE (11), trainer (6), runner (6), distillation (13) |
| Gate 2.5 + cross-paradigm | 27 | ICH M15 credibility (11), cross-paradigm ranking (16) |
| Integration tests | 30 | Mock R pipeline (3), Discovery lane dispatch (12), Suite B NODE validation (15) |
| Suite A benchmark tests | 36 | Classical scenarios (24), NODE scenarios (3), specific property tests (9) |
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
uv run pytest tests/ -q                    # all 850 tests
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
| A5 | TMDD quasi-steady-state (SC mAb) | TMDD vs. 2-cmt confusion |
| A6 | 1-cmt oral + allometric WT + renal covariate | Covariate structure recovery |
| A7 | 2-cmt + NODE saturable absorption | NODE shape recovery + surrogate fidelity |

## Benchmark Suite B

NODE-specific validation (Phase 2).

| Scenario | Test | Key Assertion |
|----------|------|---------------|
| B1 | NODE absorption recovery | Mock NODE fit passes Gate 1+2 discovery |
| B2 | Sparse data + NODE dispatch | Lane Router blocks NODE when data insufficient |
| B3 | Cross-paradigm ranking | Gate 3 correctly ranks mixed nlmixr2 + jax_node candidates |

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
                            (CL, V, ka,          (dispatch constraints)
                             multi-dose              │
                             AUC_tau)                 ↓
                                    │          SearchEngine.run()
                                    ↓           ├─ Root candidates
DSL text ──→ Lark parser ──→ DSLSpec  ──→       ├─ Warm-start children
                                │               ├─ Pareto frontier
                        Semantic validator      └─ BIC scoring
                                │                      │
                        split_subjects()               ↓
                        (k-fold, LORO)     SearchEngine._select_runner()
                                            ├─ Classical → Nlmixr2Runner
                                            └─ NODE     → NodeBackendRunner
                                                       │
                                               BackendResult
                                                       │
                                          ┌────────────┼────────────┐
                                          ↓            ↓            ↓
                                    Gate 1:       Gate 2:      Gate 2.5:
                                    Technical     Lane         Credibility
                                    Validity      Admissibility (ICH M15)
                                    (7 checks)    (6 checks)   (5 checks)
                                          │            │            │
                                          └────────────┴────────────┘
                                                       │
                                                  Gate 3:
                                                  Ranking
                                            (within-paradigm BIC
                                             or cross-paradigm
                                             VPC/NPE composite)
                                                       │
                                                       ↓
                                    BundleEmitter → JSON/JSONL artifacts
                                    (evidence_manifest, initial_estimates,
                                     gate_decisions, search_trajectory,
                                     failed_candidates, candidate_lineage,
                                     ranking, credibility_report)
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
- **Phase 1 Month 3-4** (complete): Data profiler, NCA estimates, automated search, data splitting
- **Phase 1 Month 4-5** (complete): Governance gates (1+2+3), orchestrator, dispatch constraints, seed stability
- **Phase 1 Month 5-6** (complete): CLI (Typer), structlog, Lane Router + dispatch wiring, seed stability (top-k, configurable CV), ranking persistence, boundary estimate check, multi-signal state trajectory, split integrity (SplitGOFMetrics), Phase 2 prep models, Benchmark Suite A CI + integration assertions
- **Phase 2** (in progress): Hybrid NODE backend (JAX/Diffrax/Equinox), Bram-style MLP with RE on input-layer weights, 5 constraint templates, log-space mechanistic params, functional distillation (surrogate fitting + AUC/Cmax BE fidelity), Gate 2.5 ICH M15 credibility (5 checks), Gate 3 cross-paradigm ranking (VPC concordance + NPE + composite), credibility report generator, policy gate2_5 thresholds, execution_mode config, **SearchEngine multi-backend dispatch** (classical → nlmixr2, NODE → jax_node), **Discovery lane end-to-end** with both backends, **Benchmark Suite A completion** (A5 TMDD QSS, A6 covariates, A7 NODE absorption), **Benchmark Suite B** (B1-B3 NODE validation)

## Code Review Provenance

Six rounds of multi-model code review have been conducted:

| Date | Models | Focus | Key Fixes |
|------|--------|-------|-----------|
| 2026-04-13 (R1) | codex, gemini, droid, crush | DSL compiler, emitter, bundle models | IOV syntax, TMDD QSS KD/KSS, BLQ composition |
| 2026-04-13 (R2) | codex, gemini, GLM-5, MiniMax, droid | Data pipeline, NCA, search, profiler, security | A2 CMT bug, NCA terminal slope, NaN/Inf guards, route certainty, Spearman ties |
| 2026-04-13 (R3) | codex, gemini, crush, opencode, droid | Gates, search engine, orchestrator, bundle emitter | Gate pass-on-missing→fail, NaN OFV filtering, Pareto tie-handling, lane validation, broad exception catch |
| 2026-04-13 (R4) | codex, gemini, crush, opencode, droid | Lane Router, Gate 3, Gate 2.5, CLI, dispatch constraints | Warm-start constraint propagation, submission NODE exclusion validator, lane validation, CLI policy path check |
| 2026-04-13 (R5) | codex, gemini, crush, opencode, droid | Orchestrator wiring, seed stability, ranking, Phase 2 models | CV ddof=1, non-positive param plausibility, NaN-safe BIC sort, path traversal guard, seed_stability_n=1, Ranking cross-validation |
| 2026-04-13 (R6) | gemini-3.1-pro | Phase 2 NODE backend, Gate 2.5, cross-paradigm ranking | Log-space mechanistic params, trainable Saturable scale, ss_tot unbound fix, ParameterEstimate type restoration, pooled NLL docstring |

## Known Limitations (Phase 1)

- `TimeVaryingElim.decay_fn`: only `exponential` supported; `half_life` and `linear` rejected by validator
- TMDD QSS uses `KD` as approximation for `KSS = (koff + kint)/kon`; when `kint >> koff`, KD underestimates KSS and can overestimate complex formation
- `kdeg` in TMDD models is a heuristic initial estimate (`koff` for Core, `kint` for QSS), not a separately estimable parameter
- Zero-order absorption uses `dur(centr)` modeled-duration pattern; requires compatible dose event coding (CMT=central, modeled duration flag)
- NCA initial estimates: per-subject terminal-phase log-linear regression (last 3 post-Tmax points); multi-dose uses AUC_tau but requires sorted dose records
- Route certainty assessment is conservative: CMT=1 without RATE/DUR is classified as "inferred" not "confirmed"
- Gate 1 split integrity: checks train/test CWRES divergence via `SplitGOFMetrics`; `split_manifest` is passed from orchestrator to runner for all estimation and seed stability calls
- Automated search warm-start explores error model + IIV structure variants; structural/covariate expansion deferred
- SearchEngine multi-backend dispatch implemented; Discovery lane dispatches to both nlmixr2 and jax_node based on spec type
- NODE training uses pooled population NLL (no per-subject RE); Laplace approximation deferred to Phase 3
- NODE subject loop in trainer is Python-list based (not vmap); scales to ~50 subjects, not 500+
- Functional distillation Lineweaver-Burk MM fit is approximate; nonlinear least-squares fit deferred
- VPC concordance target (0.90) is hardcoded; policy-configurable target deferred
- Agentic LLM backend is Phase 3
