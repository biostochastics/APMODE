# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to continue Phase 2 completion and begin Phase 3 prep.

---

```
Continue APMODE Phase 2 completion + Phase 3 prep.

Key docs:
  - PRD_APMODE_v0.3.md — source of truth (§4.2.4 NODE, §4.3.1 Gates, §5 Suites, §8 Phases)
  - ARCHITECTURE.md — technical architecture
  - README.md — current status

## Current state (as of 2026-04-14)

Phase 1 complete. Phase 2 ~70% complete.
850 tests passing, mypy --strict clean (46 files), ruff clean.
All changes committed and pushed (main branch).

### Phase 2 implemented (prior sessions):
  - JAX/Diffrax/Equinox/Optax NODE backend (6 modules: constraints, model, ode, trainer, runner, distillation)
  - Gate 2.5 ICH M15 credibility qualification (5 checks)
  - Gate 3 cross-paradigm ranking (VPC concordance + NPE + composite)
  - Credibility report generator
  - SearchEngine multi-backend dispatch (classical → nlmixr2, NODE → jax_node)
  - BackendRunner protocol updated with split_manifest
  - Orchestrator wires both runners; seed-stability dispatches to correct runner
  - Gate 2 reproducible_estimation recognizes adam/jax_node
  - Discovery lane end-to-end integration (12 tests)
  - Benchmark Suite A completion (A5 TMDD QSS, A6 covariates, A7 NODE absorption)
  - Benchmark Suite B (B1 absorption recovery, B2 sparse dispatch, B3 cross-paradigm ranking)
  - R simulation code for A5-A7 in simulate_all.R (not yet run — needs R 4.4+)

### Key files for remaining work:
  - src/apmode/search/engine.py — SearchEngine with _select_runner(), runners dict
  - src/apmode/orchestrator/__init__.py — Orchestrator with node_runner param
  - src/apmode/backends/node_runner.py — NodeBackendRunner
  - src/apmode/backends/node_trainer.py — train_node()
  - src/apmode/backends/node_distillation.py — distill_node()
  - src/apmode/governance/gates.py — evaluate_gate1/2/2_5/3
  - src/apmode/governance/ranking.py — cross-paradigm ranking
  - src/apmode/benchmarks/suite_a.py — 7 scenarios (A1-A7)
  - src/apmode/benchmarks/suite_b.py — 3 NODE validation scenarios

## Phase 2 remaining items:

### P2.C — NODE initial estimate strategy (ARCHITECTURE.md §2.5)
NODE backends need pre-trained weight initialization for faster convergence:
  - Pre-trained weight library for 1-cmt/2-cmt reference dynamics
  - Transfer learning from classical backend's best-fit parameter estimates
  - File: src/apmode/backends/node_init.py (new)
  - Tests: tests/unit/test_node_init.py (new)

### P2.D-fixture — Generate Suite A CSV fixtures for A5-A7
  - R simulation scripts are written in benchmarks/suite_a/simulate_all.R
  - Need to run: Rscript benchmarks/suite_a/simulate_all.R tests/fixtures/suite_a
  - This generates a5_tmdd_qss.csv, a6_1cmt_covariates.csv, a7_2cmt_node_absorption.csv
  - 20 integration tests are currently SKIPPED because these fixtures don't exist yet
  - Requires R 4.4+ with rxode2, jsonlite, lotri

### P2.E — DSL → Stan codegen + lowering test suite
Deferred from Phase 1 to Phase 2+ per v0.3 changes:
  - Per-backend lowering test suite (ensure DSLSpec → backend code is correct)
  - Stan codegen emitter for probabilistic backends
  - File: src/apmode/dsl/stan_emitter.py (new)
  - Tests: tests/unit/test_stan_emitter.py (new)

### P2.F — Basic web UI
Minimal interface for run configuration and results viewing:
  - Not started. Consider Streamlit or Panel for MVP.
  - Should support: upload CSV, select lane, view results, browse bundle

### P2.G — Flowcept + Aim integration
  - Flowcept (ORNL): W3C PROV-compliant task lineage
  - Aim v3.29: self-hosted experiment tracking
  - Both are Phase 2 scope per ARCHITECTURE.md §2.8
  - Can be deferred if time-constrained

## Known Phase 2 limitations to address:

1. NODE training uses pooled NLL (no per-subject RE). Laplace approximation
   for RE on input-layer weights is the Phase 3 step for population-level NODE.

2. NODE subject loop in trainer is Python-list based (not jax.vmap). Scales
   to ~50 subjects. Fix: pad to uniform length, use jax.vmap over subject dim.

3. VPC concordance target (0.90) is hardcoded in ranking.py. Should be
   configurable via GatePolicy.

4. Orchestrator auto-generates context_of_use for Gate 2.5. Real usage needs
   user-provided COU statement via CLI or config.

5. Distillation Lineweaver-Burk MM fit is approximate. Nonlinear least-squares
   (scipy.optimize.curve_fit) would be more robust.

6. dt0=0.1 hardcoded in Diffrax solver. Should use dt0=None for auto-selection.

## Phase 3 scope (PRD §8, for planning only):

- Agentic LLM backend (DSL transforms only, ≤25 iterations/run)
  - Agent operates exclusively through typed DSL transforms
  - Cannot emit raw code; every proposal validated before compilation
  - Langfuse self-hosted (prompt/response/cost logging) + redaction layer
  - temperature=0 + verbatim output caching
  - LLM model-version escrow in bundle (agentic_trace/)
  - run_lineage.json for multi-run provenance
- Optimization lane with LORO-CV
- Report generator with credibility framework (extend existing report/credibility.py)
- Benchmark Suite C (expert comparison pilot)
- REST API

## Open questions from PRD §10 affecting remaining work:

- Q2: VPC concordance binning strategy needs pharmacometrician sign-off
- Q3: DSL extensibility process — affects Stan codegen and new NODE constraint templates
- Q4: Covariate missingness — NODE trainer uses complete-case; full-information likelihood deferred
- Q5: Regulatory engagement timing — after Phase 2 benchmarks

## Recommended build order for remaining Phase 2:
1. P2.C NODE initial estimate strategy
2. P2.D-fixture Generate A5-A7 CSV fixtures (if R available)
3. P2.E Stan codegen (parallel track)
4. P2.F Web UI (last, MVP only)
5. P2.G Flowcept/Aim (can defer to Phase 3)
```
