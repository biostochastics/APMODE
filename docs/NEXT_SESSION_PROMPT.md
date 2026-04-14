# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to continue Phase 2 completion and begin Phase 3 prep.

---

```
Continue APMODE Phase 2 completion + Phase 3 prep.

Key docs:
  - PRD_APMODE_v0.3.md — source of truth (SS4.2.4 NODE, SS4.3.1 Gates, SS5 Suites, SS8 Phase 2/3)
  - ARCHITECTURE.md — technical architecture (SS6 Phase 2/3 scope)
  - docs/plans/2026-04-13-phase2-node-discovery-lane.md — Phase 2 plan (all 13 tasks completed)
  - README.md — current status

## Current state (as of 2026-04-13)

Phase 1 complete. Phase 2 NODE backend + governance implemented.
797 tests passing, mypy --strict clean (45 files), ruff clean.
All changes UNCOMMITTED — working tree is dirty (16 modified, 17 untracked).

### Phase 2 implemented (this session):
  - JAX/Diffrax/Equinox/Optax dependencies (jax 0.9.2, diffrax 0.7.2, equinox 0.13.6)
  - NODE constraint templates: 5 types in node_constraints.py (monotone+/-, bounded, saturable, smooth)
  - Bram-style NODE sub-model: node_model.py (MLP with RE on input-layer weights)
  - Hybrid ODE system: node_ode.py (mechanistic PK skeleton + NODE, Diffrax Tsit5)
    - Log-space mechanistic params (ka, V, V2, Q) for positivity during optimization
    - Trainable Saturable.scale as jax.Array
    - Static n_cmt/node_position fields for JAX tracing compatibility
  - NODE training loop: node_trainer.py (Optax Adam, early stopping, population NLL)
  - NodeBackendRunner: node_runner.py (BackendRunner protocol, DSLSpec -> BackendResult)
  - Functional distillation: node_distillation.py (visualization, MM surrogate, AUC/Cmax BE fidelity)
  - Gate 2.5: Real ICH M15 credibility qualification (5 checks replacing Phase 1 scaffold)
    - context_of_use, limitation_to_risk, data_adequacy, sensitivity, ml_transparency
  - Gate 3: Cross-paradigm ranking (ranking.py)
    - VPC concordance + NPE + composite score for mixed-backend survivors
    - Within-paradigm BIC preserved for single-backend
    - Qualified comparison flag for cross-paradigm
  - Credibility report generator: report/credibility.py
  - Policy updates: gate2_5 thresholds in discovery.json + optimization.json
  - Orchestrator: Gate 2.5 wired between Gate 2 and Gate 3, execution_mode in RunConfig
  - Bundle models: CredibilityContext added

### Phase 2 code review applied (Gemini 3.1 Pro):
  - Log-space mechanistic params (prevents negative ka/V during optimization)
  - Trainable Saturable scale (jax.Array, not static float)
  - ss_tot computed unconditionally in distillation (prevents UnboundLocalError)
  - ParameterEstimate return type restored (fixes type erasure)
  - Pooled NLL docstring clarified (Phase 2 limitation)

### 118 new tests across 8 test files:
  - test_node_constraints.py (32)
  - test_node_model.py (16)
  - test_node_ode.py (11)
  - test_node_trainer.py (6) — slow ~14s
  - test_node_runner.py (6) — slow ~25s
  - test_node_distillation.py (13)
  - test_cross_paradigm_ranking.py (16)
  - test_credibility_report.py (7)
  - Gate 2.5 tests added to test_gates.py (11)

## IMMEDIATE: Commit the working tree

All Phase 2 work is uncommitted. First action: review changes and commit.

## Phase 2 completion (remaining items from ARCHITECTURE.md SS6):

### P2.A — SearchEngine multi-backend dispatch (HIGH PRIORITY)
SearchEngine currently only dispatches to nlmixr2. Needs:
  - Accept dict[str, BackendRunner] keyed by backend name
  - When candidate spec has NODE modules, dispatch to node_runner
  - When classical, dispatch to nlmixr2_runner
  - Orchestrator passes both runners to SearchEngine
  - Integration test: discovery lane dispatches both backends
  - File: src/apmode/search/engine.py (modify _evaluate_candidate)
  - File: src/apmode/orchestrator/__init__.py (pass node_runner to SearchEngine)
  - Test: tests/integration/test_discovery_lane.py (new)

### P2.B — Discovery lane integration test
End-to-end test with mock backends:
  - Mock Nlmixr2Runner returns classical BackendResult
  - Mock NodeBackendRunner returns NODE BackendResult
  - Orchestrator runs discovery lane with both
  - Verify: Gate 2.5 runs, cross-paradigm Gate 3 ranking produced
  - Verify: NODE excluded from submission lane
  - File: tests/integration/test_discovery_lane.py

### P2.C — NODE initial estimate strategy (ARCHITECTURE.md SS2.5)
NODE backends need pre-trained weight initialization:
  - Pre-trained weight library for 1-cmt/2-cmt reference dynamics
  - Transfer learning from classical backend's best-fit
  - File: src/apmode/backends/node_init.py (new)

### P2.D — Benchmark Suite A expansion (A5-A8) + Suite B
PRD SS5 defines 8 Suite A scenarios (we have 4):
  - A5: 2-cmt + IOV
  - A6: 1-cmt + parallel linear+MM + BLQ
  - A7: 2-cmt + NODE nonlinear absorption (ground truth custom function)
  - A8: 1-cmt + time-varying clearance + covariate
Suite B (NODE-specific):
  - B1: NODE absorption recovery
  - B2: NODE elimination under sparse data (should produce data_insufficient)
  - B3: Cross-paradigm ranking correctness
  - Files: benchmarks/suite_a/simulate_all.R, benchmarks/suite_b/ (new)
  - Files: src/apmode/benchmarks/suite_b.py (new)

### P2.E — DSL -> Stan codegen + lowering test suite
Deferred from Phase 1 to Phase 2+ per v0.3 changes:
  - Per-backend lowering test suite
  - Stan codegen emitter
  - File: src/apmode/dsl/stan_emitter.py (new)

### P2.F — Basic web UI
Minimal interface for run configuration and results viewing.
  - Not started. Consider Streamlit or Panel for MVP.

### P2.G — Flowcept + Aim integration
  - Flowcept (ORNL): W3C PROV-compliant task lineage
  - Aim v3.29: self-hosted experiment tracking
  - Both are Phase 2 scope per ARCHITECTURE.md SS2.8

## Known Phase 2 limitations to address:

1. NODE training uses pooled NLL (no per-subject RE). Laplace approximation for RE on
   input-layer weights is the next step for population-level mixed-effects NODE.

2. NODE subject loop in trainer is Python-list based (not jax.vmap). Scales to ~50
   subjects but will OOM/slow-compile for larger datasets. Fix: pad to uniform length,
   use jax.vmap over subject dimension.

3. VPC concordance target (0.90) is hardcoded in ranking.py. Should be configurable
   via GatePolicy.

4. Orchestrator auto-generates context_of_use for Gate 2.5. Real usage needs
   user-provided COU statement via CLI or config.

5. Distillation Lineweaver-Burk MM fit is approximate. Nonlinear least-squares
   (scipy.optimize.curve_fit) would be more robust.

6. dt0=0.1 hardcoded in Diffrax solver. Should use dt0=None for auto-selection.

## Phase 3 scope (PRD SS8, for planning only):

- Agentic LLM backend (DSL transforms only, <=25 iterations/run)
- Langfuse self-hosted (prompt/response/cost logging) + redaction layer
- LLM model-version escrow in bundle (agentic_trace/)
- temperature=0 + verbatim output caching
- run_lineage.json for multi-run provenance
- Optimization lane with LORO-CV
- Report generator with credibility framework
- Benchmark Suite C
- REST API

## Open questions from PRD SS10 affecting remaining work:

- Q2: Cross-paradigm NLPD comparability — simulation-based metrics implemented,
  but VPC concordance binning strategy needs pharmacometrician sign-off
- Q3: DSL extensibility process — affects Stan codegen and new NODE constraint templates
- Q4: Covariate missingness — NODE trainer uses complete-case for covariate inputs;
  full-information likelihood deferred
- Q5: Regulatory engagement timing — after Phase 2 benchmarks, before Phase 3 NODE integration

## Recommended build order for remaining Phase 2:
1. COMMIT all current work (immediate)
2. P2.A SearchEngine multi-backend dispatch
3. P2.B Discovery lane integration test
4. P2.D Suite A expansion + Suite B (validates NODE backend)
5. P2.C NODE initial estimate strategy
6. P2.E Stan codegen (parallel track)
7. P2.F Web UI (last)
8. P2.G Flowcept/Aim (can be deferred)
```
