# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to continue Phase 2 completion and begin Phase 3 prep.

---

```
Continue APMODE Phase 2 completion + Phase 3 prep.

Key docs:
  - PRD_APMODE_v0.3.md — source of truth (§4.2.4 NODE, §4.3.1 Gates, §5 Suites, §8 Phases)
  - ARCHITECTURE.md — technical architecture
  - README.md — current status

## Current state (as of 2026-04-13)

Phase 1 complete. Phase 2 ~85% complete.
940 tests passing, mypy --strict clean (48 files), ruff clean.
All changes committed and pushed (main branch).

### Phase 2 implemented (all sessions):
  - JAX/Diffrax/Equinox NODE backend (6 modules: constraints, model, ode, trainer, runner, distillation)
  - NODE initial estimate strategy (node_init.py): pre-trained weight library + transfer learning
  - Gate 2.5 ICH M15 credibility qualification (5 checks)
  - Gate 3 cross-paradigm ranking (VPC concordance + NPE + composite, configurable target)
  - Credibility report generator
  - SearchEngine multi-backend dispatch (classical → nlmixr2, NODE → jax_node)
  - BackendRunner protocol updated with split_manifest
  - Orchestrator wires both runners; seed-stability dispatches to correct runner
  - Gate 2 reproducible_estimation recognizes adam/jax_node
  - Discovery lane end-to-end integration (12 tests)
  - Benchmark Suite A complete: A1-A7 all CSV fixtures generated via rxode2
  - Benchmark Suite B (B1-B3 NODE validation)
  - Stan codegen emitter (stan_emitter.py) + per-backend lowering test suite (43 tests)
  - R simulation script fixed for rxode2 5.x compatibility
  - Profiler: TMDD inverse curvature detection, read-only array fix
  - Distillation: scipy curve_fit for MM surrogate (replaces Lineweaver-Burk)
  - Diffrax dt0=None for auto step-size selection
  - VPC concordance target configurable via GatePolicy.vpc_concordance_target
  - BIC uses n_observations (not n_subjects) in NodeBackendRunner
  - CLI hardening (rich.markup.escape, validation improvements)

### Key files for remaining work:
  - src/apmode/dsl/stan_emitter.py — Stan codegen (IOV/BLQ/maturation not yet supported)
  - src/apmode/backends/node_init.py — Weight library + transfer learning
  - src/apmode/backends/node_ode.py — NOTE: absorption mode uses ka as CL proxy (design limitation)
  - src/apmode/governance/ranking.py — rank_cross_paradigm with vpc_concordance_target
  - src/apmode/governance/policy.py — GatePolicy.vpc_concordance_target field

## Phase 2 remaining items:

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

3. Orchestrator auto-generates context_of_use for Gate 2.5. Real usage needs
   user-provided COU statement via CLI or config.

4. NODE absorption mode uses ka as CL proxy (node_ode.py lines 117-118).
   HybridPKODE needs a log_CL field to support NODE-absorption independently.
   Pre-existing design limitation, not a regression.

5. Stan codegen does not yet support IOV, BLQ M3/M4, or maturation covariates.

6. Weight library singleton (node_init.py) persists across test runs. Consider
   adding a reset function for test isolation.

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
1. P2.F Web UI (Streamlit MVP)
2. P2.G Flowcept/Aim (can defer to Phase 3)
3. Address NODE absorption ka-as-CL limitation if blocking
4. Stan codegen maturation/IOV extensions (can be Phase 3)
```
