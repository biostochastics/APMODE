# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to finish Phase 2 and begin Phase 3.

---

```
Finish APMODE Phase 2 + begin Phase 3.

Key docs:
  - PRD_APMODE_v0.3.md — source of truth (§4.2.4 NODE, §4.3.1 Gates, §5 Suites, §8 Phases)
  - ARCHITECTURE.md — technical architecture
  - CHANGELOG.md — keepachangelog format, current session's changes
  - README.md — current status

## Current state (as of 2026-04-14)

Phase 1 complete. Phase 2 ~90% complete.
949 tests passing, mypy --strict clean (50 files), ruff clean.
Pre-commit hooks active (ruff format, ruff check, mypy strict, trailing-ws, etc.)
Automated versioning via hatch-vcs (v0.1.0 tag at Phase 1 completion).
All changes committed and pushed (main branch).

### Phase 2 fully implemented:
  - JAX/Diffrax/Equinox NODE backend (7 modules: constraints, model, ode, trainer, runner, distillation, init)
  - NODE initial estimate strategy: pre-trained weight library + transfer learning
  - Independent CL parameter in HybridPKODE (ka-as-CL proxy fixed)
  - Gate 2.5 ICH M15 credibility qualification (5 checks)
  - Gate 3 cross-paradigm ranking (VPC concordance + NPE, configurable target)
  - SearchEngine multi-backend dispatch (classical → nlmixr2, NODE → jax_node)
  - Discovery lane end-to-end integration
  - Benchmark Suite A complete (A1-A7, all CSV fixtures generated via rxode2)
  - Benchmark Suite B (B1-B3 NODE validation)
  - Stan codegen emitter (IOV, BLQ M3/M4, covariates, LOO-CV log_lik)
  - Per-backend lowering test suite (nlmixr2 vs Stan structural parity)
  - CLI: run, explore, datasets (14 public datasets), log, diff, inspect, validate
  - Dataset local cache, exit code documentation
  - Pre-commit hooks, automated versioning, CHANGELOG.md

## Phase 2 remaining items:

### P2.F — Basic web UI (Streamlit MVP)
Minimal interface for run configuration and results viewing:
  - Not started. Consider Streamlit or Panel for MVP.
  - Should support: upload CSV, select lane, view results, browse bundle
  - Could reuse the Rich panels from `explore` as Streamlit components

### P2.G — Flowcept + Aim integration (deferrable)
  - Flowcept (ORNL): W3C PROV-compliant task lineage
  - Aim v3.29: self-hosted experiment tracking
  - Can be deferred to Phase 3 if time-constrained

### Remaining CLI polish:
  - Per-candidate live progress during search (Rich Live display)
  - `validate`: cross-file consistency checks (bundle integrity)
  - Dry-run / `--plan` mode (show what would run without executing)

## Phase 3 scope (PRD §8):

### P3.A — Agentic LLM backend (primary deliverable)
  - Agent operates exclusively through typed DSL transforms
  - Cannot emit raw code; every proposal validated before compilation
  - ≤25 iterations/run, temperature=0 + verbatim output caching
  - Langfuse self-hosted (prompt/response/cost logging) + redaction layer
  - LLM model-version escrow in bundle (agentic_trace/)
  - run_lineage.json for multi-run provenance
  - File: src/apmode/backends/agentic_runner.py (new)
  - Key constraint: agent can only propose DSLSpec transforms, not raw code

### P3.B — Optimization lane with LORO-CV
  - Leave-one-run-out cross-validation for predictive performance
  - All backends eligible (classical + NODE + agentic)
  - Uses data splitter's LORO mode (already implemented)

### P3.C — Report generator with credibility framework
  - Extend existing report/credibility.py
  - Generate regulatory-ready PDF/HTML reports
  - Include ICH M15 credibility assessment, parameter tables, VPC plots

### P3.D — Benchmark Suite C (expert comparison pilot)
  - Primary metric: fraction-beats-median-expert (≥60%)
  - NLPD gap demoted to secondary metric
  - Requires external pharmacometrician expert models for comparison

### P3.E — REST API
  - FastAPI wrapper around Orchestrator
  - Async run submission, status polling, bundle download
  - Authentication, rate limiting

## Known limitations to address:
1. NODE training: pooled NLL (no per-subject RE); Laplace approximation needed
2. NODE scaling: Python-list subject loop (not vmap); pad + vmap for 500+ subjects
3. Stan codegen: maturation covariate form not yet supported
4. Orchestrator auto-generates COU for Gate 2.5; needs user-provided COU via CLI

## Open questions from PRD §10:
- Q2: VPC concordance binning strategy needs pharmacometrician sign-off
- Q3: DSL extensibility process — affects new NODE constraint templates
- Q4: Covariate missingness — NODE trainer uses complete-case
- Q5: Regulatory engagement timing — after Phase 2 benchmarks

## Recommended build order for remaining work:
1. P2.F Streamlit web UI (finishes Phase 2)
2. P3.A Agentic LLM backend (core Phase 3 deliverable)
3. P3.B Optimization lane + LORO-CV
4. P3.C Report generator
5. P3.E REST API
6. P3.D Suite C (requires external data)
7. P2.G Flowcept/Aim (can slot in anywhere)
```
