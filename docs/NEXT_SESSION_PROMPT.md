# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to continue Phase 1 completion.

---

```
Continue APMODE Phase 1 Month 5-6 completion.

Key docs:
  - PRD_APMODE_v0.3.md — source of truth
  - ARCHITECTURE.md — technical architecture
  - README.md — current status

Previous sessions completed:
  - Month 1-2: DSL compiler, nlmixr2 emitter, bundle scaffolding
  - Month 2-3: Classical NLME backend, R harness, data ingestion, benchmarks
  - Month 3-4: Data Profiler (AMT-based nonlinear CL), NCA (multi-dose AUC_tau,
    extrapolation fraction), automated search (SearchEngine + Pareto frontier),
    data splitter, E2E integration test
  - Month 4-5: Governance gates (Gate 1: 7 checks, Gate 2: 6 checks, Gate 3: ranking),
    dispatch constraints (BLQ→M3/M4, heterogeneous→IOV), seed-stability multi-run,
    Lane Router, Gate 2.5 scaffold, report provenance, Typer CLI, structlog
  - 604 tests passing, mypy strict clean, ruff clean
  - 4 rounds of multi-model code review (codex, gemini, crush, opencode, droid)

Repo: github.com/biostochastics/APMODE (private)

## What's left for Phase 1 Month 5-6 completion

### 1. Wire Lane Router into Orchestrator
- Replace hardcoded backend selection with routing.route() call
- Use DispatchDecision.backends to filter search space
- Log constraints from dispatch decision

### 2. Seed stability for all gate-passing candidates (not just top 3)
- Current: only top 3 by BIC get seed stability runs
- Gemini review flagged: all non-top-3 candidates auto-fail Gate 1
- Fix: either run seeds for all, or make seed stability optional for non-top

### 3. Persist seed stability results to bundle
- Currently seed runs used but not written to results/
- Need: {candidate_id}_seed_{n}_result.json or similar

### 4. Gate 3 ranking persistence
- Write ranking.json with full ordered candidate list
- Currently only top candidate's GateResult written

### 5. Full Benchmark Suite A in CI
- 4 scenarios with structure recovery assertions
- Parameter bias checks (within 20% of ground truth for mock)
- Suite A subset as CI integration test

### 6. Phase 2 prep models
- CredibilityReport Pydantic model (ARCHITECTURE.md §4.4)
- Agentic trace models (AgenticTraceInput/Output/Meta)
- RunLineage model
- ReportSummary model

### 7. Remaining test gaps (from multi-model review)
- Gate 1: dedicated tests for parameter_plausibility, state_trajectory
- Gate 3: tie-breaking test (equal BIC)
- Lane Router: boundary values (blq_burden=0.20 exactly)
- Dispatch: compound constraints (BLQ + heterogeneous simultaneously)

### 8. Documentation
- Update CLAUDE.md with build/test commands (per CLAUDE.md instruction)
- Ensure README phasing section is current
```
