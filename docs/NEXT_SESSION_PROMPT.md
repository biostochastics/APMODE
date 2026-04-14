# APMODE Next Session Continuation Prompt

Copy this into your next Claude Code session to continue Phase 1 completion and begin Phase 2 prep.

---

```
Continue APMODE Phase 1 Month 6 completion and Phase 2 prep.

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
  - Month 5-6 (current session):
    * Lane Router wired into orchestrator + SearchEngine (allowed_backends)
    * Seed stability: top-k by BIC, uses candidate's own fitted estimates, configurable CV
    * Seed results persisted to bundle ({candidate_id}_seed_{n}_result.json)
    * Gate 3 ranking persisted to ranking.json (RankedCandidateEntry + Ranking models)
    * Gate 1 hardened:
      - Multi-signal state trajectory (R² ≥ 0.30, CWRES SD ∈ [0.5, 2.0], gradient norm ≤ 100, rounding errors)
      - Split integrity via SplitGOFMetrics (train/test CWRES divergence)
      - Boundary estimate detection (≤ 1e-4 or ≥ 1e5)
      - Non-positive param plausibility (CL=0 fails)
      - NaN-safe BIC sort in Gate 3
      - Configurable seed_stability_cv_max in policy
    * Path traversal guard in BundleEmitter
    * Phase 2 prep models: CredibilityReport, AgenticTrace*, RunLineage, ReportSummary
    * Benchmark Suite A: fixed R simulation (NONMEM EVID format), generated CSV fixtures,
      22 CI tests + 28 integration tests
    * R harness: SplitGOFMetrics computation when split_manifest provided
    * RunConfig.lane typed as Literal (removed type: ignore)
    * Ranking model cross-validation (n_survivors matches list length)
    * 5 rounds of multi-model code review (codex, gemini, crush, opencode, droid)
  - 679 tests passing, mypy strict clean, ruff clean

Repo: github.com/biostochastics/APMODE (private)

## What's left for Phase 1 completion

### 1. Pass split_manifest from orchestrator to runner for seed stability runs
- The runner now accepts split_manifest parameter
- The orchestrator needs to pass the SplitManifest to runner calls during
  seed stability and regular estimation runs
- This enables the R harness to compute SplitGOFMetrics

### 2. Data Profiler nonlinear CL detection sensitivity
- Currently fails to detect nonlinear CL in A2 (ParallelLinearMM) and A4 (MM) scenarios
- The Spearman correlation test may need tuning or additional heuristics
- Integration test documents this limitation but doesn't assert it

### 3. GitHub Actions CI finalization
- Existing ci.yml has basic lint/test/benchmark/policy-validation jobs
- Needs: astral-sh/setup-uv@v7 (currently v3), Python matrix, R-integration job
- Benchmark Suite A CI tests should be included in the test job

### 4. Phase 1 wrap-up tasks
- Update NEXT_SESSION_PROMPT.md for Phase 2 handoff
- Final multi-model review of all Month 5-6 changes
- Verify all Known Limitations in README are still accurate

## Phase 2 prep (next major phase)

### Phase 2 scope (from PRD §8, ARCHITECTURE.md §6):
- JAX/Diffrax NODE backend (NodeBackendRunner)
- GPU scheduling (Flyte 2 vs Temporal evaluation)
- Functional distillation (learned sub-function visualization, surrogate fitting)
- Gate 2.5: Credibility Qualification (ICH M15 checks)
- Gate 3: Cross-paradigm ranking (VPC concordance, AUC/Cmax BE, NPE)
- Discovery lane activation
- DSL → Stan codegen + lowering test suite
- Benchmark Suites A (full) + B
- Basic web UI

### Phase 2 models already scaffolded:
- CredibilityReport (models.py) — fields match ARCHITECTURE.md §4.4
- ReportSummary (models.py) — for report/summary.json
- SplitGOFMetrics (models.py) — for split integrity
- RunLineage, AgenticTrace* (models.py) — Phase 3 forward-compat
```
