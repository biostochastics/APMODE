<div align="center">

  # APMODE

  **Adaptive Pharmacokinetic Model Discovery Engine**

  [![Phase](https://img.shields.io/badge/phase-2%20(85%25)-blue)](PRD_APMODE_v0.3.md)
  [![Tests](https://img.shields.io/badge/tests-940%20passing-success)]()
  [![License](https://img.shields.io/badge/license-GPL--2.0--or--later-green)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.12%2B-yellow)]()
  [![mypy](https://img.shields.io/badge/mypy-strict%20%E2%9C%93-blue)]()

</div>

---

## What is APMODE?

APMODE is a **governed meta-system** that composes four population PK modeling paradigms into a single discovery workflow — so pharmacometricians can evaluate classical, automated, neural, and hybrid approaches under one roof with consistent evidence standards.

**Four paradigms, one pipeline.** Classical NLME (nlmixr2), automated structural search, hybrid mechanistic-NODE (JAX/Diffrax), and agentic LLM model construction (Phase 3) — each dispatched, evaluated, and ranked through a unified governance funnel.

**Evidence gates, not vibes.** Every candidate passes Gate 1 (technical validity), Gate 2 (lane admissibility), Gate 2.5 (ICH M15 credibility), and Gate 3 (cross-paradigm ranking) before it can be recommended. NODE models are never eligible for regulatory submission — this is a hard rule, not a tunable weight.

**Reproducibility is the unit of output.** Every run emits a versioned JSON bundle — data manifest, search trajectory, gate decisions, candidate lineage DAG, compiled specs — so any result can be audited or replayed.

**A typed PK DSL is the control surface.** Models are specified in a structured grammar (`Absorption x Distribution x Elimination x Variability x Observation`), compiled to a typed AST, validated against pharmacometric constraints, and lowered to backend-specific code. The agentic LLM backend (Phase 3) operates exclusively through DSL transforms — it cannot emit raw code.

> **Status**: Phase 2 in progress (~85%). 940 tests passing. `mypy --strict` clean. `ruff` clean. Seven rounds of multi-model code review completed; all critical and high-severity issues fixed.

---

## Quick Start

### Prerequisites

- **Python**: 3.12+
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Optional**: R 4.4+ with `nlmixr2`, `rxode2`, `jsonlite`, `lotri` for real estimation (mock R subprocess tests work without R)

### Installation

```bash
git clone https://github.com/biostochastics/apmode.git
cd apmode

# Install all dependencies
uv sync --all-extras

# Verify
uv run apmode --help
```

### Run a Discovery Pipeline

```bash
# Validate a NONMEM-format dataset
uv run apmode validate <dataset.csv>

# Run the full pipeline (ingest → profile → NCA → search → gates → bundle)
uv run apmode run <dataset.csv> --lane submission

# Inspect a reproducibility bundle
uv run apmode inspect <bundle_dir>
```

### Run the Test Suite

```bash
uv run pytest tests/ -q                    # all 940 tests
uv run mypy src/apmode/ --strict           # type checking (0 errors)
uv run ruff check src/apmode/ tests/       # linting (0 errors)
```

---

## The PK DSL

APMODE's typed domain-specific language is the single source of truth for model specification. It compiles to a validated AST and lowers to nlmixr2 R code or Stan programs.

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

| DSL Axis | Supported Forms |
|----------|----------------|
| **Absorption** | FirstOrder, ZeroOrder, Transit(n), Sequential, Mixed, WeibullAbs |
| **Distribution** | OneCmt, TwoCmt, ThreeCmt, EffectCmt, VirtualPeripheral |
| **Elimination** | Linear, MichaelisMenten, ParallelLinearMM, InductionInhibition, TimeVaryingElim |
| **Variability** | IIV (diagonal/block), IOV, CovariateLink (power/exponential/linear/step) |
| **Observation** | Proportional, Additive, Combined, LogNormal, BLQ (M1/M3/M4 composition) |

---

## Architecture

```
NONMEM CSV ──→ Ingest + Validate ──→ Canonical PK Schema (Pandera)
                                            │
                                  ┌─────────┴──────────┐
                                  ↓                    ↓
                          Data Manifest          Evidence Manifest
                          (SHA-256,              (richness, route,
                           covariates)            CL linearity, BLQ)
                                  │                    │
                                  ↓                    ↓
                          NCA Estimator          Search Space
                          (CL, V, ka,            (dispatch constraints)
                           multi-dose)                 │
                                  │                    ↓
DSL text ──→ Lark parser ──→ AST ──→        Search Engine
                          │                  ├─ Candidate generation
                  Semantic validator         ├─ Multi-backend dispatch
                          │                  │   ├─ Classical → nlmixr2
                  split_subjects()           │   └─ NODE     → JAX/Diffrax
                  (k-fold, LORO)             └─ Pareto frontier + BIC scoring
                                                       │
                                          ┌────────────┼────────────┐
                                          ↓            ↓            ↓
                                    Gate 1:       Gate 2:      Gate 2.5:
                                    Technical     Lane         ICH M15
                                    Validity      Admissibility Credibility
                                    (7 checks)    (6 checks)   (5 checks)
                                          └────────────┴────────────┘
                                                       │
                                                  Gate 3: Ranking
                                            (within-paradigm BIC or
                                             cross-paradigm VPC/NPE)
                                                       │
                                                       ↓
                                        Reproducibility Bundle (JSON/JSONL)
```

### Key Components

| Component | Path | Role |
|-----------|------|------|
| PK DSL grammar | `src/apmode/dsl/pk_grammar.lark` | Full Lark EBNF for PK model specs |
| AST models | `src/apmode/dsl/ast_models.py` | Typed Pydantic nodes for all DSL axes |
| Semantic validator | `src/apmode/dsl/validator.py` | Constraint table enforcement (PRD §4.2.5) |
| nlmixr2 emitter | `src/apmode/dsl/nlmixr2_emitter.py` | DSL AST → R code for nlmixr2/rxode2 |
| Stan emitter | `src/apmode/dsl/stan_emitter.py` | DSL AST → Stan program |
| Data pipeline | `src/apmode/data/` | Ingestion, profiling, NCA estimates, splitting |
| Classical backend | `src/apmode/backends/nlmixr2_runner.py` | Async subprocess runner with file-based IPC |
| NODE backend | `src/apmode/backends/node_*.py` | Bram-style hybrid MLP, Diffrax ODE, Optax training |
| Governance | `src/apmode/governance/` | Gates 1/2/2.5/3, cross-paradigm ranking, policy files |
| Search engine | `src/apmode/search/engine.py` | Multi-backend dispatch, BIC scoring, Pareto frontier |
| Orchestrator | `src/apmode/orchestrator/` | Full pipeline: ingest → gates → bundle |
| Bundle emitter | `src/apmode/bundle/` | All reproducibility bundle artifacts per PRD §5 |
| CLI | `src/apmode/cli.py` | Typer CLI: `run`, `validate`, `inspect` |

---

## Governance Funnel

APMODE enforces a **gated funnel** — not a weighted sum. Each gate is disqualifying; only survivors advance.

| Gate | Purpose | Checks |
|------|---------|--------|
| **Gate 1** | Technical Validity | Convergence, parameter plausibility, CWRES normality, VPC coverage, condition number, seed stability, split integrity |
| **Gate 2** | Lane Admissibility | Interpretability, shrinkage, identifiability (profile-likelihood CI), NODE exclusion (Submission lane) |
| **Gate 2.5** | Credibility Qualification | ICH M15 context-of-use, data adequacy, ML transparency, limitation-risk mapping, operational qualification |
| **Gate 3** | Ranking | Within-paradigm BIC; cross-paradigm VPC concordance + NPE + composite score |

Gate thresholds are **versioned policy artifacts** in `policies/*.json` — not hard-coded constants.

---

## Three Operating Lanes

APMODE routes work through separate pipelines with different admissible backends and evidence standards:

| Lane | Purpose | Admissible Backends | NODE Eligible? |
|------|---------|-------------------|----------------|
| **Submission** | Regulatory-grade models | Classical NLME only | No (hard rule) |
| **Discovery** | Broad exploration | Classical + NODE | Yes (Gate 2.5 required) |
| **Translational Optimization** | LORO-CV prediction (Phase 3) | All backends | Yes |

---

## Benchmark Suites

### Suite A — Structure and Parameter Recovery

Simulated PK datasets with known ground truth (PRD §5). Requires R 4.4+ with `rxode2`.

```bash
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

### Suite B — NODE-Specific Validation

| Scenario | Test | Assertion |
|----------|------|-----------|
| B1 | NODE absorption recovery | Mock NODE fit passes Gate 1+2 Discovery |
| B2 | Sparse data + NODE dispatch | Lane Router blocks NODE when data insufficient |
| B3 | Cross-paradigm ranking | Gate 3 correctly ranks mixed nlmixr2 + jax_node candidates |

---

## Test Suite

940 tests across multiple strategies:

```bash
uv run pytest tests/unit/ -q               # ~690 unit tests
uv run pytest tests/integration/ -q         # 30 integration tests (mock R pipeline + Discovery lane)
uv run pytest tests/property/ -q            # ~30 Hypothesis property-based tests
uv run pytest tests/golden/ -q              # 21 syrupy golden master snapshots
uv run pytest tests/ --snapshot-update      # update snapshots after emitter changes
```

| Category | Count | Coverage |
|----------|-------|----------|
| Unit tests | ~690 | All modules: DSL, data, backends, search, governance, routing, bundle, benchmarks |
| NODE backend | 101 | Constraints, sub-model, ODE, trainer, runner, distillation, init strategy |
| Stan codegen | 43 | Stan emitter + cross-backend lowering validation |
| Gate 2.5 + ranking | 27 | ICH M15 credibility + cross-paradigm ranking |
| Integration | 30 | Mock R pipeline, Discovery lane dispatch, Suite B NODE validation |
| Suite A benchmarks | 48 | All 7 scenarios (A1-A7) with property tests |
| Golden masters | 21 | Syrupy snapshots for pharmacometrician-validated R output |
| R syntax validation | 168 | Balanced delimiters, eta/param consistency |

---

## Phasing

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 0** | Schemas, protocols, grammar, error taxonomy, sparkid integration | Complete |
| **Phase 1** (6 months) | DSL + compiler, classical NLME backend, automated search, Gates 1-3, reproducibility bundle, CLI, Suite A | Complete (679 tests) |
| **Phase 2** (4 months) | Hybrid NODE backend, functional distillation, Discovery lane, cross-paradigm ranking, Suite B, Stan codegen | In progress (~85%, 261 new tests) |
| **Phase 3** (4 months) | Agentic LLM backend (DSL transforms only), Optimization lane + LORO-CV, report generator, Suite C, API | Planned |

---

## Pharmacometric References

- **TMDD full binding**: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532
- **TMDD QSS**: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591
- **Transit compartments**: Savic et al. (2007), J Pharmacokinet Pharmacodyn 34:711-726
- **Allometric scaling**: Anderson & Holford (2008), Clin Pharmacokinet 47:455-467
- **BLQ M3/M4**: nlmixr2 censoring via CENS/LIMIT data columns
- **NCA**: Linear trapezoidal AUC, terminal log-linear kel, CL = Dose / AUC_inf

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`PRD_APMODE_v0.3.md`](PRD_APMODE_v0.3.md) | Product requirements (current source of truth) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Technical architecture (v0.2) |
| [`CLAUDE.md`](CLAUDE.md) | Contributor/AI guidance |
| [`policies/*.json`](policies/) | Gate threshold policy files per lane |

---

## Known Limitations

- **NODE training**: Pooled population NLL (no per-subject RE); Laplace approximation deferred to Phase 3
- **NODE scaling**: Python-list subject loop (not vmap); scales to ~50 subjects, not 500+
- **Stan codegen**: Does not yet support IOV or BLQ M3/M4 (Phase 3)
- **TMDD QSS**: Uses KD as approximation for KSS; when kint >> koff, this can overestimate complex formation
- **TimeVaryingElim**: Only `exponential` decay supported; `half_life` and `linear` rejected by validator
- **Context of use**: Orchestrator auto-generates COU for Gate 2.5; production use needs user-provided COU via CLI or config
- **Agentic LLM backend**: Phase 3 scope

See the full list in [`PRD_APMODE_v0.3.md` §10](PRD_APMODE_v0.3.md).

---

## Code Review Provenance

Seven rounds of multi-model code review completed:

| Round | Models | Focus |
|-------|--------|-------|
| R1 | codex, gemini, droid, crush | DSL compiler, emitter, bundle models |
| R2 | codex, gemini, GLM-5, MiniMax, droid | Data pipeline, NCA, search, profiler, security |
| R3 | codex, gemini, crush, opencode, droid | Gates, search engine, orchestrator, bundle emitter |
| R4 | codex, gemini, crush, opencode, droid | Lane Router, Gate 3, Gate 2.5, CLI, dispatch constraints |
| R5 | codex, gemini, crush, opencode, droid | Orchestrator wiring, seed stability, ranking |
| R6 | gemini-3.1-pro | Phase 2 NODE backend, Gate 2.5, cross-paradigm ranking |
| R7 | multi-model consensus | Full Phase 2 review; all critical/high-severity issues fixed |

---

## License

Licensed under [GPL-2.0-or-later](LICENSE).

The primary engine is nlmixr2 (R), which is GPL-2 licensed. This license choice is deliberate and affects build structure from Phase 1.

---

<div align="center">

**[Quick Start](#quick-start)** &bull;
**[DSL Reference](#the-pk-dsl)** &bull;
**[Architecture](#architecture)** &bull;
**[PRD](PRD_APMODE_v0.3.md)**

</div>
