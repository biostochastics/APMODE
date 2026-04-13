# APMODE Technical Architecture

**Version:** 0.2 — Revised per GPT-5.2-Pro review  
**Date:** 2026-04-13  
**Status:** Draft  
**Derived from:** PRD v0.3 (§3-§8), multi-model consensus (MiniMax M2.7, GLM-5, Kimi K2.5, MIMO-v2-Pro, Gemini 3.1 Pro), GPT-5.2-Pro review

---

## 1. Design Principles

1. **Custom-first, framework-ready.** Phase 1 builds a thin, purpose-built Python orchestrator. The interface contracts are designed so Flyte or Temporal can be swapped in for Phase 2 GPU scheduling without rewriting backend adapters.
2. **Process isolation is non-negotiable.** Every backend (R/nlmixr2, Python/JAX, LLM API) runs in its own process. A segfault in R must not crash the orchestrator.
3. **The DSL is the moat.** All model specifications flow through the typed PK DSL. No backend may emit or consume raw model code outside DSL transforms.
4. **Governance is a funnel, not a score.** Gates are sequential disqualifiers. Survivors are ranked; failures are logged with per-check reasons.
5. **Reproducibility is structural.** Every run emits a self-contained bundle matching PRD §4.3.2 exactly. The bundle schema is a Pydantic contract, not a framework artifact.
6. **Determinism is auditable.** All RNG seeds are escrowed. Non-deterministic boundaries (GPU, LLM) are documented, and their outputs cached.
7. **Data privacy by design.** LLM inputs never include raw row-level patient data. Observability traces are redacted of identifiers before storage.

---

## 2. Technology Stack

### 2.1 Core Stack

| Layer | Technology | License | Rationale |
|-------|-----------|---------|-----------|
| **Language (orchestrator)** | Python 3.12+ | -- | Flyte/Temporal SDKs, JAX, LLM SDKs all Python-native |
| **Language (PK engine)** | R (nlmixr2) | GPL-2 | Primary engine per PRD §4.2.2; non-negotiable |
| **Language (NODE backend)** | Python / JAX + Diffrax | Apache-2 | Phase 2; GPU-accelerated Neural ODE |
| **Package management (Python)** | uv + pyproject.toml | -- | Fast resolver, lockfile, deterministic installs |
| **Package management (R)** | renv + groundhog | GPL | renv for lockfile; groundhog for date-pinned CRAN |
| **ID generation** | sparkid | MIT | 21-char, time-sortable, monotonic, Base58; Python + Rust + JS |
| **Deployment** | Docker Compose | -- | Single-machine Phase 1; K8s deferred to Phase 2 |
| **Project license** | GPL-2-or-later | -- | Consistent with nlmixr2; compatible with Apache-2 deps (see §2.10) |

### 2.2 DSL and Compiler

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Parser** | Lark (Python, EBNF) | Pure Python; Earley/LALR modes; Transformer pattern for AST |
| **AST types** | Pydantic v2 models | Typed nodes; JSON-serializable for reproducibility bundle |
| **Semantic validator** | Custom Python (first-class compiler phase) | Constraint table enforcement: volumes > 0, rates >= 0, NODE dim <= lane ceiling, constraint_template in enumerated set |
| **Backend lowering** | Per-backend emitters (Phase 1: nlmixr2 only) | DSL AST -> R code strings; Phase 2 adds Stan codegen |
| **Testing** | Hypothesis (property-based) + pytest-syrupy (golden master) | Generate valid/invalid DSL specs; snapshot validated R output |

**Rust upgrade note:** If semantic validation exceeds ~500 LOC or compilation time impacts automated search wall-clock, consider migration to Rust via LALRPOP + PyO3 at Phase 2 start. This is a risk item (§7), not a Phase 1 requirement.

### 2.3 Orchestration

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Orchestrator** | Custom Python (asyncio + subprocess) | Gates are sequential control flow; 2-4 person team; no K8s overhead |
| **Process isolation** | `asyncio.create_subprocess_exec` + Docker SDK | Each backend in its own container; crash = task failure, not cascade. On timeout, kill entire process group (`os.killpg`). |
| **Backend interface** | `BackendRunner` protocol (Python Protocol class) | Swap-in point for Flyte tasks or Temporal activities in Phase 2 |
| **R invocation** | `Rscript` via subprocess, result written to file | See §4.2 for hardened wire contract |
| **Retry / timeout** | tenacity (Python) | Per-backend retry policy from policy file; retries write to new `attempt_id/` subdir |
| **CLI** | Typer | Local-first; `--remote` flag reserved for Phase 2 |

**Phase 2 migration path:** The `BackendRunner` protocol defines `run(spec: DSLSpec) -> BackendResult`. Phase 2 replaces subprocess calls with Flyte `@task` decorators or Temporal Activity wrappers -- the protocol interface does not change.

### 2.4 Data Ingestion and Validation

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Ingestion adapters** | Custom Python per format | Three adapters: `nonmem_csv`, `nlmixr2_event_table`, `cdisc_adam` (PRD §4.2.0) |
| **Canonical schema** | Pandera v0.18+ (DataFrameModel) | Typed columns: NMID, TIME, DV, MDV, EVID, AMT, CMT, RATE, DUR, BLQ_FLAG, LLOQ, OCCASION, STUDY_ID, plus covariate columns with time-variance metadata |
| **Policy file validation** | Pydantic v2 models | Gate threshold schemas; CI-enforced on every policy change |
| **Evidence manifest validation** | Pydantic v2 models | Typed manifest emitted by Data Profiler; consumed by Lane Router |
| **Batch validation** | `lazy=True` mode | Surface all violations in one pass, not fail-fast |

### 2.5 Initial Estimate Strategy

| Source | Method | Applies to |
|--------|--------|-----------|
| **NCA-derived** | Per-subject NCA -> CL, V, ka, t1/2; multi-cmt: macro-constants A, B, alpha, beta | Root candidates in classical + automated search |
| **Warm-start** | Parent model best-fit parameters -> child candidate initial estimates | Automated search (non-root candidates) |
| **Fallback** | Population-level naive-averaged NCA when per-subject NCA infeasible (sparse data) | All backends when individual NCA fails |
| **NODE init** | Pre-trained weight library (1-cmt/2-cmt reference dynamics) + transfer from classical best-fit | NODE backend (Phase 2) |

The `InitialEstimator` component runs between Data Profiler and Backend Dispatch. Output: `initial_estimates.json` keyed by `candidate_id`, with provenance (`source: nca | warm_start | fallback`, inputs used).

### 2.6 Reproducibility and Artifacts

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Bundle schema** | Pydantic v2 models -> JSON/JSONL | PRD §4.3.2 specifies exact file contract; no framework lock-in |
| **Run IDs** | sparkid (`generate_id()`) | Time-sortable, monotonic, 21-char; used for runs, candidates, gate decisions |
| **Data hashing** | hashlib SHA-256 | Content-addressable data manifests |
| **Bundle structure** | See §5 below | Fixed directory layout per run, matching PRD §4.3.2 exactly |
| **Validation scope** | JSON/JSONL artifacts are Pydantic-validated before writing; binary outputs (PNGs, model weights) are checksummed and referenced via manifest | Prevents silent bundle corruption |

### 2.7 Determinism and Seed Management

| Scope | Mechanism | Notes |
|-------|-----------|-------|
| **Python / NumPy** | `seedall` library | One-call seeding from root seed |
| **JAX** | `jax.random.PRNGKey` from root seed | CPU: deterministic. GPU: non-deterministic boundary (see §2.7.1) |
| **R** | `set.seed(seed)` + `RNGkind("L'Ecuyer-CMRG")` | Escrowed in bundle; `.Random.seed` state captured |
| **R environment** | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in container | Eliminates BLAS threading non-determinism |
| **LLM (Phase 3)** | `temperature=0` + verbatim output caching | Model version escrowed; re-execution flagged as non-reproducible |
| **Seed registry** | `seed_registry.json` in bundle | `{root_seed, r_seed, r_rng_kind, np_seed, jax_key, backend_seeds: {...}}` |

#### 2.7.1 GPU Non-Determinism Boundary

JAX on GPU is not guaranteed deterministic even with fixed seeds. Mitigation:

- Run config includes `execution_mode: "cpu_deterministic" | "gpu_fast"` -- stored in bundle.
- In `cpu_deterministic` mode: `JAX_PLATFORM_NAME=cpu` forces CPU execution.
- In `gpu_fast` mode: trained weights, predictions, and diagnostics are cached with composite key `(root_seed, data_hash, code_version, hardware_descriptor)`. Cached outputs are replayed for reproducibility.

### 2.8 Transparency, Logging and Observability

| Layer | Technology | Purpose | Data boundary |
|-------|-----------|---------|--------------|
| **Structured logging** | structlog (Python) | JSON-structured logs; context-bound (run_id, candidate_id, gate) | Bundle is source of truth; logs are operational mirrors |
| **Gate audit trail** | OpenTelemetry spans | Each gate decision is an OTel span with per-check evidence | References `run_id` and gate decision files in bundle |
| **Scientific provenance** | Flowcept (ORNL, Phase 2) | W3C PROV-compliant task lineage | References bundle paths |
| **LLM observability** | Langfuse (self-hosted, Phase 3) | Prompt versioning, response logging, cost tracking | Prompts redacted of subject IDs/dates before logging |
| **Experiment tracking** | Aim (v3.29, Phase 2) | Self-hosted, query runs via Python expressions | References `run_id` |
| **R backend logging** | R -> JSON to file | R process writes structured JSON log; orchestrator reads after completion | Avoids stdout contamination |

### 2.9 Data Security and Privacy

- **LLM inputs (Phase 3):** LLM prompts must contain only aggregate diagnostics, evidence manifest summaries, and DSL specs. Raw row-level patient data is never sent to LLM APIs.
- **Trace redaction:** A redaction layer strips subject identifiers, dates, and site/study free-text from all Langfuse traces and agentic_trace/ artifacts before storage.
- **Bundle access:** Bundles may contain pseudonymized subject-level data references. Access controls (filesystem permissions, encryption-at-rest) are deployment-specific and documented in the deployment guide.
- **Retention:** Langfuse traces and OTel spans follow the same retention policy as the run bundle.

### 2.10 License Compatibility

The project is licensed **GPL-2-or-later** (not GPL-2-only). This resolves compatibility with Apache-2.0 dependencies (JAX, Diffrax, OpenTelemetry, structlog, tenacity), which are compatible with GPL-3.0+ (and therefore GPL-2-or-later). All dependencies are audited in Phase 0:

| Dependency | License | GPL-2+ compatible? |
|-----------|---------|-------------------|
| nlmixr2 | GPL-2 | Yes (origin) |
| Lark | MIT | Yes |
| Pydantic | MIT | Yes |
| Pandera | MIT | Yes |
| sparkid | MIT | Yes |
| Typer | MIT | Yes |
| structlog | Apache-2/MIT | Yes |
| tenacity | Apache-2 | Yes |
| OTel SDK | Apache-2 | Yes |
| JAX + Diffrax | Apache-2 | Yes |
| Hypothesis | MPL-2.0 | Yes |

Phase 0 checklist item: full license compatibility audit with legal review.

### 2.11 Testing

| Layer | Technology | Scope |
|-------|-----------|-------|
| **Python unit tests** | pytest | Orchestrator, DSL compiler, gate logic, bundle serialization |
| **Property-based tests** | Hypothesis | Generate valid/invalid DSL programs; verify compiler rejects bad PK models |
| **Golden master tests** | pytest-syrupy | DSL -> nlmixr2 R code matches pharmacometrician-validated snapshots |
| **R unit tests** | testthat | Backend adapter correctness; nlmixr2 integration |
| **Data contract tests** | Pandera + pytest | All 3 ingestion formats; canonical schema; evidence manifest |
| **Benchmark Suite A** | Custom (in CI) | 4 synthetic recovery scenarios; pass/fail on structure recovery, parameter bias, CI coverage |
| **Benchmark Suite B** | Custom (nightly) | Semi-synthetic perturbation; BLQ burden, sparse data, protocol heterogeneity |
| **Benchmark Suite C** | Custom (quarterly) | Expert comparison; fraction-beats-median-expert >= 60% |
| **Policy file validation** | Pydantic + CI hook | Every PR changing a policy file must pass schema validation |

### 2.12 CI/CD

| Component | Technology |
|-----------|-----------|
| **CI platform** | GitHub Actions |
| **Python CI** | uv install -> pytest -> mypy -> ruff |
| **R CI** | renv restore -> R CMD check -> testthat |
| **Matrix builds** | Python 3.12/3.13 x R 4.4/4.5 |
| **Integration tests** | Docker Compose -> Suite A subset (2 scenarios) |
| **License headers** | pre-commit hook (GPL-2-or-later) |
| **License audit** | CI step: scan dependencies for license compatibility |
| **Nightly** | Full Suite A + Suite B |

---

## 3. System Architecture

```
+-----------------------------------------------------------------+
|                        CLI (Typer)                               |
|  apmode run <dataset> --lane submission --config policy.json    |
+-----------------------------+-----------------------------------+
                              |
                   +----------v----------+
                   |   Orchestrator      |
                   |   (Python asyncio)  |
                   |                     |
                   |  +---------------+  |
                   |  | Data Ingester |  |
                   |  | + Pandera     |  |
                   |  | (3 adapters)  |  |
                   |  +------+--------+  |
                   |         |           |
                   |  +------v--------+  |
                   |  | Data Profiler |  |
                   |  | -> Evidence   |  |
                   |  |    Manifest   |  |
                   |  +------+--------+  |
                   |         |           |
                   |  +------v--------+  |
                   |  | Initial       |  |
                   |  | Estimator     |  |
                   |  | (NCA/warm)    |  |
                   |  +------+--------+  |
                   |         |           |
                   |  +------v--------+  |
                   |  | Data Splitter |  |
                   |  | (per-lane     |  |
                   |  |  strategy)    |  |
                   |  +------+--------+  |
                   |         |           |
                   |  +------v--------+  |
                   |  | Lane Router   |  |
                   |  | (by intent +  |  |
                   |  |  manifest)    |  |
                   |  +------+--------+  |
                   |         |           |
                   |    +----+-----+     |
                   |    | Dispatch |     |
                   |    +----+-----+     |
                   +---------+-----------+
                             |
         +-------------------+-------------------+
         |                   |                   |
  +------v------+    +------v------+    +-------v-------+
  |  Classical   |    |  Automated  |    |  NODE / LLM   |
  |  NLME        |    |  Search     |    |  (Phase 2/3)  |
  |              |    |             |    |               |
  | [R container]|    | [R container]|   | [Py container]|
  |  nlmixr2     |    |  nlmixr2    |    |  JAX/Diffrax  |
  |  subprocess  |    |  subprocess |    |  subprocess   |
  |  file I/O    |    |  file I/O   |    |  file I/O     |
  +------+-------+    +------+------+    +-------+-------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                   +---------v----------+
                   |  Governance Layer   |
                   |                    |
                   |  Gate 1: Technical |--- OTel span (per-check)
                   |  Validity          |
                   |         |          |
                   |  Gate 2: Lane      |--- OTel span (per-check)
                   |  Admissibility     |
                   |         |          |
                   |  Gate 2.5:         |--- OTel span (per-check)
                   |  Credibility       |
                   |  Qualification     |
                   |  (Phase 2)         |
                   |         |          |
                   |  Gate 3: Ranking   |--- OTel span
                   |  (within-paradigm  |
                   |   + cross-paradigm)|
                   +---------+----------+
                             |
                   +---------v----------+
                   |  Bundle Emitter    |
                   |  (Pydantic -> JSON)|
                   |  + Report Gen      |
                   |  (Phase 2+)        |
                   +--------------------+
```

**Dispatch constraint flow:** Evidence Manifest fields directly constrain Lane Router dispatch per PRD §4.2.1. Examples: `richness_category=sparse` + inadequate absorption coverage -> NODE not dispatched; `nonlinear_clearance_signature=true` -> automated search includes MM candidates; `blq_burden > 0.20` -> all backends must use BLQ-aware likelihood.

---

## 4. Key Interface Contracts

### 4.1 BackendRunner Protocol

```python
from typing import Protocol
from pydantic import BaseModel
from enum import Enum

class BackendError(Exception):
    """Base error for all backend failures."""

class ConvergenceError(BackendError): ...
class TimeoutError(BackendError): ...
class CrashError(BackendError): ...
class InvalidSpecError(BackendError): ...

class ParameterEstimate(BaseModel):
    name: str
    estimate: float
    se: float | None
    rse: float | None          # relative standard error (%)
    ci95_lower: float | None
    ci95_upper: float | None
    fixed: bool                # was this parameter fixed?
    category: str              # "structural", "iiv", "iov", "residual"

class ConvergenceMetadata(BaseModel):
    method: str                # "saem", "focei", "jax_adam", etc.
    converged: bool
    iterations: int
    gradient_norm: float | None
    minimization_status: str   # "successful", "terminated", "boundary"
    wall_time_seconds: float

class GOFMetrics(BaseModel):
    cwres_mean: float
    cwres_sd: float
    outlier_fraction: float    # fraction |CWRES| > 4
    obs_vs_pred_r2: float | None

class VPCSummary(BaseModel):
    percentiles: list[float]   # [5, 50, 95]
    coverage: dict[str, float] # {"p5": 0.93, "p50": 0.97, "p95": 0.94}
    n_bins: int
    prediction_corrected: bool

class IdentifiabilityFlags(BaseModel):
    condition_number: float | None
    profile_likelihood_ci: dict[str, bool]  # param -> has valid profile CI
    ill_conditioned: bool

class BLQHandling(BaseModel):
    method: str                # "none", "m3", "m4"
    lloq: float | None
    n_blq: int
    blq_fraction: float

class DiagnosticBundle(BaseModel):
    gof: GOFMetrics
    vpc: VPCSummary | None
    identifiability: IdentifiabilityFlags
    blq: BLQHandling
    diagnostic_plots: dict[str, str]  # name -> relative path in bundle

class DSLSpec(BaseModel):
    """Compiled DSL specification -- typed AST serialized to Pydantic."""
    model_id: str              # sparkid
    absorption: AbsorptionModule
    distribution: DistributionModule
    elimination: EliminationModule
    variability: VariabilityModule
    observation: ObservationModule

class BackendResult(BaseModel):
    """Standardized result from any backend."""
    model_id: str
    backend: str               # "nlmixr2", "jax_node", "agentic_llm"
    converged: bool
    ofv: float | None
    aic: float | None
    bic: float | None
    parameter_estimates: dict[str, ParameterEstimate]
    eta_shrinkage: dict[str, float]
    convergence_metadata: ConvergenceMetadata
    diagnostics: DiagnosticBundle
    wall_time_seconds: float
    backend_versions: dict[str, str]
    initial_estimate_source: str  # "nca", "warm_start", "fallback"

class BackendRunner(Protocol):
    """Interface contract for all backends.
    Phase 1: subprocess implementation.
    Phase 2: Flyte @task or Temporal Activity wrapper.
    """
    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,  # from policy file
    ) -> BackendResult: ...
```

### 4.2 R Subprocess Contract (Hardened)

The R subprocess communicates via **files**, not stdout, to avoid R's stdout contamination from warnings/messages.

```
Orchestrator                              R Container
    |                                          |
    |-- write: /tmp/{request_id}/request.json -|
    |   {                                      |
    |     "schema_version": "1.0",             |
    |     "request_id": "1ocmpHE1bF...",       |
    |     "run_id": "1ocmpHE1bF...",           |
    |     "candidate_id": "1ocmpHE1bG...",     |
    |     "spec": <DSLSpec>,                   |
    |     "data_path": "/mnt/data/pk.csv",     |
    |     "seed": 42,                          |
    |     "rng_kind": "L'Ecuyer-CMRG",         |
    |     "initial_estimates": {...},           |
    |     "estimation": ["saem", "focei"]      |
    |   }                                      |
    |                                          |
    |-- exec: Rscript run_backend.R            |
    |         /tmp/{request_id}/               |
    |                                          |-- nlmixr2 fit
    |                                          |
    |                                          |-- write:
    |                                          |   /tmp/{request_id}/response.json
    |                                          |   {
    |                                          |     "schema_version": "1.0",
    |                                          |     "status": "success"|"error",
    |                                          |     "error_type": null|"convergence"|"crash",
    |                                          |     "result": <BackendResult>,
    |                                          |     "r_session_info": {...},
    |                                          |     "random_seed_state": <.Random.seed>
    |                                          |   }
    |                                          |
    |                                          |-- write:
    |                                          |   /tmp/{request_id}/logs.jsonl
    |                                          |   (structured log lines)
    |                                          |
    |<- read response.json + logs.jsonl -------|
    |                                          |
    Exit codes: 0=success, 1=R error, 137=killed, 139=segfault
    On no response.json: classify as CrashError
    On timeout: kill process group, classify as TimeoutError
```

### 4.3 Gate Interface

```python
class GateCheckResult(BaseModel):
    """Result of a single gate check (e.g., 'convergence', 'seed_stability')."""
    check_id: str              # e.g., "convergence", "cwres_mean", "seed_stability"
    passed: bool
    observed: float | bool | str  # the measured value
    threshold: float | str | None # the policy threshold
    units: str | None
    evidence_ref: str | None   # path to evidence file in bundle

class GateResult(BaseModel):
    gate_id: str               # sparkid
    gate_name: str             # "technical_validity", "lane_admissibility", etc.
    candidate_id: str          # sparkid of the model candidate
    passed: bool               # True iff ALL checks passed
    checks: list[GateCheckResult]  # per-check outcomes
    summary_reason: str        # human-readable rollup of failures
    policy_version: str        # version of the policy file used
    timestamp: str             # ISO 8601

class GateEvaluator(Protocol):
    def evaluate(
        self,
        result: BackendResult,
        policy: LanePolicy,
        lane: Lane,
    ) -> GateResult: ...
```

**Gate 2.5 Credibility Qualification (Phase 2):** Inputs required: evidence manifest fields, model complexity summary (parameter count, structural degrees of freedom), sensitivity analysis outputs. Output schema includes per-check pass/fail for: context-of-use statement, limitation-to-risk mapping, data adequacy vs model complexity, sensitivity analysis, AI/ML transparency (when NODE/agentic). Evidence pointers link to `sensitivity_analysis.json` and `context_of_use.json` in bundle.

### 4.4 Credibility Assessment Report Schema (Phase 2+)

Per PRD §4.3.3, each recommended model's report includes:

```python
class CredibilityReport(BaseModel):
    candidate_id: str
    context_of_use: str           # what decision, what risk level
    model_credibility: dict       # estimation method, convergence evidence, uncertainty
    data_adequacy: str            # evidence manifest vs model complexity assessment
    limitations: list[str]        # known boundaries, population restrictions
    ml_transparency: str | None   # role of ML, guardrails, mechanistic validation (if applicable)
    sensitivity_results: dict     # parameter perturbation -> decision stability
    evidence_refs: dict[str, str] # links to bundle artifacts used
```

---

## 5. Reproducibility Bundle Structure

Matches PRD §4.3.2 exactly. Names are canonical and must not drift.

```
runs/
+-- {run_id}/                              # sparkid
    |-- data_manifest.json                 # SHA-256, column mapping, BLQ coding, ingestion format
    |-- split_manifest.json                # subject-level train/test/validation assignments, fold seed
    |-- evidence_manifest.json             # Data Profiler output
    |-- initial_estimates.json             # per-candidate NCA/warm-start estimates with provenance
    |-- seed_registry.json                 # {root_seed, r_seed, r_rng_kind, np_seed, jax_key, ...}
    |-- policy_file.json                   # versioned gate thresholds (copy)
    |-- backend_versions.json              # nlmixr2, R, Python, JAX, container image digests, git SHA
    |-- search_trajectory.jsonl            # ordered log: DSL spec, backend, fit status, scores, gates
    |-- failed_candidates.jsonl            # rejected models with per-check gate failures
    |-- candidate_lineage.json             # DAG of candidate parentage
    |-- gate_decisions/
    |   |-- gate1_{candidate_id}.json      # GateResult with per-check outcomes
    |   |-- gate2_{candidate_id}.json
    |   |-- gate2_5_{candidate_id}.json    # Phase 2
    |   +-- gate3_{candidate_id}.json
    |-- compiled_specs/
    |   |-- {candidate_id}.json            # DSL AST (Pydantic-serialized)
    |   +-- {candidate_id}.R               # lowered R code
    |-- results/
    |   |-- {candidate_id}_result.json     # BackendResult
    |   +-- {candidate_id}_diagnostics/
    |       |-- vpc.png
    |       |-- gof.png
    |       +-- cwres.png
    |-- agentic_trace/                     # Phase 3
    |   |-- {iteration_id}_input.json      # redacted LLM input
    |   |-- {iteration_id}_output.json     # verbatim LLM output
    |   +-- {iteration_id}_meta.json       # model ID, version, prompt hash, tokens, cost
    |-- run_lineage.json                   # Phase 3: links to prior run IDs
    |-- report_provenance.json             # timestamps, component versions, generator identity
    +-- report/                            # Phase 2+
        |-- summary.json                   # structured summary
        +-- {candidate_id}_credibility.json  # CredibilityReport per recommended model
```

JSON/JSONL artifacts are Pydantic-validated before writing. Binary outputs (PNGs, model weights) are checksummed and referenced via manifest entries in the parent JSON.

---

## 6. Phased Build Sequence

### Phase 0 (2 weeks) -- Decisions and Contracts

- [ ] **License compatibility audit** (GPL-2-or-later vs all deps)
- [ ] Data format canonical schema (Pandera DataFrameModel for all 3 ingestion formats)
- [ ] R subprocess request/response JSON schema (Pydantic) with schema versioning
- [ ] Gate policy file schema (Pydantic) + CI validation hook
- [ ] Reproducibility bundle schema (Pydantic models for ALL artifacts in §5)
- [ ] Error taxonomy: `BackendError -> {ConvergenceError, TimeoutError, CrashError, InvalidSpecError}`
- [ ] sparkid integration -- ID format for runs, candidates, gates
- [ ] `BackendRunner` protocol definition
- [ ] DSL grammar draft (Lark EBNF) -- parse-only, no lowering

### Phase 1 (6 months) -- Core Platform

**Month 1-2: DSL + Compiler + Scaffolding**
- [ ] Lark grammar for full PK DSL (§4.2.5)
- [ ] Pydantic AST models (typed nodes for each module)
- [ ] Semantic validator (constraint table, dim ceilings, lane admissibility)
- [ ] nlmixr2 lowering emitter (DSL AST -> R code)
- [ ] Hypothesis property-based tests for grammar
- [ ] pytest-syrupy golden master tests for lowering
- [ ] **Minimal bundle emitter scaffolding** (data_manifest, seed_registry, compiled_specs, backend_versions) -- supports integration testing from Month 2

**Month 2-3: Classical NLME Backend**
- [ ] R Docker container with nlmixr2 + renv lockfile + `OMP_NUM_THREADS=1`
- [ ] `NlmixrBackendRunner` implementing `BackendRunner` protocol
- [ ] Hardened file-based I/O contract (§4.2)
- [ ] SAEM + FOCEI estimation
- [ ] Standardized `BackendResult` extraction (all fields including GOF, VPC, identifiability)
- [ ] testthat tests for R adapter
- [ ] Crash recovery: process group kill on timeout, typed error classification, retry to new attempt_id/

**Month 3-4: Data Pipeline + Automated Search**
- [ ] Data Ingester (3 adapters: NONMEM, nlmixr2 eventTable, CDISC ADaM)
- [ ] Pandera canonical schema validation
- [ ] Data Profiler -> Evidence Manifest
- [ ] Initial Estimator (NCA-derived + warm-start from parent)
- [ ] Data Splitter (subject-level for classical; split_manifest.json)
- [ ] Automated search: structural x covariate x random effects x residual error
- [ ] Candidate generation from DSL combinatorics
- [ ] Scoring: AIC/BIC for nested; cross-validated NLPD for non-nested
- [ ] Pareto frontier (parsimony vs. fit)
- [ ] Search trajectory logging (JSONL)

**Month 4-5: Governance Layer**
- [ ] Gate 1: Technical validity (7 checks per PRD: convergence, parameter plausibility, state trajectory, CWRES, VPC coverage, split integrity, seed stability)
- [ ] Gate 2: Lane-specific admissibility (per PRD table: interpretability, shrinkage, identifiability, NODE exclusion for Submission)
- [ ] Gate evaluators with per-check `GateCheckResult` and OTel span instrumentation
- [ ] Policy file loading + Pydantic validation
- [ ] Failed candidate logging with per-check structured reasons

**Month 5-6: Orchestrator + CLI + Integration**
- [ ] Orchestrator: asyncio event loop, `asyncio.create_subprocess_exec`, BackendRunner dispatch, gate pipeline, cancellation/cleanup semantics
- [ ] Lane Router (by intent + evidence manifest dispatch constraints per PRD §4.2.1)
- [ ] Full reproducibility bundle emitter (all artifacts in §5)
- [ ] Typer CLI: `apmode run`, `apmode validate`, `apmode inspect`
- [ ] structlog integration (JSON structured logs)
- [ ] Benchmark Suite A (4 synthetic scenarios in CI)
- [ ] Suite A pass/fail assertions in GitHub Actions

### Phase 2 (4 months) -- NODE + Discovery Lane

- [ ] JAX/Diffrax NODE backend (`NodeBackendRunner`)
- [ ] GPU scheduling -- evaluate Flyte 2 vs Temporal for `BackendRunner` migration
- [ ] `execution_mode` config: `cpu_deterministic` | `gpu_fast`
- [ ] Functional distillation (learned sub-function visualization, surrogate fitting)
- [ ] Gate 2.5: Credibility Qualification (ICH M15 checks per PRD §4.3.1)
- [ ] Gate 3: Cross-paradigm ranking (VPC coverage concordance, AUC/Cmax bioequivalence, NPE)
- [ ] Discovery lane activation
- [ ] Credibility Assessment Report generator (§4.4)
- [ ] Flowcept integration (W3C PROV provenance)
- [ ] Aim experiment tracking
- [ ] DSL -> Stan codegen + lowering test suite
- [ ] Benchmark Suites A (full) + B
- [ ] Basic web UI

### Phase 3 (4 months) -- Agentic LLM + Optimization Lane

- [ ] Agentic LLM backend (DSL transforms only, <= 25 iterations)
- [ ] Langfuse self-hosted (prompt/response/cost logging) + redaction layer
- [ ] LLM model-version escrow in bundle (`agentic_trace/`)
- [ ] `temperature=0` + verbatim output caching
- [ ] `run_lineage.json` for multi-run provenance
- [ ] Optimization lane with LORO-CV (leave-one-regimen-out)
- [ ] Data Splitter: regimen-level splitting for Optimization lane
- [ ] Report generator with credibility framework
- [ ] Benchmark Suite C
- [ ] REST API

---

## 7. Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| **R segfault crashes orchestrator** | Process isolation via subprocess + Docker; kill process group on timeout; typed error classification; retry to new attempt_id/ |
| **R stdout contamination** | File-based I/O contract (§4.2); R sinks stdout/stderr; orchestrator reads response.json |
| **Custom orchestrator becomes tech debt** | `BackendRunner` protocol provides clean swap-in boundary for Flyte/Temporal |
| **DSL grammar too rigid** | Extensibility via new module types (PRD §10 Q1); Lark EBNF is easy to extend |
| **JAX GPU non-determinism** | `execution_mode` flag; CPU-only for reproducibility; GPU outputs cached with composite key |
| **LLM provider model versioning** | Model version escrowed; output caching for replay; multi-run provenance |
| **License incompatibility** | GPL-2-or-later; Phase 0 audit; CI license scanner |
| **PHI/PII leakage in LLM traces** | Redaction layer; LLM inputs limited to aggregate diagnostics/DSL specs |
| **Phase 0 scope creep** | Fixed 2-week timebox; schemas and protocols only, no implementation |
| **Phase 1 Month 5-6 overload** | Bundle scaffolding moved to Month 1-2; data pipeline moved to Month 3-4 |
| **Cross-paradigm NLPD incomparability** | Deferred to Phase 2; simulation-based metrics used instead |
| **Retry non-idempotency** | Each retry writes to new `attempt_id/` subdir; only final attempt promoted |

---

## 8. Dependency Summary

### Python (pyproject.toml)

```
[project]
requires-python = ">=3.12"
license = "GPL-2.0-or-later"
dependencies = [
    "lark>=1.2",              # MIT
    "pydantic>=2.7",          # MIT
    "pandera>=0.18",          # MIT
    "sparkid>=0.2",           # MIT
    "typer>=0.12",            # MIT
    "structlog>=24.1",        # Apache-2/MIT
    "tenacity>=8.3",          # Apache-2
    "opentelemetry-api>=1.25",  # Apache-2
    "opentelemetry-sdk>=1.25",  # Apache-2
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "hypothesis>=6.100",      # MPL-2.0
    "pytest-syrupy>=4.0",     # Apache-2
]
```

### R (renv.lock -- key packages)

```
nlmixr2 >= 3.0      # GPL-2
rxode2 >= 3.0        # GPL-2
jsonlite             # MIT
testthat             # MIT
```

---

## 9. Open Decisions (to resolve in Phase 0)

1. **DSL extensibility process** -- how are new module types proposed, reviewed, and added? (PRD §10 Q1)
2. **Covariate missingness strategy** -- full-information likelihood vs. multiple imputation when `fraction_incomplete > 0.15` (PRD §10 Q3)
3. **Initial estimate strategy specifics** -- allometric defaults, literature priors, or data-driven NCA? (PRD §4.2.0.1; component is architected in §2.5, strategy is open)
4. **Phase 2 orchestrator selection** -- Flyte 2 vs Temporal; decision deferred to end of Phase 1 based on team experience and GPU infrastructure availability
5. **Regulatory engagement timing** -- when to seek informal FDA/EMA feedback on credibility framework (PRD §10 Q4)
