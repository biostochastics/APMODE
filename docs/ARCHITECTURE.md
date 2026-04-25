# APMODE Technical Architecture

**Version:** 0.3
**Date:** 2026-04-25
**Status:** Current (tracks APMODE 0.6.1-rc1; Phase 3 in progress)
**Derived from:** PRD v0.3 (§3–§8)
**Supersedes:** v0.2 (2026-04-13). Change summary: Phase 2/3 framing removed where
shipped (Bayesian backend, NODE, agentic LLM, Gate 2.5, FREM, Gate 3 ranking all
active); observability stack re-grounded in the real dependency set; bundle layout
regenerated from `bundle/emitter.py`; contracts regenerated from `bundle/models.py`;
speculative migrations (Flyte/Temporal, Docker-Compose, Rust-parser, Langfuse/Aim/
Flowcept) moved to §10 Non-goals.

---

## 1. Design Principles

1. **Custom-first, framework-ready.** A thin purpose-built Python orchestrator (asyncio
   + subprocess) is the main loop. The `BackendRunner` protocol is designed so Flyte
   or Temporal could be swapped in for GPU scheduling without rewriting adapters, but
   no such migration is planned (see §10).
2. **Process isolation is non-negotiable.** Every backend boundary (R/nlmixr2,
   Python/JAX, LLM SDK calls, cmdstanpy/CmdStan) is isolated from the orchestrator
   where practical. A segfault in R must not crash the
   orchestrator.
3. **The DSL is the moat.** All model specifications flow through the typed PK DSL
   (Formular). No backend emits or consumes model code outside of a deterministic
   function of the compiled `DSLSpec`.
4. **Governance is a funnel, not a score.** Gates are sequential disqualifiers
   (Gate 1 → Gate 2 → Gate 2.5 → Gate 3). Survivors are ranked; failures are logged
   with per-check reasons in the bundle.
5. **Reproducibility is structural.** Every run emits a self-contained bundle that
   is atomically sealed with a `_COMPLETE` sentinel carrying a SHA-256 digest of
   every artifact. The bundle schema is a Pydantic contract (`bundle/models.py`).
6. **Determinism is auditable.** All RNG seeds are escrowed in `seed_registry.json`.
   Non-deterministic boundaries (GPU, LLM) are documented and their outputs cached.
7. **Data privacy by design.** LLM inputs never include raw row-level patient data —
   the allow-list gate in `diagnostic_summarizer.redact_for_llm()` is the single
   enforcement point; unknown fields fail closed.

---

## 2. Technology Stack

### 2.1 Core stack — what is actually installed

Source of truth: `pyproject.toml`. Core dependencies install with `uv sync`; extras
(`node`, `bayesian`, `llm`, `test`, `dev`) are opt-in.

| Role | Package | Extra | License |
|------|---------|-------|---------|
| Orchestrator language | Python 3.12–3.14 | core | — |
| Primary PK engine | R 4.4+ / nlmixr2 / rxode2 | — (subprocess) | GPL-2 |
| Parser | `lark` ≥ 1.2 | core | MIT |
| AST / schemas | `pydantic` ≥ 2.7 | core | MIT |
| Data schemas | `pandera` ≥ 0.18 | core | MIT |
| IDs | `sparkid` ≥ 0.2 | core | MIT |
| CLI | `typer` ≥ 0.12 | core | MIT |
| Structured logging | `structlog` ≥ 24.1 | core | Apache-2 / MIT |
| Numerics | `numpy` ≥ 1.25, `pandas` ≥ 2.0, `scipy` ≥ 1.11 | core | BSD-3 |
| Terminal UI | `rich` ≥ 13 | core | MIT |
| Bayesian backend | `cmdstanpy` ≥ 1.2, `arviz` ≥ 0.17, `pyarrow` ≥ 15 | `bayesian` | BSD-3 / Apache-2 |
| NODE backend | `jax[cpu]`, `diffrax`, `equinox`, `optax`, `jaxtyping` | `node` | Apache-2 / MIT |
| LLM providers | `anthropic`, `openai`, `google-genai`, `ollama`, `litellm` | `llm` | MIT / Apache-2 |
| Tests | `pytest`, `pytest-xdist`, `pytest-asyncio`, `hypothesis`, `syrupy` | `test` | MIT / MPL-2 / Apache-2 |
| Dev tooling | `ruff`, `mypy`, `pre-commit`, `bandit`, `pip-audit` | `dev` | MIT |

**Project license:** GPL-2-or-later (compatible with nlmixr2 GPL-2 and Apache-2
dependencies).

### 2.2 DSL and compiler

See `docs/FORMULAR.md` for the language reference.

| Component | Technology |
|-----------|-----------|
| Parser | Lark (Earley mode) via `src/apmode/dsl/grammar.py` (10 KB input guard) |
| AST | Pydantic v2 models in `src/apmode/dsl/ast_models.py` (6 fields: 5 grammar blocks + `priors: list[PriorSpec]`) |
| Parse-tree → AST | `src/apmode/dsl/transformer.py` |
| AST canonicalization | `src/apmode/dsl/normalize.py` |
| Semantic validator | `src/apmode/dsl/validator.py::validate_dsl(spec, lane=...)` — lane-aware, non-fail-fast |
| Priors + admissibility | `src/apmode/dsl/priors.py` (families, target taxonomy, `_VALID_FAMILIES` matrix) |
| Transforms | `src/apmode/dsl/transforms.py` (9) + `prior_transforms.py::SetPrior` (1) — 10 total |
| Emitters | `nlmixr2_emitter.py`, `stan_emitter.py`, `frem_emitter.py` |
| Testing | Hypothesis property tests + syrupy golden masters for emitter output |

### 2.3 Orchestration

| Component | Technology |
|-----------|-----------|
| Orchestrator | Custom Python (asyncio) in `src/apmode/orchestrator/` — sequential gate control flow |
| Process isolation | `asyncio.create_subprocess_exec`; on timeout, kill the full process group (`os.killpg`) |
| Backend interface | `BackendRunner` Protocol (`src/apmode/backends/protocol.py`) |
| R invocation | `Rscript src/apmode/r/harness.R` via subprocess; file-based I/O (§4.2) |
| Stan invocation | `cmdstanpy.CmdStanModel` via `src/apmode/bayes/harness.py` wrapped by `bayesian_runner.py` |
| Retry/timeout | Bespoke logic in each runner; timeout from policy file, killed attempts write to new `attempt_id/` subdir |
| CLI | Typer (`src/apmode/cli.py`) — 16 direct commands plus registered `bundle` / `completion` groups |

**Note on deployment posture.** The codebase runs natively on the user's machine via
`uv`; there is no Docker-Compose stack, no R container, no K8s. Users who want
containerization manage it themselves.

### 2.4 Data ingestion and validation

| Component | Path |
|-----------|------|
| Ingestion | `src/apmode/data/ingest.py` + `adapters.py` (NONMEM CSV primary; others extend via adapter) |
| Canonical schema | Pandera schemas in `src/apmode/data/schema.py` |
| Dose expansion (ADDL/II, infusions) | `src/apmode/data/dosing.py` |
| Profiler (Evidence Manifest) | `src/apmode/data/profiler.py` — structured `nonlinear_clearance_signals` |
| Profiler policy | `src/apmode/data/policy.py` loads `policies/profiler.json`; `policy_sha256` embedded in every `EvidenceManifest` |
| NCA + initial estimates | `src/apmode/data/initial_estimates.py` — Huang 2025 λz selector, linear-up/log-down AUC, unit-aware CL heuristic |
| Data splitter (k-fold, LORO-CV) | `src/apmode/data/splitter.py` |
| Missing-data directive resolver | `src/apmode/data/missing_data.py::resolve_directive` |
| Multiple-imputation providers (R-backed) | `src/apmode/data/imputers.py` (`R_MiceImputer`, `R_MissRangerImputer`) + `src/apmode/r/impute.R` |
| Categorical-encoding auto-remap | `src/apmode/data/categorical_encoding.py::auto_remap_binary_columns` |

### 2.5 Initial-estimate strategy

| Source | Method | Applies to |
|--------|--------|-----------|
| NCA-derived | Per-subject PKNCA-style (λz by curve-stripping + adj-R² tie-break, linear-up/log-down AUC) with QC gates | Root candidates |
| Warm-start | Parent's best-fit parameters → child | Phase 3 warm-start children |
| Fallback | Population-median NCA when ≥50 % of per-subject fits fail QC; `RunConfig.fallback_estimates` (dataset-card `published_model.key_estimates`) or conservative defaults | Degraded data |
| NODE init | Pre-trained weight library + transfer from classical fits | NODE backend |

Per-subject diagnostics are emitted as `nca_diagnostics.jsonl`; the unit-aware scale
factor (when applied) is recorded as `_unit_scale_applied` in
`initial_estimates.json`.

### 2.6 Reproducibility and artifacts

| Component | Technology |
|-----------|-----------|
| Bundle schema | Pydantic v2 models in `src/apmode/bundle/models.py` → JSON/JSONL |
| Bundle emitter | `src/apmode/bundle/emitter.py` (per-artifact `write_*` methods) |
| Run / candidate IDs | `sparkid.generate_id()` (`src/apmode/ids.py`) |
| Data hashing | `hashlib.sha256` over content |
| Atomic seal | `BundleEmitter.seal()` writes `_COMPLETE` sentinel with SHA-256 digest of every file; `apmode validate` refuses unsealed bundles |
| Schema version | `_COMPLETE_SCHEMA_VERSION = 2` — adds per-candidate `ScoringContract` on `DiagnosticBundle` |

### 2.7 Determinism and seed management

| Scope | Mechanism |
|-------|-----------|
| Python / NumPy | Root seed from CLI `--seed`; one-call seeding |
| JAX | `jax.random.PRNGKey(root_seed)`; CPU-only is deterministic |
| R | `set.seed(seed)` + `RNGkind("L'Ecuyer-CMRG")` in `harness.R`; `.Random.seed` captured |
| R threads | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` set by the harness to eliminate BLAS non-determinism |
| Bayesian (Stan) | `cmdstanpy` seeds per chain; warmup + sampling counts in `SamplerConfig` |
| LLM | `temperature=0` enforced (non-zero raises `ValueError`); SHA-256 payload hashes + `ReplayClient` for offline replay |
| Seed registry | `seed_registry.json` in every bundle |

**GPU boundary.** JAX on GPU is not guaranteed deterministic even with fixed seeds.
The current codebase is CPU-oriented; GPU execution is not wired into the CLI. If
GPU is introduced later, bundle artifacts will gain an `execution_mode` field and a
`hardware_descriptor` in `backend_versions.json` to key the replay cache.

### 2.8 Observability — what the project actually uses

This project does not ship with SaaS observability integrations. Prior revisions of
this doc mentioned Langfuse, Aim, Flowcept, and a full OpenTelemetry stack — none
are in `pyproject.toml`. The operational picture:

| Layer | Mechanism |
|-------|-----------|
| Structured logging | `structlog` JSON logs, context-bound (run_id, candidate_id, gate). Configuration in `src/apmode/logging.py`. |
| Terminal UI | `rich` tables / progress; the CLI's `inspect`, `log`, `trace`, `graph`, `policies`, `ls` commands render to the terminal |
| Run ledger | The bundle is the source of truth. `search_trajectory.jsonl`, `failed_candidates.jsonl`, `gate_decisions/`, and `ranking.json` carry the full audit trail. |
| Gate audit | Per-gate `GateResult` under `gate_decisions/gate{1,2,2_5,3}_{candidate_id}.json` with per-check `GateCheckResult` |
| LLM audit | `agentic_trace/` — per-iteration `{input.json, output.json, meta.json}` including prompt hash, token counts, cost; `classical_checkpoint.json` enables `--resume-agentic` |
| Report provenance | `report_provenance.json` captures generator identity, component versions, timestamps |
| Bundle completeness | `_COMPLETE` sentinel with SHA-256 digest |
| R-side logging | R harness writes `logs.jsonl` to the request tempdir; orchestrator reads after completion |

External observability (Langfuse, Aim, Flowcept, OTel spans/exporters) is a **non-goal**
for 0.x — see §10.

### 2.9 Data security and privacy

- **LLM inputs.** `agentic_runner` summarizes fit diagnostics via
  `diagnostic_summarizer.redact_for_llm()`, which uses an allow-list of field names;
  unknown fields fail closed. Raw row-level patient data is never sent.
- **Imputation-stability cherry-picking guard.** When MI is active, the LLM sees only
  pooled + stability diagnostics (not per-imputation draws), so it cannot exploit a
  single lucky imputation — see
  `diagnostic_summarizer.summarize_stability_for_llm`.
- **Bundle access.** Bundles may contain pseudonymized subject-level references.
  Filesystem permissions and encryption-at-rest are deployment-specific.

### 2.10 License compatibility

GPL-2-or-later. Apache-2 dependencies (JAX, Diffrax, structlog, tenacity, OTel SDK
if added later) are compatible. License audit is a CI item.

### 2.11 Testing

| Layer | Location |
|-------|----------|
| Unit | `tests/unit/` — DSL, data, backends, search, governance, routing, bundle, report |
| Integration | `tests/integration/` — mock R pipeline, Discovery lane, LLM providers, Suite A/B/C E2E, BLQ flows |
| Property-based | `tests/property/` — Hypothesis on DSL round-trip, transforms, LORO-split invariants |
| Golden-master | `tests/golden/` — syrupy snapshots of emitter output |
| Fixtures | `tests/fixtures/` — Suite A CSVs + stored policies |
| Live-gated | `-m live` marker — LLM providers, R subprocess, CmdStan |
| Policy-file validation | CI hook `governance/validate_policies.py` |

Current collected count is auto-synced into README/CLAUDE.md by
`scripts/sync_readme.py` via `<!-- apmode:AUTO:tests -->` markers — do not hard-code
here.

### 2.12 CI / CD

GitHub Actions: `uv sync --all-extras` → pytest → mypy strict → ruff check + format.
Pre-commit runs ruff + mypy + the policy validator. Matrix is Python 3.12 / 3.13 /
3.14. Suite C has a dedicated workflow in `.github/workflows/suite_c_phase1.yml`;
other benchmark cadences are operator-driven unless a workflow exists in `.github/workflows/`.

---

## 3. System Architecture

```
              ┌────────────────────────────────────────────┐
              │            apmode CLI (Typer)              │
              │    16 direct commands + registered groups    │
              │    datasets | explore | diff | log | trace  │
              │    lineage | graph | report | doctor | ls   │
              │    policies | bundle                        │
              └──────────────────────┬─────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │      Orchestrator       │
                        │    (asyncio pipeline)   │
                        └────────────┬────────────┘
                                     │
         Ingest ──► Profiler ──► Initial Estimator ──► Splitter
              │                                             │
              ▼                                             ▼
       Missing-data                                   Lane Router
       directive + MI / FREM                    ┌──────────┼──────────┐
                                                ▼          ▼          ▼
                                         Classical    NODE        Bayesian
                                         (nlmixr2)   (JAX/Diffrax) (Stan/Torsten)
                                                │          │          │
                                                └───────┐  │  ┌───────┘
                                                        ▼  ▼  ▼
                                                    Agentic LLM
                                                    (transforms only,
                                                     ≤25 iters)
                                                        │
                                                        ▼
              ┌──────────────┬─────────────┬───────────────────┐
              │    Gate 1    │   Gate 2    │    Gate 2.5       │
              │  Technical   │    Lane     │   Credibility     │
              │  Validity    │ Admissibility│   (ICH M15)      │
              │  (PIT/NPDE-  │  (shrinkage, │                   │
              │   lite,      │   identif.,  │                   │
              │   CWRES,     │   NODE excl, │                   │
              │   R̂/ESS for  │   LORO-CV,   │                   │
              │   Bayesian)  │   priors)    │                   │
              └──────────────┴─────────────┴───────────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   Gate 3    │
                              │  Ranking    │
                              │ (Borda or   │
                              │  weighted;  │
                              │  VPC + NPE  │
                              │  + AUC/Cmax)│
                              └──────┬──────┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │  Reproducibility Bundle       │
                     │  (Pydantic → JSON/JSONL       │
                     │   + _COMPLETE sentinel with   │
                     │   SHA-256 digest)             │
                     └──────────────────────────────┘
```

### 3.1 Component inventory (verified against HEAD)

All paths rooted at `src/apmode/`.

#### DSL

| File | Role |
|------|------|
| `dsl/pk_grammar.lark` | Lark EBNF (5 blocks) |
| `dsl/grammar.py` | `compile_dsl` entry (parse + AST build, 10 KB guard) |
| `dsl/transformer.py` | Parse tree → AST |
| `dsl/ast_models.py` | `DSLSpec` + all module Pydantic nodes |
| `dsl/normalize.py` | AST canonicalization |
| `dsl/validator.py` | Lane-aware `validate_dsl` |
| `dsl/transforms.py` | 9 structural transforms + `FormularTransform` union |
| `dsl/prior_transforms.py` | `SetPrior` transform (10th) |
| `dsl/priors.py` | Prior families + `_VALID_FAMILIES` admissibility matrix |
| `dsl/nlmixr2_emitter.py` | AST → nlmixr2 R code |
| `dsl/stan_emitter.py` | AST → Stan program (IOV, NODE, maturation covariates, and v0.7 absorption preview forms ⇒ `NotImplementedError`) |
| `dsl/frem_emitter.py` | AST → FREM-augmented nlmixr2 |
| `dsl/_emitter_utils.py` | Shared emitter helpers |

#### Data

| File | Role |
|------|------|
| `data/schema.py` | Pandera canonical-PK schema |
| `data/ingest.py` + `data/adapters.py` | Format-specific ingestion |
| `data/dosing.py` | ADDL/II expansion, infusion events, event-table builder |
| `data/profiler.py` | Evidence Manifest with structured `nonlinear_clearance_signals` |
| `data/policy.py` | `ProfilerPolicy` loader |
| `data/initial_estimates.py` | NCA + unit-aware CL heuristic |
| `data/splitter.py` | k-fold + LORO-CV |
| `data/missing_data.py` | Lane-tiered directive resolver |
| `data/imputers.py` | R-backed MI providers |
| `data/categorical_encoding.py` | Binary auto-remap |
| `data/datasets.py` | Public dataset registry (nlmixr2data auto-fetch) |
| `data/types.py` | Shared typed records |

#### Backends

| File | Role |
|------|------|
| `backends/protocol.py` | `BackendRunner` Protocol + `Lane` enum |
| `backends/nlmixr2_runner.py` | Classical NLME (SAEM/FOCEi) via R subprocess |
| `backends/bayesian_runner.py` | Bayesian backend (Stan/Torsten via `cmdstanpy`) |
| `bayes/harness.py` | `cmdstanpy` driver + Vehtari 2021 R̂/ESS + E-BFMI + Pareto-k |
| `backends/frem_runner.py` | FOCE-I FREM driver for missing-data workflow |
| `backends/node_runner.py` | JAX/Diffrax NODE backend |
| `backends/node_model.py` | Bräm-style MLP with RE on input-layer weights |
| `backends/node_ode.py` | Mechanistic skeleton + NODE sub-function (Diffrax Tsit5) |
| `backends/node_trainer.py` | Optax Adam with log-space params + early stopping |
| `backends/node_constraints.py` | 5 enumerated constraint templates |
| `backends/node_init.py` | Pre-trained weights + transfer learning |
| `backends/node_distillation.py` | Sub-function viz, surrogate fitting, fidelity |
| `backends/agentic_runner.py` | Closed-loop LLM improvement (≤25 iters, transforms only) |
| `backends/diagnostic_summarizer.py` | LLM-facing redaction + stability summarization |
| `backends/llm_client.py` + `llm_providers.py` | Anthropic, OpenAI, Gemini, Ollama, OpenRouter, litellm |
| `backends/transform_parser.py` | JSON (LLM) → `list[FormularTransform]` |
| `backends/predictive_summary.py` | Canonical VPC / NPE / AUC-Cmax-BE builder (single path) |
| `backends/r_schemas.py` | R ↔ Python Pydantic wire schemas |
| `backends/prompts/` | LLM prompt templates |
| `r/harness.R` | nlmixr2 SAEM/FOCEi harness |
| `r/impute.R` | mice / missRanger dispatch |
| `r/install_deps.R` | R-side install helper |

#### Search, governance, orchestration, bundle, reporting

| File | Role |
|------|------|
| `search/candidates.py` | Phase 1/3 candidate generation |
| `search/engine.py` | Multi-backend dispatch, BIC scoring, Pareto frontier |
| `search/stability.py` | Rubin pooling + rank-stability metrics |
| `governance/gates.py` | Gates 1, 2, 2.5, 3 evaluators |
| `governance/policy.py` | Pydantic schema for lane policies |
| `governance/ranking.py` | Cross-paradigm ranking (Borda / weighted sum, uniform-drop rule) |
| `governance/validate_policies.py` | CI policy-file validator |
| `orchestrator/__init__.py` | Full pipeline: ingest → profile → NCA → search → FREM/MI → gates → bundle → report |
| `bundle/models.py` | All Pydantic schemas (≈60 classes) |
| `bundle/emitter.py` | Per-artifact `write_*`; `seal()` with `_COMPLETE` sentinel |
| `bundle/scoring_contract.py` | Cross-paradigm scoring-contract helper |
| `report/renderer.py` | HTML + Markdown regulatory report |
| `report/credibility.py` | ICH-M15-aligned credibility assessment |
| `evaluation/` | (benchmark scoring utilities) |
| `benchmarks/` | Suite A/B/C fixtures + runners |

#### CLI + framework

| File | Role |
|------|------|
| `cli.py` | Typer app — 16 direct commands plus `bundle` / `completion` groups |
| `paths.py` | `APMODE_POLICIES_DIR` env override + pyproject-walk fallback |
| `routing.py` | Lane Router — evidence-manifest-driven dispatch |
| `logging.py` | `structlog` configuration |
| `errors.py` | `BackendError` + subtypes |
| `ids.py` | `sparkid`-backed run / candidate ID generation |
| `_version.py` | Generated by `hatch-vcs` |

---

## 4. Key interface contracts

### 4.1 BackendRunner Protocol

```python
# src/apmode/backends/protocol.py

from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

class Lane(StrEnum):
    SUBMISSION   = "submission"
    DISCOVERY    = "discovery"
    OPTIMIZATION = "optimization"

@runtime_checkable
class BackendRunner(Protocol):
    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
    ) -> BackendResult: ...
```

Keyword arguments carry optional context needed by predictive-diagnostics helpers
and the cross-paradigm ranker (Gate 3).

### 4.2 BackendResult (current schema)

```python
class BackendResult(BaseModel):
    model_id: str
    backend: Literal["nlmixr2", "jax_node", "agentic_llm", "bayesian_stan"]
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
    initial_estimate_source: Literal["nca", "warm_start", "fallback"]

    # Bayesian-only (populated by BayesianRunner, None otherwise):
    posterior_diagnostics: PosteriorDiagnostics | None
    sampler_config:        SamplerConfig | None
    posterior_draws_path:  str | None  # bundle-relative
    prior_manifest_path:   str | None  # bundle-relative
    simulation_protocol_path: str | None  # bundle-relative
```

### 4.3 DiagnosticBundle (canonical, 0.5)

```python
class DiagnosticBundle(BaseModel):
    gof: GOFMetrics
    split_gof: SplitGOFMetrics | None
    vpc: VPCSummary | None
    pit_calibration: PITCalibrationSummary | None   # 0.4.2+: the Gate 1 gated metric
    identifiability: IdentifiabilityFlags
    blq: BLQHandling  # method: "none" | "m1" | "m3" | "m4" | "m6_plus" | "m7_plus"
    npe_score: float | None
    auc_cmax_be_score: float | None
    auc_cmax_source: Literal["observed_trapezoid"] | None
    diagnostic_plots: dict[str, str]
    scoring_contract: ScoringContract  # per-candidate record of NLPD kind,
                                       #   RE treatment, integrator, BLQ, obs model,
                                       #   float precision
```

**Canonical predictive-diagnostics path.**
`backends/predictive_summary.py::build_predictive_diagnostics` is the single
function from per-subject simulation matrices to `DiagnosticBundle.{vpc, npe_score,
auc_cmax_be_score, auc_cmax_source}`. Backends **must** call it atomically — partial
population (e.g. VPC without NPE) is banned.

### 4.4 R subprocess contract

Unchanged in substance from v0.2. The R harness reads
`/tmp/{request_id}/request.json`, writes `response.json` + `logs.jsonl`, and uses
non-zero exit codes only for process failures (nlmixr2 convergence failures are
reported as `status="error"` with `error_type` populated, and the runner raises a
typed `ConvergenceError`). Timeouts kill the whole process group (`os.killpg`);
absence of `response.json` after subprocess exit classifies as `CrashError`.

### 4.5 Gate interface

```python
class GateCheckResult(BaseModel):
    check_id: str                   # e.g. "convergence", "pit_median", "shrinkage"
    passed: bool
    observed: float | bool | str
    threshold: float | str | None
    units: str | None
    evidence_ref: str | None        # bundle-relative path

class GateResult(BaseModel):
    gate_id: str
    gate_name: str
    candidate_id: str
    passed: bool                    # True iff all checks pass
    checks: list[GateCheckResult]
    summary_reason: str
    policy_version: str
    timestamp: str                  # ISO 8601
```

**Gate 3 contract.** Gate 3 is a ranking gate (every survivor passes, but
`Ranking.ranked_entries` orders them). Configuration on `Gate3Config`:

| Field | Role |
|---|---|
| `composite_method` | `"weighted_sum"` (Submission) or `"borda"` (Discovery / Optimization) |
| `vpc_weight`, `npe_weight`, `bic_weight`, `auc_cmax_weight` | Component weights |
| `auc_cmax_nca_min_eligible` + `auc_cmax_nca_min_eligible_fraction` | AND-combined NCA eligibility floor; below either, `auc_cmax_be_score` is set to `None` and the uniform-drop rule removes that component from the composite for every candidate |
| `n_posterior_predictive_sims` | Backend-emitted draws per candidate |
| `vpc_n_bins` | Post-hoc time bins for VPC coverage |
| `vpc_concordance_target` | Target coverage for the concordance score |

Uniform-drop rule: if any survivor lacks a score component, that component is
removed for *every* survivor so ranking stays apples-to-apples.

**Gate 1 PIT / NPDE-lite (0.4.2).** The gated calibration metric is PIT / NPDE-lite on
the posterior-predictive matrix (Brendel 2006, Comets 2008), subject-robust-aggregated
across `p ∈ {0.05, 0.50, 0.95}`. Tolerance is `tol(p, n) = max(floor, z_α ·
sqrt(p(1−p)/n_subjects))`. Lane-specific `z_α` and floor values are in each lane
policy. `VPCSummary` is retained for reporting and within-paradigm concordance but is
no longer a Gate 1 gate.

### 4.6 Credibility report schema

```python
class CredibilityContext(BaseModel):
    candidate_id: str
    lane: Lane
    context_of_use: str
    data_adequacy_statement: str
    ai_ml_role: str | None           # present for NODE / agentic
    limitations: list[str]
    sensitivity_refs: list[str]      # bundle-relative
    prior_justification_refs: list[str]  # Bayesian — points at prior_manifest.json

class CredibilityReport(BaseModel):
    candidate_id: str
    context: CredibilityContext
    risk_map: dict[str, str]         # limitation → risk class
    evidence_refs: dict[str, str]
```

Consumed by `report/credibility.py` and rendered into the HTML/Markdown report.

---

## 5. Reproducibility bundle structure

Layout matches PRD §4.3.2 as extended through 0.5. Names are canonical; drift breaks
`apmode validate`.

```
runs/
└── {run_id}/                              # sparkid
    ├── _COMPLETE                          # JSON: {schema_version, run_id,
    │                                      #   file_digests: {path → sha256}}
    ├── data_manifest.json
    ├── split_manifest.json
    ├── evidence_manifest.json             # profiler policy SHA embedded
    ├── missing_data_directive.json
    ├── imputation_stability.json          # present when MI is active
    ├── categorical_encoding_provenance.json
    ├── initial_estimates.json
    ├── nca_diagnostics.jsonl
    ├── seed_registry.json
    ├── policy_file.json                   # versioned gate thresholds (copy of lane policy)
    ├── backend_versions.json              # Python / R / nlmixr2 / CmdStan / hardware
    ├── search_trajectory.jsonl            # per-candidate BIC/OFV/convergence
    ├── failed_candidates.jsonl            # per-check gate failures
    ├── candidate_lineage.json             # DAG edges (parent → child + label)
    ├── search_graph.json                  # full DAG for `apmode graph` (when --agentic)
    ├── classical_checkpoint.json          # enables `--resume-agentic`
    ├── ranking.json                       # Gate 3 output
    ├── report_provenance.json
    ├── gate_decisions/
    │   ├── gate1_{candidate_id}.json
    │   ├── gate2_{candidate_id}.json
    │   ├── gate25_{candidate_id}.json
    │   └── gate3_{candidate_id}.json
    ├── compiled_specs/
    │   ├── {candidate_id}.json            # DSLSpec (Pydantic)
    │   └── {candidate_id}.R               # lowered R code
    ├── results/
    │   ├── {candidate_id}_result.json          # BackendResult
    │   └── {candidate_id}_seed_{i}_result.json # multi-seed runs
    ├── bayesian/                          # when a Bayesian candidate was fit
    │   ├── prior_manifest.json            # prior_manifest_path points here
    │   ├── simulation_protocol.json
    │   ├── mcmc_diagnostics.json          # R̂/ESS/E-BFMI/Pareto-k
    │   └── posterior_draws/{candidate_id}.parquet
    ├── loro_cv/                           # Optimization lane only
    │   └── {candidate_id}_folds.json
    ├── credibility/
    │   └── {candidate_id}_credibility.json
    ├── agentic_trace/                     # when --agentic
    │   ├── {iteration_id}_input.json
    │   ├── {iteration_id}_output.json
    │   └── {iteration_id}_meta.json
    ├── run_lineage.json                   # multi-run provenance
    ├── report.html                        # regulatory report (HTML)
    └── report.md                          # regulatory report (Markdown)
```

JSON/JSONL artifacts are Pydantic-validated before writing. Binary outputs (PNGs,
parquet draws, model weights) are checksummed and referenced via manifest entries.
`_COMPLETE` is written atomically as the last step of a successful run; its absence
signals an incomplete bundle and its SHA-256 manifest catches post-hoc tampering.

---

## 6. Phasing — historical

Phases 1 and 2 are complete. Phase 3 is in progress per CLAUDE.md. The per-month
task list from v0.2 has been removed from this doc; it is preserved in the git
history at `docs/ARCHITECTURE.md@v0.2` and summarized in `CHANGELOG.md`.

**What is active today (0.6.1-rc1):**

- DSL grammar + compiler + validator + 10 typed transforms.
- Classical NLME backend (nlmixr2, SAEM/FOCEi) with warm-start children.
- Bayesian backend (Stan / Torsten via `cmdstanpy`) with R̂ / ESS / E-BFMI /
  Pareto-k Gate 1 integration.
- NODE backend (Bräm-style hybrid MLP, Diffrax Tsit5, Optax Adam).
- Agentic LLM backend (6 providers: Anthropic / OpenAI / Gemini / Ollama /
  OpenRouter / litellm; ≤ 25 iterations, transforms only).
- FREM + MI-PMM + MI-missRanger missing-data pipelines.
- Gate 1 (PIT / NPDE-lite) + Gate 2 (lane-specific) + Gate 2.5 (ICH M15) +
  Gate 3 (Borda / weighted ranking with uniform-drop rule).
- Reproducibility bundle with `_COMPLETE` sentinel and RO-Crate projection.
- Typer CLI (`run`, bundle inspection/reporting, HTTP `serve`, RO-Crate/SBOM subcommands) + HTML / Markdown regulatory report.
- Benchmark Suite A (8 scenarios), Suite B perturbation anchors, and Suite C literature fixtures.
- FastAPI HTTP surface (`apmode serve`) with loopback default, static API-key floor for non-loopback binds, SQLite run store, and cancellation lifecycle.

**What remains for Phase 3 completion:** NODE posterior-predictive simulation
(currently inert stub), Stan-side IOV + maturation-covariate lowering, full
Stan/Torsten support for the v0.7 absorption preview forms, and broader
production hardening around public API deployments.

---

## 7. Risk mitigations

| Risk | Mitigation |
|------|-----------|
| R segfault | Process isolation; kill process group on timeout; typed error classification; retry to new `attempt_id/` subdir |
| R stdout contamination | File-based I/O contract (§4.2) |
| Custom orchestrator becomes tech debt | `BackendRunner` Protocol preserves a clean swap-in boundary — no migration required today |
| DSL grammar too rigid | Two-track extensibility (new module ADR vs. enum extension); see FORMULAR §"Extensibility" |
| JAX GPU non-determinism | CPU-first posture; GPU not wired into CLI; future GPU execution will key cache on `(root_seed, data_hash, code_version, hardware_descriptor)` |
| LLM provider versioning | Model version escrowed in `agentic_trace/*_meta.json`; verbatim output caching via `ReplayClient` |
| License incompatibility | GPL-2-or-later + CI license scanner |
| PHI/PII leakage in LLM traces | Redaction via `diagnostic_summarizer.redact_for_llm()` allow-list; imputation-stability cherry-picking guard |
| Bundle drift | `_COMPLETE` sentinel with SHA-256 digest; `apmode validate` refuses unsealed bundles |
| Cross-paradigm NLPD incomparability | `ScoringContract` is per-candidate; cross-paradigm ranking uses simulation-based metrics (VPC + NPE + AUC/Cmax BE) not raw NLPD |
| Agentic LLM cherry-picking across imputations | `summarize_stability_for_llm` substitutes pooled + stability scores for raw per-imputation diagnostics |

---

## 8. Dependency summary

See `pyproject.toml` and §2.1. Binding version floors: `python >= 3.12,<3.15`,
`pydantic >= 2.7`, `lark >= 1.2`, `pandera >= 0.18`, `cmdstanpy >= 1.2` (bayesian
extra), `jax >= 0.4.30` (node + test extras), `anthropic >= 0.39` (llm extra).

---

## 9. Open decisions (current)

Decided items from v0.2 are closed (licensing: GPL-2-or-later; covariate
missingness: lane-tiered FREM / MI-PMM / MI-missRanger per `data/missing_data.py`;
initial-estimate strategy: NCA-seeded + warm-start per §2.5). Items still open:

1. **DSL extensibility process** — Track 1 (new module types) needs an ADR template
   and a pharmacometric-review workflow. PRD §10 Q1.
2. **Regulatory engagement timing** — when to seek informal FDA / EMA feedback on
   the credibility framework (PRD §10 Q4).
3. **NODE posterior-predictive simulation** — `NodeBackendRunner.sample_posterior_predictive`
   is a discoverable inert stub; Phase 3 completion item. Concrete integration
   point: `backends/node_trainer.py`.

---

## 10. Non-goals

Items intentionally **not** in scope for 0.x. If and when these are revisited,
this section is the canonical record of why they were deferred.

- **Docker / Docker-Compose / K8s.** Users run natively via `uv`; containerization
  is out of scope.
- **Flyte / Temporal migration.** `BackendRunner` is the escape hatch if it
  becomes necessary, but the current scale does not justify the orchestrator rewrite.
- **Langfuse / Aim / Flowcept integrations.** The bundle is the run ledger; SaaS
  observability is not planned. Users who need it can shim on top of
  `structlog` JSON output.
- **Rust parser migration.** Lark in Python is adequate; the Phase-2 LALRPOP+PyO3
  branch was considered and rejected.
- **Web UI.** A minimal browser UI was prototyped and removed; the CLI + HTML
  report is the current UX.

---

## Cross-references

- `CLAUDE.md` — operational guidance for Claude Code sessions in this repo.
- `docs/PRD_APMODE_v0.3.md` — product requirements (source of truth for scope).
- `docs/FORMULAR.md` — language reference for the DSL.
- `docs/PROFILER_REFINEMENT_PLAN.md` — derivation + citations for profiler policy defaults.
- `docs/adr/` — Architecture Decision Records; review `0001-review-deferrals.md`
  before re-filing a finding on `from __future__ import annotations`, Pyright, God-module
  decomposition, FREM goldens, `type: ignore` audits, or module-level Rich Consoles.
- `policies/*.json` — versioned gate policies per lane.
- `CHANGELOG.md` — per-release deltas; version history for this document's factual
  claims.
