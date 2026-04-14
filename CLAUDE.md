# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

**Phase 1 Month 5-6 — in progress.** 679 tests passing. `mypy --strict` clean. `ruff` clean.

- `PRD_APMODE_v0.2.md` — Initial RFC (2026-04-11)
- `PRD_APMODE_v0.3.md` — Revised per multi-model stress-test (2026-04-13, **current**)
- `ARCHITECTURE.md` — Technical architecture (v0.2, derived from PRD v0.3)

## Build, Test, and Run Commands

```bash
# Install dependencies (Python 3.12+, uv required)
uv sync --all-extras

# Run the full test suite (679 tests)
uv run pytest tests/ -q

# Run specific test categories
uv run pytest tests/unit/ -q              # unit tests
uv run pytest tests/integration/ -q       # E2E mock R pipeline
uv run pytest tests/property/ -q          # Hypothesis property-based
uv run pytest tests/golden/ -q            # syrupy golden master snapshots
uv run pytest tests/ --snapshot-update    # update snapshots after emitter changes

# Type checking and linting (both must be clean)
uv run mypy src/apmode/ --strict
uv run ruff check src/apmode/ tests/
uv run ruff format src/apmode/ tests/

# CLI
uv run apmode run <dataset.csv> --lane submission
uv run apmode validate <dataset.csv>
uv run apmode inspect <bundle_dir>

# Benchmark Suite A (requires R 4.4+ with rxode2, jsonlite, lotri)
Rscript benchmarks/suite_a/simulate_all.R [output_dir]
```

## Product: APMODE

APMODE (Adaptive Pharmacokinetic Model Discovery Engine) is a governed meta-system that composes four PK modeling paradigms — classical NLME, automated structural search, agentic LLM model construction, and hybrid mechanistic-NODE — into a single workflow for population PK model discovery.

## Architectural Concepts That Span the PRD

Understanding these requires reading multiple sections of the PRD together. They are load-bearing for any implementation decision:

- **Three operating lanes, not one configurable loop.** Submission, Discovery, and Translational Optimization are separate pipelines with different admissible backends, stopping rules, and evidence thresholds (§3). NODE/agentic models are **not eligible for "recommended" status in the Submission lane** — this is a hard rule, not a tunable weight.

- **Evidence Manifest gates dispatch.** The Data Profiler emits a typed manifest (§4.2.1) that downstream backends must consume. It is not descriptive metadata — it constrains which backends run. Example: `richness_category = sparse` + inadequate absorption coverage → NODE backends receive `data_insufficient` flag or are not dispatched.

- **The PK DSL is the moat and must be built first.** §4.2.5 defines a typed grammar (`Absorption × Distribution × Elimination × Variability × Observation`) with a fixed set of agent transforms. The agentic LLM backend (Phase 3) operates **exclusively** through these transforms — it cannot emit raw code. Every module and transform change must flow through the DSL compiler and audit trail. Build order: DSL/compiler/validator → classical backend → automated search → NODE → agentic.

- **Governance is a gated funnel, not a weighted sum** (§4.3.1). Gate 1 (technical validity) and Gate 2 (lane-specific admissibility) are disqualifying; only survivors enter Gate 3 ranking. Gate thresholds are **versioned policy artifacts** in per-lane policy files, not hard-coded constants.

- **Cross-paradigm NLPD comparability is an architectural blocker** (§4.3.1, §10 Q2). Before the Phase 2 Discovery lane can do cross-paradigm ranking in production, a formal observation-model comparability protocol must be specified. Treat this as a scientific design problem, not an integration detail.

- **Reproducibility bundle is the unit of reproducibility** (§4.3.2). Every run emits a fixed set of JSON/JSONL artifacts (data/split/seed/backend manifests, search trajectory, failed candidates, candidate lineage DAG, compiled specs). Any new backend or scoring path must plug into this bundle.

- **Hybrid NODE is a constrained approximation, not a population-PK engine** (§4.2.4, R6). Phase 2 scope is *one* architecture: Bräm-style low-dimensional hybrid with random effects on NODE input-layer weights. Full latent ODE (Maurel) and Uni-PK cross-compound integration are research branch only. Interpretability output is **functional distillation** (learned sub-function visualization + parametric surrogate fitting + fidelity quantification) — not SHAP.

- **Primary engine is nlmixr2 (R).** NONMEM and Pumas are optional adapters. nlmixr2's GPL-2 license interacts with the licensing model (§10 Q5) — a decision that affects build structure from Phase 1.

## Phasing (from §8)

- **Phase 1 (6 mo):** DSL + compiler + audit trail, classical NLME backend, automated search, Gate 1 + Submission-lane Gate 2, reproducibility bundle, CLI, Benchmark Suite A (4 scenarios).
- **Phase 2 (4 mo):** Hybrid NODE backend, functional distillation, Discovery lane, Gate 3 cross-paradigm ranking, Suites A (full) + B, basic web UI.
- **Phase 3 (4 mo):** Agentic LLM backend (DSL transforms only, ≤25 iterations/run), Optimization lane with LORO-CV, report generator with credibility framework, Suite C, API.

## Working With This Repo

- `PRD_APMODE_v0.3.md` is the current source of truth. When asked to implement, trace back to the specific §/table and cite it in design discussions. v0.2 is retained for history only.
- Open questions in §10 are not yet decided — surface them rather than inventing answers (esp. DSL extensibility process, covariate missingness strategy, licensing model).

## Key v0.2 → v0.3 Changes

- **Phase 0 reduced** (2 weeks): licensing decided (GPL-2 open source), data format schema, gate threshold policy format. Process isolation retained for modularity/crash isolation, not licensing.
- **Cross-paradigm NLPD replaced** with simulation-based metrics (VPC coverage concordance, AUC/Cmax bioequivalence, NPE) in Gate 3. NLPD retained within-paradigm only.
- **Gate 2.5 Credibility Qualification** added between admissibility and ranking (ICH M15 context-of-use, limitation-to-risk mapping).
- **Stan codegen deferred** from Phase 1 to Phase 2+ with per-backend lowering test suite.
- **NODE constraints tightened**: enumerated constraint templates replace free-form fields; dim ceilings per lane (≤8 Discovery, ≤4 Optimization).
- **LLM reproducibility hardened**: temperature=0, verbatim output caching, model-version escrow, multi-run provenance.
- **New sections**: §4.2.0 Data Ingestion Contract, §4.2.0.1 Initial Estimate Strategy.
- **Suite C primary metric** changed to fraction-beats-median-expert (≥60%), NLPD gap demoted to secondary.
- **Surrogate fidelity** changed from 20% rate-function to 80–125% AUC/Cmax bioequivalence.
- **Risks R8/R9** added: diagnostic-side leakage in agentic iterations; LLM provider model versioning.
- **Identifiability-based filtering** (profile-likelihood CI, condition number) added to Submission Gate 2 for all backends, not just NODE exclusion.
