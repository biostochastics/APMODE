# APMODE Documentation Index

This directory holds APMODE's product, architecture, DSL, policy, and governance
documentation. Docs fall into two tiers: **public/tracked** files (readable by
anyone who clones the repo) and **internal-only** files that exist on disk in a
full working tree but are excluded from the public repo via `.gitignore`
(`PRD_APMODE_*.md`, `CLAUDE.md`, `docs/plans/`, and
`docs/DIAGNOSTICS_REFINEMENT_ROADMAP.md`). Sections below are marked
accordingly. Everything else listed here (`ARCHITECTURE.md`, the `FORMULAR_*.md`
set, the policy references, `docs/adr/`, `docs/discovery/`) is git-tracked and
ships with the public repo.

## Product & Architecture

| Doc | Purpose |
| --- | --- |
| `PRD_APMODE_v0.3.md` *(internal-only, gitignored)* | Product requirements (v0.3, Status: RFC) — three operating lanes, Evidence-Manifest dispatch contract, PK DSL grammar (§4.2.5), four-gate governance funnel, Phase 0-3 roadmap. Declared "source of truth" but its Phase 2+/3 framing (Stan codegen, agentic backend) predates their actual shipping and is not yet corrected in-document. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture (derived from PRD v0.3 §3-§8, tracks the current release) — real dependency stack, DSL/backend/governance/bundle component inventory verified against `src/apmode/`, `BackendRunner`/`Gate`/`CredibilityReport` interface contracts, reproducibility-bundle directory layout. |

## Formular DSL Reference

| Doc | Purpose |
| --- | --- |
| [FORMULAR.md](FORMULAR.md) | Entry-point overview of the Formular PK DSL — grammar shape, module tables, the 10 typed agentic transforms, compile→validate→emit pipeline, per-backend lowering examples. Start here. |
| [FORMULAR_SEMANTICS.md](FORMULAR_SEMANTICS.md) | Formal Phase 2 module-by-module grammar/semantics reference — every absorption/distribution/elimination/variability/observation/prior variant, macros, transform provenance, the full Lark grammar, with backend-lowering status. |
| [FORMULAR_ERROR_CODES.md](FORMULAR_ERROR_CODES.md) | Canonical `FRM-{TAXON}-NNN` registry for every coded validation error (SYN/AST/SEM/LANE/BE/DATA/POLICY/PRIOR), CI-enforced to stay in sync with `src/apmode/dsl/validator.py`. |
| [FORMULAR_MIGRATION_v0.6_to_v0.7.md](FORMULAR_MIGRATION_v0.6_to_v0.7.md) | Step-by-step rewrite recipe for the two breaking v0.6→v0.7 grammar changes (inline calibration values → `initial:`, `CovariateLink` → top-level `covariates:` block), plus `apmode formular fmt --migrate` automation. |

## Policy & Governance References

| Doc | Purpose |
| --- | --- |
| [FINGERPRINT_MIGRATION.md](FINGERPRINT_MIGRATION.md) | Versioning policy for the four `DSLSpec` sha256 fingerprints (structure/spec/initial/justification) and when `CANONICAL_SCHEMA_VERSION` must bump so cross-version digests are never silently treated as equal. |
| [PROFILER_REFINEMENT_PLAN.md](PROFILER_REFINEMENT_PLAN.md) | Field-by-field derivation and literature citation for every threshold in `policies/profiler.json`, mapped to its consuming code path. |

## Architecture Decision Records

ADRs in `docs/adr/` are a permanent historical record: once accepted, an ADR is
not edited to reflect later drift — new decisions get new, numbered ADRs, and
this index (or the ADR itself, sparingly) notes when its cited facts have gone
stale.

| Doc | Purpose |
| --- | --- |
| [adr/0001-review-deferrals.md](adr/0001-review-deferrals.md) | Records six engineering-hygiene findings from the April 2026 audit (future-annotations, Pyright, god-modules, FREM goldens, `type: ignore`, module-level Console) deliberately deferred rather than fixed, each with an explicit re-evaluation trigger. Consult before re-filing any of these findings. |
| [adr/0003-sota-absorption-extension.md](adr/0003-sota-absorption-extension.md) | Justifies the three absorption forms (Erlang, ParallelFirstOrder, SumIG) added to the PK DSL in v0.7 — AST/emitter/validator/lane-gating design, literature-grounded priors, NLPD comparability protocol for ranking against existing forms. |

## Discovery & Decision Notes

| Doc | Purpose |
| --- | --- |
| [discovery/eleveld_propofol_coverage.md](discovery/eleveld_propofol_coverage.md) | Explains why Eleveld 2018 propofol was rejected as a Phase-1 Bayesian benchmark fixture (missing derived-covariate, age-decay, and Stan-maturation DSL primitives) and names vancomycin (Roberts 2011) as the substitute — the engineering record behind Suite C's fixture set. |

## Internal Roadmap

*Not in the public repo — gitignored (`docs/DIAGNOSTICS_REFINEMENT_ROADMAP.md`).*

| Doc | Purpose |
| --- | --- |
| `DIAGNOSTICS_REFINEMENT_ROADMAP.md` | Tiered (A/B/C/D) backlog of dataset- and model-layer diagnostics work — FREM deprecation, cross-paradigm scoring protocol, influence diagnostics, SBC, multi-start stability — each item naming its source file, policy knob, bundle artifact, and failure mode. |

## Active Implementation Plans

*Not in the public repo — gitignored (`docs/plans/`).*

| Doc | Purpose |
| --- | --- |
| `plans/2026-04-13-phase2-node-discovery-lane.md` | 12-task TDD plan for the Phase 2 hybrid NODE backend (JAX/Diffrax/Equinox), Gate 2.5 credibility qualification, Gate 3 cross-paradigm ranking, functional distillation, Discovery-lane orchestrator wiring — largely still open. |
| `plans/2026-04-13-phase2-node-discovery-lane.md.tasks.json` | Machine-readable per-task status sidecar for the above. |
| `plans/2026-04-14-deep-inspection-cli.md` | 6-task plan that specified `apmode trace`/`lineage`/`graph` — ~90% shipped; remaining live value is the unimplemented `graph --ancestor-of/--descendant-of/--gate` filters. |
| `plans/2026-04-14-phase3-remaining.md` | Terse status stub tracking Phase 3 LORO-CV/report/API remainder as of 2026-04-14; superseded in detail by the 2026-04-24 v0.6-completion plan. |
| `plans/2026-04-24-apmode-v0.6-completion.md` | 45-task, 4-block plan (Bayesian orchestrator dispatch + SBC scaffold, Typst PDF report, REST API, Suite C literature benchmark) for the v0.5→v0.6 milestone; 36/45 tasks done, Typst PDF / SBC nightly runner / API tail still open. |
| `plans/2026-04-24-apmode-v0.6-completion.md.tasks.json` | Machine-readable per-task status/dependency mirror of the v0.6-completion plan. |
| `plans/2026-04-24-apmode-v0.6-completion.NEXT-SESSION-PROMPT.md` | Resumable session-continuation prompt for finishing the v0.6-completion plan's remaining Typst/API/SBC/release tasks. |
| `plans/2026-07-08-formular-research-notes.md` | Seven-section externally-sourced survey (NONMEM/Pumas/Stan prior syntax, LSP LOC sizing, cognitive-load research, parser-first sequencing) grounding the companion sharpening plan's design decisions. |
| `plans/2026-07-08-formular-sharpening-and-adoption-design.md` | 12-section, 4-phase-plus-5-track design plan making Formular a full pharmacometric contract language; Phases 0-2 marked SHIPPED in-doc, leaving Phase 3 (LSP) and Adoption Tracks A-E open. |
