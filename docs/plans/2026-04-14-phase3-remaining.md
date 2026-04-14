# Phase 3 Remaining Work — Implementation Plan

> **Date:** 2026-04-14
> **Status:** In progress (P3.B partially complete)
> **Evidence sources:** Exa research (Khandokar 2025, Comets 2008, PAGE 2025 Typst/Quarto)

---

## Build Order

```
P3.B Phase 1 (folds/models)       ← COMPLETE
    ↓
P3.B Phase 2 (execution/gates)    ← IN PROGRESS (execution engine done, orchestrator wiring next)
    ↓                    ↓
P3.C (reports)     P3.E (API)     [parallel after P3.B]
    ↓
P3.D (Suite C)                    [last, needs external data]
```

## P3.B — Optimization Lane + LORO-CV

### Phase 1 — COMPLETE
- [x] LOROFoldResult, LOROMetrics, LOROCVResult in bundle/models.py
- [x] LORO policy fields in Gate2Config (npde_mean_max, npde_variance_min/max, vpc_coverage_min, min_folds, budget_top_n)
- [x] loro_cv_splits() with regimen-signature fold generation in splitter.py
- [x] write_loro_cv_result() in bundle emitter
- [x] Gate 2 _check_loro_requirement replaced with real LORO threshold checks
- [x] evaluate_loro_cv() execution engine with per-fold fitting + metric aggregation
- [x] 31 new tests (13 fold + 11 Gate 2 + 7 execution)

### Phase 2 — REMAINING
- [ ] P3.B-5: Wire LORO-CV into Orchestrator (Stage 6 restructure)
- [ ] P3.B-6: Integration + property tests

### Key Design Decisions
- **Regimen signature**: Modal dose amount per subject (not total AMT)
- **Evaluation mode**: Fixed-parameter held-out eval (fast, default). Refit mode available for strict CV.
- **LORO only for Gate 1 survivors**: Computational efficiency; budget cap via policy
- **Separate bundle artifact**: loro_cv/{candidate_id}.json (not modifying SplitManifest)
- **Gate 2 backward compatible**: loro_metrics is optional kwarg with None default

## P3.C — Report Generator (Typst + HTML)

Validated by PAGE 2025 (Pumas-AI) poster. Use typst PyPI package for PDF, Jinja2 for HTML.

## P3.E — REST API (FastAPI)

POST /runs → job_id → GET /status → GET /bundle pattern. asyncio.create_task for background execution.

## P3.D — Suite C

Framework + fixture dry-run first. External expert data as explicit handoff gate.
