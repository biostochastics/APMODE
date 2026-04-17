# APMODE — consensus review brief

Root: `/Users/biostochastics/APMODE`. Python 3.12+, GPL-2.0-or-later. Governed PK modelling meta-system composing four paradigms (nlmixr2 NLME, Stan Bayes, hybrid NODE, agentic-LLM). Three lanes (Submission / Discovery / Optimization). Gated funnel governance (Gate 1+2 disqualifying, Gate 3 ranking). DSL is the moat for agentic transforms. Reproducibility bundle sealed with SHA-256 `_COMPLETE`; RO-Crate is a read-only projection on top. `predictive_summary.py` is the canonical VPC/NPE/AUC-Cmax helper — atomic population of diagnostics is required.

## Already-gathered findings — VERIFY, EXTEND, or REFUTE (state CONFIRMED/PARTIAL/FALSE_POSITIVE/INSUFFICIENT_INFO with line evidence)

DSL (droid pass):
1. CRITICAL — Stan vs nlmixr2 proportional-error likelihood mismatch → blocks cross-paradigm NLPD. `dsl/stan_emitter.py` vs `dsl/nlmixr2_emitter.py` (observation blocks).
2. HIGH — `IVBolus` route missing from grammar/validator (`dsl/grammar.py`, `dsl/validator.py`).
3. HIGH — Transformer loses source positions (`dsl/transformer.py`) → breaks audit trail.
4. HIGH — BLQ sigma validation gap (`dsl/validator.py`).
5. HIGH — Stan TMDD QSS state naming inconsistency (`dsl/stan_emitter.py`).
6. HIGH — BLQ vs non-BLQ proportional likelihood inconsistency in Stan.
7. MEDIUM — Silent fallbacks in emitters swallow untranslatable nodes.
8. MEDIUM — InvGamma misparam on SD targets (`dsl/priors.py`).
9. MEDIUM — `normalize_param_name` missing from pruner.
10. MEDIUM — Cross-emitter sigma semantics drift.
11. LOW — IVBolus empty-string influx fragility.
12. LOW — `structural_param_names` exhaustiveness gap.

Governance / search / eval / benchmarks (gemini pass):
A. CRITICAL — `evaluation/loro_cv.py` ~line 82: `warm_estimates` from full-data posteriors fed to test folds, no `fixed_parameter=True` → data leakage if runner refits.
B. HIGH — `search/candidates.py` ~line 268: `SearchNode` mutable (`update_score`); no `.seal()`; DAG mutable post-seal.
C. HIGH — `governance/gates.py` ~line 122: hardcoded `lower_bound=1e-4`, `upper_bound=1e5`, `rse>200`, `ofv_abs_spread<0.1` should be in Gate1Config / policy JSON.
D. HIGH — `data/missing_data.py` / `search/stability.py` ~line 265: `adaptive_m` / `m_max` defined but `run_with_imputations` loops exactly `m` — adaptive not implemented.
E. MEDIUM — `governance/ranking.py` ~line 479: `metrics.sort(key=lambda m: m.composite_score)` — no `(score, candidate_id)` tiebreak → non-deterministic under float ties.
F. MEDIUM — `governance/policy.py`: Pydantic fields lack `Field(..., ge=, le=)` + units.
G. MEDIUM — `benchmarks/scoring.py` ~line 126: universal median-abs-error for NPE mismatches proportional/additive likelihoods; NPDE/PIT preferred.
H. MEDIUM — `benchmarks/perturbations.py`: missing BSV scaling + saturating-clearance perturbations.
I. LOW — `governance/gates.py:105`: Gate 1 `passed = all(c.passed for c in checks)` — correct disqualifier.
J. LOW — `governance/ranking.py:310`: uniform-drop `[None] * len(...)` — correct None contract.

## Gaps that still need your pass (add NEW findings, max 8 per model)
- Backends: `backends/predictive_summary.py` atomic-population path; `backends/agentic_runner.py` 25-iter cap + raw-text path; `backends/bayesian_runner.py` divergence/R-hat surface; `backends/nlmixr2_runner.py` subprocess safety; `backends/node_*.py` Bräm invariant.
- Plumbing: `bundle/emitter.py` digest stability + seal atomicity; `bundle/rocrate/importer.py` ZIP-slip under Windows-zero-mode; `data/profiler.py` determinism; `orchestrator/__init__.py` seed-idempotency + crash graceful-degrade; `cli.py` refactor seams (4239 LoC); `report/renderer.py` score provenance; `bundle/models.py` Pydantic validator gaps.

## Output format (fenced JSON only, terse)
```json
{
  "verifications": [{"id":"1","verdict":"CONFIRMED","file":"…","line":N,"evidence":"…","fix":"…","notes":""}],
  "new_findings": [{"file":"…","line":N,"category":"bug|correctness|security|math|wiring|perf|docs|style","severity":"critical|high|medium|low","description":"…","evidence":"…","fix":"…","confidence":"high|medium|low"}],
  "cross_subsystem": [{"description":"…","files":["…"],"severity":"…","fix":"…"}],
  "priority_shortlist": ["top 5 to fix first, ordered"]
}
```

Do NOT hallucinate line numbers. If you cannot locate, say INSUFFICIENT_INFO.
