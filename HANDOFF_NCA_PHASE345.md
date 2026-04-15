# HANDOFF — NCA refinement + error-model heuristic PR (Phases 3–5)

Paste this into the next session if it needs to resume.

## Current State (already landed in working tree, NOT yet committed)

Phase 1 (NCA refinement) + Phase 2 (error-model heuristic) are IMPLEMENTED.
Phase 3 tests WRITTEN. `mypy --strict` is clean. 1259 tests pass (48 new,
1211 existing). Remaining: multi-agent review, CHANGELOG/README, commit.

## Files modified (see `git status`)

- `src/apmode/data/initial_estimates.py` — NCAEstimator rewritten with
  PKNCA-style curve-stripping (`_select_lambda_z`), linear-up/log-down AUC
  (`_auc_lin_up_log_down`, Purves 1992), QC gates
  (adj_r²≥0.80, extrap≤20%, span_ratio≥1, n_λz≥3), literature-prior fallback
  via new `fallback_estimates` param, `LambdaZFit` + `NCAResult` dataclasses,
  per-subject diagnostics list, optional matplotlib plot emission.
- `src/apmode/data/profiler.py` — new `recommend_error_model()` decision tree
  (Beal 2001 / Ahn 2008): BLQ≥10%→BLQ_M3 with prop+comb only (never
  additive); LLOQ/Cmax>5% OR terminal log MAD>0.35 → combined;
  range>50+CV<80 → proportional; narrow+low-CV → additive; default
  proportional. Also adds `_compute_cmax_dynamic_range`,
  `_compute_dv_cv_percent`, `_compute_terminal_log_residual_mad`,
  `_cmax_median`.
- `src/apmode/search/candidates.py` — `SearchSpace.from_manifest` honors the
  new `error_model_preference`; legacy BLQ>20% fallback preserved when
  preference is None.
- `src/apmode/bundle/models.py` — new Pydantic models `ErrorModelPreference`
  (primary, allowed, confidence, rationale) and `NCASubjectDiagnostic`
  (one row per subject in `nca_diagnostics.jsonl`). Added 4 new fields on
  `EvidenceManifest`: `error_model_preference`, `cmax_p95_p05_ratio`,
  `dv_cv_percent`, `terminal_log_residual_mad`.
- `src/apmode/bundle/emitter.py` — new `write_nca_diagnostics()` (JSONL) and
  `nca_plots_dir()` helper.
- `src/apmode/orchestrator/__init__.py` — `RunConfig.fallback_estimates`
  added; Stage 2 wires it into NCAEstimator, emits `nca_diagnostics.jsonl`
  + per-subject plots, logs source (nca | dataset_card | defaults).
- `tests/unit/test_nca_refinement.py` — 12 new tests.
- `tests/unit/test_error_model_recommendation.py` — 10 new tests.

## Remaining work

1. **Multi-agent review** via `xen clink` — droid, crush, gemini, codex on
   initial_estimates.py, profiler.py, candidates.py.
2. **Tests + typing + lint** — pytest (1259+), `mypy --strict`, `ruff check`.
3. **CHANGELOG** + **README** updates (see blocks below).
4. **Single commit** on `main`.

## Known limitations (for commit body, not blocking)

- Benchmark harness (`src/apmode/benchmarks/`) does not yet thread
  `DatasetCard.published_model.key_estimates` into
  `RunConfig.fallback_estimates`. Follow-up PR.
- pyright IDE noise about pandas-stubs Series/DataFrame unions;
  `mypy --strict` is the project gate and is clean.

## Watch out for

- `xen clink` can stall. If parallel execution doesn't return in ~2 min,
  kill and retry with fewer CLIs or run sequentially.
- RUF002/RUF003 on Greek σ, ×, – in docstrings — already scrubbed in `src/`.
- Do not revert auto-formatter changes from the prior session.

## Success criteria

1. Multi-agent review complete (≥3 of 4 CLIs responded); high-signal
   fixes applied.
2. `ruff check` + `mypy --strict` clean.
3. `pytest tests/` passes (1259+).
4. mavoglurant Suite B Gate 1 passes with top CL ∈ [3, 15] L/h (if R available).
5. CHANGELOG + README updated.
6. Single commit on `main`.
