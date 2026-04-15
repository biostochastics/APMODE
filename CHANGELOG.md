# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — Evidence profiler refactor (BREAKING, PRD §10 Q2 follow-up)

The data profiler's nonlinear-clearance heuristic was producing false
positives on real-world multi-compartment linear datasets (warfarin: 0/24
candidate convergence in the rc1 real-fit campaign). Multi-model
consensus (gpt-5.2-pro, gemini-3-pro, mimo-v2-pro, glm-5) plus targeted
exa.ai literature search drove a substantial refactor.

**Breaking schema change**: `EvidenceManifest.nonlinear_clearance_signature: bool`
is **removed**. Replaced by:

- `nonlinear_clearance_evidence_strength: Literal["none","weak","moderate","strong"]`
  — multi-signal voting, where each enabled signal contributes one vote
  out of the *eligible* signals (not absolute count), so single-dose-level
  studies do not get penalised by an inactive third signal.
- `compartmentality: Literal["one_compartment","two_compartment","multi_compartment_likely","insufficient"]`
  — derived from terminal-monoexp R² + early/late curvature ratio; lets
  downstream search prefer 2-cmt linear over MM when the evidence
  profile matches that signature.
- `multi_dose_detected: bool` — drives a new dose-interval slicing path
  that prevents multi-dose accumulation from contaminating the terminal-
  monoexp R² fit and the peak detector (root cause of warfarin's
  terminal R² = -0.49 across the full profile, which recovers to ~0.95
  within the last dosing interval).

**New diagnostic fields** on EvidenceManifest, all derived from the same
shared per-subject helpers so they cannot drift relative to the
categorical fields above:

- `terminal_fit_adj_r2_median`
- `auc_extrap_fraction_median`
- `lambda_z_analyzable_fraction`
- `curvature_ratio_median`
- `peak_prominence_fraction`

**Routing change** in `src/apmode/search/candidates.py`:

- `nonlinear_clearance_evidence_strength == "strong"` → full MM
  cross-product (3-of-3 signals)
- `nonlinear_clearance_evidence_strength == "moderate"` → MM as
  sentinel only (≥50% of eligible signals; 2-cmt false positives like
  warfarin land here, which is much cheaper than the previous bool=True
  full-MM blowout)
- `nonlinear_clearance_evidence_strength in ("weak","none")` → linear
  only

**Peak detector**: `_count_prominent_peaks` now uses
`scipy.signal.find_peaks` with prominence ≥ max(10% of dynamic range,
5% of Cmax) and a minimum inter-peak distance of ≥2 samples. Replaces
the naive local-max counter that registered every descending-limb noise
wiggle as a "peak" (warfarin: 100% subjects falsely flagged multi-phase
absorption pre-refactor).

**Search-space change**: `SearchSpace.from_manifest` no longer removes
``transit`` from the absorption types when ``absorption_complexity ==
"simple"``. The new prominence-based detector classifies
transit-chain absorption datasets (Suite A3 ground truth) as "simple"
correctly, so transit candidates must remain discoverable via BIC
ranking. Test ``test_simple_absorption_deprioritizes_transit`` renamed
+ inverted accordingly.

**Empirical validation** (`benchmarks/data/{theo_sd,warfarin,mavoglurant}.csv`,
nlmixr2data corpus):

| Dataset | Before (bool) | After (enum + compart) |
|---|---|---|
| theo_sd (12 subj, 1-cmt linear) | True (incorrect) | strength=none, compartmentality=one_compartment |
| warfarin (32 subj, 2-cmt linear oral) | True (false positive → 0/24 conv) | strength=moderate, compartmentality=multi_compartment_likely |
| mavoglurant (120 subj, 2-cmt linear oral, multi-dose) | True (false positive) | strength=moderate, compartmentality=multi_compartment_likely, multi_dose_detected=True |

**Suite A integration tests** (7 synthetic recovery scenarios) all
pass: A2/A4 (true MM) correctly produce ``moderate``/``strong`` (1/2
eligible signals → moderate via the eligibility-fraction scoring); A3
(transit) recovers transit candidates without relying on accidental
peak-counter false positives.

**Files touched**:

- `src/apmode/data/profiler.py` — added `_detect_multi_dose`,
  `_last_dose_window`, `_windowed_profile`, `_count_prominent_peaks`,
  `_peak_prominence_fraction`, `_per_subject_terminal_monoexp_r2`,
  `_curvature_ratios_per_subject`, `_classify_nonlinear_clearance_evidence_strength`,
  `_assess_compartmentality`, `_lambda_z_analyzable_fraction`,
  `_auc_extrap_fraction_median`. Existing helpers updated to accept
  ``doses`` + ``multi_dose`` so the dose-interval slicing applies
  uniformly.
- `src/apmode/bundle/models.py` — schema additions + bool removal.
- `src/apmode/search/candidates.py` — routing logic consumes enum,
  transit no longer pruned from "simple" data.
- `src/apmode/report/renderer.py`, `src/apmode/cli.py`,
  `src/apmode/orchestrator/__init__.py`, `src/apmode/benchmarks/suite_b.py`
  — bool field references updated.
- 11 test files updated for the new schema; 1 fixture-coverage test
  inverted to match the corrected absorption-search-space behavior.
- Added `scipy.*` to mypy ignore-missing-imports and dropped a
  now-unused `# type: ignore` in `node_distillation.py`.

**Known follow-up (rc3)**: warfarin's terminal R² = -0.49 is partly a
multi-endpoint contamination (the dataset mixes `cp` warfarin
concentrations with `pca` PD measurements via DVID; APMODE currently
treats both as PK observations). DVID-aware endpoint filtering at
profile time is the next item.

### Added

- **Claude Skill for the APMODE CLI** (`.claude/skills/apmode/SKILL.md`): a
 project-level skill that teaches LLM assistants (Claude Code, Codex, Gemini,
 Droid, etc.) how to correctly drive `apmode run/validate/inspect/datasets/
 explore/diff/log/trace/lineage/graph`. Includes verified flag defaults,
 lane-selection guidance, bundle-artifact names, typical workflows, and
 gotchas (agentic silent no-op on the submission lane, `explore -y` timeout
 override, `--gate` vs `--top` exclusivity, `--iteration` requires `--mode`).
 Cross-reviewed with Droid, Crush, Gemini, and Codex for accuracy.

## [0.3.0-rc1] — 2026-04-15

First release candidate for the v0.3.0 Phase 3 milestone. All pre-release
hygiene items (LICENSE, SPDX headers, CLI documentation, public API) cleared.
1537 fast-path tests passing, `mypy --strict` clean across 80 source files,
`ruff` clean.

### Added

- **CLI `--binary-encode` flag** (`apmode run --binary-encode COL=VAL1:0,VAL2:1`,
 repeatable): per-column override for the categorical encoding auto-detector.
 Parsed into `RunConfig.binary_encode_overrides` and threaded through to
 FREM and MI paths. Validated at the CLI layer so malformed flags fail before
 ingestion.
- **INFO-level structured logging on applied remaps** — `auto_remap_binary_columns`
 now emits a `categorical_encoding.remap_applied` event whenever it rewrites a
 column, so downstream log aggregation captures the auto-detection decisions
 without requiring an `apmode inspect` pass.
- **HTML report export** (`report.html`): every run bundle now emits a
 standalone HTML render of `report.md` alongside the existing Markdown. The
 HTML is self-contained (no external CSS/JS), rendered via `rich.markdown`
 (already a project dependency), safe to archive or host statically. For
 publication-quality HTML, pipe `report.md` through `pandoc`.

### Fixed

- **Boolean/numeric collision in categorical detection** — `detect_encoding`
 now flags object-dtype columns that mix `bool` and non-bool numeric values
 (e.g., `pd.Series([True, 2], dtype=object)`) as `multi_level` with a clear
 rationale, instead of silently producing a meaningless remap. Raised by
 the last review.

### Deferred to v0.3.0-rc2 (documented limitations)

- **Gate 2.5 limitation-to-risk mapping**: currently a presence check
 (threshold calibration requires scientific work — see PRD §10 Q2).
- **Gate 2.5 sensitivity-analysis execution**: presence check only; actual
 ±20% structural-parameter perturbation sweep is rc2 scope.
- **LORO-CV orchestrator integration test**: unit coverage is comprehensive
 (722 LoC across `test_loro_cv.py`, `test_loro_execution.py`,
 `test_loro_gate2.py`), but no end-to-end optimization-lane integration
 test yet.
- **Cross-paradigm NLPD comparability protocol** (PRD §10 Q2): within-paradigm
 NLPD is supported; cross-paradigm ranking uses simulation-based metrics
 (VPC concordance, AUC/Cmax BE, NPE) per the v0.3 design.
- **Full nlmixr2data benchmark sweep with real fits**: FREM sweep currently
 runs compile-only; nightly real-fit sweep is rc2 infrastructure.

### Added — Categorical encoding auto-detection + provenance recording
APMODE now detects and remaps non-canonical binary categorical
covariate encodings automatically, with a documented format contract
and an explicit override path. Closes the "we should be able to let
people know what format we are expecting and how to override" gap
flagged after the benchmark sweep needed per-dataset
``binary_encode`` configs.

- **`src/apmode/data/categorical_encoding.py`** (new, ~430 LOC).
 - `detect_encoding(series)` classifies a column as one of:
 `binary_zero_one` (no remap), `binary_one_two` (1-indexed),
 `binary_string_pair` (recognized M/F, Yes/No, etc., or
 deterministic alphabetic-order default with `UserWarning` for
 unknown pairs), `binary_boolean`, `multi_level`, `continuous`,
 `constant`, `all_missing`.
 - `auto_remap_binary_columns(df, columns, overrides=..., apply=...)`
 applies the remap (or just inspects when `apply=False`) with
 **completeness validation**: silent NaN-on-incomplete-remap was
 the highest-impact bug both were flagged in the
 review; we now raise `ValueError` if any observed
 non-null value would be left unmapped, or if any target value
 is outside `{0, 1}`.
 - **Case-folded deduplication**: `["M", "m", "F"]` → recognized as
 binary M/F with the `m` variant remapping to the same target as
 `M`, instead of falsely classifying as multi-level.
 - **Mixed-type columns rejected**: `["M", 1]` (string + int) now
 returns a multi-level diagnostic with an actionable rationale,
 rather than an alphabetic mis-remap.
 - Recognized string pairs: `M/F`, `Male/Female`, `Yes/No`, `Y/N`,
 `True/False`, `Pos/Neg`, `Positive/Negative`, `Absent/Present`,
 `Control/Case`, `Placebo/Active`. Unknown pairs default to
 alphabetic-order with an explicit `UserWarning` so programmatic
 callers see the polarity choice.
 - Module exports `EXPECTED_BINARY_FORMAT` — a single-source
 docstring describing the contract for the README and CLI
 `inspect`.
- **Wired into the FREM pipeline**:
 `summarize_covariates(transforms={"sex": "binary"})` and
 `prepare_frem_data` both invoke `auto_remap_binary_columns`
 transparently. The orchestrator no longer needs per-dataset
 encoding hacks; the benchmark sweep harness was simplified
 accordingly (its `binary_encode` per-dataset config is gone).
- **Provenance recording**:
 - New `CategoricalEncodingEntry` and `CategoricalEncodingProvenance`
 Pydantic models in `src/apmode/bundle/models.py`.
 - New `BundleEmitter.write_categorical_encoding_provenance` writes
 `categorical_encoding_provenance.json` so reviewers can trace
 exactly how every raw categorical value (e.g., `"male"` /
 `"female"`) was mapped to 0/1, whether the polarity came from
 auto-detection or a caller override, and what the per-column
 rationale was.
 - `summarize_covariates` accepts an optional
 `encoding_hints_out: list | None = None` sink that the
 orchestrator can pass to capture every encoding decision for
 the bundle artifact (in-place append; backward-compatible).
- **22 new unit tests** in `tests/unit/test_categorical_encoding.py`
 covering every encoding kind, validation guards, the case-folded
 reduction, mixed-type rejection, override polarity, the
 alphabetic warning, and end-to-end integration with
 `summarize_covariates`. Total fast-path suite: **1512 passing**.
- **Benchmark sweep regenerated** with auto-detection: theophylline
 (12 subj, WT), warfarin (32 subj, sex strings, auto-remapped),
 mavoglurant (120 subj, SEX 1/2 codes, auto-remapped) — all 3 ✓
 with **zero per-dataset configuration**, the original purpose of
 the auto-detection module.

Review (multi-reviewer) consensus issues fixed in this pass:
- HIGH: silent NaN on incomplete override remap → `ValueError`
- HIGH: `summarize` and `prepare` could disagree on polarity when
 override supplied → both now share the same auto/override logic
- MED: case-sensitive uniqueness counted `["M", "m", "F"]` as
 multi-level → case-folded dedup reduces to binary
- MED: mixed-type two-level columns got bogus alphabetic remap →
 rejected with multi-level diagnostic
- MED: silent alphabetic-order default on unknown pairs →
 `UserWarning` so programmatic callers see the polarity choice

### Added — FREM benchmark sweep across the nlmixr2data corpus
- **`scripts/benchmark_frem_sweep.py`** (new, 350 LOC). Runnable
 benchmark harness that exercises the full FREM emitter pipeline
 (`summarize_covariates` → `prepare_frem_data` → `emit_nlmixr2_frem`
 → `nlmixr2(fn)` compile) on every nlmixr2data-backed dataset
 with covariate coverage. Each dataset config carries a
 `binary_encode` map so categorical covariates with non-{0,1}
 representations (warfarin's `"male"`/`"female"` strings,
 mavoglurant's 1-indexed `1`/`2` integers) are remapped before
 `summarize_covariates` validates them.
- **First-run results** (`docs/FREM_BENCHMARK_RESULTS.md`):

 | Dataset | Subjects | Rows | Covariates | Missing | Compile | Status |
 |---|---:|---:|---|---:|---:|:-:|
 | theophylline | 12 | 144 | WT | 3 | 0.77s | ✓ |
 | warfarin | 32 | 515 | wt, age, sex (binary) | 6 | 0.77s | ✓ |
 | mavoglurant | 120 | 2678 | WT, AGE, SEX (binary) | 18 | 0.77s | ✓ |

 All 3 nlmixr2data benchmarks compile cleanly through the FREM
 pipeline. Total wall time including R startup: ~30 s. This is the
 first end-to-end demonstration that the emitter handles real PK
 data at every registered size tier (12 → 120 subjects), every
 transform branch (continuous + log + binary), and the
 contaminating-column / DVID guards on real layouts.

### Added — Live end-to-end FREM + Rubin-pool test coverage
Before this commit, FREM binary/time-varying support and the
orchestrator MI execution path were only validated by string/dataframe
unit assertions — neither the emitted model nor the full pooling loop
had actually run under nlmixr2 on data. New live tests in
`tests/unit/test_frem_features_live.py` (all marked `live` + most
also `slow`; skipped by default) close those gaps:

- **`test_binary_frem_fits_end_to_end`** — FOCE-I fit on n=12
 synthetic subjects with a 0/1 `SEX` covariate. Proves
 `transform="binary"` produces a model nlmixr2 accepts and fits
 (previously unit-tested only on string/dataframe assertions).
- **`test_time_varying_frem_fits_end_to_end`** — compile-only live
 check on n=10 synthetic subjects with a time-varying CRCL covariate
 (auto-detected by `summarize_covariates`). Confirms the TV model
 (estimable `sig_cov_*`, per-(subject, TIME) augmentation) compiles
 in nlmixr2; full FOCE-I convergence on TV endpoints is deferred to
 estimator tuning work since it is not an emitter correctness issue.
- **`test_rubin_pooling_populates_from_real_m_fits`** — full
 `_run_mi_stage` run with m=3 imputations, real `R_MiceImputer`,
 and real nlmixr2 FOCE-I fits per imputation. Asserts
 `ImputationStabilityEntry.pooled_parameters` is populated from
 actual `(estimate, SE)` tuples with finite Rubin statistics — the
 first end-to-end verification that the Rubin decomposition
 actually runs through real R infrastructure.
- **`test_dvid_collision_with_realistic_multianalyte_layout`** and
 **`test_prepare_frem_data_clears_pk_context_columns`** — realistic
 parent/metabolite DVID scheme + every contaminating column
 (`CENS`/`LIMIT`/`BLQ_FLAG`/`RATE`/`DUR`/`SS`/`II`) on source rows,
 proving the guards actually fire on plausible multi-analyte data.
- **`test_frem_fits_on_theophylline_with_induced_missingness`** —
 canonical Boeckmann/Sheiner/Beal 1994 theophylline benchmark
 (nlmixr2data::theo_sd, 12 subjects × 11 obs, WT covariate). Drops
 WT for 3 subjects to induce the missingness scenario, runs the
 full Python emitter pipeline (`summarize_covariates` →
 `prepare_frem_data` → `emit_nlmixr2_frem`), and asserts nlmixr2
 compiles the model with the expected joint Ω size. Scope stops at
 compile because FOCE-I on a 144-row FREM-augmented dataset
 regularly exceeds 15 min even with minimal iteration caps — that
 belongs to a nightly benchmark. The compile path still exercises
 DSL emit → prepare_frem_data → nlmixr2 parse/validate → rxode2
 model generation, which is the critical emitter-correctness chain.

Full suite: **1490 fast-path tests passing** (62 new in the missing-data
sweep) + 6 live nlmixr2 integration tests (2 fit, 2 compile-only, 1
multi-analyte, 1 contamination guard). `mypy --strict` clean across 79
source files; `ruff format` + `check` clean.

Residual not covered by this live test pass:
- **Mavoglurant / Suite A–C datasets** through the FREM path remain
 a future integration milestone — theophylline is covered, these
 larger benchmarks require multi-minute FOCE-I fits that belong in
 a nightly/weekly suite.
- **Full `_run_frem_stage` live orchestrator test** was prototyped
 during this sweep but excluded from CI because FOCE-I with the
 covariance step on even tiny FREM-augmented data regularly exceeds
 the 200 s runner default timeout. The FREM orchestrator path is
 covered indirectly via (a) the stub-runner composition test in
 `test_frem_runner.py`, (b) the binary/TV live fits in this file,
 and (c) the `_run_mi_stage` live Rubin test.

### Fixed — orchestrator FREM work directory
- `Orchestrator._run_frem_stage` now creates `run_dir/frem/` before
 invoking `run_frem_fit` so `prepare_frem_data` can write
 `frem_augmented.csv` without raising the pandas "Cannot save file
 into a non-existent directory" error surfaced by the live test.
- The same directory is passed to the default FOCE-I
 `Nlmixr2Runner.work_dir`, so the R subprocess and the augmented
 CSV land next to each other under the bundle.

### Fixed — NCA initial-estimate refinement (PKNCA-compatible)
- Replaced last-3-point terminal regression with PKNCA-style curve-stripping
 (`_select_lambda_z`): iterate window sizes 3..n_post-Cmax, fit last-k
 points, filter `kel>0`, maximize adj-R² with a `1e-4` tiebreak on most
 points (PKNCA `pk.calc.half.life` `allow.tmax.in.half.life=FALSE`).
- Replaced linear trapezoidal AUC with linear-up/log-down
 (`_auc_lin_up_log_down`, Purves 1992); zero-zero intervals contribute 0
 and declining positive intervals use `dt*(C1-C2)/ln(C1/C2)` with a
 numerical guard that falls back to linear when `C1≈C2`.
- Added per-subject QC gates (adj-R²≥0.80, extrap≤20%, span≥1·t½, n_λz≥3);
 subjects failing QC are excluded from the population median.
- Added literature-prior fallback via `RunConfig.fallback_estimates` when
 ≥50% of subjects fail QC (wires `DatasetCard.published_model.key_estimates`
 when present; otherwise conservative defaults).
- Plumbed `last_dose_time` into `_compute_nca_single_subject` so the
 steady-state AUC_tau window anchors on the true last-dose interval
 `[last_dose_time, last_dose_time + tau]`, not the first observation.
- Removed per-subject CL/V/ka clipping so the unit-scale heuristic receives
 unbiased raw estimates; `estimate_population_level` now applies the same
 unit-scale so single-subject / sparse fallbacks are not silently 1000x off.
- Hardened `_detect_unit_scale_factor`: requires `dose_median ≥ 1.0`
 (mg-scale human dosing) and medians only over positive doses, so
 preclinical studies and placebo arms do not trigger a spurious x1000
 correction.
- Clipped regression R² to [0, 1] in `_log_linear_fit` so floating-point
 noise on near-constant log(C) windows cannot push adj-R² above 1 and
 corrupt the λz tiebreak.
- Emits `nca_diagnostics.jsonl` and optional `diagnostics/nca_plots/*.png`
 in the reproducibility bundle (one row per subject with λz fit quality,
 AUC extrapolation, exclusion reason).
- New `NCASubjectDiagnostic` Pydantic model in `apmode.bundle.models`.
- Fixes the production-blocking mavoglurant Suite B failure where NCA seed
 `CL=48 L/h` caused SAEM to converge on degenerate `lCL=-3.4`.

### Added — Profiler-stage error-model selection heuristic
- New `ErrorModelPreference` Pydantic model + `recommend_error_model()`
 decision tree (Beal 2001, Ahn 2008): BLQ≥10% → BLQ_M3 with
 proportional/combined only (never additive-only, per Ahn 2008);
 LLOQ/Cmax>5% or terminal log MAD>0.35 → combined; dynamic range>50 +
 CV<80 → proportional; narrow range + low CV → additive (biomarker-like);
 default proportional with combined fallback.
- New `EvidenceManifest` fields: `error_model_preference`,
 `cmax_p95_p05_ratio`, `dv_cv_percent`, `terminal_log_residual_mad`.
- `_compute_cmax_dynamic_range` now takes p95/p05 of per-subject Cmax
 (not of all DV values), so tail samples near LLOQ do not inflate the
 range for narrow-Cmax studies.
- `_compute_dv_cv_percent` switches to population CV (ddof=0) to match
 the calibration of `recommend_error_model` thresholds.
- `_nonlinear_clearance_confidence` now filters BLQ-zero tails and uses
 the same Cmax-inclusive post-Tmax slice as `_detect_curvature_nonlinearity`
 so the boolean detector and confidence score always operate on the same
 per-subject ratios.
- `_get_dose_auc_pairs` normalizes by median AMT instead of `iloc[-1]` so
 loading/titration/maintenance regimens do not skew the Spearman
 dose–AUC correlation used in nonlinear-clearance detection.
- `_assess_absorption_coverage` uses the positive-DV Tmax so BLQ zeros
 cannot masquerade as peak times.
- `SearchSpace.from_manifest` consumes `error_model_preference` before
 search runs; legacy `BLQ>20%` override retained for manifests emitted
 before the heuristic existed. Sparse filter preserves `"none"` so
 IV-sparse datasets still generate an IV candidate. Warns when BLQ
 M3/M4 is forced but `manifest.lloq_value` is missing.
- `_build_spec` rejects `additive` in the BLQ M3/M4 residual-error slot
 (Ahn 2008: additive sigma absorbs censored mass under M3); falls back
 to proportional.

### Added — FREM binary/time-varying covariates + orchestrator execution
- **`FREMCovariate.transform="binary"`** — accepts 0/1-coded
 categorical covariates with an additive-normal endpoint and strict
 validation on both `summarize_covariates` and `prepare_frem_data`.
 The off-diagonal Ω entry between the binary covariate eta and a PK
 eta estimates the linear association between the PK parameter and
 the binary group — the standard "categorical-FREM compromise"
 (Karlsson 2011). Multi-level categorical covariates are handled by
 one-hot encoding upstream into k−1 binary indicators.
- **`FREMCovariate.time_varying`** — when True, `prepare_frem_data`
 emits one observation row per (subject, TIME) where the covariate
 value is observed, and `emit_nlmixr2_frem` leaves `sig_cov_*`
 estimable (instead of `fix(...)`) so the residual absorbs
 within-subject variation while the joint Ω entry continues to
 carry the between-subject component. Dedup keys on `(TIME,)` per
 subject and raises on conflicting values at the same timepoint.
- **Auto-detection**: `summarize_covariates` now sets
 `time_varying=True` automatically when any subject has more than
 one distinct non-NaN value for the covariate.
- **`Orchestrator._run_frem_stage`** (new) — refits the best healthy
 classical candidate (filtered to `backend="nlmixr2"` + finite BIC +
 not `ill_conditioned`) via `run_frem_fit` after the classical
 search. The FREM fit is appended to `search_outcome.results` as
 `frem_<model_id>` and emitted to the bundle as a standard backend
 result. Defaults to a FOCE-I-configured `Nlmixr2Runner` when
 `Orchestrator(frem_runner=...)` is not provided.
- **`Orchestrator._run_mi_stage`** (new) — drives the m-imputation
 loop after classical search: produces m imputed CSVs via
 `R_MiceImputer` / `R_MissRangerImputer` (or an injected
 `mi_provider`), refits the **frozen classical candidate set** on
 each imputation so candidate_ids align, applies Rubin's rules to
 the pooled `(estimate, SE)` tuples, and emits
 `imputation_stability.json`. Gate 1 consumes the per-candidate
 stability entries to enforce `convergence_rate ≥ 0.5` and
 `rank_stability ≥ 1 − policy.imputation_stability_penalty`.
- **Stage 5c** in `apmode run` now branches on
 `directive.covariate_method`: FREM → `_run_frem_stage`, MI-* →
 `_run_mi_stage`, otherwise no-op. Failures are logged with
 `exc_info=True` and do not abort the classical search results.
- **`Nlmixr2Runner.run(compiled_code_override=...)`** kwarg — the
 FREM path supplies pre-emitted nlmixr2 model code via
 `emit_nlmixr2_frem` without monkey-patching the DSL compiler.
- **`Orchestrator(frem_runner, mi_provider)`** — optional
 constructor-level dependency injection points so tests and custom
 deployments can substitute stubs or specialized backends without
 re-invoking the defaults.

### Added — Rubin's rules for per-parameter pooling
- **`apmode.search.stability.rubin_pool`** (new). Implements the
 canonical Rubin (1987) decomposition for a scalar parameter across
 imputations:
 - `Q̄` — pooled point estimate (mean of per-imputation estimates)
 - `Ū` — within-imputation variance (mean of per-imputation SE²)
 - `B` — between-imputation variance (sample variance of per-imp estimates)
 - `T` — total variance = `Ū + (1 + 1/m) * B`
 - `dof` — Barnard–Rubin degrees of freedom for inference
 ``None`` SEs (backend did not compute them) degrade gracefully to a
 between-only decomposition instead of throwing.
- **`PerImputationFit.parameter_estimates`** (new optional field)
 carries per-parameter ``(estimate, se | None)`` from the backend.
 When set, `aggregate_stability` invokes `_rubin_pool_candidate` per
 converged candidate and populates the new
 `ImputationStabilityEntry.pooled_parameters` field with the full
 5-tuple (`pooled_estimate`, `within_var`, `between_var`,
 `total_var`, `dof`) keyed by parameter name.
- **Bundle surface**: `imputation_stability.json` entries now include
 a `pooled_parameters` dict so downstream reports can quote the
 Rubin-pooled estimates + proper variance decomposition instead of
 just the arithmetic means of OFV/AIC/BIC.
- **5 new unit tests** in `tests/unit/test_missing_data.py::TestRubinPool`:
 single imputation identity case, two-imputation variance
 decomposition, `None` SE degradation, misaligned-input validation,
 end-to-end aggregate pooling from `PerImputationFit` rows.

### Fixed — CLI hardening + multi-model review (2026-04-15)

Two rounds of review on the
Typer CLI surface uncovered a class of JSON-parsing and exit-code bugs
that were never CLI-tested. All fixed; the new `tests/unit/test_cli.py`
(60 tests, ~0.8s wall time) now covers every top-level command.

- **`log --top` was unusable without the flag** — default=0 collided
 with `min=1`, so every invocation returned Typer exit 2. Lowered
 `min=0` and documented `0 = disabled`.
- **`_load_json` crashed on missing files and non-dict JSON** — only
 `JSONDecodeError` was caught, so a missing optional artifact raised
 `FileNotFoundError`, and a `ranking.json` containing `[1,2,3]` (valid
 JSON, not an object) crashed every `.get()` call downstream. Now
 returns `None` with a warning on `FileNotFoundError`, `PermissionError`,
 `IsADirectoryError`, `OSError`, `JSONDecodeError`, and non-dict values.
- **JSONL consumers crashed on non-object rows** — every `json.loads(line).get(...)`
 path in `inspect`, `log --failed`, `_show_bundle_overview`, and the
 agentic trace parsers has been routed through a new `_parse_json_dict_row`
 helper that validates the row is a dict.
- **Empty JSONL reported "1 total, 0 converged"** — `"".split("\n") == [""]`
 made `_show_bundle_overview` count a blank trajectory as one candidate.
- **`_launch_run` swallowed all non-zero exit codes** —
 `contextlib.suppress(SystemExit)` meant `apmode explore -y` returned
 0 even when the inner pipeline failed. Also: `typer.Exit` is
 `click.exceptions.Exit` (a `RuntimeError` subclass), NOT `SystemExit`,
 so the old suppressor never fired on the actual error path. The new
 handler catches both types explicitly and propagates non-zero codes.
- **`_validate_file` / `_validate_jsonl` crashed on adversarial bundles** —
 neither caught `OSError`. A required path being a directory or
 unreadable file now reports a clean validation failure.
- **`graph --output nested/path/out.json` crashed** — parent dirs were
 never created before `write_text`.
- **`graph` did not distinguish missing vs. malformed `search_graph.json`** —
 missing still exits 0 (non-agentic bundle); malformed now exits 1.
- **`explore` discarded `fetch_dataset`'s returned path** and ran
 profile / NCA / search-space generation unguarded. All three stages
 are now wrapped with clean error messages and exit 1 on failure.
- **Dead code**: the standalone `_show_trace_cost` (59 LOC) was never
 called — all sites used `_show_trace_cost_multi`. Deleted.
- **CLI docs in CLAUDE.md were wrong**: `--agentic anthropic` (actually
 a boolean flag + `--provider`), `validate <dataset.csv>` (actually
 `<bundle_dir>`). Fixed.

Net delta: `src/apmode/cli.py` ~-30 LOC after deleting dead code and
centralising JSON parsing through helpers. `mypy --strict` and `ruff`
remain clean. Full unit suite 1297 passing.

### Fixed — multi-agent review of orchestrator MI/FREM wiring (2026-04-14)

Five consensus-level bugs surfaced during review
of the new `_run_frem_stage` / `_run_mi_stage` methods and the FREM
emitter's time-varying / categorical support.

- **HIGH: MI stage refit was misaligned.** `_run_mi_stage` previously
 built `ref_specs` from the classical search but never used them —
 each imputation ran a *fresh* `SearchEngine`, generating different
 warm-started children per draw. Stability metrics (convergence_rate,
 rank_stability) were therefore computed against misaligned
 candidate_ids and meaningless for the primary search candidates
 . The refactor now invokes the main runner
 directly over a frozen `ref_specs` list per imputation so
 candidate_ids match the classical search exactly. NODE specs are
 excluded at the filter step because they cannot be fit via
 `Nlmixr2Runner`.
- **HIGH: FREM warm-start accepted unhealthy bases.** `_run_frem_stage`
 picked `min(converged, key=bic)` without filtering, so NODE specs
 or ill-conditioned classical fits could seed the joint-Ω refit and
 destabilize estimation . Now requires
 `backend == "nlmixr2"`, finite BIC, and
 `not ill_conditioned` before selecting the best candidate.
- **HIGH: time-varying FREM was never actually emitted from the
 orchestrator path.** `summarize_covariates` unconditionally emitted
 `time_varying=False` on every `FREMCovariate`, so the Stage 5c FREM
 fit always produced baseline-only rows even when routing had
 flipped the directive based on `EvidenceManifest.time_varying_covariates`
 . `summarize_covariates` now auto-detects TV per covariate
 by checking within-subject variation and propagates the flag to
 `FREMCovariate.time_varying` — downstream `prepare_frem_data`
 correctly produces per-(subject, TIME) rows and `emit_nlmixr2_frem`
 leaves `sig_cov_*` estimable.
- **HIGH: Augmented rows leaked PK-context columns.** `_build_aug_row`
 previously copied every source-row column onto covariate observation
 rows; `CENS` / `LIMIT` / `BLQ_FLAG` / `RATE` / `DUR` / `SS` / `II`
 then routed the covariate through the PK BLQ likelihood or treated
 it as part of an infusion envelope . These
 columns are now explicitly cleared on augmentation rows.
- **HIGH: Time-varying dedupe key was too permissive.** Dedup was
 `(TIME, value)`, so conflicting values at the same subject/TIME
 collapsed silently instead of surfacing the input-data ambiguity
 . Dedup now keys on `(TIME,)` per subject and
 raises `ValueError` with a pointer when the same timepoint has
 distinct covariate values, forcing callers to aggregate upstream.
- **Hygiene: Stage 5c exception handling.** Broadened `except
 Exception` narrowed to
 `(BackendError, RuntimeError, ValueError, NotImplementedError)`
 with `exc_info=True` so the warning surfaces the traceback instead
 of silently masking real bugs .
- **Test hygiene**: `test_low_blq_burden_does_not_force_m3` updated
 from the legacy 20% BLQ threshold to 5% to match the new Beal 2001
 / Ahn 2008 10% `recommend_error_model` threshold in the profiler.

### Fixed — full-codebase review sweep (2026-04-14)

Nine issues surfaced by a comprehensive Python 3.12+ code review
(2 critical, 7 high) plus one assertion-under-`-O` hardening.

- **CRIT: Double NCA invocation.** `NCAEstimator.build_entry`
 (`src/apmode/data/initial_estimates.py`) previously re-invoked
 `estimate_per_subject()` internally, doubling NCA compute and
 overwriting the diagnostics list produced by the earlier call. The
 method now accepts a keyword-only `estimates` parameter reused
 verbatim when provided; the orchestrator passes the pre-computed
 `nca_estimates`.
- **CRIT: Runtime Pydantic validation failure on `source` field.**
 `build_entry(source: str = "nca")` accepted any string but
 `InitialEstimateEntry.source` is
 `Literal["nca","warm_start","fallback"]`, so off-Literal values
 raised at runtime without type-checker warning. Narrowed to the
 Literal.
- **HIGH: Hardcoded version string in reproducibility bundle.**
 `Orchestrator.run` wrote `apmode_version="0.2.0-dev"` in both
 `BackendVersions` and `ReportProvenance` despite hatch-vcs
 deriving the real version. Switched to `apmode.__version__`.
- **HIGH: CWD-dependent gate-policy lookup.** `_load_policy` resolved
 `Path("policies") / f"{lane}.json"` against the process CWD; any
 invocation outside the repo root silently skipped all governance
 gates. Resolution now anchors to the package directory
 (`Path(__file__).resolve().parents[3] / "policies"`).
- **HIGH: VPC gate unconditionally failed Phase 1 candidates.**
 `_check_vpc_coverage` returned `passed=False` when
 `diagnostics.vpc is None`, but no Phase 1 backend populates VPC
 yet — every candidate failed Gate 1. Added
 `Gate1Config.vpc_required: bool = True`; when `False`, missing VPC
 passes with `observed="vpc_not_configured"`. All three shipped
 lane policies set `vpc_required: false` until the backend VPC
 pipeline lands.
- **HIGH: Private-attribute mutation of `AgenticRunner._trace_dir`.**
 `_run_agentic_stage` assigned `agentic._trace_dir = …` to redirect
 Mode 1 (refine) vs Mode 2 (independent) trace output, coupling the
 orchestrator to the runner's private state. Introduced
 `AgenticRunner.with_trace_dir()` context manager; the orchestrator
 now uses `with agentic.with_trace_dir(…):`.
- **HIGH: MI inner engine emitted no bundle artifacts.**
 `_run_mi_stage._fit_one_imputation` called `SearchEngine.run()`
 with no `emitter`, so per-imputation compiled specs and trajectory
 entries were never written — a reproducibility-bundle gap (PRD
 §4.3.2). The outer `BundleEmitter` is now plumbed through.
- **HIGH: Agentic exhaustion indistinguishable from transient
 failure.** `AgenticRunner.run` raised a bare `RuntimeError` when
 all iterations completed without a converged result; the
 orchestrator's `except (BackendError, RuntimeError)` logged it as
 a generic warning. Added
 `apmode.errors.AgenticExhaustionError(BackendError)` carrying the
 iteration count; orchestrator handles it distinctly with its own
 log message.
- **HIGH: `SearchSpace.apply_directive` in-place mutation.** The
 method mutated `self` while documented as returning `self` for
 chaining — reusing a space across directive applications silently
 accumulated state. Now returns a fresh `SearchSpace` via
 `dataclasses.replace`; tests updated to use the returned value.
- **Hardening: `assert isinstance(self._agentic_runner, AgenticRunner)`
 under `python -O`.** Replaced with an explicit `TypeError` so the
 guard remains effective when assertions are stripped.

Verification: `uv run pytest tests/` → 1447 passed / 7 skipped,
`uv run mypy src/apmode/ --strict` clean, `uv run ruff check`
and `ruff format --check` clean.

### Added — Policy-driven missing-data pipeline
**Covariate imputation (MI) and Full Random Effects Models (FREM) are now
first-class options; BLQ method selection is lane-driven rather than
heuristic.** The full lifecycle is: policy → router directive → bundle
artifact → backend execution helper → credibility report.

#### Policy + directive
- **`MissingDataPolicy`** block in `src/apmode/governance/policy.py`
 with lane-tiered defaults in `policies/{submission,discovery,optimization}.json`
 (policy files bumped to `0.2.0`). Selects:
 - Covariate method: `MI-PMM`, `MI-missRanger`, `FREM`, `FFEM`, `exclude`.
 - BLQ method: `M7+` default, `M3` above `blq_m3_threshold` (or when
 `blq_force_m3=true`).
 - Imputation budget: `m_imputations`, optional `adaptive_m` / `m_max`.
 - Agentic guards: `llm_pooled_only`, `imputation_stability_penalty`.
 - Lane tiers: Submission m=20 → 40, Discovery m=5 + penalty 0.25,
 Optimization m=10 → 20 + penalty 0.5.
- **`MissingDataDirective`** (`src/apmode/bundle/models.py`) — binding
 router output carrying the resolved method, m, BLQ method, LLM
 guard, penalty weight, and rationale. Resolver in
 `src/apmode/data/missing_data.py::resolve_directive` cites Nyberg
 2024, Wijk 2025 (DiVA), Bräm CPT:PSP 2022.
- **Router integration**: `src/apmode/routing.py::route` accepts the
 policy and attaches the directive to every `DispatchDecision`.
- **Bundle artifact**: `BundleEmitter.write_missing_data_directive`
 serializes the directive as `missing_data_directive.json` in the
 reproducibility bundle.

#### Profiler + search
- `EvidenceManifest.time_varying_covariates` (new) populated by
 `_detect_time_varying_covariates` in `src/apmode/data/profiler.py`;
 drives FREM preference via `frem_for_time_varying`.
- `SearchSpace.apply_directive` maps directive BLQ method → DSL
 emission (`M3`/`M4` → BLQ observation models; `M7+`/`M6+`/`M1` →
 preprocessing-side handling). `blq_strategy` field records the
 Beal tag for auditing.

#### MI execution infrastructure
- **R imputer harness** (`src/apmode/r/impute.R`): JSON-typed
 request/response contract; dispatches to `mice::mice(method="pmm")`
 or **`missRanger::missRanger(num.trees=100, pmm.k=10)`** — the
 modern ranger-backed successor to missForest (Mayer CRAN 2.6.x,
 verified via ref.tools).
- **Python adapters** (`src/apmode/data/imputers.py`): `R_MiceImputer`
 and `R_MissRangerImputer` implement the `ImputationProvider`
 protocol.
- **Stability aggregation** (`src/apmode/search/stability.py`):
 `run_with_imputations` drives the m-fit loop via an injected search
 callable; `aggregate_stability` computes Rubin-pooled OFV/AIC/BIC,
 convergence rate, top-K rank stability, within/between variance
 proxy, and covariate sign consistency. Emitted to the bundle as
 `imputation_stability.json`.

#### FREM execution infrastructure
- **FREM emitter** (`src/apmode/dsl/frem_emitter.py`, new). Emits an
 nlmixr2 program where covariates are observations drawn from a joint
 MVN with the PK random effects — a single joint Ω matrix carries
 PK-covariate covariances; missingness is handled inside the NLME
 likelihood (no imputation, no Rubin pooling). References: Karlsson
 2011, Yngman 2022, Nyberg 2024.
 - Public API: `FREMCovariate`, `summarize_covariates`,
 `prepare_frem_data`, `emit_nlmixr2_frem`.
 - Scope: static subject-level continuous covariates;
 `transform="log"` supported for positive/right-skewed covariates
 (Yngman 2022 conditioning).
 - Correctness details: covariate means emitted as estimable thetas
 (`<-`), residuals fixed via `fix(...)` so the random effect
 absorbs BSV at one-obs-per-subject, `pd.isna()` null detection,
 DVID collision guard, duplicate-name rejection, baseline (min
 TIME) row selection.
 - **Endpoint routing is data-driven via the `DVID` column.** nlmixr2
 assigns endpoints DVID 1 (PK), 2, 3, … in declaration order.
 No `| <expr>` is emitted on endpoint RHS — nlmixr2 5.0 rejects
 any condition after `|` with "the condition '…' must be a simple
 name" (empirically confirmed 2026-04-14).
 `_FREM_DVID_OFFSET = 2` to align with nlmixr2's implicit numbering.
- **FREM runner** (`src/apmode/backends/frem_runner.py`, new).
 `run_frem_fit` composes `summarize_covariates` + `prepare_frem_data`
 + `emit_nlmixr2_frem` with `Nlmixr2Runner.run(compiled_code_override=...)`.
 `Nlmixr2Runner.run` gains `compiled_code_override` kwarg so the FREM
 path supplies its own R code without monkey-patching
 `emit_nlmixr2`.

#### Governance + agentic guard
- **Gate 1 imputation-stability check** (`src/apmode/governance/gates.py`).
 For MI runs: `convergence_rate ≥ 0.5` (hard floor) and
 `rank_stability ≥ 1 − imputation_stability_penalty`. Non-MI runs
 mark the check `not_applicable` and pass.
- **Agentic pooled-only guard**. When `directive.llm_pooled_only=true`
 the agentic runner substitutes a pooled-only LLM summary
 (`summarize_stability_for_llm` in `diagnostic_summarizer.py`) — the
 LLM never sees per-imputation diagnostics, closing the
 imputation-cherry-picking path.
- **Credibility report** auto-appends `OMEGA_POOLING_CAVEATS`
 (Rubin-pool limits, log-Cholesky note, EBE non-poolability) whenever
 MI is active. Gate 2.5 now documents the Ω-pooling caveat
 automatically.

### Tests
- **49 new unit tests** in the default fast path:
 - `tests/unit/test_missing_data.py` (31): resolver branches,
 stability aggregation, SearchSpace directive overlay, Gate 1
 imputation check.
 - `tests/unit/test_frem_emitter.py` (18): `FREMCovariate` validation,
 `summarize_covariates` (identity + log transforms, duplicates,
 baseline), `prepare_frem_data` (DVID collisions, log-scale DV,
 missing-subject handling), `emit_nlmixr2_frem` (joint Ω block,
 fixed residuals, covariate-link stripping, no DVID pipe).
- **Live end-to-end tests** (marked `live`/`slow`; skipped by default):
 - `tests/unit/test_frem_emitter.py::TestFREMLiveIntegration`: spawns
 real Rscript + nlmixr2; verifies (a) the emitted FREM model
 compiles and exposes the expected endpoint count, (b) FOCE-I
 actually fits the joint Ω and learns a non-degenerate
 `eta.cov.WT` on tiny synthetic data (regression guard against
 SAEM collapse observed during development).
 - `tests/unit/test_imputers_live.py`: spawns real Rscript against
 `mice` and `missRanger`; verifies imputed CSVs appear, no residual
 NaN in imputed columns, and between-imputation variance proves MI
 is functioning (not collapsing to single imputation).
 - `tests/unit/test_frem_runner.py`: Python-side composition test
 for `run_frem_fit` using a stub runner (fast; runs every PR).
- **Full suite**: 1391 passing in the default fast path. ruff clean,
 `mypy --strict` clean across 79 source files.

### Known constraints and residuals
- **SAEM is not supported for FREM.** nlmixr2 SAEM treats subject-level
 covariate observations as dynamic sampling targets and collapses the
 random-effect variance to zero (verified 2026-04-14). The FREM runner
 must use `Nlmixr2Runner(estimation=["focei"])`.
- **Orchestrator MI/FREM execution** now wired end-to-end
 (`Orchestrator._run_frem_stage` / `_run_mi_stage` called from
 Stage 5c of `apmode run` when the directive demands FREM or MI-*;
 bundle artifacts `frem_augmented.csv`, `imputation_stability.json`
 and the refit results emitted automatically). Defaults construct
 `Nlmixr2Runner(estimation=["focei"])` for FREM and
 `R_MiceImputer` / `R_MissRangerImputer` for MI; both can be
 overridden via `Orchestrator(frem_runner=..., mi_provider=...)`.
- **Categorical covariates in FREM** supported via
 `FREMCovariate.transform="binary"` (0/1-coded additive-normal
 endpoint; multi-level via k−1 one-hot indicators). Proper
 logit-likelihood binomial endpoints remain future work.
- **Time-varying covariates in FREM** supported via
 `FREMCovariate.time_varying=True` with auto-detection in
 `summarize_covariates`. Per-(subject, TIME) augmentation rows are
 emitted and `sig_cov_*` is left estimable so the residual absorbs
 within-subject variation while the eta continues to carry BSV.

### Fixed — NCA unit-scaling on heterogeneous-unit datasets
- **NCA initial estimates are now unit-aware** (`src/apmode/data/initial_estimates.py`).
 Discovered via Suite B mavoglurant run: NCA computed `CL = Dose / AUC`
 directly with no unit conversion, producing `CL=0.05 L/h` (1000x too low)
 when dose is in mg but DV is in ng/mL — a routine pharmacometric convention.
 SAEM then converged to degenerate parameter regions (`lCL=-3.6`, i.e.
 CL=0.028 L/h), and Gate 1's `parameter_plausibility` check correctly
 rejected all candidates.
- **`_detect_unit_scale_factor()` heuristic**: when `CL < 0.5 L/h` and
 `DV magnitude > 50`, infers ng/mL DV convention and applies `x1000`
 scaling to CL and V. Multiplier exposed as `_unit_scale_applied` in NCA
 results for downstream auditability. Per-magnitude tier:
 - `CL < 0.0001` and `DV > 50000` → `x1e6` (pg/mL DV)
 - `CL < 0.5` and `DV > 50` → `x1000` (ng/mL DV)
 - otherwise → `x1.0` (units commensurate)
- **Verified across three datasets**: theophylline (mg + mg/L → no scaling,
 CL=2.8 L/h), mavoglurant (mg + ng/mL → x1000, CL=49 L/h), A1 synthetic
 (commensurate → no scaling).
- 8 new regression tests in `tests/unit/test_nca_unit_scaling.py`
 covering commensurate units, ng/mL detection, edge cases, and a
 parametrized dose/DV-scale matrix.

### Fixed — BLQ pipeline LLOQ propagation
- **Profiler now extracts LLOQ from data** (`src/apmode/data/profiler.py`).
 When `BLQ_FLAG=1` rows exist, `_extract_lloq_value()` reads the explicit
 `LLOQ` column (preferred) or falls back to the `DV` value of censored rows
 (M3 convention sets DV=LLOQ). Result populated as `EvidenceManifest.lloq_value`.
- **SearchSpace propagates the actual LLOQ into BLQ_M3/M4 candidates**.
 Prior bug: `force_blq_method="m3"` was set when `blq_burden > 0.20`, but
 `lloq_value` stayed at the default `1.0`. On the mavoglurant Suite B run
 (actual LLOQ=32.8), this caused all candidates to fit the wrong M3
 censoring integral and Gate 1 to reject every candidate. Now the SearchSpace
 reads `manifest.lloq_value` and passes the correct LLOQ into all BLQ_M3
 observation models.
- 3 new integration tests (`tests/integration/test_blq_integration.py`):
 high BLQ + auto-computed LLOQ, low BLQ (below 20% threshold) does not
 force M3, profiler fallback to DV when LLOQ column is missing.

### Added — Bayesian backend (5th paradigm, Phase 2+)
- **Bayesian PK backend via Stan/Torsten** (`src/apmode/backends/bayesian_runner.py`,
 `src/apmode/bayes/harness.py`). Joins classical NLME, automated search,
 agentic LLM, and hybrid NODE to give APMODE **five paradigms**, covering
 FDA's Jan 2026 draft guidance on Bayesian methodology for primary inference
 (FDA-2025-D-3217) and the growing Project Optimus adoption of Bayesian
 designs. Drives cmdstanpy through a subprocess harness; file-based JSON
 request/response mirroring the `Nlmixr2Runner` pattern with asyncio +
 `start_new_session=True` + SIGKILL on process-group timeout. End-to-end
 smoke-tested against the Boeckmann 1994 theophylline dataset — posterior
 recovered CL=2.55 L/h, V=38.75 L, ka=1.72 /h (literature values ~2.8, ~35,
 ~1.5).
- **DSL prior AST** (`src/apmode/dsl/priors.py`). Ten prior families
 (Normal, LogNormal, HalfNormal, HalfCauchy, Gamma, InvGamma, Beta, LKJ,
 Mixture, HistoricalBorrowing — Schmidli 2014 robust MAP) with a
 parameterization-schema validator that rejects invalid (target, family)
 pairs at compile time. FDA-aligned `PriorSpec.source` taxonomy
 (uninformative / weakly_informative / historical_data / expert_elicitation
 / meta_analysis) enforces justification rules via Pydantic validators.
- **`SetPrior` as the 7th `FormularTransform`** (`src/apmode/dsl/prior_transforms.py`).
 The agentic LLM can now propose priors under the DSL ceiling with full
 audit trail; replace-or-append semantics on `DSLSpec.priors`.
- **Stan emitter consumes user priors** (`src/apmode/dsl/stan_emitter.py`).
 Emits user-declared priors on structural log-scale params, IIV omegas,
 IOV omegas, residual sigmas, and covariate betas. Mixture priors compile
 to numerically stable `target += log_sum_exp([log(w) + lpdf, ...])` form
 (zero-weight components dropped). Robust-MAP historical borrowing compiles
 to a two-component mixture with a wide (Normal(0, 10) on log-scale) weak
 component.
- **BackendResult extensions for Bayesian submissions**
 (`src/apmode/bundle/models.py`): new backend literal `"bayesian_stan"`;
 new `PosteriorDiagnostics` (R-hat max, ESS bulk/tail min, divergences,
 max-treedepth hits, E-BFMI, MCSE, Pareto-k counts); new `SamplerConfig`
 (chains/warmup/sampling/adapt_delta/max_treedepth + cmdstan_version /
 stan_version / compiler_id for reproducibility). `ParameterEstimate` now
 carries posterior summaries (posterior_sd, q05/q50/q95) alongside the
 frequentist fields. New `PriorManifest` and `SimulationProtocol` models
 for FDA-required prior-justification and prospective-simulation artifacts.
- **Bayesian bundle artifacts** (`src/apmode/bundle/emitter.py`): new
 `write_prior_manifest`, `write_simulation_protocol`, `write_mcmc_diagnostics`,
 and `copy_posterior_draws` methods emit to a `bayesian/` subdirectory,
 matching the existing per-candidate artifact pattern (credibility/, loro_cv/).
- **Gate-1 Bayesian thresholds** (`src/apmode/governance/policy.py`):
 `BayesianThresholds` config (R-hat ≤ 1.01, ESS ≥ 400, divergences = 0,
 E-BFMI ≥ 0.3, Pareto-k ≤ 0.7) applied only when backend == "bayesian_stan",
 following Vehtari et al. 2021 rank-normalized R-hat recommendations.
- **CLI `--backend bayesian_stan` flag** (`src/apmode/cli.py`) plus
 `--bayes-chains`, `--bayes-warmup`, `--bayes-sampling`, `--bayes-adapt-delta`,
 `--bayes-max-treedepth` for sampler configuration. Installs via
 `uv sync --extra bayesian` (cmdstanpy, arviz, pyarrow).
- **Integration plan documentation** (`docs/plans/2026-04-14-phase2-bayesian-backend.md`).

### Fixed — review sweep
- **CRITICAL: `_prune_stale_variability` silently dropped all priors** during
 structural swaps (`SwapModule` / `ReplaceWithNODE`). The `DSLSpec` rebuild
 omitted `priors=`, erasing user/agentic-declared priors. All non-SetPrior
 transforms now carry priors forward; `_prune_stale_variability` additionally
 prunes orphaned priors (e.g., a prior on `ka` after swap to `IVBolus`).
 Regression tests in `tests/unit/test_prior_transforms.py::TestPriorsSurviveStructuralSwap`.
- **`IVBolus` was missing from `_validate_swap_position`** — a valid
 `AbsorptionModule` variant that couldn't be swapped to. Added.
- **`SearchGraphNode.backend` literal was missing `"bayesian_stan"`** —
 would have caused Pydantic validation failures on any search graph
 containing a Bayesian candidate.
- **Half-family lpdfs in mixture priors dropped the log(2) normalization
 constant** (`src/apmode/dsl/stan_emitter.py::_component_lpdf`). When mixed
 with fully-supported Gamma/InvGamma components, HalfNormal/HalfCauchy were
 artificially down-weighted by 50%. Now emits
 `(normal_lpdf(x | 0, sigma) + log(2))` for half-family components.
- **HistoricalBorrowing weak component was too narrow**
 (Normal(0, 2) on log scale ≈ 95% CI [0.02, 55]) — would penalize plausible
 PK parameter values like V > 100 L or CL > 50 L/h. Widened to Normal(0, 10).
- **Mixture weights floor replaced with zero-weight filter** — cleaner Stan
 AST and avoids emitting `log(1e-300)` placeholders.
- **`_compute_eta_shrinkage` assumed omega=1** — computed
 `1 - var(eta_raw_mean)` instead of the correct
 `1 - var(omega·eta_raw_mean) / E[omega²]`. Shrinkage gate decisions were
 systematically wrong.
- **Time-varying covariates silently collapsed to baseline** (first-per-subject
 value) in the harness. Now detected and rejected with a clear error.
- **ADDL/II/SS NMTRAN semantics silently ignored** in the harness —
 multi-dose ADDL rows were dropped, steady-state rows were treated as
 regular doses. Now rejected with a preprocessing prompt.
- **DV=0 observations crashed the lognormal likelihood** (pre-dose baseline
 rows in theophylline) — the harness now drops them with a warning pointing
 to BLQM3/M4 as the correct handling.
- **`preexec_fn=os.setsid` replaced with `start_new_session=True`** for
 safer process-group isolation on macOS/Linux.
- **Response write-path outside try/except** — disk failures would escape
 error classification. Now caught with last-ditch stderr report.
- **Widened ID column detection** — harness now accepts `ID`, `NMID`,
 `USUBJID`, `SUBJECT_ID`, or `PATIENT_ID`.
- **Covariate prior schema now accepts `Mixture` and `HistoricalBorrowing`**
 (in addition to Normal) — enables meta-analytic priors on allometric
 exponents and other covariate coefficients.

### Added — Pre-Bayesian (unchanged from prior release candidate)
- **Per-mode agentic trace inspection in CLI**: `apmode trace` gained a
 `--mode {refine|independent}` flag to filter which agentic mode's history
 is shown. Cost aggregation via `--cost` now reports per-mode breakdowns
 and a grand total when multiple modes are present. New helper
 `_discover_agentic_mode_dirs()` transparently supports both the new
 subdirectory layout (`agentic_trace/refine/`, `agentic_trace/independent/`)
 and legacy flat layouts (`agentic_trace/iter_*.json`) for backward
 compatibility with pre-fix bundles. `apmode lineage` and `apmode inspect`
 also updated.
- **Agentic candidates surfaced in `candidate_lineage.json`**: previously
 only the search DAG drove the lineage file, so agentic candidates (which
 bypass the classical search engine) were missing from the reproducibility
 bundle's top-level lineage. They now appear with `transform="agentic_llm"`
 and `parent_id=None`; the full iteration-level chain remains in
 `agentic_trace/<mode>/agentic_lineage.json`.

### Fixed — review sweep
- **Agentic trace collision between refine and independent modes**: both
 modes wrote to the same `agentic_trace/iter_*.json` files, silently
 overwriting each other. Each mode now writes to its own subdirectory
 (`agentic_trace/refine/`, `agentic_trace/independent/`).
- **`_trace_dir` restoration leak**: the original restore-in-`finally` only
 wrapped the independent mode; an uncaught exception in refine mode would
 leave the runner's `_trace_dir` mutated. The whole two-mode block is now
 wrapped in a single outer `try/finally` so the base path is always
 restored, even on uncaught exceptions.
- **Unbounded conversation history in the agentic loop**: the LLM received
 the full iteration history every turn. With up to 25 iterations and
 dense diagnostic markdown, this could quadratically bloat tokens and
 exceed the provider's context window. A sliding window (last 24
 messages ≈ 12 iterations) is now applied at message-construction time;
 full history is still captured in trace files.
- **Prompt-injection defense for relayed backend errors**: nlmixr2 or R
 error messages are now passed through `_sanitize_for_prompt()` before
 being embedded in LLM user turns. The helper strips code fences,
 collapses role markers, and truncates to 500 chars. Prevents a crafted
 R error from manipulating the LLM's behavior or audit trail.
- **Redaction gate now covers `summarize_for_llm`**: the allow-list filter
 was applied only to the dict stored in trace files, not the markdown text
 actually sent to the LLM. The allow-list is now enforced on every field
 reaching a third-party LLM (PRD §10 aggregate-only invariant).

### Fixed (code review sweep)
- **Orchestrator:** replaced `if "ranking" in dir()` sentinel with explicit
 `Ranking | None` initialization (C1). The prior pattern was fragile under
 refactors and masked scope-tracking errors.
- **Agentic runner:** a single `run_id` is now minted at the top of
 `AgenticRunner.run()` and reused across every `agentic_trace/*_input.json`
 and the run-level `RunLineage` artifact (C2). Previously each iteration
 received a freshly-generated candidate ID, breaking PRD §4.2.6 multi-run
 provenance.
- **Gate 3 cross-paradigm ranking:** orphan candidate IDs are now logged and
 skipped rather than raising `RuntimeError` mid-bundle-emission (C3).
- **Concurrency:** migrated three `asyncio.gather` sites (search engine,
 seed-stability runs, LORO-CV) to `asyncio.TaskGroup` for structured
 cancellation (H1, PEP 654).
- **Rich CLI formatters:** `isinstance(bic_val, float | int)` now correctly
 rejects booleans via `_is_real_number`; `str(True)` no longer renders as
 `"1"` in the LLM diagnostics summary (H2).
- **JAX platform:** the process-global `jax.config.update("jax_platform_name",
 "cpu")` is now routed through `configure_jax_platform()`, which warns on a
 conflicting request instead of silently drifting (H3).
- **Errors:** renamed `apmode.errors.TimeoutError` to `BackendTimeoutError`
 to stop shadowing `builtins.TimeoutError` at importers (M1).
- **Node subject records:** introduced `SubjectRecord` TypedDict; dropped
 `# type: ignore[dict-item]` on the event-driven subject dict (M2).
- **Profiler:** replaced the `[]`-sentinel + `isinstance(np.ndarray)` guard
 in lag detection with an explicit `(times > 0).any()` branch (M7).
- **Profiler thresholds:** the magic constants (0.7, 0.4, 1.8, 0.3, 0.5,
 0.15) are now named module-level constants with intent comments pointing
 toward a future `policies/profiler.json` artifact (M8).
- **Broad excepts:** narrowed agentic / seed / LORO catches from
 `except Exception` to `except BackendError` (plus `RuntimeError` where the
 agentic runner intentionally raises on exhaustion) (M10).
- **Ollama client:** the recorded `request_payload_hash` now covers the
 `options` dict (including `temperature`) so the hash matches the actual
 request (M11, PRD §4.2.6 model-version escrow).
- **CLI:** dropped the blanket `console.status(...)` spinner around the
 pipeline run so structlog stage messages are no longer suppressed (M14).

### Added (code review sweep)
- **DSL `IVBolus` absorption variant** (H7) — replaces the
 `FirstOrder(ka=100.0)` stiff-ODE hack previously used to approximate IV
 bolus dosing. Emitted explicitly by `nlmixr2_emitter` (no depot ODE;
 dose routes to the central compartment via CMT=1).
- **LLM diagnostics redaction layer** (H5) —
 `diagnostic_summarizer.redact_for_llm()` enforces an allow-list of
 aggregate fields before anything leaves the process to a third-party LLM
 provider (PRD §10, ARCHITECTURE.md §11). Unknown keys are dropped
 fail-closed.
- **Provider registry typing** (M6) — `_PROVIDER_REGISTRY` is now typed as
 `dict[str, Callable[[LLMConfig], LLMClientProtocol]]` and
 `register_provider` preserves concrete subclass types via a TypeVar.
- **CI security job** (H4) — new job runs `uv run bandit -r src/apmode/`
 and `uv run pip-audit` on every push/PR. `bandit` and `pip-audit` added
 to the `dev` extras group.
- **Test parallelism** (L6) — `pytest-xdist` added to the `test` extras;
 the CI `test` job now runs `pytest -n auto` without `-x`.
- **Python version matrix widened to 3.14** (M3) — `requires-python` is now
 `">=3.12,<3.15"` and the classifier block advertises 3.14. The CI matrix
 remains 3.12/3.13 until JAX/Diffrax publish 3.14 wheels.

### Changed (code review sweep)
- **Agentic backend default** (H6) — `apmode run --agentic` now defaults to
 `False`. The prior auto-on-when-API-key-present behavior would silently
 exfiltrate diagnostics as soon as an `ANTHROPIC_API_KEY` was in the shell
 env. Opt-in is explicit.
- **Ruff config** (M4) — removed the deprecated `UP038` entry from the
 ignore list.
- **Pre-existing pandas `# type: ignore[import-untyped]` pragmas** — removed
 throughout `data/` now that pandas is covered by the mypy overrides block.

### Added
- **Deep inspection CLI commands** (Phase 3 deep inspection, PRD §4.3.2)
 - `apmode trace <bundle>` — inspect agentic LLM iteration traces: summary table,
 per-iteration detail (`--iteration N`), token/cost aggregation (`--cost`),
 JSON export (`--json`)
 - `apmode lineage <bundle> <candidate_id>` — trace the DSL transform chain from
 root to any candidate, with gate status at each step; merges agentic lineage
 automatically; optional `--spec` for DSL snapshots, `--no-gate` to hide gates
 - `apmode graph <bundle>` — visualize the full search DAG as Rich tree (default),
 Graphviz DOT (`--format dot`), Mermaid (`--format mermaid`), or JSON
 (`--format json`); filters: `--converged`, `--backend`, `--depth`; file export
 via `--output`; cycle-safe rendering
 - `apmode inspect` now shows deep inspection hints when trace/graph/lineage
 artifacts are present in the bundle
 - New Pydantic models: `SearchGraphNode`, `SearchGraphEdge`, `SearchGraph`,
 `AgenticIterationEntry` in `bundle/models.py`
 - `BundleEmitter.write_search_graph()` for `search_graph.json` artifact
 - `SearchDAG.iter_nodes()` / `to_edges()` public API for graph building
 - `IterationRecord` now tracks `transforms_rejected` and `validation_feedback`
 for complete audit trail in `agentic_iterations.jsonl`
 - 25 new unit tests covering models, emitter, and all three CLI commands
- **Parallel model evaluation** (`--parallel-models N` / `-j N` CLI flag)
 - Concurrent candidate evaluation in SearchEngine via `asyncio.gather` + `Semaphore`
 - Parallelized seed-stability runs and LORO-CV evaluation in the orchestrator
 - Sequential fast-path when `max_concurrency=1` (default, zero overhead)
 - `RunConfig.max_concurrency` field with `__post_init__` clamping for API safety
 - Lazy semaphore creation to avoid event-loop binding issues
 - All shared-state mutations (DAG, trajectory writes, emitter) remain sequential post-gather
 - 6 new async unit tests covering semaphore bounds, result ordering, and error isolation
- **Comprehensive benchmark suite expansion** (Phase 3, PRD §5/§9)
 - **Benchmark infrastructure** (`benchmarks/models.py`): Pydantic v2 models for
 `DatasetCard`, `BenchmarkCase`, `BenchmarkScore`, `PerturbationRecipe`,
 `SuiteSummary`, `SuiteReport`; `SuiteName` Literal type alias; `AccessTier`
 enum (open/scripted/credentialed) for CI eligibility
 - **Suite A-External** (`suite_a_external.py`): Schoemaker 2019 / ACOP 2016 grid
 integration — 12 nlmixr2data datasets (bolus/infusion/oral x 1-cmt/2-cmt x
 linear/MM), weekly cadence with nightly and CI smoke subsets
 - **Suite B Extended** (`suite_b_extended.py`): 7 real-data benchmark cases:
 B4 theophylline NODE anchor (Bram 2023 continuity), B5 mavoglurant BLQ 25%/40%,
 B6 mavoglurant 5% outlier injection, B7 mavoglurant sparse absorption,
 B8 mavoglurant null covariate FPR (5 random covariates), B9 gentamicin IOV
 - **Suite C** (`suite_c.py`): Expert head-to-head framework — mavoglurant,
 gentamicin IOV (DDMoRe CC0), Eleveld propofol (OpenTCI); 5-fold subject-level
 CV, NPE primary metric, 2% win margin, pre-registered evaluation protocol;
 published models as reference anchors, blinded expert panel as primary baseline
 - **7 perturbation types** (`perturbations.py`): inject_blq (M3-style CENS/LLOQ),
 remove_absorption_samples, add_null_covariates, inject_outliers (additive
 fallback for near-zero DV), sparsify, add_protocol_pooling (STUDY_ID +
 optional sampling/LLOQ variation per protocol), add_occasion_labels (vectorized)
 - **Scoring harness** (`scoring.py`): structure recovery (DSLSpec-wired), parameter
 bias (NaN for missing estimates), parameter coverage (None for missing CI per
 GPT-5.2-pro: unscorable, not free pass), NPE, prediction interval calibration,
 expert comparison (empty-list guard), dispatch assertion checking, convergence
 taxonomy; `SuiteName`-typed `aggregate_suite()`
 - **Dataset cards and prepare scripts** for 7 public PK datasets:
 nlmixr2data Schoemaker grid (prepare.R), theophylline (prepare.R),
 warfarin PK/PD (prepare.R), mavoglurant (prepare.R), DDMoRe gentamicin IOV
 (prepare.py, CC0), Eleveld propofol (prepare.R, OpenTCI), MIMIC-IV vancomycin
 (README + access instructions, Tier 2 credentialed)
 - **Dataset registry** (`benchmarks/datasets/registry.yaml`): master index of all
 datasets with access tiers, suite assignments, and prepare script references
 - 77 new integration tests covering all three suites, all 7 perturbation types,
 scoring harness, expert comparison methodology, recipe validation, and
 protocol pooling; total test count: 1232 passing
- **PerturbationRecipe cross-field validators**: Pydantic model_validator enforces
 required fields per perturbation type (blq_fraction for BLQ, outlier_fraction
 for outliers, target_obs_per_subject for sparsify, etc.); Field constraints
 (blq_fraction in [0,1], outlier_magnitude > 1, target_obs_per_subject >= 1)
- **SuiteSummary n_passed <= n_cases constraint**: model_validator prevents
 impossible summary states
- `__all__` exports on all benchmark modules for explicit public API

### Changed
- **BLQ perturbation uses M3-style censoring**: DV set to LLOQ (not 0.0) with
 CENS=1 column per nlmixr2/NONMEM M3 convention; DV=0 was ambiguous and biased
 M3 likelihood (GPT-5.2-pro + Gemini 3.1 Pro consensus)
- **Observation identification uses (EVID==0) & (MDV==0)**: all perturbation
 functions now exclude EVID=0/MDV=1 rows (missing samples) per NONMEM convention;
 prevents corrupting missingness semantics in M3/M4 handling
- **score_parameter_coverage returns None (not True) for missing CI**: SAEM
 covariance failures are unscorable, not free passes; aggregators decide policy
- **score_structure_recovery wired to DSLSpec**: compares discovered spec's
 absorption/distribution/elimination types against ExpectedStructure; returns
 None when no spec available (unscorable) instead of always True
- **score_parameter_bias flags missing estimates with NaN**: reference params
 absent from BackendResult get bias=NaN, not silent skip
- **Occasion labeling vectorized**: replaced O(n*m) row loop with groupby+cumsum;
 sorts by (NMID, TIME, EVID) before labeling to handle unsorted input
- **PublishedModel defined before DatasetCard**: eliminates forward-reference
 risk in Pydantic v2 schema generation
- **aggregate_suite accepts SuiteName Literal** (was str): removes noqa/type-ignore
- Added `pandas.*` to mypy ignore_missing_imports (no stubs installed)

### Fixed
- Mock convergence test: `_mock_result(converged=False)` now sets
 `minimization_status="terminated"` (was "successful" — tautological assertion)
- Outlier injection handles near-zero DV: uses additive offset instead of
 multiplicative 0*magnitude=0

### Reviewed by
- multi-reviewer code review
- GPT-5.2-pro and Gemini 3.1 Pro consultation on PK-domain fixes
- All 15 identified issues addressed

### Added
- **Agentic LLM backend wired to CLI and orchestrator** (Phase 3, PRD §4.2.6)
 - `--agentic/--no-agentic` flag on `apmode run` (default: on)
 - `--provider` (anthropic/openai/gemini/ollama/openrouter), `--model`, `--max-iterations`
 - Auto-activates in discovery/optimization lanes when API key found; gracefully
 disables with warning when unavailable
 - Submission lane hard-blocks agentic (PRD §3 rule enforced in lane router)
 - **Post-search agentic stage** in orchestrator with two modes:
 - **Refine mode**: takes best classical candidate, LLM proposes targeted transforms
 - **Independent mode**: starts from minimal 1-cmt oral spec, LLM builds from scratch
 - Both modes feed results back into gate funnel alongside classical candidates
 - In discovery/optimization lanes, LLM can also propose NODE transforms
- **Multi-turn conversation history in agentic loop**: LLM now sees full history
 across iterations (prior diagnostics, its own proposals, validation outcomes)
 instead of stateless per-iteration prompts
- **Validation feedback to LLM**: transform validation failures, parse errors, and
 DSL semantic violations are fed back as structured user messages so the LLM can
 correct rejected proposals (was silently swallowed with `continue`)
- **Live LLM provider integration tests** (`test_llm_providers_live.py`):
 8 tests across Anthropic, OpenAI, OpenRouter, Ollama, and full agentic loop;
 `@_skip_on_billing` decorator for graceful skip on quota/auth errors;
 `live` pytest marker (`-m live` to run, excluded from default suite)
- **`llm` optional dependency group** in pyproject.toml: `uv sync --extra llm`
 installs anthropic, openai, google-genai, ollama, litellm

### Fixed
- **R harness compatibility with nlmixr2 5.0**: fit$time (data.frame not vector),
 fit$conditionNumberCov (renamed from conditionNumber), fit$shrink (data.frame
 replacing fit$shrinkage), CWRES fallback to IWRES (SAEM doesn't compute CWRES
 by default), OFV/AIC/BIC via Gaussian quadrature, empty-list-to-dict JSON
 serialization for profile_likelihood_ci and diagnostic_plots
- **Gemini client multi-turn**: replaced string concatenation with proper
 `types.Content` objects preserving multi-turn conversation structure
- **OpenRouter auth**: client now reads `OPENROUTER_API_KEY` directly instead
 of requiring users to alias it as `OPENAI_API_KEY`

### Added
- **Full multi-dose support across all backends (ADDL/II/SS)**
 - **Data schema**: Added optional ADDL (additional doses), II (inter-dose interval),
 SS (steady-state flag) columns to CanonicalPKSchema with cross-column validation
 (ADDL>0 requires II>0, SS in {1,2} requires dose EVID, ADDL only on dose rows)
 - **Data manifest**: Added `has_multidose` and `has_steady_state` flags to DataManifest
 for downstream backend dispatch decisions
 - **Dose expansion** (`apmode.data.dosing`): New module with `expand_addl()` to
 materialize ADDL/II into explicit dose rows, `expand_infusion_events()` for
 infusion stop event generation, `build_event_table()` for complete event table
 construction with deterministic EVID-priority sorting
 - **nlmixr2 backend**: Pass-through support — rxode2 handles ADDL/II/SS natively;
 columns are passed through in the data without code changes
 - **Stan emitter**: Event-based piecewise ODE integration with merged dose+observation
 timeline (chronological single-pass loop). Analytical solutions use superposition
 principle with reset-awareness (EVID=3/4 invalidate prior dose contributions)
 - **NODE/JAX backend**: Piecewise eager integration via `_solve_multidose_eager()` with
 merged dose+observation chronological timeline. Single-dose-at-t=0 uses legacy JIT
 path. Multi-dose/delayed-dose/reset subjects use event-driven Python control flow.
 Infusion data explicitly rejected with clear error until Phase 3
 - **Tests**: 19 new unit tests for dosing module covering expand_addl, infusion events,
 event table construction, subject extraction, and schema validation

### Fixed
- **NODE backend: 14 fixes from code review**
 - **[Critical]** 2-cmt y0 dimension: runner now builds 3-element state vector for
 2-compartment models (was always 2-element, causing shape crash)
 - **[Critical]** BIC/AIC parameter count: uses actual trainable param count from model
 tree (~25 for typical NODE) instead of `len(param_estimates)` (~5), which severely
 underpenalized NODE in Gate 3 cross-paradigm ranking
 - **[High]** NLL missing log(2π): added `0.5*n*log(2π)` constant for cross-backend
 comparability with nlmixr2 (was ~1.84/obs offset)
 - **[High]** Monotone constraint semantics: renamed/documented that MonotoneIncreasing
 guarantees positive output, not input-output monotonicity; MonotoneDecreasing now
 returns positive bounded output via `1/(1+sum(softplus))` (was returning negative)
 - **[High]** Multi-dose rejection: NODE runner raises `InvalidSpecError` for subjects
 with >1 dose event (was silently using only the first dose)
 - **[High]** ODE integration start: `solve()` now integrates from `t0=0.0` (dose time)
 instead of `times[0]` (first observation), preventing time-shift errors
 - **[High]** Saturable constraint: MM-like `Vmax*s/(Km+s)` form with learned log-space
 parameters replaces bounded sigmoid (which didn't model rate saturation)
 - **[High]** Convergence semantics: patience exhaustion no longer unconditionally marks
 `converged=True`; returns granular `minimization_status` (successful/plateau/nan/max)
 - **[High]** V/V2 volume scaling: trainer uses V2 for peripheral compartment observations
 instead of always dividing by V
 - **[Medium]** WeightLibrary thread safety: added `threading.Lock` for concurrent async runs
 - **[Medium]** Time sorting: CSV loader sorts observations by time before Diffrax solve
 - **[Low]** Numerical stability: replaced `log1p(exp(x))` with `jax.nn.softplus(x)`
 - **[Low]** Documented GMR direction in `FidelityResult` (surrogate=test, NODE=reference)

### Added
- Real-data NODE integration tests (12 tests):
 - theo_sd (12 subjects, 1-cmt): CSV load, train convergence, distillation, full E2E runner
 - Oral_2CPT: multi-dose rejection guard verified
 - 2-cmt mock mode: correct y0 shape, training without crash
 - All 5 constraint templates trained on real data without NaN
- GPU-not-required note in README prerequisites
- **Phase 3 P3.B: LORO-CV for Optimization lane** (PRD §3.3)
 - `LOROFoldResult`, `LOROMetrics`, `LOROCVResult` Pydantic models in bundle/models.py
 - `loro_cv_splits()` fold generation with regimen-signature grouping (modal dose
 per subject, not total AMT)
 - Gate 2 `_check_loro_requirement` now evaluates real LORO thresholds:
 pooled NPDE mean/variance, VPC coverage concordance, minimum fold count
 - LORO policy fields in `Gate2Config`: `loro_npde_mean_max`, `loro_npde_variance_min/max`,
 `loro_vpc_coverage_min`, `loro_min_folds`, `loro_budget_top_n`
 - `evaluate_loro_cv()` execution engine with per-fold fitting, metric aggregation,
 law of total variance for pooled variance (E[Var] + Var[E])
 - `write_loro_cv_result()` emitter writing to `loro_cv/{candidate_id}.json`
 - New `src/apmode/evaluation/` package for cross-validation and predictive performance
 - Orchestrator wiring: LORO-CV runs between Gate 1 and Gate 2 for optimization lane,
 budget-capped by `loro_budget_top_n`, regimen labels preserved for auditability
 - Evidence-grounded design: Khandokar (2025 Uppsala) CV for PK model selection,
 Comets et al. (2008) NPDE, Bergstrand et al. (2011) pcVPC
 - Multi-model consensus reviewed (MIMO-v2-Pro, GPT-5.2-Pro):
 - Fixed variance proxy formula (use cwres_sd² consistently, not outlier_fraction)
 - VPC missing-evidence → fail-closed (0.0, not convergence fraction fallback)
 - Gate 2 NaN safety: explicit `np.isfinite` checks prevent silent pass
 - Gate 2.5 no longer overwrites Gate 2 artifact (gate_number=25)
 - `_find_spec_for_candidate` replaced with typed `_spec_map` dict
 - 39 new tests: fold generation (13), Gate 2 LORO (14), execution engine (7),
 Hypothesis property tests (5)
- `TimeVaryingElim.kdecay` as first-class AST field (was phantom parameter
 with hardcoded initial estimate); grammar accepts optional `kdecay=` syntax
- BLQ grammar extended: `BLQ_M3`/`BLQ_M4` now accept optional `error_model`,
 `sigma_prop`, `sigma_add` parameters in DSL text (not just programmatically)
- Semantic validation: TMDD distribution requires LinearElim (prevents
 undefined `CL` in emitted code)
- Semantic validation: duplicate IIV parameters across blocks rejected
- Semantic validation: duplicate CovariateLinks (same param+covariate) rejected
- Semantic validation: Transit `n` cannot have IIV/IOV/CovariateLink (emitters
 don't apply eta to its back-transform)
- IOV pruning after module swap transforms (was only IIV + CovariateLink)
- Stan emitter: TMDD_QSS prediction now uses Cfree (algebraic QSS solve),
 not Ctot (was scientific correctness bug)
- Stan emitter: TMDD Core/QSS receptor ODE uses explicit kdeg derivation,
 consistent with nlmixr2 emitter
- Stan emitter guardrails: ZeroOrder/MixedFirstZero absorption and IOV
 raise NotImplementedError instead of emitting broken code
- Lark parser cached via `lru_cache` (was re-instantiated per parse call)
- Formular docs: semantic validation rules table (docs/FORMULAR.md)

### Previously added
- `apmode explore` interactive wizard: step-by-step dataset exploration
 (fetch -> ingest -> profile -> NCA -> search space -> optional pipeline launch)
 with `--non-interactive` flag for CI
- `apmode datasets` command: browse and download 14 public PK datasets from
 nlmixr2data (5 real clinical, 9 simulated ground-truth)
- `apmode log` command: query gate decisions (`--gate`), failed candidates
 (`--failed`), and top-N parameter estimates (`--top N`) from bundles
- `apmode diff` command: side-by-side comparison of evidence manifests,
 rankings, and gate pass rates between two bundles
- NODE initial estimate strategy (`node_init.py`): pre-trained weight library
 for 1-cmt/2-cmt reference dynamics + transfer learning from classical fits
- Stan codegen emitter (`stan_emitter.py`): DSL -> complete Stan program with
 ODE/analytical solvers, IIV, IOV, BLQ M3/M4, covariates, LOO-CV log_lik
- Per-backend lowering test suite: validates nlmixr2/Stan structural parity
- Suite A CSV fixtures A5-A7 generated via rxode2 (TMDD QSS, covariates,
 NODE saturable absorption)
- Independent `log_CL` field in `HybridPKODE` for proper absorption-mode
 elimination parameterization
- `WeightLibrary.reset()` and `reset_weight_library()` for test isolation
- Dataset local cache: skip re-fetch if CSV already exists
- Exit code documentation (0/1/2/130) in CLI help

### Changed
- VPC concordance target now configurable via `GatePolicy.vpc_concordance_target`
 (was hardcoded 0.90) and threaded through `rank_cross_paradigm()`
- Diffrax solver uses `dt0=None` for automatic step-size selection (was 0.1)
- Distillation Michaelis-Menten surrogate fit uses `scipy.optimize.curve_fit`
 (was Lineweaver-Burk linearization)
- BIC in `NodeBackendRunner` uses total observation count (was subject count)
- `np.polyfit` in distillation called once instead of twice
- Data profiler detects TMDD inverse curvature (ratio < 0.3) in addition to
 MM curvature (ratio > 1.8)

### Fixed
- NODE absorption mode no longer uses `ka` as CL proxy — `CL/V` is used for
 elimination, with CL as an independently trainable parameter
- Stan emitter: `maturation` covariate form now raises `NotImplementedError`
 instead of silently dropping the effect
- Profiler: `corr_matrix.values` read-only array error on pandas 2.x fixed
 via `to_numpy(copy=True)`
- R simulation script: rxode2 5.x compatibility (id column for iCov, EVID
 101->1 normalization, duplicate RENAL column merge)
- CLI: `rich.markup.escape()` for user-supplied paths, `--verbose`/`--quiet`
 mutual exclusion, `dataset.is_file()` check, `timeout min=1`

## [0.1.0] - 2026-04-13

### Added
- Phase 1 complete: DSL compiler + Lark grammar, nlmixr2 emitter, semantic
 validator, classical NLME backend (nlmixr2 R subprocess), data ingestion
 (NONMEM CSV + Pandera), data profiler (evidence manifest), NCA estimator,
 data splitter (k-fold, LORO-CV), automated search engine with Pareto
 frontier, governance gates (1/2/2.5/3), lane router, orchestrator,
 reproducibility bundle emitter, Typer CLI, structlog logging
- Phase 2 in progress: JAX/Diffrax/Equinox NODE backend (6 modules),
 Gate 2.5 ICH M15 credibility, Gate 3 cross-paradigm ranking,
 SearchEngine multi-backend dispatch, Discovery lane integration,
 Benchmark Suites A (7 scenarios) + B (3 NODE validation)
- 679 tests (Phase 1) + 270 new tests (Phase 2) = 949 total
- mypy --strict clean, ruff clean
- Seven rounds of code review

[Unreleased]: https://github.com/biostochastics/APMODE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/biostochastics/APMODE/releases/tag/v0.1.0
