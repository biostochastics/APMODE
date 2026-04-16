# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-rc4] — 2026-04-16

### Gate 2 fix: shrinkage unit mismatch

- **`_check_shrinkage` now compares on the correct scale.**
  `harness.R` was emitting SD shrinkage as a **percentage** (0–100; e.g., 2.91)
  while `Gate2Config.shrinkage_max = 0.3` encodes the threshold as a
  **fraction** (0–1; 30%).  Every submission-lane candidate with low
  shrinkage (2–5 %) was therefore failing Gate 2 despite being well within
  the 30 % policy limit.  The R harness now outputs fractions: `1 -
  sqrt(pmax(0, 1 - var_shrinkage))` without the trailing `× 100`.  The
  policy value is unchanged.

## [0.3.0-rc3] — 2026-04-15

### Gate 1 semantic fixes

- **`parameter_plausibility` now back-transforms log-space thetas.**
  nlmixr2 parameterises structural PK thetas in log-space (`lCL`, `lV`,
  `lka`, `lke`, `lktr`, `lQ`, `lVm`, `lKm`, `ln`). The previous check
  compared the raw estimate to zero and flagged every negative log-
  theta as "non-positive", disqualifying ~100 % of converged candidates
  across every benchmark dataset. The check now detects log-space by
  the `l` + alpha-char prefix convention and applies the sanity bounds
  (positivity, `1e-4`–`1e5`) to `exp(estimate)`.
- **`seed_stability` no longer disqualifies candidates the orchestrator
  chose not to probe.** The orchestrator runs seed replicates only for
  the top-K candidates by BIC; Gate 1 previously failed the remaining
  candidates with "insufficient_seeds", eliminating otherwise-healthy
  fits on the basis of an orchestrator optimisation. The check now
  reports `not_probed` and passes when no replicates were supplied;
  top-K candidates that actually have replicates still face the real
  OFV-CV comparison.



First release in which `apmode run` drives nlmixr2 end-to-end on every
shipped benchmark dataset (Suite A fixtures + warfarin / theo_sd /
mavoglurant real data).

### Highlights

- **Structured signal provenance on `EvidenceManifest`.** Nonlinear-
  clearance evidence is emitted as
  `dict[SignalId, NonlinearClearanceSignal]`, one record per signal
  with algorithm, citation, policy-key JSON pointer, threshold,
  observed value + 90% CI, eligibility reason, vote, and typed extras.
  `manifest_schema_version = 3`.
- **nlmixr2 adapter fixed.** `to_nlmixr2_format` now filters non-PK
  observation rows (DVID allowlist — drops e.g. warfarin's `pca` PD
  rows), collapses single-endpoint `DVID`, and auto-remaps binary
  string covariates (e.g. `SEX="male"/"female" → 1/0`) before handing
  the CSV to the R subprocess. Without this, SAEM rejected every
  Suite A / real-data fit with `mis-match in nbr endpoints` or an
  empty-rows ID replacement.
- **Profiler rewrite.** Huang 2025 λz selector (nlmixr2autoinit) with
  Phoenix-style window guard, Smith 2000 dose-proportionality power
  model with aggregated per-dose-level geometric means, population-
  bootstrap 90% CI on curvature ratio, steady-state gating before
  dose–AUC fits, BLQ-aware Tmax / slope geometry, Wagner-Nelson ka
  seed, Richardson 2025 flip-flop flag, TAD-contamination detection,
  Pharmpy-style NODE design-feasibility budget.
- **Versioned policy artifact.** Every profiler threshold lives in
  `policies/profiler.json` (loaded via `ProfilerPolicy`) and the
  `policy_sha256` is embedded in the manifest.

### Known limits

- NODE / agentic LLM Discovery-lane backends remain behind lane-
  router gates; Suite A is exercised classically by nlmixr2.
- Full benchmark output is written to
  `benchmarks/runs/full-<timestamp>/`; a summary is included in the
  release notes of this tag.
