# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-rc3] — 2026-04-15

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
