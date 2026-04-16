# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-rc5] — 2026-04-16

### Gate 3 cross-paradigm ranking: policy-driven composite, Borda, NPE unification

Lands Tier 1 + Tier 2 of the ranking cleanup flagged in `v0.3.0-rc5`'s
profiler-refinement commit. Multi-model review via xen clink (droid,
crush, gemini, codex, opencode) informed the hardening pass.

**Tier 1 — plumbing (no behavior change on default paths):**

- `governance/ranking.py` — `compute_npe` renamed to
  `compute_cwres_npe_proxy`. The name is now honest: this is a CWRES
  proxy, not the simulation-based NPE that
  `apmode/benchmarks/scoring.compute_npe` computes for Suite C.
- `governance/ranking.py` — `rank_cross_paradigm` now **requires**
  `gate3: Gate3Config` and `vpc_concordance_target: float` as keyword
  arguments. No defaults, no hidden magic. `compute_vpc_concordance`
  loses its 0.90 default for the same reason.
- `governance/gates.py` — the previously dead
  `GatePolicy.vpc_concordance_target` is now threaded into the
  `rank_cross_paradigm` call site (`gates.py:865`).
- `routing.py` — the unversioned `0.20` BLQ advisory fallback has been
  removed. The live orchestrator always passes a `MissingDataPolicy`;
  legacy callers that don't now get no BLQ advisory rather than a
  number that never matched any lane policy (submission 0.05,
  optimization 0.10, discovery 0.15).
- `bundle/models.py` — `DiagnosticBundle.npe_score: float | None =
  Field(default=None, ge=0.0)` is the canonical simulation-based NPE
  slot. Backends populate it via
  `apmode.benchmarks.scoring.compute_npe`. Ranking prefers this over
  the CWRES proxy and records which one was used via
  `CrossParadigmMetrics.npe_source: Literal["simulation",
  "cwres_proxy"]`.

**Tier 2 — policy-driven composite (opt-in behavior change):**

- `governance/policy.py` — new `Gate3Config`:
  - `composite_method: Literal["weighted_sum", "borda"] = "weighted_sum"`
  - `vpc_weight: float = 0.5`, `npe_weight: float = 0.5`,
    `bic_weight: float = 0.0` — **PRD §10 Q2 default: cross-paradigm
    BIC off** (likelihood incomparability across observation models).
    Within-paradigm ranking uses pure BIC and is unchanged. Legacy
    0.4/0.4/0.2 split is reachable by explicit override.
  - `npe_cap: float = 5.0`, `bic_norm_scale: float = 1000.0` — the
    previously hard-coded weighted-sum normalization constants, now
    policy fields.
  - Invariant: weights sum to 1.0 within 1e-6.
- `governance/ranking.py` — added Borda-count aggregation. Each
  enabled metric (weight > 0) ranks candidates via average-rank on
  ties; composite is the rank sum (lower wins). Scale-invariant;
  preferred by multi-model consensus for cross-paradigm Gate 3. The
  weighted-sum path is retained as the default for backward-
  compatibility-of-semantics with v0.3.0-rc4 benchmarks.
- Non-finite metric inputs (NaN / ±Inf) are now coerced to the worst
  value for their ordering direction before ranking (gemini review
  catch): VPC → -Inf, NPE/BIC → +Inf. Python's unstable NaN sort can
  no longer land a broken candidate at rank 1.

**Test coverage (`tests/unit/test_cross_paradigm_ranking.py`):**

- Borda correctness: rank-sum under explicit ordering, tie averaging,
  scale-invariance under pathological NPE (multiplying CWRES by 1000
  does not change rank).
- NPE unification: `diagnostics.npe_score` always overrides the
  CWRES proxy when finite and non-negative; `None` or NaN falls back
  with `npe_source == "cwres_proxy"`.
- `Gate3Config` validators: sum-to-1, all-zero rejection, round-trip.
- Zero-BIC scenarios under both weighted-sum and Borda — identical
  metrics with wildly different BIC produce identical composite
  scores when `bic_weight = 0`.
- NaN sanitization: a NaN CWRES cannot beat a clean candidate under
  Borda; missing VPC cannot beat a valid candidate under
  weighted-sum.

**Test updates (`tests/unit/test_lane_router.py`):**

- BLQ advisory assertions rewritten to exercise the directive-driven
  string (`"BLQ method M3 selected"`) instead of the removed legacy
  `"M3/M4 required"` string. New test asserts that a policy-less
  call path emits no BLQ advisory.

**Deferred:**

- True AUC/Cmax bioequivalence scoring (Smith 2000 GMR vs NCA
  reference) remains unwired. The infrastructure is ready
  (`compute_npe` exists in `benchmarks/scoring.py`); once backends
  emit posterior-predictive simulations alongside the VPC summary,
  a GMR metric can plug into the same composite pipeline without
  API churn.
- Lane-specific `gate3` policy blocks (submission.json /
  discovery.json / optimization.json) are not yet populated. Every
  lane currently falls back to the `Gate3Config` default (weighted
  sum, BIC off). Adding explicit opt-ins (e.g., Discovery lane opts
  into Borda) is a separate PR once benchmark baselines quantify
  the behavioral delta.

### Profiler policy: drift remediation, dead-policy wiring, schema v2.1.0

`policies/profiler.json` bumped from `v2.0.0` → `v2.1.0`. All changes
preserve default behaviour (thresholds equal prior hard-coded values);
the point of the bump is that deployments tuning the JSON now actually
see the change propagate. Multi-model consensus (gpt-5.2-pro, glm-5.1)
informed the approach; consensus notes are in the PR description.

#### Drift eliminated (bare literals → policy-sourced constants)

Eight heuristic call sites in `src/apmode/data/profiler.py` previously
used bare numeric literals that mirrored (but did not source from)
policy fields. Editing the JSON had no effect. All now go through
module-level `_POLICY.*` constants:

| Function (file:line) | Before | Now sourced from |
|---|---|---|
| `recommend_error_model` (profiler.py:670) | `dv_cv_percent < 30.0` | `low_cv_additive_ceiling` |
| `_compute_node_dim_budget` (profiler.py:400) | `4 / 8 / 10 / 20` literals | `node_{discovery,optimization}_{min_subjects,min_median_samples,budget}` |
| `_classify_richness` | `< 4`, `<= 8` | `min_obs_per_subject_{moderate,rich}` |
| `_assess_identifiability` | `4 / 8 / 10 / 20` literals | `min_obs_per_subject_*` + `node_*_min_subjects` |
| `_assess_absorption_coverage` | `>= 2.0` | `absorption_coverage_min_pre_tmax` |
| `_assess_elimination_coverage` | `>= 3.0` | `elimination_coverage_min_post_tmax` |
| `_assess_flip_flop_risk` | `1.5 *`, `>= 0.85`, `>= 4` | `flip_flop.{ka_lambdaz_ratio_possible,quality_adj_r2_min,quality_min_npts}` |
| `_assess_protocol_heterogeneity` | `> 0.5` CV | `protocol_heterogeneity.obs_per_subject_cv_threshold` |

#### Dead policy field wired

`huang_2025_lambda_z.adj_r2_threshold = 0.7` was loaded into
`ProfilerPolicy.lambdaz_adj_r2_threshold` but never consumed. Now
used as the **advisory quality floor** for flip-flop detection:
below this, the profiler returns `"unknown"` rather than
`"possible"` / `"likely"`. Distinct from the stricter
`flip_flop.quality_adj_r2_min = 0.85` (Richardson 2025) required to
retain `"likely"`.

#### New policy fields (backward-compatible — defaults match prior hard-coded behaviour)

- `flip_flop.quality_adj_r2_min` (0.85) — Richardson 2025 strict
  terminal-fit adj-R² for flip-flop classification.
- `flip_flop.quality_min_npts` (4) — companion min terminal-window
  size for the strict-quality gate.
- `protocol_heterogeneity.obs_per_subject_cv_threshold` (0.5) —
  across-study CV of observations/subject above which a pooled study
  is classified `pooled-heterogeneous`.

#### Advisory vs disqualifying taxonomy (documented)

Profiler fields are **advisory** — they populate the
`EvidenceManifest` and inform Lane Router / Gate 3 ranking but do not
themselves disqualify candidates. **Disqualification** is driven by
`GatePolicy` (Gate 1/2/2.5 evaluators in `governance/gates.py`). The
distinction is now called out in
`docs/PROFILER_REFINEMENT_PLAN.md`, the module docstring of
`apmode/data/profiler.py`, and the README's parameter dictionary.

#### Drift guard

Added `tests/unit/test_profiler_policy_consistency.py` (34 tests):

1. Every active JSON leaf in `profiler.json` must map to a
   `ProfilerPolicy` field.
2. Every `_POLICY.*` module-level constant must equal its JSON source.
3. AST-level scan of eight drift-prone heuristic functions flags
   bare numeric literals (allowlist covers loop bounds, indices,
   numerical-stability floors, and display fraction→percent
   conversions).
4. `policy_id` and `policy_version` must agree.
5. `docs/PROFILER_REFINEMENT_PLAN.md` exists (referenced from the
   JSON description).

#### Documentation

- **New**: `docs/PROFILER_REFINEMENT_PLAN.md` — derivation and
  primary-source citation for every profiler policy field.
- **README**: `Data Profiler Parameter Dictionary` updated to
  v2.1.0; missing error-model / subject-quality rows added
  (`high_cv_ceiling`, `narrow_range_additive`,
  `low_cv_additive_ceiling`, `min_subjects_for_dynamic_range`,
  `min_obs_per_subject_moderate`, `min_obs_per_subject_rich`).
- **README**: new **Gate policy parameters** table covering
  Gate 1 / Gate 2 / Gate 2.5 / missing-data defaults across all
  three lanes, with primary-source citations.

#### Deferred (needs baseline regression sign-off)

Gate 3 cross-paradigm ranking cleanup (zeroing BIC weight, Borda
rank aggregation, AUC/Cmax bioequivalence scoring, wiring dead
`GatePolicy.vpc_concordance_target`, removing magic-number caps) is
out of scope for this commit per consensus — scheduled as a
separate `gate3-composite-policy` PR once benchmark baselines are
refreshed.

### CLI: comprehensive overhaul (pharmacometrician UX)

All changes to `src/apmode/cli.py`.

#### New commands

- **`apmode doctor`** — R/nlmixr2/rxode2/CmdStan/Python package and LLM
  provider health check.  Probes R via subprocess, checks `packageVersion()`
  for nlmixr2 and rxode2, queries API key environment variables for all
  supported providers, and pings Ollama at `localhost:11434`.
- **`apmode ls [RUNS_DIR]`** — List run bundles with a summary table (lane,
  candidate count, best BIC, best candidate ID).  Supports `--sort` (time,
  lane, bic, status) and `--limit`.
- **`apmode policies [LANE]`** — List and inspect gate policy files from the
  `policies/` directory.  Shows policy version, gate coverage, and validation
  status.  With a single lane argument, expands the full threshold table.
  `--validate` checks required keys.
- **`apmode report`** upgraded — Now checks for existing `report.md` /
  `report.html` artifacts and displays them (HTML: `webbrowser.open`;
  MD: `console.pager`).  `--no-browser` prints path instead.  Shows the
  Phase 3 stub only when no artifacts exist.

#### Existing command improvements

- **`apmode run --dry-run`** — Profiles data, runs NCA, enumerates the
  search space, and prints a dispatch-plan panel (root candidates, nlmixr2 /
  JAX-NODE split, compartments, absorption types, elimination types, lane)
  without executing any R backends.
- **`apmode run --yes/-y`** — Bypass the agentic data-sharing confirmation
  prompt (useful for CI).
- **Agentic submission-lane note** — `--agentic` on the submission lane now
  prints a one-liner explaining why it is disabled (PRD §3) instead of
  silently ignoring the flag.
- **Policy version in Data Summary panel** — Reads the lane's policy file
  before the run starts and shows `policy_version` so the user can verify
  they are running the expected thresholds.
- **Top-Model panel at run completion** — Loads `ranking.json` and the
  winner's `results/{id}_result.json` to display OFV, BIC, AIC, and
  η-shrinkage immediately after the run.
- **`apmode inspect` ranking table** — Added OFV and η-shrinkage columns
  (loaded from `results/{id}_result.json` per candidate).  Fixed Gate 3
  display (winner, metric, ΔBIC vs runner-up, summary_reason).
- **`apmode log --failed`** — Fixed field names to match `FailedCandidate`
  model (`candidate_id`, `gate_failed`, `summary_reason`, `failed_checks`).
- **`apmode log --top`** — Added SE/RSE%/CI95 and η-shrinkage rows per
  candidate; OFV in subtitle.
- **`apmode diff`** — Fixed `candidate_id` key (with `model_id` fallback);
  added BIC_A/BIC_B comparison columns.
- **`_show_gate_details`** — Now handles `list[GateCheckResult]` (not just
  dict), shows observed/threshold values, adds policy_version and
  summary_reason rows, raised column widths to 100/120.

#### Colorblind accessibility

All status indicators now pair color with a text symbol — ✓/✗/⚠/~ — so
they are unambiguous without relying on hue alone.  Neutral-good values use
cyan instead of green; ✗/✓/~ replace pure-color-only statuses throughout
`_validate_file`, `_pass_fraction`, `_node_label`, trace/lineage views,
`_bool_badge`, and all new panels.

---

## [0.3.0-rc4] — 2026-04-16

### Benchmark: 10/10 scenarios recommended (Suite A + real datasets)

Full run of `scripts/run_full_benchmark.sh` (all Suite A fixtures +
warfarin / theo_sd / mavoglurant) with Gate 1 + Gate 2 fixes in place.

| Dataset | Lane | Gate 1 | Gate 2 | Rec | Best BIC |
|---|---|---|---|---|---|
| a1_1cmt_oral_linear | submission | 19/34 | 19/19 | ✓ | 2848.3 |
| a2_2cmt_iv_parallel_mm | discovery | 8/41 | 8/8 | ✓ | 4396.7 |
| a3_transit_1cmt_linear | submission | 9/34 | 9/9 | ✓ | 2623.2 |
| a4_1cmt_oral_mm | discovery | 4/41 | 4/4 | ✓ | 3113.3 |
| a5_tmdd_qss | discovery | 7/34 | 7/7 | ✓ | 4858.8 |
| a6_1cmt_covariates | submission | 17/34 | 17/17 | ✓ | −317.5 |
| a7_2cmt_node_absorption | discovery | 22/34 | 22/22 | ✓ | 807.4 |
| warfarin | submission | 19/33 | 19/19 | ✓ | 960.1 |
| theo_sd | submission | 10/25 | 8/10 | ✓ | 396.3 |
| mavoglurant | discovery | 4/37 | 4/4 | ✓ | 28906.9 |

Total: 119/347 (34%) Gate 1 survivors → 117/119 (98%) Gate 2 →
**10/10 scenarios with ≥1 recommended candidate.**

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
