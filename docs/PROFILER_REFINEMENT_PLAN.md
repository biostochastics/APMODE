# Profiler Refinement Plan

This document is the derivation/citation companion to
[`policies/profiler.json`](../policies/profiler.json). Every threshold in the
JSON has an entry here with its primary source, a brief rationale, and a
pointer to the consuming code path.

The JSON is the source of truth. The Python `ProfilerPolicy` dataclass
(`apmode/data/policy.py`) is a typed view of the JSON, and module-level
`_POLICY.*` constants in `apmode/data/profiler.py` are derived from the
dataclass. Drift between these three layers is guarded by
`tests/unit/test_profiler_policy_consistency.py`.

## Policy versioning

| policy_id / policy_version | Change summary |
|---|---|
| `profiler/v2.0.0` | Initial externalization of thresholds; all heuristic constants moved from profiler code into JSON. |
| `profiler/v2.1.0` | Drift remediation: wired dead `lambdaz_adj_r2_threshold` as advisory floor for flip-flop detection; added `flip_flop.quality_adj_r2_min`, `flip_flop.quality_min_npts`, and `protocol_heterogeneity.obs_per_subject_cv_threshold`. Documented every field against primary sources. |

## Policy sections

### `covariate`

- **`correlation_threshold_abs_r = 0.7`** — Pairwise covariate correlation
  above this magnitude flags multicollinearity risk during search.
  Standard heuristic in regulatory pharmacometric workflows (see
  FDA/EMA population-PK guidance notes referencing |r| ≥ 0.7 as a
  correlation concern). Consumed by `_check_covariate_correlation`.
- **`missingness_full_information_cutoff = 0.15`** — Above this %-missing,
  the profiler recommends full-information likelihood (FREM / MI) over
  single-imputation strategies. Aligns with Wahlby-Jonsson-Karlsson
  (2002, AAPS PharmSci 4:E27) and Ribbing-Jonsson (2004, JPKPD 31:109)
  covariate-model selection practice. Consumed by
  `_assess_covariate_missingness` and threaded into the
  `MissingDataDirective` resolver.

### `nonlinear_clearance`

Multi-signal voting for MM / saturable clearance detection.

- **`mm_curvature_ratio = 1.8`** — Signal 1 vote threshold. Median
  early/late post-Cmax log-slope ratio > 1.8 → elevated curvature
  indicative of saturable elimination. Richardson et al. (2025, Commun
  Med 5:327) note that 1-compartment linear drugs produce ratios ≈ 1,
  2-compartment linear (alpha+beta phases) span ≈ 1.3–2.5, and MM
  systems climb past 2.5. The 1.8 threshold is a conservative MM
  trigger that still fires on clear 2-cmt + MM composites.
- **`tmdd_curvature_ratio = 0.3`** — Reserved for target-mediated
  drug disposition screening (reverse signal: late-phase *increases* in
  slope). Not yet wired; retained for future Phase 3 work.
- **`compartmentality_curvature_ratio = 1.3`** — Separates 1-cmt from
  2-cmt classification in `_assess_compartmentality`. Distinct from
  `mm_curvature_ratio` because a 2-cmt drug with alpha/beta ≈ 1.3–2.5
  should not be misclassified as MM-like.
- **`terminal_monoexp_r2_linear_threshold = 0.85`** — Signal 2 vote
  threshold. Median per-subject adj-R² of monoexponential fit to the
  terminal ~30% of post-Cmax samples; below 0.85 signals monoexponential
  failure (evidence against clean linear elimination). Used jointly
  with Signal 1 to avoid single-signal false positives.

### `smith_2000_dose_proportionality`

- **`theta_low = 0.80`, `theta_high = 1.25`** — Translated
  bioequivalence acceptance bounds per Smith, Vandenhende, DeSante, et
  al. (2000, Pharm Res 17:1278–1283; DOI `10.1023/a:1026451721686`).
  Dose proportionality is declared if the 90% CI for the power-model
  slope β lies within the critical region derived from these bounds
  and the observed dose ratio.
- **`min_dose_levels = 3`, `min_dose_ratio = 3.0`** — Eligibility gates
  for the Smith power-model fit. Fewer than 3 distinct doses or a dose
  span < 3-fold makes the fit degenerate; Signal 3 abstains with an
  explicit eligibility reason.

### `huang_2025_lambda_z`

- **`min_points = 3`, `tolerance = 1e-4`** — Huang, Fidler, Lan, Cheng,
  Kloprogge, Standing (2025, J Pharmacokinet Pharmacodyn 52:60; DOI
  `10.1007/s10928-025-10000-z`) `find_best_lambdaz` algorithm. Minimum
  terminal-window size and adj-R² tie-break tolerance.
- **`phoenix_constraint = true`** — APMODE-specific extension: constrain
  the candidate terminal window to start no earlier than
  `Tmax + (Tlast - Tmax) / 2`. Without this, max-adj-R² over-truncates
  noisy 2-cmt beta-phases (Phoenix WinNonlin convention).
- **`adj_r2_threshold = 0.7`** — Routine λz quality advisory floor. Used
  by `_assess_flip_flop_risk` as a hard floor: median terminal adj-R²
  below this → return `"unknown"` because the fit is too poor to support
  *any* flip-flop call. Distinct from the stricter
  `flip_flop.quality_adj_r2_min` (0.85) used for `"likely"` upgrades.
  **Advisory, not disqualifying** — Gate 1 identifiability checks live
  in `GatePolicy`, not here.

### `steady_state`

- **`n_half_lives_required = 3`** — Primary SS criterion: elapsed
  dosing time must cover ≥ 3 terminal half-lives (≈ 87.5% steady-state
  accumulation). Classical popPK convention.
- **`n_doses_alt = 5`** — Fallback criterion when half-life is
  unavailable: require ≥ 5 evenly-spaced doses.
- **`interval_tolerance = 0.25`, `dose_tolerance = 0.20`** — Tolerance
  for "evenly spaced" and "uniform AMT" checks; stricter tolerances
  reject pseudo-SS regimens that contaminate Smith-2000 fits.
- **`min_doses = 3`** — Absolute floor; below this the SS test is
  not meaningful.

### `shape_detection`

- **`multi_peak_fraction_threshold = 0.3`** — Fraction of subjects
  with ≥ 2 prominent peaks for the dataset to be classified as
  multi-phase absorption. Below 0.3 the multi-peak signal is treated
  as a minority (possibly noise) and the dataset stays `simple`.
- **`lag_signature_fraction_threshold = 0.5`** — Fraction of subjects
  with low early concentrations for the dataset to be classified as
  `lag-signature`. Stricter than multi-peak because lag is more easily
  confounded with dense sampling noise.
- **`lag_early_conc_fraction = 0.05`, `lag_early_time_percentile = 25`**
  — Per-subject lag detection: observations in the first 25% of
  post-dose TIME with concentrations below 5% of Cmax.
- **`peak_prominence_range_fraction = 0.10`, `peak_prominence_cmax_floor = 0.05`**
  — Peak prominence floor is max of (10% of dynamic range, 5% of Cmax).
  Prevents descending-limb noise wiggles from registering as peaks in
  `scipy.signal.find_peaks`.
- **`peak_min_distance_intervals = 2.0`** — Minimum separation (in
  sampling intervals) between counted peaks.

### `subject_quality`

- **`min_subjects_for_median = 4`** — Below this, population-median
  statistics are not computed (single-digit medians are unstable).
- **`min_concs_for_profile = 5`** — Per-subject minimum for any
  profile-shape heuristic (terminal fit, curvature, Wagner-Nelson).
- **`min_subjects_for_dynamic_range = 10`** — Cmax p95/p05 dynamic-range
  ratio requires ≥ 10 positive-Cmax subjects to be stable.
- **`min_obs_per_subject_rich = 8`, `min_obs_per_subject_moderate = 4`**
  — Richness classification bands (`_classify_richness`): `< moderate`
  → sparse, `moderate…rich` → moderate, `> rich` → rich. Aligned with
  nlmixr2autoinit and Pharmpy AMD (Chen et al. 2024) sampling-density
  conventions.
- **`absorption_coverage_min_pre_tmax = 2.0`** — Average pre-Tmax
  observations per subject required for `adequate` absorption coverage.
  Below this ka is not reliably identifiable.
- **`elimination_coverage_min_post_tmax = 3.0`** — Average post-Tmax
  observations per subject required for `adequate` elimination coverage.
  Minimum for a defensible terminal-phase fit.

### `error_model`

Beal (2001, JPKPD 28:481–504) M3 selection + residual-error heuristics.

- **`blq_m3_trigger = 0.10`** — BLQ burden threshold above which the
  profiler forces M3 (likelihood-based censored-data handling) and
  excludes additive-only error models. Follows Ahn, Karlsson, Dunne,
  Ludden (2008, JPKPD 35:401) which shows M3 + proportional performs
  best under heavy censoring; additive-only corrupts CL estimates.
- **`dynamic_range_proportional = 50.0`** — Cmax p95/p05 ratio above
  which proportional error is preferred (sigma scales with
  concentration over the observed range).
- **`high_cv_ceiling = 80.0`** — Protect against extreme-CV datasets:
  above this CV% the proportional recommendation is suppressed and
  `combined` is preferred.
- **`lloq_cmax_combined = 0.05`** — LLOQ/Cmax_median > 5% → combined
  error needed (additive component matters near the LLOQ).
- **`terminal_log_mad_combined = 0.35`** — Terminal log-residual MAD
  > 0.35 → combined error needed.
- **`narrow_range_additive = 5.0`, `low_cv_additive_ceiling = 30.0`**
  — Narrow dynamic-range (< 5) + low CV (< 30%) biomarker-like signal
  → additive error plausible.

### `node_readiness`

PRD §4.2.4 R6 NODE design feasibility (Pharmpy AMD design audit).

- **`min_subjects.{discovery,optimization} = {20, 10}`**
- **`min_median_samples.{discovery,optimization} = {8, 4}`**
- **`dim_budget.{discovery,optimization} = {8, 4}`** — NODE input-dim
  ceilings per lane. Drives the `node_dim_budget` EvidenceManifest
  field; Lane Router excludes NODE when budget = 0.

### `flip_flop`

Richardson et al. (2025, Commun Med 5:327) flip-flop automation pitfall.

- **`ka_lambdaz_ratio_likely = 1.0`** — Population-median W-N ka < λz
  → "likely" flip-flop (absorption slower than elimination in the
  terminal slope).
- **`ka_lambdaz_ratio_possible = 1.5`** — Population-median W-N ka <
  1.5·λz → "possible" flip-flop (weaker separation).
- **`quality_adj_r2_min = 0.85`** — **Strict** terminal-fit quality
  required to upgrade a "possible" call to "likely". Richardson 2025
  methodology: flip-flop detection is structurally sensitive to
  terminal-slope quality; routine 0.70 (Huang 2025) is insufficient.
- **`quality_min_npts = 4`** — Minimum median terminal-window size for
  the strict-quality check.

Tiered decision in `_assess_flip_flop_risk`:

| median adj-R² | behavior |
|---|---|
| `< lambdaz_adj_r2_threshold` (0.70) | return `"unknown"` (floor) |
| `∈ [floor, strict)` | `"likely"` downgraded to `"possible"` |
| `≥ strict` (0.85) AND npts ≥ 4 | `"likely"` retained |

### `tad_consistency`

- **`in_window_fraction_clean = 0.80`** — For multi-dose datasets, ≥ 80%
  of observations must fall within a dose interval `[d_i, d_i+τ]` for
  TIME to be treated as TAD-equivalent. Below this threshold,
  shape-heuristic dispatch signals are down-weighted (TAD contamination
  flag → NODE dispatch removed in Lane Router).

### `protocol_heterogeneity`

- **`obs_per_subject_cv_threshold = 0.5`** — For pooled studies,
  coefficient of variation of observations-per-subject across studies
  above this value flags `pooled-heterogeneous` (triggers IOV testing
  per Lane Router constraint notes).

### `dvid_filter`

- **`pk_dvid_allowlist = ["cp", "1", "conc", "concentration"]`** —
  Observation rows with DVID values matching this allowlist are
  retained as PK; non-matching rows are dropped before shape geometry.
  Prevents warfarin-style datasets (cp + pca) from routing PD samples
  through PK heuristics.
- **`fail_open_when_no_match = true`** — If no observation row matches
  the allowlist, the filter abstains and logs a warning rather than
  dropping all rows. Set to `false` for strict-DVID deployments.

## Advisory vs disqualifying taxonomy

This file describes the **profiler** policy. Profiler fields produce
EvidenceManifest entries that are **advisory** — they shape downstream
dispatch (Lane Router) and ranking inputs (Gate 3), but a single
manifest field never on its own removes a candidate from consideration.

**Disqualifying** thresholds live in `GatePolicy` (see
`apmode/governance/policy.py`) and drive Gate 1/2/2.5 evaluators. When
a field is described as "drives" or "triggers" dispatch here, follow
the code path through to the router or gate evaluator to see the actual
hard/soft decision.

## How to propose a new threshold

1. Open a PR touching `policies/profiler.json` (new key).
2. Add the matching `ProfilerPolicy` field in
   `apmode/data/policy.py` and a loader entry in `_load_from_path`.
3. Add a derived `_POLICY.*` constant in `apmode/data/profiler.py`
   next to its siblings.
4. Use the constant at every call site; do not introduce bare
   literals inside heuristic functions.
5. Update this file with a new entry (primary source + rationale).
6. Bump `policy_version` in the JSON and add a row to the table above.
7. CI runs `tests/unit/test_profiler_policy_consistency.py` which
   enforces (a) JSON↔dataclass mapping, (b) constant↔policy equality,
   (c) AST-level absence of bare literals in the drift-prone
   heuristic functions.
