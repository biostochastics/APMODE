# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### rc9 Scope 1 — orchestrator threading of posterior-predictive kwargs

The rc8 pipeline built the posterior-predictive diagnostics machinery
(R-harness `predicted_simulations` payload → `build_predictive_diagnostics`
→ `DiagnosticBundle.{vpc,npe_score,auc_cmax_be_score,auc_cmax_source}`)
and the runner-side wiring in `Nlmixr2Runner`, but the orchestrator
never actually forwarded `gate3_policy` or `nca_diagnostics` to the
runner, so every rc8 run silently fell back to the CWRES NPE proxy.
This pass threads both kwargs end-to-end:

- **`BackendRunner` protocol** extended with the two kwargs
  (`backends/protocol.py`). Every concrete runner now accepts them for
  Protocol-conformance and forwards (or explicitly defers) accordingly.
- **`Nlmixr2Runner`** already consumed both kwargs in rc8 — no change.
- **`AgenticRunner`** forwards verbatim into `self._inner.run(...)` on
  every iteration so the LLM loop sees the same cross-paradigm signal
  Gate 3 will evaluate (`backends/agentic_runner.py`).
- **`run_frem_fit`** forwards into the wrapped `Nlmixr2Runner`
  (`backends/frem_runner.py`).
- **`NodeBackendRunner` / `BayesianRunner`** accept and explicitly defer
  — NODE posterior-predictive is the Phase 3 random-effects stub
  (Scope 4); the Bayesian path lands in Scope 2 (Stan `y_pred`
  generated-quantities emission) (`backends/node_runner.py`,
  `backends/bayesian_runner.py`).
- **`SearchEngine`** accepts the kwargs in its constructor, stores them,
  and forwards at `runner.run` in `_evaluate_candidate`
  (`search/engine.py`).
- **`evaluate_loro_cv`** forwards the kwargs per-fold so the per-fold
  VPC populates from the posterior-predictive path instead of
  degenerating to the empty-dict CWRES fallback (`evaluation/loro_cv.py`).
- **Orchestrator** pulls `gate3_policy = policy.gate3` and
  `nca_diagnostics = estimator.diagnostics` once after the
  initial-estimates stage and threads them to every runner dispatch
  site: `SearchEngine` ctor, the seed-stability loop (`_seed_run`),
  `_run_agentic_stage` (both refine + independent modes),
  `_run_frem_stage` → `run_frem_fit`, `_run_mi_stage::_fit_one_imputation`,
  and `_run_loro_cv::_eval_one` → `evaluate_loro_cv`
  (`orchestrator/__init__.py`).
- **Threading test** pins the contract: the new
  `TestDiscoveryLaneIntegration::test_orchestrator_threads_gate3_policy_to_runner`
  wires a recording mock runner against the submission-lane policy and
  asserts the last dispatch carried a `Gate3Config` — regressions where
  the orchestrator silently drops the policy fail this test loudly
  (`tests/integration/test_discovery_lane.py`).
- **FREM stub-runner test** (`tests/unit/test_frem_runner.py`) updated
  to accept the new kwargs so the plumbing test stays green.

With this threading in place, an end-to-end run now activates the full
rc8 posterior-predictive pipeline when the R harness supplies
`predicted_simulations`. When rxode2 simulation fails the existing
structured warning in `Nlmixr2Runner._parse_response` surfaces and Gate
3 falls back to CWRES — no change to that path.

## [0.4.0] — 2026-04-16

### Version bump + README single-source automation (+0.1 from 0.3.0-rc series)

**Summary.** Rolls up rc7/rc8 + the post-rc8 consensus-review patches into
0.4.0. Tightens the README ↔ codebase contract so numeric claims can't
drift again.

- **Gate policy version unified to `0.4.0`** across all three lanes.
  Submission and Discovery were previously `0.3.1`; Optimization was
  already `0.4.0` (gate3 weight-shape change). Uniform tag now lets CI
  hooks enforce a single version for the whole policy set
  (`policies/submission.json`, `policies/discovery.json`,
  `tests/unit/test_gate_policy.py`).
- **`fallback-version` in pyproject bumped** from `0.2.0.dev0` →
  `0.4.0.dev0` so hatch-vcs editable installs without a recent tag
  report a sensible near-release version.
- **`scripts/sync_readme.py` (new)** — single-sources 9 numeric claims
  from the codebase into `README.md` + `CLAUDE.md` via
  `<!-- apmode:AUTO:<key> -->…<!-- apmode:/AUTO:<key> -->` markers:
  `version`, `version_tag`, `tests`, `tests_nonlive`, `policy_gate`,
  `policy_profiler`, `profiler_manifest`, `transforms`, `cli_cmds`,
  `datasets`, `backends`. `--check` exits 1 on drift — wire into
  pre-commit/CI.
- **README comprehensively rewritten**:
  - Architecture ASCII diagram now shows all four backends (nlmixr2 /
    Bayesian-Stan / NODE / Agentic LLM), not two.
  - Key Components table adds rows for `bayesian_runner.py`,
    `bayes/harness.py`, `frem_runner.py`, `predictive_summary.py`,
    `report/`, `paths.py`.
  - Agentic Transforms section corrected to 7 (was 6 — missed
    `set_prior`).
  - Bayesian Gate 1 thresholds table gains `ebfmi_min` (0.30) and
    `pareto_k_max` (0.70) rows to match what the prose already claimed.
  - CLI Reference extended from 7 to 14 commands with `report`,
    `doctor`, `ls`, `policies`, `trace`, `lineage`, `graph`.
  - Adds env-var documentation (`APMODE_POLICIES_DIR`, provider keys,
    `OLLAMA_HOST`) and a "worked end-to-end walkthrough" seven-step
    script.
  - Adds the Bayesian backend, FREM emitter, and FREM runner to the
    Phasing table (formerly only in prose).
  - Fixes profiler policy version (README had said
    `manifest_schema_version = 3`; the JSON was always `2`).
  - Adds Known Limitations entry for
    `NodeBackendRunner.sample_posterior_predictive` returning `None`
    stub (post-rc8 review item) and the Optimization-lane Gate 3
    uniform-drop caveat.

### Consensus review fixes (folded in from the post-rc8 Unreleased block)

Post-rc8 multi-model review identified six ship-now items and a
deferred-research list. All six ship items landed in this pass;
research items are tracked for the next cycle.

**Ship items (this pass):**

- `auc_cmax_source` literal renamed from `"observed_nca"` to
  `"observed_trapezoid"` — our implementation is plain
  `np.trapezoid` on the observed sample range with no λz-based
  extrapolation to AUC-inf. Calling it "observed_nca" was a
  label-vs-substance gap for a regulatory-adjacent pipeline.
  `DiagnosticBundle.auc_cmax_source`, `CrossParadigmMetrics.auc_cmax_source`,
  `PredictiveSummaryBundle.auc_cmax_source`, and all tests touching the
  literal now carry the honest name (`bundle/models.py`,
  `governance/ranking.py`, `backends/predictive_summary.py`,
  `benchmarks/scoring.py`, `tests/unit/test_*.py`).
- `NodeBackendRunner.sample_posterior_predictive` stub — shipped as a
  type-checked discoverable method that returns `None` with an audit-
  trail `UserWarning`. The ranker's uniform-drop rule already treats
  `None` as "backend omitted sims", so the stub formalizes the public
  API contract without changing Gate 3 behavior. When Phase 3 random-
  effects infrastructure lands in `node_trainer.py`, this method
  becomes the concrete integration point (`backends/node_runner.py`).
- `BackendResult.reject_predicted_simulations_field` — Pydantic
  `model_validator(mode="before")` that raises loudly if a caller
  forgets to strip the out-of-band `predicted_simulations` carrier key
  from the harness response dict. Guards the runner pattern in
  `Nlmixr2Runner._parse_response` against future schema drift
  (`bundle/models.py`).
- `build_predictive_from_draws` type hints tightened from `Any` to
  `numpy.typing.ArrayLike` + concrete Pydantic / Gate3Config types.
  Restores mypy-strict parity with the rest of the codebase
  (`bayes/harness.py`).
- Structured logging on R-sim fallback — when a `Gate3Config` is
  provided but `predicted_simulations` is absent from the harness
  response, `Nlmixr2Runner._parse_response` now emits a
  `logging.WARNING` naming the candidate + the requested sim count +
  the CWRES-proxy fallback decision. Previously silent
  (`backends/nlmixr2_runner.py`).
- **Optimization-lane transitional-policy disclosure** — the rc8
  policy flip (auc_cmax_weight=0.30) in a mixed nlmixr2+NODE/Bayesian
  survivor set triggers the uniform-drop rule for all candidates,
  effectively reverting to the rc7 two-component composite until the
  predictive-diagnostic integration lands for NODE / Bayesian. This is
  intentional (creates implementation pressure) — now explicitly
  called out here so a future reviewer reads it as policy, not bug.

**Research-item follow-up pass (all landed in this cycle):**

- `Gate3Config.npe_aggregation: Literal["flatten", "per_subject_median"]`
  — default `"flatten"` preserves rc8 behavior (pool all obs/sim-median
  pairs before the median-absolute-error); `"per_subject_median"`
  computes NPE per subject then medians across subjects, reducing bias
  toward dense-sampled subjects in unbalanced designs. Provenance of
  the active choice is persisted on `PredictiveSummaryBundle.
  npe_aggregation` so the bundle audit trail can detect drift.
- `Gate3Config.auc_cmax_aggregation: Literal["median_trajectory",
  "median_of_aucs"]` — default `"median_trajectory"` preserves the rc8
  point-estimate path (collapse sims to per-sim median trajectory then
  trapezoid once); `"median_of_aucs"` trapezoidates each sim separately
  per subject and takes the median of those scalar AUC/Cmax values,
  preserving distributional uncertainty for nonlinear profiles.
- `Gate3Config.vpc_include_prediction_corrected: bool = False` —
  Python-side policy plumbing shipped; R harness second-pass
  (pooled-grid `rxSolve`) tracked as a follow-up commit. When enabled
  in a future cycle, the bundle's `VPCSummary.prediction_corrected`
  becomes `True` for regulatory-facing runs expecting a smoothed VPC.
- `PredictiveSummaryBundle.mask_drop_reasons: dict[str, int]` — fixed
  taxonomy bucket count of NCA-ineligibility reasons
  (`{absorption, elimination, blq, span, lambda_z, auc_extrap,
  missing, other}`). Reviewers use this to distinguish "8/12 eligible
  with 4 λz-failures" from "8/12 eligible with 4 extrapolation-
  failures" — information loss from mask-drop is less concerning when
  failures are uncorrelated with structural-model discrimination signal.
- `apmode log --top` now renders per-percentile VPC coverage
  (``VPC[p5=0.87/p50=0.91/p95=0.94]``) in each candidate's subtitle
  when populated — the ranker's scalar concordance hides the per-
  percentile signal pharmacometricians need to read failure modes.
- `Gate3Config.vpc_n_bin_collapse_warn_ratio: float = 0.5` —
  `_compute_vpc_from_sims` now emits a `logging.WARNING` when the
  effective bin count (after `np.unique` on quantile-tied edges) drops
  below the policy-configured fraction of `vpc_n_bins`. Coverage
  estimates on sparse-sampling designs are noisier than the bin count
  suggests; the warning surfaces the audit event rather than hiding it.

**Verification:**

- `uv run ruff check src/apmode/ tests/` — clean.
- `uv run mypy src/apmode/ --strict` — clean.
- `uv run pytest tests/ -m "not live and not slow"` — 1655 passed.

### Code-quality pass (2026-04-16)

Ruff + mypy-strict clean; 1655 tests pass.

**Correctness / robustness:**

- Orchestrator now reads `agentic.trace_dir` via a new public
  read-only property on `AgenticRunner` instead of the private
  `_trace_dir` attribute (`backends/agentic_runner.py`,
  `orchestrator/__init__.py`).
- `_gate3_cross_paradigm` raises `RuntimeError` when ranking returns
  an orphan candidate id not present in survivors. Previously logged
  and silently skipped — masking a ranking-module invariant violation
  and producing a misaligned ranked list (`governance/gates.py`).
- `apmode.paths.policies_dir()` / `policy_path_for_lane()` centralise
  policy-file resolution so the CLI and orchestrator can't drift on
  `Path(__file__).parents[N]` parent-count heuristics. Walks up for
  `pyproject.toml`, honours `APMODE_POLICIES_DIR`, falls back to
  `importlib.resources` for future packaged-policies installs
  (`paths.py` new, `orchestrator/__init__.py`, `cli.py`).
- `candidate_lineage.json` load on `--resume-agentic` narrowed from a
  bare `except Exception` to
  `(json.JSONDecodeError, ValidationError, OSError)` with a
  `warning`-level log of the failed path
  (`orchestrator/__init__.py`).
- `compute_auc_cmax_be_score(n=0)` now returns `None` unconditionally.
  Empty cohort is "undefined" (no data), not "BE-failed"; the old
  `0.0` poisoned downstream `max()` aggregation. The uniform-drop
  rule in `rank_cross_paradigm` already handles `None` cleanly.
  `test_empty_subjects_returns_zero` and
  `test_empty_inputs_legacy_returns_zero` renamed / updated to
  assert `is None` (`benchmarks/scoring.py`,
  `tests/unit/test_cross_paradigm_ranking.py`,
  `tests/unit/test_predictive_summary.py`).
- `run_with_imputations` signature widened from
  `Callable[[Path, int], ...]` to `Callable[[Path, int, int], ...]`
  so the search callable receives both seed and imputation index.
  The runner now asserts the returned `imputation_idx` matches the
  loop index instead of silently overwriting. Orchestrator's
  `_fit_one_imputation` sets the idx directly
  (`search/stability.py`, `orchestrator/__init__.py`).
- `score_parameter_bias(ref_value=0)` returns `float("nan")` instead
  of `abs(est)` — the old fallback silently mixed unit-ful absolute
  error with unitless relative bias under the same dict key,
  conflating scales across parameters in downstream `max()`
  aggregation (`benchmarks/scoring.py`).
- `classical_checkpoint.json` schema 1.0 → 1.1. Spec and result are
  nested dicts (`spec`, `result`) instead of JSON-encoded strings
  (`spec_json`, `result_json`). Easier to inspect with `jq`, no
  double-parse on load; the loader still accepts 1.0 bundles
  (`orchestrator/__init__.py`).
- `_launch_run` accepts a `timeout: int = 900` parameter matching
  `run`'s own default. Previously hard-coded `timeout=600`, silently
  overriding the default in `apmode explore -y` (`cli.py`).
- `_show_bundle_overview` prefers `candidate_id` with `model_id`
  fallback for legacy bundles — the pattern used elsewhere in the
  CLI (`cli.py`).
- `_run_agentic_stage` now carries a comment explaining why refine +
  independent modes cannot be parallelised as-is: the shared
  `Nlmixr2Runner.work_dir` would race on R subprocess intermediate
  files. Any future parallelisation must give each mode its own
  `Nlmixr2Runner` with a distinct `work_dir`, and must verify the
  LLM client has no shared mutable state
  (`orchestrator/__init__.py`).

**Quality / style:**

- `SubjectSimulation.__post_init__` flips `arr.flags.writeable = False`
  on the three numpy arrays — defence-in-depth against accidental
  in-place mutation of the simulation matrix
  (`backends/predictive_summary.py`).
- Dropped redundant `edges.copy()` in `_bin_edges_from_pooled_times`
  (`np.unique` already returns a fresh array).
- Hoisted `import re` out of the nested `_mermaid_id` in the graph
  renderer to module scope (`cli.py`).
- `datetime` import consolidated to the top of
  `orchestrator/__init__.py` (removed the per-method local import).
- Split-integrity magic numbers (`cwres_drift=0.5`,
  `outlier=2*x+0.05`) moved to `Gate1Config.split_cwres_drift_max`,
  `split_outlier_ratio_slope`, `split_outlier_ratio_intercept`
  (`governance/policy.py`, `governance/gates.py`). Defaults match
  the prior hard-coded values — no behaviour change.
- Added `A` (flake8-builtins) to Ruff selectors with `A002`/`A003`
  ignored (Typer option-keyword shadowing of `format`/`type` is
  intentional) (`pyproject.toml`).
- `_check_loro_requirement` marks unused `result` parameter via
  `del result` — matches the pattern in `_check_imputation_stability`
  (`governance/gates.py`).

**Verification:**

- `uv run ruff check src/apmode/ tests/` — clean.
- `uv run mypy src/apmode/ --strict` — clean.
- `uv run pytest tests/ -m "not live and not slow"` — 1655 passed,
  0 failed.

## [0.3.0-rc8] — 2026-04-16

### Gate 3 predictive-diagnostics foundation: shared helper + per-subject NCA eligibility

Foundation commit for the backend VPC/simulation pipelines work tracked
post-rc7. Ships the helper and policy knobs so the next commit can land
the nlmixr2 R-harness VPC as a drop-in.

**1 — `src/apmode/backends/predictive_summary.py` (new, unit-tested):**

- `SubjectSimulation` frozen dataclass carrying `(n_sims, n_obs_i)` per
  subject, observed DV/time vectors, and an optional
  `NCASubjectDiagnostic`. One forward-solve per subject at observed
  times supplies VPC, NPE, and AUC/Cmax atomically — no separate VPC
  simulation grid, no interpolation.
- `build_predictive_diagnostics(per_subject_sims, *, policy, ...)` →
  `PredictiveSummaryBundle`. Validates shape invariants (ndim, n_obs
  alignment, uniform `n_sims` across subjects) before any math; raises
  `ValueError` on malformed inputs instead of producing silent NaN.
  Populates all three diagnostics from one pass so backends cannot
  cherry-pick which metric to report (partial-population ban — see
  `DiagnosticBundle` docstring).
- `_compute_vpc_from_sims` implements xpose4/PsN "percentile-based VPC
  with confidence intervals": for each percentile `p` and post-hoc
  time bin, compute the empirical percentile of observations and the
  central `coverage_target`·100% CI of the per-sim simulated
  percentile, count "hits" when the former lies inside the latter.
  `coverage[f"p{p}"]` reports the hit fraction — matches the ranker's
  `compute_vpc_concordance(target)` convention (single scalar target
  per percentile).
- `_per_subject_auc_cmax` — NaN-safe trapezoidal AUC and Cmax per
  subject; returns `(NaN, NaN)` on fewer than 2 finite points
  (undefined AUC) rather than crashing.

**2 — `benchmarks/scoring.py`: per-subject NCA eligibility:**

- `is_nca_eligible_per_subject(NCASubjectDiagnostic)` — per-subject
  sibling of the pooled `is_nca_eligible_for_auc_cmax`. Delegates to
  `NCASubjectDiagnostic.excluded` + `excluded_reason` (profiler already
  encodes adj-R² on λz, AUC extrapolation, span ratio, and minimum
  λz-point count there). Reuses the profiler's QC instead of
  duplicating the logic — no dodge surface because the mask is
  observed-data-only, candidate-independent.
- `compute_auc_cmax_be_score` extended with keyword-only
  `eligible_mask`, `min_eligible`, `min_eligible_fraction`. Return type
  widened to `float | None`:
  - Legacy path (no mask, no floors) returns `float` on non-empty
    input and `None` on empty cohorts (empty is now semantically
    "undefined / no data," not "BE-failed"; previously returned 0.0
    — consumers that compared against `== 0.0` must use
    `is None`). Non-empty legacy behavior is unchanged.
  - With mask: numerator = BE-pass AND eligible; denominator = eligible
    count (**mask-drop**, not BE-fail). Candidates cannot dodge the
    metric because eligibility is run-level observed-data QC. Both
    floors AND-combined; below either → `None` → ranker's uniform-drop
    rule removes the component for all candidates.
- `test_gate_policy.TestLanePoliciesGate3Contract::test_all_lanes_policy_version_bumped`
  updated from `0.3.0` → `0.3.1`.

**3 — `governance/policy.py::Gate3Config` new knobs:**

| Field | Default | Bounds | Purpose |
|---|---|---|---|
| `n_posterior_predictive_sims` | 500 | `[100, 5000]` | Backend ETA draws per candidate (Bergstrand 2011 VPC convention) |
| `vpc_n_bins` | 10 | `[3, 100]` | Post-hoc bins for VPC coverage aggregation |
| `auc_cmax_nca_min_eligible` | 8 | `≥1` | Absolute per-subject NCA-eligible floor |
| `auc_cmax_nca_min_eligible_fraction` | 0.5 | `[0, 1]` | Eligible-fraction floor (AND with absolute) |

Defaults: 500 sims matches the VPC convention, 8-subject floor avoids
"2–4 subjects drive the score" artifacts, 0.5 fraction prevents skewed
cohort representation. None of these are written to the per-lane policy
JSON — the repo convention is that non-default values are explicit,
defaults are implicit (matches existing `npe_cap` / `bic_norm_scale`
handling).

**4 — CLI: `apmode policies <lane>` surfaces flat Gate 3 fields:**

Pre-rc8 the `policies` command iterated gate dicts expecting a nested
`checks` key, and silently rendered **nothing** for the flat Gate 3
block. Fixed so each `gate3.*` leaf (composite_method, weights, and
any policy-overridden floors) now appears in the
`apmode policies <lane>` output.

**5 — Policy file version bump:** `policy_version` ticks from `0.3.0`
→ `0.3.1` across all three lane policies. No default behaviour change
— the new Gate3Config fields preserve existing ranking math until a
backend emits a simulation matrix (next commit).

**Not implemented (intentionally deferred):**

- **nlmixr2 R-harness VPC** — next commit. The contract documented at
  `r/harness.R:204` and `node_runner.py:202` lets backends populate
  diagnostics with a one-line call to `build_predictive_diagnostics`.
- **NODE posterior-predictive** — pending `node_trainer.py` audit;
  Laplace-approximation MVP planned.
- **Bayesian harness** — `bayes/harness.py` still absent; Stan
  `generated quantities` block emission queued.
- **`policies/optimization.json` `auc_cmax_weight` flip to 0.30** —
  scheduled for the same commit as nlmixr2 VPC.
- **`apmode log --top` NPE / AUC / Cmax columns** — deferred until
  backends populate the fields; cosmetic CLI polish today.

**Verification:**

- `uv run ruff format`, `uv run ruff check` — clean.
- `uv run mypy --strict src/apmode/backends/predictive_summary.py
  src/apmode/benchmarks/scoring.py src/apmode/governance/policy.py` —
  clean.
- `uv run pytest tests/unit/test_predictive_summary.py` — 29/29 passed.
- `uv run pytest tests/unit/test_{cross_paradigm_ranking,gate_policy,gates,bundle_models,missing_data}.py` — 186/186 passed.
- `uv run python -m apmode.governance.validate_policies policies/` — 4/4 OK.
- `uv run apmode policies optimization` — Gate 3 fields now render.

## [0.3.0-rc7] — 2026-04-16

### Gate 3 follow-up: BLQ enum expansion, AUC/Cmax BE metric, lane-specific composites

Addresses the deferred-list work from the post-c0fc317 review. Verified
scope narrower than the original briefing on #2 because backend VPC /
simulation pipelines do not yet exist — see "Contract, not
implementation" notes below.

**1 — BLQHandling enum expansion (correctness bug):**

- `bundle/models.py::BLQHandling.method` — expanded from
  `Literal["none", "m3", "m4"]` to
  `Literal["none", "m1", "m3", "m4", "m6_plus", "m7_plus"]` (models.py:136). The
  policy side (`MissingDataPolicy.blq_method`) already emitted `"M7+"` via
  `data/missing_data.py::resolve_directive` but the bundle enum had no slot
  for it, so the directive was silently coerced at the boundary. `_`-separated
  lowercase form preserved to stay a valid Python `Literal` member.
- `data/missing_data.py::normalize_blq_method_for_bundle` — new helper at
  the policy→bundle boundary maps `"M1"/"M3"/"M4"/"M6+"/"M7+"` → the bundle
  enum. Raises `ValueError` on unknown inputs rather than coercing (coercion
  is exactly what was masking M7+). Canonical map in `_BLQ_POLICY_TO_BUNDLE`.
- `tests/unit/test_cross_paradigm_ranking.py::test_same_backend_m3_vs_m7_plus_forces_simulation`
  — regression test covering the silent-drop hole: within-paradigm
  M3-vs-M7+ pair must trigger `ranking_requires_simulation_metrics=True`
  because the two observation models have incomparable likelihood scales.
- `tests/unit/test_missing_data.py::TestNormalizeBLQMethodForBundle` —
  round-trips every policy-side method and asserts unknown inputs raise.

**2 — NPE population contract (wire-ready, not full implementation):**

- `backends/node_runner.py` (line ~202) and `r/harness.R` (line ~204) —
  explicit docstring at each `DiagnosticBundle` construction site pointing
  to `apmode.benchmarks.scoring.compute_npe` as the canonical NPE source,
  with the invariant that VPC + `npe_score` must be derived from the *same*
  posterior-predictive simulation matrix. Concrete VPC generation is a
  per-backend implementation task not covered here (R harness currently
  emits `vpc = NULL`; NODE and Bayesian runners produce no posterior-
  predictive samples).
- `tests/unit/test_cross_paradigm_ranking.py::TestAgenticRunnerPropagatesNPEScore`
  — regression test confirming that `AgenticRunner.run` end-of-loop
  re-stamps `backend="agentic_llm"` via `diagnostics=best_result.diagnostics`
  (agentic_runner.py:656), preserving any `npe_score` the inner backend
  populated. Without this, agentic candidates would silently demote to
  the CWRES proxy.
- `tests/unit/test_cross_paradigm_ranking.py::test_compute_npe_end_to_end_wire`
  — runs `compute_npe(observed, sims)` on deterministic arrays, stores on
  `diagnostics.npe_score`, asserts the ranker picks up exactly that value
  with `npe_source="simulation"`. Contract, not integration.

**4 — AUC/Cmax bioequivalence metric (PRD §4.3.1):**

- `benchmarks/scoring.py` — new `compute_auc_cmax_be_score` (pure math,
  no eligibility knowledge) returns the per-subject BE pass fraction:
  fraction of subjects whose candidate/NCA GMRs fall in [0.80, 1.25]
  for *both* AUC and Cmax (Smith 2000 FDA goalposts). Uses per-subject
  pass vectors — subjects with non-finite or non-positive inputs count
  as BE-fail rather than being dropped (so partial-data issues are
  visible in the score). All-or-nothing scoring rejected — would
  collapse to 0 under moderate variability.
- `benchmarks/scoring.py::is_nca_eligible_for_auc_cmax` — policy gate for
  admitting an observed-data NCA reference: `absorption_phase_coverage ==
  "adequate"` + `elimination_phase_coverage == "adequate"` + BLQ burden
  below threshold (default 0.20 per Thway 2018; policy-configurable via
  `Gate3Config.auc_cmax_nca_max_blq_burden`). Returns `(eligible, reason)`
  for audit-trail surfacing. Candidate-derived fallback (median GMR)
  explicitly rejected — circular herd bias.
- `governance/policy.py::Gate3Config` — added `auc_cmax_weight: float =
  0.0` (default off; lanes opt in). `weights_sum_to_one` validator now
  sums 4 weights (vpc + npe + bic + auc_cmax == 1.0, 1e-6 tolerance).
  Added `auc_cmax_nca_max_blq_burden: float = 0.20` (eligibility knob).
- `bundle/models.py::DiagnosticBundle` — added `auc_cmax_be_score:
  float | None` (bounded [0, 1]) and
  `auc_cmax_source: Literal["observed_nca"] | None`. Minor-version
  schema bump: consumers reading DiagnosticBundle JSON with a strict
  schema must accept the new optional fields.
- `governance/ranking.py::CrossParadigmMetrics` — added `auc_cmax_be` +
  `auc_cmax_source` top-level fields (minor-version schema bump). New
  fields chosen over repurposing `component_scores` for clarity.
- `governance/ranking.py::_resolve_auc_cmax` — mirrors `_resolve_npe`
  but has *no fallback*: missing → `(None, None)` so the uniform-drop
  rule can trigger. Non-finite scores are treated as missing.
- `governance/ranking.py::_apply_uniform_auc_cmax_drop` — **uniform-drop
  rule**: when any candidate in the survivor set has `auc_cmax_be=None`
  AND `auc_cmax_weight > 0`, the component is dropped for *every*
  candidate and the remaining weights are proportionally renormalized.
  Per-candidate renormalization was explicitly rejected — it lets
  data-poor candidates dodge metrics their siblings are scored on,
  defeating cross-paradigm comparability (the whole point of Gate 3).
  Pathological case (auc_cmax was the only enabled component) falls
  back to equal vpc/npe=0.5 with documented reason.
- `governance/ranking.py::_weighted_sum_composite` and `_borda_composite`
  — accept `auc_cmax_be` (scalar) / `auc_cmax_list` (list); apply
  `(1 - auc_cmax_be) * auc_cmax_weight` under weighted sum; average-rank
  with `lower_is_better=False` under Borda. The uniform-drop guarantees
  either all entries are finite or `auc_cmax_weight` was zeroed.
- `governance/ranking.py::ranking_requires_simulation_metrics` — added
  optional keyword `gate3: Gate3Config | None = None`. When provided
  and `gate3.auc_cmax_weight > 0`, returns `(True, reason)` even for
  homogeneous survivor sets (single backend, uniform BLQ). Without this
  a policy enabling AUC/Cmax would see the BIC within-paradigm path
  take over and silently ignore the policy.
- `governance/gates.py::evaluate_gate3` — updated call site to pass
  `gate3=policy.gate3` so the policy opt-in actually takes effect.

**5 — Lane-specific Gate3Config in policy JSON:**

- `policies/submission.json` — kept default (`weighted_sum`, vpc=0.5,
  npe=0.5, bic=0.0, auc_cmax=0.0). BIC off is correct cross-paradigm
  even for the submission lane (PRD §10 Q2). `policy_version` bumped
  to `0.3.0`.
- `policies/discovery.json` — added explicit `gate3` block: Borda
  aggregation, vpc=0.5, npe=0.5, bic=0.0, auc_cmax=0.0. Borda chosen
  for scale-invariance across paradigms; auc_cmax held at 0.0 until
  the backend VPC pipeline produces simulation matrices — enabling it
  today would trigger uniform-drop on every ranking for no benefit.
  `policy_version` 0.3.0.
- `policies/optimization.json` — same Gate3Config shape as discovery
  for now. Migration note: flip `auc_cmax_weight` to a non-zero value
  (likely 0.3, with vpc/npe dropping to 0.35/0.35) once the nlmixr2 R
  harness and NODE runner emit posterior-predictive simulations. That
  switch will make simulation-based ranking the unconditional path for
  this lane (regulatory-adjacent), consistent with PRD §4.3.1.
  `policy_version` 0.3.0.
- `tests/unit/test_gate_policy.py::TestLanePoliciesGate3Contract` —
  per-lane assertions pinning the on-disk Gate3Config shape. An
  accidental edit can no longer silently flip discovery back to
  weighted_sum.

**6 — Low-priority cleanups:**

- `governance/gates.py` — deduped the two nested `_safe_bic` definitions
  (formerly at ~790 and ~872) to a single module-level helper near the
  imports. Shared sort-key for within-paradigm BIC ranking and the
  simulation-based pipeline; missing/non-finite BIC coerced to `+inf`
  so unstable-sort cannot land a missing-BIC candidate at rank 1.
- `governance/ranking.py::_average_rank` — replaced exact float equality
  (`indexed[j][1] == indexed[i][1]`) with
  `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)`. Metric values here
  flow through arithmetic (e.g. `1 - mean(deviations)` for VPC
  concordance) so real-valued ties can differ by one ULP after rounding.
  Exact `==` would silently break such ties in arbitrary sort order.
- `bundle/models.py::DiagnosticBundle` — documented the intentional
  mutability asymmetry. Evidence-driven decision after auditing the
  bundle module: the entire result/diagnostic family (GOFMetrics,
  IdentifiabilityFlags, BLQHandling, SplitGOFMetrics, BackendResult,
  DiagnosticBundle) is mutable because it gets populated in stages;
  the frozen classes are all input-side (EvidenceManifest, signals,
  directives). Freezing DiagnosticBundle alone would produce more
  inconsistency, not less. Preferred documenting the design over
  refactoring 25+ test mutation sites with no functional benefit.

**Not implemented (deferred from briefing):**

- **#3 benchmark regression for the Gate3Config default flip** — the
  user explicitly requested this be skipped for this cycle. Method
  revised: 3-cell same-commit A/B (old weights vs new weights,
  isolating proxy→real-NPE separately) instead of the
  b1567aa-vs-c0fc317 worktree comparison that would confound weight
  and metric changes. Deferred until backends emit real NPE.
- **Full backend VPC/simulation pipelines** — nlmixr2 R harness, NODE
  posterior-predictive, Stan Bayesian predictive. Each is a multi-week
  per-backend implementation; none is in this release. The contract
  documented at each construction site lets them land as one-liner
  additions when implemented.

**Verification:**

- `uv run ruff format`, `uv run ruff check` — clean.
- `uv run mypy --strict src/apmode/` — clean on 82 source files.
- Targeted unit sweep (governance + bundle + missing-data + diagnostic + FREM
  + agentic + search + LORO + benchmarks + lane router + error model +
  BLQ integration): 320 passed, 4 deselected (live-marked).
- `uv run pytest tests/unit/{test_cross_paradigm_ranking,test_gate_policy,test_gates,test_bundle_models,test_missing_data,test_credibility_report,test_loro_gate2,test_bundle_emitter,test_diagnostic_summarizer}.py` — 225 passed.
- `uv run pytest tests/property/ tests/golden/` — 39 passed (21 golden snapshots).
- `uv run python -m apmode.governance.validate_policies policies/` — 4/4 OK.
- Full `uv run pytest tests/ -q` (~11 min) deliberately not re-run this cycle;
  schema-touching targeted sweep covers the regressive surface.

**Next release plan (rc8 → backend VPC/simulation pipelines):**

The deferred "Full backend VPC/simulation pipelines" item is scoped and
planned for the next cycle. Approach:

- New shared module `src/apmode/backends/predictive_summary.py::build_predictive_diagnostics`
  — owns the canonical math from `(n_sims, n_subjects, n_timepoints)` arrays
  to `(VPCSummary, npe_score, auc_cmax_be_score, auc_cmax_source)` as a single
  atomic step, so the four diagnostics cannot drift apart across backends.
- Per-backend simulation steps, in dependency order:
  1. nlmixr2 R harness via `rxode2::rxSolve` — biggest single-step win,
     flips classical cross-paradigm ranking from CWRES proxy to real NPE.
     `src/apmode/r/harness.R` and `src/apmode/backends/r_schemas.py` extend
     to carry a typed `predicted_simulations` sub-schema.
  2. NODE posterior-predictive — Laplace-approximation sampling of
     input-layer random-effect weights (Bräm 2022), re-solve via the
     existing `_solve_multidose_eager`. Add to `node_trainer.py`.
  3. Bayesian Stan harness — largest effort; the harness file
     (`src/apmode/bayes/harness.py`) does not exist yet. Stan's
     `generated quantities` block provides predictive samples for free;
     reuse the `predicted_simulations` sub-schema.
- Policy migration: once nlmixr2 emits real scores, flip
  `policies/optimization.json::gate3.auc_cmax_weight` to ~0.3 (with
  vpc/npe dropping to 0.35/0.35). `policy_version` → `0.4.0`.
- Benchmark regression (#3 from rc7 briefing) becomes meaningful after
  at least one backend emits real NPE. Use a 3-cell same-commit A/B
  design, not the b1567aa-vs-c0fc317 worktree comparison that would
  confound weight-change with metric-definition-change.
- Open design questions deliberately surfaced rather than decided:
  `n_sims` default (proposed 500 per Bergstrand 2011, policy-versioned
  via `Gate3Config.n_posterior_predictive_sims`); non-uniform per-subject
  time grid handling (shared-grid + interpolate vs. per-subject lists);
  NODE Laplace vs. variational posterior if training already fits Ω;
  per-subject eligibility for AUC/Cmax (pooled today, per-subject is
  follow-up scope).

## [0.3.0-rc6] — 2026-04-16

### Checkpoint/resume, transform-parser fixes, and corrected PK defaults

A pre-release review identified four correctness bugs in the
checkpoint/resume path and one in the transform parser. All are fixed
in this release. The `_MODULE_DEFAULTS` fallback initial estimates
were corrected against nlmixr2 SOTA documentation.

**Checkpoint/resume (`--resume-agentic`):**

- `cli.py` — new `--resume-agentic` flag. Skips Stage 5 (classical SAEM search)
  and loads `classical_checkpoint.json` from the existing bundle directory. Use
  after an agentic API failure to restart the LLM loop without re-running the
  full multi-hour SAEM search.
- `orchestrator/__init__.py` — DAG lineage preservation on resume: when
  `_checkpoint_loaded=True`, the orchestrator now reads the existing
  `candidate_lineage.json` and seeds `lineage_entries` from it. Previously, the
  empty `SearchDAG()` default produced zero lineage entries, silently destroying
  the classical search history in the output bundle (gemini critical finding).
- `orchestrator/__init__.py` — duplicate `failed_candidates.jsonl` entries on
  resume: Gate 1 re-evaluates all loaded checkpoint candidates and
  `append_failed_candidate` appends; on resume this doubled every failure entry.
  Fixed by unlinking the file before the Gate 1 loop when `_checkpoint_loaded`.
- `orchestrator/__init__.py` — fail-fast on ambiguous bundle state: if
  `--resume-agentic` is requested but zero bundle directories exist, a
  `RuntimeError` is raised with an actionable message. If more than one directory
  exists, a `RuntimeError` names the count and asks the user to remove all but
  the target. Previously this silently fell back to a fresh run.

**Transform parser (`backends/transform_parser.py`):**

- `_parse_single_transform` — `swap_module` now accepts the LLM short-form
  `"new_module": "MichaelisMenten"` (bare string) in addition to the explicit
  dict form `{"type": "MichaelisMenten", ...}`. Previously the parser called
  `.get("type")` on a string and crashed.
- `_MODULE_DEFAULTS` — added fallback initial estimates for all 15 module types.
  When the LLM provides a short-form string without explicit parameter values,
  the defaults supply the required fields. LLM-provided values always override.
- `_MODULE_DEFAULTS` — corrected volume compartment ordering per nlmixr2
  official documentation and nlmixr2autoinit population-level fits:
  - `OneCmt.V`: 50 → **90 L** (nlmixr2 theophylline example; lVc = log(90))
  - `TwoCmt`: V1=30, V2=50 → **V1=50, V2=20, Q=2.0** (central > peripheral;
    nlmixr2autoinit shows Vc≈60 L, Vp≈10 L, Q≈1 L/hr for typical oral drug)
  - `ThreeCmt`: V1=20, V2=50, V3=100 → **V1=50, V2=20, V3=10** (central
    compartment is largest for most small-molecule drugs)

**Housekeeping:**

- `orchestrator/__init__.py` — removed dead `evidence: EvidenceManifest`
  parameter from `_run_agentic_stage` signature and its call site. The parameter
  was never consumed inside the method body (Pyright warning, pre-existing).
- `orchestrator/__init__.py` — renamed unused loop variable `_sr` → `_` in
  Gate 2 for loop (Pyright `not accessed` warning, pre-existing).
- `checkpoint: dict[str, object]` annotation in `_write_classical_checkpoint`
  narrowed to `dict[str, Any]` to satisfy `mypy --strict`.

## [0.3.0-rc5] — 2026-04-16

### Gate 3 cross-paradigm ranking: policy-driven composite, Borda, NPE unification

Lands Tier 1 + Tier 2 of the ranking cleanup flagged in `v0.3.0-rc5`'s
profiler-refinement commit.

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
  the preferred aggregator for cross-paradigm Gate 3 because
  likelihood scales are incomparable (PRD §10 Q2). The weighted-sum
  path is retained as the default for backward-compatibility-of-
  semantics with v0.3.0-rc4 benchmarks.
- Non-finite metric inputs (NaN / ±Inf) are now coerced to the worst
  value for their ordering direction before ranking: VPC → -Inf,
  NPE/BIC → +Inf. Python's unstable NaN sort can no longer land a
  broken candidate at rank 1.

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
see the change propagate.

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
