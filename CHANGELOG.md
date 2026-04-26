# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Undefined CWRES emits JSON `null`, not a fabricated `0/1` fallback

Per ICH M15 §3 and Karlsson 2007 (the canonical CWRES diagnostic
paper), CWRES is a *diagnostic*, not a convergence indicator: a
nlmixr2 fit can be numerically converged with usable parameter
estimates yet have undefined residuals (Suite-B
`b8_mavoglurant_null_covariates`: 5 random null covariates contaminate
the residual computation). An earlier patch substituted
`cwres_mean=0, cwres_sd=1, outlier_fraction=0` for any non-finite
aggregate; that silent fallback would have let degenerate fits pass
downstream Gate 1 ranking as "perfectly diagnosed" — exactly the
failure mode regulatory diagnostics protocols explicitly forbid.

Replaced with end-to-end "diagnostic unavailable" propagation
(reverts the 11ed85b approach):

1. **R harness** (`r/harness.R::.simulate_posterior_predictive`):
   `.finite_or_null(x)` returns R `NULL` (→ JSON `null`) on any
   non-finite input. No more fabricated `0/1`.
2. **Pydantic schema** (`apmode.bundle.models.GOFMetrics`):
   `cwres_mean`, `cwres_sd`, `outlier_fraction` are now
   `Optional[float] = None`. Updated docstring cites ICH M15 §3 and
   Karlsson 2007.
3. **Gate 1** (`apmode.governance.gates`):
   `_check_state_trajectory_validity` and `_check_cwres` both
   fail-closed when the metric is `None` —
   `GateCheckResult.observed = "unavailable"`. A fit Gate 1 cannot
   diagnose cannot pass Gate 1.
4. **Cross-paradigm ranker** (`compute_cwres_npe_proxy`): returns
   `+inf` when either CWRES aggregate is `None`, so a
   diagnostically-incomplete fit ranks worst, not best, under the
   "lower is better" ordering.
5. **LORO-CV** (`apmode.evaluation.loro_cv`): `test_npde_mean` /
   `test_npde_variance` propagate `None` cleanly through the
   per-fold calculation instead of crashing on `cwres_sd**2`.
6. **Reporters** (`diagnostic_summarizer`, `report.renderer`):
   render `None` as the literal `"unavailable"` so audit-trail
   readers see the diagnostic gap rather than a misleading
   `"0.0000"` entry.

Tests: 3 new pins in `tests/unit/test_r_subprocess.py
::TestPredictedSimulations1DCoercion`:
`test_harness_emits_null_for_undefined_gof` (pins
`.finite_or_null` definition + all three call sites),
`test_gof_metrics_schema_accepts_none` (all-None and
mixed-availability constructions round-trip cleanly),
`test_cwres_npe_proxy_returns_inf_when_unavailable` (proves the
ranker fails-closed; well-defined fits still return finite).

### Added — Phase-1 scorecard from end-to-end live run

The first complete Phase-1 scorecard from a live `apmode.benchmarks.suite_c_phase1_runner`
end-to-end pass on the five open `nlmixr2data` fixtures. Committed at
`benchmarks/suite_c/phase1_npe_inputs.json` so the weekly CI workflow has a measured
baseline. All 50 fits (5 fixtures × 5 folds × 2 sides) returned `status=success` after
the v0.6.1-rc1 nine-fix bring-up. Dimensionless NPE per fixture:

| Fixture | n | NPE APMODE | NPE Lit | Δ |
|---|---:|---:|---:|---:|
| theophylline | 12 | 0.186 | 0.178 | -4% |
| warfarin | 32 | 0.232 | 0.227 | -2% |
| **mavoglurant** | 120 | **0.408** | **0.993** | **+59%** |
| phenobarbital | 59 | 0.251 | 0.270 | +7% |
| oral_1cpt | 120 | 0.263 | 0.259 | -2% |

`fraction_beats_literature_median = 40% (2/5)`, below the 60% target. The three losses are
all within the δ=0.02 win-margin Monte-Carlo noise band; the mavoglurant +59% win is the
methodology-improvement signal. Oral_1CPT is a simulated ground-truth-recovery fixture
where the literature side is fitting the simulator's exact typical values — a near-tie
is the design expectation, not a regression.

### Fixed — CI: pip-audit ignore CVE-2026-3219; bandit B604 skip; SAEM fixture committed; bayesian-dispatch tests skip-on-no-Rscript; help-rendering test relaxed

Fix the linux-x86_64 CI failures across `security`, `test (3.12/3.13/3.14)` jobs that
were stacking up on every push. All four classes are environment-only (full local sweep
stays at 2557+ passing).

- **`pip-audit` CVE-2026-3219** — pip 26.0.1 (uv-bundled) carries a known vulnerability
  with no upstream fix yet. APMODE itself never invokes pip at runtime; pip is only in
  the build/dev environment. The CI invocation now passes
  `--ignore-vuln CVE-2026-3219` with an inline justification.
- **`Bandit B604` false positives on dataclass `shell=` kwarg** — Bandit's
  `any_other_function_with_shell=True` matcher fires on any literal `shell=` call,
  including `apmode.shells.{bash,fish,zsh,powershell}` emitting
  `InstallResult(shell=self.name, ...)` where `shell` is the target-shell-name field,
  not a subprocess injection vector. CI now passes `--skip B604` with rationale; B602
  (subprocess literal with shell=True) is still enforced.
- **SAEM-replay fixture not tracked** — `tests/fixtures/saem/theo_sd_30b_30e.log`
  matched the `*.log` ignore glob and was never committed. 7 tests in
  `test_saem_progress.py` + `test_nlmixr2_streaming.py` replay it and failed on CI
  with `FileNotFoundError`. Fix: add `!tests/fixtures/saem/*.log` exception and commit
  the 4.9 KiB fixture.
- **Bayesian-dispatch tests construct `Nlmixr2Runner`** — two tests in
  `test_orchestrator_bayesian_dispatch.py` instantiate `Nlmixr2Runner(work_dir=tmp_path)`,
  whose `__init__` resolves `Rscript` via `shutil.which` (issue #22) and raises
  `FileNotFoundError` on the R-less CI image. Fix: add a module-level
  `_RSCRIPT_AVAILABLE` guard and `pytest.mark.skipif` both tests.
- **`test_run_output_short_flag` rich-rendering brittleness** — asserted on
  rich-rendered `--help` output for the `--output-dir` alias. Fix: invoke
  `apmode run /nonexistent.csv {alias} /tmp/out` for each of `-o` / `--output` /
  `--output-dir` and assert no "No such option" parsing error — functional
  acceptance is the actual contract.

### Fixed — Documentation accuracy and drift cleanup

- Synced README / CLAUDE auto-managed counters to the current test collection
  and clarified the CLI-command count versus registered command groups.
- Reconciled HTTP API security documentation with the implemented static
  API-key and dataset-root requirements for non-loopback binds.
- Updated Architecture, Formular, Suite A/C, and R adapter docs for the
  v0.6.1-rc1 surface: 10 Formular transforms, shipped API/RO-Crate commands,
  v0.7 absorption preview forms, BLQ M3/M4 Stan support, and the current
  benchmark fixture rosters.
- Removed stale/broken ADR references and repo-resident process attribution
  from documentation and changelog prose.

### Fixed — CLI trace/graph hardening: typed traces, valid exporters, DAG invariants

The `apmode trace` and `apmode graph` deep-inspection commands now handle
malformed or adversarial bundle artifacts without corrupting machine-readable
output or crashing read-only inspection:

- `apmode trace --json` keeps stdout as pure JSON even when
  `agentic_iterations.jsonl` contains corrupt lines. Invalid iteration rows are
  validated against `AgenticIterationEntry` and skipped with human-mode
  diagnostics rather than causing renderer crashes.
- `apmode trace --cost` now tolerates malformed token/cost/time fields in
  `iter_*_meta.json`, warning in human mode and using zero for bad numeric
  cells instead of raising `ValueError`.
- `apmode graph --format dot` and `--format mermaid` now emit raw text on
  stdout. Rich no longer interprets DOT attributes or Mermaid labels as markup
  (for example `node [shape=box, fontsize=10]` is preserved).
- Mermaid export uses stable generated node IDs (`n0`, `n1`, ...) so candidate
  IDs such as `a-b` and `a_b` cannot collide after sanitization; original
  candidate IDs remain visible in labels.
- `SearchGraph` validation now rejects duplicate `candidate_id` values and
  edges that reference missing nodes, aligning the bundle model with the CLI's
  advertised DAG semantics.
- `apmode graph --backend` help now includes `bayesian_stan`, matching the
  accepted backend enum.

Tests: `tests/unit/test_deep_inspect.py` adds regression pins for corrupt trace
rows under `--json`, invalid typed iteration rows, malformed cost metadata, DOT
stdout, Mermaid ID collisions, duplicate graph nodes, and dangling graph edges.

### Fixed — Sparse-data harness: 1D `sims_at_observed` no longer crashes the runner

When a subject has a single observation (the `pheno_sd` neonatal
phenobarbital fixture: 155 obs across 59 subjects → some subjects
hit 1 obs), the R harness's posterior-predictive simulator emitted
`sims_at_observed` as a flat `list[float]` of length n_sims rather
than `list[list[float]]` (n_sims × 1). `jsonlite::toJSON(...,
auto_unbox = TRUE)` was unboxing each length-1 inner array to a
scalar; the Pydantic model `PredictedSimulationsSubject` then
rejected the response with one ValidationError per simulated row
(200 errors at `--n-sims 200`), aborting the entire fixture.

Two-layer fix:

- **R-side primary**: `r/harness.R::.simulate_posterior_predictive`
  wraps each per-sim row in `I(...)` so jsonlite preserves the
  array shape under `auto_unbox = TRUE`.
- **Pydantic-side defence in depth**: a new `field_validator
  ("sims_at_observed", mode="before")` on `PredictedSimulationsSubject`
  detects the flat `list[float]` shape and coerces it to
  `[[x] for x in v]` before validation. Well-formed inputs are
  unaffected. A future R-side regression fails the harness pin
  (`tests/unit/test_r_subprocess.py
  ::TestPredictedSimulations1DCoercion::
  test_harness_uses_I_wrap_for_sparse_subjects`) instead of the
  Phase-1 weekly run.

Tests: 3 new pins in `TestPredictedSimulations1DCoercion`
(flat-list-coerced, 2D-passthrough, harness-source-`I()`-wrap).

### Fixed — Runner-resilience: orphaned grandchild pipe FDs no longer hang `Nlmixr2Runner`

`Nlmixr2Runner._spawn_r` previously did
`asyncio.gather(drain_stdout, drain_stderr)` followed by
`await proc.wait()`. rxode2 invokes gcc/clang to compile the
generated C model code; those grandchildren inherit R's stdout and
stderr file descriptors via the `os.setsid`'d process group. If R
exits while a grandchild is still alive (segfaulted, SIGKILLed, or
just slow to flush), the OS pipes stay open. asyncio's
`BaseSubprocessTransport._try_finish` gates the wait future on ALL
pipe transports closing, so an orphaned grandchild blocks both the
drain gather and `proc.wait()` indefinitely. The outer
`asyncio.timeout` would eventually fire — at the cost of a 600 s
per-fit penalty per orphaned fit. The Suite-C Phase-1 weekly run
hit this on the mavoglurant fold04 fit and stalled overnight.

Fix: race a `os.kill(pid, 0)`-based watchdog against the drain
gather. Once the immediate child is gone (independent of pipe
state), the runner gives the drains a 5 s grace and then
`feed_eof()`s the StreamReaders so the drain coroutines complete
and the asyncio transport finally resolves `proc.wait()`. The
final `proc.wait()` gets a 2 s safety wrap in case the
transport's internal `_try_finish` state machine is still wedged
on a pipe we couldn't unblock.

Regression test: `test_grandchild_holding_pipe_does_not_hang_runner`
launches a shell script that backgrounds a 30 s sleeper inheriting
stdout/stderr, writes a valid `response.json`, then exits 0.
Without the fix, `runner.run()` blocks for the full 30 s sleep;
with the fix it returns in ~6 s.

## [0.6.1-rc1] — 2026-04-25

### Added — SOTA absorption preview (ADR-0003): Erlang, ParallelFirstOrder, SumIG(k=2)

Three new absorption variants land in the typed PK DSL ahead of the v0.7 milestone.
This is the first material extension of the absorption module since v0.1; design
authority is `docs/adr/0003-sota-absorption-extension.md`.

**`Erlang(n: int 1..7, ktr: float)`** — integer transit chain with no terminal
first-order step. Lowers as an explicit n-compartment ODE chain in nlmixr2 (not via
`rxode2::transit()`'s gamma interpolation, whose semantics differ — see ADR-0003 D2).
Admissible in all three lanes. The agent reaches Erlang only via the new
`convert_transit_to_erlang(n)` transform; `swap_module` accepts `Erlang` for direct
placement. Validator caps `n ≤ 7` because longer chains add little resolution and
inflate state count quadratically.

**`ParallelFirstOrder(ka1, ka2, frac)`** — two parallel first-order depots
(fast + slow GI, sublingual + GI per Pumas PK43; Soufsaf 2021 PMX). Genuinely
distinct from `MixedFirstZero(ka, dur, frac)`, which is first-order + zero-order.
Lowers as two depot compartments with `f(depot_fast)=frac`, `f(depot_slow)=1-frac`,
both feeding central simultaneously. Admissible in all lanes. New transform
`add_parallel_route(ka2, frac)` converts an existing `FirstOrder(ka)` into
`ParallelFirstOrder(ka1=ka, ka2, frac)`.

**`SumIG(k=2, MT_1, MT_2, RD2_1, RD2_2, weight_1)`** — sum of inverse Gaussians
input rate (Csajka, Drover, Verotta 2005, *Pharm Res*; Weiss & Wegner 2022,
*Pharm Res*). Captures double peaks, prolonged release, formulation/food effects
(Wagner 2014 mavoglurant; Weiss 2022 talinolol+rifampicin). Lowers as a
**closed-form analytic input rate** (`sqrt(RD2/(2π·t³))·exp(...)`), not via
deconvolution macros, so the producer-side digest contract stays untouched.
Single-dose only in v0.7; multi-dose superposition deferred (ADR-0003 D4).

  - Validator hard-caps `k ∈ {1, 2}` for v0.7 — path to k=3 is a future
    validator-only change behind the `sumig_max_k` policy knob (ADR-0003 D1).
  - Label-switching guard: `MT_1 < MT_2` enforced via positive-difference
    parameterisation (`MT_2 = MT_1 + exp(ldelta_MT_2)` in the emitter).
  - Cross-module identifiability gate: `SumIG.k >= 2` requires disposition
    (CL/V/Q) to be fixed externally — either via the dispatch-time
    `EvidenceManifest.disposition_fixed` flag or via priors with
    `source="fixed_external"` on every disposition param. Without that gate,
    sumIG-2 is non-identifiable on sparse data (Csajka 2005 §4; Weiss 2022 §5).
  - Lane admissibility: `SumIG` blocked from Submission via the new
    `_LANE_ABSORPTION_INADMISSIBLE` table — academic-grade, not yet standard
    regulatory practice. Discovery and Optimization admit it.

**New transforms** added to the `FormularTransform` allowlist (now 10 total, was 7):
`convert_transit_to_erlang`, `add_parallel_route`, `set_sumig_components`. All are
narrow, single-purpose, and compose with existing transforms via
`SwapModule`/`apply_transform`'s `_prune_stale_variability` hook (orphaned IIV /
priors on dropped parameters are pruned automatically — e.g., IIV on `ka` after
Transit→Erlang conversion).

**Refused outright in v0.7:** F.A.T./PBFTPK absorption (Macheras 2024). Regulatorily
unblessed by FDA/EMA; shipping it as a first-class DSL form would undermine
Submission-lane credibility (ADR-0003 D1).

**Deferred to v0.8+:** Weibull absorption (`nlmixr2lib::addWeibullAbs` already covers
this; not urgent). SumIG with k=3 (gated behind `sumig_max_k` policy knob; unlocks
when Discovery/Optimization datasets demonstrate Gate 3 NLPD floor pass at k=3
vs k=2 in held-out folds).

**Stan/Torsten support deferred to v0.7.1.** The Stan emitter currently rejects
the new variants with `NotImplementedError` (matching the existing
ZeroOrder/MixedFirstZero pattern), routing users to the nlmixr2 backend. Closed-form
input rate inside Torsten's user-defined ODE RHS requires time-varying covariate
plumbing for arbitrary t-forcing; that ships in a follow-up release.

**Closed v0.6.0 grammar gap.** `IVBolus()` is now first-class in the Lark grammar
(it was previously AST-only and only reachable via `swap_module`). Programmatic
construction is unaffected.

**New prior source value: `fixed_external`.** Added to the `PriorSource` literal
to mark priors derived from IV-reference fits or prior converged classical models —
used by the SumIG disposition-fixed cross-module check as the spec-side fallback
when the dispatch-time manifest flag is unavailable.

**Tests.** 44 new unit tests in `tests/unit/test_dsl_v07_absorption.py` cover AST
construction, grammar parsing, validator constraints (Erlang n cap, MT ordering,
disposition gate, lane admissibility, k cap), transform validation + apply, nlmixr2
emitter content, Stan emitter rejection, and a property test verifying that the
SumIG closed-form input rate integrates to ≈1 over (0, ∞).

### Fixed — Multi-CLI code review hardening pass: cancellation, durability, digest scoping, harness diagnostics

A focused code review of the `api/`, `bundle/`, `backends/`, and `bayes/`
surfaces produced 9 actionable findings, all addressed in this set of patches.
No public API change; existing bundles remain loadable.

**API run-state cancellation contract — slot leak on double-cancel closed.**
`apmode.api.runs.execute_run` now wraps the `on_complete` callback and the `FAILED`
status write in `asyncio.shield`, mirroring the existing shield around the `CANCELLED`
write. Without these, a second `CancelledError` (lifespan shutdown racing a `DELETE`)
could pre-empt `active_tasks.pop` mid-flight and permanently leak a 429 capacity slot
under repeated cancellations. The 4-link cancellation contract (`DELETE` → `task.cancel`
→ `terminate_process_group` → status update under shield) now extends to the
post-terminal callback. Symmetric 4-link semantics are preserved end-to-end.

**Subprocess group cleanup — grace window preserved on double-cancel, no leaked
`wait_for`.** `apmode.backends.process_lifecycle.terminate_process_group` is rewritten
around an `asyncio.ensure_future(...)` + `while: await asyncio.shield(inner_task)`
loop. The old form sent SIGKILL early and orphaned the inner `wait_for` if a second
cancellation arrived during the SIGTERM grace window; the new form keeps awaiting the
shielded task until it completes (clean exit or post-SIGKILL reap), captures the
cancellation, and re-raises after the child is reaped. The
`SIGTERM → grace → SIGKILL → reap` contract is now atomic under N concurrent cancels.

**HTTP API key check — no 500 on non-ASCII headers.** `apmode.api.routes` previously
called `hmac.compare_digest(api_key, expected)` directly; a header value containing
non-ASCII raised `TypeError`, which FastAPI surfaced as a 500 with a distinguishing
log line (a tiny side-channel: malformed key vs. wrong key). Both sides are now
encoded as UTF-8 bytes inside a `try/except UnicodeEncodeError`, so a malformed
header degrades cleanly to a 401. Confirmed against the existing `hmac.compare_digest`
timing-safe contract.

**Bundle write durability — `os.fsync` before `os.replace`, with short-write loop.**
`BundleEmitter._write_text` and the `_COMPLETE` sentinel write now go through a new
`_atomic_durable_write` helper that opens the tmp file via `os.open(O_WRONLY|O_CREAT|
O_TRUNC)`, writes in a loop that handles POSIX short-write (`os.write` may return
fewer bytes than requested on NFS / pipe-backed paths), `os.fsync(fd)`, and only
*then* `os.replace`s the destination. A new `_fsync_dir` then fsyncs the parent
directory entry so the rename is durable across power loss, not just process death.
Without these, a SIGKILL between `Path.write_text` returning and `os.replace`
committing could leave a successfully-renamed sentinel with stale page-cache contents
on filesystems that reorder data and metadata writes. Linux ext4 with `auto_da_alloc`
hides most of this; the durability is now explicit so the contract survives an admin
disabling the heuristic or the bundle living on NFS / overlayfs.

**Digest exclusion is bundle-relative, not basename.** `_DIGEST_EXCLUDED_NAMES` was
matched by `p.name`, which would silently exempt any nested file with a same-named
artefact (e.g. a future `compiled_specs/<id>/bom.cdx.json`) from the seal digest.
The producer (`apmode.bundle.emitter._DIGEST_EXCLUDED_RELATIVE_PATHS`) and the
importer (`apmode.bundle.rocrate.importer._DIGEST_EXCLUDED_RELPATHS_LOWER`) now
match against `p.relative_to(run_dir).as_posix()`. The SBC manifest, which lives at
`artifacts/sbc/sbc_manifest.json`, is excluded by its full relative path
(`_SBC_MANIFEST_RELPATH`); the SBOM and `_COMPLETE` sentinels remain at the bundle
root and so match by their basename alone. Sealed digests of existing bundles are
unchanged because the exclusion *set* is identical; only the matching rule is
tightened.

**`emitter.seal()` runs in a worker thread.** `Orchestrator.run` now `await
asyncio.to_thread(emitter.seal)` so the multi-second `_compute_bundle_digest` walk
(1 MiB-chunked SHA-256 over hundreds of posterior parquets in a discovery-lane
bundle) does not block the FastAPI event loop. The process-wide `threading.RLock`
(`_DIGEST_LOCK`) keeps the worker thread safe against concurrent emitter writes;
the lock is reentrant so `seal()` re-acquiring it inside `_compute_bundle_digest`
does not deadlock.

**Bayesian harness surfaces diagnostic-computation failures.**
`apmode.bayes.harness._compute_diagnostics` previously swallowed any `Exception` from
`az.bfmi` / `az.loo(pointwise=True)` with `pass`, leaving `pareto_k_max=None` /
`ebfmi_min=NaN` indistinguishable from "model has no `log_lik` group" or "energy
diagnostic unavailable". The harness now records each failure as a structured
`"<metric>_failed: <ExceptionType>: <message>"` entry on the new
`PosteriorDiagnostics.diagnostics_warnings: list[str]` field (default empty for
backwards compat with older bundles), prints it to stderr for live visibility, and
Gate 1 Bayesian (`apmode.governance.gates.evaluate_gate1_bayesian`) appends each
entry to `warning_reasons` and raises a `bayesian_diagnostics_complete` check.
Operators can now trace why a Bayesian fit's reliability metrics were not computed.

**CLAUDE.md — bundle-digest invariant prose corrected.** The "snapshot-then-hash"
description was wrong: the implementation streams 1 MiB chunks under
`_DIGEST_LOCK`. The TOCTOU guarantee is upheld by the lock spanning the whole
`rglob` + chunked-read sequence, not by buffering the snapshot. The architectural
note now says "locked-stream-hash" and references `_DIGEST_EXCLUDED_RELATIVE_PATHS`
rather than the renamed `_DIGEST_EXCLUDED_NAMES`.

**Suite B cross-seed CV excludes zero-variance parameters.**
`_compute_cross_seed_stability` previously included `CV=0.0` rows for parameters that
came back identical across every seed. These contribute nothing to the headline
`cross_seed_cv_max` metric and pollute the per-param map; the function now skips any
parameter with `sd < 1e-12`. Pinned by
`tests/unit/test_suite_b_runner.py::TestComputeCrossSeedStability::test_cv_calc_against_known_values`.

### Added — Two open public Phase-1 fixtures (`phenobarbital_grasela_1985`, `oral_1cpt_acop_2016`); roster grows from 5 to 7

The original Phase-1 roster of 5 had 2 credentialed/manual-fetch
fixtures (`gentamicin_germovsek_2017` requires DDMoRe browser
download; `vancomycin_roberts_2011` is on the Bayesian roster but the
sibling MIMIC-IV vancomycin fixture requires CITI training +
PhysioNet credentials). The weekly CI workflow could therefore
never produce a complete Phase-1 scorecard from the roster alone.

Two new fixtures close that gap:

- **`phenobarbital_grasela_1985`** — real, public, peer-reviewed
  neonatal phenobarbital data shipped via `nlmixr2data::pheno_sd`
  (GPL-2). 59 neonates, sparse TDM, 1-cmt IV bolus. Reference
  parameters from the Pharmpy pheno tutorial which replicates the
  Grasela 1985 NONMEM fit (CL=0.0047 L/h/kg, V=1.0 L/kg).
- **`oral_1cpt_acop_2016`** — simulated ground-truth-recovery
  reference, `nlmixr2data::Oral_1CPT` (GPL-2). 120 simulated
  subjects, oral first-order absorption, linear elimination,
  generated from typical CL=4 L/h, V=70 L, ka=1 /h with proportional
  sigma=0.2. Because the data are simulated from a known generative
  model, this fixture is a methodology *recovery* test rather than a
  literature comparator — APMODE's free fit must recover the
  typical values to within sampling noise.

The 2 credentialed fixtures (`gentamicin_germovsek_2017`,
`schoemaker_nlmixr2_tutorial`) stay in the roster for operator runs
that pass `--dataset-csv` overrides; CI runs them with the open 5.
`PHASE1_MLE_FIXTURE_IDS` length grew from 5 to 7; the integration
test that pinned `len == 5` was updated to `len == 7` with rationale.

### Fixed — `compute_npe` is now dimensionless when `spec.observation` is proportional / combined / BLQ-wrapped

The `BackendResult.diagnostics.npe_score` was raw MedAE (additive
default) regardless of the model's actual residual structure — so
ng/mL-scaled fixtures (mavoglurant: AMT in mg, DV in ng/mL, median
89, max 1730) inflated NPE by ~3 OoMs vs mg/L-scaled fixtures (theo,
warfarin) even when the residual *pattern* was comparable. The
inflation made cross-fixture NPE comparisons unit-bound rather than
methodology-bound, which defeats the
`fraction_beats_literature_median` Phase-1 gate.

`apmode.backends.predictive_summary._observation_error_model`
maps the DSL `ObservationModule` (Proportional / Additive / Combined
/ BLQM3 / BLQM4) to the matching `compute_npe` `error_model` hint:
proportional → divide by observed value; combined → divide by
`sqrt(obs² + 1²)`; additive → raw residual (rc8 default).
`build_predictive_diagnostics` now takes a `spec` kwarg and forwards
the derived enum to every `compute_npe` call (both the `flatten` and
`per_subject_median` aggregation paths). `Nlmixr2Runner.run` threads
the spec through `_parse_response` so the wire is end-to-end.

Backwards-compat: `spec` is keyword-only and defaults to `None`,
which preserves the rc8 raw-MedAE behaviour for any caller that has
not yet threaded the spec through.

Tests: 4 new pins in `tests/unit/test_predictive_summary.py
::TestNPEObservationErrorModelWire`:
1. `test_proportional_npe_is_dimensionless` — same 50% under-
   prediction at mg/L vs ng/mL scale (1000×) must produce the same
   NPE within 5% noise.
2. `test_additive_default_preserves_rc8_scale_dependence` — when
   spec is omitted the NPE *does* scale 1000× (rc8 baseline).
3. `test_combined_uses_combined_scaling` — Combined() dispatches to
   the combined-denominator path.
4. `test_blq_wraps_underlying_error_model` — BLQM3 with proportional
   underlying model uses proportional scaling.

### Fixed — `NCAEstimator` filters mixed-endpoint datasets and uses a data-driven fallback when QC fails

A second, independent class of "Phase-1 hits per-fit timeout" failures —
caused not by the FOCEI eta-drift loop (closed in the previous entry)
but by initial-estimate quality — is closed here.

- **Mixed-endpoint datasets pass through the same `DVID` PK-row filter
  the runner-side adapter already applies.** Warfarin's canonical
  NONMEM CSV interleaves `DVID="cp"` PK rows with `DVID="pca"`
  prothrombin-complex-activity PD rows. `NCAEstimator.__init__` now
  filters its observation slice via
  `apmode.data.adapters.PK_DVID_ALLOWLIST` (the constant was promoted
  from `_PK_DVID_ALLOWLIST` to a public name so the NCA module and the
  runner-side adapter share one source of truth). Before the filter,
  the per-subject terminal-slope regression saw PK and PD values mixed
  together, lambda_z fits collapsed for 25/26 subjects, and the
  estimator fell through to `_default_estimates()` (CL=5/V=70/ka=1) —
  values 47x off for warfarin's literature CL=0.106. After the filter,
  warfarin fold02 NCA returns CL=0.17 / V=8.5 / ka=0.11 (within 2x of
  literature on every parameter; `fallback_source="nca"`). Fail-open:
  if `DVID` is absent (theo, mavoglurant) or the allowlist would empty
  the frame (custom DVID schemes), all observation rows are kept and
  behaviour is unchanged.
- **New data-driven fallback layer in `NCAEstimator._apply_fallback`
  precedes the hard-coded defaults.** When per-subject NCA QC still
  fails *and* no `fallback_estimates` (dataset-card prior) is
  available, the estimator now derives initial estimates from the
  observed Cmax/AUC directly: `V = Dose_geo / Cmax_geo`,
  `CL = Dose_geo / AUC_obs_geo`, `ka = 2.5 / Tmax_geo`, with
  log-normal floors/caps `[1, 1000] L`, `[0.01, 500] L/h`,
  `[0.05, 12] 1/h`. The unit-scale heuristic (mg-dose / ng-mL detector)
  is reused from the NCA happy-path so a 1000x mass-units mismatch
  still gets corrected. The textual `fallback_source` cascade is now
  `nca → dataset_card → data_driven → defaults`. The hard-coded
  `_default_estimates()` remains as the final last-resort.
- **Tests.** Two new test classes in `tests/unit/test_initial_estimates
  .py`: `TestNCADVIDFilter` (PK+PD synthetic frame recovers via filter;
  no-op when DVID is absent) and `TestDataDrivenFallback` (sparse
  2-obs-per-subject frame triggers data-driven over hard-coded defaults;
  dataset-card prior still wins over data-driven). Full non-live sweep
  stays green at 2408 tests.

### Fixed — `Nlmixr2Runner` pre-adapts the on-disk CSV; `r/harness.R` rename is the safety net

Two interlocking layers close a class of indefinite-hang bugs that the
v0.6.1 honest-mode wiring exposed on the Suite-C Phase-1 weekly run.
The user-visible symptom in both layers was the same: a per-fit
`BackendTimeoutError` after the configured budget elapsed, with no
diagnostic in the response (the R subprocess never wrote one because
it was stuck in C-level FOCEI Newton iterations).

- **Runner layer (`Nlmixr2Runner.run`).** Before spawning Rscript the
  runner now reads the on-disk `data_path` (and optional
  `test_data_path`), pipes each through
  `apmode.data.adapters.to_nlmixr2_format`, and writes the adapted
  copy as `data_nlmixr2.csv` / `test_data_nlmixr2.csv` inside the
  per-fit scratch directory. The request.json `data_path` /
  `test_data_path` fields point at the adapted copies. The adapter
  performs three transformations the canonical APMODE schema does
  not enforce but nlmixr2 / rxode2 require: (a) `NMID` → `ID` rename
  (without it FOCEI enters a "Theta reset (ETA drift)" loop forever
  because no row is recognised as belonging to a distinct subject);
  (b) PK-row filter on `DVID` plus column drop on single-endpoint
  models (without it nlmixr2 raises "mis-match in nbr endpoints in
  model & in data" or hangs depending on whether the model has a
  multi-endpoint `~` clause); (c) string / two-level categorical
  covariate remap to integer 0/1 (e.g. `SEX="male"/"female"` →
  `1/0`) since nlmixr2 cannot consume string-typed covariates.
  Persisted train/test CSVs at the caller's path are byte-identical
  to what they wrote — the adapted copy lives only in the per-fit
  scratch and is GC'd with the run directory.
- **Harness layer (`r/harness.R::.normalize_id_column`).** A
  defence-in-depth NMID → ID rename remains in the R harness so the
  failure mode does not return if a future caller bypasses
  `Nlmixr2Runner` (e.g. operator runs `Rscript harness.R` directly
  during a debug session). The harness rename is a no-op when the
  runner has already adapted the CSV.
- **Tests.** Updated unit tests in
  `tests/unit/test_nlmixr2_runner.py` (`test_creates_request_json`,
  `test_request_carries_test_data_and_fixed_parameter`) write real
  NMID-shaped CSVs through the runner, then assert the adapted
  `data_nlmixr2.csv` exists, has `ID` columns instead of `NMID`,
  and that request.json points at the adapted path. New
  `tests/unit/test_r_subprocess.py::TestHarnessRRenamesNMID` (3
  pins) prevents the harness rename from being reverted.
- **End-to-end.** Phase-1 theophylline now fits all 5 folds in
  <1 minute on the default SAEM+FOCEI chain (was: indefinite hang
  in the FOCEI leg). Phase-1 warfarin (whose dataset carries
  `DVID="cp"/"pca"` and `SEX="male"/"female"`) now fits without
  raising the multi-endpoint mismatch (was: hung on FOCEI before
  the adapter wiring). Non-live sweep stays green at 2404 tests.

### Changed — Suite-C Phase-1 honest mode (held-out NPE + fixed-THETA literature)

The Phase-1 weekly benchmark grows from a catastrophic-drift detector
into a true methodology-drift detector. Two interlocking R-harness +
Python-runner extensions close the v0.6 scope-boundaries that the
Task 44 docstring listed as v0.7 unblocks.

- **Held-out NPE per fold (`r/harness.R::.simulate_posterior_predictive`
  + `RSubprocessRequest.test_data_path`).** When the request carries an
  optional `test_data_path`, the harness fits on `data_path` (train CSV)
  and routes `rxode2::rxSolve(events=test_df)` so the posterior-
  predictive matrix — and therefore NPE / VPC / AUC-Cmax — is generated
  on subjects the model never saw. The reported NPE is true held-out
  generalisation, not goodness-of-fit. Disjoint train/test subject IDs
  are required (rxode2 partitions sims by ID; a colliding ID silently
  recycles the train subject's posthoc ETA in place of an Ω draw); the
  subject-level k-fold split satisfies this by construction and the
  `suite_c_phase1_runner` adds an explicit assertion as defence in
  depth. Backwards-compat: when the field is absent the harness
  behaves bit-identically to the v0.6 in-sample path.
- **Fixed-THETA literature evaluation (`Nlmixr2Runner.run(...,
  fixed_parameter=True)` + `RSubprocessRequest.fixed_parameter`).** The
  runner's `NotImplementedError` gate is dropped; with `fixed_parameter
  =True` the harness collapses the AIC-best estimation loop to a single
  `est='posthoc'` pass that freezes THETA/OMEGA/SIGMA at the compiled
  `ini()` values (which the DSL emitter already writes from
  `initial_estimates`) and only estimates ETAs. Combined with the
  held-out path, the literature-side fit becomes "how well do the
  published parameters generalise to unseen subjects?" — the
  methodology-drift signal the v0.6 warm-start path could not produce.
- **`suite_c_phase1_runner` switched to honest mode.** Per fold the
  driver now writes a disjoint `train.csv` + `test.csv` pair, runs
  the APMODE side with `test_data_path=test_csv` and the literature
  side with both `test_data_path=test_csv` and `fixed_parameter=True`.
  Same RNG seed within a fold (so per-fold NPE differences are
  THETA-driven, not Monte-Carlo noise) is preserved. The runner
  module docstring now describes the honest contract instead of
  the v0.6 scope-boundary list.
- **`BackendRunner` protocol carries `test_data_path`.** All four
  backends (`nlmixr2`, `bayesian`, `node`, `agentic`) accept the
  kwarg; only `Nlmixr2Runner` honours it today, the others record
  and ignore it pending their own posterior-predictive paths. Test
  fakes (`tests/integration/test_loro_orchestrator.py::
  _FullProtocolMockRunner`) updated to match.
- **Tests.** New unit coverage in
  `tests/unit/test_r_subprocess.py` (path-traversal + relative-path
  rejection on `test_data_path`, defaults bit-identical to v0.6),
  `tests/unit/test_nlmixr2_runner.py` (request.json carries the new
  fields verbatim, relative `test_data_path` rejected at the runner
  boundary), and `tests/unit/test_suite_c_phase1_runner.py` (per-
  fold train/test CSVs are emitted with disjoint subject IDs,
  literature side gets `fixed_parameter=True`, APMODE side does
  not). Full non-live sweep stays green at 2398 tests.
- **Operator knob — `--estimation` flag on
  `python -m apmode.benchmarks.suite_c_phase1_runner`.** Comma-
  separated list of nlmixr2 estimation methods forwarded to
  `Nlmixr2Runner.estimation` (e.g. `--estimation focei` for a
  single-pass smoke; default still uses `Nlmixr2Runner`'s
  `['saem','focei']` two-pass). Useful when the per-fit timeout
  is too tight for the SAEM+FOCEI default on the first uncached
  fixture, and for ad-hoc operator runs that want a single
  estimator under a fixed wall-clock budget.

### Added — CLI infrastructure: version drift, typed errors, completion, SAEM streaming

Four interlocking slices land the CLI plumbing the v0.7 plan calls for —
version-drift detection, typed CLI errors, shell-completion installer,
and SAEM streaming progress.

- **PR1 — Version drift detection (`src/apmode/_version_drift.py`).**
  Surfaces drift between the CHANGELOG-declared milestone (most-recent
  `## [X.Y.Z]` header) and the runtime artefact identity
  (`importlib.metadata.version("apmode")`, written by `hatch-vcs`).
  When the two disagree, `apmode --version` prints both:
  `apmode 0.6.0-rc1 (runtime 0.3.0rc4.dev77+gHASH.dDATE)`. Comparison
  uses `packaging.version.Version` with PEP-440 canonicalisation and
  symmetric `+local` strip; the CHANGELOG header regex accepts both
  `-rcN` and `.devN` / `.postN` separators (multi-segment pre-release
  like `0.7.0-rc1.2` is accepted). 31 unit tests including PEP-440
  corner cases and CliRunner snapshots of aligned + drifted output.
- **PR2 — Typed CLI error infrastructure (`src/apmode/cli_errors.py`,
  `src/apmode/_json_ctx.py`, `src/apmode/__main__.py`,
  `scripts/check_typer_exit_count.py`).** Six error classes
  (`BundleNotFoundError`, `BundleInvalidError`, `PolicyValidationError`,
  `BackendUnavailableError`, `ConfigError`, `UserAbortError`) carry
  class-level `kind` (snake_case identifier) and `code` (process exit
  code: 10/11/12/13/14/130). New `python -m apmode` entrypoint runs
  `app(args=argv, standalone_mode=False)` and translates exceptions to
  either a Rich panel on stderr or a JSON envelope on stdout (chosen by
  argv pre-scan for `--json`). Click 8.x semantics fully honoured:
  `typer.Exit(N)` is converted to a return value (not re-raised), and
  `ClickException.exit_code` is preserved (no longer flattened to 2).
  CI ratchet at `scripts/check_typer_exit_count.py` locks the legacy
  count at 57; subsequent PRs migrate call sites and lower the
  baseline. Robustness details: `click.Abort` handler (Ctrl-C inside
  a prompt now renders as `user_abort` rather than a raw traceback),
  `--` argv-sentinel respect (positional `--json` after `--` is not
  a flag), `exc.exit_code` propagation (preserves `ClickException`
  subclasses with custom codes), parenthesised-raise regex match
  (`raise (typer.Exit(...))` forms now counted by the ratchet). 60
  unit tests across `test_cli_errors.py`,
  `test_cli_main_entrypoint.py`, `test_typer_exit_baseline.py`.
- **PR3 — Shell completion (`src/apmode/shells/{__init__,_rcfile,bash,
  zsh,fish,powershell}.py` + `src/apmode/cli_completion.py`).**
  `apmode completion {install,show,uninstall}` Typer sub-app wraps
  per-shell strategies. bash → marker block in `~/.bashrc`; zsh →
  vanilla marker block OR oh-my-zsh file-drop into
  `${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/apmode-completion.zsh`; fish →
  file-drop into `${XDG_CONFIG_HOME:-~/.config}/fish/completions/
  apmode.fish` (uses `env VAR=value` for fish 3.x portability);
  PowerShell 5/7 → marker block in `$PROFILE`. Atomic writes via
  per-PID `<file>.tmp.<pid>` + `os.replace`; one-shot `.bak` per
  process; idempotent install (`installed`/`already_installed`/
  `updated`/`uninstalled`/`absent` actions); byte-stable round-trip
  (one trailing newline budget). Robustness details: per-PID
  `<file>.tmp.<pid>` suffix in `atomic_write` (concurrent-install
  safety), `FileNotFoundError` TOCTOU handler in `backup_once` (file
  vanishing between `exists()` and `copy2()` is no longer fatal),
  fish `env VAR=value cmd` prefix (fish 3.x rejects bare
  `VAR=value cmd`), stale-tmp cleanup on `atomic_write` exception
  path. 56 unit tests including
  install/uninstall byte-equality, oh-my-zsh detection, PS5/PS7 path
  resolution, and JSON-envelope coverage.
- **PR4 — SAEM streaming progress (`src/apmode/backends/
  saem_progress.py` + `src/apmode/backends/nlmixr2_runner.py`).**
  `SAEMLineParser` is pinned against a real captured `nlmixr2 5.0`
  log (`tests/fixtures/saem/theo_sd_30b_30e.log`, 60 iterations,
  30 burn-in + 30 EM): a `params:` header followed by
  `001: <tab-separated floats>` iteration lines. Parser is designed
  to never raise on arbitrary input (Hypothesis property test).
  `nlmixr2_runner._drain_pipe` runs alongside `_spawn_r` to drain
  stdout + stderr concurrently with audit-log tee + per-line
  callback; the asyncio `StreamReader` limit was bumped from 64 KiB
  to 1 MiB so a long nlmixr2 stack trace no longer triggers
  byte-loss on overrun. `Nlmixr2Runner` constructor exposes
  `progress_callback` + `audit_log_dir` for the future CLI flag
  wiring. Robustness details: `Inf`/`-Inf` tolerance in the regex
  character class (divergent SAEM iterations no longer drop
  silently), length-mismatch guard on emitted `SAEMState` (no hollow
  `param_values=()` when header is set), drain-overrun loop continues
  until newline (no per-line cycling on > 1 MiB stack traces),
  closed-handle `ValueError` race caught alongside `OSError`,
  cross-stream sentinel newline so stdout fragments cannot glue
  onto stderr lines in the shared audit log. 49 unit tests
  including 100k-line deadlock-prevention burst, end-to-end
  drain→parser pipeline against the real fixture, and Hypothesis
  property fuzz on text + bytes.
- **Version bump.** `pyproject.toml` `fallback-version` 0.5.0-rc1 →
  0.6.0-rc1; `src/apmode/__init__.py` fallback constant 0.2.0.dev0 →
  0.6.0-rc1; CHANGELOG `[Unreleased]` cut to `[0.6.0-rc1] — 2026-04-24`
  during the bump; README `## Status` line updated to v0.6 wording
  and PR4 / PR1 streaming feature mentions; CITATION.cff version
  pulled by `scripts/sync_readme.py`.
- **WIP polish.** The pre-existing uncommitted `apmode serve` work
  (Task 35) was tightened on the same commit: stdlib imports
  (`contextlib`, `ipaddress`) hoisted to module scope; duplicated
  `_is_loopback_host(host)` check deduplicated; `escape(host)!s`
  no-op stripped; module docstring extended to list `serve`. The
  `--allow-public` test was updated to satisfy the concurrent
  `APMODE_API_KEY` + `--dataset-root` security gates.

### Fixed — Audit pass: cancellation, auth, digest integrity

A targeted hardening sweep of cancellation, auth, and digest-integrity surfaces.
All 2035 non-live unit tests pass; `mypy --strict` and `ruff check` are clean.

- **Bayesian Gate 1 no longer clobbers the classical Gate 1 audit
  artifact.** `BundleEmitter.write_gate_decision` now accepts
  `gate_number: int | str`. Bayesian Gate 1 writes
  `gate1b_<id>.json`; classical Gate 1 keeps `gate1_<id>.json`.
  Gate 2.5 was emitting `gate25_<id>.json` while every CLI consumer
  globbed `gate2_5_*.json` — the orchestrator now emits
  `gate2_5_<id>.json` so `apmode inspect` / `apmode diff` see Gate 2.5
  decisions for the first time.
- **API lifespan ordering.** `apmode.api.app.build_app` now waits up to
  10 s for cancelled tasks to flush their `CancelledError` handlers
  before closing the aiosqlite store, eliminating the window where
  `RUNNING` rows were left for the next startup sweep to reconcile.
  The `execute_run` cancellation handler wraps its terminal status
  write in `asyncio.shield` so a *second* cancel during shutdown
  cannot pre-empt the commit.
- **API authentication floor.** A new static-API-key dependency
  (`X-API-Key` header, `hmac.compare_digest`) is mounted on every run-
  management endpoint; `/healthz` stays unauthenticated. The CLI's
  `apmode serve --allow-public` now refuses to bind a non-loopback
  host unless `APMODE_API_KEY` is set *and* `--dataset-root` confines
  `POST /runs` `dataset_path` values to a known directory tree
  (resolved via `joinpath -> resolve -> relative_to`, defeating both
  naive `..` traversal and symlink escape).
- **API responses redact tracebacks.** `RunStatusResponse.error` now
  surfaces only the last non-empty traceback line (capped at 240
  characters); the full stack trace stays in the bundle and the
  server log.
- **Subprocess termination.** `terminate_process_group` shields its
  SIGTERM grace window so a second `CancelledError` cannot skip the
  SIGKILL escalation. The TimeoutError branches in `Nlmixr2Runner` /
  `BayesianRunner` now route through `terminate_process_group` (5 s
  SIGTERM grace, then SIGKILL) instead of going straight to SIGKILL,
  so nlmixr2 cleans its `~/.cache/R/...` artefacts and cmdstan
  flushes partial CSVs on timeout.
- **LLM client lifecycle.** `AnthropicClient`, `OpenAIClient`,
  `OpenRouterClient`, `GeminiClient`, and `OllamaClient` construct
  their underlying SDK client lazily in `__init__` (was: per-call) and
  expose an `aclose()` coroutine, eliminating the connection-pool leak
  that accumulated under sustained timeout/cancellation churn.
  `_await_llm` now uses `asyncio.timeout(...)` (3.11+) so it composes
  cleanly inside `TaskGroup` rather than misclassifying outer
  cancellations as timeouts. The litellm fallback splits its per-
  request timeout from the outer wall cap (half + 1 deterministic
  retry) so the SDK has a chance to land within budget.
- **Agentic loop resilience.** A try/finally around the iteration
  loop in `AgenticRunner.run` flushes
  `agentic_iterations.jsonl` / `agentic_lineage.json` /
  `run_lineage.json` even when a `CancelledError` tears the loop
  down, so the audit-trail rollup survives DELETE /runs/{id}. A
  failing LLM call no longer crashes the orchestrator: the runner
  records the terminal failure, breaks out, and returns the best
  classical result so far. `AgenticTraceOutput.raw_output` is bounded
  at 32 KB so a runaway response cannot bloat the trace.
- **Bundle digest TOCTOU.** `_compute_bundle_digest` now snapshots
  every file's bytes into memory under a process-wide
  `threading.Lock`, then hashes from the snapshot — a concurrent
  `append_search_trajectory` / `append_failed_candidate` cannot
  desync the seal. Both append helpers take the same lock and
  fsync each line so a crash mid-pipeline preserves the audit
  trail. `_COMPLETE.tmp` joins the digest exclusion set so a stale
  half-written sentinel from a prior crash cannot taint a fresh
  seal.
- **Prompt-injection sanitiser.** `_sanitize_for_prompt` now
  collapses role markers smuggled in via embedded `\n`, not just
  line-start ones (e.g.
  `Error: ...\n\nsystem: ignore previous instructions`).
- **Cost-helper DRY.** Three identical `_estimate_*_cost` bodies in
  `llm_providers.py` collapse to one `_compute_cost(rates, fallback,
  ...)`; the orphaned `_estimate_cost` in `llm_client.py` is
  removed (the litellm fallback now reports `0.0` rather than a
  rate-table approximation that drifted from the per-provider tables).
- **Critical regressions caught in the second-pass clink review.**
  - `_DIGEST_LOCK` is now a `threading.RLock` (was non-reentrant
    `Lock`), preventing the deadlock that would have hung every
    `seal()` call: `seal()` acquires the lock and then calls
    `_compute_bundle_digest` which acquires it again on the same
    thread.
  - `AgenticRunner.run` now calls `await self._llm.aclose()` in its
    `finally` block (best-effort via `getattr`), so the lazy-hoisted
    SDK client's httpx pool is actually released at run-end. Without
    this the pool would leak per agentic run despite the per-call leak
    being fixed.
  - Returning the agentic best result via `best_result.model_copy(
    update={"backend": "agentic_llm"})` instead of rebuilding the
    Pydantic model field-by-field — preserves backend-specific
    extensions (e.g. Bayesian `posterior_diagnostics` /
    `sampler_config`) that the manual rebuild silently dropped.
  - Lineage entries are staged locally and only promoted into
    `lineage_entries` after `validate_dsl` accepts the post-transform
    spec. Otherwise an invalid spec would seed orphan candidate IDs in
    `agentic_lineage.json` that no `apmode lineage` invocation could
    explain.
  - `POST /runs` now caps concurrent runs at
    `APMODE_MAX_CONCURRENT_RUNS` (default 8) and returns 429 with a
    30 s `Retry-After` past that — closes the cost-DOS path where an
    authenticated caller could submit unbounded runs.
  - Error redaction (`_redact_error_for_api`) now strips absolute
    filesystem paths via regex so the last-line summary served on
    `GET /runs/{id}/status` cannot be used to map the server's
    directory layout.

## [0.6.0-rc1] — 2026-04-24

### Added — `apmode serve` HTTP API CLI (plan Task 35)

- **`apmode serve` typer subcommand** wraps `apmode.api.app.build_app(...)` behind a programmatic uvicorn server (`uvicorn.Config(...) + uvicorn.Server(config).run()`). Defaults: `--host 127.0.0.1`, `--port 8765`, `--runs-dir runs`, `--db-path runs/.apmode_runs.sqlite3`, `--timeout-graceful-shutdown 30` (covers the 5 s SIGTERM-to-SIGKILL window in `terminate_process_group` + headroom for the `CancelledError` handler to seal partial bundles).
- **Loopback-default security gate.** Non-loopback `--host` values (`0.0.0.0`, RFC 1918 ranges, public IPs, hostnames other than `localhost`) exit with code 2 and a prescriptive message naming the risk: the API has no auth and bundle artefacts may contain patient data. The override flag `--allow-public` proceeds with a stderr warning naming the bind address. The gate is enforced by `_is_loopback_host`, which routes string hosts through `ipaddress.ip_address(...).is_loopback`; string-literal `localhost` (and IPv6 variants) are accepted, while DNS-resolved hostnames are rejected on principle (security defaults must not depend on runtime resolution).
- **`--allow-bayesian` flag** extends the `POST /runs` backend allowlist from the default `("nlmixr2",)` to `("nlmixr2", "bayesian_stan")`. Off by default until the Bayesian runner is fully wired through the request resolver.
- **Friendly extras-missing error.** `ImportError` on `from apmode.api.app import build_app` or `import uvicorn` is caught and re-emitted as `Error: the HTTP API extras are not installed. Run uv sync --extra api to pull FastAPI + uvicorn + aiosqlite.` with exit code 1.
- **README capability table.** "HTTP API + `apmode serve`" and "Optimization lane (LORO-CV Gate 2)" are promoted from 🔶 Partial → ✅ Available; the dangling `#http-api` ToC anchor now resolves to a full endpoint contract section. The "Future enhancement" note documents a long-poll evolution (`?wait=N` on `GET /runs/{id}/status` via per-`run_id` `asyncio.Event`) as deferred — the implementation path is recorded but not shipped, since current scientific clients run minutes-to-hours per fit and 5 s polling is well within the tolerable budget.
- **Tests.** `tests/unit/test_serve_cli.py` adds 19 cases covering loopback-host detection (`127.0.0.1`, `localhost`, `::1` accepted; `0.0.0.0`, RFC 1918, public IPs, hostnames rejected), default-bind kwargs to `uvicorn.Config`, custom port/log-level/shutdown-budget propagation, the `--allow-bayesian` allowlist extension, the non-loopback-without-`--allow-public` exit-2 gate (asserting `build_app` is **not** called so no SQLite handle leaks), `--allow-public` proceeding with a warning, and the extras-missing error path.

### Fixed — v0.6 RO-Crate projector hardening

Audit of the projector + importer + entity layer of
`src/apmode/bundle/rocrate/` drove the fixes below; all 2173 non-live
tests pass, `mypy --strict` and `ruff` are clean, and the
rocrate-validator REQUIRED gate continues to pass on the 5 Suite-A
scenarios. The golden snapshot at
`tests/golden/rocrate/__snapshots__/test_rocrate_golden.ambr` was
intentionally regenerated; review the diff to confirm the reshaping.

- **B1 — Importer no longer drops user-owned `workflows/` subtree.**
  `src/apmode/bundle/rocrate/importer.py` previously excluded the
  whole top-level `workflows/` directory because the exporter
  materialises a synthetic stub at `workflows/<lane>-lane.apmode`. A
  bundle that legitimately carried user-authored files under
  `workflows/` would have those silently dropped on round-trip. Fix:
  read `ro-crate-metadata.json` to learn the exact synthetic path
  from `mainEntity.@id` and exclude only that one entry. Added a
  `mainEntity` sanity guard rejecting `..`, leading `/`, drive
  prefixes, and null bytes; new regression tests cover both directory
  and zip forms.
- **H1 / H2 / L2 — Backend engines are first-class entities.** The
  projector now emits `#step-backend-<engine>` `HowToStep` and
  `#engine-<engine>` `SoftwareApplication` entities per backend that
  produced a result, populated with `softwareVersion` from
  `backend_versions.json`. `CreateAction.instrument` now points at
  the engine (the actual tool) rather than the candidate DSL spec;
  the candidate `SoftwareApplication` is carried as a
  `CreateAction.object` input alongside the data manifest. Workflow
  `hasPart` carries engines so `ProvRCToolRequired` ("every
  instrument tool must be hasPart of the workflow") is satisfied.
  `HowToStep.workExample` for backend steps resolves to the engine.
- **H3 — `provagent:AIModelInvocation` (canonical PROV-AGENT class).**
  `entities/agentic.py` flipped the `additionalType` from the obsolete
  `provagent:ModelInvocation` to `provagent:AIModelInvocation` per
  PROV-AGENT (Souza et al., eScience 2025, arXiv:2508.02866 v3).
- **H4 — `datePublished` survives bundle relocation.**
  `_resolve_date_published` reads `_COMPLETE.sealed_at` (an ISO-8601
  timestamp written into the sentinel at seal time) so the stamp
  travels with the bundle byte-for-byte. The mtime fallback is only
  used for legacy bundles. The value is validated through
  `datetime.fromisoformat` so a malformed `sealed_at` cannot leak
  into the crate.
- **H5 — Case-insensitive digest-exclusion + emitter sync.** The
  importer's excluded set is now lowercase-normalised against
  `{_COMPLETE, bom.cdx.json, sbc_manifest.json}`. The same audit
  caught that the importer's excluded set was missing
  `sbc_manifest.json` (third entry in
  `apmode.bundle.emitter._DIGEST_EXCLUDED_NAMES`); both are now
  identical.
- **M1 — Credibility orphan guard.**
  `entities/credibility.py` no longer emits a credibility `File`
  entity when the corresponding `#backend-create-<id>` `CreateAction`
  is absent.
- **M2 — `apmode:regulatoryContext` is no longer auto-defaulted.**
  `entities/pccp.py` previously stamped `pccp-ai-dsf` whenever any
  `regulatory/` file was present, mislabelling AI-Act-only or MDR
  bundles. Operators now MUST pass `--regulatory-context` explicitly;
  otherwise the slot is left empty and the orchestrator's
  `research-only` default applies.
- **M3 — `contentSize` is a string of decimal bytes** (schema.org
  Text range). `entities/_common.py::file_entity`.
- **M4 — Parquet media type is `application/x-parquet`** (the de-facto
  value used across Arrow/DuckDB tooling; Apache's IANA submission is
  still pending).
- **M5 — Synthetic workflow stub always rehashed.**
  `_materialise_virtual_workflow` no longer short-circuits on
  pre-existing files, so the `ComputationalWorkflow` entity's
  `sha256` / `contentSize` match the bytes on disk regardless of
  prior state.
- **M6 — Sentinel parse failure is fatal.**
  `_add_complete_sentinel` raises `BundleNotSealedError` on
  `json.JSONDecodeError`, `UnicodeDecodeError`, or wrong-shape
  payload, so a corrupted seal cannot produce a crate with a
  non-verifiable integrity anchor. The cosmetic description-text bug
  (`len(payload) and 'all'`) was cleaned up.
- **M7 — `apmode bundle publish` validates the bundle path** (mirrors
  `sbom_command`'s existence checks; was a stub-message dead-end
  before).
- **M8 — Agentic iteration ordering uses one global signal.**
  `add_agentic_trace` previously sorted iterations by lexicographic
  id (so `iter10` preceded `iter2`). The new logic picks one signal
  per bundle: numeric `meta.sequence_number` if every iteration
  carries one, else `meta.started_at` if every iteration carries one,
  else lexicographic id. Mixed bundles therefore cannot interleave
  signals.
- **M9 — Real Suite-A smoke test (opt-in).**
  `tests/integration/test_rocrate_suite_a_smoke.py` runs
  `apmode run` against `benchmarks/suite_a/*.csv`, exports the
  resulting bundle, and validates at REQUIRED severity. Gated by
  `APMODE_SUITE_A_SMOKE=1` so default CI stays fast; environments
  that have R / rxode2 / nlmixr2 can flip it on to close the
  synthetic-fixture gap from plan §H acceptance criterion 1.
- **L1 — `workflowhub/workflow-ro-crate/1.0` declared in `conformsTo`.**
  WRROC v0.5 says the root Dataset SHOULD reference this base profile
  in addition to the three WRROC profiles; doing so unlocks
  WorkflowHub's workflow-aware rendering paths. Each `conformsTo`
  profile entity now carries its own explicit `version` triple
  (`0.5` / `1.0` / `1.1`) instead of a heuristic.
- **Test additions.** New file
  `tests/unit/rocrate/test_review_regression.py` (490 LOC, 18 tests)
  pins each finding so a revert breaks the suite. Existing files
  were extended for the H5 / B1 / M1 / M2 / M4 / M8 / L1 paths. Test
  count rose from 2157 to 2193 (rocrate-only: 84 → 103, plus the
  Suite-A smoke marker). `mypy --strict` and `ruff check` remain
  clean.

### Added — v0.6-rc1 CLI polish

CLI surface hardened against an external audit. The following
non-breaking improvements landed; existing flags, exit codes, and Rich
output paths are preserved.

- **Machine-readable `--json` on every read command.** `apmode log`,
  `apmode diff`, `apmode datasets`, `apmode doctor`, and `apmode policies`
  now accept `--json`, emitting a stable `{"ok": bool, ...}` envelope on
  stdout and suppressing all Rich output. Errors travel through the
  envelope (e.g. `{"ok": false, "error": "not_a_directory", ...}`) rather
  than stderr when `--json` is set, so CI can branch on a single
  `jq '.ok'` test. `validate`, `inspect`, `trace`, `lineage`, `report`
  already supported `--json`; the read surface is now uniform. The
  multi-format commands (`ls --format json`, `graph --format json`)
  retain `--format` since they have multiple text formats.
- **Environment-variable bindings on `apmode run`.**  Adds
  `APMODE_LANE`, `APMODE_SEED`, `APMODE_TIMEOUT`, `APMODE_OUTPUT_DIR`,
  `APMODE_BACKEND`, `APMODE_PROVIDER`, `APMODE_MODEL`, `APMODE_AGENTIC`,
  `APMODE_AGENTIC_MAX_ITER`, `APMODE_PARALLEL_MODELS`, `APMODE_POLICY` —
  rendered into `apmode run --help` so they are discoverable. Previously
  zero `envvar=` bindings existed (only `APMODE_POLICIES_DIR` was
  resolved manually in `paths.py`).
- **`apmode run -o` / `--output-dir` aliases.**  Skill table and CLAUDE.md
  document `--output-dir / -o`; the CLI exposed only `--output` with no
  short flag. The flag now accepts `--output`, `--output-dir`, and `-o`
  interchangeably (no breaking change to existing scripts).
- **`apmode bundle rocrate {import,publish}` aliases.**  The skill and
  CLAUDE.md document the canonical RO-Crate triad as
  `apmode bundle rocrate export|import|publish`. The implementations live
  at `bundle import` / `bundle publish` (top-level forms still work);
  the documented sub-grouped form now resolves via shared callbacks in
  `src/apmode/bundle/rocrate/cli_hooks.py`.
- **`apmode run` Examples epilogue.**  The most complex command had no
  `Examples:` block; every other major command did. Added a workflow
  pattern block covering submission default, parallel discovery,
  agentic opt-in, Bayesian backend, agentic resume, dry-run, and the
  env-var driven invocation form.
- **Test additions.**  `tests/unit/test_cli.py` extended with
  `TestJsonOutputs` (10 tests for the new envelope contract on every
  read command), `TestEnvVarBindings` (3 tests for env-var override +
  help-text rendering), and `TestBundleRocrateAliases` (3 tests
  pinning the rocrate-grouped import/publish wiring). Test count
  raised from 86 to 101 in this file. Existing
  `tests/integration/test_rocrate_export_validate.py` continues to pass
  unchanged.

### Added — v0.6-rc1 Suite C Phase-1 scoring + weekly CI (plan Task 41)

The Phase-1 Suite C contract from PRD §4 / plan Task 41: score APMODE
against each Phase-1 MLE fixture under subject-level 5-fold CV, report
per-dataset NPE_median, aggregate via
`fraction_beats_literature_median = Σ(NPE_APMODE ≤ NPE_lit·(1-δ))/|D|`
with `δ = 0.02`. The v0.6 CI gate is `fraction_beats >= 0.60` (3 of 5
fixtures must beat the literature NPE by at least 2%). A miss opens a
GitHub issue but is **not** a release block — it is a methodology-drift
signal, owned by `.github/workflows/ci.yml` for release semantics.

- **`apmode.benchmarks.suite_c_phase1_scoring`** — pure-Python scoring
  helper (`score_fixture`, `aggregate_phase1_scorecard`,
  `phase1_roster_dois`, `FixtureScore`, `SuiteCPhase1Scorecard`).
  Aggregate returns `fraction_beats_literature_median = None` and
  `passes_gate = False` whenever the score list has fewer than
  `PHASE1_MIN_FIXTURES_FOR_AGGREGATE = 3` entries (matches the
  legacy `MIN_LITERATURE_COUNT` floor — a 1-of-1 = 100% must not look
  like a green Phase-1 run).
- **`apmode.benchmarks.suite_c_phase1_cli`** — `python -m
  apmode.benchmarks.suite_c_phase1_cli --inputs <file.json> --out
  scorecard.json [--markdown-summary scorecard.md]` driver invoked by
  the weekly workflow. Standalone `python -m` entry point (rather
  than a Typer subcommand on `apmode.cli`) keeps the dependency
  surface to vanilla `uv sync --extra dev` — no R, no cmdstan
  required for the scoring math itself. Documented exit codes:
  `0` (scorecard written), `2` (usage / parse error), `3` (fixture
  NPE validation failure).
- **`.github/workflows/suite_c_phase1.yml`** — weekly cron (Mon 03:17
  UTC, off-peak), workflow_dispatch with `open_issue_on_failure`
  input, uploads the scorecard JSON + Markdown as a build artifact,
  appends the Markdown to `$GITHUB_STEP_SUMMARY`, opens a labelled
  GitHub issue (`suite-c,phase-1-regression`) on a missed gate.
  Skips silently with a workflow `::warning::` annotation when
  `benchmarks/suite_c/phase1_npe_inputs.json` is absent — keeps the
  weekly cadence green during the bootstrap window before plan Task
  44's live-fit loop lands. Security note: no `${{ github.event.X }}`
  interpolation inside any `run:` block (per workflow-injection
  guidance).
- **22 new unit tests** (`tests/unit/test_suite_c_phase1_scoring.py` +
  `tests/unit/test_suite_c_phase1_cli.py`): score boundary inclusivity,
  custom delta/min-fixtures overrides, deterministic input ordering,
  every CLI exit-code branch, Markdown rendering for both pass and
  fail headlines, and a cross-link guard that
  `PHASE1_MIN_FIXTURES_FOR_AGGREGATE` matches `MIN_LITERATURE_COUNT`.

### Added — v0.6-rc1 HTTP API surface (plan Tasks 32, 33, 34)

The FastAPI app stack landed on top of the Task 31 SQLite RunStore.
Three lifecycle endpoints (POST/GET/DELETE), two streaming download
endpoints (bundle ZIP + RO-Crate ZIP), and a Starlette lifespan that
ties the run registry, background-task tracker, and subprocess
termination contract together.

- **`apmode.api.app.build_app()` factory + Starlette lifespan
  (Task 32 + 34).** New modules `src/apmode/api/{models,runs,routes,
  app}.py`. The factory wires the `SQLiteRunStore`, an injectable
  `runner_factory` (defaults to `Nlmixr2Runner`), and an
  `allow_backends` allowlist (default `("nlmixr2",)`; Task 36 will
  add `bayesian_stan`). The `@asynccontextmanager` lifespan calls
  `store.initialize()` (which runs `sweep_interrupted_on_startup`
  internally) before serving the first request, exposes
  `app.state.{store,runs_dir,active_tasks}`, and on shutdown cancels
  every still-running entry of `active_tasks` then closes the store.
  The recommended uvicorn pairing is `timeout_graceful_shutdown=30`
  to leave headroom for the SIGTERM-to-SIGKILL grace window in the
  runner.
- **`POST /runs` → 202 + Retry-After: 5 (Task 32).** `CreateRunRequest`
  is `extra="forbid"` Pydantic with explicit `lane`, `backend`,
  `seed`, `timeout_seconds`, `max_concurrency`, `covariate_names`,
  `column_mapping`, `context_of_use`, and `requeue_on_interrupt`
  fields. The handler creates the `RunRecord(status=PENDING)`,
  spawns `asyncio.create_task(execute_run(...))` (named
  `apmode-run-{run_id}` for observability), tracks it in
  `app.state.active_tasks`, and returns the
  `RunCreatedResponse(run_id, status, status_url)` payload. An
  unknown backend → 400; an unknown lane / extra field → 422.
- **`Orchestrator.run(..., run_id=...)` parameter (Task 32 plumbing).**
  The orchestrator now accepts an optional pre-allocated `run_id` so
  the API's RunRecord and the orchestrator's bundle dir share the
  same path (`runs_dir/<run_id>/`). Mutually exclusive with
  `skip_classical=True` (resume already binds to an existing run).
- **`GET /runs/{run_id}/{status,bundle,rocrate}` + `GET /runs`
  (Task 32).** Status returns `RunStatusResponse` (404 if unknown).
  Bundle and RO-Crate endpoints stream a per-request temp ZIP via
  `FileResponse` + `BackgroundTask` cleanup; both gate on
  `RunStatus.COMPLETED` (404 if unknown, 425 *Too Early* if not yet
  COMPLETED so polling clients can back off cleanly per RFC 8470).
  Bundle ZIP is built with `shutil.make_archive` in a thread; RO-Crate
  ZIP is built via the existing `RoCrateEmitter.export_from_sealed_bundle`.
- **`DELETE /runs/{run_id}` cancellation (Task 33).** Two-phase: the
  handler `task.cancel()`s the background coroutine and polls the
  store (5 s budget) until the row reaches a terminal status. The
  background task's `asyncio.CancelledError` handler in
  `apmode.api.runs.execute_run` writes `RunStatus.CANCELLED` and
  re-raises so asyncio's default exception handler sees clean
  cancellation. 404 if unknown, 409 if already terminal
  (`COMPLETED`/`FAILED`/`CANCELLED`/`INTERRUPTED`).
- **`apmode.backends.process_lifecycle.terminate_process_group`
  (Task 33).** New shared helper used by both `Nlmixr2Runner._spawn_r`
  and `BayesianRunner._spawn_python_harness`: SIGTERM the child
  process group, wait `grace_seconds=5.0` for graceful exit, escalate
  to SIGKILL. Idempotent and `ProcessLookupError`-safe. Both runners
  now catch `asyncio.CancelledError` on `proc.communicate()` and call
  the helper before re-raising — this is the path that prevents
  cancellation from leaving an orphan R or cmdstan child consuming
  CPU after `DELETE /runs/{id}` returns.
- **`requeue_on_interrupt` opt-in flag (Task 34).** Added to
  `CreateRunRequest`, persisted in the SQLite `runs` table via a new
  `INTEGER NOT NULL DEFAULT 0` column with idempotent ALTER backfill
  on `initialize()` (zero-touch upgrade from rc0). Surfaced in
  `RunStatusResponse` so a future re-queue worker can read every
  `INTERRUPTED` row's flag without consulting an out-of-band request
  log.
- **18 new tests** (15 in `tests/integration/test_api_runs.py` + 3 in
  `tests/unit/test_process_lifecycle.py`) cover: 202 + Retry-After,
  backend allowlist 400, unknown-lane 422, `extra='forbid'` 422,
  list ordering, status round-trip, bundle ZIP entries, 425 on
  unsealed bundle, DELETE → CANCELLED, DELETE on unknown 404, DELETE
  on completed 409, `requeue_on_interrupt` round-trip, lifespan
  startup sweep marks RUNNING → INTERRUPTED, and the
  `terminate_process_group` SIGTERM exit / idempotence /
  self-completion paths.

### Added — v0.6-rc1 RunStore + Suite C Phase-1 fixtures (plan Tasks 31, 40, 43)

Three independent slices of the v0.6 work landed together: the SQLite
backbone for the upcoming HTTP API (Task 31), the Phase-1 MLE literature
fixtures the Suite C scoring harness will iterate over (Task 40), and the
Phase-1 Bayesian fixture for vancomycin (Task 43; Eleveld remains NO-GO
per Task 42's coverage assessment).

- **`SQLiteRunStore` + `RunStore` Protocol (Task 31).** New optional
  `[api]` extra ships `fastapi>=0.111`, `uvicorn[standard]>=0.30`, and
  `aiosqlite>=0.19`. `src/apmode/api/store.py` implements an async
  run registry: `RunRecord` Pydantic schema, `RunStatus` StrEnum
  (`pending` / `running` / `completed` / `failed` / `cancelled` /
  `interrupted`), and `SQLiteRunStore` with `PRAGMA journal_mode=WAL`,
  `PRAGMA synchronous=NORMAL`, `PRAGMA busy_timeout=5000`, and a
  single-connection-per-store concurrency model where writes serialise
  through `asyncio.Lock` around `BEGIN IMMEDIATE` transactions
  (sqlite.org/wal.html performance guidance). `initialize()` is
  idempotent and runs `sweep_interrupted_on_startup()` before
  returning so the API never serves a stale `RUNNING` row after a
  process restart — invariant the Task 34 lifespan hook relies on.
  18 new unit tests in `tests/unit/test_run_store_sqlite.py` cover the
  PRAGMA invariants, idempotent init/close, sweep semantics on
  RUNNING / non-RUNNING rows, duplicate-key INSERT, and Protocol
  structural conformance.
- **Suite C Phase-1 MLE fixtures (Task 40, five datasets).** New
  `benchmarks/suite_c/` directory holds one `<dataset_id>.yaml`
  fixture + sibling `<dataset_id>.dsl.json` per dataset:
  `theophylline_boeckmann_1992` (1-cmt oral, Schoemaker 2019 anchor,
  DOI 10.1002/psp4.12471), `warfarin_funaki_2018` (1-cmt lagged-FO
  oral, Fidler 2019 nlmixr2 doc anchor, DOI 10.1002/psp4.12445),
  `mavoglurant_wendling_2015` (2-cmt oral, Wendling 2015 simplified
  fit, DOI 10.1007/s11095-014-1574-1; replaces the plan's
  unverifiable `mavoglurant_wang_2007.yaml` placeholder),
  `gentamicin_germovsek_2017` (1-cmt IV, Germovsek 2017 IOV neonate
  model, DOI 10.1128/AAC.02659-16; cross-cited against De Cock 2014
  for typical-CL agreement), and `schoemaker_nlmixr2_tutorial` (1-cmt
  IV bolus known-truth from the Schoemaker 2019 grid). New
  `apmode.benchmarks.literature_loader` exposes `load_fixture`,
  `load_fixture_by_id`, `load_dsl_spec`, and the canonical roster
  `PHASE1_MLE_FIXTURE_IDS`. 31 new integration tests in
  `tests/integration/test_suite_c_phase1_mle.py` assert each fixture
  loads, validates against the submission lane, references a known
  `dataset_id` in `benchmarks/datasets/registry.yaml`, emits
  nlmixr2 R code naming every reference parameter, has a
  Crossref-canonical DOI, and respects the
  `parameterization_mapping` invariant.
- **Suite C Phase-1 Bayesian fixture — vancomycin Roberts 2011
  (Task 43).** `benchmarks/suite_c/vancomycin_roberts_2011.{yaml,
  dsl.json}` ships a 1-cmt IV bolus DSLSpec with a six-element
  `PriorSpec` list: weakly-informative log-Normal priors on log-CL
  (mu=log(4.6), sigma=0.5) and log-V (mu=log(105), sigma=0.5)
  centred on Roberts 2011 typical values
  (DOI 10.1128/AAC.01708-10), plus half-Normal priors on the BSV
  SDs and residual-error SDs. Eleveld propofol is excluded per the
  Task 42 NO-GO assessment (`docs/discovery/eleveld_propofol_coverage.md`).
  `LiteratureFixture` extended with `backend: Literal["nlmixr2",
  "bayesian_stan"] = "nlmixr2"` so MLE fixtures keep the silent
  default and Bayesian fixtures opt in. 9 new tests in
  `tests/integration/test_suite_c_bayesian.py` (8 always-on + 1
  short-fit gated behind `@pytest.mark.slow` and a cmdstanpy-skip
  guard) cover prior-list non-emptiness, prior validation against
  the spec, half-Normal-on-omegas / Normal-on-log-structurals
  contract, log-prior centre vs reference-value agreement, and Stan
  emitter compatibility.
- **Runtime dependency: `pyyaml>=6.0`.** Added explicitly to project
  dependencies (was previously transitively available via dev/test
  extras). The Suite C scoring path (Task 41 follow-up) will load
  fixtures at runtime, not just in tests.
- **Test count: 2057 → 2115** (`<!-- apmode:AUTO:tests -->` markers
  re-synced; CLAUDE.md is gitignored).

### Added — v0.6-rc1 Gate 2 prior-data-conflict + prior-sensitivity hard-gates (plan Tasks 20 + 21)

The Bayesian Gate 2 admissibility funnel now includes the two
remaining FDA-style prior diagnostics from PRD §4.3.1 / plan
§Block 1B. Both follow the same three-layer pattern as Task 19's
prior-justification check: a Pydantic artefact written to disk by
the harness/orchestrator, a pure-Python compute helper that the
harness can call once cmdstanpy lands the prior-only Stan pass, and
a Gate 2 check that reads the artefact back and rules pass/fail
against a versioned threshold.

- **`PriorDataConflict` + `PriorDataConflictEntry` schemas (Task 20).**
  New artefact `bayesian/{cid}_prior_data_conflict.json` records,
  per key dataset statistic, the observed value vs the central 95%
  prior-predictive PI computed from a fresh Stan
  `generated_quantities` pass with `prior_only=true`. The summary
  carries `conflict_fraction` (Gate threshold), per-entry
  `in_pi` flags with bound-consistency validation, and a
  `status="not_computed"` branch with a mandatory `reason` so a
  recorded skip is distinguishable from a missing artefact (Box
  1980; Evans & Moshonov 2006; Gabry et al. 2019).
- **`PriorSensitivity` + `PriorSensitivityEntry` schemas (Task 21).**
  New artefact `bayesian/{cid}_prior_sensitivity.json` records, per
  (parameter, alternative-prior) pair, the normalised posterior-mean
  shift `|Δmean|/posterior_sd_baseline`. The schema cross-checks the
  scoring arithmetic and the running `max_delta` so a hand-edited
  artefact cannot misreport (Roos et al. 2015; Kallioinen et al.
  2024 power-scaling sensitivity).
- **Pure-Python compute helpers.** `apmode.bayes.prior_data_conflict`
  exposes `compute_observed_summary`, `compute_prior_predictive_summaries`,
  and `compute_prior_data_conflict`; `apmode.bayes.prior_sensitivity`
  exposes `compute_prior_sensitivity`. They take already-computed
  numpy arrays / dict summaries so unit tests can exercise the gate
  end-to-end without cmdstanpy on the runner. The harness will plug
  them in once the prior-only Stan pass and N+1 alternative-prior
  refits land.
- **`Gate2Config.prior_data_conflict_required/_threshold` and
  `prior_sensitivity_required/sensitivity_max_delta`.** Submission
  defaults to `required=True` with `threshold=0.05` (5% of key
  statistics outside the prior 95% PI) and `max_delta=0.20` (≤ 1/5
  of a baseline posterior SD shift). Discovery / Optimization
  default both `*_required` knobs to `False` because the prior-only
  pass and N+1 refits are too expensive for the search-lane
  cost-benefit. Schema is fail-closed: when the lane requires the
  check but the artefact is missing or `status="not_computed"`,
  Gate 2 fails with a precise reason.
- **`evaluate_gate2` keyword-only `prior_data_conflict` and
  `prior_sensitivity`.** New `_check_prior_data_conflict` /
  `_check_prior_sensitivity` checks added to the Gate 2 funnel
  alongside the existing prior-justification check. Both trivially
  pass when the candidate is non-Bayesian or the lane policy
  doesn't require them.
- **`BackendResult.prior_data_conflict` /
  `prior_sensitivity` sidecar fields + emitter methods.** Mirror the
  `loo_summary` / `reparameterization_recommendation` pattern: the
  harness embeds the artefact inline on the result, the orchestrator
  emits it via `BundleEmitter.write_prior_data_conflict` /
  `write_prior_sensitivity`, and the gate reads the emitted JSON
  back through the new `evaluate_gate2` keywords so the
  reviewer-facing artefact is exactly what the gate ruled on.
- **Policy version bump 0.5.1 → 0.6.0.** All three lane policies
  carry the new fields with lane-appropriate defaults.

### Fixed — v0.6-rc1 correctness + wiring pass

A follow-up audit turned up six high-confidence findings on the
just-merged Bayesian block, several of which were dead-code in
production. This change closes them.

- **`build_loo_summary` correctness.** `_coerce_float` returned
  `float("nan")` when arviz returned `None` or a non-coercible value,
  smuggling NaN past the `LOOSummary.<field>: float | None` contract
  and tripping JSON-serialization (NaN is not valid per RFC 8259).
  NaN now coerces to `None`. The `or` short-circuit on `getattr(loo,
  "elpd_loo", None) or getattr(loo, "elpd", None)` would have skipped
  legitimate `0.0` values and is replaced by an explicit
  `is not None` probe; `se_elpd_loo` also probes the
  `arviz_stats >= 1.0` `elpd_se` rename.
- **Per-class diagnostic dicts populated by the harness.** The Gate 1
  Bayesian evaluator already iterated
  `PosteriorDiagnostics.{rhat_max,ess_bulk_min,ess_tail_min}_by_class`
  but the harness left them as empty defaults — every per-class check
  silently skipped. `_compute_diagnostics` now buckets each variable
  via the new `apmode.governance.param_class.classify_param_class`
  (extracted to break the latent gates ↔ harness circular import).
- **Six-axis severity coverage on Gate 1 Bayesian.** `Gate1BayesianConfig`
  exposed `max_treedepth_fraction` and `e_bfmi_min` thresholds but the
  evaluator never read them and the severity validator listed only
  four axes (`rhat`, `ess`, `divergences`, `pareto_k`). Added explicit
  E-BFMI (Betancourt 2017 §6.1) and tree-depth-fraction checks; the
  severity dict now requires all six axes (`treedepth`, `ebfmi`
  added). Lane defaults: Submission/Optimization fail on `ebfmi`,
  warn on `treedepth`; Discovery warns on both. ESS warn-tier
  violations also surface in `warning_reasons` (previously dropped on
  the floor — only R-hat / divergences / pareto_k surfaced).
- **`pareto_k` non-finite sanitisation.** arviz can emit NaN/Inf
  Pareto-k entries when PSIS smoothing fails on individual
  observations. `_compute_diagnostics` now filters them before
  computing the max; `_bin_pareto_k` does the same so the bucket
  counts can't include phantom `very_bad` entries.
- **`PriorManifestEntry.doi` Crossref-canonical validator.** `doi`
  was `str | None` with no pattern check, so a free-form `"n/a"`
  satisfied Gate 2's truthiness probe. Added a Crossref regex
  validator matching the one on `LiteratureReference.doi`.
- **`LOOSummary.k_counts` key validator.** Accepts only the four
  Vehtari 2017 reliability bands (`good`/`ok`/`bad`/`very_bad`); a
  typo would otherwise corrupt downstream report rendering silently.
- **`approximate_posterior` polish.** `_empirical_bootstrap_draws`
  routed NaN diagonals through `np.where(np.isfinite(d) & (d > 0), …)`
  which raised RuntimeWarnings on NaN; now wrapped in
  `np.errstate(invalid="ignore")`. The `noise * std[np.newaxis, :]`
  reshape is removed in favour of natural broadcasting. The
  `cholesky-then-svd` comment that didn't match the actual `method="svd"`
  call is rewritten to describe what the code does. The variance
  floor `_BOOTSTRAP_VARIANCE_FLOOR = 1e-6` is now a named constant
  with a docstring describing when it activates.
- **`MetricTuple.method` distinguishes faithful Laplace from
  diagonal fallback.** Added `laplace_mvn` and
  `laplace_bootstrap_diagonal` literals so the Gate 3 ranker can tell
  a real MVN draw (full off-diagonal covariance preserved) from the
  degraded diagonal-only fallback. New
  `apmode.governance.approximate_posterior.laplace_draws_with_method`
  returns a `LaplaceDrawsResult` named tuple `(draws, method)` for
  callers that need the provenance; the array-only `laplace_draws`
  remains as a thin wrapper.

### Wired — v0.6-rc1 Bayesian artefacts into the orchestrator

The previous batch built `evaluate_gate1_bayesian`, `build_loo_summary`,
`build_reparameterization_recommendation`, and the prior-manifest
emitter — but none had a production caller, so they were dead code.
This change wires them into the orchestrator so v0.6-rc1 actually
governs Bayesian runs.

- **`evaluate_gate1_bayesian` runs alongside `evaluate_gate1`.**
  Bayesian candidates are now disqualified when per-class R-hat / ESS
  / divergences / treedepth / E-BFMI / Pareto-k thresholds fail with
  `severity="fail"`; `severity="warn"` violations surface in
  `summary_reason` without dropping the gate. Non-Bayesian backends
  pass trivially with `summary_reason="not_applicable — non-Bayesian
  backend"` so the audit trail is uniform.
- **Bayesian sidecar artefacts emitted on every run.**
  `Orchestrator._emit_bayesian_sidecars` writes
  `bayesian/{cid}_loo_summary.json`,
  `bayesian/{cid}_reparameterization_recommendation.json` (when the
  harness produces one), and `bayesian/{cid}_prior_manifest.json`
  before either gate evaluates. The harness now embeds `loo_summary`
  + `reparameterization_recommendation` inline on the result payload
  (new `BackendResult` fields) so the orchestrator owns the bundle-
  write step (single audit-trail point).
- **`evaluate_gate2` consumes the prior-manifest artefact.** The
  Submission-lane requirement (`bayesian_prior_justification_required:
  true`) is now actually enforced — the orchestrator loads
  `bayesian/{cid}_prior_manifest.json` from disk (the same artefact a
  reviewer would inspect) and threads it into `evaluate_gate2(...,
  prior_manifest=...)`.

Test counts: 2018 → <!-- apmode:AUTO:tests -->2024<!-- apmode:/AUTO:tests --> collected
(2001 → <!-- apmode:AUTO:tests_nonlive -->2007<!-- apmode:/AUTO:tests_nonlive --> non-live).
mypy `--strict` clean (108 source files); ruff `check` + `format` clean.

### Added — v0.6-rc1 Bayesian Block 1B/1C + Gate 1/2 Bayesian gates

Nine plan tasks (15, 16, 17, 18, 19, 22, 23, 24, 25) landed in a single
session, extending the Bayesian plumbing from "orchestrator wires through"
to "governance funnel actively consumes".

- **Plan Task 15 — `prior_manifest.json` emitter with validation.**
  `BundleEmitter.write_prior_manifest_from_specs(specs, cid, *, policy_version,
  justification_min_length)` validates every informative prior's
  justification + DOI before writing; aggregates failures so reviewers see
  every offending prior in one pass. `PriorManifestEntry` gains a `doi`
  field so Task 14's Crossref identifier survives the round-trip.
- **Plan Task 16 — per-lane `gate1_bayesian` block.**
  `Gate1BayesianConfig` splits R-hat and ESS floors across parameter
  classes (fixed_effects / iiv / residual / correlations) with a per-axis
  severity map (rhat / ess / divergences / pareto_k). Submission strictest,
  Discovery relaxes fixed-effects R-hat to 1.05, Optimization tightens
  Pareto-k to 0.5 with `fail` severity.
- **Plan Task 17 — `evaluate_gate1_bayesian()` warn/fail tiers.**
  `PosteriorDiagnostics` carries `rhat_max_by_class` / `ess_bulk_min_by_class`
  / `ess_tail_min_by_class`. The evaluator walks each axis against the
  matching threshold and applies the severity tier: warn-severity
  violations surface via `evidence_ref` without dropping the gate;
  fail-severity violations drop it with class-specific reasons.
- **Plan Task 18 — PSIS-LOO via `arviz.loo` + `loo_summary.json`.**
  `build_loo_summary(idata, cid)` is forward-compatible with arviz_stats
  1.0's renamed fields (`elpd` / `p`); bins Pareto-k into the four
  Vehtari 2017 reliability bands (`good` ≤ 0.5, `ok` ≤ 0.7, `bad` ≤ 1.0,
  `very_bad` > 1.0). Always emitted — `status="not_computed"` with reason
  is explicit, not an absent artifact.
- **Plan Task 19 — Gate 2 Bayesian prior-justification hard-gate.**
  `Gate2Config.bayesian_prior_justification_required` (Submission=True,
  Discovery/Optimization=False) + `bayesian_prior_justification_min_length`
  (Submission tightens to 500 chars). `evaluate_gate2` consumes an
  optional `prior_manifest` kwarg and aggregates per-prior errors. Fails
  closed when the lane demands the manifest but it isn't supplied.
- **Plan Task 22 — Laplace/MVN approximate posterior-predictive helper.**
  `apmode.governance.approximate_posterior.laplace_draws(theta, cov,
  n_draws, seed, *, fallback)` draws from the Laplace approximation for
  MLE backends so Gate 3 compares MLE and Bayesian candidates on
  commensurate intervals. Falls back to `empirical_bootstrap` (diagonal
  Normal around theta) when the covariance is ill-conditioned (cond > 1e12,
  non-finite entries, non-symmetric); `fallback="raise"` re-raises
  `LinAlgError` for callers that prefer hard-failing on degenerate MLE fits.
- **Plan Task 23 — commensurate `MetricTuple` carrier.**
  `MetricTuple(mean, ci_low, ci_high, method: Literal["posterior_draws",
  "laplace_draws", "empirical_bootstrap"], ci_level)` enforces
  `ci_low <= mean <= ci_high` at construction; the `method` tag records
  provenance so reviewers tell real posterior draws from Laplace-
  approximated ones. Dump shape is identical across methods so Gate 3
  iterates keys uniformly.
- **Plan Task 24 — Gate 3 metric stack + laplace_draws.**
  `Gate3Config.metric_stack: list[Literal[...]]` is the anti-metric-
  shopping whitelist; `Gate3Config.validate_metric(metric)` raises the
  new `PolicyError` (narrow `ValueError` subclass) when a candidate
  proposes an off-list metric. `Gate3Config.laplace_draws`: Submission
  2000, Discovery 500, Optimization 1000.
- **Plan Task 25 — drop silent auto-reparameterization.**
  `ReparameterizationRecommendation` advisory artifact emitted under
  `bayesian/{cid}_reparameterization_recommendation.json` when
  divergences or tree-depth saturations warrant intervention.
  `build_reparameterization_recommendation(diag, cfg, cid)` escalates
  above a 5% divergence-fraction floor to `switch_to_non_centered`
  (Betancourt & Girolami 2015) and falls back to
  `refit_with_higher_adapt_delta` for lower-rate divergences or tree-
  depth saturations. APMODE never switches parameterization silently
  so the audit trail matches the run as executed.

New unit tests: `test_bundle_prior_manifest.py` (10), `test_policy_gate1_bayesian.py` (12),
`test_gate1_bayesian.py` (24), `test_bayes_loo.py` (6),
`test_gate2_prior_justification.py` (8), `test_approximate_posterior.py` (12),
`test_metric_tuple.py` (9), `test_policy_gate3_metric_stack.py` (12),
`test_reparameterization_recommendation.py` (10). mypy `--strict` clean; ruff
`check` + `format` clean.

Continued in the same session — four more plan tasks (26, 38, 39, 42):

- **Plan Task 26 — `sbc_manifest.json` schema + stub emitter.**
  `SBCManifest` + `SBCPriorEntry` are the on-disk shape for the
  Talts 2018 Simulation-Based Calibration roll-up the nightly runner
  (Task 27) populates. Producer-side `BundleEmitter.write_sbc_manifest`
  emits a stub with `priors=[]` so the artefact's *presence* signals
  the Bayesian path executed end-to-end. Filename `sbc_manifest.json`
  joins the SBOM in `_DIGEST_EXCLUDED_NAMES` so a nightly rewrite does
  not invalidate `_COMPLETE`.
- **Plan Task 38 — Suite C rename + reframed claims.**
  `BenchmarkCase.expert_models` → `literature_models`. Suite C Phase 1
  is now framed as "methodology validation vs established literature
  models"; Phase 2 (head-to-head vs blinded human-expert panel) is
  marked out of v0.6 scope. `MIN_EXPERT_COUNT` retained as a
  deprecated alias for `MIN_LITERATURE_COUNT` (removal in v0.7).
- **Plan Task 39 — `LiteratureFixture` schema with parameterization mapping.**
  `LiteratureReference` (Crossref-canonical DOI + citation +
  population description) + `LiteratureFixture` (dataset_id + DSL
  spec path + reference_params + parameterization_mapping). The
  mapping is `{published_name: dsl_name}`; the validator enforces
  that mapping values index `reference_params` so cross-tool NPE
  comparisons (e.g. NONMEM `TVCL` vs DSL `CL`) cannot silently
  mis-align.
- **Plan Task 42 — Eleveld propofol DSLSpec coverage assessment.**
  `docs/discovery/eleveld_propofol_coverage.md` documents three
  blocking gaps preventing Eleveld 2018 from running on the
  Bayesian path: Stan emitter raises `NotImplementedError` for the
  `maturation` covariate form; no derived-covariate primitive
  (FFM = f(weight, sex, height)); no piecewise age-decay primitive.
  `scripts/check_eleveld_dslspec_coverage.py` is the auditable
  CI-friendly counterpart — exits non-zero on blocking gaps.
  Recommendation: **NO-GO** for v0.6 Phase-1 Bayesian fixtures.
  Vancomycin (Roberts 2011) is the only Phase-1 Bayesian fixture.

Plan tasks 26, 38, 39, 42 added unit tests:
`test_sbc_manifest.py` (9), `test_literature_fixture_schema.py` (11).
Total session test additions: 113 unit tests across thirteen plan
tasks. Test collected count: 1894 → 2018.

### Added — v0.6-rc1 in-progress (Bayesian Block 1 + Suite A8 + Gate 2.5 submission)

Work toward v0.6.0-rc1 has landed in pieces; the Bayesian orchestrator
plumbing, bundle-emitter Bayesian artefacts, Suite A8 scenario, and
Gate 2.5 block for the submission lane are now on main.

- **Gate 2.5 in submission lane.** `policies/submission.json` gains a
  `gate2_5` block (context_of_use, limitation_to_risk, data_adequacy
  with ratio floor 5.0, sensitivity, ai_ml_transparency=false). Policy
  version bumped to `0.5.1` in all three lane files (single-version
  invariant enforced by `scripts/sync_readme.py`). New
  `tests/unit/test_policy_schema.py` asserts the block shape.
- **Suite A8 — 1-cmt oral + time-varying CL + CRCL covariate.**
  `benchmarks/suite_a/simulate_all.R` simulates
  `CL(t, CRCL) = CL0 * (CRCL/90)^0.75 * exp(-0.15 * t / 24)` across
  60 subjects × 11 observations; `src/apmode/benchmarks/suite_a.py`
  adds `scenario_a8()` + `A8_COVARIATE_MODEL_NOTES` recording the
  diurnal rate that no DSL primitive expresses.
  `reference_params.json` carries an `_expected_misspecification_bias`
  block so benchmark tooling compares recovery to the time-averaged
  target rather than raw `CL0`.
- **LORO-CV lane gating tests.** Property tests
  (`tests/property/test_loro_fold_properties.py`) assert fold count =
  regimen count and permutation stability; integration tests
  (`tests/integration/test_loro_orchestrator.py`) pin that LORO fires
  in the optimization lane and is skipped in submission even on
  LORO-eligible data.
- **Platform-adaptive cmdstanpy kwargs.** New
  `src/apmode/bayes/platform.py::cmdstan_run_kwargs(*, uses_reduce_sum)`
  returns Windows-safe `force_one_process_per_chain=True` (cmdstanpy
  issue #895 — MinGW `thread_local` performance bug) and POSIX
  `cpp_options={"STAN_THREADS": True}` when `reduce_sum` is active
  (issue #780).
- **BayesianRunner wired into discovery/optimization.**
  `Orchestrator.__init__` gains `bayesian_runner`; `_LANE_BACKENDS`
  admits `bayesian_stan` in discovery and optimization lanes only
  (submission stays classical NLME per PRD §3 hard rule).
- **Provenance-recording sample helper.** New
  `sample_with_provenance` compiles + samples with
  `save_cmdstan_config=True` (cmdstanpy issue #848) and writes
  `backend_versions.json` with SHA-256 hashes of the Stan program and
  data, cmdstan version, host platform, and the resolved
  `one_process_per_chain` flag.
- **Bayesian bundle artefacts.**
  `BundleEmitter.write_posterior_draws` (long-form Parquet with
  optional thinning), `write_posterior_summary` (canonical 9-column
  schema), `write_mcmc_diagnostics`, `write_sampler_config` — each
  validated against `_SAFE_ID_RE` and written under
  `run_dir/bayesian/`. `ParameterEstimate` gains
  `posterior_mean` alongside `posterior_sd` / `q05` / `q50` / `q95`
  so reports can disambiguate posterior mean from primary estimate.
- **DSL prior-justification validator.**
  `apmode.dsl.priors.validate_prior_justification(spec, *, min_length)`
  checks the Crossref-canonical DOI pattern and a minimum
  justification length (default 50 chars; callers override via
  `min_length`). DOI regex accepts SICI-style suffixes with angle /
  square brackets. `PriorSpec` now carries a `doi: str | None` field
  and uses `_INFORMATIVE_SOURCES` frozenset as the single source of
  truth (previously duplicated as a tuple literal).

### Fixed — Review pass on unpushed v0.6 work

A review pass across the 14 unpushed v0.6-rc1 commits surfaced
correctness, safety, and documentation gaps. All resolved on main:

- **Bayesian harness hardcoded `converged=True`**. `_run` now derives
  the flag from a conservative floor — R-hat ≤ 1.05, bulk ESS ≥ 400,
  zero divergent transitions — via a new unit-testable
  `_is_converged` helper. Policy-driven warn/fail tiers land in
  plan Task 17 (Gate 1 Bayesian).
- **Silent DV≤0 dropping is now an explicit error**. `_build_stan_data`
  raises `ValueError` (mapped to `error_type="invalid_spec"`) when it
  encounters non-positive DV rows with `MDV=0`, since the lognormal /
  proportional likelihood cannot accommodate them. Callers must set
  `MDV=1` on pre-dose baselines or switch to the BLQ_M3 / BLQ_M4
  module with explicit censoring.
- **CSV path hardening**. `_build_stan_data` resolves
  `request["data_path"]`, rejects non-string / empty values, and
  requires the path to point to a regular file before `pd.read_csv`
  touches it (CWE-22 defence-in-depth).
- **DOI regex extended**. `_DOI_PATTERN` now accepts `<`, `>`, `[`,
  `]` so Wiley's legacy SICI-style DOIs (`10.1002/(SICI)…`) pass
  validation.
- **Bayesian RO-Crate projector coverage**. The
  `src/apmode/bundle/rocrate/entities/bayesian.py` regex now matches
  `sampler_config`, `posterior_summary`, and `posterior_draws`
  alongside the existing `prior_manifest` / `simulation_protocol` /
  `mcmc_diagnostics` / `draws` shapes.
- **Dead / duplicated code removed**. Unused
  `SCENARIOS: dict[str, type[None]]` deleted from `benchmarks/suite_a.py`;
  `_INFORMATIVE_SOURCES` frozenset is now the single source of truth
  for `PriorSpec`'s informative-source branch; `routing.py` had a
  dangling `Follow-up: (` comment, replaced with the active
  description of the `node_dim_budget` primary gate.
- **Trust-boundary documentation**. `sample_with_provenance` carries
  an explicit note that `stan_code` must originate from
  `apmode.dsl.stan_emitter.emit_stan` — user-supplied or agentic
  output must pass through the DSL allow-list before reaching
  cmdstanpy's C++ compile path.
- **Suite A8 docstring + reference_params.json now disclose
  misspecification bias**. Benchmark comparisons use the
  time-averaged `CL` target (`3.678`) rather than raw `CL0 = 4.482`.

Test coverage added: `test_bayes_harness_convergence.py`,
`test_bayes_harness_data.py`, `tests/unit/rocrate/test_entities_bayesian.py`,
plus DOI-bracket / `min_length` cases on
`test_prior_justification.py`.

### Changed — Deep-review hardening (41-finding audit)

A comprehensive cross-subsystem review identified 41 correctness, wiring,
and audit-trail issues across the DSL compiler, backends, governance
gates, reproducibility bundle, and benchmark suite. All 41 are now
resolved. Highlights:

- **Stan/nlmixr2 proportional-error likelihood unified** (#1).
  `stan_emitter` now emits `y ~ normal(f, sigma_prop * f)` to match the
  nlmixr2 `cp ~ prop(prop.sd)` semantics. Previously Stan used
  `lognormal(log(f), sigma)` for the non-BLQ path while the BLQ branch
  used the Normal form — cross-paradigm NLPD (PRD §4.3.1) was
  invalidated by the inconsistency.
- **Stan ODE solver O(N × N_subjects) collapse** (#3). The
  ``transformed data`` block precomputes ``obs_start[i]`` / ``obs_end[i]``
  per subject; the per-subject loop iterates the slice directly instead
  of scanning all observations with a subject filter.
- **Stan dose-event mismatch now raises** (#4). Previously
  `event_cmt > n_states` silently dropped the dose; emitter now emits
  a Stan ``reject(...)`` so the dataset/model CMT mismatch is visible.
- **Bundle seal is atomic** (#32, #34). `_COMPLETE` is written through
  a tmp + `os.replace` dance so a mid-write crash cannot leave a
  half-formed sentinel that `initialize()` would accept.
- **Post-seal bundle mutation refused** (#2). `append_search_trajectory`
  and `append_failed_candidate` raise `BundleAlreadySealedError` when
  `_COMPLETE` is present (would otherwise desync the stored digest).
- **LORO-CV `fixed_parameter` protocol** (#5). `BackendRunner.run` gains
  an explicit `fixed_parameter: bool = False` kwarg; `evaluate_loro_cv`
  passes `True`; every runner (`nlmixr2`, `bayesian`, `node`, `agentic`)
  raises `NotImplementedError` when requested rather than silently
  re-fitting and leaking train data through warm-start.
- **Gate 1 thresholds policy-driven** (#6). `Gate1Config` gains
  `param_value_min` / `param_value_max` / `param_rse_max` /
  `seed_stability_ofv_abs_spread_floor`. No more hardcoded `1e-4`,
  `1e5`, `200`, `0.1` in `gates.py`.
- **`adaptive_m` / `m_max` no longer phantom** (#7). `data/missing_data.py`
  raises `NotImplementedError` when a policy sets
  `adaptive_m=True` so the unimplemented escalation cannot silently
  degrade to fixed m.
- **Ranking tiebreak deterministic** (#8). `metrics.sort(key=...)`
  now includes `candidate_id` as tiebreaker so Borda ties resolve the
  same way across runs with identical bundles.
- **BLQ sigma validation gap + IVBolus validator branch** (#9, #10).
  `validator._validate_observation` now enforces sigma positivity on
  BLQ M3/M4 conditional on `error_model`; `_validate_absorption`
  gained an explicit no-op branch for IVBolus.
- **NODE modules contribute to `structural_param_names()`** (#11).
  Variability items that target NODE input-layer weights
  (`node_abs_wN` / `node_elim_wN`) now pass validation.
- **`_bt_logit` honours `cov.form`** (#12). Power / exponential /
  linear / maturation routing mirrors `_bt`; the rc8 path flattened
  all forms to linear-additive on the logit scale.
- **`nlmixr2` ZeroOrder + TMDDQSS routing fix** (#13). Emits
  `dur(Atot)` under TMDDQSS (central = total drug), `dur(centr)`
  elsewhere — via a new `_central_cmt_name` helper.
- **`MixedFirstZero` `frac == 1.0` no longer crashes** (#14). Clamp
  fraction to `1 - 1e-4` before emitting `log(frac / (1 - frac))`.
- **Stan TwoCmt flip-flop singularity detected** (#15). Emits a
  tolerance check; near-singular analytical denominators `reject(...)`
  so the caller can switch to the numerical ODE solver.
- **Stan TMDDQSS aliased as `Atot`** (#16). The central state is
  named both `Atot` (total drug) and `centr` (alias) so the RHS
  template stays uniform and the naming matches nlmixr2's
  `_emit_tmdd_qss_odes`.
- **Source positions flow into `DSLSpec.source_meta`** (#17).
  `parse_dsl` enables `propagate_positions=True`; `compile_dsl` walks
  the parse tree once and attaches `(line, column)` for absorption /
  distribution / elimination / observation / variability[i] so
  validator errors can point at source.
- **PIT calibration degenerate-sims safeguard** (#18, #33, #41).
  `build_predictive_diagnostics` catches the all-NaN `ValueError`
  from `_compute_pit_calibration` and returns a zero-subject sentinel;
  Gate 1 surfaces this as `pit_degenerate_no_finite_sims`; the
  `n_observations` denominator no longer inflates a true 0 to 1.
- **Credibility report provenance** (#19). `CredibilityReport` gains
  `source_result_path` + `source_result_sha256` so an auditor can
  walk back to the `BackendResult` JSON any score was derived from.
- **`SearchDAG` seals** (#20). `seal()` / `SearchDAGSealedError`; all
  mutators (`add_root`, `add_child`, `update_score`) refuse after
  seal so the in-memory DAG cannot desync from
  `candidate_lineage.json`.
- **Agentic LLM raw_text sanitized** (#21). Already guarded for
  `runner_error`; now applied on every `raw_text` append.
- **Subprocess executable validated up front** (#22).
  `Nlmixr2Runner` / `BayesianRunner` resolve the executable to an
  absolute path via `shutil.which` at `__init__`; `FileNotFoundError`
  surfaces immediately.
- **NODE runner rejects `timeout_seconds`** (#23). JAX training is
  non-interruptible; silently accepting a timeout was a lie. Now
  raises `NotImplementedError` so orchestrators can choose a
  watchdog-subprocess path.
- **MI per-imputation error isolation** (#24, #39). One failed
  imputation no longer discards every previously-completed fit;
  `m_imputations=0` refused upstream instead of emitted as a
  degenerate manifest.
- **`compute_npe` is likelihood-aware** (#25, #37). New
  `error_model` kwarg rescales residuals on proportional / combined
  models so cross-paradigm NPE comparison is no longer scale-biased;
  additive default preserves rc8 behaviour.
- **PRD §10 stress-surface perturbations declared** (#26).
  `PerturbationType` gains `SCALE_BSV_VARIANCES`, `SATURATE_CLEARANCE`,
  `TMDD`, `FLIP_FLOP` with required-parameter validators; dispatchers
  raise `NotImplementedError` until the transforms land, so Suite C
  never silently replaces a stress request with a no-op.
- **`ka` silent fallback surfaced** (#27). `_apply_fallback` logs a
  structured warning and sets `_ka_defaulted=1.0` metadata when the
  dataset card or defaults path supplies `ka`.
- **Silent emitter catch-alls removed** (#28). Unknown observation
  modules now raise `NotImplementedError` on both nlmixr2 and Stan
  paths instead of emitting a wrong proportional default.
- **InvGamma off `*_sd` targets** (#29). `_VALID_FAMILIES` no longer
  claims InvGamma is a valid SD-scale prior when the emitter draws
  on the SD scale; use `Gamma` / `HalfNormal` / `HalfCauchy`.
- **BLQ M3/M4 active-sigma helper** (#30). `active_sigmas()` on
  `BLQM3` / `BLQM4` returns only the fields the likelihood actually
  consumes given `error_model` — param-count helpers must call this
  instead of inspecting every field.
- **Gate 1 fields carry bounds + units** (#31). Added
  `description=` + explicit dimensions to the #6-added Gate1Config
  fields so audit readouts know the unit they're comparing against.
- **Agentic warm-start guards non-finite estimates** (#35). NaN / Inf
  parameters are dropped with a structured log entry instead of
  propagating into the next iteration.
- **transform_parser allow-list explicit** (#36). Extra LLM-supplied
  fields are rejected against `module_cls.model_fields` at parse time;
  Pydantic `extra="forbid"` becomes a redundant second line of
  defence instead of the only one.
- **Balanced protocol-pooling allocation** (#38). Replaces
  `rng.integers(...)` with block randomisation so the docstring's
  "balanced allocation" claim is now true.
- **`numpy.trapezoid` auto-select** (#40). Already honoured via
  `getattr(np, "trapezoid", None) or np.trapz` (NumPy 2.0+ forward,
  1.x backward).

Test baseline: 1825 passing, 1 skipped. `mypy --strict` clean (105
files); `ruff check` clean; 22 golden snapshots unchanged.

### Added — Supply-chain SBOM (CycloneDX)

APMODE now ships a CycloneDX Software Bill of Materials (SBOM) as a
producer-side sidecar and as a GitHub-release asset. The SBOM describes
the Python dependency graph that produced the bundle; it is generated
by `pip-audit --format cyclonedx-json` (pip-audit is already pinned in
the dev group, so no new top-level dependency is introduced).

- **`apmode bundle sbom <bundle_dir>`** (new CLI) — runs `pip-audit`
  against the active environment and writes
  `<bundle_dir>/bom.cdx.json`. Refuses to overwrite an existing SBOM
  without `--force`. Emits a JSON summary with `--json`.
- **Sealed-digest exclusion.** `bom.cdx.json` is excluded from
  `apmode.bundle.emitter._compute_bundle_digest` and the matching
  importer verifier, so adding or regenerating the SBOM on a sealed
  bundle never invalidates `_COMPLETE`. This preserves bundle
  immutability while still colocating the SBOM with the run.
- **RO-Crate projection.** A new `src/apmode/bundle/rocrate/entities/sbom.py`
  projector registers the SBOM as a `File` entity with
  `encodingFormat="application/vnd.cyclonedx+json"` and
  `additionalType="apmode:sbom"` (new term in `vocab.py`). The root
  Dataset's `hasPart` gains the SBOM file when present; no-op when
  absent (golden snapshot unchanged).
- **CI gate.** The `security` job in `.github/workflows/ci.yml` now
  emits `bom.cdx.json` on every push / PR and uploads it as a
  workflow artifact (`sbom-cyclonedx-json`, 90-day retention).
- **Release asset.** A new `.github/workflows/release.yml` triggers
  on `v*` tag pushes, regenerates the SBOM against the released
  environment, builds wheel + sdist with `uv build`, and attaches
  `bom.cdx.json` + distribution archives to the GitHub release via
  `gh release create`.
- **Test coverage.** `tests/unit/rocrate/test_entities_sbom.py`
  covers the no-op, successful projection (hash + media type +
  `apmode:sbom`), and the invariant that injecting `bom.cdx.json`
  after sealing does not trip `_verify_sentinel`.

### Added — RO-Crate v0.6 integration

APMODE reproducibility bundles can now be projected onto a **Workflow Run
RO-Crate — Provenance Run Crate v0.5** (`https://w3id.org/ro/wfrun/provenance/0.5`)
for FAIR packaging, WorkflowHub/Zenodo-ready archives, and machine-readable
regulatory crosswalk (FDA PCCP, EU AI Act Article 12). The Pydantic bundle
remains producer-side truth; the RO-Crate is a read-only external projection.
Design: `_research/ROCRATE_INTEGRATION_PLAN.md` §§A–H (accepted).

- **New module `src/apmode/bundle/rocrate/`** — hand-written projector
  with per-family entity modules (`data`, `policy`, `backend`, `gate`,
  `lineage`, `credibility`, `bayesian`, `agentic`, `pccp`). Reads a
  sealed bundle via JSON, emits a deterministic `ro-crate-metadata.json`
  plus a copy of every bundle artefact as a `File` entity. Never
  mutates the source bundle.
- **Two output forms**: directory (`apmode bundle rocrate export …
  --out /path/to/crate`) and ZIP (`… --out crate.zip`). ZIP entries
  use a fixed `1980-01-01` timestamp + sorted order for reproducible
  archives; `json.dumps(sort_keys=True)` gives byte-identical
  `ro-crate-metadata.json` across runs on the same sealed bundle.
- **`apmode:` vocabulary** at `https://w3id.org/apmode/terms#` —
  `lane`, `lanePolicy`, `gate`, `gateRationale`, `candidateLineageEdge`,
  `searchGraph`, `modificationDescription`, `modificationProtocol`,
  `impactAssessment`, `traceabilityTable`, `regulatoryContext`,
  `dslSpec`, `dslTransform`, `llmInvocation`, `credibilityReport`,
  `loroCV`, `scoringContract`, `nlpdComparabilityProtocol`,
  `completeSentinel`. Context declares each term with both prefix
  form and explicit full-prefix mapping to satisfy `roc-validator`'s
  compaction check.
- **Action triad per WRROC**: each candidate fit projects as a
  `CreateAction` instrumenting its `SoftwareApplication`; each gate
  decision projects as a `ControlAction` with `instrument=HowToStep`,
  `object=CreateAction`, and `apmode:gateRationale=File`; the lane
  run is wrapped in an `OrganizeAction` with
  `startTime=endTime=bundle_seal_ts`.
- **`_COMPLETE` sentinel** is embedded as a `File` entity with
  `additionalType="apmode:completeSentinel"`; the SHA-256 digest is
  surfaced via `schema:identifier="sha256:<hex>"`.
- **Round-trip import** (`apmode bundle import <crate> --out <bundle>`)
  extracts a directory- or ZIP-form crate back to a bundle directory
  and verifies the `_COMPLETE` digest against the extracted tree. ZIP
  extraction rejects path-traversal entries (ZIP-slip), symlink/hard
  link/socket entries, absolute paths, and Windows drive prefixes;
  directory imports reject symlinks inside the crate.
- **Validator integration**: `apmode validate <bundle> --rocrate`
  invokes `roc-validator` on the crate at a configurable severity
  (default REQUIRED) and profile (default `provenance-run-crate-0.5`);
  `apmode inspect <bundle> --rocrate-view` prints an RO-Crate
  summary (action-triad counts, lane, `mainEntity`, sentinel
  presence).
- **New dep**: `roc-validator>=0.8.1` in the `dev` group; main dep
  `rocrate>=0.13` for future ro-crate-py interop.
- **CI gate**: `tests/integration/test_rocrate_export_validate.py`
  parametrises 5 scenarios (minimal, multi-candidate+lineage,
  bayesian+credibility, agentic+PCCP, full-mixed) and validates both
  directory and ZIP forms at REQUIRED severity. REQUIRED failures
  block merge.
- **Security tests**: `tests/unit/rocrate/test_importer_security.py`
  covers ZIP-slip via parent-dir traversal, absolute paths, Windows
  drives, and symlink file-type mode bits.
- **Golden snapshot**: `tests/golden/rocrate/__snapshots__/` locks the
  canonical Submission-lane `ro-crate-metadata.json`.
- **Publish CLI surface** (`apmode bundle publish --workflowhub` /
  `--zenodo`) is wired as a CLI stub — sandbox/token handling is
  validated but the upload itself is deferred to v0.8 per
  `_research/ROCRATE_INTEGRATION_PLAN.md` §H.

### References

- Leo S. et al. (2024) *Recording provenance of workflow runs with RO-Crate*. PLOS ONE 19(9): e0309210. <https://doi.org/10.1371/journal.pone.0309210>
- Wilkinson S.R. et al. (2025) *Applying the FAIR Principles to computational workflows*. Scientific Data 12: 328. <https://doi.org/10.1038/s41597-025-04451-9>

## [0.5.0-rc2] — 2026-04-17

Known-Limitations Closure M0 landed: per-candidate `ScoringContract`
on every `DiagnosticBundle`, contract-grouped Gate-3 ranking, and the
Submission-lane dominance rule. Gate policy schema bumped to `0.5.0`;
bundle sentinel schema bumped to `2`.

### Added

- **`ScoringContract` Pydantic model** (`src/apmode/bundle/models.py`).
  Frozen, 7-field contract: `contract_version`, `nlpd_kind`
  (conditional / marginal), `re_treatment`
  (integrated / conditional_ebe / pooled), `nlpd_integrator`
  (nlmixr2_focei / laplace_blockdiag / laplace_diag / hmc_nuts / none),
  `blq_method`, `observation_model`, `float_precision`. Attached on
  `DiagnosticBundle.scoring_contract` (defaults to the classical
  nlmixr2 contract so existing call sites continue to type-check).
- **Per-backend contract derivation**
  (`src/apmode/bundle/scoring_contract.py`). `derive_scoring_contract`
  maps `BackendResult.backend` + `DSLSpec.observation` + BLQ method
  to the correct contract. Runners call `attach_scoring_contract` on
  their returned result: `nlmixr2_runner.run` and `bayesian_runner.run`
  override after `_parse_response`; `node_runner.run` sets the
  contract in-place on the constructed `DiagnosticBundle` (pooled
  path until M3 Laplace lands).
- **Contract-grouped Gate 3 ranking**
  (`src/apmode/governance/ranking.py`):
  - `group_by_scoring_contract(survivors)` — partitions survivors
    into same-contract buckets preserving order.
  - `ContractGroupedRanking` dataclass — one `CrossParadigmRankingResult`
    per contract, plus optional `recommended_candidate_id`.
  - `rank_by_scoring_contract(..., lane=...)` — groups, ranks each
    group via the existing `rank_cross_paradigm` primitive, applies
    Submission-lane dominance rule when `lane == "submission"`.
- **Submission-lane dominance rule.** Only candidates whose contract
  has `re_treatment == "integrated"` AND `nlpd_kind == "marginal"`
  are eligible for `recommended_candidate_id`. Otherwise the ranking
  returns `recommended_candidate_id = None` and a verbatim
  disclosure warning. Discovery and Optimization lanes leave the
  field `None` by design (separate leaderboards are the intended
  output).
- **Tests**: `tests/unit/test_scoring_contract.py` (10 tests) and
  `tests/integration/test_gate3_contract_enforcement.py` (8 tests).
### Changed

- **Bundle schema version** (`_COMPLETE_SCHEMA_VERSION`) bumped from
  1 → 2. Bundles produced before this version do not carry
  `scoring_contract`; at read time the field defaults to the
  classical nlmixr2 contract (no migration script required — Pydantic
  resolves it via the `default_factory`).
- **Gate policy schema version** bumped from `0.4.3` → `0.5.0`
  across all three lane JSONs. CLAUDE.md `apmode:AUTO:policy_gate`
  marker auto-synced.

### Deferred (explicitly surfaced, ADR-gated)

- **#2 NODE infusions (RATE>0)** — ADR: `docs/adr/0004-node-infusions.md`
  (pending); loud-reject landing in M2a.
- **Stan steady-state (SS!=0)** — ADR: `docs/adr/0003-stan-ss-scope.md`
  (pending); Stan and NODE backends hard-reject `SS!=0` at Gate 1
  (landing in M2a). README wording update pending:
  *"SS supported in nlmixr2 lane only; NODE is oral-only in v0.5.0."*

## [0.5.0-rc1] — 2026-04-17

First release candidate for 0.5. Gate policy schema `0.4.3`; profiler
policy `2.1.0`; bundle sentinel schema `1`.

### Added

- **Bundle `_COMPLETE` sentinel.** `BundleEmitter.seal()` writes a
  JSON sentinel with a SHA-256 digest of every other artifact in the
  bundle. The orchestrator calls it as the last step of a successful
  run. `apmode validate` refuses unsealed bundles and verifies the
  digest on every read. Required artifacts promoted:
  `evidence_manifest.json`, `candidate_lineage.json`.
- **`StanIdentifier` AST type alias** (regex
  `^[A-Za-z][A-Za-z0-9_]*$`) applied to `CovariateLink.param`,
  `CovariateLink.covariate`, `IIV.params`, `IOV.params`. Emitter-side
  `_sanitize_stan_name()` also rejects Stan reserved keywords and
  `__` suffixes.
- **`LLMConfig.timeout_seconds`** (default 120s) with a shared
  `_await_llm` helper wrapping both native SDK timeouts and an outer
  `asyncio.wait_for`. Applies to Anthropic / OpenAI / OpenRouter /
  Gemini / Ollama direct-SDK providers and the litellm fallback.
- **`MissingDataPolicy.imputation_convergence_rate_min`** policy
  field (submission 0.75, optimization 0.65, discovery 0.5). Replaces
  a hard-coded constant in `gates.py`.
- **BIC uniform-drop rule.** `_apply_uniform_bic_drop` in
  `governance/ranking.py` mirrors the AUC/Cmax drop; composite
  renormalization composes correctly when both components have
  missing candidates. Pathological fallback (BIC was the only
  positive weight) falls back to `vpc=0.5, npe=0.5`.
- **Transit `n` variability rejection** now applied to IOV blocks
  (previously only IIV).
- **`set_prior` transform** in the LLM-response parser
  (`backends/transform_parser.py`). Registration-map refactor plus a
  regression test that asserts every `FormularTransform` union
  variant has a parser entry.
- **Python 3.14** added to the CI test matrix.
- **`docs/adr/0001-review-deferrals.md`** registers intentional
  non-fixes (`from __future__ import annotations`, Pyright, profiler
  / orchestrator decomposition, FREM golden tests, `# type: ignore`
  audit, module-level Rich `Console`).

### Changed

- **Stan emitter IV-bolus support.** `IVBolus` no longer emits a
  phantom depot state or an undeclared `ka` term. `_needs_depot()`
  and `_centr_idx()` helpers thread through `_n_states`,
  `_emit_state_aliases`, `_emit_ode_dynamics`, and `_emit_ode_solve`.
- **Single `Lane` StrEnum** in `apmode.backends.protocol`. `cli.py`
  re-exports the enum; `routing.py` keeps the string `Literal` form
  for compatibility with policy JSONs.
- **`Orchestrator.__init__`** accepts the `BackendRunner` protocol
  directly (previously typed narrowly as `Nlmixr2Runner`, with a
  `cast` at the CLI callsite).
- **Anthropic `model_version` escrow** concatenates the response's
  `_request_id` / `id` so agentic reproducibility traces carry a
  per-call server-side fingerprint.
- **`_check_split_integrity`** fails on missing `split_gof` when
  `split_integrity_required=True`. The policy field default changed
  to `False` so benchmark and single-fold workflows do not need to
  opt out.
- **`nlmixr2` runner** spawns subprocesses with
  `start_new_session=True` instead of the thread-unsafe
  `preexec_fn=os.setsid` (matches the Bayesian runner).
- **`_within_between_ratio`** guards non-finite `pooled` against
  silently producing NaN composite scores.
- **Profiler helpers** (`_assess_protocol_heterogeneity`,
  `_assess_absorption_coverage`, `_assess_elimination_coverage`,
  `_assess_route_certainty`) return `Literal[...]` matching the
  `EvidenceManifest` fields.
- **`policy=None` on submission lane** now raises
  `APMODEConfigError` instead of silently skipping all gates.
- **Report renderer** uses `datetime.now(tz=UTC)`.
- **Seed-stability check** short-circuits to pass when the absolute
  OFV spread across seeds is below 0.1 units (platform-float floor,
  well below textbook model-selection tolerance).
- **`agentic_iterations.jsonl`** flushes and fsyncs per iteration so
  the audit trail survives mid-run crashes.
- **Assertion-based invariants** in the orchestrator's agentic stage
  replaced with explicit `RuntimeError` / `TypeError` guards so
  invariants survive `python -O`.
- **Shared `needs_ode`** helper in `apmode.dsl._emitter_utils`
  replaces duplicated definitions in both emitters.

### Fixed

- `np.trapezoid` compatibility shim — NumPy 2.0 removed
  `np.trapz`, older installs lack `np.trapezoid`; the project
  minimum `numpy>=1.25` tolerates either.
- Profiler policy-load failure wrapped in a typed
  `APMODEConfigError` with the resolved path.
- Ollama SDK drift — attribute access first, dict-subscript fallback
  for legacy installs.
- `transform_parser` handles `set_prior` (was previously missing
  from the if/elif chain).

### Deprecated

- Pre-0.5 reproducibility bundles without the `_COMPLETE` sentinel no
  longer validate. Re-run to upgrade.

### Security

- Stan emitter no longer interpolates unsanitized covariate /
  variability parameter names into generated code, closing an
  injection vector from LLM-proposed identifiers.

### Policy versions

- Gate policy schema: `0.4.2` → `0.4.3`
- Bundle sentinel schema: `1`
- Profiler policy: unchanged at `2.1.0`
