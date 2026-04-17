# Changelog

All notable changes to APMODE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
