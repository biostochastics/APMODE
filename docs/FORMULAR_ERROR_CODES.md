# Formular DSL Error Codes

## Related documentation

- [FORMULAR.md](FORMULAR.md) — primary Formular language reference; already links to this registry at line 551, so the relationship should be bidirectional.
- [FORMULAR_SEMANTICS.md](FORMULAR_SEMANTICS.md) — per-module semantic reference that independently documents FRM-SEM-010/FRM-PRIOR-001 triggers in prose; describes the same validation surface from a different angle.
- [adr/0003-sota-absorption-extension.md](adr/0003-sota-absorption-extension.md) — D1/D2/D5/D6 design rationale cited by bare name four times (FRM-SEM-005/006/007/008, FRM-LANE-003) without a path or link.
- `plans/2026-07-08-formular-sharpening-and-adoption-design.md` *(internal-only, gitignored)* — source design plan for the P1.5/P1.8/P2.1 phase work that introduced several error-code taxa (SEM_UNITS, BE, DATA, POLICY, PRIOR, AST_MACRO_*), cited by number/section only.

Canonical registry of the structured `FRM-{TAXON}-NNN` error codes emitted
by `apmode.dsl.validator.validate_dsl`. This is the single source of truth:
`tests/unit/test_dsl_error_codes.py` asserts every code referenced in
`src/apmode/dsl/validator.py` appears in this document, so an added check
that forgets to document its code fails CI.

Codes are defined in `src/apmode/dsl/errors.py::FrmCode` and attached to
every `ValidationError` returned by `validate_dsl` via the `code` field.
Each error additionally carries a `severity` (`"error"` today for every
check below — no check is currently downgraded to `"warning"`/`"info"`)
and, where a single obviously-correct fix exists, a `remediation` string.

A handful of codes are the exception: `FRM-AST-010`/`FRM-AST-011`
(block-cardinality violations) are detected on the raw parse tree by
`apmode.dsl.grammar.compile_dsl` *before* a `DSLSpec` exists, and
`FRM-AST-016`/`FRM-AST-017` (unknown/duplicate `use` macro) are detected by
`apmode.dsl.macros.expand_macros`, invoked from `compile_dsl` immediately
after the main transform pass. All four are raised as
`apmode.dsl.errors.FormularCompileError` (with `.code`/`.message`
attributes) rather than returned as a `ValidationError` from `validate_dsl`.

## Taxonomy

| Taxon | Meaning |
|---|---|
| `SYN` | Grammar/parse-level errors. |
| `AST` | AST-shape / structural-integrity errors — duplicates, missing required content, dangling references, incompatible module pairings. Not a numeric-bound check. |
| `SEM` | Semantic/numeric constraint violations — positivity, ranges, integer floors, capped values, cross-field numeric relations. |
| `LANE` | Lane-admissibility rejections (PRD §3) — the construct is well-formed and numerically valid but not admissible in the requested lane. |
| `BE` | Backend-capability errors — a capability tag the spec requires that a named emitter has no working code path for, or an unregistered emitter name. Emitted by `apmode.dsl.validator.validate_backend_bound` (Phase 1, P1.8). |
| `DATA` | Data-bound checks that require the bound dataset, not just the spec. Emitted by `apmode.dsl.validator.validate_data_bound` (Phase 1, P1.8). |
| `POLICY` | Policy-bound checks against a loaded `GatePolicy` — lane/policy mismatch, NODE-eligibility mismatch. Emitted by `apmode.dsl.validator.validate_policy_bound` (Phase 1, P1.8). |
| `PRIOR` | Prior-related validation. See [FRM-PRIOR — prior declaration/lowering errors](#frm-prior--prior-declarationlowering-errors). |

## FRM-SYN — grammar/parse-level errors

No `FRM-SYN-*` code is emitted by `validate_dsl` today. Malformed DSL text
fails inside `apmode.dsl.grammar.compile_dsl` (a Lark `UnexpectedInput`
subclass raised during parsing) *before* a `DSLSpec` exists, so there is
nothing for the post-parse validator to code. The taxon is reserved for a
future pass that re-surfaces parse diagnostics through this same coded
channel.

## FRM-SEM — semantic / numeric constraint violations

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-SEM-001` | A structural parameter (volume, rate, sigma, ...) must be `> 0`. | `absorption: FirstOrder(ka)` with `initial: { ka = -1.0 }` | Set the parameter to a value `> 0`. |
| `FRM-SEM-002` | A structural parameter must be `>= 0` (non-negative). | `absorption: LaggedFirstOrder(ka, tlag)` with `initial: { ka = 1.0, tlag = -0.5 }` | Set the parameter to a value `>= 0`. |
| `FRM-SEM-003` | A fraction/weight parameter must lie strictly in `(0, 1)`. | `absorption: MixedFirstZero(ka, dur, frac)` with `initial: { ka = 1.0, dur = 2.0, frac = 1.0 }` | Set the parameter strictly between 0 and 1. |
| `FRM-SEM-004` | An integer count (Transit/Erlang `n`, NODE `dim`) must be `>= 1`. | `absorption: Transit(n=0, ktr, ka)` with `initial: { ktr = 1.0, ka = 1.0 }` | Set the count to an integer `>= 1`. |
| `FRM-SEM-005` | Erlang chain length `n` exceeds the v0.7 cap of 7 (ADR-0003 D2). | `absorption: Erlang(n=8, ktr)` with `initial: { ktr = 1.0 }` | Reduce `n` to `<= 7`, or use Transit absorption for longer chains. |
| `FRM-SEM-006` | SumIG `k` is outside the v0.7-supported range `[1, 2]` (ADR-0003 D1). | `absorption: SumIG(k=3, MT_1, MT_2, RD2_1, RD2_2, weight_1)` | Set `k` to a value in `[1, 2]`. |
| `FRM-SEM-007` | SumIG requires `MT_1 < MT_2` (label-switching guard, ADR-0003 D1). | `absorption: SumIG(k=2, MT_1, MT_2, RD2_1, RD2_2, weight_1)` with `initial: { MT_1 = 5.0, MT_2 = 2.0, ... }` | Swap `MT_1`/`MT_2` (and their `weight_1` pairing) so `MT_1 < MT_2`. |
| `FRM-SEM-008` | SumIG `k >= 2` requires CL/V/Q fixed externally (ADR-0003 D5). | `SumIG(k=2, ...)` with no `fixed_external` prior on CL/V/Q | Add priors with `source="fixed_external"` on every disposition parameter, or supply IV reference data so the manifest sets `disposition_fixed`. |
| `FRM-SEM-009` | NODE `dim` exceeds the max dim for its `constraint_template`. | `NODEAbsorption(dim=6, constraint_template="monotone_increasing")` (max 4) | Reduce `dim`, or choose a `constraint_template` with a higher max dim. |
| `FRM-SEM-010` | The `units:` block is dimensionally inconsistent: a recognized token resolved to the wrong category (e.g. `volume` given a mass unit), or `concentration` is not a compound `mass/volume` unit. Spec-internal (no dataset needed), hence `SEM` rather than `DATA` — see `apmode.dsl.units` module docstring for the exact algorithm, including why an *unrecognized* token is never treated as a hard mismatch. | `units: { time = h, amount = mg, concentration = ng/mL, volume = mg }` (volume given a mass unit, not a volume unit) | Declare `volume` as a recognized metric volume unit (e.g. `L`, `mL`) reachable from `amount`/`concentration` via `Volume = Amount / Concentration`. |

## FRM-AST — AST-shape / structural-integrity errors

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-AST-001` | A structural parameter appears in more than one IIV block. | Two `IIV(params=[CL, ...])` blocks both listing `CL` | Remove the parameter from all but one IIV block, or merge the blocks. |
| `FRM-AST-002` | The same `(param, covariate)` `CovariateLink` pair is declared twice. | Two `CL <- WT.power(...)` entries in `covariates:` | Remove the duplicate `CovariateLink` declaration. |
| `FRM-AST-003` | An IIV/IOV block declares an empty `params` list. | `IIV(params=[], structure=diagonal)` | Add at least one structural parameter to the block's `params` list. |
| `FRM-AST-004` | `structure="block"` IIV requires at least 2 params. | `IIV(params=[CL], structure=block)` | Add a second parameter, or switch `structure` to `diagonal`. |
| `FRM-AST-005` | An IIV block references a name with no matching structural parameter. | `IIV(params=[nonexistent], ...)` | Replace the name with one of the spec's structural parameters. |
| `FRM-AST-006` | An IOV block references a name with no matching structural parameter. | `IOV(params=[nonexistent], ...)` | Replace the name with one of the spec's structural parameters. |
| `FRM-AST-007` | A `CovariateLink` references a name with no matching structural parameter. | `nonexistent <- WT.power(theta=0.75, ref=70)` in `covariates:` | Replace the name with one of the spec's structural parameters. |
| `FRM-AST-008` | IIV/IOV declared on a parameter the emitters do not apply eta to (e.g. Transit `n`). | `IIV(params=[n], ...)` on a `Transit` absorption model | Remove the parameter from the IIV/IOV block's `params` list. |
| `FRM-AST-009` | TMDD distribution requires Linear elimination (provides CL for `kel = CL/V`). | `TMDDCore(...)` distribution paired with `MichaelisMenten` elimination | Change elimination to `Linear()` (with `CL` set in the `initial:` block) when using a TMDD distribution module. |
| `FRM-AST-010` | A required top-level block (`absorption:`/`distribution:`/`elimination:`, or the `observation:`/`observations:` group) is absent. Raised by `apmode.dsl.grammar.compile_dsl` as a `FormularCompileError` on the raw parse tree, before a `DSLSpec` can be constructed — not by `validate_dsl`. | `model { absorption: FirstOrder(ka) distribution: OneCmt(V) observation: Proportional(sigma_prop=0.1) initial: { ka=1, V=1 } }` (missing `elimination:`) | Add the missing block. |
| `FRM-AST-011` | A singleton top-level block appears more than once, mutually exclusive observation blocks are both present, or a map-like declaration repeats a key that would otherwise be overwritten during compilation. Raised by `apmode.dsl.grammar.compile_dsl` as a `FormularCompileError`, same layer as `FRM-AST-010`. | Two `absorption: ...` blocks in one `model { }`; both `observation: Proportional(...)` and `observations: { ... }`; duplicate `observations: { plasma: ... }` entries; or duplicate `initial: { ka = ... }` entries | Remove the duplicate block or duplicate keyed entry. For the observation group, keep only `observation:` for a single endpoint or only `observations:` for multiple analytes. |
| `FRM-AST-012` | A calibration parameter used by a structural module has no value in the `initial:` block. | `absorption: FirstOrder(ka)` with `initial: { V=70.0, CL=5.0 }` (no `ka`) | Add `ka = <value>` to the `initial:` block. |
| `FRM-AST-013` | The `initial:` block declares a value for a parameter no structural module references. | `initial: { ka=1.0, V=70.0, CL=5.0, extra=1.0 }` where no module uses `extra` | Remove the unused entry from `initial:`, or reference it from a structural module's parameter list. |
| `FRM-AST-014` | Two entries in an `observations:` block declare the same `dvid`. | `observations: { plasma: { dvid=1, prediction=C_central, error=Proportional(sigma_prop=0.1) }, metabolite: { dvid=1, prediction=C_central, error=Additive(sigma_add=0.2) } }` (both `dvid=1`) | Assign a distinct `dvid` to each entry. |
| `FRM-AST-015` | An `observations:` entry's `prediction` does not name a known state variable of the compiled model (see `DSLSpec.known_prediction_variables`). | `observations: { plasma: { dvid=1, prediction=C_metabolite, error=Proportional(sigma_prop=0.1) } }` on a model with `OneCmt` distribution (no `C_metabolite` state exists) | Set `prediction` to one of the model's known prediction variables (`"C_central"` always; `"C_target_total"` additionally for `TMDD_QSS` distribution). |
| `FRM-AST-016` | A top-level `use <name>` statement names a macro not present in `apmode.dsl.macros.MACRO_REGISTRY` (Formular sharpening plan §4 Phase 2, P2.1). Raised by `apmode.dsl.macros.expand_macros` as a `FormularCompileError`, invoked from `apmode.dsl.grammar.compile_dsl` after the main transform pass — not by `validate_dsl`. Only a small vetted standard-library registry is supported; there are no user-defined macros in this phase. | `use pkstd.nonexistent_macro` | Use one of the registered macro names (e.g. `pkstd.standard_iiv`, `pkstd.standard_priors`, `pkstd.standard_error_model`), or remove the `use` statement. |
| `FRM-AST-017` | The same macro name appears in more than one `use` statement within a single spec. Raised by `apmode.dsl.macros.expand_macros`, same layer as `FRM-AST-016`. Rejected rather than silently deduplicated because re-running a macro's expansion twice is a correctness hazard (e.g. `pkstd.standard_iiv` would double-declare IIV on the same parameters), not merely redundant. | `use pkstd.standard_iiv` appearing twice in one `model { }` | Remove the duplicate `use` statement. |

## FRM-LANE — lane-admissibility rejections

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-LANE-001` | NODE absorption/elimination modules are not admissible in Submission lane. | `NODEAbsorption(...)` validated with `lane=Lane.SUBMISSION` | Replace the NODE module with a classical form, or run in Discovery/Optimization lane. |
| `FRM-LANE-002` | NODE `dim` exceeds the requested lane's dimension ceiling. | `NODEAbsorption(dim=6, ...)` in Optimization lane (ceiling 4) | Reduce `dim`, or run in a lane with a higher NODE dimension ceiling. |
| `FRM-LANE-003` | Absorption form (e.g. SumIG) is not admissible in the requested lane (ADR-0003 D6). | `SumIG(...)` validated with `lane=Lane.SUBMISSION` | Use a regulatorily conventional absorption form, or run in Discovery/Optimization lane. |
| `FRM-LANE-004` | NODE variant used without `DSLSpec.experimental.node` opt-in. No emitter has a working code path for NODE modules yet (all raise `NotImplementedError`); this check fires regardless of lane, prior to and independent of `FRM-LANE-001`/`FRM-LANE-002`. | `NODEAbsorption(...)` with `experimental=ExperimentalFlags(node=False)` (the default) | Add `experimental=ExperimentalFlags(node=True)` to opt in, or remove the NODE variant. |

## FRM-BE — backend-capability errors

Emitted by `apmode.dsl.validator.validate_backend_bound`, part of the
seven-level validator API (`apmode.dsl.validation_levels.validate`,
Phase 1 P1.8). Delegates entirely to `apmode.dsl.capabilities.report` —
these checks add no new capability knowledge, they turn a non-`"supported"`
status for any capability tag the spec exercises into a coded error.

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-BE-001` | The requested backend name is not a registered DSL emitter. | `validate_backend_bound(spec, backend="unknown_backend")` | Use one of the registered emitters (`nlmixr2`/`stan`/`frem`). |
| `FRM-BE-002` | The spec exercises a `CapabilityTag` the named backend does not report `"supported"` for (`explicitly_unsupported`, `unknown_gap`, or `experimental_no_stable_backend`). | `NODEAbsorption(...)` validated against `backend="stan"` (stan has no NODE code path) | Remove the unsupported construct from the spec, or choose a backend that supports it. |

## FRM-DATA — data-bound checks

Emitted by `apmode.dsl.validator.validate_data_bound` (P1.8). Conservatively
scoped to existence checks tied directly to what the spec's
`observations:`/`covariates:` blocks reference — not a general data-profiling
subsystem (see `apmode.data.profiler` for that).

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-DATA-001` | The spec declares a multi-analyte `observations:` block but the bound dataset has no `DVID` column. | `spec.observations` non-empty, `data.columns` has no `DVID` | Add a `DVID` column to the dataset identifying each analyte's rows. |
| `FRM-DATA-002` | A `covariates:` entry references a covariate with no matching column in the bound dataset. | `CL <- WT.power(...)` in `covariates:` with no `WT`/`wt` column in `data` | Add the missing column to the dataset, or remove the covariate link. |

## FRM-POLICY — policy-bound checks

Emitted by `apmode.dsl.validator.validate_policy_bound` (P1.8). A
deliberately minimal first pass: two checks that need only the spec, the
requested lane, and the loaded `GatePolicy` object already in hand. Deeper
policy validation — gate threshold sanity against actual fitted-candidate
metrics (CWRES, VPC coverage, Gate 3 composite weights, etc.) — needs a
candidate result, not just the compiled spec, and is a Phase 2 gap; see
`apmode.governance.gates` for where those checks already run once a
candidate exists.

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-POLICY-001` | The loaded policy's `lane` does not match the lane validation was requested for. | `GatePolicy(lane="discovery", ...)` validated with `lane=Lane.SUBMISSION` | Load the policy for the requested lane, or pass the lane matching the loaded policy. |
| `FRM-POLICY-002` | The spec uses a NODE absorption/elimination module but the loaded policy's `gate2.node_eligible` is false. | `NODEAbsorption(...)` validated against `policies/submission.json` (`node_eligible=false`) | Set `gate2.node_eligible=true` in the policy, or remove the NODE module(s) from the spec. |

## FRM-PRIOR — prior declaration/lowering errors

| Code | Description | Example trigger | Remediation |
|---|---|---|---|
| `FRM-PRIOR-001` | A `priors:` grammar-block entry failed `apmode.dsl.priors.build_prior_spec` construction: unresolvable target, family/target-kind mismatch, or an informative source missing `justification`/`historical_refs`. Raised by `apmode.dsl.grammar.compile_dsl` as a `FormularCompileError` (same shape as `FRM-AST-010`/`FRM-AST-011`), not by `validate_dsl`. | `priors: { CL ~ HalfCauchy(scale=1.0) }` (HalfCauchy is not a valid family for a structural target) | Fix the target name, choose a family compatible with the target kind (see `apmode.dsl.priors._VALID_FAMILIES`), or supply `justification`/`historical_refs` required for the declared `source`. |

`apmode.dsl.priors` already enforces evidence-quality / justification
checks for FDA Gate 2 (PRD §4.3.1) at its Python-level entry points:

- `PriorSpec`'s `model_validator` raises `pydantic.ValidationError` when
  an informative-source prior (`historical_data`, `expert_elicitation`,
  `meta_analysis`) has an empty `justification`, or when
  `source="historical_data"` has no `historical_refs`.
- `validate_prior_justification` and `validate_priors` return
  `list[str]` prose errors (minimum justification length, DOI format,
  prior family/target-kind mismatch, duplicate/unresolvable prior
  target) rather than `apmode.dsl.validator.ValidationError` instances.

Those functions' `list[str]` / exception-based return contracts are
depended on directly by `apmode.bundle.emitter`,
`apmode.backends.transform_parser`, `apmode.benchmarks.models`,
`apmode.dsl.prior_transforms`, `apmode.dsl.stan_emitter`, and by test
assertions that pattern-match on message substrings (e.g.
`tests/unit/test_prior_justification.py`). Rewiring *those* call sites to
emit coded `ValidationError` objects remains out of scope — it is an
API-breaking change across that call graph, and `priors.py` itself is
unaffected by `FRM-PRIOR-001`.

**Why now:** Formular sharpening plan §4 Phase 1 (P1.5) introduces a
genuinely new call site — the `priors:` grammar block, letting a human
author priors directly in Formular text — with no legacy callers
depending on a prose error shape. `compile_dsl` lowers each parsed entry
through `build_prior_spec` (the same canonical factory `SetPrior` routes
through) and, on failure, wraps the resulting `ValueError` in a
`FormularCompileError` carrying `FRM-PRIOR-001` instead of letting a bare
`ValueError` escape. This is the first `FrmCode` member to carry the
`PRIOR` taxon; every other prior-validation gap the Phase 0 note listed
(duplicate prior target, justification length, DOI format — when invoked
directly via `priors.py` rather than through the grammar) remains
prose/exception-based and un-coded.
