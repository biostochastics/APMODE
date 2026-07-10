# Formular Semantics Reference

## Related documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — §4.2.5-derived architecture-level "why" behind the PK DSL this document specifies at the grammar/semantics level.
- [FORMULAR.md](FORMULAR.md) — predecessor overview doc that explicitly defers to this one as the Phase 2 spec; the two form a matched pair.
- [FORMULAR_ERROR_CODES.md](FORMULAR_ERROR_CODES.md) — canonical FRM-* error code registry for the validation/identifiability caveats described throughout this reference.
- [FORMULAR_MIGRATION_v0.6_to_v0.7.md](FORMULAR_MIGRATION_v0.6_to_v0.7.md) — documents the breaking v0.6→v0.7 grammar rewrites that this document's Phase 1 top-level-block sections describe as already-landed end states.
- [adr/0003-sota-absorption-extension.md](adr/0003-sota-absorption-extension.md) — design-decision record for Erlang/ParallelFirstOrder/SumIG, cited by short-form throughout the Absorption axis and Transforms Reference sections.

**Status:** Phase 2 — macros, transform provenance, equations view, compact signature,
and the formal grammar reference have landed on top of the Phase 1 grammar reboot. This
document is the specification the Formular compiler implements — not a post-hoc
description. It is filled in incrementally: each subsequent phase of the plan in
`docs/plans/2026-07-08-formular-sharpening-and-adoption-design.md` §4 fills the
sections its work touches. Where a section cannot yet be verified against the actual
codebase, it says **TBD — Phase N** rather than speculating.

**Scope of this version:** describes the DSL as implemented after Formular sharpening
plan §4 Phase 1 (P1.1–P1.9) *and* Phase 2 (P2.1–P2.6). Phase 1 established:
order-insensitive top-level blocks (`src/apmode/dsl/pk_grammar.lark`), the
structural/calibration split (structural modules declare no inline values; every
calibration number lives in the `initial:` block), the
`metadata:`/`units:`/`initial:`/`priors:`/`covariates:` top-level blocks, the
multi-analyte `observations:` block, the seven-level validator API
(`src/apmode/dsl/validation_levels.py`), and the `apmode formular
{fmt,lint,validate,explain,diff,lower,compat}` CLI tree (`src/apmode/cli_formular.py`).
Phase 2 adds, concretely:

- A closed, vetted **macro registry** (`src/apmode/dsl/macros/__init__.py`,
  `src/apmode/dsl/macros/stdlib.py`) reachable via a new `use <dotted.name>` top-level
  statement (`use_block` in `pk_grammar.lark`), expanded post-transform by
  `expand_macros` and recorded in `DSLSpec.macros_used` (P2.1). See
  [Macros](#macros).
- **`rationale: str`** and **`expected_diagnostic_effect: list[str]`** fields added to
  all 10 `FormularTransform` union members in `src/apmode/dsl/transforms.py` and
  `src/apmode/dsl/prior_transforms.py`, plus a matching `rationale` /
  `expected_diagnostic_effect` / `applied_at` triple added to the bundle lineage records
  `CandidateLineageEntry` and `SearchGraphEdge` in `src/apmode/bundle/models.py` (P2.2).
  See [Transforms Reference](#transforms-reference).
- A read-only **symbolic ODE-system view**, `src/apmode/dsl/equations.py`
  (`build_equations`/`render_equations`, sympy-backed), wired into
  `apmode formular explain --equations` (P2.3). See
  [CLI: `explain --equations` and `signature`](#cli-explain---equations-and-signature).
- A **compact one-line model signature**, `build_signature` in
  `src/apmode/dsl/serializer.py`, wired into the new `apmode formular signature
  <spec-file>` command (P2.4). See the same CLI subsection.
- This document's own **Formal Grammar Reference** section, a human-readable rendering
  of `src/apmode/dsl/pk_grammar.lark` (P2.5).
- The macros/transforms/equations/signature sections below (P2.6), plus the
  Phase-1-era Phase-2-TBD markers throughout this document that could now be resolved.

Read against `src/apmode/dsl/ast_models.py`, `validator.py`, `transforms.py`,
`prior_transforms.py`, `priors.py`, `units.py`, `capabilities.py`, `canonical.py`,
`equations.py`, `serializer.py`, `pk_grammar.lark`, `cli_formular.py`,
`macros/__init__.py`, `macros/stdlib.py`, `nlmixr2_emitter.py`, `stan_emitter.py`, and
`frem_emitter.py` as they exist today.

**Two now-resolved forward references from the Phase 0 skeleton:**
- *Capability matrix* — `src/apmode/dsl/capabilities.py` exists and is populated: 31
  `CapabilityTag` entries (10 absorption + 5 distribution + 5 elimination +
  5 variability + 6 observation), `SUPPORTS`/`EXPLICITLY_UNSUPPORTED` frozensets on each of the
  three registered emitters (`nlmixr2`, `stan`, `frem`), enforced by
  `scripts/verify_capability_coverage.py`. Every "see `src/apmode/dsl/capabilities.py`
  for current support status" pointer below is also reachable interactively via
  `apmode formular compat [<spec-file>] [--backend]` (P1.9) — the CLI is a thin
  renderer over the same code-derived matrix, not a second source of truth.
- *`Lane` enum* — `apmode.dsl.lane.Lane(StrEnum)` (`SUBMISSION`, `DISCOVERY`,
  `OPTIMIZATION`) is now the canonical definition, versioned via
  `LANE_TAXONOMY_VERSION = "1.0.0"`. `apmode.backends.protocol.Lane` re-exports this same
  object (not a parallel copy) so pre-existing backend call sites are unaffected. This
  document cites `apmode.dsl.lane.Lane` throughout; any residual `apmode.backends.protocol.Lane`
  reference in older prose means the identical enum.

## Table of contents

- [Cross-cutting notes](#cross-cutting-notes)
- Top-level spec blocks (Formular sharpening plan §4 Phase 1)
  0a. [`metadata:` block](#metadata-block)
  0b. [`units:` block](#units-block)
  0c. [`initial:` block](#initial-block)
  0d. [`covariates:` block (top-level)](#covariates-block-top-level)
  0e. [`priors:` block (human-authored prior syntax)](#priors-block-human-authored-prior-syntax)
  0f. Multi-level validator API + `apmode formular` CLI tree — see
      [Cross-cutting notes](#cross-cutting-notes)
- Absorption axis
  1. [IVBolus](#ivbolus)
  2. [FirstOrder](#firstorder)
  3. [ZeroOrder](#zeroorder)
  4. [LaggedFirstOrder](#laggedfirstorder)
  5. [Transit](#transit)
  6. [MixedFirstZero](#mixedfirstzero)
  7. [Erlang](#erlang)
  8. [ParallelFirstOrder](#parallelfirstorder)
  9. [SumIG](#sumig)
  10. [NODEAbsorption](#nodeabsorption)
- Distribution axis
  11. [OneCmt](#onecmt)
  12. [TwoCmt](#twocmt)
  13. [ThreeCmt](#threecmt)
  14. [TMDDCore](#tmddcore)
  15. [TMDDQSS](#tmddqss)
- Elimination axis
  16. [LinearElim](#linearelim)
  17. [MichaelisMenten](#michaelismenten)
  18. [ParallelLinearMM](#parallellinearmm)
  19. [TimeVaryingElim](#timevaryingelim)
  20. [NODEElimination](#nodeelimination)
- Variability axis
  21. [IIV](#iiv)
  22. [IOV](#iov)
  23. [CovariateLink](#covariatelink)
  24. [Occasion specs (ByStudy / ByVisit / ByDoseEpoch / Custom)](#occasion-specs)
- Observation axis
  25. [Proportional](#proportional)
  26. [Additive](#additive)
  27. [Combined](#combined)
  28. [BLQM3](#blqm3)
  29. [BLQM4](#blqm4)
  29a. [Multi-analyte observations: block (ObservationEndpoint)](#multi-analyte-observations-block-observationendpoint)
- Prior axis
  30. [Prior families overview](#prior-families-overview)
  31. [NormalPrior](#normalprior)
  32. [LogNormalPrior](#lognormalprior)
  33. [HalfNormalPrior](#halfnormalprior)
  34. [HalfCauchyPrior](#halfcauchyprior)
  35. [GammaPrior](#gammaprior)
  36. [InvGammaPrior](#invgammaprior)
  37. [BetaPrior](#betaprior)
  38. [LKJPrior](#lkjprior)
  39. [MixturePrior](#mixtureprior)
  40. [HistoricalBorrowingPrior](#historicalborrowingprior)
- Macros (Formular sharpening plan §4 Phase 2, P2.1)
  41. [Macros](#macros)
- Transforms (Formular sharpening plan §4 Phase 2, P2.2)
  42. [Transforms Reference](#transforms-reference)
- CLI additions (Formular sharpening plan §4 Phase 2, P2.3/P2.4)
  43. [CLI: `explain --equations` and `signature`](#cli-explain---equations-and-signature)
- Formal grammar (Formular sharpening plan §4 Phase 2, P2.5)
  44. [Formal Grammar Reference](#formal-grammar-reference)
- [Verification notes](#verification-notes)

---

## Cross-cutting notes

- **Dimensional homogeneity is checked at the spec level, not enforced by unit
  conversion.** `src/apmode/dsl/units.py` (Phase 1, P1.3) is a dimensional-homogeneity
  checker, not a Pint-style unit-conversion library: it never converts a value between
  units. Every "Parameters" table below states an *intended* physical dimension (e.g.
  `CL: volume/time`) for pharmacometric readability; the optional top-level `units: {
  time, amount, concentration, volume }` block declares the spec's GLOBAL units (there is
  still no per-parameter unit annotation syntax), and `apmode.dsl.units.check_units_consistency`
  verifies `volume` is dimensionally reachable from `amount`/`concentration` via
  `Volume = Amount / Concentration` (accounting for standard metric mass/volume prefixes).
  `validate_dsl` raises `FRM-SEM-010` on a genuine mismatch; a spec with no `units:` block
  is unaffected. `apmode.dsl.units.unit_coverage_report` produces a per-parameter
  `UnitCoverageReport` ({checked, unchecked, mismatched}) written into
  `compiled_specs/{id}_fingerprints.json` under the `units_coverage` key. This does not
  check that a dataset's actual time/concentration columns match the declared units —
  that cross-check against the bound dataset remains a documented gap (Phase 2
  candidate).
- **All AST node classes are frozen Pydantic models** (`model_config = ConfigDict(frozen=True)`
  in `ast_models.py`). Transforms never mutate; they always construct a new `DSLSpec` with a
  fresh `model_id` (`apmode.ids.generate_candidate_id()`).
- **Two independent emitters exist today**: `nlmixr2_emitter.py` (R/rxode2, primary
  engine) and `stan_emitter.py` (Stan/Torsten via cmdstanpy, Phase 2+). A third,
  `frem_emitter.py`, is a specialised nlmixr2 variant for Full Random Effects Models
  (covariates modeled as joint observations rather than `CovariateLink`) and is
  documented separately where relevant. NODE modules are lowered by neither R emitter —
  they are trained by a dedicated JAX/Diffrax/Equinox runner
  (`src/apmode/backends/node_runner.py`, `node_ode.py`, `node_trainer.py`), which is an
  implemented, non-stub backend as of this writing (not a raw-code escape hatch — see the
  NODE sections below for what "lowering" means there).
- **Lane enum**: `apmode.dsl.lane.Lane` (`StrEnum`): `SUBMISSION`, `DISCOVERY`,
  `OPTIMIZATION`, versioned via `LANE_TAXONOMY_VERSION = "1.0.0"` (Phase 0, P0.3).
  `apmode.backends.protocol.Lane` re-exports this same object — there is exactly one
  `Lane` type, not two independently-defined enums that happen to agree.
  `src/apmode/dsl/validator.py` is the sole place lane-admissibility rules for the DSL
  are enforced today (`_LANE_ABSORPTION_INADMISSIBLE`, `_LANE_DIM_CEILING`,
  `_validate_node_constraints`, plus the P0.8 NODE experimental gate,
  `FrmCode.LANE_NODE_EXPERIMENTAL_GATE`). Per-lane governance policy files
  (`policies/<lane>.json`) are a separate, downstream concern (Gate 1/2/2.5/3); the new
  `validate_policy_bound` level (P1.8, below) cross-checks a loaded `GatePolicy` against
  lane/NODE-eligibility but does not re-implement Gate 1/2/2.5/3 candidate-metric
  thresholds.
- **`structural_param_names()`** (`DSLSpec` method) is the authoritative list of which
  parameter names exist for a given compiled spec; the variability/prior/covariate
  machinery validates against this set, not against a fixed schema. Section 3
  ("Parameters") below lists the names this method contributes for each module variant.
- **Structural-vs-calibration split (Formular sharpening plan §4 Phase 1, P1.4).** Every
  structural module (absorption/distribution/elimination) now declares its calibration
  parameters with **no inline value** — e.g. `absorption: FirstOrder(ka)`, not
  `FirstOrder(ka=1.2)`. Every numeric starting estimate for those parameters lives in a
  single flat top-level `initial: { ka = 1.2, V = 35, CL = 4.1 }` block
  (`DSLSpec.initial: dict[str, float]`). `DSLSpec.calibration_param_names()` is the
  authoritative list of which `structural_param_names()` entries require an `initial:`
  value — it excludes names that are structural-but-not-calibrated: chain-length
  integers (`Transit.n`, `Erlang.n`; `Erlang.n` is set only by the
  `ConvertTransitToErlang` transform) and NODE input-layer weight names
  (`node_abs_w*`/`node_elim_w*` — no DSL primitive gives them an initial value; that is a
  NODE-backend concern). `FrmCode.AST_INITIAL_VALUE_MISSING` /
  `FrmCode.AST_INITIAL_VALUE_UNUSED` catch, respectively, a calibration parameter with no
  `initial:` entry and an `initial:` entry naming a parameter no structural module
  references. Every per-module "Parameters" subsection below states, for each field,
  whether it is structural (inline, no value) or calibration (lives in `initial:`).
- **Nine typed Formular transforms** live in `src/apmode/dsl/transforms.py`
  (`FormularTransform` union): `SwapModule`, `AddCovariateLink`, `AdjustVariability`,
  `SetTransitN`, `ToggleLag`, `ReplaceWithNODE`, `ConvertTransitToErlang`,
  `AddParallelRoute`, `SetSumIGComponents`, plus `SetPrior`
  (`src/apmode/dsl/prior_transforms.py`) — ten counting `SetPrior`. Each module's
  "Supported transforms" section below lists precisely which of these ten can target it.
  `AddCovariateLink` mirrors `CovariateLink`'s per-form field contract exactly (P1.6) and
  now appends into `DSLSpec.covariates` — the top-level list, not a `VariabilityItem` —
  see [`covariates:` block](#covariates-block-top-level) and the `CovariateLink` section.
  All ten now carry `rationale: str = ""` / `expected_diagnostic_effect: list[str] =
  []` provenance fields (P2.2) — see [Transforms Reference](#transforms-reference) for
  the exact field shape and where `applied_at` actually lives (not on the transform).
- **Multi-level validator API (Formular sharpening plan §4 Phase 1, P1.8).**
  `apmode.dsl.validation_levels.validate(spec, *, level, lane, data=None, backend=None,
  policy=None) -> ValidationReport` runs one or more of seven independent levels — `syntax
  → ast → semantic → data_bound → lane_bound → backend_bound → policy_bound` — and
  returns a `ValidationReport` keyed by level (`report.by_level`, `report.levels_run`,
  `report.ok`). `validate()` takes an already-compiled `DSLSpec`, so a genuine parse
  failure (Lark `UnexpectedInput` / `FormularCompileError` from
  `apmode.dsl.grammar.compile_dsl`) short-circuits *before* `validate()` is ever called —
  the CLI's `_compile_or_exit` helper (below) is where that short-circuit actually lives.
  The `syntax` level itself is consequently a reserved, currently-always-empty slot in
  `ValidationReport.by_level`: no `FRM-SYN-*` code is emitted by `validate_dsl` today (see
  `errors.py`), so selecting `syntax` returns no findings even though the level exists in
  the taxonomy for a future pass that re-surfaces parse diagnostics through this same
  coded channel. `data_bound` needs a bound `pandas.DataFrame` (checks
  `FrmCode.DATA_REQUIRED_COLUMN_MISSING` / `DATA_COVARIATE_COLUMN_MISSING`);
  `backend_bound` needs an emitter name and delegates to
  `apmode.dsl.capabilities.report()` (`FrmCode.BE_UNKNOWN_BACKEND` /
  `BE_CAPABILITY_UNSUPPORTED`); `policy_bound` needs a loaded
  `apmode.governance.policy.GatePolicy` (`FrmCode.POLICY_LANE_MISMATCH` /
  `POLICY_NODE_INELIGIBLE`). Every `ValidationError` this API returns carries
  `source_span: SourceSpan | None`, `code: FrmCode`, `severity`, and `remediation` (Phase
  0, P0.1/P0.2) — the CLI (below) is a renderer over this same API, not a second
  implementation.
- **`apmode formular` CLI tree (Formular sharpening plan §4 Phase 1, P1.9; extended
  Phase 2, P2.3/P2.4).** `src/apmode/cli_formular.py` registers `apmode formular {fmt,
  lint, validate, explain, diff, lower, compat, signature}` into the main Typer app.
  `fmt` re-serializes a spec in canonical block order (warns on stderr that this
  invalidates external line-number references); `lint` runs the
  `ast`/`semantic`/`lane_bound` levels (the ones needing no extra input); `validate`
  exposes the full seven-level API with `--level`/`--data`/`--backend`/`--policy`/
  `--lane`; `explain` prints a human-readable module-choice summary (including
  `metadata:`/`units:`/`initial:`/`covariates:`/`priors:`/`observations:` blocks) and,
  as of P2.3, `--equations` renders the derived symbolic ODE system via
  `apmode.dsl.equations.build_equations`/`render_equations` instead of stubbing; `diff`
  canonicalizes both specs (block and entry order) before comparing so reordering never
  shows as a diff; `lower` runs a capability pre-flight (`validate_backend_bound`)
  before emitting nlmixr2/Stan/FREM code, failing fast with the exact missing
  `CapabilityTag`s rather than emitting broken output; `compat` prints the code-derived
  capability matrix, optionally scoped to one spec and/or one backend; `signature`
  (new, P2.4) prints `apmode.dsl.serializer.build_signature`'s compact one-line
  module-choice summary. `fmt --migrate` (P1.11) is fully implemented: it routes the raw
  pre-0.7 text through `apmode.dsl.migration.migrate_v06_to_v07` (a best-effort
  text-pattern rewriter — inline calibration values → `initial: {}`, compact
  `CovariateLink(...)` → arrow-syntax `covariates: {}` — reproducing the exact
  pre-existing nlmixr2-emitter default reference values so a migrated-then-compiled
  spec lowers to numerically identical backend code), writes the result in place
  (`--in-place`) or to stdout, prints any per-construct warnings to stderr, and exits 1
  only if warnings were produced (0 otherwise; unrelated file-not-found errors exit 1
  via the usual CLI error path). See
  [CLI: `explain --equations` and `signature`](#cli-explain---equations-and-signature)
  for the full detail on the two P2.3/P2.4 additions.

---

## Top-level spec blocks (Formular sharpening plan §4 Phase 1)

Six optional/new top-level blocks were added or reshaped in Phase 1. Grammar source:
`src/apmode/dsl/pk_grammar.lark`; AST: `src/apmode/dsl/ast_models.py`; lowering:
`apmode.dsl.grammar.compile_dsl`. All top-level blocks may appear in any order
(P1.1) — `metadata:`/`units:`/`initial:` are optional and at most one each;
`covariates:`/`priors:` are optional, at most one each, with zero-or-more entries
inside; exactly one of `observation:`/`observations:` is required.

### `metadata:` block

1. **Synopsis.** Free-text spec provenance: `metadata: { title="...", intent="...",
   context_of_use="...", analyte="...", version="..." }`. Every field is an optional
   string.
2. **Grammar.** `metadata_block: "metadata:" "{" (metadata_item ("," metadata_item)*)? "}"`;
   at most one per model, zero-or-more of the five named items in any order/subset.
3. **AST.** `apmode.dsl.ast_models.Metadata` (frozen Pydantic, all fields
   `str | None = None`); `DSLSpec.metadata: Metadata | None`, `None` when the block is
   absent or the spec was built programmatically pre-Phase-1.
4. **Compile-time behavior.** None of the five fields affect compilation, validation, or
   emission — purely descriptive.
5. **Fingerprint scope (`apmode.dsl.canonical`).** Excluded from both
   `structure_fingerprint` and `spec_fingerprint` entirely (free-text prose, per the
   design plan's invariant 7) — editing `metadata:` never invalidates a fingerprint or a
   content-addressed cache hit.
6. **Backend lowering notes.** Not read by `nlmixr2_emitter.py`/`stan_emitter.py`/
   `frem_emitter.py`. Carried through to the reproducibility bundle manifest
   (`BundleEmitter`) for human/audit consumption, and rendered by `apmode formular
   explain` (P1.9).
7. **Known limitations.** No CI lint on `metadata:` prose content (unlike
   `docs/REGULATORY_POSTURE.md`'s banned-marketing-language lint, §6.2 of the design
   plan — planned, Track B.2 of `docs/plans/2026-07-08-formular-sharpening-and-adoption-design.md`;
   not yet created) — a `title`/`intent` claiming e.g. "validated for clinical use" would
   compile without complaint. Not currently in scope; flagged for awareness, not fixed here.

### `units:` block

1. **Synopsis.** Declares the spec's GLOBAL measurement units:
   `units: { time = h, amount = mg, concentration = ng/mL, volume = L }`. Not
   per-parameter unit annotation — Formular has no syntax to attach a unit to an
   individual `CL`/`V`/`ka` value.
2. **Grammar.** `units_block: "units:" "{" "time" "=" unit_expr "," "amount" "=" unit_expr
   "," "concentration" "=" unit_expr "," "volume" "=" unit_expr "}"` — all four fields
   required when the block is present (the checker needs all three base units plus the
   derived `volume` to verify self-consistency); `unit_expr` is a bare `NAME` (`h`, `L`)
   or a compound `NAME "/" NAME` (`ng/mL`), reusing the `NAME` lexer terminal rather than
   a dedicated unit-token terminal so it never competes for lexer priority.
3. **AST.** `apmode.dsl.ast_models.UnitsDeclaration` (frozen, `extra="forbid"`);
   `DSLSpec.units: UnitsDeclaration | None`.
4. **Dimensional-homogeneity checker (`apmode.dsl.units`), not a unit-conversion
   library.** Never converts a number between units and has no Pint dependency. Answers
   exactly two questions: (a) is the declaration internally self-consistent — is
   `volume` dimensionally reachable from `amount`/`concentration` via `Volume = Amount /
   Concentration` (`check_units_consistency`, `UnitConsistencyResult`)? (b) for each
   calibration parameter, what dimension does its structural *role* imply
   (`_ROLE_DIMENSIONS`, a fixed table: `CL`→Clearance, `V`/`V1`/`V2`/`V3`→Volume,
   `ka`/`ktr`/`kdecay`→Rate, `Vmax`→Amount/Time, `Km`/`sigma_add`→Concentration,
   `tlag`/`dur`→Time, `frac`/`sigma_prop`→Unitless), and does that dimension depend on
   the (possibly broken) volume-reachability check? Recognized token vocabulary is
   deliberately small: mass (`g`/`mg`/`mcg`/`ug`/`ng`), volume (`L`/`mL`), time
   (`h`/`hr`/`min`/`day`/`d`) — every prefixed mass token collapses to category `"mass"`
   (conversion factors are irrelevant to a homogeneity check and never computed).
5. **Three-way field-status semantics.** Each of `units:`'s leaf fields (`time`,
   `amount`, `volume`, `concentration_num`, `concentration_den`) resolves to `"ok"`
   (recognized, correct category), `"mismatch"` (recognized but wrong category — e.g.
   `volume = "mg"`, mass-shaped), or `"unresolved"` (token outside the recognized
   vocabulary, e.g. a typo or `"lb"` — never silently guessed at or treated as a
   false-positive mismatch). A calibration parameter's role-dimension status is
   `"checked"`/`"unchecked"`/`"mismatched"` accordingly (`_resolve_role_status`), and only
   a genuine `"mismatched"` role raises `FrmCode.SEM_UNITS_INCONSISTENT`
   (`FRM-SEM-010`) via `validate_dsl`. A spec with no `units:` block is entirely
   unaffected — silence, not a hard requirement.
6. **`UnitCoverageReport` (bundle artifact).** `unit_coverage_report(spec)` returns
   `status="not_declared"` (all lists empty) when `spec.units is None`, else
   `status="checked"` with `checked`/`unchecked`/`mismatched: list[UnitMismatch]` lists
   over `DSLSpec.calibration_param_names()` plus the observation module's *active*
   sigma fields (via `active_sigmas()`, so `BLQM3`/`BLQM4`'s always-present-but-possibly-
   inactive sigma default is never reported as checked/mismatched). Written into
   `compiled_specs/{model_id}_fingerprints.json` under a `units_coverage`-shaped key —
   every model with a `units:` block gets this report; a model without one gets the
   explicit `not_declared` status rather than silent omission (the design plan's "silence
   is documented, not implicit" invariant).
7. **`sigma_prop`/`sigma_add` are standard-deviation scale, not variance —
   dimensional grounding for a real, previously-live confusion point.**
   `CHANGELOG.md`'s Suite A benchmark-simulator notes document exactly this failure mode:
   "`σ_prop` and `σ_add` in the simulator and `reference_params.json` are standard
   deviations on the data scale. NONMEM's `SIGMA` block uses variance; square before
   comparing." Formular's `sigma_prop`/`sigma_add` fields (on `Proportional`/`Additive`/
   `Combined` and the always-present `BLQM3`/`BLQM4` fields) follow the simulator's
   SD-scale convention throughout — a spec author porting a NONMEM `$SIGMA` block must
   take the square root before writing the value into Formular, not copy it verbatim.
   `unit_coverage_report` additionally runs `_sigma_prop_heuristic_warnings`: a narrowly
   scoped, non-fatal heuristic flagging `sigma_prop > 1.0` (an SD-scale *fraction*
   implying a residual SD larger than the predicted concentration itself) as unusual
   enough to warrant a warning — surfaced on `UnitCoverageReport.sigma_prop_warnings`,
   never a hard validation error, and deliberately only catches the large-value
   direction (a copied-in NONMEM variance without squaring makes the value *smaller*, not
   larger, so this heuristic does not catch every SD/variance mix-up — documented as a
   partial, not a complete, safeguard).
8. **Backend lowering notes.** Not read by any emitter — purely a compile-time/bundle
   diagnostic. Rendered by `apmode formular explain` alongside the coverage summary
   (`{status}, {n} mismatched`).
9. **Fingerprint scope.** Included in `spec_fingerprint` (semantic content that affects
   interpretation of `initial:` values), excluded from `structure_fingerprint` (per
   design plan invariant 7).
10. **Known limitations.** Does not check that a bound dataset's actual time/
    concentration columns match the declared units — that cross-check against real data
    remains a documented gap (Phase 2 candidate). No per-parameter unit override syntax
    exists; a spec mixing e.g. mg-dosed and mcg-dosed cohorts within one `units:`
    declaration cannot express that distinction.

### `initial:` block

1. **Synopsis.** The *only* way to give starting/calibration estimates for structural
   parameters: `initial: { ka = 1.2, V = 35, CL = 4.1 }`. Every structural module
   declares its parameters with no inline value (e.g. `absorption: FirstOrder(ka)`) — see
   the [structural-vs-calibration split](#cross-cutting-notes) cross-cutting note.
2. **Grammar.** `initial_block: "initial:" "{" (initial_item ("," initial_item)*)? "}"`;
   `initial_item: NAME "=" NUMBER`; optional, at most one per model, zero-or-more
   entries.
3. **AST.** `DSLSpec.initial: dict[str, float]` — a flat parameter-name → value map, not
   a typed per-module field. `DSLSpec.get_initial(name, default=None)` is the lookup
   helper; `kdecay` on `TimeVaryingElim` is the one calibration parameter with a
   conventional non-error default (`0.1`) when omitted, applied by callers passing
   `default=0.1` explicitly — `get_initial` itself has no per-name opinion.
4. **Validation.** `FrmCode.AST_INITIAL_VALUE_MISSING` (a calibration parameter used by a
   structural module has no `initial:` entry) and `FrmCode.AST_INITIAL_VALUE_UNUSED` (an
   `initial:` entry names a parameter no structural module references) — both AST-shape
   checks, run against `DSLSpec.calibration_param_names()`.
5. **Transform interaction.** `apply_transform` (`transforms.py`) merges each
   transform's own `initial_overrides` (if any) on top of the carried-forward
   `spec.initial`, then prunes any name no longer referenced by the resulting structural
   modules and raises if any newly-required calibration parameter is still missing —
   e.g. `SwapModule(absorption: FirstOrder → ZeroOrder)` requires the caller to supply
   `dur` via `initial_overrides` since `ZeroOrder` needs a value `FirstOrder` never had.
6. **Fingerprinting.** `apmode.dsl.canonical.initial_fingerprint(spec)` hashes exactly
   this dict, independent of the rest of the spec — enables cache-hits/re-fits that
   share structure but differ only in starting values. Also included (not excluded)
   inside `spec_fingerprint`.
7. **Backend lowering notes.** Every emitter reads `spec.get_initial(...)`/
   `spec.initial` to seed the `ini()` block (nlmixr2) or prior-centering /
   `initial_estimates` overrides (Stan) — this is the sole source of starting values;
   there is no separate "default initial value" fallback baked into the emitters
   themselves (beyond `kdecay`'s documented `0.1`, item 3).
8. **Known limitations.** No per-parameter bounds/transform-scale declaration in
   `initial:` itself (log-scale vs. natural-scale is an emitter convention, not spec
   text) — a Phase 2 candidate if authors need to express bounds explicitly rather than
   relying on emitter defaults.

### `covariates:` block (top-level)

Full per-form field semantics, back-transform expressions, and fingerprint scope are
documented in the [`CovariateLink`](#covariatelink) section under Variability axis
(retained there for historical section-numbering continuity even though, as of P1.6,
`CovariateLink` is no longer a `VariabilityItem`). This entry covers only the block-level
grammar/relocation story.

1. **Synopsis.** `covariates: { CL <- WT.power(theta=0.75, ref=70), CL <-
   SEX.categorical(reference="M"), CL <- PMA.maturation(tm50=45, hill=3) }` — arrow
   syntax (`param <- covariate.form(...)`), a dedicated top-level block distinct from
   `variability:`.
2. **Grammar.** `covariates_block: "covariates:" "{" (covariate_entry ("," covariate_entry)*)?
   "}"`; `covariate_entry: NAME "<-" NAME "." covariate_form_call`; five `covariate_form_call`
   alternatives (`power`, `exponential`, `linear`, `categorical`, `maturation`), each with
   its own named-argument field list matching `CovariateLink`'s per-form contract exactly.
3. **Relocation rationale (P1.6).** Pre-Phase-1, covariate effects were a third
   `VariabilityItem` kind alongside `IIV`/`IOV`, with a function-call syntax and hardcoded
   per-form reference constants (e.g. `power` silently centering on a hardcoded 70 kg
   reference weight, Anderson & Holford 2008). Moving covariates to their own top-level
   list/block reflects that a covariate link is a fixed-effect structural relationship,
   not a random-effect variance-component declaration — grouping them under
   `variability:` conflated two different kinds of "things that make a parameter vary
   across subjects." The old function-call form (without explicit reference values) is
   removed entirely, not kept as a deprecated alternate syntax.
4. **AST.** `DSLSpec.covariates: list[CovariateLink]` (top-level field, default empty
   list) — see the `CovariateLink` section for the full model.
5. **Backend/capability-tag notes.** `CapabilityTag.VARIABILITY_COVARIATE_LINK` /
   `VARIABILITY_COVARIATE_MATURATION_FORM` keep their pre-P1.6 dotted-string values
   (`"variability.covariate_link"`) even though covariates are no longer structurally
   part of the variability module — a deliberate choice to avoid a gratuitous rename of a
   stable capability-contract string (`apmode.dsl.capabilities.tags_for_spec` iterates
   `spec.covariates`, not `spec.variability`, to derive these tags).
6. **Known limitations.** See the `CovariateLink` section, item 11.

### `priors:` block (human-authored prior syntax)

1. **Synopsis.** `priors: { CL ~ LogNormal(mu=log(4.0), sigma=0.25) source=historical_data
   doi="10.1002/..." justification="..." }` — lets a human author Bayesian priors
   directly in Formular text, with parity to the agentic `SetPrior` transform. Prior to
   P1.5, `PriorSpec` had no textual grammar block at all — it was Python-API-only
   (`build_prior_spec()`/`SetPrior`).
2. **Grammar.** `priors_block: "priors:" "{" prior_entry* "}"`;
   `prior_entry: NAME "~" prior_family prior_attr*`; `prior_attr` covers `source`, `doi`,
   `justification`, `historical_refs` (a `string_list`). Ten `prior_family` alternatives
   mirror `apmode.dsl.priors.PriorFamily` **field-for-field** — same field names, same
   units (e.g. `LogNormal`'s `mu`/`sigma` are log-space, matching Stan/Pumas convention)
   — so a human-authored prior lowers to the exact same `PriorSpec` a Python caller gets
   from `build_prior_spec()` with equivalent arguments. A minimal `numexpr` grammar
   (`NUMBER`, or `log(numexpr)`) lets an author write `mu=log(4.0)` directly rather than
   pre-computing the log by hand.
3. **Lowering (parity guarantee, P0.4/P1.5).** Each `prior_entry` is lowered through
   `apmode.dsl.priors.build_prior_spec` — **the same canonical factory the agentic
   `SetPrior` transform routes through** (`apmode.dsl.prior_transforms.validate_set_prior`/
   `apply_set_prior`) — inside `apmode.dsl.grammar.compile_dsl`, not inside the Lark
   `Transformer` itself. Reason: Lark wraps any exception a `Transformer` callback raises
   in `lark.exceptions.VisitError`, which would bury `build_prior_spec`'s plain
   `ValueError`/`FormularCompileError` behind an extra unwrap every caller would need to
   perform — lowering after the parse tree is fully built avoids that.
4. **Error handling.** A `build_prior_spec` failure at the grammar-compile boundary
   (unresolvable target, family/target-kind mismatch, or an informative source missing
   `justification`/`historical_refs`) is caught and re-raised as `FormularCompileError`
   carrying `FrmCode.PRIOR_INVALID_DECLARATION` (`FRM-PRIOR-001`) — the first, and so
   far only, coded `FRM-PRIOR-*` error (see `errors.py`'s "Why FRM-PRIOR now has one
   emitted code" for why every other prior-validation gap remains prose/exception-based:
   `priors.py`'s existing entry points are depended on by six-plus call sites and
   message-substring-asserting tests, so rewiring them to coded `ValidationError`s would
   be an API-breaking change out of scope here — this grammar-compile call site is new,
   with no legacy string-matching callers to break).
5. **Justification enforcement preserved.** Same FDA Gate 2 (PRD §4.3.1) rules as
   Python-API construction: non-empty `justification` + non-empty `historical_refs` for
   `historical_data` (enforced by `PriorSpec`'s own model validator, raised as part of
   `build_prior_spec`'s construction step); the separate, finer-grained
   `validate_prior_justification` (length ≥ 50 chars, Crossref-canonical DOI pattern) is
   run downstream by the emitter before writing `prior_manifest.json`, not by the grammar
   compile step itself.
6. **Fingerprint scope.** Prior *presence* (`target` + `family` only) is in
   `structure_fingerprint`; full prior content (hyperparameters, `source`,
   `historical_refs`) is in `spec_fingerprint`; `justification`/`doi` text feeds
   `justification_hash` instead, kept separate so prose edits don't invalidate
   `spec_fingerprint`.
7. **Backend lowering notes.** Same as the Prior axis section below — Stan-only, ignored
   entirely by `nlmixr2_emitter.py`. See [Prior axis](#prior-axis) for per-family detail.
8. **Known limitations.** Every other prior-validation gap (duplicate prior target,
   justification below minimum length, malformed DOI) called via `priors.py` directly
   remains prose/exception-based, not `FrmCode`-coded (item 4). `fmt`/`diff` canonicalize
   `priors:` entry order (see `apmode.dsl.serializer`) so reordering priors never shows
   as a diff.

---

## Absorption axis

### IVBolus

1. **Synopsis.** IV bolus dosing — no absorption phase; dose enters the central
   compartment directly.
2. **State variables introduced.** None. No depot compartment is emitted (both R and Stan
   emitters special-case this: `_needs_depot(spec)` in `stan_emitter.py` returns `False`;
   `nlmixr2_emitter._emit_ode_dynamics` skips the depot branch entirely).
3. **Parameters.** None (`structural_param_names()` contributes nothing for `IVBolus`).
4. **ODE / algebraic contribution.** `_abs_influx = ""` (nlmixr2) / `"0"` (Stan); the dose
   event must route directly into the central compartment via `CMT=1` in the NONMEM-style
   event table. No absorption term appears in `d/dt(centr)`.
5. **Observation contribution.** N/A (absorption axis).
6. **Backend lowering notes.** Supported by both `nlmixr2_emitter.py` and
   `stan_emitter.py`. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented in `validator.py`.
8. **Lane admissibility.** Not restricted by `_LANE_ABSORPTION_INADMISSIBLE` in any lane.
9. **Supported transforms.** `SwapModule` (as `new_module` for `position="absorption"`,
   validated by `_validate_swap_position`). No dedicated narrow transform targets
   `IVBolus` directly.
10. **Known limitations.** None documented.

### FirstOrder

1. **Synopsis.** First-order absorption via a single depot compartment with rate
   constant `ka`.
2. **State variables introduced.** `depot`.
3. **Parameters.** `ka` — calibration value, dimension Rate (1/time); declared with no
   inline value (`absorption: FirstOrder(ka)`) and must have a matching entry in
   `initial:` (`FrmCode.AST_INITIAL_VALUE_MISSING` otherwise). When a `units:` block is
   present, `ka`'s Rate dimension is checked by `apmode.dsl.units` against the declared
   `time` unit (see [`units:` block](#units-block)).
4. **ODE / algebraic contribution.**
   - nlmixr2: `d/dt(depot) <- -ka * depot`; influx to central `= ka * depot`. When the
     spec does not require ODE mode (`needs_ode(spec)` false), the emitter instead uses
     the `linCmt()` shorthand (`_emit_lincmt_dynamics`) rather than an explicit ODE.
   - Stan: `dydt[1] = -ka * depot;`; same influx term. Stan additionally has a closed-form
     analytical superposition path (`_emit_analytical_solve`) for `OneCmt` + `FirstOrder`
     / `LaggedFirstOrder` and for `TwoCmt` + `FirstOrder` under `LinearElim`, used instead
     of `ode_rk45` when applicable.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Supported by both emitters, including the analytical
   fast-path in Stan. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented in `validator.py` beyond `ka > 0`
   (`_positive`).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule`; `ToggleLag` (on: `FirstOrder → LaggedFirstOrder`
   with `tlag=0.5`); `AddParallelRoute` (`FirstOrder(ka) → ParallelFirstOrder(ka1=ka, ka2,
   frac)`, requires current absorption to be exactly `FirstOrder` — `validate_transform`
   rejects otherwise).
10. **Known limitations.** None documented.

### ZeroOrder

1. **Synopsis.** Zero-order (constant-rate) absorption over a fixed duration `dur`.
2. **State variables introduced.** None as a separate depot state; rxode2's `dur(<cmt>)`
   modeled-duration infusion mechanism drives the central compartment directly
   (nlmixr2). No depot compartment is created.
3. **Parameters.** `dur` — calibration value, dimension Time; declared with no inline
   value and must have a matching `initial:` entry.
4. **ODE / algebraic contribution.**
   - nlmixr2: `dur(<cmt>) <- dur` where `<cmt>` is resolved by `_central_cmt_name` (`centr`
     for `OneCmt`/`TwoCmt`/`ThreeCmt`, `Atot` for TMDD distributions — hardcoding `centr`
     would fail rxode2 compilation under TMDD, per inline comment `#13`). `_abs_influx = ""`
     since the infusion mechanism handles influx implicitly.
   - Stan: `ZeroOrder` under ODE mode is explicitly **not supported** —
     `stan_emitter.emit_stan` raises `NotImplementedError` for `ZeroOrder` (and
     `MixedFirstZero`, `Erlang`, `ParallelFirstOrder`, `SumIG`) whenever `_needs_ode(spec)`
     is true, directing the caller to the nlmixr2 backend instead.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 only; Stan raises `NotImplementedError`. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented beyond `dur > 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** No Stan/Torsten lowering (nlmixr2-only today).

### LaggedFirstOrder

1. **Synopsis.** First-order absorption with a lag time `tlag` before absorption begins.
2. **State variables introduced.** `depot`.
3. **Parameters.** `ka` (Rate), `tlag` (Time) — both calibration values, declared with no
   inline value; both must have matching `initial:` entries.
4. **ODE / algebraic contribution.**
   - nlmixr2: `alag(depot) <- tlag` then `d/dt(depot) <- -ka * depot`; influx `= ka *
     depot`. Under the `linCmt()` (non-ODE) path, `alag(depot) <- tlag` is emitted
     alongside `cp <- linCmt()`.
   - Stan: the ODE RHS remains first-order (`dydt[1] = -ka * depot;`, influx `ka * depot`),
     and the event loop delays dose application by `tlag_i` before adding dose amount to the
     depot. The analytical superposition path applies the same lag with
     `t_eff = fmax(t_since_dose - tlag_i, 0)`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** See `src/apmode/dsl/capabilities.py` for current support
   status.
7. **Identifiability caveats.** None documented beyond `ka > 0`, `tlag >= 0`
   (`_non_negative`).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule`; `ToggleLag` (off: `LaggedFirstOrder →
   FirstOrder()`, `ka` carried forward unchanged under the same `initial:` key,
   `tlag` pruned from `initial:` — Formular sharpening plan §4 Phase 1, P1.4).
10. **Known limitations.** Reset+dose (`EVID=4`) events in Stan ODE mode are delayed as a
    combined event under `LaggedFirstOrder`; plain dose events (`EVID=1`) carry the intended
    lag semantics.

### Transit

1. **Synopsis.** Transit-compartment absorption (Savic et al. 2007) — an `n`-compartment
   chain with rate `ktr` feeding a first-order depot with rate `ka`.
2. **State variables introduced.** nlmixr2 exposes `depot` (rxode2's `transit(n, mtt)`
   intrinsic handles the transit chain's internal states implicitly). Stan emits explicit
   `transit_1..transit_n` states plus a terminal `depot`.
3. **Parameters.** `n` — **structural**, declared inline on the module
   (`absorption: Transit(n=5, ktr, ka)`), never estimated and never in `initial:`. `ktr`
   (Rate), `ka` (Rate) — both **calibration** values, must have `initial:` entries.
4. **ODE / algebraic contribution.**
   - nlmixr2: `d/dt(depot) <- transit(n, mtt) - ka * depot` where `mtt <- (n + 1) / ktr`
     is the mean transit time fed to rxode2's `transit()` intrinsic.
   - Stan: no `transit()` intrinsic exists, so `_emit_ode_dynamics` emits an explicit
     integer chain: `dydt[1] = -ktr * transit_1`, middle states
     `ktr * transit_{i-1} - ktr * transit_i`, terminal depot
     `dydt[n+1] = ktr * transit_n - ka * depot`, and central influx `ka * depot`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters support `Transit`, but via materially
   different lowerings (rxode2 intrinsic vs explicit integer chain; see item 4). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented in `validator.py` beyond positivity
   constraints (`n >= 1` int, `ktr > 0`, `ka > 0`).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule`; `SetTransitN` (requires current absorption to
   be `Transit`, changes only `n`, preserves `ktr`/`ka`); `ConvertTransitToErlang`
   (requires `Transit`, drops terminal `ka`, locks `n` to an integer, inherits `ktr`).
10. **Known limitations.** nlmixr2 and Stan approximate the transit chain differently
    (item 4) — cross-paradigm parameter comparability for `Transit` absorption is not
    guaranteed to be numerically identical between backends.

### MixedFirstZero

1. **Synopsis.** Mixed first-order + zero-order absorption: fraction `frac` of dose goes
   through a first-order depot, the remainder enters central as a duration-controlled
   zero-order input.
2. **State variables introduced.** `depot_fo`; the zero-order route uses event-level
   `dur(centr)`/`f(centr)` rather than a synthetic depot.
3. **Parameters.** `ka` (Rate), `dur` (Time), `frac` (Unitless, unit interval) — all
   three are **calibration** values, declared with no inline value, each requiring an
   `initial:` entry.
4. **ODE / algebraic contribution.**
   - nlmixr2: `d/dt(depot_fo) <- -ka * depot_fo`; `f(depot_fo) <- frac`;
     `dur(centr) <- dur`; `f(centr) <- 1 - frac`; influx `= ka * depot_fo`.
   - Stan: **not supported** in ODE mode — `emit_stan` raises `NotImplementedError`
     unconditionally for `MixedFirstZero` (see the `ZeroOrder` entry above; same guard
     clause).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 only. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** `frac == 1.0` (perfect first-order bioavailability, no
   zero-order leg) or `frac == 0.0` produces a singular logit transform
   (`log(frac / (1 - frac))`); the emitter clamps `frac` to `[1e-4, 1 - 1e-4]` and emits an
   inline comment documenting the clamp (`nlmixr2_emitter.py` inline note, referenced as
   `APMODE #14`). This is an emitter-level numerical safeguard, not a validator rejection
   — `_unit_interval` in `validator.py` already requires `frac` strictly in `(0, 1)` at
   the AST level, so the clamp only matters for values extremely close to the boundary.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** No Stan/Torsten lowering.

### Erlang

1. **Synopsis.** Explicit `n`-compartment Erlang chain absorption with shared rate `ktr`
   and no terminal first-order `ka` step (ADR-0003 D2) — distinct from `Transit`, which
   uses rxode2's gamma-interpolated `transit()` intrinsic plus a terminal `ka`.
2. **State variables introduced.** `E1 ... En` (one state per chain link).
3. **Parameters.** `ktr` (Rate) is the only estimable/variability-eligible parameter —
   **calibration**, declared with no inline value, requires an `initial:` entry. `n` is
   **structural**, declared inline with a value (`absorption: Erlang(n=4, ktr)`), set
   only via the `ConvertTransitToErlang` transform (not estimated, not exposed to
   IIV/priors/covariates, and excluded from `calibration_param_names()` — never appears
   in `initial:` — `structural_param_names()` contributes only `"ktr"` for `Erlang`).
4. **ODE / algebraic contribution.**
   - nlmixr2: for `i in 1..n`: `d/dt(E1) <- -ktr * E1`; for `i > 1`, `d/dt(Ei) <- ktr *
     E(i-1) - ktr * Ei`; influx to central `= ktr * E{n}`.
   - Stan: **not supported** in ODE mode — `emit_stan` raises `NotImplementedError`
     unconditionally for `Erlang`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 only. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** `n` capped at `_ERLANG_MAX_N = 7` in `validator.py`
   (`_validate_absorption` → explicit `erlang_max_n` violation) — documented rationale:
   "longer chains add little resolution and inflate state count" (quadratically, per the
   `Erlang` class docstring in `ast_models.py`); the validator directs users to `Transit`
   for `n > 7`.
8. **Lane admissibility.** Not restricted by `_LANE_ABSORPTION_INADMISSIBLE` (only
   `SumIG` is currently lane-restricted).
9. **Supported transforms.** `SwapModule`; `ConvertTransitToErlang` is the *only* path the
   agent has to reach `Erlang` from a fresh spec per the transform's own docstring ("the
   agent's only path to Erlang" — ADR-0003 D2, bounded search-space expansion). No
   dedicated transform mutates an existing `Erlang`'s `n`/`ktr` in place other than
   replacing the whole module via `SwapModule`.
10. **Known limitations.** No Stan/Torsten lowering; `n` is not variability- or
    prior-eligible.

### ParallelFirstOrder

1. **Synopsis.** Two simultaneous first-order absorption routes — fast (`ka1`) and slow
   (`ka2`) — with fraction `frac` of dose entering the fast route (Pumas PK43; Soufsaf
   2021). Distinct from `MixedFirstZero` (first-order + zero-order).
2. **State variables introduced.** `depot_fast`, `depot_slow`.
3. **Parameters.** `ka1` (Rate), `ka2` (Rate), `frac` (Unitless, unit interval) — all
   three **calibration**, declared with no inline value, each requiring an `initial:`
   entry.
4. **ODE / algebraic contribution.**
   - nlmixr2: `d/dt(depot_fast) <- -ka1 * depot_fast`; `d/dt(depot_slow) <- -ka2 *
     depot_slow`; `f(depot_fast) <- frac`; `f(depot_slow) <- 1 - frac`; influx `= ka1 *
     depot_fast + ka2 * depot_slow`.
   - Stan: **not supported** in ODE mode — raises `NotImplementedError`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 only. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** `frac` clamped to `[1e-4, 1 - 1e-4]` before the logit
   transform for the same reason as `MixedFirstZero` (see item 7 there); no distinctness
   constraint between `ka1` and `ka2` is enforced (unlike `SumIG`'s `MT_1 < MT_2`
   label-switching guard) — **not documented as a caveat in `validator.py`**, so if
   `ka1 == ka2` label-switching is possible but not currently validated against; noted
   here as an observed gap, not invented.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule`; `AddParallelRoute` is the only transform that
   constructs `ParallelFirstOrder` (from `FirstOrder`, requires current absorption to be
   exactly `FirstOrder`).
10. **Known limitations.** No Stan/Torsten lowering; no validator-level distinctness
    guard between `ka1`/`ka2`.

### SumIG

1. **Synopsis.** Sum of two Inverse Gaussian absorption-rate components — a closed-form
   analytical input-rate function (Csajka 2005; Weiss & Wegner 2022) rather than a
   compartmental ODE chain.
2. **State variables introduced.** None — the input rate `sumig_input` is an algebraic
   function of time `t`, not a compartment state.
3. **Parameters.** `k` — **structural**, declared inline with a value
   (`absorption: SumIG(k=2, MT_1, MT_2, RD2_1, RD2_2, weight_1)`; hard-restricted to
   `{1, 2}` in v0.7 — not estimated, not variability-eligible, excluded from
   `calibration_param_names()`). `MT_1`, `MT_2` (Time, mean transit times per component);
   `RD2_1`, `RD2_2` (intended as relative dispersion-squared terms per component; exact
   dimension not stated in code and not in `_ROLE_DIMENSIONS`, so `units:` coverage
   reports these as "unchecked" rather than guessing); `weight_1` (Unitless, unit
   interval; `weight_2 = 1 - weight_1` is implicit and not stored on the AST) — these five
   are all **calibration** values, declared with no inline value, each requiring an
   `initial:` entry.
4. **ODE / algebraic contribution.**
   - nlmixr2 only: emits the closed-form density
     `I(t) = D·F · Σᵢ wᵢ · sqrt(RD2ᵢ / (2π·t³)) · exp(-RD2ᵢ·(t-MTᵢ)² / (2·MTᵢ²·t))` as R
     code (`ig_1`, `ig_2`, `sumig_input`), guarded by a `_t_safe <- ifelse(t > 1e-6, t,
     1e-6)` floor to avoid a `0^(-3/2)` singularity at `t=0` in rxode2's LSODA output-time
     grid evaluation. Influx to central `= SUMIG_DOSE * sumig_input`; the nlmixr2 data
     adapter supplies `SUMIG_DOSE` as a persistent per-subject single-dose scalar because
     rxode2's reserved `amt` is not available after the event row.
   - Stan: **not supported** — `emit_stan` raises `NotImplementedError` unconditionally
     for `SumIG`. The docstring in `stan_emitter.py` explains the deferral is not a
     simple omission: "Torsten user-defined ODE RHS does not have access to arbitrary
     t-forcing without time-varying covariate plumbing" (ADR-0003 D4).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 only, single-dose only in v0.7 (multi-dose
   superposition explicitly deferred per ADR-0003 D4). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats (from `validator.py::_validate_sumig`, real, documented):**
   - `k` restricted to `{1, 2}` for v0.7 (`sumig_k_range` violation otherwise); path to
     `k=3` is described as "gated behind the sumig_max_k policy knob" (ADR-0003 D1) but no
     such policy knob exists in code today — **TBD — future release**.
   - `MT_1 < MT_2` required when `k >= 2` (`sumig_mt_ordering` violation) — a
     positive-difference parameterisation (`MT_2 = MT_1 + exp(ldelta_MT_2)` in the
     emitter) that prevents label switching during FOCEI, since without a canonical
     ordering the same density is reachable from two parameter combinations.
   - When `k >= 2`, disposition parameters (`CL`/`V`/`V1`/`V2`/`V3`/`Q`/`Q2`/`Q3` —
     whichever are present in the compiled spec) must be fixed externally
     (`sumig_disposition_fixed` violation otherwise) — checked via
     `_disposition_priors_fixed`, which requires every present disposition parameter to
     have a `PriorSpec` with `source="fixed_external"`. The validator note explicitly says
     the dispatch-time `EvidenceManifest.disposition_fixed` flag is a separate,
     out-of-validator-scope check; the spec-side fallback here is the fixed-external prior
     tag only.
8. **Lane admissibility.** Explicitly rejected in the Submission lane
   (`_LANE_ABSORPTION_INADMISSIBLE[Lane.SUBMISSION] = frozenset({"SumIG"})`,
   `_validate_lane_absorption_admissibility`) — rationale in the error message: "SumIG
   academic-grade; not yet standard regulatory practice." Admissible in Discovery and
   Optimization lanes without restriction.
9. **Supported transforms.** `SwapModule` (initial placement of `SumIG`); `SetSumIGComponents`
   (requires current absorption to already be `SumIG`, updates `MT_1`/`MT_2`/`RD2_1`/
   `RD2_2`/`weight_1` in place, preserves `k`; re-enforces `MT_1 < MT_2` at
   `validate_transform` time before it ever reaches `_validate_sumig`).
10. **Known limitations.** No Stan/Torsten lowering (ADR-0003 D4, architectural, not
    incidental); single-dose only (multi-dose superposition deferred); `k` capped at 2;
    Submission-lane inadmissible by design.

### NODEAbsorption

1. **Synopsis.** Neural-ODE absorption module — a learned, constrained sub-function
   replacing the depot/absorption phase (Discovery/Optimization lanes only).
2. **State variables introduced.** `dim` latent NODE states (backend-specific; not
   exposed as named compartments in the DSL AST itself — the JAX/Diffrax runner owns the
   concrete state representation).
3. **Parameters.** `dim` (int) and `constraint_template` (categorical) — both
   **structural**, declared inline with values
   (`absorption: NODE_Absorption(dim=4, constraint_template=bounded_positive)`), never in
   `initial:`. One input-layer weight per dimension, contributed to
   `structural_param_names()` as `node_abs_w0 ... node_abs_w{dim-1}` (Bräm-style hybrid,
   PRD §4.2.4 layout) so downstream `Variability` items targeting NODE weights pass
   `_validate_variability` instead of being rejected on a `valid_params` miss (documented
   inline as `APMODE #11`) — these weight names are explicitly **excluded** from
   `calibration_param_names()` (no DSL primitive gives them an `initial:` value; that is a
   NODE-backend concern, not the `initial:` block's).
4. **ODE / algebraic contribution.** Neither `nlmixr2_emitter.py` nor `stan_emitter.py`
   lowers `NODEAbsorption` — both raise `NotImplementedError` at the top of `emit_stan`/
   `emit_nlmixr2` whenever `spec.has_node_modules()` is true, before reaching any
   module-specific branch. The actual ODE contribution is defined inside the JAX/Diffrax
   hybrid ODE system (`src/apmode/backends/node_ode.py::HybridPKODE`), which is a
   separate code path from the DSL emitters described in this document — **TBD — full
   description of the JAX ODE contribution is out of scope for the nlmixr2/Stan-emitter
   semantics this document currently covers; a dedicated NODE semantics section is future
   work.**
5. **Observation contribution.** N/A (absorption axis).
6. **Backend lowering notes.** Not lowered by either R/Stan emitter. Trained via
   `src/apmode/backends/node_runner.py` (JAX/Equinox/Diffrax), which is an implemented,
   working backend as of this writing — not a stub — but it is architecturally separate
   from the DSL-compiler-to-R/Stan lowering path this document otherwise describes. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** `dim` must not exceed the `constraint_template`'s maximum
   dimension (`_TEMPLATE_MAX_DIM`: `monotone_increasing`=4, `monotone_decreasing`=4,
   `bounded_positive`=6, `saturable`=4, `unconstrained_smooth`=8) — violation:
   `node_template_max_dim`.
8. **Lane admissibility.** Rejected outright in Submission
   (`node_lane_admissibility` violation, `_validate_node_constraints`); dimension ceiling
   in Discovery is 8, in Optimization is 4 (`_LANE_DIM_CEILING`) — violation:
   `node_lane_dim_ceiling` if exceeded.
9. **Supported transforms.** `SwapModule`; `ReplaceWithNODE` (`position="absorption"`,
   the primary agent-facing path to construct a `NODEAbsorption` from any prior
   absorption module; `dim` capped at the transform level to `[1, 8]` via Pydantic
   `Field(ge=1, le=8)`, independent of the lane/template caps enforced later by the
   validator).
10. **Known limitations.** No DSL-emitter (nlmixr2/Stan) lowering by architectural design
    — NODE backends use a dedicated JAX/Diffrax emitter, not the Formular compiler
    described in the rest of this document.

---

## Distribution axis

### OneCmt

1. **Synopsis.** One-compartment distribution.
2. **State variables introduced.** `centr` (central compartment amount).
3. **Parameters.** `V` — **calibration** value, dimension Volume; declared with no
   inline value (`distribution: OneCmt(V)`), requires an `initial:` entry.
4. **ODE / algebraic contribution.** `d/dt(centr) <- <abs_influx> - <elim_expr>`;
   `cp <- centr / V`. Under linear elimination + non-lag-requiring absorption, both
   emitters may bypass the ODE entirely: nlmixr2 emits `cp <- linCmt()`
   (`_emit_lincmt_dynamics`); Stan uses the closed-form superposition path
   (`_emit_analytical_solve`) for `FirstOrder`/`LaggedFirstOrder` absorption under
   `LinearElim`.
5. **Observation contribution.** N/A (distribution axis).
6. **Backend lowering notes.** Supported by both emitters, including analytical
   fast-paths. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented beyond `V > 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only (no dedicated distribution-axis narrow
   transform exists in `transforms.py` today — distribution changes are always whole-module
   swaps).
10. **Known limitations.** None documented.

### TwoCmt

1. **Synopsis.** Two-compartment distribution (central + one peripheral).
2. **State variables introduced.** `centr`, `periph`.
3. **Parameters.** `V1` (Volume, central), `V2` (Volume, peripheral), `Q`
   (intercompartmental clearance, intended dimension Volume/Time) — all three
   **calibration**, declared with no inline value (`distribution: TwoCmt(V1, V2, Q)`),
   each requiring an `initial:` entry. `V1`/`V2` have a `_ROLE_DIMENSIONS` entry in
   `apmode.dsl.units` (Volume); `Q` does not, so it reports "unchecked" under a `units:`
   declaration (Phase 2 candidate to extend the role table).
4. **ODE / algebraic contribution.**
   `d/dt(centr) <- <abs_influx> - <elim_expr> - Q/V1*centr + Q/V2*periph`;
   `d/dt(periph) <- Q/V1*centr - Q/V2*periph`; `cp <- centr / V1`. Stan additionally has a
   closed-form analytical path for `TwoCmt` + `FirstOrder` + `LinearElim`
   (`_emit_analytical_solve`), including an explicit flip-flop-singularity guard: when
   `ka_i` is within `1e-6 * max(1, ka_i)` of either hybrid rate constant `a1`/`a2` (or
   `a1 ≈ a2`), the Stan model calls `reject(...)` for that subject rather than silently
   producing NaNs, per inline comment `#15`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters; Stan analytical fast-path has a documented
   flip-flop guard (falls back to `reject()`, not a silent numerical failure). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** `V1`, `V2`, `Q` must be `> 0`; no cross-parameter
   identifiability check (e.g. flip-flop `ka ≈ a1/a2`) exists in `validator.py` itself —
   the flip-flop guard is emitter-level (Stan analytical path only, item 4), not a
   validator rejection.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** Flip-flop near-singularity is guarded only in the Stan
    analytical solve path; nlmixr2's `linCmt()`/ODE paths and Stan's own ODE (non-analytical)
    path have no equivalent guard — **TBD — Phase N** whether this is an intentional
    scope limit or a gap.

### ThreeCmt

1. **Synopsis.** Three-compartment distribution (central + two peripheral).
2. **State variables introduced.** `centr`, `periph1`, `periph2`.
3. **Parameters.** `V1`, `V2`, `V3` (Volume), `Q2`, `Q3` (intercompartmental clearances,
   intended dimension Volume/Time) — all five **calibration**, declared with no inline
   value, each requiring an `initial:` entry. `V1`/`V2`/`V3` have a `_ROLE_DIMENSIONS`
   entry (Volume); `Q2`/`Q3` do not, so they report "unchecked" under a `units:`
   declaration (same gap as `TwoCmt.Q`).
4. **ODE / algebraic contribution.**
   `d/dt(centr) <- <abs_influx> - <elim_expr> - Q2/V1*centr + Q2/V2*periph1 -
   Q3/V1*centr + Q3/V3*periph2`; `d/dt(periph1) <- Q2/V1*centr - Q2/V2*periph1`;
   `d/dt(periph2) <- Q3/V1*centr - Q3/V3*periph2`; `cp <- centr / V1`. Both emitters use
   explicit ODE only — no analytical/linCmt or superposition shortcut is implemented for
   `ThreeCmt` in either emitter.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters, ODE-only (no analytical fast-path). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented beyond positivity of `V1`, `V2`, `V3`,
   `Q2`, `Q3`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** No analytical/superposition fast-path in either emitter (ODE
    solve required in all cases).

### TMDDCore

1. **Synopsis.** Full target-mediated drug disposition binding model (Mager & Jusko
   2001).
2. **State variables introduced.** `centr` (aliased `Atot`-style naming is *not* used
   here — `TMDDCore`'s central state is named `centr`, unlike `TMDDQSS`; see the
   `_central_cmt_name` helper), `R` (free receptor concentration), `RC` (drug-receptor
   complex concentration).
3. **Parameters.** `V` (Volume, central), `R0` (concentration, baseline receptor level),
   `kon` (1/(concentration·time), association rate), `koff` (1/time, dissociation rate),
   `kint` (1/time, internalization rate) — all five **calibration**, declared with no
   inline value, each requiring an `initial:` entry. Only `V` has a `_ROLE_DIMENSIONS`
   entry in `apmode.dsl.units` today; `R0`/`kon`/`koff`/`kint` have no role-dimension
   lookup, so a `units:`-declared spec reports them as "unchecked," not "checked" or
   "mismatched" (Phase 2 candidate to extend the role table).
4. **ODE / algebraic contribution.**
   `L <- centr/V` (drug concentration); `d/dt(centr) <- <abs_influx> - kel*centr -
   kon*L*R*V + koff*RC*V`; `d/dt(R) <- ksyn - kdeg*R - kon*L*R + koff*RC`;
   `d/dt(RC) <- kon*L*R - koff*RC - kint*RC`; initial condition `R(0) <- R0`;
   `cp <- centr/V`. Derived rates: `kdeg <- koff` (receptor degradation approximated by
   `koff`'s initial estimate — an explicit approximation, per inline comment, not a
   distinct estimated parameter), `ksyn <- kdeg * R0` (synthesis at steady state),
   `kel <- CL/V` (requires `CL` from a `LinearElim` elimination module — see item 7).
   Both nlmixr2 and Stan emit materially the same equations (Stan's version operates on
   `y[]`-indexed states rather than named R variables but is algebraically identical).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** TMDD distribution modules apply the selected classical
   elimination module to free-drug amount in both emitters. `NODEElimination` remains
   incompatible because no NODE+TMDD lowering exists. All five `TMDDCore` parameters must
   be `> 0`.
8. **Lane admissibility.** Not restricted by `_LANE_ABSORPTION_INADMISSIBLE` (that map is
   absorption-axis only); no distribution-axis lane restriction exists in `validator.py`
   today.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** `kdeg` is approximated as equal to `koff`'s initial value
    rather than estimated as an independent parameter — documented in the emitter as a
    simplifying approximation, not a bug, but a real modeling limitation worth surfacing
    to a reader comparing this to the full Mager & Jusko (2001) parameterisation.

### TMDDQSS

1. **Synopsis.** TMDD quasi-steady-state approximation (Gibiansky et al. 2008).
2. **State variables introduced.** `Atot` (total drug amount — the central-compartment
   state under this distribution holds *total* drug, i.e. free + bound, not free drug
   only; `_central_cmt_name` returns `"Atot"` for both TMDD variants, and inline comment
   `#16` in the emitter explicitly flags this naming subtlety), `Rtot` (total receptor
   concentration).
3. **Parameters.** `V` (Volume, central), `R0` (concentration, baseline receptor), `KD`
   (concentration, equilibrium dissociation constant — an *approximation* of the true
   quasi-steady-state constant `KSS = (koff + kint)/kon`; the emitter's own inline comment
   states "When kint is significant, KSS > KD; using KD underestimates KSS, which can
   overestimate complex formation and target-mediated elimination" and recommends
   converting to `TMDDCore` if the full `KSS` is needed), `kint` (1/time, internalization
   rate) — all four **calibration**, declared with no inline value, each requiring an
   `initial:` entry. As with `TMDDCore`, only `V` has a `_ROLE_DIMENSIONS` entry; `R0`,
   `KD`, `kint` report "unchecked" under a `units:` declaration.
4. **ODE / algebraic contribution.**
   `KSS <- KD` (documented approximation, item 3); `Ctot <- Atot/V`; algebraic QSS solve:
   `Cfree <- 0.5*((Ctot - Rtot - KSS) + sqrt((Ctot - Rtot - KSS)^2 + 4*KSS*Ctot))`;
   `Rfree <- Rtot*KSS/(KSS + Cfree)`; `RC <- Ctot - Cfree`;
   `d/dt(Atot) <- <abs_influx> - elim(Cfree*V) - kint*RC*V`;
   `d/dt(Rtot) <- ksyn - kdeg*Rfree - kint*RC`; initial conditions `Atot(0) <- 0`,
   `Rtot(0) <- R0`; `cp <- Cfree` (note: the observed/predicted concentration is the
   algebraically-derived free concentration, not `Atot/V`). Derived rates: `kdeg <- kint`
   (approximated equal to `kint`'s initial value), `ksyn <- kdeg * R0`; classical
   elimination is applied to free-drug amount.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters (Stan's ODE solve path additionally computes
   `Cfree`/`Rtot` per-observation-time inline in `_emit_ode_solve`'s TMDDQSS branch, using
   `y_state`-derived `Ctot_n`/`Rtot_n`). See `src/apmode/dsl/capabilities.py` for current
   support status.
7. **Identifiability caveats.** Same NODE-elimination incompatibility as `TMDDCore`
   (item 7 there). `KD ≈ koff/kon` is documented in the class docstring as an
   approximation that "differs from KD when kint > 0," i.e. the QSS parameterisation
   trades off exactness against `TMDDCore` for a lower-dimensional, better-behaved
   estimation problem — a real, literature-grounded (Gibiansky 2008) identifiability
   trade-off, not a code defect. All four parameters must be `> 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** `KSS ≈ KD` approximation (item 3) systematically
    underestimates `KSS` when `kint` is non-negligible relative to `koff` — this is an
    inherent property of the QSS reduction, not something the DSL can fix; users needing
    exact `KSS` must use `TMDDCore` instead.

---

## Elimination axis

### LinearElim

1. **Synopsis.** Linear (first-order) elimination.
2. **State variables introduced.** None (modifies the rate term in the distribution
   module's central-compartment ODE).
3. **Parameters.** `CL` — **calibration** value, dimension Clearance (Volume/Time);
   declared with no inline value (`elimination: Linear(CL)`), requires an `initial:`
   entry.
4. **ODE / algebraic contribution.** Elimination-rate expression `CL/V * centr` (or
   `CL/V1 * centr` for two/three-compartment distributions), subtracted from the central
   compartment's balance in both emitters (`_elimination_rate_expr` in nlmixr2,
   `_stan_elim_expr` in Stan — functionally identical expressions).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters; also the only elimination module compatible
   with the Stan analytical (non-ODE) superposition path and the only one compatible with
   TMDD distributions. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None beyond `CL > 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented.

### MichaelisMenten

1. **Synopsis.** Michaelis-Menten (saturable) elimination.
2. **State variables introduced.** None.
3. **Parameters.** `Vmax` (Amount/Time per `_ROLE_DIMENSIONS`), `Km` (Concentration) —
   both **calibration**, declared with no inline value (`elimination:
   MichaelisMenten(Vmax, Km)`), each requiring an `initial:` entry.
4. **ODE / algebraic contribution.** Rate expression
   `Vmax * (centr/V) / (Km + centr/V)` — both emitters express the MM term in
   concentration units (`centr/vol`) "to ensure dimensional consistency (Km is in
   concentration units)" per the shared docstring in both `_elimination_rate_expr` and
   `_stan_elim_expr`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters; forces `needs_ode(spec) = True`
   (`_emitter_utils.needs_ode`) — no analytical/linCmt shortcut is possible with
   saturable elimination. Compatible with TMDD distributions; the TMDD emitters apply the
   MM term to free-drug amount. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None beyond `Vmax > 0`, `Km > 0`. No sparse-data /
   flat-likelihood warning is implemented for MM elimination under limited concentration
   ranges — **TBD — Phase N** (this is a known pharmacometric concern for MM models in
   general, but nothing in `validator.py` checks for it today; not invented here as
   present).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented beyond forcing ODE mode.

### ParallelLinearMM

1. **Synopsis.** Parallel linear + Michaelis-Menten elimination (dual first-order and
   saturable clearance pathways).
2. **State variables introduced.** None.
3. **Parameters.** `CL` (Clearance), `Vmax`, `Km` (as above) — all three **calibration**,
   declared with no inline value, each requiring an `initial:` entry.
4. **ODE / algebraic contribution.** Rate expression
   `(CL/V*centr + Vmax*(centr/V)/(Km + centr/V))` — the parenthesized sum of the linear
   and MM terms, identical form in both emitters.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters; forces ODE mode. **Explicitly excluded**
   from TMDD compatibility — `_validate_module_compatibility` singles this out: "has CL
   but its MM term is not wired into TMDD dynamics, so allowing it would silently drop
   Vmax/Km" — i.e. `ParallelLinearMM` under a TMDD distribution is rejected even though it
   nominally provides `CL`, specifically because the TMDD ODE templates
   (`_emit_tmdd_core_odes`/`_emit_tmdd_qss_odes`) only ever reference a bare `CL/V` term
   and would silently discard the MM component if it were allowed through. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** `CL`, `Vmax`, `Km` each `> 0`; no additional
   parallel-pathway identifiability check (e.g. distinguishing linear-dominant vs.
   MM-dominant regimes) exists in `validator.py`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** Cannot be combined with `TMDDCore`/`TMDDQSS` distribution
    (validator-enforced, see item 6).

### TimeVaryingElim

1. **Synopsis.** Time-varying elimination — `CL` changes over time according to one of
   three decay forms.
2. **State variables introduced.** None.
3. **Parameters.** `CL` (Clearance) — **calibration**, declared with no inline value
   (`elimination: TimeVarying(CL, decay_fn=exponential)`), requires an `initial:` entry.
   `kdecay` (Rate) — **calibration**, but with a twist: it is not a named field in the
   grammar signature at all (unlike every other calibration parameter); it is purely an
   optional `initial:` key that defaults to `0.1` when omitted
   (`DSLSpec.get_initial("kdecay", default=0.1)`, per the class docstring). `decay_fn`
   (categorical: `"exponential" | "half_life" | "linear"`) — **structural**, declared
   inline with a value, no default — required.
4. **ODE / algebraic contribution.** Three decay-form rate expressions, identical in both
   emitters:
   - `exponential`: `CL * exp(-kdecay * t) / V * centr`
   - `half_life`: `CL / (1 + kdecay * t) / V * centr`
   - `linear`: `max(CL * (1 - kdecay * t), 0) / V * centr` (nlmixr2: `max(...)`; Stan:
     `fmax(...)`) — floored at zero to prevent negative clearance past the zero-crossing
     time `t = 1/kdecay`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters; forces ODE mode. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats (real, from `validator.py` inline commentary on
   `TimeVaryingElim`):** the `linear` decay form's `max(..., 0)` floor produces a C0
   (non-smooth) kink at `t = 1/kdecay`. rxode2's LSODA integrator with FOCEI forward
   sensitivities handles the resulting step-size reduction natively, but "Stan's
   `ode_rk45` error estimator assumes C1 smoothness and may over-refine near the
   zero-crossing, producing HMC divergences at large `kdecay`." When an initial-estimate
   `kdecay` implies the zero-crossing falls within the typical observation window
   (`1/kdecay < t_obs_max`), the Stan emitter still runs, but the validator comment
   explicitly documents *reduced convergence reliability* as an accepted, known
   limitation rather than a rejected configuration — "the math itself is correct...so we
   accept the adjoint limitation." A `t_max`-aware Gate 1 warning is named as a tracked
   follow-on requiring the manifest's `t_obs_max` threaded to the validator, not yet
   implemented.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** Stan HMC convergence reliability degrades for `linear` decay
    when the zero-crossing falls inside the observed time window (item 7); no automated
    warning exists yet.

### NODEElimination

1. **Synopsis.** Neural-ODE elimination module — a learned, constrained sub-function
   replacing the elimination-rate term (Discovery/Optimization lanes only).
2. **State variables introduced.** `dim` latent NODE states (backend-owned, not named DSL
   compartments — see `NODEAbsorption` item 2 for the same caveat).
3. **Parameters.** `dim` (int) and `constraint_template` (categorical) — both
   **structural**, declared inline with values, never in `initial:` (same as
   `NODEAbsorption`). One input-layer weight per dimension, contributed as
   `node_elim_w0 ... node_elim_w{dim-1}` in `structural_param_names()` (mirrors
   `NODEAbsorption`'s `node_abs_w*` naming) — excluded from `calibration_param_names()`
   for the same reason (no DSL primitive gives NODE weights an `initial:` value).
4. **ODE / algebraic contribution.** Not lowered by either DSL emitter — both raise
   `NotImplementedError` via the same `spec.has_node_modules()` guard described under
   `NODEAbsorption`. **TBD** — see `NODEAbsorption` item 4 for the same caveat about the
   JAX/Diffrax runner owning the actual contribution.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Same as `NODEAbsorption` — trained via
   `node_runner.py`/`node_ode.py`/`node_trainer.py`, not compiled through
   `nlmixr2_emitter.py`/`stan_emitter.py`. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** Same `_TEMPLATE_MAX_DIM` ceiling as `NODEAbsorption`
   (`node_template_max_dim` violation).
8. **Lane admissibility.** Same rules as `NODEAbsorption`: rejected outright in
   Submission; ceiling 8 in Discovery, 4 in Optimization (`_LANE_DIM_CEILING`,
   `_validate_node_constraints` iterates over both absorption- and elimination-position
   NODE modules identically).
9. **Supported transforms.** `SwapModule`; `ReplaceWithNODE` (`position="elimination"`).
10. **Known limitations.** Same as `NODEAbsorption` item 10 — no DSL-emitter lowering by
    design.

---

## Variability axis

### IIV

1. **Synopsis.** Inter-individual variability on one or more structural parameters, with
   either a diagonal (independent) or block (correlated) covariance structure.
2. **State variables introduced.** N/A — variability axis items modify how structural
   parameters are back-transformed per-subject, they do not add compartment states.
3. **Parameters.** `params: list[StanIdentifier]` (the structural parameter names this
   IIV block targets — must exist in `spec.structural_param_names()`, verified
   case-insensitively via `normalize_param_name`); `structure: "diagonal" | "block"`.
4. **ODE / algebraic contribution.** Not a direct ODE contribution; enters the
   back-transform of each targeted structural parameter as an additive term on the
   log scale: nlmixr2 `_bt(param, log_name)` appends `+ eta.<param>`; Stan
   `_emit_transformed_parameters_block` appends `+ omega_<param> * eta_raw[i, idx]`
   (non-centered parameterisation). `ini()` block emission differs by structure:
   `diagonal` emits one `eta.<p> ~ 0.1` line per parameter; `block` emits a joint
   `eta.p1 + eta.p2 + ... ~ c(...)` lower-triangular covariance matrix with `0.1` on the
   diagonal and `0.01` off-diagonal initial values (nlmixr2 only — Stan represents block
   IIV via the shared `eta_raw` matrix and per-parameter `omega_<p>` scale, i.e. it does
   not currently emit a genuine correlated block covariance in the `parameters{}` block;
   **TBD — Phase N**, confirm whether Stan's non-centered `eta_raw` matrix actually
   induces correlation or treats each column independently — worth a dedicated code trace
   before asserting either way).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both emitters support `IIV`, but the `block` structure's
   actual correlation semantics may differ between nlmixr2 (explicit covariance matrix)
   and Stan (independent `omega_<p>` per parameter unless a distinct block-covariance
   code path exists that was not found in this reading — see item 4). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats (from `validator.py::_validate_variability`, real,
   documented):**
   - A parameter may appear in **at most one** IIV block across the whole spec
     (`iiv_no_duplicate_params` violation otherwise).
   - `block` structure requires `>= 2` params (`block_min_params` violation otherwise).
   - Every targeted param must resolve to a structural parameter
     (`iiv_param_exists` violation).
   - `Transit`'s `n` topology field is explicitly **disallowed** from IIV
     (`_NO_VARIABILITY_PARAMS = frozenset({"n"})`, `no_variability_on_param` violation) —
     `n` is structural topology, not an estimated parameter with an eta-bearing
     back-transform.
8. **Lane admissibility.** Not restricted by lane.
9. **Supported transforms.** `AdjustVariability` (`action: "add" | "remove" |
   "upgrade_to_block"` — `_apply_adjust_variability` targets the *first* `IIV` block
   containing, or appropriate for, the given param; `upgrade_to_block` only upgrades the
   block actually containing that param, not all blocks).
10. **Known limitations.** Possible Stan/nlmixr2 semantic divergence on `block` structure
    correlation (item 4/6) — flagged as TBD rather than asserted as confirmed broken or
    confirmed correct.

### IOV

1. **Synopsis.** Inter-occasion variability on one or more structural parameters, keyed
   to an occasion specification.
2. **State variables introduced.** N/A.
3. **Parameters.** `params: list[StanIdentifier]`; `occasions: OccasionSpec` (one of
   `OccasionByStudy` / `OccasionByVisit` / `OccasionByDoseEpoch` / `OccasionCustom`).
4. **ODE / algebraic contribution.** nlmixr2: `ini()` block emits
   `eta.iov.<p> ~ 0.05 | occ(<column>)` per param (nlmixr2's pipe syntax for occasion
   binding, requires nlmixr2 >= 2.1 per inline comment; no standalone `occ()` call is
   needed in the `model()` block — `_emit_iov_occasion` emits only a documentation
   comment). Back-transform appends `+ eta.iov.<param>` to the log-scale expression.
   Stan: **fully implemented** — a non-centered, occasion-indexed back-transform
   (Metrum Torsten idiom: `theta[occ] = TV * covariates * exp(eta_iiv + eta_iov[occ])`).
   `emit_stan` declares `N_occ`/`occ[n]` data, `omega_iov_<param>` + a
   `matrix[N_subjects * N_occ, n_iov_params] eta_iov_raw` parameter
   (`to_vector(eta_iov_raw) ~ std_normal()`), and emits `array[N_occ] real <param>_i`
   back-transforms indexed by `(i - 1) * N_occ + occ_k` for every IOV-carrying
   parameter — non-IOV parameters stay plain `real`. Presence of any `IOV` item forces
   the numerical-ODE codegen path even for an otherwise analytically-solvable structural
   model, since `_emit_analytical_solve`'s closed-form multi-dose superposition assumes
   time-invariant per-subject parameters and cannot represent occasion-varying kinetics.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both nlmixr2 and Stan are functional. See
   `src/apmode/dsl/capabilities.py` for current support status (`variability.iov` is
   `supported` for both emitters).
7. **Identifiability caveats.** Each targeted param must resolve to a structural
   parameter (`iov_param_exists` violation); `Transit`'s `n` is disallowed from IOV for
   the same reason as IIV (`no_variability_on_param`, mirrors the IIV check — "the
   nlmixr2 and Stan emitters do not apply IOV eta to Transit `n` either").
8. **Lane admissibility.** Not restricted by lane.
9. **Supported transforms.** `AdjustVariability`'s `action` field carries two IOV-targeting
   values alongside the pre-existing IIV actions (`add`/`remove`/`upgrade_to_block`):
   `add_iov` and `remove_iov`. `add_iov` merges the param into an existing `IOV` item that
   shares the same `occasions` spec (union of `params`, no duplicates), or creates a new
   `IOV` item when none matches; `occasions` defaults to `OccasionByStudy()` when omitted,
   matching `search/candidates.py`'s `force_iov` root-candidate default. Requesting
   `add_iov` for a param that already has IOV under a *different* `occasions` spec is
   rejected by `validate_transform` (a param cannot have two IOV occasion partitions at
   once). `remove_iov` drops the param from any `IOV` item that has it, dropping the item
   entirely once its `params` list empties; removing a param with no existing IOV is a
   no-op, mirroring the pre-existing `remove` action's IIV semantics.
10. **Known limitations.** Stan lowers diagonal IOV only (item 4/6): each
    IOV-carrying parameter's occasion draws are independent
    (`to_vector(eta_iov_raw) ~ std_normal()`), with no correlation structure
    either across occasions or across the IOV etas of different parameters —
    the IOV analogue of the `IIV` section's block-structure gap. Correlated
    IOV is not on the current implementation roadmap.

### CovariateLink

1. **Synopsis.** A covariate effect on a structural parameter, with a specified
   functional form and first-class reference values (Formular sharpening plan §4
   Phase 1, P1.6). Lives in its own top-level `DSLSpec.covariates` list — *not* in
   `DSLSpec.variability`, whose sole item kinds are `IIV`/`IOV` — declared via the
   `covariates: { param <- covariate.form(...) }` arrow-syntax block. Distinguishing
   covariates from IIV/IOV at the top level (rather than as a third
   `VariabilityItem` union member) mirrors the fact that a covariate link is a
   fixed-effect structural relationship, not a random-effect variance-component
   declaration — grouping them under `variability:` conflated two different kinds of
   "things that make a parameter vary across subjects."
2. **State variables introduced.** N/A.
3. **Parameters.** `param: StanIdentifier` (target structural parameter), `covariate:
   StanIdentifier` (data column name), `form: "power" | "exponential" | "linear" |
   "categorical" | "maturation"`, plus the form's own explicit, named reference-value
   fields (exactly the fields required by `form`; every other field must be `None` —
   enforced by `_require_covariate_fields`):
   - `power`: `theta` (coefficient's initial/starting estimate) and `ref` (fixed
     reference covariate value the formula centers on, e.g. `ref=70` for a 70 kg
     reference weight, Anderson & Holford 2008).
   - `exponential` / `linear`: `theta` only (no reference-centering in either formula).
   - `categorical`: `reference` only (the baseline level's *name*, a string — the
     numeric 0/1 encoding of non-reference levels is a data-adapter concern, not a DSL
     one). The coefficient itself is not yet configurable here (Phase 2 candidate).
   - `maturation`: `tm50` and `hill` (initial/starting estimates for the TM50 and
     Hill-exponent parameters respectively).
4. **ODE / algebraic contribution.** Enters the back-transform of the targeted parameter
   as an additive log-scale (or logit-scale, for parameters like `frac` that use
   `_bt_logit`) term, per `form`:
   - `power`: `+ beta * log(covariate / ref)` — `ref` is the declaration's own reference
     value (no hardcoded constant).
   - `exponential`: `+ beta * covariate`.
   - `linear`: `+ log(1 + beta * covariate)`.
   - `categorical`: `+ beta * covariate` (same expression as `exponential` in both
     emitters today — i.e. `categorical` is not currently distinguished from
     `exponential` at the math level, only semantically at the DSL type level; **worth
     flagging as a real observation**, not an invented one).
   - `maturation`: `+ log(covariate^beta / (covariate^beta + TM50^beta))` — introduces an
     additional `TM50_<param>_<covariate>` parameter, initialized from the declaration's
     own `tm50` field. **Not implemented in Stan** — `_covariate_expr` raises
     `NotImplementedError` for `form="maturation"`.
   Coefficient naming convention: `beta_<param>_<covariate>`, initialized from the
   declaration's own `theta` (`power`/`exponential`/`linear`) or `hill` (`maturation`,
   paired with `TM50_<param>_<covariate> <- tm50`) — `categorical` has no configurable
   coefficient yet and keeps the pre-P1.6 hardcoded starting value of `0` (nlmixr2
   `ini()` block); Stan priors default to `Normal(theta, 0.5)` /
   `Normal(hill, 0.5)` matching the same per-form rule, `Normal(0, 1)` for
   `categorical`, all overridable via `spec.priors`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** nlmixr2 supports all five forms; Stan supports
   `power`/`exponential`/`categorical`/`linear` but raises `NotImplementedError` for
   `maturation`. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** No duplicate `CovariateLink` for the same
   `(param, covariate)` pair, checked case-insensitively on both param (via
   `normalize_param_name`) and covariate (`.upper()`) —
   `covariate_link_no_duplicate` violation otherwise. Target param must resolve to a
   structural parameter (`covariate_param_exists`). No identifiability check exists for
   collinear covariates (e.g. two covariates both linked to the same param with highly
   correlated data) — not implemented, not claimed here.
8. **Lane admissibility.** Not restricted by lane.
9. **Fingerprint classification** (`apmode.dsl.canonical`): `param`/`covariate`/`form`
   are structural (`structure_fingerprint` changes if any changes); `theta`/`ref`/`tm50`/
   `hill` are calibration-like (excluded from `structure_fingerprint`, included in
   `spec_fingerprint` only — same treatment as `sigma_prop`/`sigma_add`); `reference`
   (the categorical baseline level name) is structural despite being a field on the
   calibration-carrying model, since it identifies which level is baseline rather than
   a re-estimable numeric value.
10. **Supported transforms.** `AddCovariateLink` (validates target param exists and no
   duplicate link already exists before constructing).
11. **Known limitations.** `categorical` form is mathematically identical to
    `exponential` in both emitters (item 4) — the DSL distinguishes them as a type but not
    (yet) as a distinct equation; `maturation` form has no Stan lowering.

### Occasion specs

*(`OccasionByStudy`, `OccasionByVisit`, `OccasionByDoseEpoch`, `OccasionCustom` —
grouped here since they are not independent variability items but configuration carried
by `IOV.occasions`.)*

1. **Synopsis.** Four ways to define the occasion-grouping column for IOV:
   one-occasion-per-study (no column needed), per-visit, per-dose-epoch, or a
   fully custom column name.
2. **State variables introduced.** N/A.
3. **Parameters.** `OccasionByStudy`: none. `OccasionByVisit` / `OccasionByDoseEpoch` /
   `OccasionCustom`: `column: str` (data column name).
4. **ODE / algebraic contribution.** Resolved by `_get_occasion_column` (nlmixr2 emitter)
   to a concrete column name string used in the `eta.iov.<p> ~ ... | occ(<column>)`
   syntax: `OccasionByStudy` → hardcoded `"STUDY_ID"` (per canonical schema, PRD §4.2.0);
   `OccasionByVisit`/`OccasionByDoseEpoch`/`OccasionCustom` → the user-supplied `column`,
   sanitized via `_sanitize_r_name`; unrecognised types fall back to `"OCC"`.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Both nlmixr2 and Stan (see `IOV` item 4/6 above — Stan's
   IOV back-transform is fully implemented, not merely declared). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None documented at the occasion-spec level; column
   existence in the actual dataset is a data-binding concern, not validated by
   `validator.py`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** No standalone transform mutates an existing `IOV` item's
   `OccasionSpec` in place, but `AdjustVariability`'s `add_iov` action (see `IOV` item 9)
   accepts an optional `occasions: OccasionSpec` — defaulting to `OccasionByStudy()` when
   omitted — so an agent can select the occasion spec at the point a new `IOV` item is
   created.
10. **Known limitations.** No transform changes an *existing* `IOV` item's occasion spec
    without removing and re-adding it under the new spec (`add_iov` under a different
    `occasions` value than an already-covered param is rejected, not merged — see `IOV`
    item 9).

---

## Observation axis

### Proportional

1. **Synopsis.** Proportional residual error model.
2. **State variables introduced.** N/A.
3. **Parameters.** `sigma_prop` — intended dimension dimensionless (fractional
   coefficient of variation on the natural scale).
4. **ODE / algebraic contribution.** N/A (observation axis, not a dynamics contribution).
5. **Observation contribution.**
   - nlmixr2: `cp ~ prop(prop.sd)`.
   - Stan: `dv[n] ~ normal(f[n], sigma_prop * f[n])` for every `n` — the emitter's own
     comment (`#1`) states this Normal-with-proportional-variance form was deliberately
     chosen to "unify proportional likelihood with nlmixr2," calling out that an earlier
     revision used `lognormal` here while using Normal in the BLQ branch, making Stan
     "internally inconsistent AND silently diverged from nlmixr2, invalidating every
     cross-paradigm NLPD comparison (PRD §4.3.1)" — i.e. the current form is a documented
     fix for a real historical cross-paradigm-comparability bug, not an arbitrary choice.
6. **Backend lowering notes.** Both emitters, deliberately kept numerically consistent
   (item 5). `apmode.backends.predictive_summary._observation_error_model` maps this to
   the `compute_npe` `error_model="proportional"` hint (residuals divided by observed
   value) per the CLAUDE.md-documented NPE wiring. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None beyond `sigma_prop > 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented.

### Additive

1. **Synopsis.** Additive residual error model.
2. **State variables introduced.** N/A.
3. **Parameters.** `sigma_add` — intended dimension same units as the observed
   concentration.
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.**
   - nlmixr2: `cp ~ add(add.sd)`.
   - Stan: `dv ~ normal(f, sigma_add)` (vectorized over all `N`, unlike the `Proportional`
     and `Combined` branches which loop per-observation because their variance term
     depends on `f[n]`).
6. **Backend lowering notes.** Both emitters. `_observation_error_model` maps this to
   `error_model="additive"` (raw residual, unscaled — the rc8 default per CLAUDE.md).
   See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None beyond `sigma_add > 0`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented.

### Combined

1. **Synopsis.** Combined proportional + additive residual error model.
2. **State variables introduced.** N/A.
3. **Parameters.** `sigma_prop` (dimensionless), `sigma_add` (concentration units).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.**
   - nlmixr2: `cp ~ prop(prop.sd) + add(add.sd)`.
   - Stan: `dv[n] ~ normal(f[n], sqrt(square(sigma_prop * f[n]) + square(sigma_add)))` per
     observation.
6. **Backend lowering notes.** Both emitters. `_observation_error_model` maps this to
   `error_model="combined"` (`sqrt(obs² + 1²)`-style scaling per CLAUDE.md). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** None beyond both sigmas `> 0`; no check that `sigma_prop`
   and `sigma_add` are jointly identifiable given the observed concentration range (a
   known general pharmacometric concern for combined error models at low concentrations)
   — not implemented in `validator.py`, not claimed here as present.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented beyond the general joint-identifiability
    caveat in item 7 (not validator-enforced).

### BLQM3

1. **Synopsis.** Below-limit-of-quantification handling via the M3 method
   (likelihood-based left-censoring), composing with an underlying residual error model.
2. **State variables introduced.** N/A.
3. **Parameters.** `loq_value` (limit of quantification, concentration units);
   `error_model: "proportional" | "additive" | "combined"` (default `"proportional"`);
   `sigma_prop` (default `0.1`), `sigma_add` (default `0.5`) — **both fields are always
   present on the model regardless of `error_model`** (documented rationale, `#30`: "keeps
   `==` comparisons stable and avoids plumbing `Optional[float]` through every downstream
   consumer"). Use `active_sigmas()` (a method on `BLQM3`) to get only the sigma names
   that actually enter the likelihood for the configured `error_model`, to avoid
   double-counting vestigial defaults in parameter-count/prior-coverage logic.
4. **ODE / algebraic contribution.** N/A. Censoring is data-driven, not a model-block
   function.
5. **Observation contribution.**
   - nlmixr2: censoring is expressed via `CENS`/`LIMIT` data columns, **not** in the
     model-block syntax; the emitted model block carries only a documentation comment
     (`# BLQ M3: set CENS=1 and DV=LLOQ=<loq_value> in data for BLQ obs`) plus the
     composed error model's standard observation line (`cp ~ prop(prop.sd)` /
     `cp ~ add(add.sd)` / both), selected by `error_model`.
   - Stan: explicit per-observation branch —
     observed (`cens[n]==0`): `target += normal_lpdf(dv[n] | f[n], <sigma_expr>)`;
     censored (`cens[n]==1`): `target += normal_lcdf(loq | f[n], <sigma_expr>)` (i.e.
     `P(Y < LOQ)`), where `<sigma_expr>` is `sigma_add`, `sigma_prop*f[n]`, or the combined
     `sqrt(...)` form depending on `error_model`. The `generated quantities` block mirrors
     this exactly for `log_lik` (LOO-CV compatibility).
6. **Backend lowering notes.** Both emitters. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** `loq_value > 0`; the corresponding sigma(s) for the
   selected `error_model` must be `> 0` (checked explicitly in
   `validator.py::_validate_observation` — a zero or negative SD "silently produces
   degenerate likelihoods in the emitters," per inline comment).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented beyond the CENS/LIMIT data-column contract
    (an nlmixr2 API characteristic, not an APMODE limitation per se).

### BLQM4

1. **Synopsis.** Below-limit-of-quantification handling via the M4 method (censoring with
   an explicit positivity constraint — interval censoring on `[0, LOQ]` rather than M3's
   simple left tail).
2. **State variables introduced.** N/A.
3. **Parameters.** Identical shape to `BLQM3`: `loq_value`, `error_model`, `sigma_prop`
   (default `0.1`), `sigma_add` (default `0.5`), same always-present-fields rationale and
   `active_sigmas()` helper.
4. **ODE / algebraic contribution.** N/A; data-driven censoring as with `BLQM3`.
5. **Observation contribution.**
   - nlmixr2: comment `# BLQ M4: set CENS=1, DV=LLOQ=<loq_value>, LIMIT=0 in data`, plus
     the same composed error-model observation line as `BLQM3`.
   - Stan: observed branch identical to `BLQM3`; censored branch differs — M4 uses
     `target += log_diff_exp(normal_lcdf(loq | f[n], <sigma_expr>), normal_lcdf(0 |
     f[n], <sigma_expr>))`, i.e. `P(0 < Y < LOQ)` rather than M3's `P(Y < LOQ)`, correctly
     excluding the physically-impossible negative-concentration mass from the censored
     likelihood.
6. **Backend lowering notes.** Both emitters. See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** Same as `BLQM3` item 7.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SwapModule` only.
10. **Known limitations.** None documented beyond the CENS/LIMIT/LIMIT=0 data-column
    contract.

---

### Multi-analyte `observations:` block (`ObservationEndpoint`)

1. **Synopsis.** Formular sharpening plan §4 Phase 1 (P1.7). An optional, additive
   alternative to the singular `observation:` sugar above: `observations: { <name>: {
   dvid=<int>, prediction=<NAME>, error=<observation_type> }, ... }` declares one or more
   named, DVID-routed analytes, each with its own residual error model. `observation:` and
   `observations:` are **mutually exclusive alternatives for the same required concept** —
   exactly one must appear in a `model { }` (neither present, or both present, fails to
   compile with `FrmCode.AST_MISSING_REQUIRED_BLOCK` / `FrmCode.AST_DUPLICATE_BLOCK`
   respectively). The singular form remains fully supported and is not deprecated — it is
   the common case and is expected to stay so.
2. **State variables introduced.** None by the block itself; `prediction` *references* an
   existing state/output the compiled model already produces (see item 4).
3. **Parameters.** Per entry: `dvid` (int, must be unique across all entries in the same
   block — `FrmCode.AST_OBSERVATIONS_DVID_COLLISION` otherwise), `prediction` (a
   `StanIdentifier` naming a known prediction variable — `FrmCode.
   AST_OBSERVATIONS_PREDICTION_UNKNOWN` if it does not resolve), `error` (any
   `ObservationModule` variant, validated identically to the singular `observation:` field).
4. **Known prediction variables (`DSLSpec.known_prediction_variables()`).** `"C_central"` is
   always valid — the canonical name for the primary disposition compartment's concentration
   (nlmixr2's `cp`). `TMDDQSS` distribution additionally exposes `"C_target_total"` (the
   `Rtot` ODE state — total target/receptor concentration), giving a genuine second analyte
   for TMDD assay designs measuring free drug plus total target (Gibiansky et al. 2008).
   `TMDDCore` does **not** expose an equivalent state today (total target there is the sum of
   two separate states, `R` + `RC`, which no emitter synthesizes as one named output). No
   other structural module exposes a second named prediction state — metabolite/parent-child
   compartment topology does not exist in the DSL (Phase 2 candidate).
5. **Unified accessor (`DSLSpec.observation_endpoints()`).** The one accessor downstream code
   should use instead of branching on which syntax form a spec used. Returns
   `list[ObservationEndpoint]`: when `observations:` was used, the dict's values in
   declaration order; otherwise a single synthetic endpoint named `"default"` with `dvid=1`
   and `prediction="C_central"` wrapping the legacy `observation:` field. Every pre-P1.7
   single-endpoint spec (including every spec compiled before this section existed) produces
   exactly one endpoint through this accessor with identical shape.
6. **`DSLSpec.observation` back-compat proxy.** When `observations:` is used, the mandatory
   `observation` field is synthesized from the *first* entry (insertion order) so every
   pre-existing consumer that reads `spec.observation` directly — `apmode.dsl.canonical`,
   `apmode.dsl.units`, `apmode.bundle.scoring_contract`, `apmode.backends.predictive_summary`
   — keeps working unchanged, treating that first entry as representative. Full
   multi-endpoint awareness in those modules (e.g. per-endpoint NPE/VPC, per-endpoint unit
   coverage) is a **Phase 2 candidate**, not implemented here.
7. **Backend lowering notes — honest per-emitter status (P1.7):**
   - **nlmixr2: full support.** `_emit_sigma_ini` / `_emit_observation_model` iterate
     `observation_endpoints()`; the single-endpoint path is byte-identical to pre-P1.7
     output. For 2+ endpoints, each entry's `prediction` is resolved to its R variable via a
     `_PREDICTION_STATE_NAMES` table (`"C_central"` → `cp`, `"C_target_total"` → `Rtot`), and
     a per-endpoint `<var> ~ <error>` statement is emitted with sigma names suffixed by
     endpoint name (e.g. `prop.sd.plasma`) so multiple endpoints never share a sigma
     variable. **Not covered by this emission**: routing observed data rows to the correct
     endpoint by `dvid` at the data-adapter/runner layer (`Nlmixr2Runner`'s two-layer adapter
     contract currently filters non-PK `DVID` rows *out* rather than routing them to a second
     endpoint) — wiring that end-to-end is a Phase 2 candidate.
   - **Stan: Phase 2 gap, explicitly rejected.** `emit_stan` raises `NotImplementedError`
     immediately when `spec.observations is not None` — every data/parameters/likelihood/
     log_lik helper in the Stan emitter reads the singular `spec.observation` field directly,
     and correct multi-endpoint codegen would need per-endpoint data arrays and likelihood
     terms (a disproportionately large rewrite deferred to Phase 2). Declare a single
     `observation:` block, or use the nlmixr2 backend, for a multi-analyte spec today.
   - **FREM: Phase 2 gap, explicitly rejected.** `emit_nlmixr2_frem` raises
     `NotImplementedError` when `spec.observations is not None`, even though it delegates PK
     model-block emission to nlmixr2's (now multi-endpoint-capable) `_emit_model`. The reason
     is FREM's *own* covariate-observation endpoints are DVID-numbered by declaration order
     (PK first = DVID 1, first covariate = DVID 2, ...; see `_emit_frem_model`), which would
     collide with explicit `observations:` dvids in an unverified way — rejected defensively
     rather than risking a silent DVID clash.
8. **Fingerprinting (`apmode.dsl.canonical`).** `CANONICAL_SCHEMA_VERSION` bumped to `2.2.0`;
   both `structure_fingerprint` and `spec_fingerprint` gained a new `"observations"` key
   (sorted-by-canonical-json list of per-entry projections, empty when unset) alongside the
   unchanged `"observation"` key, so two multi-analyte specs differing only in their
   `observations:` content no longer collide.
9. **Lane admissibility.** Not restricted by lane (P1.7 adds no lane-specific rule).
10. **Supported transforms.** None yet — the agentic transform grammar has not been extended
    for multi-analyte endpoints (Phase 2 candidate).
11. **Known limitations.** Duplicate endpoint *names* within one `observations:` block are
    not flagged as an error (last one wins, since names are a plain dict key in the
    transformer) — only duplicate `dvid`s and unresolvable `prediction`s are checked. Stan
    and FREM multi-endpoint codegen, and end-to-end multi-analyte data routing for nlmixr2,
    remain Phase 2 candidates (see item 7).

---

## Prior axis

Priors (`src/apmode/dsl/priors.py`) are carried on `DSLSpec.priors: list[PriorSpec]` — a
sixth semantic axis. As of Formular sharpening plan §4 Phase 1 (P1.5), priors **do**
have a textual grammar block of their own — the `priors: { ... }` top-level block (see
[`priors:` block](#priors-block-human-authored-prior-syntax) above) — in addition to the
pre-existing Python-API path (`build_prior_spec()` / the `SetPrior` transform); both
paths route through the same canonical factory and produce field-identical
`PriorSpec`s. Consumed by `stan_emitter.py` (injected into the `model{}` block) and
ignored by `nlmixr2_emitter.py`, `node_runner.py`, and the agentic LLM backend unless
explicitly wired elsewhere.

### Prior families overview

Each `PriorSpec` has a `target: str` (resolved against the compiled spec via
`classify_target` into one of `structural | iiv_sd | iov_sd | residual_sd | corr_iiv |
covariate`), a `family: PriorFamily` (one of the ten variants below), a `source:
PriorSource` (`uninformative | weakly_informative | historical_data |
expert_elicitation | meta_analysis | fixed_external`), and `justification`/`doi`/
`historical_refs` fields whose presence is enforced for "informative" sources
(`historical_data`, `expert_elicitation`, `meta_analysis`) by
`PriorSpec.justification_required_for_informative_sources` (non-empty `justification`
required; `historical_data` additionally requires non-empty `historical_refs`) and,
separately, by `validate_prior_justification` (justification length `>= 50` chars,
`doi` matching the Crossref-canonical pattern `^10\.\d{4,9}/...$`) — this second check is
run by the emitter before writing `prior_manifest.json`, not by the Pydantic model itself.
The `(target_kind, family)` parameterization schema is enforced by `_VALID_FAMILIES` in
`priors.py`:

| target_kind    | allowed families |
|----------------|-------------------|
| `structural`   | `Normal`, `LogNormal`, `Mixture`, `HistoricalBorrowing` |
| `iiv_sd`       | `HalfNormal`, `HalfCauchy`, `Gamma` |
| `iov_sd`       | `HalfNormal`, `HalfCauchy`, `Gamma` |
| `residual_sd`  | `HalfNormal`, `HalfCauchy`, `Gamma` |
| `corr_iiv`     | `LKJ` only |
| `covariate`    | `Normal`, `Mixture`, `HistoricalBorrowing` |

Note `InvGamma` is deliberately excluded from every `*_sd` row — inline comment `#29`
explains that `InvGamma` is conventionally a variance prior, and allowing it directly on
an SD-scale target without a matching `sqrt` transform in the Stan emitter "would silently
constrain the variance while the emitter draws on the SD scale"; the comment names a
future `InvGammaOnSquare` wrapper as the intended escape hatch, not yet implemented.
`validate_prior_family`'s rejection message for this case names the variance/SD
distinction and points at the working `MixturePrior`-component path explicitly, rather
than reading like a generic family/target-kind mismatch. `corr_iiv` accepting `LKJ` is
itself only half-wired: the schema accepts it, but `stan_emitter.py` does not declare a
`corr_iiv` correlation-matrix parameter, so `emit_stan` raises a dedicated
`NotImplementedError` — naming the target and the remedy — before any code generation
whenever an `LKJPrior` is declared on `corr_iiv`; it is accepted at the schema level "so
agentic transforms can plan for it," not because correlated-IIV emission is functional
today.

**Backend lowering (applies to all ten families below):** Stan only. `nlmixr2_emitter.py`
does not read `spec.priors` at all — priors have no effect on the nlmixr2/rxode2 lowering
path today. **Lane admissibility:** not restricted by lane in `validator.py` (priors are
validated by `validate_priors`/`validate_prior_family` in `priors.py`, which has no
lane parameter). **Supported transforms (all ten families):** `SetPrior`
(`target`, `family`, `source`, `justification`, `historical_refs`) is the sole transform
that creates or replaces a prior — semantics are declare-or-idempotently-replace
(`validate_set_prior` checks the target resolves and the family matches the
parameterization schema for that target kind before `apply_set_prior` performs the
insert-or-replace).

### NormalPrior

1. **Synopsis.** `Normal(mu, sigma)` on a real-valued (unbounded) target.
2. **State variables introduced.** N/A.
3. **Parameters.** `mu: float`, `sigma: float` (`> 0`, Pydantic `Field(gt=0)`).
4. **ODE / algebraic contribution.** N/A (prior, not a dynamics term).
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ normal(mu, sigma);`. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** Valid for `structural` and `covariate` target kinds only
   (see the table above); rejected by `validate_prior_family` for `iiv_sd`/`iov_sd`/
   `residual_sd`/`corr_iiv` targets.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** nlmixr2 does not consume priors at all (see overview).

### LogNormalPrior

1. **Synopsis.** `LogNormal(mu, sigma)` — positive-valued alternative for structural
   params expressed on the natural scale.
2. **State variables introduced.** N/A.
3. **Parameters.** `mu: float`, `sigma: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: if the target Stan variable is itself already on the
   log scale (`on_log_scale=True`, the case for every structural param in this codebase,
   since `_emit_parameters_block` always declares `real log_<name>;`), `LogNormal(mu,
   sigma)` on the natural-scale variable becomes `Normal(mu, sigma)` on the log-scale
   Stan variable — `_emit_user_prior` performs this transformation automatically. If
   `on_log_scale=False`, it emits `<stan_param> ~ lognormal(mu, sigma);` directly. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** Same target-kind restriction as `NormalPrior`
   (`structural`, `covariate` only).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** None documented beyond the general nlmixr2 non-consumption
    caveat.

### HalfNormalPrior

1. **Synopsis.** `HalfNormal(sigma)` — positive-valued, the default weakly-informative
   family for IIV/IOV/residual SD parameters.
2. **State variables introduced.** N/A.
3. **Parameters.** `sigma: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ normal(0, sigma);`, relying on the
   parameter's `<lower=0>` declaration in the `parameters{}` block to truncate to the
   half-line (Stan does not require an explicit half-normal distribution family; the
   declared bound handles it). Inside mixtures, the correctly-normalized log-density
   `normal_lpdf(x | 0, sigma) + log(2)` is used instead (see `MixturePrior` item 4) — the
   `+log(2)` correction is required so half-family components are not artificially
   down-weighted 50% relative to fully-supported components like `Gamma`/`InvGamma` in a
   mixture (documented rationale in `_component_lpdf`). See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** Valid for `iiv_sd`/`iov_sd`/`residual_sd` target kinds
   only.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** None documented beyond the general nlmixr2 non-consumption
    caveat.

### HalfCauchyPrior

1. **Synopsis.** `HalfCauchy(scale)` — positive-valued, heavy-tailed; used as the default
   prior family throughout the codebase's own convenience constructors
   (`default_iiv_prior`, `default_residual_prior`) and as the Stan emitter's own default
   when no user `SetPrior` exists (`omega_<p> ~ cauchy(0, 1)`, `sigma_prop/sigma_add ~
   cauchy(0, <init>)`).
2. **State variables introduced.** N/A.
3. **Parameters.** `scale: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ cauchy(0, scale);`. Half-Cauchy
   log-density inside mixtures: `cauchy_lpdf(x | 0, scale) + log(2)` (same normalization
   rationale as `HalfNormalPrior`). See `src/apmode/dsl/capabilities.py` for current
   support status.
7. **Identifiability caveats.** Valid for `iiv_sd`/`iov_sd`/`residual_sd` only.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** None documented beyond the general nlmixr2 non-consumption
    caveat.

### GammaPrior

1. **Synopsis.** `Gamma(alpha, beta)` — positive-valued, conjugate for precision
   parameters; also valid on SD-scale targets per the schema (unusual but permitted).
2. **State variables introduced.** N/A.
3. **Parameters.** `alpha: float` (`> 0`), `beta: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ gamma(alpha, beta);`. See
   `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** Valid for `iiv_sd`/`iov_sd`/`residual_sd` only (not
   `structural` or `covariate`, per the schema table).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** None documented beyond the general nlmixr2 non-consumption
    caveat.

### InvGammaPrior

1. **Synopsis.** `InverseGamma(alpha, beta)` — positive-valued; the classical choice for
   variance parameters, but see item 7 for why it is currently excluded from every
   variability/residual target kind.
2. **State variables introduced.** N/A.
3. **Parameters.** `alpha: float` (`> 0`), `beta: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ inv_gamma(alpha, beta);` when used
   directly (e.g. as a `MixturePrior` component); `_component_lpdf` supports
   `inv_gamma_lpdf` for mixture use. See `src/apmode/dsl/capabilities.py` for current
   support status.
7. **Identifiability caveats.** **Not valid for any of the six target kinds as a
   top-level `SetPrior` family** — `_VALID_FAMILIES` excludes `InvGamma` from every row
   (`structural`, `iiv_sd`, `iov_sd`, `residual_sd`, `corr_iiv`, `covariate`). It is only
   reachable indirectly, as a *component* inside a `MixturePrior` (whose components list
   type does include `InvGammaPrior`). This is a deliberate, documented restriction
   (inline comment `#29`, see the prior-axis overview above), not an oversight to be
   "fixed" without also adding the described `sqrt`-transform wrapper.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior` — but will fail `validate_prior_family` for
   every target kind when used as the top-level family (item 7); can only be reached as a
   `MixturePrior` component today.
10. **Known limitations.** Effectively unusable as a standalone `SetPrior` target
    (item 7) — this is the DSL's most restrictive prior family.

### BetaPrior

1. **Synopsis.** `Beta(alpha, beta)` — supported on `[0, 1]`; suited to bioavailability
   `F` or mixing fractions. No current structural parameter is literally named `F`, but
   unit-interval structural params exist today (e.g. `MixedFirstZero.frac`,
   `ParallelFirstOrder.frac`, `SumIG.weight_1`). The prior schema permits `Beta` only on
   known unit-interval structural targets (`frac`, `weight_1`), not on positive
   structural parameters such as `CL` or `V`.
2. **State variables introduced.** N/A.
3. **Parameters.** `alpha: float` (`> 0`), `beta: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `<stan_param> ~ beta(alpha, beta);` — note this
   requires the target Stan parameter itself to be declared on `[0,1]` for the density to
   be meaningful; the DSL/emitter does not cross-check that a `Beta`-prior-bearing
   parameter is actually unit-interval-constrained in the `parameters{}` block (e.g.
   `frac`-typed structural params are represented via a `logit_frac` real-valued Stan
   variable in the nlmixr2 emitter's back-transform convention, and the Stan emitter's
   parameter declarations for structural params are all unconstrained `real log_<name>`
   per `_emit_parameters_block` — meaning a `Beta` prior on a `frac`-like target would be
   placed on an unconstrained real Stan parameter, which is very likely a real
   emitter/prior mismatch worth flagging as **TBD — needs verification**, not asserted as
   confirmed-broken without tracing an actual generated Stan program).
7. **Identifiability caveats.** Valid for `structural`/`covariate` target kinds per the
   schema.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** Possible target/parameterization mismatch for unit-interval
    structural parameters under the Stan emitter's uniformly-unconstrained `real
    log_<name>` parameter declarations (item 6) — flagged as an open question, not a
    confirmed bug.

### LKJPrior

1. **Synopsis.** `LKJ(eta)` on a correlation matrix — `eta=1` is uniform over
   correlation matrices, `eta>1` shrinks toward the identity (independence).
2. **State variables introduced.** N/A.
3. **Parameters.** `eta: float` (`> 0`).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan: `stan_emitter.py` does not declare a `corr_iiv`
   correlation-matrix variable in the `parameters{}` block, so it cannot honour a prior
   on it; the emitter no longer contains any `lkj_corr()` codegen path at all (the dead
   `_emit_user_prior` branch that used to emit `<stan_param> ~ lkj_corr(eta);` against
   that undeclared variable — code that could never have compiled — has been removed).
   Instead, `emit_stan` looks up `corr_iiv`'s prior via `_find_prior` *before* any code
   generation and raises a dedicated `NotImplementedError` naming the target and the
   remedy (remove the prior, or use the nlmixr2 backend for correlated IIV) whenever it
   is an `LKJPrior` — a loud, actionable rejection rather than the prior being silently
   dropped or unconsulted. **Not functional end-to-end**; only the rejection path is
   implemented. See `src/apmode/dsl/capabilities.py` for current support status.
7. **Identifiability caveats.** Only valid target kind is `corr_iiv`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior` — schema-valid but not emitter-functional (item
   6).
10. **Known limitations.** `corr_iiv` is schema-accepted so agentic transforms can plan
    ahead for correlated IIV before the emitter supports it, but declaring an `LKJPrior`
    on it now raises a dedicated `NotImplementedError` at the top of `emit_stan` instead
    of being silently dropped. Correlated (block-structure) IIV emission itself remains
    unimplemented — this closes the *silent-drop* gap, not the underlying feature gap.

### MixturePrior

1. **Synopsis.** A weighted mixture of two or more component prior families — the core
   primitive for robust MAP historical borrowing (Schmidli et al. 2014).
2. **State variables introduced.** N/A.
3. **Parameters.** `components: list[...]` (`>= 2`, each one of `Normal | LogNormal |
   HalfNormal | HalfCauchy | Gamma | InvGamma | Beta` — notably **not** `LKJ`,
   `Mixture`, or `HistoricalBorrowing` themselves, i.e. no nested mixtures and no LKJ
   components); `weights: list[float]` (`>= 2`, validated at construction time by
   `weights_match_components`: must equal `len(components)`, must sum to `1.0` within
   `1e-6`, must all be non-negative).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan only: emits
   `target += log_sum_exp([log(w_1) + lpdf_1(...), log(w_2) + lpdf_2(...), ...]);`
   (Stan User's Guide §13.1 mixture-modeling form). Zero-weight components are dropped
   before emission (`_emit_mixture_prior` filters `w > 0.0`); if *every* component has
   zero weight, `_emit_mixture_prior` raises `ValueError` (a genuinely degenerate
   mixture, not silently emitted as empty code). See `src/apmode/dsl/capabilities.py` for
   current support status.
7. **Identifiability caveats.** Valid for `structural`/`covariate` target kinds only (per
   the schema table) — i.e. cannot mixture-prior an SD or correlation target directly
   (though `HistoricalBorrowingPrior`, which compiles to a `MixturePrior`, is itself
   restricted to log-scale structural targets only — see next entry).
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** No nested mixtures; no `LKJ` mixture components (schema
    restriction, not emitter restriction — the type union itself excludes them).

### HistoricalBorrowingPrior

1. **Synopsis.** A robust MAP (meta-analytic-predictive) prior built from historical
   dataset summaries (Schmidli et al. 2014) — compiles at emit time into a two-component
   `MixturePrior`.
2. **State variables introduced.** N/A.
3. **Parameters.** `map_mean: float`, `map_sd: float` (`> 0`), `robust_weight: float`
   (`>= 0`, `<= 1`, default `0.2`), `historical_refs: list[str]` (`>= 1` entry required).
4. **ODE / algebraic contribution.** N/A.
5. **Observation contribution.** N/A.
6. **Backend lowering notes.** Stan only, via `_emit_historical_borrowing_prior`: builds
   `map_component = Normal(map_mean, map_sd)` and `weak_component = Normal(0.0, 10.0)`
   (chosen, per inline comment, because PK parameters on the log scale can span `V`
   easily exceeding 100 L and `CL` ranging 0.1–100 L/h, so a narrower `Normal(0, 2)` would
   be "too narrow and would penalize true values"), combines them as a `MixturePrior`
   with weights `[1 - robust_weight, robust_weight]`, then delegates to
   `_emit_mixture_prior`. **Hard restriction:** `_emit_historical_borrowing_prior` raises
   `NotImplementedError` when `on_log_scale=False` — i.e. `HistoricalBorrowingPrior` is
   usable *only* on log-scale structural targets, since every structural parameter in
   this codebase is represented log-scale in Stan's `parameters{}` block, this in
   practice means `HistoricalBorrowingPrior` works for `structural` targets and (per the
   schema table permitting it on `covariate` too) would need verification for covariate
   targets specifically, since covariate coefficients (`beta_<param>_<covariate>`) are
   *not* log-scale Stan variables (`_emit_parameters_block` declares them as plain `real
   beta_<p>_<c>;`) — **this is a real, verifiable mismatch**: `HistoricalBorrowingPrior`
   is schema-permitted on `covariate` targets but would hit the `on_log_scale=False`
   `NotImplementedError` branch for any covariate target, since `_emit_user_prior` is
   always called with `on_log_scale=False` for covariate coefficients in
   `_emit_model_block`. Flagged here as confirmed by direct code trace, not speculation.
7. **Identifiability caveats.** `robust_weight` must be in `[0, 1]`; `historical_refs`
   must be non-empty (both enforced at the AST level via Pydantic `Field` constraints,
   not the validator module). The `PriorSpec` model-level check
   (`justification_required_for_informative_sources`) additionally requires non-empty
   `justification` for `source in {historical_data, expert_elicitation, meta_analysis}`,
   and specifically `historical_refs` for `source="historical_data"`.
8. **Lane admissibility.** Not restricted.
9. **Supported transforms.** `SetPrior`.
10. **Known limitations.** Effectively `structural`-target-only in practice despite being
    schema-permitted for `covariate` targets too (item 6) — using it on a covariate
    coefficient will raise `NotImplementedError` at emit time, not silently misfire.

---

## Macros

Formular sharpening plan §4 Phase 2, P2.1. A macro is vetted, closed-registry sugar over
the existing AST — it expresses nothing a hand-authored `variability:`/`priors:` block
could not already say. There is no user-defined-macro extension point in this phase;
`MACRO_REGISTRY` (`src/apmode/dsl/macros/__init__.py`) is populated only by
`src/apmode/dsl/macros/stdlib.py` at import time.

**Syntax.** A top-level `use <dotted.name>` statement (`use_block` in
`pk_grammar.lark`, built on a `DOTTED_NAME: NAME ("." NAME)*` terminal) names one
registered macro. Like `variability_block`, `use_block` is deliberately absent from
every set `apmode.dsl.grammar._validate_block_cardinality` checks, so a spec may
contain zero, one, or many `use` statements with no bespoke cardinality rule. Multiple
`use` statements are expanded in source order; the *same* macro name may not be used
twice in one spec (`FrmCode.AST_MACRO_DUPLICATE_USE`), and a name absent from the
registry is rejected (`FrmCode.AST_MACRO_UNKNOWN`) — see
[`docs/FORMULAR_ERROR_CODES.md`](FORMULAR_ERROR_CODES.md) `FRM-AST-016`/`FRM-AST-017`.
Both are raised as `FormularCompileError` from `apmode.dsl.macros.expand_macros`,
called from `apmode.dsl.grammar.compile_dsl` immediately after the main Lark
transform pass — i.e. a macro sees the fully-assembled spec, including anything the
author already declared by hand.

**The three stdlib macros** (`src/apmode/dsl/macros/stdlib.py`), all currently
versioned `"v1"`:

| Macro name | Version | What it does | No-op / idempotency behavior |
|---|---|---|---|
| `pkstd.standard_iiv` | `v1` | Adds one diagonal `IIV(...)` entry covering every structural parameter (`DSLSpec.structural_param_names()`, excluding `"n"`) not already covered by an existing `IIV`/`IOV` item. NODE `node_abs_w*`/`node_elim_w*` weight names ARE eligible. | Returns `spec` unchanged (does not append an empty `IIV`) if every eligible parameter is already covered. |
| `pkstd.standard_priors` | `v1` | Adds a weakly-informative prior (`source="weakly_informative"`, built only via `apmode.dsl.priors.build_prior_spec` — never hand-constructed) on every structural parameter (again excluding `"n"`) lacking a declared prior. Family: `LogNormal(mu=log(initial) or 0.0, sigma=1.0)` for ordinary structural params; `Normal(0, 1)` for NODE `node_abs_w*`/`node_elim_w*` weights specifically, since a LogNormal default would silently bias those (possibly-negative) weights positive. | Returns `spec` unchanged if every eligible parameter already has a declared prior. |
| `pkstd.standard_error_model` | `v1` | Adds a `HalfNormal(sigma=1.0)` prior on the residual-error SD target(s) implied by `spec.observation`'s type (`Proportional`→`sigma_prop`, `Additive`→`sigma_add`, `Combined`→both; `BLQM3`/`BLQM4` via `.active_sigmas()`) for each target not already declared. | Single-endpoint case: no-op if every implied sigma target already has a prior. **Multi-analyte case** (`spec.observations` set): documented no-op, unconditionally — `classify_target`'s `sigma_prop`/`sigma_add` namespace is flat with no per-endpoint disambiguation, so a per-endpoint expansion could silently collide/overwrite rather than do something correct; extending the DSL with per-endpoint sigma targets is left as a Phase 2+ candidate outside this macro's scope. |

**Provenance: `DSLSpec.macros_used: list[str]`.** Each successful expansion appends
`"{name}@{version}"` (e.g. `"pkstd.standard_iiv@v1"`) to this field, in source order,
recording exactly which macro version ran — relevant if a future release changes what
a given macro name expands to.

**Audit artifact: `compiled_specs/{model_id}/expanded.formular`.** `BundleEmitter.
write_compiled_spec` (`src/apmode/bundle/emitter.py`) writes this file — the
post-expansion spec re-serialized via `serialize_spec` — **iff** `spec.macros_used` is
non-empty. It is a normal pre-seal bundle artifact, not exempted from the sealed-bundle
digest (unlike `_COMPLETE`/`bom.cdx.json`/`sbc_manifest.json`).

**Fingerprint exclusion (verified against `src/apmode/dsl/canonical.py`).**
`macros_used` is excluded from both `structure_fingerprint` and `spec_fingerprint`, and
this required **no** `CANONICAL_SCHEMA_VERSION` bump — confirmed by reading
`_structure_dict`/`_spec_dict`: both hand-build their projection dict from an explicit,
named field allowlist rather than a raw `model_dump()`, so a new top-level `DSLSpec`
field is excluded by construction unless a maintainer deliberately adds it to those two
functions. `CANONICAL_SCHEMA_VERSION` remains `"2.2.0"` (the P1.7 multi-analyte value)
as of this writing. Practical consequence: two specs that are byte-identical after
macro expansion fingerprint identically whether or not either used a `use` shortcut to
get there — "sugar, not semantics," matching the property a macro is supposed to have.

**Known limitation.** No user-defined macros exist in this phase; the registry is a
closed, small, vetted set. Extending it (or adding a spec-author-defined macro
mechanism) is out of scope here and not currently on the plan — **TBD — Phase 3+** if
ever revisited.

---

## Transforms Reference

Formular sharpening plan §4 Phase 2, P2.2. `FormularTransform` (`src/apmode/dsl/
transforms.py`) is a 10-member discriminated union (`Field(discriminator="type")`);
nine members live in `transforms.py`, the tenth (`SetPrior`) in
`src/apmode/dsl/prior_transforms.py`. Every module's own "Supported transforms"
subsection earlier in this document names which of these ten can target it; this
section documents the union itself plus the P2.2 provenance fields.

| Transform | `type` literal | What it does |
|---|---|---|
| `SwapModule` | `"swap_module"` | Replace an entire axis module (absorption/distribution/elimination/observation) with a new one; `initial_overrides` supplies any calibration values the new module needs that the old one didn't have. |
| `AddCovariateLink` | `"add_covariate_link"` | Add a `CovariateLink` (power/exponential/linear/categorical/maturation) to `DSLSpec.covariates`, enforcing the same per-form required/forbidden field contract as hand-authored `covariates:` text. |
| `AdjustVariability` | `"adjust_variability"` | Add, remove, or upgrade-to-block an `IIV` entry for one structural parameter. |
| `SetTransitN` | `"set_transit_n"` | Change the transit-compartment count on a `Transit` absorption module (requires current absorption to already be `Transit`). |
| `ToggleLag` | `"toggle_lag"` | Add or remove lag time on first-order absorption (`FirstOrder` ⇄ `LaggedFirstOrder`). |
| `ReplaceWithNODE` | `"replace_with_node"` | Replace absorption or elimination with a NODE module (Discovery lane only; `dim` ≤ lane ceiling; `constraint_template` from the enumerated set). |
| `ConvertTransitToErlang` | `"convert_transit_to_erlang"` | Convert `Transit` → `Erlang(n, ktr)`, dropping the terminal first-order `ka` step and locking `n` to an integer (ADR-0003 D2; the agent's only path to `Erlang`). |
| `AddParallelRoute` | `"add_parallel_route"` | Convert `FirstOrder` → `ParallelFirstOrder(ka1, ka2, frac)`, splitting a single first-order absorption into fast + slow parallel routes. |
| `SetSumIGComponents` | `"set_sumig_components"` | Set/update `SumIG` component parameters (`MT_1`, `MT_2`, `RD2_1`, `RD2_2`, `weight_1`); requires current absorption to already be `SumIG`; validator enforces `MT_1 < MT_2` (label-switching guard). |
| `SetPrior` | `"set_prior"` | Declare or replace a prior on `target` (append if none exists, replace if one does — idempotent re-declaration); the 7th/10th admissible transform (PRD §4.2.5, Phase 2+). |

**The P2.2 provenance fields, precisely where each one lives:**

- **`rationale: str = ""`** — present on all 9 `transforms.py` members
  (`SwapModule`, `AddCovariateLink`, `AdjustVariability`, `SetTransitN`, `ToggleLag`,
  `ReplaceWithNODE`, `ConvertTransitToErlang`, `AddParallelRoute`,
  `SetSumIGComponents`). **`SetPrior` does not have a separate `rationale` field** — it
  keeps its pre-existing `justification: str = ""` field, which already serves the same
  role for that transform (avoiding a redundant duplicate field); `"rationale" not in
  SetPrior.model_fields`.
- **`expected_diagnostic_effect: list[str] = []`** — present on **all 10** members,
  including `SetPrior`.
- **`applied_at: str | None = None`** — does **not** exist on any `FormularTransform`
  member. It lives only on the two bundle lineage record types in
  `src/apmode/bundle/models.py`: `CandidateLineageEntry` and `SearchGraphEdge`, both of
  which also carry their own `rationale: str | None = None` and
  `expected_diagnostic_effect: list[str] = []` fields (independent of, but populated
  from, the producing transform's fields at the two production call sites in
  `src/apmode/orchestrator/__init__.py` and `src/apmode/backends/agentic_runner.py`).
  `applied_at` is an ISO-8601 timestamp (`datetime.now(tz=UTC).isoformat()`) set when
  the lineage record is written, not when the transform object itself was constructed.

These three fields are pure provenance: `validate_transform(spec, t)` and
`apply_transform(spec, t)` produce identical results whether or not `rationale`/
`expected_diagnostic_effect` are populated on `t`.

---

## CLI: `explain --equations` and `signature`

Formular sharpening plan §4 Phase 2, P2.3 (`explain --equations`) and P2.4
(`signature`). Both live in `src/apmode/cli_formular.py`; both are display/summary
views over an already-compiled `DSLSpec` — neither can affect backend code generation.

### `apmode formular explain --equations`

Renders the symbolic ODE system a spec compiles to, via
`src/apmode/dsl/equations.py::build_equations`/`render_equations`. This module is
**non-authoritative and read-only**: the R/Stan emitters
(`nlmixr2_emitter.py`/`stan_emitter.py`) remain the ground truth for actual numerical
execution; `equations.py` is a one-way mirror (emitter understanding → symbolic view),
and nothing in the emitter modules imports from it. Built with `sympy`
(`sympy.Eq`/`Derivative`/`Function`), following the pattern used by pharmpy's
`CompartmentalSystem` — never by string-templating R/Stan source and re-parsing it.

`EquationSystem` holds `odes` (one `sympy.Eq(Derivative(state(t), t), rhs)` per
differential state), `algebraic` (closed-form / non-differential relations),
`observation_eq` (the primary prediction variable's defining equation, or `None`), and
`notes` (non-obvious composition decisions, rendered as a `Notes:` section by
`render_equations`). Every branch is a line-for-line symbolic translation of the
corresponding nlmixr2-emitter function (`_emit_ode_dynamics`,
`_elimination_rate_expr`, `_emit_tmdd_core_odes`, `_emit_tmdd_qss_odes`), using the
same compartment names (`depot`, `centr`, `periph`, `periph1`, `periph2`, `E1..En`,
`depot_fo`, `depot_fast`/`depot_slow`, `Atot`, `Rtot`, `R`, `RC`).

**Known limitations / special-case notes**, surfaced here per this document's own
TBD/known-limitations convention rather than silently omitted:

- **`SumIG`** has no differential equation of its own — its contribution is a
  closed-form Sum-of-Inverse-Gaussians input rate `I(t)` (Csajka, Drover & Verotta
  2005; Weiss & Wegner 2022), represented as an algebraic equation feeding the central
  compartment's influx term. The numerical `_t_safe` near-`t=0` stability guard the
  emitter applies is *omitted* here (a numerical-stability detail, not a modeling
  choice). Single-dose only, mirroring the emitter's own v0.7 scope (ADR-0003 D4).
- **`ZeroOrder`** produces no synthetic depot ODE — it is a `dur(<cmt>)` infusion-
  duration mechanism (rxode2 event-level), noted rather than faked as a differential
  equation; the central-compartment ODE keeps only the elimination term.
- **`IVBolus`** is treated the same way as `ZeroOrder` for influx purposes — dose
  enters via `CMT=1` event routing, not through any ODE term.
- **`Transit`**'s depot ODE uses an opaque `Function("transit")(n, mtt, t)` term rather
  than reproducing rxode2's built-in gamma-interpolated cascade (Savic et al. 2007)
  symbolically; `mtt = (n + 1) / ktr`.
- **`TMDDCore`/`TMDDQSS`** distribution modules completely ignore `spec.elimination`
  — verified even for a spec whose `elimination` field is not `LinearElim` (the
  semantic validator requires `LinearElim` for TMDD specs via
  `FrmCode.AST_TMDD_REQUIRES_LINEAR_ELIM`, but that is a compatibility gate on the
  *declared* module, not evidence the dynamics actually read `Vmax`/`Km` — they never
  do, regardless of what is declared). `kel`/`kdeg`/`ksyn`/`KSS`/`Cfree`/etc. are
  emitted as algebraic definitions mirroring the R intermediate-variable lines exactly.
- **`NODEAbsorption`/`NODEElimination`** have no closed-form symbolic representation;
  `build_equations` raises `NotImplementedError`, mirroring
  `emit_nlmixr2`'s own `spec.has_node_modules()` guard. The CLI catches this and prints
  a red error message, exiting with code 1, rather than emitting a misleading partial
  view.

No module combination the nlmixr2 emitter itself supports is left unrepresentable —
the only unhandled case is NODE, and it is an explicit, intentional
`NotImplementedError` rather than a silent fallthrough.

### `apmode formular signature <spec-file>`

Prints a compact, one-line, pipe-delimited summary of a spec's module choices via
`src/apmode/dsl/serializer.py::build_signature`, e.g.:

```
FO absorption | 1CMT | Linear CL | IIV(CL,V,ka) diag | Combined error
```

(this exact string is pinned end-to-end against
`benchmarks/suite_c/theophylline_boeckmann_1992.dsl.json` by
`tests/unit/test_dsl_signature.py`). Intended for `apmode formular signature`, report
headers, and DAG-viewer node labels — anywhere a single grep/pipe-able line is more
useful than `apmode formular explain`'s full multi-line table. Segments, in order:

1. Absorption short code, suffixed `" absorption"` (e.g. `"FO absorption"`,
   `"ZO absorption"`) — the bare short codes read as unlabeled abbreviations more than
   distribution/elimination's do, hence the suffix.
2. Distribution short code (e.g. `"1CMT"`, `"TMDD-QSS"`).
3. Elimination short code (e.g. `"Linear CL"`, `"MM"`).
4. IIV summary — `"IIV(param,param,...) diag|block"`, one clause per `IIV`
   variability item (joined with `"; "` if multiple), params sorted; omitted entirely
   when `spec.variability` has no `IIV` items. `IOV` items are intentionally excluded
   from this segment — a discretionary scope choice, not an oversight.
5. Observation segment — the single short code (e.g. `"Prop error"`, `"BLQ-M3"`) for a
   legacy singular `observation:` spec, or, for a multi-analyte `observations:` spec,
   `"<n> endpoints (<unique error-model codes>)"` (endpoint count plus the distinct
   error-model codes in use, de-duplicated, in endpoint-name-sorted order) — this
   multi-analyte rendering was left to the implementation's discretion by the plan.

**Known follow-up (not yet implemented).** `apmode ls` does not yet surface a
signature column — `cli.py::ls_command` currently reads only small summary artifacts
(`data_manifest.json`, `policy_file.json`, `ranking.json`,
`search_trajectory.jsonl`) in a fast directory scan; adding a per-row signature would
require loading and Pydantic-validating each bundle's `compiled_specs/{best_id}.json`
before `--limit` is applied, plus defensive handling for a missing/malformed compiled
spec. Flagged in `cli.py` with a `TODO(P2.4 follow-up)` comment pointing at the design
doc. **TBD — Phase 2 follow-up or Phase 3.**

---

## Formal Grammar Reference

Formular sharpening plan §4 Phase 2, P2.5. **`src/apmode/dsl/pk_grammar.lark` is the
executable source of truth.** This section is a human-readable, faithful rendering of
that file as read on 2026-07-08 — not an independent specification. If this section and
the `.lark` file ever disagree, the `.lark` file wins; treat a disagreement as a bug in
this document, not in the grammar.

The grammar's top level is order-insensitive (Formular sharpening plan §4 Phase 1,
P1.1): `model_body` is a flat `block*` list, and `apmode.dsl.grammar.compile_dsl` runs a
post-parse cardinality pass (`_validate_block_cardinality`) rather than the grammar
itself enforcing which blocks are required/singular/repeatable.

```
start: model
model: "model" "{" model_body "}"
model_body: block*
block: absorption | distribution | elimination | variability_block
     | observation | observations_block | metadata_block | initial_block
     | units_block | priors_block | covariates_block | use_block
```

### Absorption

One `absorption: <absorption_type>` block, required exactly once. Ten variants; three
(`Erlang`, `ParallelFirstOrder`, `SumIG`) are v0.7 SOTA extensions per ADR-0003.

```
absorption: "absorption:" absorption_type
absorption_type: iv_bolus | first_order | zero_order | lagged_first_order
                | transit | mixed_first_zero | erlang | parallel_first_order
                | sum_ig | node_absorption

iv_bolus:              "IVBolus" "(" ")"
first_order:           "FirstOrder" "(" "ka" ")"
zero_order:            "ZeroOrder" "(" "dur" ")"
lagged_first_order:    "LaggedFirstOrder" "(" "ka" "," "tlag" ")"
transit:               "Transit" "(" "n" "=" INT "," "ktr" "," "ka" ")"
mixed_first_zero:      "MixedFirstZero" "(" "ka" "," "dur" "," "frac" ")"
erlang:                "Erlang" "(" "n" "=" INT "," "ktr" ")"
parallel_first_order:  "ParallelFirstOrder" "(" "ka1" "," "ka2" "," "frac" ")"
sum_ig:                "SumIG" "(" "k" "=" INT "," "MT_1" "," "MT_2" ","
                                    "RD2_1" "," "RD2_2" "," "weight_1" ")"
node_absorption:       "NODE_Absorption" "(" "dim" "=" INT ","
                                    "constraint_template" "=" CONSTRAINT_TEMPLATE ")"
```

Every calibration parameter (`ka`, `dur`, `tlag`, `ktr`, `frac`, `ka1`, `ka2`, `MT_1`,
`MT_2`, `RD2_1`, `RD2_2`, `weight_1`) is a bare name with no inline value — the value
lives in the top-level `initial:` block (Phase 1, P1.4). `n`/`k`/`dim`/
`constraint_template` are structural (inline, integer or enumerated literal).

### Distribution

One `distribution: <distribution_type>` block, required exactly once. Five variants.

```
distribution: "distribution:" distribution_type
distribution_type: one_cmt | two_cmt | three_cmt | tmdd_core | tmdd_qss

one_cmt:    "OneCmt" "(" "V" ")"
two_cmt:    "TwoCmt" "(" "V1" "," "V2" "," "Q" ")"
three_cmt:  "ThreeCmt" "(" "V1" "," "V2" "," "V3" "," "Q2" "," "Q3" ")"
tmdd_core:  "TMDD_Core" "(" "V" "," "R0" "," "kon" "," "koff" "," "kint" ")"
tmdd_qss:   "TMDD_QSS" "(" "V" "," "R0" "," "KD" "," "kint" ")"
```

### Elimination

One `elimination: <elimination_type>` block, required exactly once. Five variants.
`TimeVarying`'s `kdecay` is deliberately not a named grammar token — it is always an
optional `initial:` value (defaults to `0.1` when omitted).

```
elimination: "elimination:" elimination_type
elimination_type: linear_elim | michaelis_menten | parallel_linear_mm
                | time_varying_elim | node_elimination

linear_elim:         "Linear" "(" "CL" ")"
michaelis_menten:    "MichaelisMenten" "(" "Vmax" "," "Km" ")"
parallel_linear_mm:  "ParallelLinearMM" "(" "CL" "," "Vmax" "," "Km" ")"
time_varying_elim:   "TimeVarying" "(" "CL" "," "decay_fn" "=" DECAY_FN ")"
node_elimination:    "NODE_Elimination" "(" "dim" "=" INT ","
                                    "constraint_template" "=" CONSTRAINT_TEMPLATE ")"
```

### Variability

Zero-or-more `variability:` blocks (each holding one-or-more items, or a single bare
item without braces); this is why `variability_block` is absent from the grammar's
own required/optional/exactly-one-of-group tracking, same treatment as `use_block`.

```
variability_block: "variability:" "{" variability_item+ "}"
                 | "variability:" variability_item
variability_item: iiv | iov

iiv: "IIV" "(" "params" "=" param_list "," "structure" "=" STRUCTURE ")"
iov: "IOV" "(" "params" "=" param_list "," "occasions" "=" occasion_spec ")"
param_list: "[" NAME ("," NAME)* "]"

occasion_spec: occasion_by_study | occasion_by_visit
             | occasion_by_dose_epoch | occasion_custom
occasion_by_study:      "ByStudy"
occasion_by_visit:      "ByVisit" "(" NAME ")"
occasion_by_dose_epoch: "ByDoseEpoch" "(" NAME ")"
occasion_custom:        "Custom" "(" NAME ")"
```

### Observation (singular) and `observations:` (multi-analyte)

Exactly one of `observation:`/`observations:` must appear per `model {}` — enforced by
`_validate_block_cardinality` on the raw parse tree, not by the grammar itself (both
productions are syntactically independent `block` alternatives).

```
observation: "observation:" observation_type
observation_type: proportional_obs | additive_obs | combined_obs | blq_m3 | blq_m4

proportional_obs: "Proportional" "(" "sigma_prop" "=" NUMBER ")"
additive_obs:     "Additive" "(" "sigma_add" "=" NUMBER ")"
combined_obs:     "Combined" "(" "sigma_prop" "=" NUMBER "," "sigma_add" "=" NUMBER ")"
blq_m3: "BLQ_M3" "(" "loq_value" "=" NUMBER "," "error_model" "=" ERROR_MODEL ","
                     "sigma_prop" "=" NUMBER "," "sigma_add" "=" NUMBER ")"
      | "BLQ_M3" "(" "loq_value" "=" NUMBER ")"
blq_m4: "BLQ_M4" "(" "loq_value" "=" NUMBER "," "error_model" "=" ERROR_MODEL ","
                     "sigma_prop" "=" NUMBER "," "sigma_add" "=" NUMBER ")"
      | "BLQ_M4" "(" "loq_value" "=" NUMBER ")"

observations_block: "observations:" "{" observation_entry
                       ("," observation_entry)* "}"
observation_entry: NAME ":" "{" "dvid" "=" INT "," "prediction" "=" NAME ","
                       "error" "=" observation_type "}"
```

### Metadata (optional, at most one)

```
metadata_block: "metadata:" "{" (metadata_item ("," metadata_item)*)? "}"
metadata_item: "title" "=" STRING -> metadata_title
             | "intent" "=" STRING -> metadata_intent
             | "context_of_use" "=" STRING -> metadata_context_of_use
             | "analyte" "=" STRING -> metadata_analyte
             | "version" "=" STRING -> metadata_version
```

### Initial (optional, at most one)

The single flat namespace every calibration value (from every axis) resolves against.

```
initial_block: "initial:" "{" (initial_item ("," initial_item)*)? "}"
initial_item: NAME "=" NUMBER
```

### Units (optional, at most one)

Declares GLOBAL measurement units for the spec's data/calibration values — not
per-parameter unit annotations. All four fields are required when the block is
present.

```
units_block: "units:" "{" "time" "=" unit_expr "," "amount" "=" unit_expr ","
                 "concentration" "=" unit_expr "," "volume" "=" unit_expr "}"
unit_expr: NAME ("/" NAME)?
```

### Priors (optional, at most one top-level block; zero-or-more entries)

Each entry lowers through the same `apmode.dsl.priors.build_prior_spec` factory the
`SetPrior` transform uses (P1.5 parity guarantee) — same field names/units as
`PriorFamily` (e.g. `LogNormal`'s `mu`/`sigma` are log-space).

```
priors_block: "priors:" "{" prior_entry* "}"
prior_entry: NAME "~" prior_family prior_attr*
prior_attr: "source" "=" PRIOR_SOURCE -> prior_attr_source
          | "doi" "=" STRING -> prior_attr_doi
          | "justification" "=" STRING -> prior_attr_justification
          | "historical_refs" "=" string_list -> prior_attr_historical_refs

prior_family: normal_prior | lognormal_prior | halfnormal_prior | halfcauchy_prior
            | gamma_prior | invgamma_prior | beta_prior | lkj_prior
            | mixture_prior | historical_borrowing_prior

normal_prior:      "Normal" "(" "mu" "=" numexpr "," "sigma" "=" numexpr ")"
lognormal_prior:   "LogNormal" "(" "mu" "=" numexpr "," "sigma" "=" numexpr ")"
halfnormal_prior:  "HalfNormal" "(" "sigma" "=" numexpr ")"
halfcauchy_prior:  "HalfCauchy" "(" "scale" "=" numexpr ")"
gamma_prior:       "Gamma" "(" "alpha" "=" numexpr "," "beta" "=" numexpr ")"
invgamma_prior:    "InvGamma" "(" "alpha" "=" numexpr "," "beta" "=" numexpr ")"
beta_prior:        "Beta" "(" "alpha" "=" numexpr "," "beta" "=" numexpr ")"
lkj_prior:         "LKJ" "(" "eta" "=" numexpr ")"

mixture_component: normal_prior | lognormal_prior | halfnormal_prior
                  | halfcauchy_prior | gamma_prior | invgamma_prior | beta_prior
prior_component_list: "[" mixture_component ("," mixture_component)* "]"
numexpr_list: "[" numexpr ("," numexpr)* "]"
mixture_prior: "Mixture" "(" "components" "=" prior_component_list ","
                   "weights" "=" numexpr_list ")"

historical_borrowing_prior: "HistoricalBorrowing" "(" "map_mean" "=" numexpr ","
                   "map_sd" "=" numexpr ("," "robust_weight" "=" numexpr)? ","
                   "historical_refs" "=" string_list ")"
string_list: "[" (STRING ("," STRING)*)? "]"

numexpr: NUMBER -> num_literal
       | "log" "(" numexpr ")" -> num_log
```

### Covariates (optional, at most one top-level block; zero-or-more entries)

Distinct from `variability:` — `IIV`/`IOV` remain the sole variability item kinds;
covariate effects use arrow syntax (`param <- covariate.form(...)`) with named,
first-class reference-value fields per form.

```
covariates_block: "covariates:" "{" (covariate_entry ("," covariate_entry)*)? "}"
covariate_entry: NAME "<-" NAME "." covariate_form_call
covariate_form_call: power_form | exponential_form | linear_form
                    | categorical_form | maturation_form

power_form:        "power" "(" "theta" "=" numexpr "," "ref" "=" numexpr ")"
exponential_form:  "exponential" "(" "theta" "=" numexpr ")"
linear_form:       "linear" "(" "theta" "=" numexpr ")"
categorical_form:  "categorical" "(" "reference" "=" STRING ")"
maturation_form:   "maturation" "(" "tm50" "=" numexpr "," "hill" "=" numexpr ")"
```

### Use (macros — Formular sharpening plan §4 Phase 2, P2.1)

Zero-or-more `use <dotted.name>` statements, same "absent from cardinality tracking"
treatment as `variability_block`. See [Macros](#macros) for the expansion semantics;
this is the grammar production alone.

```
use_block: "use" DOTTED_NAME
```

### Terminals

```
STRUCTURE:            "diagonal" | "block"
CONSTRAINT_TEMPLATE:  "monotone_increasing" | "monotone_decreasing"
                     | "bounded_positive" | "saturable" | "unconstrained_smooth"
DECAY_FN:             "exponential" | "half_life" | "linear"
ERROR_MODEL:          "proportional" | "additive" | "combined"
PRIOR_SOURCE:         "uninformative" | "weakly_informative" | "historical_data"
                     | "expert_elicitation" | "meta_analysis" | "fixed_external"
INT:     /[0-9]+/
NUMBER:  /[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/
DOTTED_NAME: NAME ("." NAME)*    // e.g. "pkstd.standard_iiv"; used only by use_block

%import common.CNAME -> NAME
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
%ignore /\/\/.*/
```

---

## Verification notes

Every claim in the sections above traces to a specific, cited location in
`src/apmode/dsl/ast_models.py`, `validator.py`, `transforms.py`, `prior_transforms.py`,
`priors.py`, `units.py`, `capabilities.py`, `canonical.py`, `equations.py`,
`serializer.py`, `pk_grammar.lark`, `cli_formular.py`, `macros/__init__.py`,
`macros/stdlib.py`, `bundle/models.py`, `bundle/emitter.py`, `nlmixr2_emitter.py`,
`stan_emitter.py`, or `frem_emitter.py` as read on 2026-07-08 (Phase 0 sections) and
re-verified against the same files' post-Phase-1 state on the same date (P1.10, this
pass) — every "Parameters" subsection's structural-vs-calibration split was checked
directly against the current `pk_grammar.lark` production for that module, not carried
forward from the Phase 0 wording. The [Macros](#macros),
[Transforms Reference](#transforms-reference),
[CLI: `explain --equations` and `signature`](#cli-explain---equations-and-signature),
and [Formal Grammar Reference](#formal-grammar-reference) sections were added in the
same P2.6 documentation pass, each fact directly re-derived from the four Phase 2
implementation streams' actual on-disk code (not from their own self-reports) — in
particular the `macros_used` fingerprint-exclusion claim was independently re-verified
against `canonical.py`'s `_structure_dict`/`_spec_dict` bodies, and the
`applied_at`-lives-on-lineage-not-on-transform claim was independently re-verified
against `transforms.py`/`prior_transforms.py` (absent there) and `bundle/models.py`
(present there). Items marked **TBD — Phase N** or **TBD** without a specific phase are
places where the reading did not produce enough evidence to state a fact confidently;
they should be resolved (not guessed at) by whichever phase's work next touches that
module, per the "each subsequent phase fills the sections it touches" policy at the top
of this document.

Modules/areas where confidence was genuinely limited, listed explicitly rather than
silently left ambiguous:

- **`IIV` block-structure correlation semantics under Stan** (item 4/6 of the `IIV`
  section) — whether Stan's `eta_raw` matrix actually induces a correlated block or
  treats columns independently was not traced through the full covariance-Cholesky path;
  flagged TBD rather than asserted.
- **`LaggedFirstOrder` under Stan reset+dose events** — plain dose events are delayed in
  the ODE path, but reset+dose (`EVID=4`) is applied as one delayed combined event rather
  than an immediate reset plus delayed dose.
- **`BetaPrior`/`HistoricalBorrowingPrior` parameterization mismatches against the Stan
  emitter's uniformly-log-scale (structural) / uniformly-natural-scale (covariate)
  parameter declarations** — the `HistoricalBorrowingPrior`-on-`covariate` mismatch was
  confirmed by direct trace (`_emit_model_block`'s covariate-prior call site always
  passes `on_log_scale=False`, and `_emit_historical_borrowing_prior` raises on
  `on_log_scale=False`). `BetaPrior` is now target-restricted to unit-interval structural
  targets; unsupported Stan log-scale emission raises rather than silently applying a beta
  density to a log parameter.
- **NODE module ODE/algebraic contribution** — this document describes only what the
  nlmixr2/Stan emitters do (raise `NotImplementedError`), not the actual JAX/Diffrax
  hybrid-ODE mathematics in `node_ode.py`, which is out of scope for a Formular-compiler
  semantics document at this phase but should get its own section once NODE modules are
  in scope for a documentation pass.

**Phase 1 emitter/DSL gaps carried forward from the code-landing sessions (not dropped
here just because the code has landed):**

- **Stan and FREM reject multi-analyte `observations:` blocks outright**
  (`NotImplementedError` at emit time) — only nlmixr2 lowers 2+ endpoints today; see
  [Multi-analyte `observations:` block](#multi-analyte-observations-block-observationendpoint)
  item 7.
- **nlmixr2's own multi-analyte lowering has no end-to-end data-routing story** — the
  data-adapter/runner layer (`Nlmixr2Runner`'s two-layer adapter contract) currently
  filters non-PK `DVID` rows *out* rather than routing them to a second endpoint; wiring
  that is a Phase 2 candidate (same section, item 7).
- **`categorical` covariate coefficients are not configurable** — the form has no
  `theta`-equivalent field yet and both emitters hardcode its starting value to `0`; see
  the `CovariateLink` section, items 3/4/11.
- **`categorical` and `exponential` covariate forms share identical math** in both
  emitters — the DSL distinguishes them as a type, not (yet) as a distinct equation; same
  section, item 4/11 (a pre-existing Phase 0 observation, still true post-Phase-1).
- **`maturation` covariate form has no Stan lowering** — `_covariate_expr` raises
  `NotImplementedError` for it; `CovariateLink` section, items 4/6/11.
- **Full multi-endpoint awareness has not propagated to `apmode.dsl.canonical`,
  `apmode.dsl.units`, `apmode.bundle.scoring_contract`, or
  `apmode.backends.predictive_summary`** — all four still read the singular
  `spec.observation` (synthesized from the first `observations:` entry when that syntax
  is used) rather than iterating every endpoint; see `DSLSpec.observations` field
  docstring and item 6 of the multi-analyte section. Per-endpoint unit coverage, NPE, and
  VPC remain Phase 2 candidates.
- **The agentic transform grammar has no multi-analyte-endpoint transform** — a spec
  cannot add/remove/modify an `observations:` entry via any `FormularTransform` today;
  see the multi-analyte section, item 10.

**Phase 2 gaps carried forward (not dropped here just because the code has landed):**

- **No user-defined macros.** `MACRO_REGISTRY` is a closed, small, vetted set populated
  only by `apmode.dsl.macros.stdlib`; there is no spec-author-facing extension
  mechanism. Extending it is a **TBD — Phase 3+** candidate if ever revisited — see
  [Macros](#macros), final paragraph.
- **`apmode ls` does not surface `build_signature`'s compact model summary.** A
  `TODO(P2.4 follow-up)` comment marks the intended insertion point in
  `cli.py::ls_command`; not implemented in this phase because it would require loading
  and Pydantic-validating each bundle's compiled spec inside what is otherwise a fast,
  low-dependency directory scan. See
  [CLI: `explain --equations` and `signature`](#cli-explain---equations-and-signature),
  final paragraph.
- **The symbolic equations view (`equations.py`) has no NODE representation** — an
  intentional, documented `NotImplementedError`, not a gap to close, but noted here
  since it is the one case `apmode formular explain --equations` cannot render for any
  spec using `NODEAbsorption`/`NODEElimination`.
- **`pkstd.standard_error_model` is a no-op for multi-analyte (`spec.observations`)
  specs** — `apmode.dsl.priors.classify_target`'s flat `sigma_prop`/`sigma_add`
  namespace has no per-endpoint disambiguation; adding one is a Phase 2+ candidate
  outside this macro's scope, not attempted here. See [Macros](#macros).
