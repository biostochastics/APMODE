# Formular Migration Guide: v0.6 → v0.7

## Related documentation

- [FORMULAR.md](FORMULAR.md) — primary language reference (grammar, transforms, extensibility) this migration guide assumes as background.
- [FORMULAR_SEMANTICS.md](FORMULAR_SEMANTICS.md) — formal Phase 2 spec describing the same validation surface this guide's mechanical rewrite targets.
- [FORMULAR_ERROR_CODES.md](FORMULAR_ERROR_CODES.md) — the exact FRM-AST-012/013-class errors a user hits when this migration's rewrite is done incorrectly.
- [adr/0003-sota-absorption-extension.md](adr/0003-sota-absorption-extension.md) — the new v0.7 absorption forms (Erlang, ParallelFirstOrder, SumIG) that co-shipped with this breaking grammar change.
- `plans/2026-07-08-formular-sharpening-and-adoption-design.md` *(internal-only, gitignored)* — §4 design plan that introduced these breaking changes, currently cited by section number only.

**Status:** Current
**Scope:** Syntax changes introduced by the Formular sharpening plan, Phase 1
(`docs/plans/2026-07-08-formular-sharpening-and-adoption-design.md` §4). Every change
below is a **breaking, non-dual-supported** rewrite of the grammar — the v0.7 compiler
(`apmode.dsl.grammar.compile_dsl`) does not parse the v0.6 forms described here at all.
There is no compatibility flag; a v0.6 spec file must be rewritten before it will compile
against this version of APMODE.

Two of these changes require rewriting hand-authored `.pk` text. The rest are additive
(new optional blocks) and require no changes to an existing v0.6 spec beyond the two
breaking rewrites. Run `apmode formular fmt --migrate` to apply the mechanical rewrite
automatically for the two breaking changes (§1, §2 below); see [§4](#4-automated-migration-apmode-formular-fmt---migrate)
for what the tool does and does not handle.

---

## 1. Inline calibration values move into `initial: { ... }`

**What changed.** Every structural declaration in `absorption:`/`distribution:`/
`elimination:` used to carry its numeric starting values inline, as
`Keyword(name=value, ...)`. In v0.7, those numeric values are removed from the
declaration — the declaration keeps only the **parameter names** it introduces — and
every numeric value moves into a single top-level `initial: { name = value, ... }`
block.

This is not cosmetic. `DSLSpec.initial` is now the single flat namespace every
calibration value lives in, regardless of which module introduced the parameter name —
mirrored by `DSLSpec.calibration_param_names()`, `get_initial()`, and the
`initial_fingerprint()` (`apmode.dsl.canonical`) that governs which changes count as a
new candidate vs. a re-estimation of an existing one.

**Before (v0.6):**

```
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: Linear(CL=5.0)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
}
```

**After (v0.7):**

```
model {
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = 1.2, V = 70.0, CL = 5.0 }
}
```

### Mechanical rewrite recipe

For every structural declaration of the form `Keyword(name1=v1, name2=v2, ...)`:

1. For each `name=value` pair, decide whether `name` is **structural** or
   **calibration**:
   - **Structural** (stays inline, unchanged, as `name=value`): `Transit`'s and
     `Erlang`'s `n` (chain length, an integer topology choice); `SumIG`'s `k` (number of
     inverse-Gaussian components); `TimeVarying`'s `decay_fn` (which decay shape, a
     keyword, not a number). `NODE_Absorption`/`NODE_Elimination`'s `dim` and
     `constraint_template` are unaffected altogether (they never had a numeric
     calibration value).
   - **Calibration** (numeric starting value — move out): everything else. Concretely:
     `ka`, `dur`, `tlag`, `ktr`, `frac`, `ka1`, `ka2`, `MT_1`, `MT_2`, `RD2_1`, `RD2_2`,
     `weight_1`, `V`, `V1`, `V2`, `V3`, `Q`, `Q2`, `Q3`, `R0`, `kon`, `koff`, `kint`,
     `KD`, `CL`, `Vmax`, `Km`.
2. Rewrite the declaration to drop `=value` from every calibration name (leaving the
   bare name), while keeping every structural `name=value` pair unchanged.
3. Collect every removed `(name, value)` pair, across every module in the model, into
   one `initial: { name1 = value1, name2 = value2, ... }` block anywhere at the top
   level of `model { ... }` (block order is free in v0.7 — see §3).

**One irregular case: `TimeVarying`'s `kdecay`.** In v0.6, `TimeVarying` had two forms —
`TimeVarying(CL=X, decay_fn=Y)` and `TimeVarying(CL=X, kdecay=Y, decay_fn=Z)` — with
`kdecay` optionally inline. In v0.7, `kdecay` is not part of the call **at all**, in
either form; the call is always `TimeVarying(CL, decay_fn=Z)`. If your v0.6 spec supplied
`kdecay` explicitly, move it into `initial: { ..., kdecay = <value> }` like any other
calibration value. If it omitted `kdecay`, do nothing — `DSLSpec.get_initial("kdecay",
0.1)` supplies the same 0.1 default v0.6 used inline.

```
# v0.6, kdecay given
elimination: TimeVarying(CL=5.0, kdecay=0.2, decay_fn=exponential)
# v0.7
elimination: TimeVarying(CL, decay_fn=exponential)
initial: { CL = 5.0, kdecay = 0.2 }

# v0.6, kdecay omitted (implicit 0.1)
elimination: TimeVarying(CL=5.0, decay_fn=exponential)
# v0.7 (kdecay still implicitly 0.1 via DSLSpec.get_initial)
elimination: TimeVarying(CL, decay_fn=exponential)
initial: { CL = 5.0 }
```

`observation:` sigma values (`Proportional(sigma_prop=...)`, `Additive(sigma_add=...)`,
`Combined(...)`, `BLQ_M3(...)`, `BLQ_M4(...)`) are **unchanged** — they stay inline. They
were never part of this rewrite because a residual-error sigma is not a structural
calibration value in the same sense (see `DSLSpec.calibration_param_names()`'s
docstring, which excludes observation and covariate fields by design).

---

## 2. `CovariateLink` moves from `variability:` to a `covariates:` block with arrow syntax

**What changed.** In v0.6, a covariate effect was declared as a `CovariateLink(...)`
function call and treated as one more item inside `variability:`, alongside `IIV`/`IOV`:

```
variability: CovariateLink(param=CL, covariate=WT, form=power)
```

or, mixed with other variability items:

```
variability: {
    IIV(params=[CL], structure=diagonal)
    CovariateLink(param=CL, covariate=WT, form=power)
}
```

In v0.7, `CovariateLink` is gone from `variability:` entirely — `variability_item` is
now only `IIV | IOV`. Covariate effects get their **own** top-level `covariates: { ... }`
block, with arrow syntax (`param <- covariate.form(...)`), and — this is the substantive
change, not just a syntax reshuffle — every form now carries **explicit, named
reference/coefficient values** that v0.6 did not expose in the DSL at all (they were
hardcoded inside the nlmixr2 emitter: `power` silently centered on a hardcoded 70 kg
reference weight, `exponential`/`linear`/`categorical` silently started their
coefficient at 0, `maturation` silently started at `hill=1, tm50=1`).

**Before (v0.6):**

```
variability: {
    IIV(params=[CL], structure=diagonal)
    CovariateLink(param=CL, covariate=WT, form=power)
}
```

**After (v0.7):**

```
variability: { IIV(params=[CL], structure=diagonal) }
covariates: { CL <- WT.power(theta=0.75, ref=70) }
```

### Mechanical rewrite recipe

1. Find every `CovariateLink(param=P, covariate=C, form=F)` inside a `variability:`
   block (braced or single-item) and remove it from that block. If `variability:` had no
   other item, remove the `variability:` block entirely (it is optional in v0.7).
2. Add one entry `P <- C.F(...)` per removed link to a `covariates: { ... }` block
   (create one if the spec doesn't have one yet).
3. Supply the field(s) each `form` now requires, per `apmode.dsl.ast_models.CovariateLink`:

   | `form`        | required fields  | what v0.6 hardcoded (use as your migration default) |
   |---------------|-------------------|-------------------------------------------------------|
   | `power`       | `theta`, `ref`    | `theta=0.75`, `ref=70` (Anderson & Holford 2008 allometric weight scaling) |
   | `exponential` | `theta`           | `theta=0.0` |
   | `linear`      | `theta`           | `theta=0.0` |
   | `categorical` | `reference`       | **no equivalent** — v0.6 never recorded a baseline level name anywhere; you must choose one by hand (e.g. `reference="M"` for a `SEX` covariate with baseline male) |
   | `maturation`  | `tm50`, `hill`    | `tm50=1.0`, `hill=1.0` |

   The table's non-`categorical` defaults reproduce the old hardcoded emitter constants
   exactly, so a mechanically migrated spec lowers to numerically identical R/Stan code
   to what it produced under v0.6 — but they were placeholders even then, not
   scientifically motivated starting values (except `power`'s 70 kg reference), so treat
   them as a starting point to tune, not a final answer.
4. `categorical` is the one form the mechanical recipe **cannot** complete
   automatically: pick the actual baseline level your data uses (see
   `apmode.data.adapters` for how string/two-level categorical covariates are encoded)
   and write it in by hand.

**Field names and identifiers are otherwise unquoted, exactly as in v0.6**: `param` and
`covariate` are bare identifiers (`CL`, `WT`), not string literals.

---

## 3. Additive changes: no rewrite required

The rest of the Phase 1 grammar work only *adds* optional top-level blocks; a v0.6 spec
that has already had §1 and §2 applied compiles unchanged. You do not need to add any of
these to migrate — they are documented here so a hand-authoring user knows they now
exist:

- **Blocks may appear in any order.** v0.6 required the fixed sequence `absorption`,
  `distribution`, `elimination`, `variability`, `observation`. v0.7's `model_body` is an
  unordered list of blocks (cardinality — exactly one absorption/distribution/
  elimination/observation-or-observations, at most one metadata/units/initial/priors/
  covariates, zero-or-more variability — is checked after parsing, not by grammar
  position). Existing v0.6 files are already in a valid v0.7 order; no change needed.
- **`metadata: { title = "...", intent = "...", context_of_use = "...", analyte = "...",
  version = "..." }`** — optional, free-text, provenance-only; does not affect
  compilation or emission.
- **`units: { time = h, amount = mg, concentration = mg/L, volume = L }`** — optional,
  declares the spec's global measurement units for `apmode.dsl.units`'s dimensional-
  homogeneity checker.
- **`priors: { CL ~ LogNormal(mu=1.386, sigma=0.25) }`** — optional Bayesian prior
  declarations, lowered through the same `apmode.dsl.priors.build_prior_spec` factory
  the agentic `SetPrior` transform uses.
- **`observations: { free: { dvid=1, prediction=C_central, error=Proportional(sigma_prop=0.1) }, ... }`**
  — optional multi-analyte form, additive to (never a replacement for) the singular
  `observation:` sugar. Only meaningful for distribution modules exposing more than one
  named prediction (currently only `TMDD_QSS`'s `C_target_total`).
- **`experimental: { node = true }`** — required opt-in when a spec uses a
  `NODE_Absorption`/`NODE_Elimination` module (no working backend yet); unrelated to
  this migration.

---

## 4. Automated migration: `apmode formular fmt --migrate`

`apmode formular fmt <spec.pk> --migrate [--in-place]` applies the §1 and §2 rewrites
above mechanically. It is a **best-effort, text-pattern rewriter** — regex/string
transforms targeting exactly the two documented constructs — not a second parser for the
old grammar (the design plan explicitly rejects resurrecting a full second Lark grammar
for a one-time migration aid). Concretely:

- It recognizes one structural declaration or one `CovariateLink(...)` call per source
  line, matching the one-declaration-per-line style every existing Formular fixture in
  this repository already uses. Multiple declarations packed onto a single line are
  outside its scope.
- Every calibration value it moves is carried through as **opaque text**, never
  re-parsed as a float — so `1.50` or `1.5e-2` survive the rewrite byte-for-byte,
  with no float round-tripping surprises.
- For `CovariateLink(..., form=categorical, ...)` — the one construct with no safe
  default (see the table in §2) — it does **not** guess. It leaves that exact
  `CovariateLink(...)` text untouched in the output and prints:

  ```
  could not auto-migrate this construct near line N, please review manually: ...
  ```

  to stderr, and exits with a non-zero status. Every other recognized construct in the
  same file is still migrated; only the flagged one is left for you to fix by hand.
- Running it twice is safe: text already in v0.7 form (bare structural names, no
  `CovariateLink(...)` calls) matches nothing and passes through unchanged.

Programmatic entry point: `apmode.dsl.migration.migrate_v06_to_v07(text: str) ->
MigrationResult` (`MigrationResult.text`, `MigrationResult.warnings`). After migrating,
run `apmode formular fmt <file>` (without `--migrate`) to canonicalize block order via
the normal compile-and-reserialize path once the file compiles cleanly.

### Known gap (Phase 2 candidate)

The migrator does not attempt to reconcile multiple `CovariateLink`/structural
declarations sharing one source line, nor spec text spanning line continuations inside a
single call's argument list. Reformat such a file to one-declaration-per-line before
running `--migrate`, or perform that portion of the rewrite by hand.
