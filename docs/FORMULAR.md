# Formular — The APMODE PK Specification Language

**Version:** 0.5 (tracks APMODE 0.5.0-rc1)
**Status:** Current (Phase 3 in progress)
**Canonical source:** `src/apmode/dsl/` — grammar (`pk_grammar.lark`), AST (`ast_models.py`),
validator (`validator.py`), transforms (`transforms.py` + `prior_transforms.py`), priors
(`priors.py`), emitters (`nlmixr2_emitter.py`, `stan_emitter.py`, `frem_emitter.py`).

---

Formular is a typed domain-specific language for specifying population pharmacokinetic
models. It is the single source of truth for model structure in APMODE: every model —
whether dispatched to nlmixr2 (R), Stan/Torsten (via cmdstanpy), or a neural-ODE backend
(JAX/Diffrax) — is expressed as a Formular specification, compiled to a typed AST,
validated against pharmacometric constraints, and lowered to backend-specific code.

The name evokes *formulary* (pharmacy) and *formula* (mathematics). The language is
deliberately constrained: it captures what a pharmacometrician would write on a
whiteboard, not what a programmer would write in R or Python.

---

## Why a dedicated language?

1. **Safety boundary for agentic AI.** The agentic LLM backend operates exclusively
   through a fixed set of typed Formular transforms — it cannot emit raw code. Every
   proposed model change is a typed AST operation that the validator can accept or reject
   before compilation. This is the primary safety mechanism against unbounded code
   generation.

2. **Backend independence.** A single Formular spec lowers to nlmixr2 R code, Stan
   programs, or JAX/Diffrax neural-ODE configurations. Adding a backend means writing a
   new emitter, not redesigning the model specification.

3. **Reproducibility.** The compiled `DSLSpec` (a Pydantic model) is the serializable
   unit stored under `compiled_specs/<candidate_id>.json` in every run bundle. Any
   candidate model can be reconstructed from its Formular AST without re-running the
   search.

4. **Constraint enforcement.** Pharmacometric constraints (volumes > 0, rates > 0, NODE
   dimensions within lane ceilings, prior-family admissibility, valid constraint
   templates) are enforced at the language level, before any backend code is generated.

---

## Shape of a Formular specification

Formular has a **five-block grammar** (absorption, distribution, elimination,
variability, observation) and a **sixth semantic axis** — `priors` — that is carried on
the compiled `DSLSpec` but has no textual grammar block: priors are declared via the
`SetPrior` transform (typed Python API) or defaulted by the Bayesian pipeline, never
written in Formular text.

```
                   ┌──────────────────────────────────────┐
  TEXT:            │   model { absorption: …             │
  5 grammar        │           distribution: …           │
  blocks           │           elimination: …            │
                   │           variability: { … }        │
                   │           observation: … }          │
                   └──────────────────┬───────────────────┘
                                      │ Lark (Earley) + transformer
                                      ▼
                   ┌──────────────────────────────────────┐
  AST:             │ DSLSpec(                             │
  6 fields         │   model_id, absorption, distribution,│
                   │   elimination, variability,          │
                   │   observation,                       │
                   │   priors: list[PriorSpec]  ← set via │
                   │                               SetPrior│
                   │ )                                    │
                   └──────────────────────────────────────┘
```

Every AST field is required except `priors` (empty list by default — only populated
when `SetPrior` fires or a Bayesian run emits a default prior manifest).

---

## Grammar reference

A Formular model has five required blocks:

```
model {
    absorption:  <absorption-module>
    distribution: <distribution-module>
    elimination:  <elimination-module>
    variability:  { <variability-items...> }   # or a single item without braces
    observation:  <observation-module>
}
```

### Absorption

| Module | Syntax | Parameters |
|--------|--------|------------|
| IV bolus | `IVBolus()` | dose enters central directly; no absorption phase |
| First-order | `FirstOrder(ka=1.0)` | `ka`: absorption rate (h⁻¹) |
| Zero-order | `ZeroOrder(dur=2.0)` | `dur`: infusion duration (h) |
| Lagged first-order | `LaggedFirstOrder(ka=1.0, tlag=0.5)` | `ka`, `tlag` (h) |
| Transit | `Transit(n=4, ktr=2.0, ka=1.0)` | `n`: compartments (≥1), `ktr` (h⁻¹), `ka` |
| Mixed | `MixedFirstZero(ka=1.0, dur=2.0, frac=0.6)` | `frac ∈ (0, 1)` |
| Erlang (v0.7) | `Erlang(n=3, ktr=1.5)` | integer `n` (1..7); chain with no terminal `ka` |
| Parallel first-order (v0.7) | `ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6)` | two depots; `frac ∈ (0, 1)` |
| SumIG (v0.7) | `SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)` | closed-form IG sum; `MT_1 < MT_2`; Discovery+Optimization only |
| NODE | `NODE_Absorption(dim=4, constraint_template=monotone_increasing)` | `dim`: hidden-layer width |

`Erlang` lowers as an explicit n-compartment ODE chain in the nlmixr2 emitter (not
via `rxode2::transit()` — see ADR-0003 D2). `ParallelFirstOrder` lowers as two
depot compartments feeding central simultaneously, distinct from `MixedFirstZero`
(first-order + zero-order). `SumIG` lowers as a closed-form analytic input rate
(`sqrt(RD2 / (2π·t³)) · exp(...)`) with the dose amount applied as scaling factor;
single-dose only in v0.7. Stan/Torsten support for the three v0.7 forms is deferred
to v0.7.1; the Stan emitter raises `NotImplementedError` and routes to nlmixr2.

`SumIG.k >= 2` carries an identifiability gate: disposition parameters (CL/V/Q)
must be externally fixed, either via the `EvidenceManifest.disposition_fixed`
flag at dispatch or via priors with `source="fixed_external"` on each disposition
param. Without the gate, sumIG-2 is non-identifiable on sparse data (Csajka 2005;
Weiss 2022). `SumIG` is blocked from the Submission lane.

### Distribution

| Module | Syntax | Parameters |
|--------|--------|------------|
| One-compartment | `OneCmt(V=30.0)` | `V` (L) |
| Two-compartment | `TwoCmt(V1=30.0, V2=40.0, Q=5.0)` | `V1`, `V2`, `Q` (L/h) |
| Three-compartment | `ThreeCmt(V1=..., V2=..., V3=..., Q2=..., Q3=...)` | |
| TMDD full | `TMDD_Core(V=30.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05)` | Mager & Jusko 2001 |
| TMDD QSS | `TMDD_QSS(V=30.0, R0=10.0, KD=0.1, kint=0.05)` | Gibiansky 2008; KD ≈ KSS when `kint ≪ koff` |

### Elimination

| Module | Syntax | Parameters |
|--------|--------|------------|
| Linear | `Linear(CL=2.0)` | `CL` (L/h) |
| Michaelis–Menten | `MichaelisMenten(Vmax=50.0, Km=5.0)` | Saturable |
| Parallel linear + MM | `ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)` | Mixed |
| Time-varying | `TimeVarying(CL=2.0, kdecay=0.1, decay_fn=<fn>)` | `decay_fn ∈ {exponential, half_life, linear}`; `kdecay` optional (grammar accepts the 2-arg form and defaults) |
| NODE | `NODE_Elimination(dim=4, constraint_template=bounded_positive)` | |

**TimeVarying semantics (validator-enforced):**

```
exponential:  CL(t) = CL · exp(-kdecay · t)
half_life:    CL(t) = CL / (1 + kdecay · t)        (kdecay = ln(2)/t_half)
linear:       CL(t) = max(CL · (1 - kdecay · t), 0)
```

### Variability

Multiple items are grouped in braces; a single item can omit them:

```
variability: {
    IIV(params=[CL, V1, ka], structure=block)
    IOV(params=[CL], occasions=ByStudy)
    CovariateLink(param=CL, covariate=WT, form=power)
}
```

| Item | Options |
|------|---------|
| `IIV` | `structure`: `diagonal` or `block` (block requires ≥2 params) |
| `IOV` | `occasions`: `ByStudy`, `ByVisit(col)`, `ByDoseEpoch(col)`, `Custom(col)` |
| `CovariateLink` | `form`: `power`, `exponential`, `linear`, `categorical`, `maturation` |

### Observation

| Module | Syntax |
|--------|--------|
| Proportional | `Proportional(sigma_prop=0.1)` |
| Additive | `Additive(sigma_add=0.5)` |
| Combined | `Combined(sigma_prop=0.1, sigma_add=0.5)` |
| BLQ M3 (bare) | `BLQ_M3(loq_value=0.1)` |
| BLQ M3 (composed) | `BLQ_M3(loq_value=0.1, error_model=combined, sigma_prop=0.2, sigma_add=0.5)` |
| BLQ M4 (bare) | `BLQ_M4(loq_value=0.1)` |
| BLQ M4 (composed) | `BLQ_M4(loq_value=0.1, error_model=additive, sigma_prop=0.1, sigma_add=1.0)` |

**BLQ composition.** `BLQ_M3` and `BLQ_M4` accept an `error_model ∈ {proportional, additive, combined}` argument that selects which residual form is composed with the likelihood-based BLQ correction (Beal 2001). The bare form defaults to proportional.

---

## Complete example (5-block grammar)

```
model {
    absorption: Transit(n=4, ktr=2.0, ka=1.0)
    distribution: TwoCmt(V1=30.0, V2=40.0, Q=5.0)
    elimination: ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)
    variability: {
        IIV(params=[CL, V1, ka], structure=block)
        CovariateLink(param=CL, covariate=WT, form=power)
    }
    observation: Combined(sigma_prop=0.1, sigma_add=0.5)
}
```

Two-compartment, transit absorption, parallel linear + Michaelis–Menten elimination,
block IIV on three parameters, allometric weight scaling on clearance, combined
residual error.

---

## Priors — the sixth (semantic) axis

Priors are **not** written in Formular text. They live on `DSLSpec.priors` as a typed
list of `PriorSpec` objects, populated in one of two ways:

1. **Programmatic declaration** via the `SetPrior` transform (below).
2. **Default injection** by the Bayesian pipeline (`apmode.bayes.harness`) when no
   explicit prior is supplied — weakly-informative `Normal(log_init, 2)` on structural
   parameters, `HalfCauchy(1)` on ω, `HalfCauchy(σ_init)` on residual σ, `LKJ(2)` on
   correlation matrices.

Priors are consumed by the Stan emitter (injected into the `model` block) and the
Bayesian runner. The nlmixr2 emitter and the JAX/NODE backend currently **ignore**
priors unless explicitly wired — for those backends, priors are advisory metadata
stored in the bundle for audit.

### Prior families

| Family | Parameters | Admissible targets |
|--------|-----------|--------------------|
| `Normal(mu, sigma)` | real-valued | log-scale structural, covariate coefficients |
| `LogNormal(mu, sigma)` | positive | structural (natural scale) |
| `HalfNormal(sigma)` | positive | ω (IIV/IOV SD), residual σ |
| `HalfCauchy(scale)` | positive | ω, residual σ |
| `Gamma(alpha, beta)` | positive | ω, residual σ |
| `InvGamma(alpha, beta)` | positive | ω, residual σ |
| `Beta(alpha, beta)` | `[0, 1]` | bioavailability F, mixing fractions |
| `LKJ(eta)` | correlation | `corr_iiv` only |
| `Mixture(components, weights)` | compound | structural, covariate (robust-MAP building block) |
| `HistoricalBorrowing(map_mean, map_sd, robust_weight, historical_refs)` | compound | structural, covariate (Schmidli 2014 robust MAP; compiles to `Mixture` at emit time) |

### Target taxonomy

Prior targets resolve against the current `DSLSpec`:

| Target pattern | Kind | Example |
|---|---|---|
| Name in `structural_param_names()` | `structural` | `"CL"`, `"V1"`, `"ka"` |
| `omega_<param>` | `iiv_sd` | `"omega_CL"` |
| `omega_iov_<param>` | `iov_sd` | `"omega_iov_CL"` |
| `sigma_prop`, `sigma_add` | `residual_sd` | |
| `corr_iiv` | `corr_iiv` | |
| `beta_<param>_<covariate>` | `covariate` | `"beta_CL_WT"` |

### Parameterization-schema admissibility matrix

Enforced at `SetPrior` validation time (`apmode.dsl.priors::_VALID_FAMILIES`). Invalid
(target, family) pairs are rejected before any emission:

| Target kind | Admissible families |
|---|---|
| `structural` | `Normal`, `LogNormal`, `Mixture`, `HistoricalBorrowing` |
| `iiv_sd` / `iov_sd` / `residual_sd` | `HalfNormal`, `HalfCauchy`, `Gamma`, `InvGamma` |
| `corr_iiv` | `LKJ` only |
| `covariate` | `Normal`, `Mixture`, `HistoricalBorrowing` |

> `LKJ` on `corr_iiv` is schema-accepted; the Stan emitter does not yet declare
> `corr_iiv` in the parameters block and will raise `NotImplementedError` if it sees
> one. The schema accepts it so agentic planning can plan ahead.

### Provenance and FDA alignment

Every `PriorSpec` carries a `source ∈ {uninformative, weakly_informative,
historical_data, expert_elicitation, meta_analysis}`. Sources other than
`uninformative` / `weakly_informative` require a non-empty `justification` string
(validator-enforced); `historical_data` additionally requires `historical_refs` listing
source datasets. This aligns with FDA draft guidance FDA-2025-D-3217 (January 2026) on
Bayesian methodology — the resulting `prior_manifest.json` in the bundle is the
artifact Gate 2 consumes for prior-justification review.

---

## Formular transforms — the 10 typed mutations

The set of admissible mutations to a `DSLSpec`. Each transform is a frozen Pydantic
model with a `type` discriminator; the union `FormularTransform` is the agentic LLM
backend's output type. No other form of DSL mutation is admissible — the LLM cannot
emit raw code.

| Transform | Purpose | Payload |
|---|---|---|
| `swap_module` | Replace an absorption/distribution/elimination/observation module | `position`, `new_module` |
| `add_covariate_link` | Add a covariate effect | `param`, `covariate`, `form ∈ {power, exponential, linear, categorical, maturation}` |
| `adjust_variability` | Add/remove an IIV term or upgrade diagonal → block | `param`, `action ∈ {add, remove, upgrade_to_block}` |
| `set_transit_n` | Change transit compartment count | `n` |
| `toggle_lag` | Enable/disable absorption `tlag` | `enabled`, optional `tlag` |
| `replace_with_node` | Swap an axis to a NODE module (Discovery/Optimization only) | `position`, `dim`, `constraint_template` |
| `convert_transit_to_erlang` (v0.7) | Convert `Transit(n, ktr, ka)` → `Erlang(n, ktr)` | `n` (1..7) |
| `add_parallel_route` (v0.7) | Convert `FirstOrder(ka)` → `ParallelFirstOrder(ka1=ka, ka2, frac)` | `ka2`, `frac` |
| `set_sumig_components` (v0.7) | Update SumIG params in-place | `MT_1`, `MT_2`, `RD2_1`, `RD2_2`, `weight_1` |
| `set_prior` | Declare or replace a prior (idempotent replace-or-append) | `target`, `family`, `source`, `justification`, `historical_refs` |

**Semantics.** `apmode.dsl.transforms.apply_transform(spec, t)` returns a new
`DSLSpec` with a fresh `model_id` (sparkid). Transforms are pure — the input spec is
not mutated. `validate_transform(spec, t)` returns a list of string errors without
applying; the agentic loop runs validate before apply and feeds failures back to the
LLM for correction.

**Safety boundary.** Preconditions (e.g., `set_transit_n` rejects on a spec whose
absorption is not `Transit`; `set_prior` rejects nonsense (target, family) pairs via
the admissibility matrix) are checked at plan time. Post-apply the validator runs in
full (see below), so invalid *compound* states are caught even if each individual
transform passes its precondition.

**Builder function.** `transform_parser.py::parse_transforms(json_str)` deserializes
an LLM's JSON output into `list[FormularTransform]` — this is the only path by which
LLM-authored transforms enter the pipeline.

---

## Compilation pipeline

```
Formular text                            SetPrior / default priors
     │                                           │
     ▼                                           │
Lark parser (Earley, 10 KB guard)                │
     │                                           │
     ▼                                           │
DSLTransformer ──► DSLSpec (typed Pydantic AST) ◄┘
     │
     ▼
validate_dsl(spec, lane=<Lane>)   ── lane-aware semantic checks
     │                               (volumes > 0, NODE dim ≤ lane ceiling, …)
     ▼
Backend emitter
     ├─► emit_nlmixr2(spec)                → R code (nlmixr2 / rxode2)
     ├─► emit_stan(spec)                   → Stan program (BLQ M3/M4 & IOV ⇒ NotImplementedError)
     ├─► emit_nlmixr2_frem(spec, covs, …)  → FREM-augmented nlmixr2 code
     └─► node runner  (JAX/Diffrax; consumes spec + NODE modules directly)
```

**Entry points:**

| Purpose | Callable |
|---|---|
| Parse + build AST | `apmode.dsl.grammar.compile_dsl(text) → DSLSpec` (applies 10 KB length guard) |
| Lane-aware validation | `apmode.dsl.validator.validate_dsl(spec, lane=Lane.SUBMISSION\|DISCOVERY\|OPTIMIZATION) → list[ValidationError]` |
| Apply a transform | `apmode.dsl.transforms.apply_transform(spec, transform) → DSLSpec` |
| Apply a set-prior | `apmode.dsl.prior_transforms.apply_set_prior(spec, t) → DSLSpec` |

`compile_dsl` is parse + AST only — it does **not** run lane-aware semantic checks
(dim ceilings, lane admissibility). The orchestrator calls `validate_dsl(spec, lane=…)`
separately so the lane can be varied without recompiling.

### Key files

| File | Role |
|------|------|
| `src/apmode/dsl/pk_grammar.lark` | Lark EBNF for the 5-block grammar |
| `src/apmode/dsl/grammar.py` | `compile_dsl` entry point + Earley parser wiring |
| `src/apmode/dsl/transformer.py` | Parse tree → typed AST |
| `src/apmode/dsl/ast_models.py` | `DSLSpec` + all module Pydantic nodes |
| `src/apmode/dsl/normalize.py` | AST canonicalization (parameter ordering, name normalization) |
| `src/apmode/dsl/validator.py` | `validate_dsl` (lane-aware) + pharmacometric constraint checks |
| `src/apmode/dsl/transforms.py` | 6 structural transforms + `FormularTransform` union |
| `src/apmode/dsl/prior_transforms.py` | `SetPrior` (the 7th transform) |
| `src/apmode/dsl/priors.py` | `PriorSpec`, prior families, admissibility schema |
| `src/apmode/dsl/nlmixr2_emitter.py` | AST → nlmixr2 R code |
| `src/apmode/dsl/stan_emitter.py` | AST → Stan program |
| `src/apmode/dsl/frem_emitter.py` | AST → FREM-augmented nlmixr2 (joint-Ω covariate workflow) |
| `src/apmode/dsl/_emitter_utils.py` | Shared emitter helpers (parameter escaping, etc.) |
| `src/apmode/backends/transform_parser.py` | JSON (LLM output) → `list[FormularTransform]` |

---

## Constraint templates (NODE modules)

NODE absorption and elimination modules require a `constraint_template` that restricts
the learned function's shape:

| Template | Meaning | Template max dim |
|----------|---------|------------------|
| `monotone_increasing` | Output monotonically increases with input | 4 |
| `monotone_decreasing` | Output monotonically decreases with input | 4 |
| `bounded_positive` | Output is strictly positive | 6 |
| `saturable` | Output increases then plateaus (Michaelis–Menten-like) | 4 |
| `unconstrained_smooth` | Smooth output only; no shape constraint | 8 |

The effective dim ceiling is `min(template_max, lane_ceiling)`:

| Lane | Lane ceiling | NODE admissible? |
|------|-------------|------------------|
| Submission | — | **No** (hard rule) |
| Discovery | 8 | Yes |
| Optimization | 4 | Yes |

---

## Semantic validation rules

The validator (`apmode.dsl.validator.validate_dsl`) enforces pharmacometric
constraints before any code is generated. Violations are accumulated (non-fail-fast,
Pandera `lazy=True` philosophy) and returned as a `list[ValidationError]`.

| Rule | Constraint |
|------|-----------|
| Positivity | Volumes, rates, clearances, sigmas must be > 0 |
| Unit interval | `MixedFirstZero.frac ∈ (0, 1)` exclusive |
| Non-negative | `tlag ≥ 0` |
| Positive integer | Transit `n`, NODE `dim` must be ≥ 1 |
| NODE lane admissibility | NODE modules rejected in Submission lane |
| NODE dim ceilings | `dim ≤ min(template_max_dim, lane_ceiling)` |
| Block IIV | `structure=block` requires ≥ 2 params |
| TMDD requires Linear elim | `TMDD_Core` / `TMDD_QSS` requires `Linear` elimination |
| No duplicate IIV params | Same parameter cannot appear in multiple IIV blocks |
| No duplicate covariate links | Same `(param, covariate)` pair cannot appear twice |
| No variability on Transit `n` | Transit `n` cannot have IIV/IOV/CovariateLink |
| IIV/IOV/CovariateLink params exist | Must reference a structural parameter |
| `TimeVarying.decay_fn` | Must be one of `{exponential, half_life, linear}` (Pydantic Literal) |
| Prior target/family match | `(target_kind, family)` must appear in the admissibility matrix |
| Prior target resolves | Target must be a known parameter name or pattern |
| Informative-prior justification | `source ∈ {historical_data, expert_elicitation, meta_analysis}` requires non-empty `justification`; `historical_data` additionally requires `historical_refs` |

Covariate-column existence is **not** checked at validation time — it is a
data-binding check that occurs when the emitter composes the spec with a concrete
dataset (the column may be absent in the current dataset but present in a later run).

### Example rejections

```
MixedFirstZero(ka=1.0, dur=2.0, frac=1.5)
  → ValidationError: MixedFirstZero.frac must be in (0, 1), got 1.5

Transit(n=0, ktr=2.0, ka=1.0)
  → ValidationError: Transit.n must be >= 1, got 0

IIV(params=[CL], structure=block)
  → ValidationError: block IIV requires >= 2 params, got 1

NODE_Absorption(dim=6, constraint_template=monotone_increasing)  in Discovery
  → ValidationError: NODE dim 6 exceeds template max 4 for monotone_increasing

SetPrior(target="omega_CL", family=Normal(mu=0, sigma=1))
  → ValidationError: family 'Normal' not valid for target kind 'iiv_sd'
                     (allowed: ['Gamma', 'HalfCauchy', 'HalfNormal', 'InvGamma'])

SetPrior(target="unknown_param", family=Normal(mu=0, sigma=1))
  → ValidationError: target 'unknown_param' does not resolve to any parameter
                     (structural: ['CL', 'V', 'ka'])
```

---

## Backend lowering — one spec, three targets

The same spec:

```
model {
    absorption: FirstOrder(ka=1.0)
    distribution: OneCmt(V=30.0)
    elimination: Linear(CL=2.0)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
}
```

### nlmixr2 (R)

```r
function() {
  ini({
    tka   <- log(1.0); tcl <- log(2.0); tv <- log(30.0)
    eta.cl ~ 0.1; eta.v ~ 0.1
    add.sd <- 0.1
  })
  model({
    ka <- exp(tka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv + eta.v)
    d/dt(depot)  = -ka * depot
    d/dt(center) =  ka * depot - (cl / v) * center
    cp <- center / v
    cp ~ prop(add.sd)
  })
}
```

### Stan / Torsten

```stan
parameters {
  real log_ka;  real log_cl;  real log_v;
  vector<lower=0>[2] omega;    // [ω_CL, ω_V]
  real<lower=0> sigma_prop;
  matrix[2, N] eta;
}
model {
  // priors injected from DSLSpec.priors (defaults if empty)
  log_ka ~ normal(0, 2);  log_cl ~ normal(log(2.0), 2);  log_v ~ normal(log(30.0), 2);
  omega  ~ cauchy(0, 1);  sigma_prop ~ cauchy(0, 0.1);
  to_vector(eta) ~ std_normal();
  // likelihood uses Torsten PKModelOneCpt with log_cl + omega[1]*eta[1,i], etc.
}
```

### JAX / Diffrax (NODE or mechanistic)

```python
# Mechanistic skeleton — same ODE; IIV applied at solve-time.
def rhs(t, y, theta):
    depot, center = y
    ka, cl, v = theta
    return jnp.array([-ka * depot, ka * depot - (cl / v) * center])

# IIV: theta_i = theta * exp(eta_i), eta_i ~ N(0, diag(omega))
# Solver: Diffrax.Tsit5, adaptive step; Optax Adam on pooled NLL.
```

The emitters are in `nlmixr2_emitter.py`, `stan_emitter.py`, and the NODE harness in
`backends/node_*.py`. Any spec that passes `validate_dsl` for the target lane will
lower cleanly in nlmixr2; Stan additionally requires that BLQ M3/M4 and IOV are
absent (Phase 3 work) and that NODE modules are not present (Stan has no neural-ODE
support).

---

## Extensibility — two tracks

Adding things to Formular falls into two very different workflows:

### Track 1 — New module types (rare; requires pharmacometric review)

Adding a new `Absorption`, `Distribution`, `Elimination`, or `Observation` module
(e.g., a new mechanistic absorption model) requires:

1. A formal ODE specification (parameters, identifiability properties).
2. Grammar rule in `pk_grammar.lark`, AST class in `ast_models.py`, validator branch
   in `validator.py`, at least one emitter, and unit/property/golden tests.
3. Pharmacometrician review (this is the source-of-truth for pharmacological
   correctness).

This is tracked as PRD §10 Q1 and is expected to go through an ADR.

### Track 2 — New enum values (common; lightweight)

New `constraint_template` values, new `CovariateLink.form` values, new `IOV.occasion`
kinds, new prior families — these are typed enum extensions. The process:

1. Extend the `Literal` / enum in the relevant AST file or `priors.py`.
2. Extend the admissibility matrix (for priors) or the lane-ceiling table (for NODE
   templates).
3. Extend each emitter that lowers that axis.
4. Add tests. Benchmark-gated for NODE constraint templates.

Separating these two tracks keeps grammar-version churn decoupled from routine enum
extensions — important for the agentic LLM backend, which may propose new constraint
templates without implying a full grammar bump.

---

## Cross-references

- **README.md §"Formular — The PK DSL"** — public-facing summary (grammar overview,
  transform count, DSL-axis module table).
- **ARCHITECTURE.md §2.2 & §4.1** — compiler stack, `BackendRunner` protocol, and
  bundle schema.
- **PRD_APMODE_v0.3.md §4.2.5** — canonical specification of Formular grammar and
  constraints.
- **policies/\*.json** — lane-specific policy thresholds consumed by `validate_dsl`
  and the governance gates.

## Pharmacometric references

- **Savic et al. (2007)** — Transit compartment model, *J Pharmacokinet Pharmacodyn*
  34:711–726.
- **Mager & Jusko (2001)** — TMDD full binding, *J Pharmacokinet Pharmacodyn*
  28:507–532.
- **Gibiansky et al. (2008)** — TMDD QSS approximation, *J Pharmacokinet Pharmacodyn*
  35:573–591.
- **Beal (2001)** — BLQ M3 likelihood, *J Pharmacokinet Pharmacodyn* 28:481–504.
- **Ahn et al. (2008)** — BLQ method comparison (M1–M7), *J Pharmacokinet
  Pharmacodyn* 35:401–421.
- **Schmidli et al. (2014)** — Robust MAP priors for historical borrowing,
  *Biometrics* 70:1023–1032.
- **Anderson & Holford (2008)** — Allometric scaling defaults, *Clin Pharmacokinet*
  47:455–467.
- **Vehtari et al. (2021)** — Rank-normalized R̂ + ESS, *Bayesian Analysis* 16:667–718.
- **FDA-2025-D-3217** (Jan 2026) — Draft guidance on Bayesian methodology; source of
  the `PriorSpec.source` + `justification` taxonomy used by Gate 2.
