# Formular — The APMODE PK Specification Language

Formular is a typed domain-specific language for specifying population pharmacokinetic models. It is the single source of truth for model structure in APMODE: every model — whether dispatched to nlmixr2, Stan, or a neural ODE backend — is expressed as a Formular specification, compiled to a typed AST, validated against pharmacometric constraints, and lowered to backend-specific code.

The name evokes *formulary* (pharmacy) and *formula* (mathematics). The language is deliberately constrained: it captures what a pharmacometrician would write on a whiteboard, not what a programmer would write in R or Python.

---

## Why a dedicated language?

1. **Safety boundary for agentic AI.** The Phase 3 LLM backend operates exclusively through Formular transforms — it cannot emit raw code. Every proposed model change is a typed AST operation that the compiler can validate before compilation. This is the primary safety mechanism against unbounded code generation.

2. **Backend independence.** A single Formular spec lowers to nlmixr2 R code, Stan programs, or JAX/Diffrax neural ODE configurations. Adding a backend means writing a new emitter, not redesigning the model specification.

3. **Reproducibility.** The compiled `DSLSpec` (a Pydantic model) is the serializable unit stored in the reproducibility bundle. Any candidate model can be reconstructed from its Formular AST without re-running the search.

4. **Constraint enforcement.** Pharmacometric constraints (volumes > 0, rates > 0, NODE dimensions within lane ceilings, valid constraint templates) are enforced at the language level, before any backend code is generated.

---

## Grammar overview

A Formular model has five axes, each required:

```
model {
    absorption:  <absorption-module>
    distribution: <distribution-module>
    elimination:  <elimination-module>
    variability:  { <variability-items...> }
    observation:  <observation-module>
}
```

### Absorption

| Module | Syntax | Parameters |
|--------|--------|------------|
| First-order | `FirstOrder(ka=1.0)` | ka: absorption rate constant (h⁻¹) |
| Zero-order | `ZeroOrder(dur=2.0)` | dur: infusion duration (h) |
| Lagged first-order | `LaggedFirstOrder(ka=1.0, tlag=0.5)` | ka, tlag: lag time (h) |
| Transit | `Transit(n=4, ktr=2.0, ka=1.0)` | n: compartment count, ktr: transit rate, ka |
| Mixed | `MixedFirstZero(ka=1.0, dur=2.0, frac=0.6)` | frac: fraction first-order |
| NODE (Phase 2) | `NODE_Absorption(dim=4, constraint_template=monotone_increasing)` | dim: hidden layer width |

### Distribution

| Module | Syntax | Parameters |
|--------|--------|------------|
| One-compartment | `OneCmt(V=30.0)` | V: volume of distribution (L) |
| Two-compartment | `TwoCmt(V1=30.0, V2=40.0, Q=5.0)` | V1, V2: volumes; Q: intercompartmental CL |
| Three-compartment | `ThreeCmt(V1=30.0, V2=40.0, V3=20.0, Q2=5.0, Q3=2.0)` | |
| TMDD full | `TMDD_Core(V=30.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05)` | Target-mediated drug disposition |
| TMDD QSS | `TMDD_QSS(V=30.0, R0=10.0, KD=0.1, kint=0.05)` | Quasi-steady-state approximation |

### Elimination

| Module | Syntax | Parameters |
|--------|--------|------------|
| Linear | `Linear(CL=2.0)` | CL: clearance (L/h) |
| Michaelis-Menten | `MichaelisMenten(Vmax=50.0, Km=5.0)` | Saturable elimination |
| Parallel linear + MM | `ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)` | Mixed linear and saturable |
| Time-varying | `TimeVarying(CL=2.0, kdecay=0.1, decay_fn=exponential)` | CL with time-dependent decay; kdecay optional (default 0.1) |
| NODE (Phase 2) | `NODE_Elimination(dim=4, constraint_template=bounded_positive)` | |

### Variability

Multiple variability items are grouped in braces:

```
variability: {
    IIV(params=[CL, V1, ka], structure=block)
    IOV(params=[CL], occasions=ByStudy)
    CovariateLink(param=CL, covariate=WT, form=power)
}
```

| Item | Options |
|------|---------|
| **IIV** | `structure`: `diagonal` or `block` |
| **IOV** | `occasions`: `ByStudy`, `ByVisit(col)`, `ByDoseEpoch(col)`, `Custom(col)` |
| **CovariateLink** | `form`: `power`, `exponential`, `linear`, `categorical`, `maturation` |

### Observation

| Module | Syntax |
|--------|--------|
| Proportional | `Proportional(sigma_prop=0.1)` |
| Additive | `Additive(sigma_add=0.5)` |
| Combined | `Combined(sigma_prop=0.1, sigma_add=0.5)` |
| BLQ M3 | `BLQ_M3(loq_value=0.1)` or `BLQ_M3(loq_value=0.1, error_model=combined, sigma_prop=0.2, sigma_add=0.5)` |
| BLQ M4 | `BLQ_M4(loq_value=0.1)` or `BLQ_M4(loq_value=0.1, error_model=additive, sigma_prop=0.1, sigma_add=1.0)` |

---

## Complete example

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

This specifies a two-compartment model with transit absorption, parallel linear + Michaelis-Menten elimination, block IIV on three parameters, allometric weight scaling on clearance, and combined residual error.

---

## Compilation pipeline

```
Formular text
    │
    ↓
Lark parser (Earley)  →  Parse tree
    │
    ↓
DSLTransformer        →  DSLSpec (typed Pydantic AST)
    │
    ↓
Semantic validator     →  Constraint enforcement
    │                     (volumes > 0, NODE dims ≤ lane ceiling, etc.)
    ↓
Backend emitter        →  nlmixr2 R code / Stan program / JAX config
```

**Entry point:** `apmode.dsl.grammar.compile_dsl(text) → DSLSpec`

**Key files:**

| File | Role |
|------|------|
| `src/apmode/dsl/pk_grammar.lark` | Lark EBNF grammar |
| `src/apmode/dsl/transformer.py` | Parse tree → typed AST |
| `src/apmode/dsl/ast_models.py` | Pydantic AST node definitions |
| `src/apmode/dsl/validator.py` | Semantic constraint enforcement |
| `src/apmode/dsl/nlmixr2_emitter.py` | AST → nlmixr2 R code |
| `src/apmode/dsl/stan_emitter.py` | AST → Stan program |

---

## Constraint templates (NODE modules)

NODE absorption and elimination modules require a `constraint_template` that restricts the learned function's shape:

| Template | Meaning |
|----------|---------|
| `monotone_increasing` | Output monotonically increases with input |
| `monotone_decreasing` | Output monotonically decreases with input |
| `bounded_positive` | Output is strictly positive |
| `saturable` | Output increases then plateaus (Michaelis-Menten-like) |
| `unconstrained_smooth` | No shape constraint; smooth output only |

NODE `dim` ceilings are lane-dependent: ≤8 for Discovery, ≤4 for Translational Optimization. Submission lane does not admit NODE modules at all.

---

## Semantic validation rules

The validator enforces pharmacometric constraints before any code is generated:

| Rule | Constraint |
|------|-----------|
| Positivity | Volumes, rates, clearances, sigmas must be > 0 |
| Unit interval | `MixedFirstZero.frac` must be in (0, 1) exclusive |
| Non-negative | `tlag` must be >= 0 |
| Positive integer | Transit `n`, NODE `dim` must be >= 1 |
| NODE lane admissibility | NODE modules rejected in Submission lane |
| NODE dim ceilings | dim <= template max dim AND dim <= lane ceiling |
| Block IIV | `structure=block` requires >= 2 params |
| TMDD requires LinearElim | TMDD_Core/QSS distribution requires Linear elimination |
| No duplicate IIV params | Same parameter cannot appear in multiple IIV blocks |
| No duplicate CovariateLinks | Same param+covariate pair cannot appear twice |
| No variability on Transit n | Transit `n` cannot have IIV/IOV/CovariateLink |
| IIV/IOV/CovariateLink params exist | Must reference a structural parameter |
| Decay function support | Only `exponential` decay is implemented |

All violations are accumulated (non-fail-fast), matching the Pandera `lazy=True` philosophy.

---

## Extensibility

The Formular grammar is intentionally tight. Adding new module types requires:

1. A formal specification (ODE system, parameter constraints, identifiability requirements)
2. Pharmacometrician review
3. New NODE constraint templates require benchmark validation on Suite A scenarios

This is an open design question (PRD §10 Q1). The grammar is easy to extend (Lark EBNF), but the validation and lowering layers must be updated in concert.

---

## References

- **PRD §4.2.5** — Canonical specification of Formular grammar and constraints
- **ARCHITECTURE.md §2.2** — Compiler stack and testing strategy
- **Savic et al. (2007)** — Transit compartment model, J Pharmacokinet Pharmacodyn 34:711-726
- **Mager & Jusko (2001)** — TMDD full binding model, J Pharmacokinet Pharmacodyn 28:507-532
- **Gibiansky et al. (2008)** — TMDD QSS approximation, J Pharmacokinet Pharmacodyn 35:573-591
