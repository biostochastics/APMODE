# Eleveld Propofol DSLSpec Coverage Assessment

> Plan Task 42 — Phase-1 Bayesian fixture eligibility check.
>
> **Recommendation: NO-GO for v0.6 Phase-1 Bayesian fixtures.**
>
> Defer Eleveld propofol to discovery work; ship Phase-1 with the
> simpler vancomycin Roberts 2011 fixture only.

## Source

- Eleveld DJ, Colin P, Absalom AR, Struys MMRF (2018). Pharmacokinetic-
  pharmacodynamic model for propofol for broad application in anaesthesia
  and sedation. *Br J Anaesth* 120(5):942-959.
  DOI: [10.1016/j.bja.2018.01.018](https://doi.org/10.1016/j.bja.2018.01.018).

## Eleveld 2018 covariate model — what the paper specifies

The Eleveld PK model is a 3-compartment IV-infusion mammillary model with
the following covariate effects on the typical-value parameters:

| Parameter | Covariate effect |
|-----------|------------------|
| V1, V2, V3 | Allometric scaling on **fat-free mass** (FFM, derived from weight + sex + height/BMI) |
| CL | Allometric on FFM × **maturation** (Hill on PMA / post-menstrual age) × **age** decay (linear in years for adults) × **opioid co-administration** (categorical multiplier) |
| Q2, Q3 | Allometric on FFM × **age**-modulated factor (linear / piecewise for paediatric vs adult) |
| V2 | Linear age effect on top of allometric scaling |
| KE0 (PD) | Power-of-weight (allometric without maturation) |

There is no single covariate primitive that captures the *fat-free-mass*
substitution: FFM is itself a derived covariate computed from (weight,
height, sex) before the allometric `power` form is applied.

## DSLSpec primitive inventory (apmode/dsl/ast_models.py)

`CovariateLink.form` admits five values:

- `power` — allometric `(cov / cov_ref)^β` form
- `exponential` — `exp(β · cov)` form
- `linear` — `1 + β · cov` form
- `categorical` — multiplicative offset on a binary covariate
- `maturation` — Hill function on a single covariate
  (`cov^β / (cov^β + TM50^β)`)

There is no primitive for:

- **Derived covariates** (FFM = f(weight, height, sex)). Each
  `CovariateLink` references exactly one `covariate` field; FFM would
  need to be precomputed and supplied as a column in the input CSV.
- **Composite covariate effects on a single parameter** — e.g. CL =
  base × allometric(FFM) × maturation(PMA) × decay(age) × opioid. The
  Stan emitter applies covariates in sequence but every form is
  multiplicative; the *piecewise / age-decay* form Eleveld uses for
  CL above ~30 years is not expressible.
- **PD/effect-site compartment** — Eleveld also defines a
  pharmacodynamic ke0 parameter; APMODE's DSL is currently PK-only
  (no `EffectSite` primitive in `ast_models.DistributionModule`).

## Backend coverage of `maturation`

| Backend | `maturation` form support |
|---------|--------------------------|
| nlmixr2 (`src/apmode/dsl/nlmixr2_emitter.py`) | Implemented — emits `cov^β / (cov^β + TM50^β)` |
| Stan (`src/apmode/dsl/stan_emitter.py:782`) | **NotImplementedError** — explicit raise |

The Stan backend is the one Phase-1 Bayesian fixtures would target.
Until the Stan emitter learns the maturation form, an Eleveld fixture
that uses `maturation` cannot run on the Bayesian path at all.

## Gap summary

| Required by Eleveld | Available in DSL? |
|---------------------|------------------|
| Allometric scaling on weight | Yes (`power`) |
| Maturation function on PMA | nlmixr2 yes; **Stan no** |
| FFM derivation (weight + sex + height) | **No** — needs a precomputed column |
| Age-decay on CL above adulthood | **No** — not expressible as a single `CovariateLink.form` |
| Opioid categorical multiplier on CL | Yes (`categorical`) |
| Sex categorical on V1/V3 | Yes (`categorical`) |
| Effect-site compartment (PD) | **No** — DSL is PK-only |

## Recommendation

**NO-GO for v0.6 Phase-1 Bayesian fixtures.**

Three blocking gaps make Eleveld unsuitable for the v0.6 release scope:

1. **Stan emitter does not implement the `maturation` form.** Adding
   it is a focused but non-trivial change — needs the prior schema for
   `TM50` (HalfNormal? LogNormal?), a Stan codegen path for
   `pow(cov, beta) / (pow(cov, beta) + pow(TM50, beta))`, and golden
   tests against the nlmixr2 emitter's output. Discovery-track work.
2. **No derived-covariate path** for FFM. Either the input CSV must
   carry a precomputed `FFM` column or the DSL needs a
   `DerivedCovariate` primitive (out of scope for the v0.6 stretch
   goal of a Bayesian fixture pair).
3. **Age-decay on CL** doesn't fit any of the five covariate forms;
   it's a piecewise-linear function of age that the published model
   defines via a `gAge` weighting term. Again, discovery work.

## Phase-1 Bayesian fixture set

Ship **vancomycin (Roberts 2011)** only. It uses a simple
1-compartment model with allometric scaling on weight + creatinine
clearance on CL (linear) — every covariate is a single-primitive
`CovariateLink`, and the Stan emitter handles all three forms today.

When the gaps above are closed (likely in v0.7 alongside Stan
maturation + the agentic-LLM transform set extension), revisit
Eleveld and add the propofol fixture.

## Related artifacts

- `src/apmode/dsl/ast_models.py:CovariateLink` — primitive enumeration
- `src/apmode/dsl/stan_emitter.py:782` — maturation NotImplementedError
- `src/apmode/dsl/nlmixr2_emitter.py:230` — maturation reference impl
- `benchmarks/suite_c/propofol_eleveld_2018.yaml` — to be created in v0.7
