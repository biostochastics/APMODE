# ADR 0003 — SOTA Absorption Extension (v0.7)

**Date:** 2026-04-25
**Status:** Accepted
**Context:** v0.7 admits three new absorption forms — `Erlang(n, ktr)`,
`ParallelFirstOrder(ka1, ka2, frac)`, and `SumIG(k=2, MT, RD2, weight_1)` —
into the typed PK DSL (PRD §4.2.5). This is the first material extension of
the absorption module since v0.1 and touches grammar, AST, validator, both
emitters (nlmixr2 + Stan), agent transform allowlist, profiler manifest,
lane-admissibility table, and Gate 3 ranker policy.

This ADR exists so future reviewers don't re-litigate the decisions below.

---

## Decisions

### D1 — Three new absorption variants, not five

**Refused outright:** F.A.T./PBFTPK (Macheras 2024). Regulatorily unblessed
by FDA/EMA; shipping it as a first-class DSL form would undermine
Submission-lane credibility. If a researcher needs F.A.T. they fork APMODE.

**Deferred to v0.8+:** Weibull absorption. Already covered by
`nlmixr2lib::addWeibullAbs`; not urgent. Re-evaluate after v0.7 ships and is
benchmarked.

**Capped at k=2:** SumIG. Wagner 2014 (mavoglurant) used k=2 or k=3, but
k=3 fits required dense IV+oral data. Add `policies/<lane>.json` knob
`sumig_max_k` (default 2). The path to k=3 is a *validator* change, not a
grammar change — `SumIG.k: int` (validator restricts to k ∈ {1, 2} for v0.7).

### D2 — Erlang as a separate AST class, not constrained Transit

Erlang is mathematically equivalent to `Transit(n, ktr, ka→∞)` collapsed
to a gamma density, but the emitter for `Transit` keeps a terminal
first-order `ka` step. Hiding Erlang under "Transit with very large ka"
creates an entire class of silent emitter bugs (rxode2's `transit(n, mtt)`
handles non-integer `n` via gamma interpolation, which is not what Erlang
needs). Separate variant; ~40 lines of Pydantic + validator + two emitter
branches; eliminates the bug class. The agent reaches Erlang via a single
transform `convert_transit_to_erlang(n: int)` rather than via
`swap_module`, capping search-space expansion.

### D3 — ParallelFirstOrder is genuinely distinct from MixedFirstZero

`MixedFirstZero(ka, dur, frac)` = first-order + zero-order routes (Pumas
PK15 patterns). `ParallelFirstOrder(ka1, ka2, frac)` = two parallel
first-order routes (Pumas PK43 patterns; Soufsaf 2021 PMX). The literature
treats them as distinct and the priors differ (slow-route `ka2` is
log-normally centred ~1 log-unit below `ka1`). Distinct AST class.

### D4 — SumIG lowers as a closed-form analytic input rate, not via deconvolution

For nlmixr2: closed-form forcing function injected into `model({})`:
```
ig_i  <- sqrt(RD2_i / (2*pi*t^3)) * exp(-RD2_i*(t-MT_i)^2 / (2*MT_i^2*t))
input <- amt * F * (w_1*ig_1 + (1-w_1)*ig_2)
d/dt(centr) <- input - <elim_expr>
```
For Stan/Torsten: identical analytical rate inside the ODE RHS; explicit
hand-coded form (not `inv_gaussian_lpdf`) so we can guard `t > 0` directly.
v0.7 ships **single-dose only**; multi-dose superposition deferred.

`linCmt()` deconvolution macros are not used — they require a numeric
input function via covariate, which couples the emitter to the data
adapter and would break the producer-side digest contract.

### D5 — SumIG(k≥2) requires a "disposition fixed" gate (two-stage protocol)

Csajka 2005 §4 and Weiss 2022 §5 both document the identifiability collapse
when sumIG-k≥2 is fit jointly with disposition (CL/V/Q). Implement as a
**cross-module validator check**: `SumIG.k >= 2` requires either:
- the `EvidenceManifest.disposition_fixed` flag (IV reference detected,
  or external fixed estimates provided), **or**
- `spec.priors` contains entries for all disposition params with
  `source == "fixed_external"` (or equivalent fixed-prior signal).

Surfaces as a first-class `ValidationError` with actionable message.

### D6 — Lane admissibility table

Add `_LANE_ABSORPTION_ADMISSIBILITY` dict to `validator.py`, parallel to
`_LANE_DIM_CEILING`:

| Form               | Submission | Discovery | Optimization |
|--------------------|------------|-----------|--------------|
| FirstOrder         | ✓          | ✓         | ✓            |
| ZeroOrder          | ✓          | ✓         | ✓            |
| LaggedFirstOrder   | ✓          | ✓         | ✓            |
| Transit            | ✓          | ✓         | ✓            |
| MixedFirstZero     | ✓          | ✓         | ✓            |
| Erlang             | ✓          | ✓         | ✓            |
| ParallelFirstOrder | ✓          | ✓         | ✓            |
| SumIG              | ✗          | ✓         | ✓            |
| NODE_Absorption    | ✗          | ✓         | ✓            |

SumIG is academic-grade; not yet standard regulatory practice. Submission
lane rejects it at Gate 2 with an actionable error.

### D7 — Profiler manifest (schema_version 2 → 3)

Two new fields, both with defaults so v2 manifests load (additive
migration; no breaking change):

```python
absorption_complexity_eligible: dict[str, bool] = Field(default_factory=dict)
disposition_fixed: bool = False
```

`absorption_complexity_eligible.SumIG_k2` is derived from:
- `richness_category == "rich"`, **and**
- `absorption_phase_coverage == "adequate"`, **and**
- (`peak_prominence_fraction >= sumig_k2_min_peak_prominence`
  *or* `absorption_complexity == "multi-phase"`).

Threshold `sumig_k2_min_peak_prominence` (default 0.15, per Weiss 2022
simulation) lives in `policies/profiler.json` for tunability without code
changes. Migration: a `_migrate_v2_to_v3` shim in the manifest loader fills
defaults for old bundles (no behavioural change for existing reproducers).

### D8 — NLPD comparability protocol (within classical NLME)

For absorption-form ranking *within* the classical paradigm (Transit vs
SumIG vs Erlang), NLPD is directly comparable because all forms use the
**same observation likelihood** (Proportional / Additive / Combined / BLQ).
Only the predicted concentration trajectory changes.

Required protocol (Gate 3):
1. Frozen error model across the candidate set.
2. Identical LORO-CV folds for every candidate.
3. Penalised metric: `NLPD_AICc = mean_NLPD + AICc_penalty/n_obs`
   (frequentist) or `ELPD_LOO ± SE` (Bayesian, PSIS-LOO per Vehtari 2017).
4. Policy floor before SumIG can outrank Transit/Erlang:
   `ΔAICc > 4` AND `ΔNLPD_AICc > 0.05 nats/obs`. Bayesian:
   `ΔELPD > 2·SE`. Versioned in `policies/optimization.json` as
   `sumig_nlpd_floor`, `sumig_aicc_floor`. Tie → parsimony wins.

**Not addressed by this ADR:** PRD §10 Q2 (NODE↔classical comparability).
That remains an open scientific design problem and is **orthogonal** to the
absorption-form extension shipped here.

### D9 — Default priors per new module (literature-grounded)

- **Erlang:** `log(ktr) ~ N(log(2 / t_50_obs), 1²)`. `n` is structural,
  no prior.
- **ParallelFirstOrder:** inherit FirstOrder's `ka` prior on `ka1`;
  `log(ka2) ~ N(log(ka1) - 1, 0.5²)` (slow route ~1 log-unit below fast);
  `logit(frac) ~ N(0, 1²)`.
- **SumIG (k=2):** `log(MT_1) ~ N(log(t_median_sampling), 1²)`;
  `log(MT_2 - MT_1) ~ N(log(t_median_sampling), 1²)` (positive-difference
  parameterisation prevents label switching);
  `log(RD2_i) ~ N(0, 1²)`; `weight_1 ~ Beta(2, 2)` (regularises away from
  boundaries).

`t_median_sampling` and `t_50_obs` are sourced from
`EvidenceManifest.observation_signature`.

### D10 — Phasing

| Week | Scope                                                                        |
|------|------------------------------------------------------------------------------|
| 1    | This ADR + grammar/AST/validator/transformer for the three new variants     |
| 2    | nlmixr2 emitter branches + Stan emitter branches + golden snapshots         |
| 3    | Profiler manifest fields (v3 schema), lane admissibility table, transforms  |
| 4    | NLPD harness (LORO identical folds, AICc penalty, policy floor) + benchmarks |

### What v0.7 explicitly does NOT ship

- F.A.T./PBFTPK (refused).
- Weibull (deferred to v0.8).
- SumIG with k=3 (deferred behind `sumig_max_k`).
- Time-dependent ka(t) sigmoidal Emax / double Weibull (refused — niche).
- NODE-vs-classical NLPD protocol (still open per PRD §10 Q2).
- Multi-dose SumIG superposition (deferred).

---

## Re-evaluation triggers

- **F.A.T./PBFTPK:** if FDA/EMA issues guidance admitting τ-parameterised
  absorption, revisit refusal.
- **Weibull:** ship if a benchmarked downstream user explicitly requests
  it AND `nlmixr2lib::addWeibullAbs` integration via DSL paths is
  cleaner than dropping to raw R.
- **SumIG k=3:** ship if (a) a real Discovery/Optimization dataset
  demonstrates Gate 3 NLPD floor pass at k=3 vs k=2 in held-out fold and
  (b) at least three published reference implementations (NONMEM/Pumas/
  nlmixr2) ship a comparable parameterisation.
- **Profiler thresholds:** `sumig_k2_min_peak_prominence` is configurable
  in `policies/profiler.json` from day one — recalibrate as real-world
  evidence accumulates.
