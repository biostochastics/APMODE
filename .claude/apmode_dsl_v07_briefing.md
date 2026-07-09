# APMODE v0.7+ DSL — SOTA Absorption Extension Briefing

## Goal

Design the v0.7+ extension of the APMODE PK DSL to admit state-of-the-art absorption modeling: **sum of Inverse Gaussians (sumIG)**, **Erlang absorption**, and adjacent SOTA constructs (Weibull, parallel routes, F.A.T./PBFTPK). Produce a comprehensive, implementation-ready plan that a Phase-3+ engineer can execute without re-debating scope.

## What APMODE Is (load-bearing context)

APMODE is a governed meta-system composing five PK paradigms (classical NLME, automated structural search, agentic LLM, hybrid mechanistic-NODE, Bayesian Stan/Torsten) into a single workflow. The DSL is the moat — every backend, including the agentic LLM, operates **exclusively** through the typed grammar and a fixed set of agent transforms. The agent cannot emit raw R/Stan code.

Three operating lanes exist:
- **Submission** — regulator-facing; classical NLME only; NODE/agentic ineligible for "recommended" status.
- **Discovery** — broad search; admits NODE + agentic.
- **Optimization** — translational; LORO-CV, fixed-THETA evaluation harness contracts.

Dispatch is gated by a typed **Evidence Manifest** from the Data Profiler (richness ∈ {sparse, moderate, rich}, absorption_coverage ∈ {adequate, inadequate}, etc.). The manifest is consumed before any model is built — `richness=sparse + absorption_coverage=inadequate` already gates NODE today.

Governance is a **gated funnel** (Gate 1 technical → Gate 2 lane-admissibility → Gate 2.5 → Gate 3 ranking), with thresholds in versioned `policies/<lane>.json` (currently policy schema `0.6.0`). Reproducibility is per-bundle: every spec, transform, candidate, and gate decision is journaled.

## Current DSL Surface (v0.6)

Grammar (`src/apmode/dsl/pk_grammar.lark`) — top-level: `model { absorption distribution elimination variability_block observation }`.

**Absorption variants today** (also encoded as Pydantic AST in `ast_models.py`):
- `IVBolus()` — no absorption (added later, no params)
- `FirstOrder(ka)`
- `ZeroOrder(dur)`
- `LaggedFirstOrder(ka, tlag)`
- `Transit(n, ktr, ka)` — Savic 2007; n estimated continuously via rxode2's gamma interpolation
- `MixedFirstZero(ka, dur, frac)` — parallel zero+first-order
- `NODE_Absorption(dim, constraint_template)` — Discovery/Optimization only

**Agent transforms today** (`transforms.py`) — fixed enumerated set:
- `swap_module(position, new_module)`
- `add_covariate_link(param, covariate, form)`
- `adjust_variability(param, action ∈ {add, remove, upgrade_to_block})`
- `set_transit_n(n)`
- `toggle_lag(on)`
- `replace_with_node(position, constraint_template, dim)`
- `set_prior(target, distribution, ...)`

Validators (`validator.py`) enforce: positive params, NODE dim ≤ template ceiling × lane ceiling, frac ∈ (0,1), `Transit.n ≥ 1`, etc. All violations surface (Pandera-lazy semantics).

Emitters: `nlmixr2_emitter.py` lowers AST → R `function() { ini({}) model({}) }`; `stan_emitter.py` lowers AST → Stan/Torsten. Both must be kept in lockstep with any new variant.

`structural_param_names()` on DSLSpec returns the parameter list every variability/prior/covariate-link mechanism resolves against — adding a new absorption module means adding the parameter list it contributes.

## SOTA Constructs to Consider (from literature scan)

1. **Erlang absorption** — n integer transit compartments with shared `ktr`; analytically equivalent to `Transit(n, ktr, ka=∞)` collapsing to gamma density (shape=n, rate=ktr). Different from current `Transit` because (a) integer n, (b) no terminal first-order `ka`, (c) admits closed-form for `n ≤ ~7`. **Erlang frequency distribution** — Hong et al. 2017 (pregabalin) showed inferior to transit for that drug; still useful as a constrained baseline.

2. **Sum of Inverse Gaussians (sumIG / nIG)** — Csajka, Drover, Verotta (2005, *Pharm Res*); Weiss & Wegner (2022, *Pharm Res*, talinolol/rifampicin); Wagner et al. (2014) mavoglurant — k weighted IGs as input rate function: `I(t) = D·F·Σ wᵢ·IGᵢ(t; MTᵢ, RD²ᵢ)`, `Σwᵢ = 1`. Captures double peaks, prolonged release, food effect, formulation. **Stronger than transit** for highly variable / multi-peak absorption (e.g. mavoglurant ER, talinolol w/ P-gp interaction). Implemented in nlmixr's reference set as `invgaussian` (single IG) and ADAPT5; sumIG=2 is the most-used variant.

3. **Weibull absorption** — Pumas `@delay` macro + Weibull(k, λ); nlmixr2lib has `addWeibullAbs(ntransit, wa, wb, ka, ktr)` — combines Weibull dose-input with a transit chain. Already shipping in nlmixr2 ecosystem.

4. **Parallel first-order routes** — two simultaneous depot compartments, fraction `F1` to one and `1-F1` to the other (e.g. fast/slow GI absorption, sublingual vs. GI per Pumas PK43). Already common but not in our DSL.

5. **F.A.T. / PBFTPK** (Macheras & Tsekouras 2022–2025) — finite absorption time τ replaces ka; regulatorily disruptive but not yet in FDA/EMA guidance. **Lower priority** — research branch only.

6. **Time-dependent ka(t)** — sigmoidal Emax `ka(t) = ka·tᶜ/(t50ᶜ + tᶜ)` (Poggesi PAGE 2025); double Weibull. Niche.

## Hard Architectural Rules Any Proposal Must Respect

1. **Grammar is the single source of truth.** Every new absorption form needs (a) a Lark rule, (b) a Pydantic AST class, (c) a validator branch, (d) two emitter branches (nlmixr2 + Stan), (e) a `structural_param_names()` contribution, (f) golden snapshot tests, (g) priors registry entries.

2. **Agent transforms are an allowlist, not a sandbox.** Any new structural option means new transforms (e.g. `set_sumig_k`, `convert_to_erlang`) — each transform must validate against the current spec and produce a new spec via `apply_transform`. The agentic backend cannot reach the new construct without an enumerated transform.

3. **Profiler manifest gates dispatch.** sumIG with k≥2 is weakly identified on sparse data and typically requires IV reference + dense oral sampling + multi-formulation (Weiss approach: fix IV-derived disposition, fit input). The Profiler must produce a flag (`absorption_complexity_eligible`?) that gates `SumIG(k=2,3)` candidacy *before* it reaches the search.

4. **Cross-paradigm comparability.** sumIG / Erlang / Weibull all change the observation likelihood and the parameter dimensionality. The Gate 3 ranker assumes NLPD comparability — the protocol for comparing a SumIG fit vs. a Transit fit on the same data must be specified (this is open question §10 Q2 in PRD; treat as scientific design).

5. **Submission lane purity.** Erlang and Transit are accepted regulatory practice; sumIG is academic-grade but not a typical regulatory submission tool. Lane-admissibility table must explicitly enumerate which new absorption forms are eligible per lane (analogous to NODE rules).

6. **Lane-policy floors.** Per-lane policy JSON gains new keys (e.g. `sumig_max_k`, `sumig_min_richness`, `erlang_max_n`, `weibull_min_richness`). Bumping `policy_gate_schema_version` is mandatory.

7. **Zero coupling to nlmixr2-only.** Stan/Torsten emitter must support every new variant; sumIG via mixture analytical input is straightforward in Torsten's `pmx_solve_*` framework; Erlang via transit chain.

8. **Backwards compat.** Existing bundles, golden snapshots, and the digest contract (`_DIGEST_EXCLUDED_NAMES` set) cannot be silently broken. Adding a new variant is additive; bumping policy schema requires migration notes.

## Questions To Answer (please be specific and opinionated)

For each of the following, give a concrete recommendation with the trade-off, not a survey:

**A. Grammar shape**
1. What concrete Lark rules + Pydantic AST classes for `Erlang(n, ktr)`, `SumIG(k, ...)`, `Weibull(...)`, `ParallelFirstOrder(ka1, ka2, frac)`? Be specific about parameter names, dimensions, and any constraints (e.g. `Σwᵢ = 1` enforcement: simplex via softmax on `k-1` free params).
2. Should sumIG's per-component params (MT, RD²) be vector-valued in the AST (list[float]) or flattened (`MT_1, MT_2, ...`)? Implications for IIV, priors, covariates, validation, emission.
3. Erlang as a `Transit` special case (just constrain `n ∈ ℤ⁺`, drop terminal `ka`) vs. its own variant. Argue the trade-off.

**B. Agent transforms**
4. New transforms required for the agent to reach these absorption forms via DSL paths. Be concrete (`add_ig_component`, `set_sumig_k`, `convert_transit_to_erlang`, `weibull_from_transit`, `add_parallel_route`).
5. Should transforms validate identifiability up-front (e.g. `set_sumig_k(k=3)` rejected when `richness=sparse`)? Or is that the Profiler's job?

**C. Profiler / manifest**
6. New manifest fields to gate the new absorption forms. Concretely: what flags, what thresholds, derived from what data signatures.
7. Should the manifest export a per-form eligibility map (`{Erlang: ok, SumIG_k2: ok, SumIG_k3: blocked_sparse, Weibull: ok}`) or generic richness flags that the dispatch logic interprets?

**D. Identifiability & priors**
8. Default priors per new module (literature-grounded; cite). For sumIG: weakly informative on log(MT) ~ N(log(median sampling time), 1²), log(1/RD²) ~ N(0, 1²); a Beta(1,1) on weights or Dirichlet(α=2) on the simplex. Argue.
9. Two-stage fitting requirement for sumIG: do we *require* IV reference data (or fix disposition externally) before allowing k≥2? If so, where does that gate live?

**E. Emitters**
10. Concrete nlmixr2 lowering pattern for sumIG (closed-form input function, not ODE) — is the `invgaussian` reference in nlmixr2 directly usable, or do we need a custom function definition? sumIG-2 vs. sumIG-3 generalization.
11. Stan/Torsten lowering: analytical input rate to `pmx_solve_*`? Or hand-coded ODE with a forcing function?
12. Erlang lowering: is closed-form gamma density in nlmixr2 better than the transit chain? rxode2's `transit()` already supports both — pick one.

**F. Cross-paradigm ranking**
13. NLPD comparability protocol when one candidate uses transit and another uses sumIG. (The current Gate 3 ranker assumes the same observation likelihood.)
14. Should sumIG be admitted only into Discovery + Optimization lanes, never Submission? Or admit to Submission with an explicit "non-standard absorption" warning in the report?

**G. Roll-out / phasing**
15. Order of implementation: which to ship first? Which to gate behind a flag? Where do golden snapshots and benchmark scenarios slot in (Suite A — new mavoglurant scenario, Suite B — talinolol)?
16. Backwards-compat tax: any risk to existing bundles, digest invariants, RO-Crate projection, agent transform allowlist enumeration?
17. Should we open an **ADR** before merging anything, given this is open question §10 Q2 in the PRD?

## Output Format

Return a **concrete, opinionated plan** organized as:

1. **One-paragraph TL;DR** with the three things you'd do first.
2. **Grammar diff** — proposed Lark rules + Pydantic classes (just the additions).
3. **Transforms diff** — new transform classes with validation + apply logic sketches.
4. **Profiler diff** — new manifest fields + threshold rationale.
5. **Per-question recommendations** — A-G above; one paragraph per question; cite literature where relevant.
6. **Risks & open questions** — what still needs human/scientific judgment.
7. **Phasing** — concrete sequenced work items, ideally tied to PRD phases.

Be opinionated. We have already done the literature survey (Savic 2007; Csajka 2005; Weiss 2022/2023; Hong 2017; Macheras 2022–2025; Pumas/Monolix/MDL/PharmML reference). Skip the lit recap — go straight to design.

Length budget: ~1500–2500 words. Concise, dense, technical.
