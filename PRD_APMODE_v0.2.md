# PRD: Adaptive Pharmacokinetic Model Discovery Engine (APMODE)

**Version:** 0.2 — Revised per expert review
**Date:** 2026-04-11
**Status:** RFC

---

## 1. Problem Statement

Finding the right pharmacokinetic model for a given dataset remains an unsatisfying mix of manual craftsmanship, narrow automation, and disconnected innovation. As of mid-2026, the field operates across four paradigms (Timonen, Generable, April 2026): classical pharmacometric modeling, automated model-space exploration (nlmixr2auto; Chen et al. 2024), LLM-agentic model construction (AutoStan; Dürr et al. 2026), and neural ODE models (Lu et al. 2021; Bräm et al. 2023; Giacometti et al. 2025; Maurel et al. 2026; Uni-PK 2026).

No single paradigm is sufficient. Classical models are too rigid, automated search is too bounded, agentic LLMs lack domain grounding, and NODEs lack interpretability and regulatory acceptance. The field needs a system that **composes** these approaches into a coherent, governed workflow.

---

## 2. Product Vision

**APMODE** is an end-to-end system for population PK model discovery that orchestrates mechanistic modeling, automated structural search, LLM-driven model construction, and neural ODE fitting within a governed decision framework. The user provides concentration-time data, dosing records, and covariates. APMODE returns an admissible candidate set with lane-specific ranking, documented exclusion reasons, diagnostics, interpretability reports, and submission-supporting documentation for human review.

**Core thesis:** The right architecture is not one model class but a *governed search-and-ranking meta-system* over heterogeneous model classes, organized into intent-specific operating lanes with explicit admissibility gates.

---

## 3. Product Modes (Operating Lanes)

The PRD explicitly separates three product goals into distinct operating lanes. These are not configurable weights on a universal loop — they are different pipelines with different admissible backends, stopping rules, and evidence thresholds.

### 3.1 Submission Lane

**Purpose:** Produce a defensible model for regulatory submission (IND, NDA, BLA pharmacometrics appendix).

**Admissible backends:** Classical NLME + bounded automated search only. Automated search is bounded to pre-specified, submission-appropriate structural and covariate spaces.

**Stopping rules:** Converged estimation, acceptable GOF diagnostics (CWRES, VPC coverage within pre-specified thresholds), adequate parameter identifiability, shrinkage within acceptable bounds, covariate model justified by clinical and statistical criteria.

**Evidence standard:** Full audit trail. Interpretable parameters. Reproducible estimation pathway. Aligned with ICH M15 expectations for model credibility documentation, context-of-use specification, and risk-based assessment.

**NODE/agentic models are not eligible for "recommended" status in this lane.** They may appear in supplementary analysis as hypothesis-generating evidence.

### 3.2 Discovery Lane

**Purpose:** Find the best structural description of the data, including novel structures the modeler might not have considered.

**Admissible backends:** Classical NLME + automated search + hybrid mechanistic-NODE + agentic LLM proposals.

**Stopping rules:** Diminishing NLPD improvement across iterations; user-defined iteration budget; stability of top-ranked candidates across re-runs.

**Evidence standard:** Predictive performance is primary. Interpretability is valued but not required. Models that cannot be interpreted directly should include functional distillation outputs (see §4.2.7).

### 3.3 Translational Optimization Lane

**Purpose:** Produce a model optimized for dose regimen generalization and precision dosing applications.

**Admissible backends:** Classical NLME baseline + hybrid mechanistic-NODE + latent ODE (Maurel-style, when mature).

**Stopping rules:** Leave-one-regimen-out cross-validation performance within pre-specified thresholds; simulation faithfulness under dosing perturbation.

**Evidence standard:** Predictive accuracy under regimen transfer is primary. Extrapolation behavior must be characterized. Uncertainty quantification required.

---

## 4. System Architecture

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     APMODE Platform                         │
│                                                             │
│  ┌───────────┐   ┌──────────┐   ┌───────────────┐         │
│  │   Data     │──▶│ Evidence │──▶│  Lane Router  │         │
│  │  Profiler  │   │ Manifest │   │  (by intent)  │         │
│  └───────────┘   └──────────┘   └──────┬────────┘         │
│                                         │                   │
│          ┌──────────────────────────────┼────────┐         │
│          │              │               │        │         │
│     ┌────▼────┐   ┌────▼─────┐   ┌────▼────┐  ┌▼───────┐ │
│     │Classical│   │Automated │   │ Agentic │  │ Hybrid │ │
│     │  NLME   │   │  Search  │   │  LLM    │  │  NODE  │ │
│     │Backend  │   │ Backend  │   │ Backend │  │Backend │ │
│     └────┬────┘   └────┬─────┘   └────┬────┘  └───┬────┘ │
│          └──────────────┴──────────────┴───────────┘      │
│                              │                             │
│                    ┌─────────▼──────────┐                  │
│                    │  Governance Layer   │                  │
│                    │  ┌───────────────┐  │                  │
│                    │  │ Admissibility  │  │                  │
│                    │  │    Gates       │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │  Within-Gate   │  │                  │
│                    │  │   Ranking      │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │  Provenance &  │  │                  │
│                    │  │  Reproducib.   │  │                  │
│                    │  └───────────────┘  │                  │
│                    └─────────┬──────────┘                  │
│                              │                             │
│                    ┌─────────▼──────────┐                  │
│                    │  Report Generator  │                  │
│                    └───────────────────-┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

#### 4.2.1 Data Profiler → Evidence Manifest

The profiler is not merely descriptive — it outputs a **formal evidence manifest** that downstream modules must consume. The manifest constrains dispatch.

**Manifest fields:**

| Field | Type | Example |
|---|---|---|
| `route_certainty` | enum: confirmed / inferred / ambiguous | confirmed: IV bolus |
| `absorption_complexity` | enum: simple / multi-phase / lag-signature / unknown | multi-phase |
| `nonlinear_clearance_signature` | bool + confidence | true, 0.78 |
| `richness_category` | enum: sparse (< 4 samples/subject) / moderate (4–8) / rich (> 8) | moderate |
| `identifiability_ceiling` | enum: low / medium / high (based on design + sampling) | medium |
| `covariate_burden` | int (number of candidate covariates) + correlation flag | 12, correlated |
| `blq_burden` | fraction of BLQ observations | 0.15 |
| `missingness_pattern` | enum: MCAR / MAR / informative-suspected | MAR |
| `protocol_heterogeneity` | enum: single-study / pooled-similar / pooled-heterogeneous | pooled-similar |
| `temporal_support` | struct: {absorption_phase_coverage, elimination_phase_coverage} | {adequate, adequate} |

**Dispatch constraints derived from manifest:**

- If `richness_category` = sparse AND `temporal_support.absorption_phase_coverage` = inadequate → NODE backends receive a `data_insufficient` flag and are not dispatched (or are dispatched with explicit low-confidence warning).
- If `nonlinear_clearance_signature` = true → automated search includes MM and parallel elimination candidates.
- If `blq_burden` > 0.20 → all backends must use BLQ-aware likelihood (M3/M4 methods or equivalent).
- If `protocol_heterogeneity` = pooled-heterogeneous → IOV must be tested in classical backends.

#### 4.2.2 Classical NLME Backend

**Primary engine:** nlmixr2 (R). NONMEM and Pumas are optional adapters, not core dependencies.

Accepts model specification via the typed PK DSL (§4.2.5). Runs SAEM and/or FOCEI estimation. Returns standardized result object:

- Parameter estimates with RSE/CI
- ETA shrinkage per parameter
- OFV, AIC, BIC
- GOF diagnostics (observed vs. predicted, CWRES vs. time/pred, QQ)
- VPC (prediction-corrected where appropriate)
- Condition number / practical identifiability flags
- Convergence metadata (iterations, gradient norms, minimization status)

#### 4.2.3 Automated Search Backend

Builds on nlmixr2auto-style logic. Search dimensions:

- **Structural:** 1-cmt ↔ 2-cmt ↔ 3-cmt × absorption variants × elimination variants (all expressed as DSL module combinations)
- **Covariate:** stepwise (SCM forward/backward), or LASSO-on-ETAs for screening
- **Random effects:** diagonal vs. block omega; IIV and IOV candidates
- **Residual error:** additive, proportional, combined

Scoring: AIC/BIC for nested; cross-validated NLPD for non-nested. Emits Pareto frontier (parsimony vs. fit).

Search is bounded by the evidence manifest: e.g., if `absorption_complexity` = simple, transit absorption models are deprioritized (not excluded, but placed in a secondary tier).

#### 4.2.4 Hybrid Mechanistic-NODE Backend

**Phase 2 scope: one architecture only** — hybrid mechanistic + low-dimensional NODE (Bräm-style).

Architecture: the mechanistic skeleton (e.g., 2-cmt distribution + linear clearance) is retained. A small neural network replaces **one** sub-model — typically absorption or elimination — where the classical form shows systematic misfit.

- State dimensions correspond to interpretable compartments
- The NODE sub-model operates on interpretable inputs (concentration, time, covariates)
- Population variability: random effects on NODE input layer weights (low-dimensional, per Bräm et al.)

**What this backend does NOT include in Phase 2:**
- Full latent ODE (Maurel-style) — research branch only
- Cross-compound Uni-PK molecular-feature integration — research branch only
- Arbitrary high-dimensional NODE with opaque latent state — excluded by design

**Interpretability outputs (functional distillation, not SHAP-only):**

- Learned sub-function visualization: plot the NODE-learned clearance law over the observed concentration range; plot learned absorption kernel vs. time
- Parametric surrogate fitting: fit symbolic or parametric surrogates (MM, parallel pathways, transit chain) to the learned sub-function
- Surrogate fidelity quantification: report where the parametric surrogate matches the NODE and where it diverges, over the clinically relevant state space
- Covariate effect characterization: are effects monotone, saturable, subgroup-specific? (derived from NODE partial dependence, not generic SHAP)

This is the interpretability bridge. Its purpose is to answer pharmacometric questions ("what does the learned clearance look like?") not ML questions ("which features were important?").

#### 4.2.5 PK Domain-Specific Language (DSL) and Compiler

**This is the core platform artifact and the primary moat.** It must be built before the agentic LLM backend.

**Typed grammar:**

```
Model := AbsorptionModule × DistributionModule × EliminationModule
         × VariabilityModule × ObservationModule

AbsorptionModule :=
  | FirstOrder(ka)
  | ZeroOrder(dur)
  | LaggedFirstOrder(ka, tlag)
  | Transit(n, ktr)
  | MixedFirstZero(ka, dur, frac)
  | NODE_Absorption(dim, constraints)       # Discovery/Optimization lanes only

DistributionModule :=
  | OneCmt(V)
  | TwoCmt(V1, V2, Q)
  | ThreeCmt(V1, V2, V3, Q2, Q3)
  | TMDD_Core(R0, kon, koff, kint)          # with quasi-steady-state option

EliminationModule :=
  | Linear(CL)
  | MichaelisMenten(Vmax, Km)
  | ParallelLinearMM(CL, Vmax, Km)
  | TimeVarying(CL, decay_fn)
  | NODE_Elimination(dim, constraints)      # Discovery/Optimization lanes only

VariabilityModule :=
  | IIV(params: list, structure: diagonal | block)
  | IOV(params: list, occasions: spec)
  | CovariateLink(param, covariate, functional_form)

ObservationModule :=
  | Proportional(sigma_prop)
  | Additive(sigma_add)
  | Combined(sigma_prop, sigma_add)
  | BLQ_M3(loq_value)
  | BLQ_M4(loq_value)
```

**Allowed agent transforms (for agentic backend):**

- `swap_module(position, new_module)` — e.g., swap Elimination from Linear to MichaelisMenten
- `add_covariate_link(param, covariate, form)` — e.g., allometric weight on CL
- `adjust_variability(param, action: add | remove | upgrade_to_block)`
- `set_transit_n(n)` — change transit compartment count
- `toggle_lag(on | off)`
- `replace_submodel_with_NODE(position, dim, constraints)` — Discovery lane only

Each transform produces a new model spec that is: (a) validated against the grammar, (b) compiled to nlmixr2 or Stan code, (c) logged in the audit trail with before/after specs.

**Constraints enforced at DSL level:**

- State and flow constraints consistent with the selected model family (not a blanket "mass conservation" — which is slightly too broad for all practical implementations, e.g., models with first-pass metabolism or target-mediated disposition where the implemented state variables may not sum to total drug)
- Elimination flows ≥ 0
- Volume parameters > 0
- Rate constants > 0
- Covariate functional forms must be from an approved set (power, exponential, linear, categorical, maturation)

#### 4.2.6 Agentic LLM Backend (Phase 3)

Operates exclusively through DSL transforms (§4.2.5). Cannot write raw code.

**Inputs:** Evidence manifest, current best model(s), their residual diagnostics (CWRES patterns, VPC deficiencies), the search history so far.

**Behavior:** Proposes a sequence of transforms. Each transform is validated, compiled, fit, and evaluated. The agent observes results (NLPD, diagnostics) and proposes the next transform or stops.

**Iteration budget:** Capped at 25 rounds per run (consistent with AutoStan's demonstrated sufficiency at ~20).

**What the agent adds over automated search:** It can reason about *why* a model is misfitting (e.g., "CWRES show time-dependent bias in the elimination phase → try MM elimination") rather than brute-force enumerating. It can also propose compound transforms (swap elimination + add covariate link simultaneously) that search algorithms handle less naturally.

**What the agent cannot do:** Write arbitrary ODE code, propose structures outside the DSL grammar, override admissibility gates, or access held-out data.

### 4.3 Governance Layer

This is the architectural core of the system's credibility. Three subsections.

#### 4.3.1 Admissibility Gates (Gated Funnel, Not Weighted Sum)

Candidate models pass through sequential gates. A failure at any gate is **disqualifying**, not penalizable.

**Threshold operationalization:** All gate thresholds are **versioned policy artifacts**, not hard-coded constants. Each lane ships with a default policy file specifying numeric cutoffs (e.g., CWRES mean threshold, VPC coverage bounds, shrinkage ceiling, outlier fraction limit, seed stability criterion). Policy files are part of the reproducibility bundle and are independently versionable. Default values are calibrated during the benchmark program (§5) and updated as evidence accumulates. Users may override defaults with justification, which is logged in the audit trail.

**Gate 1: Technical Validity**

| Check | Criterion | Applies to |
|---|---|---|
| Convergence | SAEM/FOCEI converged (classical); training loss plateaued, no NaN (NODE) | All |
| Parameter plausibility | No impossible values (negative volumes, CL < 0, ka < 0) | All |
| State trajectory validity | No negative concentrations, no divergent ODE solutions | All |
| Residual diagnostics | CWRES mean < threshold, no extreme outlier fraction | Classical, automated |
| Simulation realism | VPC coverage within [X%, Y%] for 5th/50th/95th percentiles | Classical, automated |
| Split integrity | No held-out subject contamination | All |
| Seed stability | Top model consistent across ≥ 3 random seeds | All |

**Gate 2: Lane-Specific Admissibility**

| Check | Submission | Discovery | Optimization |
|---|---|---|---|
| Interpretable parameterization | Required | Not required | Preferred |
| Reproducible estimation pathway | Required | Required | Required |
| ETA shrinkage < threshold (e.g., 30%) | Required | Informational | Required for dose-dependent params |
| Practical identifiability (all key params) | Required | Informational | Required for dose-related params |
| NODE-only models eligible for "recommended" | No | Yes | Yes (with uncertainty quantification) |
| Leave-one-regimen-out performance | Not required | Not required | Required |

**Gate 3: Within-Gate Ranking**

Only models surviving Gates 1–2 enter ranking. Ranking metrics, applied in order of priority specific to each lane:

*Submission lane:* (1) Diagnostic adequacy (GOF, VPC), (2) parameter precision and identifiability, (3) parsimonious structure, (4) information criteria (AIC/BIC) within the admissible set.

*Discovery lane:* (1) Cross-validated NLPD (subject-level holdout), (2) structural novelty relative to classical baseline, (3) residual pattern improvement, (4) interpretability score (if available).

*Optimization lane:* (1) Leave-one-regimen-out NLPD, (2) simulation faithfulness under dosing perturbation, (3) uncertainty calibration, (4) NLPD on in-distribution holdout.

**Critical design rule:** Cross-validated NLPD requires a common observation model definition across backends when scoring is commensurate. If backends use different observation models (e.g., different BLQ handling), this must be flagged and the comparison qualified.

**Architectural dependency for Phase 2:** Before the hybrid NODE backend can participate in cross-paradigm ranking (Gate 3), a formal observation-model comparability protocol must be specified and validated. This protocol must define: (a) how observation noise is parameterized in NODE vs. NLME backends, (b) under what conditions NLPD scores are directly comparable, and (c) how qualified comparisons are presented when full comparability cannot be achieved. This is not a minor integration detail — it is one of the hardest scientific design problems in the platform and must be resolved before Discovery lane cross-paradigm ranking is considered production-ready.

#### 4.3.2 Provenance and Reproducibility Contract

Every run produces a machine-readable **reproducibility bundle**:

| Artifact | Contents |
|---|---|
| `data_manifest.json` | SHA-256 hash of input dataset; column mapping; BLQ coding |
| `split_manifest.json` | Subject-level train/test/validation assignments per fold; random seed |
| `backend_versions.json` | nlmixr2 version, R version, Python/JAX version (for NODE), LLM model ID + prompt hash (for agentic) |
| `seed_registry.json` | All random seeds used across all backends |
| `search_trajectory.jsonl` | Ordered log of every model attempted: DSL spec, backend, fit status, scores, gate outcomes |
| `failed_candidates.jsonl` | Models that failed gates, with specific failure reasons |
| `candidate_lineage.json` | DAG of model derivation (which model was parent of which) |
| `compiled_specs/` | Directory of compiled model code for every candidate |
| `evidence_manifest.json` | Data profiler output |
| `report_provenance.json` | Timestamps, component versions, who/what generated each section |

This bundle is the unit of reproducibility. Any APMODE result can be fully re-executed from this bundle on matching infrastructure.

#### 4.3.3 Credibility Assessment Framework

Aligned with ICH M15 and FDA AI/ML guidance. For each recommended model, the report includes:

- **Context of use statement:** What decision does this model support? What is the risk level of model failure?
- **Model credibility documentation:** Estimation method, convergence evidence, sensitivity analyses, parameter uncertainty.
- **Data adequacy assessment:** Was the data sufficient to identify the proposed model structure? (Derived from evidence manifest vs. model complexity.)
- **Limitations disclosure:** Known extrapolation boundaries, covariate range restrictions, populations not represented.
- **AI/ML transparency statement** (if NODE or agentic backend was involved): What role did ML play? What guardrails were applied? How was the result validated against mechanistic expectations?

This is **submission-supporting documentation for human review** — not "regulatory-ready" output. The distinction matters. APMODE provides evidence; the pharmacometrician and regulatory reviewer make the judgment.

---

## 5. Validation Strategy (Formal Benchmark Program)

This product lives or dies by benchmark design. Three benchmark suites.

### Suite A: Synthetic Recovery

Simulated datasets with known ground truth. Purpose: can APMODE recover the correct structure and parameters?

| Scenario | True model | Key test |
|---|---|---|
| A1 | 1-cmt oral, first-order absorption | Correct structure identification |
| A2 | 2-cmt IV + extravascular | Compartment count recovery |
| A3 | Transit absorption (n=3) | Transit chain detection; n recovery |
| A4 | Michaelis-Menten elimination | Nonlinear clearance detection |
| A5 | TMDD (quasi-steady-state) | TMDD identification vs. 2-cmt confusion |
| A6 | Mixed covariate effects (allometric WT on CL, categorical renal on CL) | Covariate structure recovery |
| A7 | 2-cmt with NODE-generated nonlinear absorption (ground truth is a custom function) | Can hybrid NODE recover the shape? Can symbolic surrogate approximate it? |

**Metrics:** Structure recovery rate (%), parameter bias (%), parameter coverage (does 95% CI contain truth?), time-to-result.

**Symbolic approximation target (scoped):** For hybrid NODE models in scenarios A4 and A7 only, test whether parametric surrogates fitted to the NODE sub-function recover the true functional form. Target: surrogate matches true function within 20% over the 5th–95th percentile of observed state space. This target applies to selected hybrid model classes, not all NODE dynamics.

### Suite B: Semi-Synthetic Perturbation

Start from real clinical datasets. Inject known perturbations to test robustness.

| Perturbation | Purpose |
|---|---|
| Increase BLQ fraction to 25%, 40% | BLQ robustness |
| Add 5% gross outliers | Outlier sensitivity |
| Remove 50% of absorption-phase samples | Sparse data degradation |
| Pool two studies with different protocols | Protocol heterogeneity handling |
| Add correlated covariates with no true effect | Covariate false-positive rate |
| Reduce to sparse sampling (< 4/subject) in Discovery lane | Rate of incorrect escalation to hybrid NODE under sparse-data perturbation (tests whether evidence manifest and dispatch constraints function correctly) |

**Metrics:** Model rank stability, parameter bias change, false-positive covariate inclusion rate, incorrect NODE dispatch rate.

### Suite C: Real-World Expert Comparison (Pilot)

Blinded pilot head-to-head against 2–3 experienced pharmacometricians on 3–5 real clinical datasets. This is scoped as an initial pilot, not a powered external proof point. Expansion to larger expert panels and additional datasets is planned as the platform matures.

| Metric | Target |
|---|---|
| Top-model NLPD | APMODE ≤ 10% worse than best expert model |
| Structure agreement | APMODE selects same structural model as ≥ 2/3 experts in ≥ 60% of cases |
| Time-to-good-model | APMODE ≤ 4 hours wall-clock; experts report their actual time |
| Revision burden | After APMODE run, how many manual changes does an expert want to make? Target: ≤ 2 substantive changes |

---

## 6. Key Technical Risks & Mitigations

### R1: NODE models not accepted by regulators
**Mitigation:** Submission lane excludes NODE from "recommended." NODE findings inform classical model structure via interpretability bridge. NODE serves as discovery/optimization tool, not submitted model.

### R2: Agentic LLM proposes implausible models
**Mitigation:** DSL grammar + typed transforms + constraint enforcement. Agent cannot escape the grammar. Every proposal is validated before compilation.

### R3: Cross-paradigm comparison is not commensurate
**Mitigation:** Common observation model definition enforced where possible. When not possible (different BLQ handling, different latent spaces), comparison is flagged as qualified rather than absolute. Gated funnel prevents a model from "winning" on NLPD while failing hard validity checks.

### R4: Computational cost
**Mitigation:** Tiered execution. Quick-fit methods first (FOCE for NLME; fast NODE training with early stopping). Full estimation (SAEM, ensemble NODE) only for top candidates surviving Gate 1. Agentic iterations capped at 25. Target: full Submission lane run ≤ 4 hours on 32-core + 1 GPU; Discovery lane ≤ 8 hours.

### R5: Sparse data + over-flexible methods
**Mitigation:** Evidence manifest flags richness category. Sparse data + regulatory intent → NODE not dispatched (default policy, not just example). Sparse data + exploratory intent → NODE dispatched with explicit `data_insufficient` warning and results marked as hypothesis-generating only.

### R6: Hybrid NODE population modeling is a constrained approximation
The Phase 2 hybrid NODE backend uses random effects on NODE input-layer weights (per Bräm et al.), which is tractable but limited. This is **not** a full replacement for mature mixed-effects population modeling as practiced with NLME. The cited NODE PK literature supports the promise of structured NODEs for PK, but does not yet demonstrate a generally solved, industrial-grade mixed-effects NODE workflow. APMODE's hybrid NODE should be understood as a constrained approximation useful for discovering structural misspecification and informing classical model revision — not as a drop-in population-PK engine. This limitation must be communicated clearly in all Discovery and Optimization lane outputs that include NODE candidates.

### R7: "NLPD as common currency" masks pharmacometric failure modes
**Mitigation:** Gated funnel with explicit checks for ETA shrinkage, parameter identifiability, simulation faithfulness under dosing perturbation, BLQ robustness, and seed stability. NLPD is ranking criterion only within the surviving set after these gates.

---

## 7. Differentiators

| Capability | nlmixr2auto | AutoStan | Uni-PK | **APMODE** |
|---|---|---|---|---|
| Classical NLME | ✅ | ❌ | ❌ | ✅ |
| Automated structural search | ✅ | ❌ | ❌ | ✅ |
| LLM-driven proposals | ❌ | ✅ (general) | ❌ | ✅ (PK-specific, DSL-constrained) |
| Hybrid mechanistic-NODE | ❌ | ❌ | Partial | ✅ |
| Cross-paradigm gated ranking | ❌ | ❌ | ❌ | ✅ |
| Intent-specific operating lanes | ❌ | ❌ | ❌ | ✅ |
| Typed PK DSL + compiler | N/A | ❌ | N/A | ✅ |
| Functional distillation (NODE → PK interpretation) | N/A | N/A | Partial (SHAP) | ✅ |
| Reproducibility bundle | ❌ | ❌ | ❌ | ✅ |
| Credibility assessment framework | ❌ | ❌ | ❌ | ✅ |

---

## 8. MVP Scope

### Phase 1 (6 months): Platform Foundation

- Data Profiler → Evidence Manifest (rule-based + lightweight ML)
- **PK DSL + compiler + validator + audit trail** ← core platform artifact, built first
- Classical NLME Backend (nlmixr2)
- Automated Search Backend (structural + covariate)
- Governance Layer: Gate 1 (technical validity) + basic Gate 2 (submission lane only)
- Reproducibility bundle generation
- CLI interface
- Benchmark Suite A (synthetic recovery), initial 4 scenarios

### Phase 2 (4 months): Hybrid NODE + Discovery Lane

- Hybrid mechanistic-NODE backend (one architecture: Bräm-style low-dimensional)
- Interpretability bridge: functional distillation (learned sub-function visualization, parametric surrogate fitting, fidelity quantification)
- Discovery lane activation (Gate 2 rules for discovery)
- Cross-paradigm ranking (Gate 3 with NLPD + lane-specific criteria)
- Benchmark Suite A completion (all 7 scenarios) + Suite B (semi-synthetic)
- Web UI (basic)

### Phase 3 (4 months): Agentic LLM + Optimization Lane

- Agentic LLM backend operating through DSL transforms
- Optimization lane activation (leave-one-regimen-out evaluation)
- Report Generator with credibility assessment framework
- Benchmark Suite C (expert comparison, 2–3 datasets)
- API for programmatic access

### Research Branch (ongoing, not mainline product)

- Latent ODE (Maurel-style) for sparse precision dosing
- Cross-compound Uni-PK molecular-feature integration
- D-PINNs for aggregated literature data
- Full mixed-effects NODE inference

---

## 9. Success Criteria

| Metric | Target | Suite |
|---|---|---|
| Structural recovery rate | ≥ 80% on Suite A scenarios | A |
| Parameter bias | ≤ 15% for key PK parameters on Suite A | A |
| Parameter coverage | 95% CI contains truth ≥ 90% of the time | A |
| Symbolic surrogate fidelity | Matches true function within 20% (5th–95th state space), hybrid models only, scenarios A4/A7 | A |
| BLQ robustness | Rank stability when BLQ fraction increases to 25% | B |
| Covariate false positive rate | ≤ 10% for null covariates | B |
| Incorrect NODE dispatch under sparse perturbation | 0% when manifest correctly flags data_insufficient | B |
| Expert comparison NLPD | ≤ 10% worse than best expert model | C |
| Structure agreement with experts | ≥ 60% agreement | C |
| Time-to-model | ≤ 4 hours (Submission), ≤ 8 hours (Discovery) | C |
| Expert revision burden | ≤ 2 substantive changes | C |
| Reproducibility | Identical or tolerance-equivalent results under pinned infrastructure and seed registry | All |

---

## 10. Open Questions

1. **DSL extensibility:** The typed grammar is intentionally tight for v1. When and how do we add new module types? Proposal: new modules require a formal specification (ODE system, parameter constraints, expected identifiability requirements) reviewed by a pharmacometrician before inclusion.

2. **Common observation model for cross-paradigm NLPD:** Elevated to an architectural dependency for Phase 2 (see §4.3.1). The observation-model comparability protocol is a blocking deliverable before Discovery lane cross-paradigm ranking goes to production.

3. **Population-level NODE inference:** Full mixed-effects NODE (analogous to NLME with omega estimation) remains computationally hard. The Bräm low-dimensional approach uses random effects on NODE input weights, which is tractable but limited. This is a known ceiling for Phase 2.

4. **Regulatory engagement timing:** When in the development process should we seek informal FDA/EMA feedback on the credibility framework? Recommendation: after Phase 1 benchmark results are available, before Phase 2 NODE integration.

5. **Commercial packaging and licensing (architectural note, not just open question):** nlmixr2 is GPL-2. The DSL/compiler/audit system is the moat. These two facts require an early licensing architecture decision. Options: (a) open-core — DSL/governance/orchestrator are proprietary, nlmixr2 integration is open-source adapter; (b) SaaS — platform is hosted, nlmixr2 runs server-side, no distribution; (c) enterprise internal-use — source obligations handled via internal deployment. This decision affects build structure from Phase 1 and should not be deferred. NONMEM and Pumas remain optional adapters with separate licensing paths.

6. **Expert benchmark dataset access:** Suite C requires real clinical PK datasets with expert-built models as comparators. This requires data-sharing agreements (e.g., via BPCA, FDA public datasets, or pharma collaborations).

---

## 11. References

- Timonen J. "Modeling pharmacokinetic data: How to find the right model?" Generable Blog, April 2026.
- Dürr O et al. "AutoStan: Autonomous Bayesian Model Improvement via Predictive Feedback." arXiv:2603.27766, March 2026.
- Lu J et al. "Neural-ODE for pharmacokinetics modeling." iScience 24:102804, 2021.
- Bräm DS et al. "Low-dimensional neural ODEs and their application in pharmacokinetics." J Pharmacokinet Pharmacodyn 51:123–140, 2024.
- Giacometti L et al. "Leveraging Neural ODEs for Population Pharmacokinetics of Dalbavancin in Sparse Clinical Data." Entropy 27(6):602, 2025.
- Maurel B et al. "Latent Neural-ODE for Model-Informed Precision Dosing." arXiv:2602.03215, February 2026.
- Uni-PK. "Toward Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs." J Chem Inf Model 66(5):2640–2650, 2026.
- Losada R. "Bridging pharmacology and neural networks." CPT Pharmacometrics Syst Pharmacol, 2024.
- D-PINNs. "A physics-informed neural network approach for estimating population-level PK parameters." J Pharmacokinet Pharmacodyn, 2026.
- nlmixr2auto. UCL Pharmacometrics. github.com/ucl-pharmacometrics/nlmixr2auto.
- Chen et al. "Automatic tool for population PK model development." Diva Portal, 2024.
- ICH M15. "General Principles for Model-Informed Drug Development." 2024–2025 drafts.
- FDA. "Artificial Intelligence and Machine Learning in Drug Development." Guidance documents, 2023–2025.
