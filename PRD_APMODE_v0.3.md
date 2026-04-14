# PRD: Adaptive Pharmacokinetic Model Discovery Engine (APMODE)

**Version:** 0.3
**Date:** 2026-04-13
**Status:** RFC

---

## Revision History

| Version | Date | Changes |
|---|---|---|
| 0.2 | 2026-04-11 | Initial RFC per expert review |
| 0.3 | 2026-04-13 | Multi-model stress-test incorporation: cross-paradigm ranking redesign, LLM reproducibility hardening, Phase 0/1 re-scoping, Gate 2.5 credibility qualification, data ingestion contract, NODE constraint tightening, surrogate fidelity metric revision, Suite C metric revision, initial estimate strategy, IOV grammar, identifiability-based Submission filtering, licensing elevation |

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

**NODE/agentic models are not eligible for "recommended" status in this lane.** They may appear in supplementary analysis as hypothesis-generating evidence. This restriction reflects current regulatory acceptance, not a scientific quality judgment — all backends in this lane are additionally subject to practical identifiability requirements (see Gate 2, §4.3.1).

### 3.2 Discovery Lane

**Purpose:** Find the best structural description of the data, including novel structures the modeler might not have considered.

**Admissible backends:** Classical NLME + automated search + hybrid mechanistic-NODE + agentic LLM proposals.

**Stopping rules:** Diminishing improvement in primary ranking metric across iterations; user-defined iteration budget; stability of top-ranked candidates across re-runs.

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
│  │  Ingestion │   │ Manifest │   │  (by intent)  │         │
│  │  + Profiler│   │          │   │               │         │
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
│                    │  │ Gate 1:       │  │                  │
│                    │  │ Technical     │  │                  │
│                    │  │ Validity      │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │ Gate 2:       │  │                  │
│                    │  │ Lane-Specific │  │                  │
│                    │  │ Admissibility │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │ Gate 2.5:     │  │                  │
│                    │  │ Credibility   │  │                  │
│                    │  │ Qualification │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │ Gate 3:       │  │                  │
│                    │  │ Within-Gate   │  │                  │
│                    │  │ Ranking       │  │                  │
│                    │  └───────┬───────┘  │                  │
│                    │  ┌───────▼───────┐  │                  │
│                    │  │ Provenance &  │  │                  │
│                    │  │ Reproducib.   │  │                  │
│                    │  └───────────────┘  │                  │
│                    └─────────┬──────────┘                  │
│                              │                             │
│                    ┌─────────▼──────────┐                  │
│                    │  Report Generator  │                  │
│                    └────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

#### 4.2.0 Data Ingestion Contract

APMODE accepts PK datasets in the following formats, converted on ingestion to a canonical internal representation:

**Accepted input formats:**

| Format | Convention | Notes |
|---|---|---|
| NONMEM-style CSV | EVID/MDV/CMT/AMT/DV columns | Most common in industry |
| nlmixr2 event table | nlmixr2 `eventTable()` or compatible data frame | Native R integration |
| CDISC ADaM (ADPC/ADNCA) | AVAL, ATPT, DTYPE, PARAMCD conventions | Submission-ready source |

**Canonical internal representation:** All ingested data are normalized to a typed internal schema with explicit fields for:

- Subject identifier, time, observation value, observation type (parent, metabolite)
- Dosing records: amount, route, rate/duration, compartment, infusion/bolus flag
- BLQ coding: numeric flag + LLOQ value (not overloaded on DV)
- Occasion identifiers (for IOV)
- Covariate values with time-variance flag (time-constant vs. time-varying)
- Study/protocol identifier (for pooled analyses)

**Validation on ingestion:** Column mapping verification, duplicate record detection, dosing-observation temporal consistency, covariate completeness assessment (feeds missingness_pattern in Evidence Manifest).

#### 4.2.0.1 Initial Estimate Strategy

Systematic initial estimate derivation is required before estimation dispatch. Poor initial estimates are a primary cause of SAEM non-convergence and NODE training failure.

**Classical backends:** Non-compartmental analysis (NCA) on per-subject data to derive initial CL, V, ka, t½ estimates. For multi-compartment models, NCA-derived macro-constants (A, B, α, β) inform micro-constant initialization. When NCA is infeasible (sparse data), population-level NCA on pooled naive-averaged profiles provides fallback estimates.

**NODE backends:** Pre-trained weight initialization from a library of reference PK dynamics (e.g., standard 1-cmt/2-cmt absorption/elimination patterns). Transfer learning from the classical backend's best-fit solution where available.

**Automated search:** Each candidate model in the search receives initial estimates derived from the best-fit parameter values of its parent model in the search DAG (warm-starting), falling back to NCA-derived estimates for the root candidates.

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
| `covariate_missingness` | struct: {pattern: MCAR/MAR/informative-suspected, fraction_incomplete: float, strategy: impute-median/impute-LOCF/full-information/exclude} | {MAR, 0.08, impute-median} |
| `blq_burden` | fraction of BLQ observations | 0.15 |
| `protocol_heterogeneity` | enum: single-study / pooled-similar / pooled-heterogeneous | pooled-similar |
| `temporal_support` | struct: {absorption_phase_coverage, elimination_phase_coverage} | {adequate, adequate} |

**Dispatch constraints derived from manifest:**

- If `richness_category` = sparse AND `temporal_support.absorption_phase_coverage` = inadequate → NODE backends receive a `data_insufficient` flag and are not dispatched (or are dispatched with explicit low-confidence warning).
- If `nonlinear_clearance_signature` = true → automated search includes MM and parallel elimination candidates.
- If `blq_burden` > 0.20 → all backends must use BLQ-aware likelihood (M3/M4 methods or equivalent).
- If `protocol_heterogeneity` = pooled-heterogeneous → IOV must be tested in classical backends.
- If `covariate_missingness.fraction_incomplete` > 0.15 → covariate search must use full-information likelihood or multiple imputation; median/LOCF imputation is flagged as provisional.

#### 4.2.2 Classical NLME Backend

**Primary engine:** nlmixr2 (R), accessed via **process-isolated interface** (subprocess or local REST). Process isolation provides backend modularity (clean adapter boundary for NONMEM/Pumas) and crash isolation (R segfault does not take down the orchestrator). NONMEM and Pumas are optional adapters with separate licensing paths, not distributed with APMODE.

Accepts model specification via the typed PK DSL (§4.2.5). Runs SAEM and/or FOCEI estimation. Returns standardized result object:

- Parameter estimates with RSE/CI
- ETA shrinkage per parameter
- OFV, AIC, BIC
- GOF diagnostics (observed vs. predicted, CWRES vs. time/pred, QQ)
- VPC (prediction-corrected where appropriate)
- Condition number / practical identifiability flags (profile-likelihood CI for key parameters)
- Convergence metadata (iterations, gradient norms, minimization status)

#### 4.2.3 Automated Search Backend

Builds on nlmixr2auto-style logic. Search dimensions:

- **Structural:** 1-cmt ↔ 2-cmt ↔ 3-cmt × absorption variants × elimination variants (all expressed as DSL module combinations)
- **Covariate:** stepwise (SCM forward/backward), or LASSO-on-ETAs for screening
- **Random effects:** diagonal vs. block omega; IIV and IOV candidates
- **Residual error:** additive, proportional, combined

Scoring: AIC/BIC for nested comparisons within the same backend. Cross-validated predictive metrics for non-nested comparisons (see §4.3.1 Gate 3). Emits Pareto frontier (parsimony vs. fit).

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

**Known limitations of the Bräm-style RE parameterization:**

The random effects on NODE input-layer weights create a mathematically valid mixed-effects model but differ fundamentally from classical NLME random effects on physiological parameters (e.g., log-normal IIV on CL, V). The NODE RE parameterization introduces per-subject perturbations in a latent computational space rather than on interpretable biological parameters. This means:

1. Individual parameter estimates from this architecture **should not be used for individual dose adjustment** without additional validation.
2. Prediction intervals must be validated against nonparametric bootstrap before VPC generation. VPCs from this architecture are flagged as "bootstrap-validated" or "unvalidated" in the reproducibility bundle.
3. The architecture is suitable for detecting structural misspecification and informing classical model revision — not as a drop-in population-PK engine for dose individualization.

This limitation must be communicated clearly in all Discovery and Optimization lane outputs that include NODE candidates.

**Interpretability outputs (functional distillation, not SHAP-only):**

- Learned sub-function visualization: plot the NODE-learned clearance law over the observed concentration range; plot learned absorption kernel vs. time
- Parametric surrogate fitting: fit symbolic or parametric surrogates (MM, parallel pathways, transit chain) to the learned sub-function
- Surrogate fidelity quantification: report where the parametric surrogate matches the NODE and where it diverges, using integrated-exposure bioequivalence criteria (see §5 Suite A)
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
  | NODE_Absorption(dim, constraint_template)   # Discovery/Optimization lanes only

DistributionModule :=
  | OneCmt(V)
  | TwoCmt(V1, V2, Q)
  | ThreeCmt(V1, V2, V3, Q2, Q3)
  | TMDD_Core(R0, kon, koff, kint)              # with quasi-steady-state option

EliminationModule :=
  | Linear(CL)
  | MichaelisMenten(Vmax, Km)
  | ParallelLinearMM(CL, Vmax, Km)
  | TimeVarying(CL, decay_fn)
  | NODE_Elimination(dim, constraint_template)  # Discovery/Optimization lanes only

VariabilityModule :=
  | IIV(params: list, structure: diagonal | block)
  | IOV(params: list, occasions: occasion_spec)
  | CovariateLink(param, covariate, functional_form)

OccasionSpec :=
  | ByStudy                                     # one occasion per study
  | ByVisit(visit_column)                       # one occasion per visit
  | ByDoseEpoch(epoch_column)                   # one occasion per dosing epoch
  | Custom(occasion_column)                     # user-defined

ObservationModule :=
  | Proportional(sigma_prop)
  | Additive(sigma_add)
  | Combined(sigma_prop, sigma_add)
  | BLQ_M3(loq_value)
  | BLQ_M4(loq_value)
```

**NODE module constraints:**

NODE modules (`NODE_Absorption`, `NODE_Elimination`) use **enumerated constraint templates** rather than free-form constraint specifications. This prevents semantic overreach through overly permissive constraint definitions.

| Template | Description | Max dim |
|---|---|---|
| `monotone_increasing` | Output monotonically increasing in primary input | 4 |
| `monotone_decreasing` | Output monotonically decreasing in primary input | 4 |
| `bounded_positive` | Output ∈ (0, upper_bound) | 6 |
| `saturable` | Output approaches asymptote (MM-like shape class) | 4 |
| `unconstrained_smooth` | L2-regularized, no shape constraint | 8 |

**Dimension ceilings by lane:**

| Lane | Max NODE dim | Rationale |
|---|---|---|
| Submission | N/A (NODE not admissible) | — |
| Discovery | ≤ 8 | Sufficient for structural exploration |
| Optimization | ≤ 4 | Parsimony for dose extrapolation |

**Allowed agent transforms (for agentic backend):**

- `swap_module(position, new_module)` — e.g., swap Elimination from Linear to MichaelisMenten
- `add_covariate_link(param, covariate, form)` — e.g., allometric weight on CL
- `adjust_variability(param, action: add | remove | upgrade_to_block)`
- `set_transit_n(n)` — change transit compartment count
- `toggle_lag(on | off)`
- `replace_submodel_with_NODE(position, constraint_template, dim)` — Discovery lane only; dim must be ≤ lane ceiling; constraint_template must be from the enumerated set

Each transform produces a new model spec that is: (a) validated against the grammar, (b) compiled to backend-specific code, (c) logged in the audit trail with before/after specs.

**Constraints enforced at DSL level:**

- State and flow constraints consistent with the selected model family
- Elimination flows ≥ 0
- Volume parameters > 0
- Rate constants > 0
- Covariate functional forms must be from an approved set (power, exponential, linear, categorical, maturation)
- NODE dim ≤ lane-specific ceiling
- NODE constraint_template must be from the enumerated template set

**Compilation targets:**

Phase 1: nlmixr2 only. Phase 2+: Stan codegen with a per-backend lowering test suite verifying semantic equivalence for parameter constraints, BLQ handling, ODE solver tolerances, and random effect parameterization. The DSL→Stan compiler is not a syntactic translation — it requires validated lowering rules for each module type.

#### 4.2.6 Agentic LLM Backend (Phase 3)

Operates exclusively through DSL transforms (§4.2.5). Cannot write raw code.

**Inputs:** Evidence manifest, current best model(s), their residual diagnostics (CWRES patterns, VPC deficiencies), the search history so far.

**Behavior:** Proposes a sequence of transforms. Each transform is validated, compiled, fit, and evaluated. The agent observes results and proposes the next transform or stops.

**Iteration budget:** Capped at 25 rounds per run (consistent with AutoStan's demonstrated sufficiency at ~20).

**Reproducibility requirements for agentic runs:**

- LLM inference must use `temperature=0` and deterministic decoding where supported by the provider.
- All LLM inputs (system prompt, user messages, tool calls) and outputs (full response text, token-level metadata) are cached verbatim in the reproducibility bundle under `agentic_trace/`.
- The bundle records: LLM provider, model ID, model version (if exposed by provider API), API endpoint, request timestamp, and a SHA-256 hash of the full request payload.
- **Model-version escrow:** If the provider does not expose a deterministic model version identifier, the run is flagged as `agentic_reproducibility: best-effort` in the bundle. Full reproducibility requires replaying from cached outputs, not re-executing LLM inference.

**Multi-run provenance:** When a user executes multiple agentic runs (different seeds, configurations, or iterative refinement), each run's bundle includes a `run_lineage` field linking to prior run IDs. The governance layer tracks cross-run candidate selection to prevent undocumented cherry-picking. All runs contributing to the final candidate set must appear in the provenance record.

**What the agent adds over automated search:** It can reason about *why* a model is misfitting (e.g., "CWRES show time-dependent bias in the elimination phase → try MM elimination") rather than brute-force enumerating. It can also propose compound transforms (swap elimination + add covariate link simultaneously) that search algorithms handle less naturally.

**What the agent cannot do:** Write arbitrary ODE code, propose structures outside the DSL grammar, override admissibility gates, access held-out data, or specify NODE modules with constraint templates or dimensions outside the enumerated set.

### 4.3 Governance Layer

This is the architectural core of the system's credibility. Four gates plus provenance.

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
| Practical identifiability (profile-likelihood CI, condition number) | Required for all key params, all backends | Informational | Required for dose-related params |
| NODE-only models eligible for "recommended" | No | Yes | Yes (with uncertainty quantification) |
| Leave-one-regimen-out performance | Not required | Not required | Required |

**Gate 2.5: Credibility Qualification**

This gate implements ICH M15's requirement for context-of-use specification and risk-based model assessment. It is a **qualifying gate**, not documentation — a model that cannot satisfy these requirements is excluded from ranking.

| Check | Submission | Discovery | Optimization |
|---|---|---|---|
| Context-of-use statement | Required: explicit decision supported, risk level of model failure | Required: discovery intent documented | Required: dosing decision supported, patient population |
| Limitation-to-risk mapping | Required: each known limitation mapped to impact on the supported decision | Not required | Required: extrapolation boundaries mapped to dosing risk |
| Data adequacy vs. model complexity | Required: evidence manifest fields matched against model parameter count and structural complexity | Informational | Required |
| Sensitivity analysis | Required: key parameters perturbed, decision stability assessed | Not required | Required: dosing recommendation stability under parameter perturbation |
| AI/ML transparency (if NODE or agentic backend involved) | N/A (not admissible) | Required: what role did ML play, what guardrails applied, how validated against mechanistic expectations | Required: same as Discovery |

**Gate 3: Within-Gate Ranking**

Only models surviving Gates 1, 2, and 2.5 enter ranking.

**Within-paradigm ranking** (models sharing the same observation model and estimation framework): AIC/BIC for nested comparisons; cross-validated NLPD for non-nested comparisons. NLPD is valid as a ranking metric when the observation model is identical across candidates.

**Cross-paradigm ranking** (models from different backends with potentially different observation models, BLQ handling, or residual error structures): NLPD is **not used** as the primary ranking metric. Instead, cross-paradigm ranking uses simulation-based predictive metrics:

| Metric | Description | Primary for lane |
|---|---|---|
| VPC coverage concordance | Fraction of time bins where simulated 5th/50th/95th percentiles contain observed data within pre-specified coverage bounds | Submission, Discovery |
| AUC/Cmax bioequivalence | Simulated individual AUC and Cmax within 80–125% geometric mean ratio relative to observed, across validation subjects | Optimization |
| Nonparametric prediction error (NPE) | Median absolute prediction error on held-out subjects, computed from posterior predictive simulations (no distributional assumptions) | Discovery, Optimization |

When cross-paradigm ranking is performed, the report explicitly states which backends were compared, which metrics were used, and why NLPD was not applicable. Cross-paradigm rankings are presented as **qualified comparisons**, not absolute orderings.

Ranking priorities per lane (applied in order within the surviving set):

*Submission lane:* (1) Diagnostic adequacy (GOF, VPC), (2) parameter precision and identifiability, (3) parsimonious structure, (4) information criteria (AIC/BIC) within the admissible set.

*Discovery lane:* (1) Primary predictive metric (VPC coverage concordance for within-paradigm; NPE for cross-paradigm), (2) structural novelty relative to classical baseline, (3) residual pattern improvement, (4) interpretability score (if available).

*Optimization lane:* (1) Leave-one-regimen-out predictive metric (AUC/Cmax bioequivalence), (2) simulation faithfulness under dosing perturbation, (3) uncertainty calibration, (4) within-paradigm NLPD on in-distribution holdout.

#### 4.3.2 Provenance and Reproducibility Contract

Every run produces a machine-readable **reproducibility bundle**:

| Artifact | Contents |
|---|---|
| `data_manifest.json` | SHA-256 hash of input dataset; column mapping; BLQ coding; ingestion format |
| `split_manifest.json` | Subject-level train/test/validation assignments per fold; random seed |
| `backend_versions.json` | nlmixr2 version, R version, Python/JAX version (for NODE), LLM model ID + version + prompt hash (for agentic) |
| `seed_registry.json` | All random seeds used across all backends |
| `search_trajectory.jsonl` | Ordered log of every model attempted: DSL spec, backend, fit status, scores, gate outcomes |
| `failed_candidates.jsonl` | Models that failed gates, with specific failure reasons |
| `candidate_lineage.json` | DAG of model derivation (which model was parent of which) |
| `compiled_specs/` | Directory of compiled model code for every candidate |
| `evidence_manifest.json` | Data profiler output |
| `initial_estimates.json` | NCA-derived or warm-start initial estimates per candidate |
| `agentic_trace/` | (Phase 3) Verbatim LLM inputs/outputs, request metadata, temperature settings |
| `run_lineage.json` | (Phase 3) Links to prior run IDs when multiple runs contribute to final candidate set |
| `report_provenance.json` | Timestamps, component versions, who/what generated each section |
| `policy_file.json` | Versioned gate threshold policy used for this run |

This bundle is the unit of reproducibility. For classical and automated backends, any APMODE result can be fully re-executed from this bundle on matching infrastructure. For agentic backends, reproducibility is achieved by replaying from cached LLM outputs in `agentic_trace/`; re-executing LLM inference may produce different proposals and is flagged accordingly.

#### 4.3.3 Credibility Assessment Reporting

Aligned with ICH M15 and FDA AI/ML guidance. Gate 2.5 enforces credibility qualification as a gate; this section specifies the **reporting format** for surviving candidates.

For each recommended model, the report includes:

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

**Symbolic surrogate fidelity target (scoped):** For hybrid NODE models in scenarios A4 and A7 only, test whether parametric surrogates fitted to the NODE sub-function produce pharmacokinetically equivalent systemic exposure. Target: surrogate-predicted AUC and Cmax within 80–125% geometric mean ratio of NODE-predicted AUC and Cmax, evaluated across the 5th–95th percentile of the simulated covariate and dose range. This applies to selected hybrid model classes, not all NODE dynamics.

### Suite B: Semi-Synthetic Perturbation

Start from real clinical datasets. Inject known perturbations to test robustness.

| Perturbation | Purpose |
|---|---|
| Increase BLQ fraction to 25%, 40% | BLQ robustness |
| Add 5% gross outliers | Outlier sensitivity |
| Remove 50% of absorption-phase samples | Sparse data degradation |
| Pool two studies with different protocols | Protocol heterogeneity handling |
| Add correlated covariates with no true effect | Covariate false-positive rate |
| Reduce to sparse sampling (< 4/subject) in Discovery lane | Rate of incorrect escalation to hybrid NODE under sparse-data perturbation |
| Inject covariate missingness at 10%, 20%, 30% | Covariate imputation strategy robustness |

**Metrics:** Model rank stability, parameter bias change, false-positive covariate inclusion rate, incorrect NODE dispatch rate, diagnostic-observation information leakage (compare agentic proposals across independent seed runs for systematic bias).

### Suite C: Real-World Expert Comparison (Pilot)

Blinded pilot head-to-head against 2–3 experienced pharmacometricians on 3–5 real clinical datasets. This is scoped as an initial pilot, not a powered external proof point. Expansion to larger expert panels and additional datasets is planned as the platform matures.

| Metric | Primary/Secondary | Target |
|---|---|---|
| Fraction of datasets where APMODE top model outperforms median expert model (by NPE) | **Primary** | ≥ 60% |
| Top-model NLPD gap vs. best expert | Secondary | ≤ 10% worse |
| Structure agreement | Secondary | APMODE selects same structural model as ≥ 2/3 experts in ≥ 60% of cases |
| Time-to-good-model | Secondary | APMODE ≤ 4 hours wall-clock; experts report their actual time |
| Revision burden | Secondary | After APMODE run, how many manual changes does an expert want to make? Target: ≤ 2 substantive changes |

---

## 6. Key Technical Risks & Mitigations

### R1: NODE models not accepted by regulators
**Mitigation:** Submission lane excludes NODE from "recommended." NODE findings inform classical model structure via interpretability bridge. NODE serves as discovery/optimization tool, not submitted model.

### R2: Agentic LLM proposes implausible models
**Mitigation:** DSL grammar + typed transforms + enumerated constraint templates + dim ceilings per lane. Agent cannot escape the grammar or specify arbitrary NODE architectures. Every proposal is validated before compilation.

### R3: Cross-paradigm comparison is not commensurate
**Mitigation:** Cross-paradigm ranking uses simulation-based predictive metrics (VPC coverage concordance, AUC/Cmax bioequivalence, NPE), not NLPD. Within-paradigm ranking may use NLPD where observation models match. Cross-paradigm rankings are explicitly flagged as qualified comparisons.

### R4: Computational cost
**Mitigation:** Tiered execution. Quick-fit methods first (FOCE for NLME; fast NODE training with early stopping). Full estimation (SAEM, ensemble NODE) only for top candidates surviving Gate 1. Agentic iterations capped at 25. Target: full Submission lane run ≤ 4 hours on 32-core + 1 GPU; Discovery lane ≤ 8 hours. Candidate count budget per lane is a policy parameter (default: 50 Submission, 100 Discovery, 30 Optimization) to bound wall-clock time.

### R5: Sparse data + over-flexible methods
**Mitigation:** Evidence manifest flags richness category. Sparse data + regulatory intent → NODE not dispatched (default policy, not just example). Sparse data + exploratory intent → NODE dispatched with explicit `data_insufficient` warning and results marked as hypothesis-generating only.

### R6: Hybrid NODE population modeling is a constrained approximation
The Phase 2 hybrid NODE backend uses random effects on NODE input-layer weights (per Bräm et al.), which is tractable but limited. This is **not** a full replacement for mature mixed-effects population modeling as practiced with NLME. The Bräm RE parameterization captures inter-individual variability but in a latent computational space rather than on physiological parameters; prediction intervals require bootstrap validation, and individual estimates should not drive dose adjustment. APMODE's hybrid NODE should be understood as a constrained approximation useful for discovering structural misspecification and informing classical model revision — not as a drop-in population-PK engine.

### R7: "NLPD as common currency" masks pharmacometric failure modes
**Mitigation:** Gated funnel with explicit checks for ETA shrinkage, practical identifiability (profile-likelihood CI), simulation faithfulness under dosing perturbation, BLQ robustness, and seed stability. Cross-paradigm ranking uses simulation-based metrics, not NLPD. NLPD is ranking criterion only within-paradigm, within the surviving set after gates.

### R8: Diagnostic-side information leakage in agentic iterations
**Risk:** The agentic backend observes CWRES/VPC patterns across up to 25 iterations × 3 seeds, providing ~75 diagnostic snapshots. While these are aggregated summaries (not individual-level held-out labels), repeated fit-evaluate cycles could enable indirect adaptive overfitting to the training data's distributional properties.
**Mitigation:** Gate 1 seed-stability check (top model must be consistent across ≥ 3 independent seeds). Suite B benchmarking will explicitly test for systematic bias in agentic proposals across independent seed runs. If detected, additional mitigations include diagnostic subsampling (presenting aggregated diagnostics from a random subset of training subjects per iteration) or iteration-budget tightening.

### R9: LLM provider model versioning
**Risk:** LLM providers may silently version, update, or retire models, making agentic run reproduction impossible via re-inference.
**Mitigation:** Reproducibility bundle caches verbatim LLM outputs. Re-execution replays from cache. Runs where the provider does not expose a deterministic version identifier are flagged as `agentic_reproducibility: best-effort`.

---

## 7. Differentiators

| Capability | nlmixr2auto | AutoStan | Uni-PK | **APMODE** |
|---|---|---|---|---|
| Classical NLME | ✅ | ❌ | ❌ | ✅ |
| Automated structural search | ✅ | ❌ | ❌ | ✅ |
| LLM-driven proposals | ❌ | ✅ (general) | ❌ | ✅ (PK-specific, DSL-constrained) |
| Hybrid mechanistic-NODE | ❌ | ❌ | Partial | ✅ |
| Cross-paradigm gated ranking | ❌ | ❌ | ❌ | ✅ (simulation-based) |
| Intent-specific operating lanes | ❌ | ❌ | ❌ | ✅ |
| Typed PK DSL + compiler | N/A | ❌ | N/A | ✅ |
| Functional distillation (NODE → PK interpretation) | N/A | N/A | Partial (SHAP) | ✅ (surrogate fidelity via AUC/Cmax BE) |
| Reproducibility bundle (with agentic trace) | ❌ | ❌ | ❌ | ✅ |
| Credibility assessment framework (ICH M15 gate) | ❌ | ❌ | ❌ | ✅ |

---

## 8. MVP Scope

### Phase 0 (2 weeks): Architectural Foundations

- **Licensing: resolved.** APMODE is open source under GPL-2, consistent with nlmixr2. Process isolation (subprocess/REST) is retained as an architectural choice for backend modularity and crash isolation — not for licensing reasons. This enables clean adapter boundaries for optional proprietary backends (NONMEM, Pumas) which maintain separate licensing paths.
- **Data format canonical schema:** Finalize internal representation (§4.2.0).
- **Gate threshold policy file format:** Define versioned policy schema.

### Phase 1 (6 months): Platform Foundation

- Data Ingestion + Data Profiler → Evidence Manifest (rule-based + lightweight ML)
- **PK DSL + compiler (nlmixr2 target only) + validator + audit trail** ← core platform artifact, built first
- Initial estimate strategy (NCA-based, §4.2.0.1)
- Classical NLME Backend (nlmixr2, process-isolated)
- Automated Search Backend (structural + covariate; bounded to 2 structural families initially, SCM forward/backward)
- Governance Layer: Gate 1 (technical validity) + Gate 2 (submission lane, including practical identifiability)
- Reproducibility bundle generation
- CLI interface
- Benchmark Suite A (synthetic recovery), initial 4 scenarios (A1–A4)

### Phase 2 (4 months): Hybrid NODE + Discovery Lane

- Hybrid mechanistic-NODE backend (one architecture: Bräm-style low-dimensional, enumerated constraint templates)
- Interpretability bridge: functional distillation (learned sub-function visualization, parametric surrogate fitting, AUC/Cmax bioequivalence fidelity quantification)
- DSL → Stan codegen with per-backend lowering test suite
- Discovery lane activation (Gate 2 rules for discovery, Gate 2.5 credibility qualification)
- Cross-paradigm ranking using simulation-based metrics (Gate 3 with VPC coverage concordance, NPE)
- Benchmark Suite A completion (all 7 scenarios) + Suite B (semi-synthetic, including diagnostic leakage monitoring)
- Web UI (basic)

### Phase 3 (4 months): Agentic LLM + Optimization Lane

- Agentic LLM backend operating through DSL transforms (enumerated NODE constraints, dim ceilings, temperature=0, verbatim output caching)
- Multi-run provenance tracking
- Optimization lane activation (leave-one-regimen-out evaluation, AUC/Cmax bioequivalence ranking)
- Report Generator with credibility assessment framework (Gate 2.5 reporting)
- Benchmark Suite C (expert comparison, 2–3 datasets, primary metric: fraction-beats-median-expert)
- API for programmatic access

### Research Branch (ongoing, not mainline product)

- Latent ODE (Maurel-style) for sparse precision dosing
- Cross-compound Uni-PK molecular-feature integration
- D-PINNs for aggregated literature data
- Full mixed-effects NODE inference
- Alternative RE parameterizations for NODE (output-space RE, hierarchical latent NODE)

---

## 9. Success Criteria

| Metric | Target | Suite |
|---|---|---|
| Structural recovery rate | ≥ 80% on Suite A scenarios | A |
| Parameter bias | ≤ 15% for key PK parameters on Suite A | A |
| Parameter coverage | 95% CI contains truth ≥ 90% of the time | A |
| Symbolic surrogate fidelity | Surrogate AUC/Cmax within 80–125% GMR of NODE AUC/Cmax (hybrid models, A4/A7) | A |
| BLQ robustness | Rank stability when BLQ fraction increases to 25% | B |
| Covariate false positive rate | ≤ 10% for null covariates | B |
| Incorrect NODE dispatch under sparse perturbation | 0% when manifest correctly flags data_insufficient | B |
| Agentic diagnostic leakage | No systematic bias in agentic proposals across independent seed runs | B |
| Fraction of datasets where APMODE beats median expert (NPE) | ≥ 60% | C |
| Top-model NLPD gap vs. best expert | ≤ 10% worse (secondary) | C |
| Structure agreement with experts | ≥ 60% agreement | C |
| Time-to-model | ≤ 4 hours (Submission), ≤ 8 hours (Discovery) | C |
| Expert revision burden | ≤ 2 substantive changes | C |
| Reproducibility | Identical or tolerance-equivalent results under pinned infrastructure and seed registry (classical/automated); cache-replay equivalent (agentic) | All |

---

## 10. Open Questions

1. **DSL extensibility:** The typed grammar is intentionally tight for v1. When and how do we add new module types? Proposal: new modules require a formal specification (ODE system, parameter constraints, expected identifiability requirements) reviewed by a pharmacometrician before inclusion. New NODE constraint templates require benchmark validation on Suite A scenarios before addition.

2. **Common observation model for within-paradigm NLPD:** Within a single backend paradigm, NLPD remains a valid ranking metric. The cross-paradigm case is handled by simulation-based metrics (§4.3.1). However, within-paradigm comparisons still require careful observation-model specification when different BLQ handling methods are used.

3. **Population-level NODE inference:** Full mixed-effects NODE (analogous to NLME with omega estimation) remains computationally hard. The Bräm low-dimensional approach uses random effects on NODE input weights, which is tractable but limited in physiological interpretability. Alternative RE parameterizations (output-space RE, hierarchical latent NODE) are research branch topics.

4. **Regulatory engagement timing:** When in the development process should we seek informal FDA/EMA feedback on the credibility framework? Recommendation: after Phase 1 benchmark results are available, before Phase 2 NODE integration.

5. **Licensing: resolved.** APMODE is open source under GPL-2, consistent with nlmixr2. Process isolation is retained for backend modularity and crash isolation, not licensing. NONMEM and Pumas adapters maintain separate licensing paths via the subprocess boundary and are not distributed as part of the APMODE open-source release.

6. **Expert benchmark dataset access:** Suite C requires real clinical PK datasets with expert-built models as comparators. This requires data-sharing agreements (e.g., via BPCA, FDA public datasets, or pharma collaborations).

7. **Covariate missingness strategy:** The evidence manifest captures missingness patterns, and dispatch constraints trigger full-information or multiple imputation when missingness exceeds 15%. However, the interaction between covariate missingness handling and the automated covariate search (SCM, LASSO-on-ETAs) requires specification: should imputed covariates be flagged in the search, and should the covariate selection criterion account for imputation uncertainty?

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
