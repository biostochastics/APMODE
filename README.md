<div align="center">

  <img src="apmode_logo.png" alt="APMODE Logo" width="480">

  # APMODE

  **Adaptive Pharmacokinetic Model Discovery Engine**

  [![Phase](https://img.shields.io/badge/phase-3%20(P3.B%20%E2%80%94%20rc5)-blue)]()

  [![Tests](https://img.shields.io/badge/tests-1558%20passing-success)]()
  [![License](https://img.shields.io/badge/license-GPL--2.0--or--later-green)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.12%E2%80%933.14-yellow)]()
  [![mypy](https://img.shields.io/badge/mypy-strict%20%E2%9C%93-blue)]()
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/biostochastics/APMODE)

</div>

---

## What is APMODE?

APMODE is a **governed meta-system** that composes five population PK modeling paradigms into a single discovery workflow — so pharmacometricians can evaluate classical, automated, neural, Bayesian, and hybrid approaches under one roof with consistent evidence standards.

**Five paradigms, one pipeline.** Classical NLME (nlmixr2), automated structural search, hybrid mechanistic-NODE (JAX/Diffrax), agentic LLM model construction (Phase 3), and **Bayesian PK (Stan/Torsten via cmdstanpy, Phase 2+)** — each dispatched, evaluated, and ranked through a unified governance funnel.

**Evidence gates.** Every candidate passes Gate 1 (technical validity), Gate 2 (lane admissibility), Gate 2.5 (ICH M15 credibility), and Gate 3 (cross-paradigm ranking) before it can be recommended. The Bayesian backend adds MCMC-specific Gate 1 thresholds (R̂ ≤ 1.01, ESS ≥ 400, divergences = 0, E-BFMI ≥ 0.3, Pareto-k ≤ 0.7) and a Gate 2 prior-justification artifact aligned with FDA's Jan 2026 draft Bayesian methodology guidance (FDA-2025-D-3217). NODE models are never eligible for regulatory submission — this is a hard rule, not a tunable weight.

**Reproducibility is the unit of output.** Every run emits a versioned JSON bundle — data manifest, search trajectory, gate decisions, candidate lineage DAG, compiled specs — so any result can be audited or replayed.

**Formular — a typed PK DSL — is the control surface.** Models are specified in [Formular](docs/FORMULAR.md), a structured grammar (`Absorption x Distribution x Elimination x Variability x Observation × Priors`), compiled to a typed AST, validated against pharmacometric constraints, and lowered to backend-specific code (nlmixr2 R, Stan/Torsten, JAX/Diffrax). The agentic LLM backend (Phase 3) operates exclusively through Formular transforms — including the new `SetPrior` transform for Bayesian workflows — it cannot emit raw code.

> **Status**: **v0.3.0-rc5** (2026-04-16) — Phase 3 release candidate 5. Comprehensive CLI overhaul: new `apmode doctor`, `apmode ls`, `apmode policies` commands; `apmode report` now opens existing artifacts; `--dry-run` dispatch preview; colorblind-safe status indicators throughout; RSE%/η-shrinkage in parameter tables; policy version in run header; agentic-on-submission lane note. 10/10 benchmark scenarios produce a recommended candidate. 1541 tests passing (non-live); `mypy --strict` clean; `ruff` clean. Supports Python 3.12–3.14.

---

## Quick Start

### Prerequisites

- **Python**: 3.12, 3.13, or 3.14
- **Package manager**: [uv](https://docs.astral.sh/uv/)
- **Optional**: R 4.4+ with `nlmixr2`, `rxode2`, `jsonlite`, `lotri` for classical NLME (mock R subprocess tests work without R)
- **Optional (Bayesian)**: CmdStan 2.36+ via `cmdstanpy.install_cmdstan()` for the Stan/Torsten backend

### Installation

```bash
git clone https://github.com/biostochastics/apmode.git
cd apmode

# Install all dependencies
uv sync --all-extras

# Verify
uv run apmode --help
```

### Explore a Public Dataset (Interactive)

```bash
# Browse 14 available PK datasets (real clinical + simulated ground-truth)
uv run apmode datasets

# Interactive exploration: fetch → profile → NCA → search space preview
uv run apmode explore theo_sd
uv run apmode explore mavoglurant --lane discovery

# Non-interactive (CI): runs full pipeline automatically
uv run apmode explore Oral_1CPTMM -y -o ./runs
```

### Run the Full Pipeline

```bash
# Run on your own NONMEM-format CSV (classical SAEM, default)
uv run apmode run <dataset.csv> --lane submission

# Download a dataset first, then run
uv run apmode datasets theo_sd -o ./data
uv run apmode run ./data/theo_sd.csv --lane discovery

# Bayesian backend (requires bayesian extras + CmdStan)
uv sync --extra bayesian
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
uv run apmode run ./data/theo_sd.csv --lane discovery \
    --backend bayesian_stan --bayes-chains 4 --bayes-warmup 1000 --bayes-sampling 1000
```

### Environment Check

```bash
# Verify R, nlmixr2, rxode2, CmdStan, Python packages, and LLM provider keys
uv run apmode doctor
```

### List and Preview Runs

```bash
# List all bundles in ./runs (most recent first)
uv run apmode ls

# Sort by BIC, show top 10
uv run apmode ls --sort bic --limit 10

# Preview a run without executing R backends
uv run apmode run <dataset.csv> --lane submission --dry-run
```

### Inspect Results

```bash
# Bundle summary
uv run apmode inspect <bundle_dir>

# Top 3 candidates with parameter estimates, RSE%, η-shrinkage
uv run apmode log <bundle_dir> --top 3

# Gate failure analysis
uv run apmode log <bundle_dir> --failed
uv run apmode log <bundle_dir> --gate gate1

# View the run report (HTML in browser, or Markdown in pager)
uv run apmode report <bundle_dir>
uv run apmode report <bundle_dir> --format md

# Compare two runs side-by-side
uv run apmode diff ./runs/run_a ./runs/run_b

# Deep inspection (Phase 3)
uv run apmode trace <bundle_dir>                         # agentic iteration summary
uv run apmode trace <bundle_dir> --iteration 5           # detail for iteration 5
uv run apmode trace <bundle_dir> --cost                  # token/cost rollup
uv run apmode lineage <bundle_dir> <candidate_id>        # transform chain root→target
uv run apmode graph <bundle_dir>                         # search DAG tree view
uv run apmode graph <bundle_dir> --format dot -o dag.dot # Graphviz export
```

### Governance

```bash
# List gate policy versions for all lanes
uv run apmode policies

# Inspect submission-lane thresholds in detail
uv run apmode policies submission

# Validate policy files
uv run apmode policies --validate
```

### Run the Test Suite

```bash
uv run pytest tests/ -q                    # 1525 collected (skip live tests with -m "not live")
uv run mypy src/apmode/ --strict           # type checking
uv run ruff check src/apmode/ tests/       # linting (0 errors)

# Bayesian smoke test against Boeckmann 1994 theophylline (~5 min)
uv run python scripts/bayesian_smoke_theophylline.py
```

---

## Formular — The PK DSL

**Formular** is APMODE's typed domain-specific language for specifying population PK models. The name evokes *formulary* (pharmacy) and *formula* (mathematics). It is the single source of truth for model structure: every model is expressed as a Formular specification, compiled to a typed AST, validated against pharmacometric constraints, and lowered to backend-specific code (nlmixr2 R, Stan, or JAX/Diffrax).

> **Full reference:** [`docs/FORMULAR.md`](docs/FORMULAR.md) — grammar, compilation pipeline, constraint templates, semantic validation rules, extensibility.

**Why a dedicated language?** (1) Safety boundary for agentic AI — the Phase 3 LLM backend operates exclusively through Formular transforms, it cannot emit raw code. (2) Backend independence — one spec, multiple targets. (3) Reproducibility — the compiled `DSLSpec` is the serializable unit in the reproducibility bundle. (4) Constraint enforcement — pharmacometric invariants are checked at the language level before any code generation.

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

**Compilation:** `Formular text → Lark parser (Earley) → DSLTransformer → DSLSpec (Pydantic AST) → Semantic validator → Backend emitter`

**Entry point:** `apmode.dsl.grammar.compile_dsl(text) → DSLSpec`

| DSL Axis | Supported Modules |
|----------|-------------------|
| **Absorption** | IVBolus, FirstOrder, ZeroOrder, LaggedFirstOrder, Transit(n), MixedFirstZero, NODE_Absorption |
| **Distribution** | OneCmt, TwoCmt, ThreeCmt, TMDD_Core, TMDD_QSS |
| **Elimination** | Linear, MichaelisMenten, ParallelLinearMM, TimeVarying(kdecay, decay_fn), NODE_Elimination |
| **Variability** | IIV (diagonal/block), IOV (ByStudy/ByVisit/ByDoseEpoch/Custom), CovariateLink (power/exponential/linear/categorical/maturation) |
| **Observation** | Proportional, Additive, Combined, BLQ_M3, BLQ_M4 (with composable error_model) |
| **Priors** (Bayesian) | Normal, LogNormal, HalfNormal, HalfCauchy, Gamma, InvGamma, Beta, LKJ, Mixture, HistoricalBorrowing (Schmidli 2014 robust MAP) |

**NODE constraint templates:** `monotone_increasing`, `monotone_decreasing`, `bounded_positive`, `saturable`, `unconstrained_smooth` — with lane-dependent dim ceilings (≤8 Discovery, ≤4 Optimization, excluded from Submission).

**Formular transforms** (agentic LLM admissible operations): `swap_module`, `add_covariate_link`, `adjust_variability`, `set_transit_n`, `toggle_lag`, `replace_with_node`, **`set_prior`** — seven typed transforms that produce new `DSLSpec` instances. `set_prior` is parameterization-schema validated: only HalfNormal/HalfCauchy/Gamma/InvGamma families are admissible on IIV ω and residual σ targets; only Normal/LogNormal/Mixture/HistoricalBorrowing on log-scale structural params; LKJ only on correlation matrices. Invalid pairs are rejected at compile time so the LLM cannot propose nonsense priors.

---

## How Candidate Models Are Generated

Candidate generation is APMODE's core algorithmic contribution. It is an **evidence-driven, two-phase search with a fixed budget** that turns a dataset into 30-50 typed model specifications ready for backend fitting. The budget is deterministic (no tunable knobs, no unbounded iteration), pruned by what the profiler observed in the data, and grown once via warm-started children of the best root candidates.

### Phase 1: Build the Search Space from Evidence

The Data Profiler emits an `EvidenceManifest` describing what the dataset supports. `SearchSpace.from_manifest()` translates each flag into allowed modules and constraints:

| Evidence flag | Effect on search space |
|---------------|------------------------|
| `nonlinear_clearance_signature=true` | Adds `MichaelisMenten` + `ParallelLinearMM` to elimination set |
| `absorption_complexity="multi-phase"` | Adds `Transit(n)` + `LaggedFirstOrder` to absorption set |
| `absorption_complexity="lag-signature"` | Adds `LaggedFirstOrder` |
| `richness_category="sparse"` + `absorption_coverage="inadequate"` | Removes NODE from backends (data insufficient) |
| `identifiability_ceiling="low"` | Caps compartment count at 1 |
| `identifiability_ceiling="medium"` | Caps at 2 compartments |
| `identifiability_ceiling="high"` | Allows up to 3 compartments |
| `covariate_burden > 0` | Tracked for SCM search and Gate 2.5 credibility checks |
| `blq_burden > policy.blq_m3_threshold` | Selects BLQ method (`M7+` below, `M3` above) per `MissingDataPolicy` |
| `covariate_missingness.fraction > 0` | Triggers policy-driven `MissingDataDirective`: MI-PMM / MI-missRanger / FREM |
| `time_varying_covariates=true` | Prefers FREM over MI-PMM (Nyberg 2024) |
| Lane = Submission | Hard-blocks NODE backends regardless of data |

This is a **disqualifying filter**, not a weighted score. If the profiler says the data doesn't support 3-cmt estimation, no 3-cmt candidate is generated — saving hours of wasted fitting on unidentifiable models.

### Missing-Data Handling (Covariates + BLQ)

Missing data is resolved by a **lane-tiered policy** (`policies/{submission,discovery,optimization}.json → missing_data`) that turns the `EvidenceManifest` into a binding `MissingDataDirective`. Two independent paths:

**Covariate missingness** (`src/apmode/data/missing_data.py::resolve_directive`):
1. No missingness → `exclude` (no directive action).
2. Time-varying covariates → **FREM** (handled in the NLME likelihood — avoids per-occasion imputation and Ω pooling). Rationale: Nyberg 2024, Jonsson 2024.
3. `fraction > frem_preferred_above` → **FREM** (MI pooling of Ω becomes the dominant error source at high missingness).
4. `fraction > mi_pmm_max_missingness` (between thresholds) → **FREM** (prefer single-fit over large-m MI).
5. Correlated covariates + `missforest_fallback` → **MI-missRanger** (nonlinear relations; Bräm 2022 concept, `missRanger` 2.6.x ranger-backed implementation).
6. Otherwise → **MI-PMM** with `m = policy.m_imputations` (Wijk 2025 DiVA).

**BLQ handling**:
- `blq_burden > policy.blq_m3_threshold` or `blq_force_m3=true` → **M3** (likelihood-based; Beal 2001).
- Otherwise → **M7+** (impute 0 + inflated additive residual error; Wijk 2025).

**Lane tiers**:

| Lane | m | adaptive | BLQ M3 threshold | Rank penalty |
|------|---|----------|------------------|--------------|
| Submission | 20 | → 40 | 5% | 0.0 (hard floor only) |
| Discovery | 5 | off | 15% | 0.25 |
| Optimization | 10 | → 20 | 10% | 0.5 |

**Imputation stability (Gate 1)**. When MI is active, each candidate gets an `ImputationStabilityEntry` (Rubin-pooled OFV/AIC/BIC, `convergence_rate`, top-K `rank_stability`, within/between variance proxy, covariate sign consistency, plus per-parameter Rubin pooling — see below). Gate 1 fails candidates with `convergence_rate < 0.5` (hard floor) or `rank_stability < 1 − policy.imputation_stability_penalty` (soft, lane-driven).

**Rubin's rules for per-parameter pooling**. `apmode.search.stability.rubin_pool(estimates, ses, m_total=...)` implements the canonical Rubin (1987) decomposition on a single scalar parameter:

- `Q̄` — pooled point estimate (mean of per-imputation estimates).
- `Ū` — within-imputation variance (mean of per-imputation `SE²`).
- `B` — between-imputation variance (sample variance of per-imputation estimates).
- `T = Ū + (1 + 1/m) · B` — total variance used for inference.
- `dof` — Barnard–Rubin degrees of freedom for interval construction.

`ImputationStabilityEntry.pooled_parameters` carries the full 5-tuple per parameter so downstream reports can quote Rubin-pooled estimates and confidence intervals with the correct variance decomposition (not just the arithmetic mean of OFV/AIC/BIC). The orchestrator populates the input tuples by having `_run_mi_stage` forward each refit's `(estimate, SE)` pairs on `PerImputationFit.parameter_estimates`. When the backend does not emit SEs (e.g., SAEM without a covariance step), the pool degrades gracefully to a between-imputation-only variance.

**Agentic LLM cherry-picking guard**. The stability manifest is the *only* per-imputation artifact the LLM sees. When `directive.llm_pooled_only=true` (default), `diagnostic_summarizer.summarize_stability_for_llm` substitutes pooled scores + stability scores for raw per-imputation diagnostics — the LLM cannot observe individual imputation draws, so it cannot select transforms that exploit a single lucky imputation.

**Ω-pooling caveats**. Whenever MI is used, the credibility report automatically appends three Rubin/log-Cholesky/EBE caveats to the limitations block for Gate 2.5 ingestion (ICH M15 alignment).

**Categorical covariate format + auto-detection**. APMODE expects binary categorical covariates encoded as canonical `{0, 1}`. Real-world PK data routinely uses other conventions; `apmode.data.categorical_encoding.auto_remap_binary_columns` recognises and remaps the following automatically (no per-dataset configuration required):

| Native form | Example datasets | Mapping |
|-------------|------------------|---------|
| `{0, 1}` | (canonical) | identity, no remap |
| `{1, 2}` 1-indexed integers | mavoglurant SEX | 1 → 0, 2 → 1 |
| `{True, False}` booleans | clinical screening flags | `False` → 0, `True` → 1 |
| `M`/`F`, `Male`/`Female` (any case) | warfarin sex | `f`/`female` → 0, `m`/`male` → 1 |
| `Yes`/`No`, `Y`/`N` | adherence flags | `no`/`n` → 0, `yes`/`y` → 1 |
| `True`/`False`, `Pos`/`Neg`, `Positive`/`Negative` | screening | smaller → 0 |
| `Absent`/`Present`, `Control`/`Case`, `Placebo`/`Active` | trial arms | smaller → 0 |
| Unknown two-level string pair | (any) | alphabetic-order default + `UserWarning` |

Multi-level categorical covariates (>2 distinct values) are **not** auto-remapped — they need k-1 one-hot binary indicators chosen and named by the analyst. Mixed-type columns (e.g., `["M", 1]` mixing strings and numbers) are rejected with an actionable diagnostic.

**Override the polarity** when the auto-detected one is wrong:

```python
from apmode.dsl.frem_emitter import summarize_covariates

covs = summarize_covariates(
    df, ["SEX"],
    transforms={"SEX": "binary"},
    binary_encode_overrides={"SEX": {"male": 0, "female": 1}},  # opposite polarity
)
```

Or call `auto_remap_binary_columns(df, ["SEX"], overrides={...})` directly before passing the DataFrame into the FREM pipeline.

**Provenance**: Every encoding decision (column, detected kind, applied remap, source = `auto`/`override`/`no_remap`, rationale) is captured in `categorical_encoding_provenance.json` in the run bundle so reviewers can trace exactly how each raw categorical value was mapped to 0/1.

**R harness**. Imputation uses `src/apmode/r/impute.R`, which dispatches to `mice::mice(method="pmm")` or `missRanger::missRanger(num.trees=100, pmm.k=10)` — the ranger-backed fast alternative to missForest (Mayer CRAN 2.6.x). Python-side providers are `R_MiceImputer` and `R_MissRangerImputer` (`src/apmode/data/imputers.py`); both are covered by live tests in `tests/unit/test_imputers_live.py` that spawn real Rscript against installed `mice`/`missRanger` and verify (a) imputed CSVs appear, (b) no residual NaN in imputed columns, (c) between-imputation variance proves MI is functioning.

**FREM emitter**. The FREM path is implemented in `src/apmode/dsl/frem_emitter.py` and executed via `src/apmode/backends/frem_runner.py::run_frem_fit`. Public API:

- `FREMCovariate(name, mu_init, sigma_init, dvid, epsilon_sd, transform, time_varying)` — per-covariate metadata. `transform="log"` for positive/right-skewed covariates (Yngman 2022 conditioning); `transform="binary"` for 0/1-coded categorical covariates (additive-normal endpoint, the standard categorical-FREM compromise — multi-level categorical covariates are one-hot encoded upstream into k−1 binary indicators). `time_varying=True` emits per-(subject, TIME) observation rows and leaves `sig_cov_*` estimable so it absorbs within-subject variation while the eta continues to capture between-subject variance.
- `summarize_covariates(df, names, transforms=...)` — compute baseline (min TIME) mean + SD per covariate; rejects duplicates and covariates with <2 observed subjects.
- `prepare_frem_data(df, covariates)` — pivot to DVID-multiplexed long format; adds one observation row per subject per covariate at the subject's baseline time, rejects DVID collisions with any existing multi-analyte scheme.
- `emit_nlmixr2_frem(spec, covariates, initial_estimates=...)` — emits the augmented nlmixr2 model function with a joint Ω block (PK IIV etas + covariate etas), estimable covariate means, fixed covariate residuals (`fix(...)`) so the eta absorbs all BSV, and one observation endpoint per covariate. **Routing is data-driven via the `DVID` column**: nlmixr2 assigns endpoints DVID 1 (PK), 2, 3, … in declaration order, and `prepare_frem_data` writes matching DVIDs. No `| DVID==N` pipe is emitted — nlmixr2 5.0 rejects conditions on the endpoint RHS (verified live 2026-04-14).
- `run_frem_fit(spec_template, df, covariate_names, runner, ...)` — composes the above with `Nlmixr2Runner` for a single-call FREM fit.

Scope covers **static + time-varying** subject-level covariates and **continuous + log-transformed + binary-categorical** encodings. `summarize_covariates` auto-detects per-covariate time-varying status by checking within-subject variation and sets `time_varying=True` accordingly. Live end-to-end tests in `tests/unit/test_frem_emitter.py::TestFREMLiveIntegration` spawn real Rscript + nlmixr2 and verify (a) the emitted model compiles, (b) FOCE-I actually fits the joint Ω and learns a non-degenerate `eta.cov.WT` variance on tiny synthetic data.

**Estimator requirement**: FREM requires **FOCE-I**. nlmixr2 SAEM treats subject-level covariate observations as dynamic sampling targets and collapses the random-effect variance to zero. The FREM runner must be constructed with `Nlmixr2Runner(estimation=["focei"])`; `Orchestrator._run_frem_stage` does this by default when no `frem_runner` is injected.

**Orchestrator execution**: `missing_data_directive.json` is written to every run's reproducibility bundle. When the directive resolves to **FREM**, `Orchestrator._run_frem_stage` automatically refits the best healthy classical candidate (filtered to `backend="nlmixr2"` + finite BIC + not `ill_conditioned`) via `run_frem_fit`. When the directive resolves to **MI-PMM / MI-missRanger**, `_run_mi_stage` freezes the classical candidate set and refits each spec on m imputed datasets (produced by `R_MiceImputer` / `R_MissRangerImputer` by default, or an injected `mi_provider`); `aggregate_stability` applies Rubin's rules, emits `imputation_stability.json`, and feeds per-candidate stability entries into Gate 1. Both stages wrap their work in a narrow `(BackendError, RuntimeError, ValueError, NotImplementedError)` handler with `exc_info=True` so classical search results survive a missing-data failure.

### Phase 2: Generate Root Candidates

`generate_root_candidates(space, base_params=nca_estimates)` enumerates the cross-product of allowed modules, seeded with parameter estimates from **per-subject PKNCA-style NCA** (adaptive lambda_z selection by curve-stripping, linear-up/log-down AUC integration per Purves 1992, and QC gates on adj-R², AUC extrapolation fraction, and span ratio; subjects failing QC are excluded from the population median, and when ≥50% are excluded APMODE falls back to `RunConfig.fallback_estimates` — e.g., a dataset card's `published_model.key_estimates` — or conservative defaults. Per-subject diagnostics are emitted as `nca_diagnostics.jsonl` in the reproducibility bundle):

```
for cmt in space.structural_cmt:              # e.g., [1, 2, 3]
    for abs_type in space.absorption_types:   # [FirstOrder, Transit-3, Lagged]
        for elim_type in space.elimination_types:  # [Linear, MM, ParallelLinearMM]
            for error_type in space.error_types:   # [Proportional, Combined]
                yield DSLSpec(...)
```

For a typical dataset with rich sampling, nonlinear CL, and high identifiability, this produces ~20-30 root candidates. NCA-seeded initial estimates dramatically reduce SAEM convergence time vs. fitting from arbitrary values.

The NCA estimator is **unit-aware**: when raw CL is implausibly small (`< 0.5 L/h`), the DV magnitude looks like ng/mL territory (`> 50`), and `dose_median ≥ 1` (mg-scale human dosing), it applies a `x1000` scaling factor — accommodating the common pharmacometric convention of dose-in-mg with concentration-in-ng/mL without requiring users to specify units explicitly. The `dose_median` guard prevents over-correction of preclinical data (small-animal studies, biologics) where a legitimate low-CL drug with ng/mL concentrations may superficially match the heuristic. The applied scale factor is recorded as `_unit_scale_applied` in the initial estimates bundle for auditability.

### Phase 3: Warm-Start Children (Fixed Budget, Max 18)

After all root candidates are fit, the SearchEngine selects the **top-3 converged roots by BIC** and generates warm-started children from each. For each parent, it produces up to 2 children per error model type (proportional, additive, combined), giving a hard ceiling of **3 parents × 3 error types × 2 children = 18 child candidates**. Duplicates (deduplicated by `model_id`) are skipped, so the actual count is usually lower.

Children **warm-start from their parent's fitted parameters** rather than NCA estimates. For an `ka`/`CL`/`V` 2-cmt parent that converged to `ka=0.8, CL=3.2, V1=45, V2=80, Q=9`, those values seed the child's initial estimates, cutting SAEM convergence time by ~3-5x. The child's structural modules can differ (e.g., different IIV structure or error model), but the shared structural parameters inherit the parent's fitted values.

Every parent → child relationship is tracked in the **Search DAG** (`candidate_lineage.jsonl`), with edge labels like `"warm_start_combined"` indicating which error model drove the child.

### Total Budget

```
total_budget = phase1_roots + phase3_children
             ≤ |cmt| × |absorption| × |elimination| × |error_types|  +  18
```

For a typical rich dataset with nonlinear CL and multi-phase absorption:

| Dimension | Typical size |
|-----------|--------------|
| Phase 1 roots | 20-30 (cross-product after `_build_spec()` filters inadmissible combos) |
| Phase 3 children | up to 18 |
| **Total** | **~30-50 candidates** |

The search **does not** iterate until convergence, apply transforms beyond warm-start children, or have a user-tunable `--max-candidates` flag. The budget is fully determined by the Evidence Manifest. This makes wall time estimable as `total_budget × avg_fit_seconds / max_concurrency`.

### What About DSL Transforms?

The DSL supports typed transforms (`add_iiv`, `change_absorption`, `add_covariate`, etc.) documented in [`docs/FORMULAR.md`](docs/FORMULAR.md). These are used by the **agentic LLM backend** (Phase 3 of APMODE, see [Agentic LLM Backend](#agentic-llm-backend) below), which applies them iteratively based on LLM proposals up to `--max-iterations`. The **automated search engine does not apply transforms iteratively** — it only does the Phase 1 cross-product + Phase 3 warm-start children. If you want iterative transform-based exploration, enable the agentic backend with `apmode run --agentic --lane discovery`.

### Example: What This Produces on Mavoglurant

Running on the 120-subject Novartis mavoglurant dataset (`nlmixr2data::mavoglurant`), the profiler detects: nonlinear CL, rich sampling, 6 potential covariates, inadequate absorption-phase coverage. The resulting search:

- **Phase 1 roots:** ~20-30 candidates = (1,2,3)-cmt × (FirstOrder, Transit, Lagged) × (Linear, MM, ParallelLinearMM) × (Proportional, Additive, Combined), minus `_build_spec()`-filtered combos
- **Phase 3 children:** top-3 converged roots warm-start up to 18 children with varied error models (some dedup, usually 10-15 actual)
- **Time per candidate:** 2-5 minutes SAEM on 120 subjects, parallelizable up to `max_concurrency`
- **Total wall time:** ~60-90 minutes for a single-threaded submission-lane run
- **Output:** `candidate_lineage.jsonl` (DAG edges), `search_trajectory.jsonl` (per-candidate BIC/OFV/convergence)

### Why This Design

1. **Evidence-driven pruning** — no wasted fits on models the data can't identify (saves 10x+ runtime vs. full grid)
2. **Deterministic budget** — reproducible runtime estimates, no runaway loops, no hyperparameters to tune
3. **Warm-start efficiency** — children inherit converged parent parameters (3-5x faster SAEM)
4. **Auditability** — every candidate has a logged origin: either a root (Phase 1) or a warm-start child of a specific parent (Phase 3). Use `apmode lineage <bundle> <candidate>` to trace it.
5. **Typed safety boundary** — transforms exist as a finite, validated DSL operation set. The automated search uses a restricted subset (error-model variation via warm-start). The full transform set is available to the agentic LLM backend, which cannot emit raw code.

---

## Architecture

```
NONMEM CSV ──→ Ingest + Validate ──→ Canonical PK Schema (Pandera)
                                            │
                                  ┌─────────┴──────────┐
                                  ↓                    ↓
                          Data Manifest          Evidence Manifest
                          (SHA-256,              (richness, route,
                           covariates)            CL linearity, BLQ)
                                  │                    │
                                  ↓                    ↓
                          NCA Estimator          Search Space
                          (CL, V, ka,            (dispatch constraints)
                           multi-dose)                 │
                                  │                    ↓
DSL text ──→ Lark parser ──→ AST ──→        Search Engine
                          │                  ├─ Candidate generation
                  Semantic validator         ├─ Multi-backend dispatch
                          │                  │   ├─ Classical → nlmixr2
                  split_subjects()           │   └─ NODE     → JAX/Diffrax
                  (k-fold, LORO)             └─ Pareto frontier + BIC scoring
                                                       │
                                          ┌────────────┼────────────┐
                                          ↓            ↓            ↓
                                    Gate 1:       Gate 2:      Gate 2.5:
                                    Technical     Lane         ICH M15
                                    Validity      Admissibility Credibility
                                    (7 checks)    (6 checks)   (5 checks)
                                          └────────────┴────────────┘
                                                       │
                                                  Gate 3: Ranking
                                            (within-paradigm BIC or
                                             cross-paradigm VPC/NPE)
                                                       │
                                                       ↓
                                        Reproducibility Bundle (JSON/JSONL)
```

### Key Components

| Component | Path | Role |
|-----------|------|------|
| PK DSL grammar | `src/apmode/dsl/pk_grammar.lark` | Full Lark EBNF for PK model specs |
| AST models | `src/apmode/dsl/ast_models.py` | Typed Pydantic nodes for all DSL axes |
| Semantic validator | `src/apmode/dsl/validator.py` | Constraint table enforcement (PRD §4.2.5) |
| nlmixr2 emitter | `src/apmode/dsl/nlmixr2_emitter.py` | DSL AST → R code for nlmixr2/rxode2 |
| Stan emitter | `src/apmode/dsl/stan_emitter.py` | DSL AST → Stan program |
| Data pipeline | `src/apmode/data/` | Ingestion, profiling, NCA estimates, splitting |
| Dose expansion | `src/apmode/data/dosing.py` | ADDL/II expansion, infusion events, event table builder |
| Classical backend | `src/apmode/backends/nlmixr2_runner.py` | Async subprocess runner with file-based IPC |
| NODE backend | `src/apmode/backends/node_*.py` | Bram-style hybrid MLP, Diffrax ODE, Optax training |
| Agentic LLM backend | `src/apmode/backends/agentic_runner.py` | Closed-loop LLM model improvement via Formular transforms |
| LLM providers | `src/apmode/backends/llm_providers.py` | Anthropic, OpenAI, Gemini, Ollama, OpenRouter, litellm |
| Governance | `src/apmode/governance/` | Gates 1/2/2.5/3, cross-paradigm ranking, policy files |
| Search engine | `src/apmode/search/engine.py` | Multi-backend dispatch, BIC scoring, Pareto frontier |
| Orchestrator | `src/apmode/orchestrator/` | Full pipeline: ingest → gates → bundle |
| Bundle emitter | `src/apmode/bundle/` | All reproducibility bundle artifacts per PRD §5 |
| Dataset registry | `src/apmode/data/datasets.py` | 14 public PK datasets from nlmixr2data with auto-fetch |
| CLI | `src/apmode/cli.py` | Typer CLI: `run`, `explore`, `datasets`, `inspect`, `log`, `diff`, `validate` |

---

## Governance Funnel

APMODE enforces a **gated funnel** — not a weighted sum. Each gate is disqualifying; only survivors advance.

| Gate | Purpose | Checks |
|------|---------|--------|
| **Gate 1** | Technical Validity | Convergence, parameter plausibility, CWRES normality, VPC coverage, condition number, seed stability, split integrity |
| **Gate 2** | Lane Admissibility | Interpretability, shrinkage, identifiability (profile-likelihood CI), NODE exclusion (Submission lane) |
| **Gate 2.5** | Credibility Qualification | ICH M15 context-of-use, data adequacy, ML transparency, limitation-risk mapping, operational qualification |
| **Gate 3** | Ranking | Within-paradigm BIC; cross-paradigm VPC concordance + NPE + composite score |

Gate thresholds are **versioned policy artifacts** in `policies/*.json` — not hard-coded constants.

---

## Three Operating Lanes

APMODE routes work through separate pipelines with different admissible backends and evidence standards:

| Lane | Purpose | Admissible Backends | NODE Eligible? |
|------|---------|-------------------|----------------|
| **Submission** | Regulatory-grade models | Classical NLME only | No (hard rule) |
| **Discovery** | Broad exploration | Classical + NODE | Yes (Gate 2.5 required) |
| **Translational Optimization** | LORO-CV prediction (Phase 3) | All backends | Yes |

---

## Benchmark Suites

### Suite A — Structure and Parameter Recovery

Simulated PK datasets with known ground truth (PRD §5). Requires R 4.4+ with `rxode2`.

```bash
Rscript benchmarks/suite_a/simulate_all.R [output_dir]
```

| Scenario | Model | Key Test |
|----------|-------|----------|
| A1 | 1-cmt oral, first-order abs, linear elim | Structure identification |
| A2 | 2-cmt IV, parallel linear+MM elim | Compartment count + nonlinear CL |
| A3 | Transit (n=3), 1-cmt, linear elim | Transit chain detection |
| A4 | 1-cmt oral, MM elimination | Nonlinear clearance detection |
| A5 | TMDD quasi-steady-state (SC mAb) | TMDD vs. 2-cmt confusion |
| A6 | 1-cmt oral + allometric WT + renal covariate | Covariate structure recovery |
| A7 | 2-cmt + NODE saturable absorption | NODE shape recovery + surrogate fidelity |

### Suite B — NODE-Specific Validation

| Scenario | Test | Assertion |
|----------|------|-----------|
| B1 | NODE absorption recovery | Mock NODE fit passes Gate 1+2 Discovery |
| B2 | Sparse data + NODE dispatch | Lane Router blocks NODE when data insufficient |
| B3 | Cross-paradigm ranking | Gate 3 correctly ranks mixed nlmixr2 + jax_node candidates |

### End-to-End Benchmark Results (v0.3.0-rc4, 2026-04-16)

Full `apmode run` pipeline on all Suite A fixtures plus the three real
public datasets. Driver: `scripts/run_full_benchmark.sh`; bundles land
in `benchmarks/runs/full-<timestamp>/`.

**Gate funnel — 10/10 scenarios produced a recommended candidate.**

| Dataset | Lane | Gate 1 | Gate 2 | Recommended | Best BIC |
|---|---|---|---|---|---|
| a1_1cmt_oral_linear | submission | 19 / 34 | 19 / 19 | ✓ | 2848.3 |
| a2_2cmt_iv_parallel_mm | discovery | 8 / 41 | 8 / 8 | ✓ | 4396.7 |
| a3_transit_1cmt_linear | submission | 9 / 34 | 9 / 9 | ✓ | 2623.2 |
| a4_1cmt_oral_mm | discovery | 4 / 41 | 4 / 4 | ✓ | 3113.3 |
| a5_tmdd_qss | discovery | 7 / 34 | 7 / 7 | ✓ | 4858.8 |
| a6_1cmt_covariates | submission | 17 / 34 | 17 / 17 | ✓ | −317.5 |
| a7_2cmt_node_absorption | discovery | 22 / 34 | 22 / 22 | ✓ | 807.4 |
| warfarin (nlmixr2data) | submission | 19 / 33 | 19 / 19 | ✓ | 960.1 |
| theo_sd (nlmixr2data) | submission | 10 / 25 | 8 / 10 | ✓ | 396.3 |
| mavoglurant (nlmixr2data) | discovery | 4 / 37 | 4 / 4 | ✓ | 28906.9 |
| **Total** | | **119 / 347 (34%)** | **117 / 119 (98%)** | **10 / 10** | |

Gate 1 rejects non-converged candidates and those failing plausibility,
CWRES normality, or trajectory validity. Gate 2 (submission lane) applies
shrinkage ≤ 30%, identifiability, and reproducible-estimation checks.
Gate 2 now passes 98% of Gate 1 survivors across all scenarios — the rc3
log-space plausibility fix and rc4 shrinkage unit fix (percentage → fraction)
resolved the two systematic false-positive rejection sources.

Every bundle passes `apmode validate` and emits the v3 structured
`nonlinear_clearance_signals` record with full provenance (algorithm,
citation, policy_key, threshold, observed value + 90% CI, eligibility
reason, vote). `apmode inspect <bundle>` renders the per-signal table;
`apmode lineage <bundle> <candidate_id>` traces the transform DAG.

---

## Test Suite

1525 tests across multiple strategies:

```bash
uv run pytest tests/unit/ -q               # unit tests
uv run pytest tests/integration/ -q        # integration tests
uv run pytest tests/property/ -q           # Hypothesis property-based tests
uv run pytest tests/golden/ -q             # syrupy golden master snapshots
uv run pytest tests/ --snapshot-update     # update snapshots after emitter changes
```

| Directory | Coverage |
|-----------|----------|
| `tests/unit/` | All modules: DSL, data, backends, search, governance, routing, bundle |
| `tests/integration/` | Mock R pipeline, Discovery lane, LLM providers |
| `tests/property/` | Hypothesis property-based tests |
| `tests/golden/` | Syrupy snapshots for emitter output |
| `tests/r_syntax/` | R code syntax validation |

---

## Phasing

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 0** | Schemas, protocols, grammar, error taxonomy, sparkid integration | Complete |
| **Phase 1** (6 months) | DSL + compiler, classical NLME backend, automated search, Gates 1-3, reproducibility bundle, CLI, Suite A | Complete |
| **Phase 2** (4 months) | Hybrid NODE backend, functional distillation, Discovery lane, cross-paradigm ranking, Suite B, Stan codegen (IOV + BLQ), NODE init strategy, dataset registry, interactive CLI | Complete |
| **Phase 3** (4 months) | Agentic LLM backend (DSL transforms only), Optimization lane + LORO-CV, report generator, Suite C, API | In progress |

---

## Pharmacometric References

- **Smith 2000 dose-proportionality power model**: Smith BP, Vandenhende FR, DeSante KA, Farid NA, Welch PA, Callaghan JT, Forgue ST (2000). Confidence interval criteria for assessment of dose proportionality. *Pharm Res* 17(10):1278-1283. doi:10.1023/a:1026451721686
- **Huang 2025 λz selector + initial-estimates pipeline**: Huang Z, Fidler M, Lan M, Cheng I-L, Kloprogge F, Standing JF (2025). An automated pipeline to generate initial estimates for population pharmacokinetic base models. *J Pharmacokinet Pharmacodyn* 52:60. doi:10.1007/s10928-025-10000-z
- **Wagner-Nelson absorption**: Wagner JG, Nelson E (1963). Percent absorbed time plots derived from blood level and/or urinary excretion data. *J Pharm Sci* 52(6):610-611. doi:10.1002/jps.2600520627
- **Richardson 2025 popPK automation**: Richardson S, Irurzun-Arana I et al. (2025). A machine learning approach to population pharmacokinetic modelling automation. *Commun Med* 5:327. doi:10.1038/s43856-025-01054-8
- **Pharmpy AMD**: Chen X, Hooker AC, Karlsson MO et al. (2024). A fully automatic tool for development of population pharmacokinetic models. *CPT Pharmacometrics Syst Pharmacol* 13:1785-1797
- **BLQ M3 likelihood**: Beal SL (2001). Ways to fit a PK model with some data below the quantification limit. *J Pharmacokinet Pharmacodyn* 28(5):481-504. doi:10.1023/a:1012299115260
- **BLQ method comparison**: Ahn JE, Karlsson MO, Dunne A, Ludden TM (2008). Likelihood-based approaches to handling data below the quantification limit using NONMEM VI. *J Pharmacokinet Pharmacodyn* 35(4):401-421. doi:10.1007/s10928-008-9094-4
- **SCM covariate selection**: Wählby U, Jonsson EN, Karlsson MO (2002). Comparison of stepwise covariate model building strategies in population pharmacokinetic-pharmacodynamic analysis. *AAPS PharmSci* 4(4):E27. doi:10.1208/ps040427
- **Covariate model power**: Ribbing J, Jonsson EN (2004). Power, selection bias and predictive performance of the population pharmacokinetic covariate model. *J Pharmacokinet Pharmacodyn* 31(2):109-134. doi:10.1023/B:JOPA.0000034404.86036.72
- **TMDD full binding**: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532
- **TMDD QSS**: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591
- **Transit compartments**: Savic et al. (2007), J Pharmacokinet Pharmacodyn 34:711-726
- **Allometric scaling**: Anderson & Holford (2008), Clin Pharmacokinet 47:455-467
- **NCA**: PKNCA-style curve-stripping for terminal lambda_z (`pk.calc.half.life`, adjusted R² with most-points tiebreak), linear-up/log-down AUC integration (Purves 1992, `pk.calc.auc`). Reference: Purves (1992) J Pharmacokin Biopharm 20:211; PKNCA vignettes at https://humanpred.github.io/pknca/

---

## Data Profiler Parameter Dictionary

Every threshold used by `apmode.data.profiler` is stored in a versioned
policy artifact so runs are reproducible without touching Python source:

- **Source of truth**: [`policies/profiler.json`](policies/profiler.json)
- **Loader**: [`src/apmode/data/policy.py`](src/apmode/data/policy.py) — `get_policy()` returns a typed frozen `ProfilerPolicy` dataclass; `policy_sha256` is embedded in every `EvidenceManifest`.
- **Current version**: `profiler/v2.1.0` (`manifest_schema_version = 3`; structured `nonlinear_clearance_signals` replaces the flat scalars/masks)
- **Derivation & citations**: [`docs/PROFILER_REFINEMENT_PLAN.md`](docs/PROFILER_REFINEMENT_PLAN.md) — one paragraph per policy group tying each default to a primary source.
- **Drift guard**: [`tests/unit/test_profiler_policy_consistency.py`](tests/unit/test_profiler_policy_consistency.py) — enforces JSON↔dataclass↔constants equality and AST-scans drift-prone heuristic functions for bare numeric literals.

| Group | Parameter | Default | Purpose |
|-------|-----------|---------|---------|
| **Smith 2000 dose-proportionality** | `theta_low` | 0.80 | Lower bound of bioequivalence-style exposure interval |
| | `theta_high` | 1.25 | Upper bound |
| | `min_dose_levels` | 3 | Minimum distinct dose levels required for eligibility |
| | `min_dose_ratio` | 3.0 | Minimum ratio of highest to lowest dose |
| **Huang 2025 λz selector** | `min_points` | 3 | Minimum points in the terminal-fit window |
| | `tolerance` | 1e-4 | Adjusted-R² tie-break tolerance |
| | `phoenix_constraint` | true | Restrict window to second half of post-Cmax (falls back to `min_points` on sparse profiles) |
| | `adj_r2_threshold` | 0.7 | Minimum acceptable terminal-fit quality |
| **Steady-state check** | `n_half_lives_required` | 3 | Elapsed duration in half-lives for SS (87.5% attainment) |
| | `n_doses_alt` | 5 | Alternative dose-count path to SS |
| | `interval_tolerance` | 0.25 | ±25% inter-dose variation |
| | `dose_tolerance` | 0.20 | ±20% dose amount variation |
| | `min_doses` | 3 | Minimum doses required in any SS branch |
| **Nonlinear clearance** | `mm_curvature_ratio` | 1.8 | Early/late slope ratio triggering MM vote |
| | `tmdd_curvature_ratio` | 0.3 | Inverse ratio triggering TMDD vote |
| | `compartmentality_curvature_ratio` | 1.3 | Separate threshold for 2-cmt linear (distinct from MM) |
| | `terminal_monoexp_r2_linear_threshold` | 0.85 | Monoexp R² above this is evidence for linear PK |
| **Covariate** | `correlation_threshold_abs_r` | 0.7 | Inter-covariate |r| flagging collinearity |
| | `missingness_full_information_cutoff` | 0.15 | Fraction above → full-information likelihood |
| **Error model** | `blq_m3_trigger` | 0.10 | BLQ fraction triggering M3 |
| | `dynamic_range_proportional` | 50.0 | Cmax_p95/p05 triggering proportional error |
| | `high_cv_ceiling` | 80.0 | Suppresses proportional recommendation at extreme CV% |
| | `lloq_cmax_combined` | 0.05 | LLOQ/Cmax median triggering combined |
| | `terminal_log_mad_combined` | 0.35 | Terminal log-residual MAD triggering combined |
| | `narrow_range_additive` | 5.0 | Dynamic range below this + low CV → additive error plausible |
| | `low_cv_additive_ceiling` | 30.0 | DV CV% ceiling for additive recommendation (biomarker-like) |
| **NODE readiness** | `dim_budget.discovery` | 8 | NODE dim cap for Discovery lane |
| | `dim_budget.optimization` | 4 | NODE dim cap for Optimization lane |
| | `min_subjects.{discovery,optimization}` | 20 / 10 | Minimum subjects per lane |
| | `min_median_samples.{discovery,optimization}` | 8 / 4 | Minimum median samples/subject |
| **Flip-flop** | `ka_lambdaz_ratio_likely` | 1.0 | `ka < λz` → flip-flop "likely" |
| | `ka_lambdaz_ratio_possible` | 1.5 | `ka < 1.5·λz` → "possible" |
| | `quality_adj_r2_min` | 0.85 | **Strict** terminal adj-R² required to retain `"likely"` (Richardson 2025) |
| | `quality_min_npts` | 4 | Minimum median terminal-window size for the strict-quality check |
| **TAD consistency** | `in_window_fraction_clean` | 0.80 | Fraction of obs within union of per-dose intervals |
| **Protocol heterogeneity** | `obs_per_subject_cv_threshold` | 0.5 | Across-study CV of obs/subject above this flags `pooled-heterogeneous` |
| **DVID filter** | `pk_dvid_allowlist` | ["cp", "1", "conc", "concentration"] | Accepted observation DVIDs |
| | `fail_open_when_no_match` | true | Keep all rows and WARN (vs raise ValueError) when allowlist matches nothing |
| **Shape detection** | `multi_peak_fraction_threshold` | 0.3 | Fraction of subjects with ≥2 prominent peaks |
| | `lag_signature_fraction_threshold` | 0.5 | Fraction with delayed-onset absorption |
| | `peak_prominence_range_fraction` | 0.10 | Prominence as fraction of dynamic range |
| | `peak_prominence_cmax_floor` | 0.05 | Prominence floor as fraction of Cmax |
| | `peak_min_distance_intervals` | 2.0 | Minimum inter-peak distance in sampling intervals |
| **Subject quality** | `min_concs_for_profile` | 5 | Minimum per-subject observations for shape analysis |
| | `min_subjects_for_median` | 4 | Minimum subjects for population-median statistics |
| | `min_subjects_for_dynamic_range` | 10 | Minimum positive-Cmax subjects for p95/p05 ratio |
| | `min_obs_per_subject_moderate` | 4 | Richness band: sparse / moderate boundary |
| | `min_obs_per_subject_rich` | 8 | Richness band: moderate / rich boundary |
| | `absorption_coverage_min_pre_tmax` | 2.0 | Mean pre-Tmax samples for "adequate" coverage |
| | `elimination_coverage_min_post_tmax` | 3.0 | Mean post-Tmax samples for "adequate" coverage |

### Advisory vs disqualifying

Profiler fields populate the EvidenceManifest — they are **advisory**
inputs that shape dispatch (Lane Router) and ranking (Gate 3). None of
them on their own remove a candidate. Hard disqualification lives in
the **Gate policies** (`policies/submission.json`, `discovery.json`,
`optimization.json`) and is evaluated by `governance/gates.py`.

The Huang-2025 `lambdaz_adj_r2_threshold = 0.7` is used as a
**quality floor** (advisory) inside `_assess_flip_flop_risk`: below
this, the profiler returns `"unknown"` because the terminal fit is
too poor to support any flip-flop call. The stricter
`flip_flop.quality_adj_r2_min = 0.85` is required to retain a
`"likely"` call — a lower-quality-but-still-usable fit downgrades to
`"possible"`.

### Gate policy parameters (`policies/{submission,discovery,optimization}.json`)

Gate policies are lane-specific, versioned, and discoverable via
`apmode policies <lane>`. The Pydantic schema lives in
[`src/apmode/governance/policy.py`](src/apmode/governance/policy.py)
and is validated by the CI hook
[`src/apmode/governance/validate_policies.py`](src/apmode/governance/validate_policies.py).

| Gate | Parameter | Submission | Discovery | Optimization | Purpose |
|---|---|---|---|---|---|
| **Gate 1 — Technical validity** | `cwres_mean_max` | 0.10 | 0.15 | 0.12 | Max magnitude of population-level CWRES mean |
| | `outlier_fraction_max` | 0.05 | 0.08 | 0.06 | Max fraction of \|CWRES\| > 4 outliers |
| | `vpc_coverage_lower` / `_upper` | 0.85 / 0.995 | 0.80 / 0.995 | 0.82 / 0.995 | Per-band VPC coverage bounds |
| | `seed_stability_n` | 3 | 3 | 3 | Required seed replicates (top-K candidates only) |
| | `seed_stability_cv_max` | 0.10 | 0.10 | 0.10 | Max CV of OFV across seeds |
| | `obs_vs_pred_r2_min` | 0.30 | 0.30 | 0.30 | Minimum R² for state-trajectory validity |
| | `cwres_sd_min` / `_max` | 0.50 / 2.0 | 0.50 / 2.0 | 0.50 / 2.0 | CWRES SD should hug 1.0 |
| | `gradient_norm_max` | 100.0 | 100.0 | 100.0 | Convergence gradient norm ceiling |
| | Bayesian `rhat_max` | 1.01 | 1.01 | 1.01 | Vehtari 2021 rank-normalized R̂ |
| | Bayesian `ess_bulk_min` / `_tail_min` | 400 / 400 | 400 / 400 | 400 / 400 | Vehtari 2021 ESS floor |
| | Bayesian `n_divergent_max` | 0 | 0 | 0 | HMC divergence ceiling |
| **Gate 2 — Lane admissibility** | `interpretable_parameterization` | required | not_required | preferred | Blocks NODE in Submission |
| | `shrinkage_max` | 0.30 | null | 0.30 | Max η-shrinkage (null = no threshold) |
| | `identifiability_required` | true | false | true | Profile-likelihood CI + condition number check |
| | `node_eligible` | false | true | true | Hard Submission exclusion (PRD §3) |
| | `loro_required` | false | false | true | LORO-CV required for Optimization |
| | LORO `loro_npde_mean_max` | — | — | 0.3 | NPE pool-mean ceiling |
| | LORO `loro_npde_variance_{min,max}` | — | — | 0.5 / 1.5 | NPE pool-variance bounds |
| | LORO `loro_vpc_coverage_min` | — | — | 0.80 | VPC coverage concordance floor |
| | LORO `loro_min_folds` | — | — | 3 | Minimum regimen folds |
| **Gate 2.5 — Credibility (ICH M15)** | `context_of_use_required` | n/a | true | true | COU statement present |
| | `limitation_to_risk_mapping_required` | n/a | false | true | Risk-mapped limitations |
| | `data_adequacy_required` | n/a | true | true | n_obs / n_params check |
| | `data_adequacy_ratio_min` | n/a | 5.0 | 5.0 | Minimum obs/params ratio |
| | `sensitivity_analysis_required` | n/a | false | true | Sensitivity artifact present |
| | `ai_ml_transparency_required` | n/a | true | true | Statement required for NODE/agentic |
| **Gate 3 — Cross-paradigm composite** | `gate3.composite_method` | `weighted_sum` | `weighted_sum` | `weighted_sum` | Aggregation: `"weighted_sum"` or `"borda"` (rank-based, scale-invariant) |
| | `gate3.vpc_weight` | 0.5 | 0.5 | 0.5 | VPC coverage concordance weight |
| | `gate3.npe_weight` | 0.5 | 0.5 | 0.5 | NPE (simulation if available, else CWRES proxy) weight |
| | `gate3.bic_weight` | 0.0 | 0.0 | 0.0 | **BIC off by default** (PRD §10 Q2 — likelihoods incomparable cross-paradigm) |
| | `gate3.npe_cap` | 5.0 | 5.0 | 5.0 | Weighted-sum only: NPE value above this clamps to 1.0 |
| | `gate3.bic_norm_scale` | 1000.0 | 1000.0 | 1000.0 | Weighted-sum only: divisor for BIC normalization |
| | `vpc_concordance_target` | 0.90 | 0.90 | 0.90 | Target coverage for concordance score |
| **Missing data** | `mi_pmm_max_missingness` | 0.20 | 0.30 | 0.25 | Above → FREM preferred |
| | `frem_preferred_above` | 0.25 | 0.40 | 0.30 | Explicit FREM trigger |
| | `m_imputations` | 20 | 5 | 10 | Default MI multiplicity |
| | `adaptive_m` | true | false | true | Escalate m up to `m_max` on variance |
| | `m_max` | 40 | 10 | 20 | Adaptive ceiling |
| | `adaptive_variance_threshold` | 0.05 | 0.10 | 0.08 | Between-imputation variance trigger |
| | `blq_m3_threshold` | 0.05 | 0.15 | 0.10 | BLQ burden → M3 (else M7+) |
| | `imputation_stability_penalty` | 0.0 | 0.25 | 0.5 | Rank-stability threshold = 1 − penalty |

All gate thresholds are sourced from primary literature:
CWRES conventions (Nguyen et al. 2017, *CPT PSP* 6:87), VPC coverage
(Bergstrand & Karlsson 2009, *AAPS J* 11:371), R̂/ESS (Vehtari et al.
2021, *Bayesian Analysis* 16:667), BLQ M3/M7 (Beal 2001, *JPKPD* 28:481;
Wijk et al. 2025), FREM (Nyberg 2024). Lane-specific tolerances
follow the PRD §4.3.1 submission/discovery/optimization risk profile.

To tune for a deployment, edit the relevant policy JSON, re-run
`apmode run ...`, and inspect the emitted
`evidence_manifest.policy_sha256` (profiler) or
`policy_file.json` (gate) in the reproducibility bundle to confirm
the change propagated.

---

## Acknowledgments

The APMODE Data Profiler stands on the shoulders of the open-source
pharmacometric community. We specifically build on:

- **nlmixr2 / nlmixr2autoinit** (Standing lab, UCL; Matt Fidler et al.) — the Huang 2025 `find_best_lambdaz`, `is_ss`, Wagner-Nelson helpers, and the automated initial-estimates pipeline are replicated faithfully from the R package. https://github.com/ucl-pharmacometrics/nlmixr2autoinit
- **Pharmpy / AMD tool** (Uppsala Pharmacometrics Research Group) — the search-space MFL, structural/IIV/RUV search algorithms, and dispatch-feasibility concepts inform APMODE's Lane Router. https://github.com/pharmpy/pharmpy
- **pyDarwin** (Certara; Sale, Sherer, Nieforth et al.) — the Bayesian-optimisation + penalty-function framework for popPK structural search motivated APMODE's penalty / plausibility signals. https://github.com/certara/pyDarwin
- **rxode2 / nlmixr2est** — canonical oral / IV / infusion model code generation.
- **PKNCA** (Denney) — the reference NCA implementation whose conventions (linear-up/log-down AUC, `pk.calc.half.life`) APMODE's profiler follows. https://github.com/humanpred/pknca
- **pandas, numpy, scipy, scikit-learn, hypothesis, syrupy** — the Python scientific and testing stack.
- **Pydantic, Lark, ruff, mypy, uv, pytest** — the type-safety, parsing, linting, and test toolchain.

Please cite the individual papers listed under *Pharmacometric References* when reporting APMODE-based analyses.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Technical architecture (v0.2) |
| [`docs/PRD_APMODE_v0.3.md`](docs/PRD_APMODE_v0.3.md) | Current PRD (v0.3, source of truth) |
| [`docs/FORMULAR.md`](docs/FORMULAR.md) | Formular DSL full reference |
| [`policies/*.json`](policies/) | Gate threshold policy files per lane |
| [`.claude/skills/apmode/SKILL.md`](.claude/skills/apmode/SKILL.md) | Claude Skill for LLM assistants (Claude Code, Codex, Gemini, Droid, …) — lane guidance, verified flag defaults, bundle artifacts, gotchas |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `apmode run <csv> --lane <lane>` | Full pipeline: ingest → profile → NCA → search → gates → bundle |
| `apmode explore <name-or-csv>` | Interactive wizard: step-by-step data exploration with optional pipeline launch |
| `apmode datasets [name]` | Browse or download 14 public PK datasets from nlmixr2data |
| `apmode inspect <bundle>` | Print summary of a reproducibility bundle |
| `apmode log <bundle> --top N` | Show top-N ranked candidates with parameter estimates |
| `apmode log <bundle> --failed` | List failed candidates with gate + reason |
| `apmode log <bundle> --gate gate1` | Per-check pass/fail details for a specific gate |
| `apmode diff <bundle-a> <bundle-b>` | Side-by-side comparison of evidence, rankings, gate pass rates |
| `apmode trace <bundle>` | Agentic iteration traces: summary, `--iteration N`, `--cost`, `--json` |
| `apmode lineage <bundle> <candidate>` | Transform chain from root to candidate with gate status |
| `apmode graph <bundle>` | Search DAG visualization: `--format tree/dot/mermaid/json`, `--converged` |
| `apmode validate <bundle>` | Validate bundle completeness |
| `apmode --version` | Print version |

### Key Options for `apmode run`

| Flag | Default | Description |
|------|---------|-------------|
| `--lane` | `submission` | Operating lane: `submission`, `discovery`, `optimization` |
| `--seed` | `753849` | Root random seed for reproducibility |
| `--parallel-models N` / `-j N` | `1` | Max concurrent model evaluations (R subprocesses). Higher values speed up search but use more memory. |
| `--timeout` | `600` | Backend timeout in seconds |
| `--agentic/--no-agentic` | **off** | Enable the agentic LLM backend (discovery/optimization lanes). OFF by default because the loop ships aggregated diagnostics to a third-party LLM provider; pass `--agentic` to opt in. |
| `--provider` | `anthropic` | LLM provider: `anthropic`, `openai`, `gemini`, `ollama`, `openrouter` |
| `--policy` | auto | Gate policy JSON file (falls back to `policies/<lane>.json`) |

**Exit codes:** `0` success, `1` input/validation error, `2` backend error, `130` user interrupt.

### Public Dataset Registry

14 datasets available via `apmode datasets`, including 5 real clinical datasets:

| Dataset | Subjects | Route | Elimination | Covariates |
|---------|----------|-------|-------------|------------|
| `theo_sd` | 12 | oral | linear | WT |
| `warfarin` | 32 | oral | linear | WT, age, sex |
| `mavoglurant` | 120 | oral | unknown | AGE, SEX, WT, HT |
| `pheno_sd` | 59 | IV | linear | WT, APGR |
| `nimoData` | 40 | IV infusion | unknown | WT |

Plus 9 simulated ground-truth datasets (1/2-cmt, oral/IV/infusion, linear/MM).

---

## Agentic LLM Backend (Phase 3)

The agentic backend is a **closed-loop model improvement system** where an LLM proposes typed PK model transforms based on diagnostic feedback, operating exclusively within the Formular DSL grammar.

> **Privacy:** The agentic backend is **off by default** (`--agentic` to enable). When enabled, aggregated fit diagnostics — but never per-subject data — are sent to the selected LLM provider. The allow-list gate in `diagnostic_summarizer.redact_for_llm()` is the single enforcement point; unknown fields fail closed.

### Operating Modes

The orchestrator runs the agentic stage **after** classical search, in two modes:

| Mode | Starting Spec | What the LLM Does |
|------|--------------|-------------------|
| **Refine** | Best classical candidate from search | Targeted improvement — add covariates, swap modules, adjust variability |
| **Independent** | Minimal 1-cmt oral spec | Build from scratch — LLM discovers structure through transforms |

In discovery/optimization lanes, the LLM can also propose `replace_with_node` transforms to introduce Neural ODE modules. All agentic candidates enter the same governance gate funnel as classical candidates.

### Iteration Loop

```
AgenticRunner.run(initial_spec, data)
 │
 FOR each iteration (max 25):
 │  1. EVALUATE  → inner_runner fits current spec (nlmixr2/JAX)
 │  2. SUMMARIZE → convergence, CWRES, shrinkage, VPC → markdown
 │  3. PROMPT    → system prompt + conversation history → LLM
 │  4. PARSE     → JSON extraction → typed FormularTransform objects
 │  5. VALIDATE  → precondition checks + post-apply semantic validation
 │  6. APPLY     → transforms applied sequentially, lineage DAG tracked
 │  7. TRACE     → input/output/meta/cached_response written per iteration
 │  8. FEEDBACK  → validation failures fed back to LLM for correction
 │
 RETURN best BackendResult by BIC across all iterations
```

### Available Transforms

The LLM cannot write raw code — it can only propose these 6 typed operations:

| Transform | Purpose | Example |
|-----------|---------|---------|
| `swap_module` | Replace absorption/distribution/elimination/observation | Linear → MichaelisMenten elimination |
| `add_covariate_link` | Add covariate effect (power/exponential/linear/categorical/maturation) | Allometric WT → CL |
| `adjust_variability` | Modify IIV structure (add/remove/upgrade_to_block) | Remove IIV on param with >30% shrinkage |
| `set_transit_n` | Change transit compartment count | Increase transit N for delayed absorption |
| `toggle_lag` | Enable/disable absorption lag time | Add tlag for delayed onset |
| `replace_with_node` | Swap to Neural ODE (discovery/optimization only) | NODE absorption with dim=4 |

### LLM Provider Support

| Provider | SDK | Auth | Cost |
|----------|-----|------|------|
| **Anthropic** | `anthropic.AsyncAnthropic` | `ANTHROPIC_API_KEY` | Per-token |
| **OpenAI** | `openai.AsyncOpenAI` | `OPENAI_API_KEY` | Per-token |
| **Google Gemini** | `google.genai.Client` | `GEMINI_API_KEY` | Per-token |
| **OpenRouter** | OpenAI-compatible | `OPENROUTER_API_KEY` | Per-token |
| **Ollama** | `ollama.AsyncClient` | None (local) | Free |
| **litellm** | `litellm.acompletion` | Per-provider | Fallback |

Install provider SDKs: `uv sync --extra llm`

### Reproducibility

- **temperature=0** enforced (non-zero raises `ValueError`)
- **Payload hashing** — SHA-256 of every request for audit
- **Cached responses** — `ReplayClient` replays from `agentic_trace/` without API calls
- **Model-version escrow** — deterministic fingerprint when available (Anthropic: `full`, others: `best-effort`)
- **Conversation history** — multi-turn context preserved across iterations

### Live Integration Tests

```bash
# Run live provider tests (requires API keys or local Ollama)
uv run pytest tests/integration/test_llm_providers_live.py -m live -v

# Tests skip gracefully on missing keys or billing/quota errors
```

---

## Known Limitations

- **Multi-dose**: ADDL/II expansion supported across all backends; SS (steady-state) pass-through for nlmixr2 only — Stan/NODE reject SS!=0
- **NODE infusions**: NODE backend rejects infusion data (RATE>0); use nlmixr2 for infusion dosing
- **NODE training**: Pooled population NLL (no per-subject RE); Laplace approximation deferred to Phase 3
- **NODE scaling**: Python-list subject loop (not vmap); scales to ~50 subjects, not 500+
- **Stan codegen**: Maturation covariate form not yet supported (raises `NotImplementedError`)
- **TMDD QSS**: Uses KD as approximation for KSS; when kint >> koff, this can overestimate complex formation
- **TimeVaryingElim**: Only `exponential` decay supported; `half_life` and `linear` rejected by validator
- **Context of use**: Orchestrator auto-generates COU for Gate 2.5; production use needs user-provided COU via CLI or config
- **Agentic LLM backend**: Requires funded API keys (Anthropic/OpenAI) or local Ollama with a chat-capable model (≥4B params recommended)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design rationale.

---

## Citation

If you use APMODE in your research, please cite:

```bibtex
@software{apmode2026,
  title        = {APMODE: Adaptive Pharmacokinetic Model Discovery Engine},
  author       = {Kornilov, Sergey A.},
  year         = {2026},
  url          = {https://github.com/biostochastics/apmode},
  license      = {GPL-2.0-or-later}
}
```

---

## License

Licensed under [GPL-2.0-or-later](LICENSE).

The primary engine is nlmixr2 (R), which is GPL-2 licensed. This license choice is deliberate and affects build structure from Phase 1.

---

<div align="center">

**[Quick Start](#quick-start)** &bull;
**[Formular DSL](#formular--the-pk-dsl)** &bull;
**[Architecture](#architecture)**

</div>
