<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Dataset Justifications — APMODE Benchmark Extensions

**Generated:** 2026-07-08 · Exa + Tavily research pass, then curl+R verification pass.
**Companion files:** `registry.yaml` (current in-tree), `candidates.yaml` (new proposals, v2 post-verification), `manifest.json` (machine-readable union), `fetch_ledger.jsonl` (source audit trail — 58 records).

This document explains **why each candidate dataset belongs in which APMODE lane / suite**, cross-referenced to `docs/PRD_APMODE_v0.3.md` §§ and `CLAUDE.md` architectural invariants.

## Verification pass results (2026-07-08)

| Outcome | Count | Detail |
|---|---|---|
| Live, license-verified | 8 | nlmixr2data (GPL-3), pmxNODE (GPL-3), pkdb code (MIT), Neural_PK (MIT), r-base (GPL-2), PK-DB API, Monolix NODE tutorial, arXiv 2602.03215 |
| Live, but **no LICENSE** → reference-only | 3 | Metrum Expo 1, Metrum Expo 4 (added), OSP observed-data |
| Live, unchecked license | 4 | saemix, npde, medicaldata, Hoffert 2024 (paywall) |
| **503 Service Unavailable** (transient?) | DDMoRe repository (all URLs; sitewide) — foundation home still 200 |
| **404 Not Found** → REMOVED | Metrum BUGSModelLibrary on Bitbucket → replaced by Torsten + Expo 4 |
| 403 bot-block (paper still resolvable) | MDPI Entropy dalbavancin paper |

**Row-count corrections found by local `Rscript` spot-check** (`nlmixr2data v2.0.9`, R 4.5.3):
- `warfarin`: registry.yaml says 519 rows → local install returns **515**
- `mavoglurant`: registry.yaml says 2346 rows → local install returns **2678**
- `pheno_sd`: registry.yaml says 155 rows / 59 subjects → local install returns **744 rows** (registry description matches the sparse NONMEM VII PHENO distribution, *not* `nlmixr2data::pheno_sd`; these are two different fixtures. Reconcile before Suite-B usage.)

**Corpus-size correction (PK-DB):**
- Prior claim: "~500 studies, all curated" (from paper snippet)
- Verified: **803 total studies**; sampled 200 → **15 marked `licence:open`**, 185 `licence:closed`. Extrapolated **redistributable open corpus ≈ 60 studies**. All 803 have `access:public` metadata (parameters, PMID, study design) — only ~60 have redistributable concentration-time data. Still highest-leverage single addition, but requires per-study license filtering in the fetcher.

## 0. Coverage matrix

| Lane (§3) | Suite | Purpose | Currently covered | Gaps closed by candidates.yaml |
|---|---|---|---|---|
| Submission | Suite A | Simulated ground-truth structural recovery | Oral_1CPT, mavoglurant | 1-cpt MM elim, 2-cpt oral, transit-comp, PKPD turnover (via `nlmixr2data_extended_grid`) |
| Submission | Suite B | Real-data classical anchors | theo_sd, warfarin, mavoglurant | Indometh (IV 2-cpt biexp), Monolix-warfarin joint (PKPD) |
| Discovery | Suite C | Literature-anchor fixtures | pheno_sd, opentci_propofol, gentamicin | DDMoRe repository index (100+ models), OSP DDI/Pediatrics |
| Discovery | Suite D (new) | NODE/hybrid benchmarks | — | pmxNODE examples, Monolix NODE zip, dalbavancin, latent-NODE tacrolimus |
| Optimization | LORO-CV | External-validation MIPD | mimic_vancomycin (Tier 2) | Hoffert 2024 tacrolimus multi-tool comparator |
| Evidence Manifest stress (§4.2.1) | any | Richness / covariate / absorption coverage variance | — | **PK-DB REST API (~500 curated studies)** — highest-leverage single addition |

## 1. Justifications per candidate

### 1.1 `nlmixr2data_extended_grid` — structural coverage
**Load-bearing PRD §:** §4.2.5 (DSL grammar: Absorption × Distribution × Elimination × Variability × Observation).

**Gap.** Current Suite A files (`a1_1cmt_oral_linear.csv` … `a8_1cmt_tvcl_covariate.csv`) plus `Oral_1CPT` cover 1-cpt oral + one 2-cpt NODE-absorption case + one TMDD. The DSL's Elimination axis (Linear / Michaelis-Menten / TMDD-QSS) is under-tested — `Bolus_1CPTMM`, `Oral_1CPTMM`, `Bolus_2CPTMM`, `Oral_2CPTMM`, `Infusion_1CPTMM`, `Infusion_2CPTMM` fill it, and `wbcSim` extends coverage to the PKPD turnover Observation branch.

**Why not simulate ourselves.** Because Schoemaker/Xiong/Wilkins/Laveille/Wang seeded the ACOP-2016 grid with published parameter distributions that a decade of pharmacometrics has calibrated against — reproducing them means comparability to a corpus of NLME papers, not a bespoke sim.

**Cost.** Zero — already installed via nlmixr2 dependency chain.

### 1.2 `saemix_theo_covariate` — covariate-discovery ground truth
**Load-bearing PRD §:** §4.2.5 (Covariate transforms in the DSL) + §10 open Q (covariate missingness strategy).

**Why unique.** `saemix::theo.saemix` is `Theoph` with a **simulated** `Sex` column added. This is exactly the setup you want for testing covariate discovery: a known synthetic covariate embedded in a familiar structural model. No FDA data restrictions, and if the discovery lane's covariate-search misses it, you have a defect not a data problem.

### 1.3 `npde_reference_sims` — VPC/NPE calibration reference
**Load-bearing CLAUDE.md invariant:** "Posterior-predictive diagnostics go through one canonical helper" — `predictive_summary.build_predictive_diagnostics`.

**Why unique.** Comets 2008 shipped 1000× posterior-predictive simulations of warfarin (base + covariate model). Your NPE / VPC-coverage calculations can be calibrated against these instead of internally-consistent random sims. If your VPC gate rejects a model that Comets 2008 accepts, that's a calibration bug, not model quality.

### 1.4 `r_base_datasets` + `medicaldata_indometh` — 2-cpt IV biexponential
**Load-bearing PRD §:** §4.2.5 (Distribution axis: 1-cpt vs 2-cpt IV).

**Gap.** The current Submission-lane real-data anchors (theo, warfarin, mavoglurant) are all **oral**. Indomethacin (Kwan 1976) is the canonical IV biexponential 2-cpt fixture: 6 subj, 11 samples each, clean geometry.

**Cost.** Zero — R core `datasets` is a transitive dependency.

### 1.5 `ddmore_repository_index` — Suite-C literature anchor pool ⚠ BLOCKED
**Load-bearing PRD §:** §4.3.1 (Gate 3 cross-paradigm ranking requires an anchor set).

**Status (2026-07-08):** `repository.ddmore.foundation` and legacy `repository.ddmore.eu` both return **503 Service Unavailable** across all URLs (root + individual model IDs). The foundation home page (`www.ddmore.foundation`) is 200 and still lists the repository as active, so this reads as a deployment outage rather than a sunset. **Do not promote to registry.yaml until the repository returns 200.** Existing `ddmore_gentamicin` entry in registry.yaml is affected by the same outage.

**Why load-bearing (unchanged).** Every DDMoRe entry ships with (a) real or simulated data with dependent-variable observations, (b) a canonical NONMEM/Monolix/WinBUGS run for byte-checkable ground truth, (c) qualification metadata (see D1_21 procedure document). This is the *only* public resource that gives you a scientific-community-validated NLME baseline per model — which is precisely what your cross-paradigm ranker needs against the classical arm.

**Prioritise these IDs first** (highest APMODE relevance):
- **DDMODEL00000238** (gentamicin neonatal IOV) — already in registry, keep.
- **DDMODEL00000248** (preterm phenobarbital) — extends pheno_sd to preterm neonates with IOV.
- **DDMODEL00000003** (Hamren tesaglitazar) — DDMoRe qualification exemplar; MDL+PharmML+NONMEM all consistent.
- **DDMODEL00000103** (Trefz PKU/Kuvan turnover-KPD) — PK/PD with turnover (Elimination axis + Observation axis).
- **DDMODEL00000130** (Karaiskos colistin) — non-MDL executable original code (Scenario 4) → tests your `bayesian_stan` backend against a NONMEM baseline.
- **DDMODEL00000243** (TTE) + **DDMODEL00000247** (IRT) — discrete-data observation models for eventual DSL extension.

**Why not just fetch everything.** 123 models is unwieldy; the above 6 exercise every DSL axis without redundancy.

### 1.6 `metrum_merge_expo1_nonmem` + `metrum_merge_expo4_torsten` — reference NONMEM + Torsten workflows
**Load-bearing CLAUDE.md invariants:** "Primary engine is nlmixr2 (R). NONMEM and Pumas are optional adapters." + "Torsten backend has the same predictive-diagnostics contract as nlmixr2."

**Licensing caveat (post-verification).** BOTH Metrum expo repositories have **NO LICENSE file at HEAD** (GitHub API returns `license: null`; direct raw fetch of `LICENSE`/`LICENSE.md` returns 404). Absence of a license means public-viewable but **not redistributable**. Both entries are downgraded to `tier: reference_only` — reproducing methodology + patterns is fine, but no code or data can be vendored into APMODE without explicit permission from MetrumRG.

**Why still valuable.** MeRGE Expo 1 (NONMEM/FOCE, pkgr + renv + MPN) is the reference workflow that regulatory-adjacent modelers already know. MeRGE Expo 4 (bbr.bayes + Stan/Torsten, `ppkexpo1` centered → `ppkexpo2` NCP reparameterisation → `ppkexpo3` allometric → `ppkexpo4` eGFR/age/albumin covariates on CL) is a hand-curated model progression that stresses APMODE's Gate-3 discrimination. Reproducing Expo 1 headline PopPK model with the classical backend and Expo 4 with the `bayesian_stan` backend is a defensible acceptance test per lane even under reference-only licensing.

**Note.** `metrum_merge_expo4_torsten` also **replaces the dead `bugs_model_library`** entry (Bitbucket 404) since it's the same author group's active Bayesian PKPD workflow.

### 1.7 `osp_observed_data` — DDI + pediatric held-out ⚠ downgraded
**Load-bearing PRD §:** §4.2.1 (Evidence Manifest richness classification) + Discovery-lane covariate handling.

**Licensing caveat (post-verification).** OSP `Database-for-observed-data` repository has **NO LICENSE file at HEAD** (verified via GitHub API + raw `LICENSE`/`LICENSE.md` fetches, both 404). My earlier claim of "LGPL-3.0" was wrong — LGPL-3 is the OSP Suite *software* license, not the observed-data database's. Entry downgraded to `tier: reference_only`.

**Why still valuable.** Only curated observed-data set explicitly built for **qualification plans** (Bayer OSP). Even reference-only use is valuable: `ObsDataPK_OSP.xlsx`, `DDI.csv`, `Pediatrics.csv` schema is a template for building analogous Suite-C fixtures from PK-DB + published parameters.

### 1.8 `pkdb_api` — Evidence Manifest stress at scale ⭐ still highest leverage (revised)
**Load-bearing PRD §:** §4.2.1 (Data Profiler / Evidence Manifest).

**Post-verification counts.** 803 total studies (not ~500 as I initially claimed from the paper snippet). In a 200-study sample: **15 marked `licence:open`, 185 `licence:closed`** — so the openly redistributable corpus is ≈ **60 studies**, not 500. All 803 have `access:public` metadata (parameters, PMID, study design). Software license on the code repository is **MIT** (verified via GitHub API), not LGPL-3 as I initially stated (LGPL-3 in the paper refers to the running PK-DB service's software).

**Why still load-bearing.** Even at 60 studies, this is the only large open PK database with (a) individual-subject concentration-time curves, (b) ChEBI/ncit/hp/doid/mondo-annotated covariates (age, weight, smoking, CYP genotype, oral contraceptives, co-medication), (c) REST API access. One REST integration adds ~60 studies of heterogeneous richness — exactly the input distribution your `richness_category` / `absorption_coverage` classifiers need to be tested against.

**API pagination gotcha.** The REST API wraps results in `response.data.data[]` with `response.data.count` for total — not the flat DRF pagination the swagger might suggest. Fetcher must handle this specifically.

**Cost estimate.** ~1 dev-day for the async fetch + `licence:open` filter + Pandera schema mapping + prepare.py; ~1 dev-day for Gate-1 sanity policy tuning against the resulting variance. Net: 2 days for ~10× the current Suite-B breadth.

**Why not a single-file dump.** PK-DB studies vary in richness, covariate presence, error models, and units. Batch-download would defeat the point — the Evidence Manifest *should* see this heterogeneity live.

### 1.9 `pmxnode_examples` + `monolix_node_case_study` — Phase-2 NODE contract validation
**Load-bearing PRD §:** §4.2.4 R6 (Bräm-style low-dimensional NODE with random effects on input-layer weights).

**Why unique.** Bräm 2025 (`doi:10.1002/psp4.13265`) is **the paper your NODE backend is architecturally derived from**. `pmxNODE` (`github.com/braemd/pmxnode`) ships the exact reference implementation in Monolix + NONMEM + nlmixr2. Fitting the same demo datasets with APMODE's NODE runner and reporting comparable RMSEs is the strongest possible claim for the Phase-2 deliverable.

**Two datasets, not one.** `pmxNODE` gives you the R-side examples; the Monolix `NODE_example.zip` is the smallest self-contained fixture for smoke-testing without pmxNODE's dependency chain.

### 1.10 `dalbavancin_node` — sparse-data NODE Gate 2 test
**Load-bearing CLAUDE.md invariant:** "`richness_category = sparse` + inadequate absorption coverage → NODE backends receive `data_insufficient` flag".

**Why unique.** Ranieri 2025 is the first published NODE eval on real-world sparse clinical data. Whether APMODE's Gate 2 fires `data_insufficient` on this dataset — and whether the NODE runner degrades gracefully when it doesn't — is a directly testable behavioural contract.

### 1.11 `latent_node_tacrolimus` + `hoffert_tacrolimus_benchmark_2024` — Optimization-lane LORO-CV
**Load-bearing PRD §:** §3 Translational-Optimization lane.

**Why paired.** Latent NODE (arXiv 2602.03215) provides a NODE-based MIPD comparator; Hoffert 2024 provides the 7-tool systematic benchmark protocol (RxStudio / PrecisePK / InsightRx / MwPharm / DoseMeRX / BestDose / ISBA). Optimization-lane LORO-CV output should be plot-comparable against Hoffert's headline figures.

**Data caveat.** Neither ships raw tacrolimus data. The reproduction path is: (a) implement published PopPK models from the systematic review, (b) simulate concentration-time under those, (c) treat as ground-truth for APMODE's Optimization lane. Not ideal, but the only defensible open-data route.

## 2. Deliberate exclusions

| Excluded | Reason | Consequence |
|---|---|---|
| **T-DM1 (Lu 2021 / Bräm 2025 cross-regimen)** | Genentech restricted patient data | Cite methodology only; use `Lu 2014` (open-access CC-BY) parameters for prior elicitation instead |
| **DrugBank PK endpoints** | Parameter-level, no curves | Useful for `initial_estimates` cascade priors only, not model fitting |
| **Old "PK/DB" (~1200 compounds)** | Superseded by PK-DB, no active maint | Ignore |

## 3. Recommended promotion sequence (revised post-verification)

Rough cost / benefit ranking for merging candidates → registry.yaml, adjusted for what verification actually found:

1. **`nlmixr2data_extended_grid`** (0 dev-days, unblocks DSL axis coverage). Verified: 25 datasets present in local install; row counts confirmed for 18. Also **reconcile row-count discrepancies** in existing `registry.yaml` (`warfarin` 519→515, `mavoglurant` 2346→2678, `pheno_sd` 155→744 — the last is likely a mislabelled fixture pointing at NONMEM VII PHENO not `nlmixr2data::pheno_sd`).
2. **`r_base_datasets`** for Indometh IV 2-cpt (0 dev-days). Verified: R 4.5.3 local install.
3. **`pkdb_api`** (2 dev-days, ~10× Suite-B breadth — revised from 100× after finding only ~7.5% of studies openly licensed). Verified live. Fetcher must filter for `licence:open` and handle the `response.data.data[]` wrap.
4. **`pmxnode_examples`** (1 dev-day, unblocks NODE backend claim of parity with Bräm 2025). Verified: GPL-3.0 code license, r-universe + github both live.
5. **`ddmore_repository_index`** targeted 7 IDs (2 dev-days) — **BLOCKED** at 503 as of 2026-07-08. Do not begin until repository returns 200. Includes `ddmore_gentamicin` already in registry.yaml — that entry is currently unfetchable.
6. **`metrum_merge_expo4_torsten`** (1 dev-day, `bayesian_stan` reference test). Live but reference-only; reproduce methodology, do not vendor.
7. **`osp_observed_data`** (1 dev-day) — Phase 3 DDI extension; reference-only.
8. **`saemix_theo_covariate`**, **`npde_reference_sims`** (0.5 dev-days each) — calibration references.
9. **`metrum_merge_expo1_nonmem`** — Phase 3 cross-tool NONMEM acceptance test; reference-only.

**No longer in list:**
- ~~`bugs_model_library`~~ — REMOVED (Bitbucket 404). Replaced by `metrum_merge_expo4_torsten` + `metrum_torsten_library`.

## 4. Provenance

Every claim above traces to a specific record in `fetch_ledger.jsonl`. Grep by `source_url`:

```bash
grep 'pk-db.com' benchmarks/datasets/fetch_ledger.jsonl | jq
grep 'doi.*10.1002/psp4.13265' benchmarks/datasets/fetch_ledger.jsonl | jq
```

Confidence field on each ledger entry: `high` = extracted from the primary source (paper/repo), `medium` = extracted from a snippet or mirror, `low` = not used.

Retrieval commands in `manifest.json` have `verified: true|false`; `false` means the command was inferred from the source documentation but not executed. Verify before promotion.
