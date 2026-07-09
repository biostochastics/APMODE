<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Benchmark Suite A: Synthetic Recovery

Simulated PopPK datasets with known ground truth for structural-recovery
benchmarking (PRD v0.3 §5).

## Usage

```bash
# Single replicate (default, ~15 s):
Rscript benchmarks/suite_a/simulate_all.R [output_dir]

# 30 replicates for recovery-rate confidence intervals (~4 min):
Rscript benchmarks/suite_a/simulate_all.R benchmarks/suite_a 30
```

Requires: R 4.4+, `rxode2`, `lotri`, `jsonlite`.

## Scenarios

| File | True Model | Key Test |
|---|---|---|
| `a1_1cmt_oral_linear.csv` | 1-cmt oral, first-order absorption, linear elimination | Structure identification |
| `a2_2cmt_iv_parallel_mm.csv` | 2-cmt IV, parallel linear + Michaelis-Menten elimination | Compartment count + nonlinear CL |
| `a3_transit_1cmt_linear.csv` | Transit (n=3, ktr=2), 1-cmt, linear elimination | Transit chain detection |
| `a4_1cmt_oral_mm.csv` | 1-cmt oral, MM elimination | Nonlinear clearance detection |
| `a5_tmdd_qss.csv` | TMDD QSS SC mAb | TMDD vs 2-cmt confusion |
| `a6_1cmt_covariates.csv` | 1-cmt oral + WT^0.75 allometry + RENAL categorical + SEX exp-effect (n=100) | Covariate structure recovery |
| `a7_2cmt_node_absorption.csv` | 2-cmt + saturable MM absorption | NODE shape recovery + surrogate fidelity |
| `a8_1cmt_autoind_covariate.csv` | 1-cmt oral + CRCL^0.75 + monotonic autoinduction `exp(−0.15·t/24)` (n=100) | Time-varying CL + covariate recovery |
| `a9_erlang_absorption.csv` | Erlang (n=3 explicit chain, ktr=2), 1-cmt, linear elimination | Erlang vs. Transit chain discrimination |
| `a10_parallel_first_order.csv` | Two simultaneous first-order depots (ka1=2, ka2=0.3, frac=0.6) | Parallel-absorption structure recovery |
| `a11_mixed_first_zero_absorption.csv` | First-order + zero-order depots (ka=1.5, dur=3, frac=0.55) | Mixed-mechanism absorption recovery |
| `a12_zero_order_absorption.csv` | Standalone zero-order absorption (dur=4, modeled infusion) | Zero-order-only structure recovery |
| `a13_sum_ig_absorption.csv` | Sum-of-2-Inverse-Gaussians (MT₁=0.5, MT₂=3.5), 1-cmt, disposition fixed | SumIG(k=2) shape recovery |
| `a14_3cmt_iv_bolus.csv` | 3-cmt IV bolus (V1=5, V2=15, V3=100, Q2=8, Q3=1.5) | Deep-compartment / terminal-phase recovery |
| `a15_tmdd_core_dosearms.csv` | TMDD full binding model (Mager & Jusko 2001), 3 dose arms (10/60/400 mg, n=60) | kon/koff identifiability via dose-ranging |
| `a16_time_varying_elim_unconfounded.csv` | 1-cmt oral, `CL(t)=CL0·exp(−0.0015·t)`, QD×14d, no covariate | Unconfounded time-varying CL recovery |
| `a17_block_iiv_additive_error.csv` | 1-cmt oral, block IIV (CL-V corr≈0.4) + additive residual error | Correlated-BSV + additive-error recovery |
| `a18_iov_occasions.csv` | 1-cmt oral, IOV on ka across 3 occasions (week-spaced, `occ` column) | Inter-occasion variability recovery |
| `a19_maturation_covariate.csv` | 1-cmt oral, Hill/sigmoid maturation covariate on CL (PNA, TM50=15, hill=3, n=100) | Maturation-form covariate recovery |
| `a20_1cmt_oral_blq_elevated_lloq.csv` | 1-cmt oral, elevated LLOQ=0.12 (~19% BLQ) | BLQ_M3 vs. BLQ_M4 likelihood comparison (paired A20a/A20b DSLSpecs) |
| `a21_tmdd_qss_multi_analyte.csv` | TMDD QSS, dual endpoint (DVID=1 free drug, DVID=2 total target) | Multi-analyte observation-model recovery |

Each `<scenario>.csv` has a companion `<scenario>_eta.csv` containing the
per-subject η draws for η-recovery diagnostics. A20a/A20b share one CSV
(`a20_1cmt_oral_blq_elevated_lloq.csv`) — see `suite_a.py::scenario_a20a`
docstring; they are a paired benchmark unit, not two independent recovery
tests, since both are scored against identical ground truth.

## Columns

- **`NMID`** — Subject ID (1..n)
- **`TIME`** — Time since dose (h)
- **`DV`** — Dependent variable = concentration (mg/L) for observation rows, 0 for dose rows
- **`MDV`** — Missing-DV flag (1 for dose or NA observation, 0 for observed)
- **`EVID`** — 0 for observation, 1 for dose
- **`AMT`** — Dose amount (mg) or 0
- **`CMT`** — Compartment: 1 for dose (depot for oral, central for IV/SC); 2 for observation in oral scenarios; 1 for observation in A2 (IV) and A5 (SC, central-tracking)
- **`BLQ`** — Below-LLOQ flag (1 = below-LLOQ, M3-imputed to `LLOQ/2`)
- **Covariates** — `WT` (kg, truncated at ≥ 40), `SEX` (0=F, 1=M), plus per-scenario extras: `RENAL` (A6), `CRCL` (A8), `PNA` (A19, weeks)
- **`OCC`** — Occasion index (A18 only, 1..3), binds `IOV` etas to dosing occasions
- **`DVID`** — Endpoint id (A21 only: 0=dose row, 1=free drug, 2=total target), multi-analyte observation routing

## Reference parameters

`reference_params.json` contains true parameter values, ω (inter-individual
variability), σ (residual error, **standard deviations on data scale — not
variances**), covariate metadata, and, for A8, the analytical time-averaged
CL over 0–48 h alongside the expected static-target bias.

## Design notes

- σ is standard deviation on the data scale. NONMEM's `SIGMA` block uses
  variance; square before comparing.
- WT is generated from a truncated normal (`pmax(rnorm(70, 15), 40)`) so
  no subject has a physically impossible mass.
- Negative simulated concentrations become `NA` with `MDV=1`; observations
  below LLOQ (when set) use M3-style imputation (`DV = LLOQ/2, BLQ=1`).
- Seeds are per-scenario, per-replicate: `BASE_SEED + scenario_idx × 10000 +
  replicate_idx × 100`; scenarios never depend on each other's stream
  state.
- The event table passed to `rxSolve` is a single-subject template that
  is replicated by `nSub`. Passing an N-subject event table *and* `nSub=N`
  causes N×N subject replication and physically impossible concentrations;
  the `build_et_icov()` helper is used only for scenarios with `iCov=`
  covariate injection (A6, A8, A19) or per-subject-varying dose amounts
  (`build_et_icov_dosearms()`, A15). `build_et_multi()` builds a
  single-subject template with more than one dose row (two parallel
  depots, or a multi-dose regimen) for A10/A11/A16; `build_nonmem_output`'s
  dose-replication branch discriminates single-subject-template vs.
  already-expanded-iCov event tables by the presence of an `id` column,
  not row count, so it generalizes to any number of dose rows per subject.
- Zero-order / modeled-duration absorption (A11, A12) requires the dosing
  record to set `RATE = -2` — a plain `AMT` record with no `RATE` is
  bolus and silently ignores the model's `dur(<cmt>) <- ...` statement.
- **SumIG (A13) ground-truth caveat**: the nlmixr2 emitter's own SumIG ODE
  (`src/apmode/dsl/nlmixr2_emitter.py`) references the reserved
  event-column variable `amt` as if it persisted across the whole
  simulation, but rxode2 5.0.2 only exposes `amt` on the exact dosing row
  (`NA` at every later observation time) — verified empirically. A literal
  port of the emitter's expression yields `NA` concentrations at every
  point but the dose. `sim_A13` in `simulate_all.R` substitutes an
  explicit `DOSE` constant instead (valid since v0.7 SumIG is single-dose
  only, so the dose amount is a compile-time constant) — this changes only
  how the dose scalar is threaded into the closed-form input function, not
  the structural form itself. The emitter's `amt`-based code path likely
  needs the same fix before a live SumIG fit would work; out of scope for
  this benchmark addition. Separately, identifiers may not start with an
  underscore in rxode2 model blocks (`_t_safe` errors with "unexpected
  symbol"; the emitter's own `_t_safe` variable name would hit this too),
  so the ground-truth R model uses `t_safe` instead.
- rxode2's nested-omega IOV mechanism (A18) requires the occasion-indexing
  event-table column to be literally named `occ` (lowercase) — `OCC`
  errors with `"could not find 'occ' in data"`. The simulator uses `occ`
  internally and renames to `OCC` only in the emitted CSV, matching
  `OccasionByDoseEpoch(column="OCC")` in the DSLSpec.

## References

- Boeckmann, Sheiner & Beal (1994). *NONMEM Users Guide.* NONMEM Project
  Group, UCSF.
- Comets, Brendel & Mentré (2008). *Comp Meth Prog Biomed* 90:154.
  [doi:10.1016/j.cmpb.2007.12.002](https://doi.org/10.1016/j.cmpb.2007.12.002)
- Gibiansky et al. (2008). *J Pharmacokinet Pharmacodyn* 35:573.
  [doi:10.1007/s10928-008-9102-8](https://doi.org/10.1007/s10928-008-9102-8)
- Duvnjak et al. (2024). *CPT:PSP*.
  [doi:10.1002/psp4.13213](https://doi.org/10.1002/psp4.13213)
- Richardson et al. (2025). *Commun Med*.
  [doi:10.1038/s43856-025-01054-8](https://doi.org/10.1038/s43856-025-01054-8)
