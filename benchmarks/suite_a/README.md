<!-- SPDX-License-Identifier: GPL-2.0-or-later -->
# Benchmark Suite A: Synthetic Recovery

Simulated PopPK datasets with known ground truth for structural-recovery
benchmarking (PRD v0.3 §5).

## Usage

```bash
# Single replicate (default, ~7 s):
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

Each `<scenario>.csv` has a companion `<scenario>_eta.csv` containing the
per-subject η draws for η-recovery diagnostics.

## Columns

- **`NMID`** — Subject ID (1..n)
- **`TIME`** — Time since dose (h)
- **`DV`** — Dependent variable = concentration (mg/L) for observation rows, 0 for dose rows
- **`MDV`** — Missing-DV flag (1 for dose or NA observation, 0 for observed)
- **`EVID`** — 0 for observation, 1 for dose
- **`AMT`** — Dose amount (mg) or 0
- **`CMT`** — Compartment: 1 for dose (depot for oral, central for IV/SC); 2 for observation in oral scenarios; 1 for observation in A2 (IV) and A5 (SC, central-tracking)
- **`BLQ`** — Below-LLOQ flag (1 = below-LLOQ, M3-imputed to `LLOQ/2`)
- **Covariates** — `WT` (kg, truncated at ≥ 40), `SEX` (0=F, 1=M), plus per-scenario extras: `RENAL` (A6), `CRCL` (A8)

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
  covariate injection (A6, A8).

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
