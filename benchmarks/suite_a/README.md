# Benchmark Suite A: Synthetic Recovery

Simulated datasets with known ground truth (PRD §5).

## Usage

```bash
Rscript benchmarks/suite_a/simulate_all.R [output_dir]
```

Requires: R 4.4+, rxode2, jsonlite, lotri

## Scenarios

| File | True Model | Key Test |
|------|-----------|----------|
| a1_1cmt_oral_linear.csv | 1-cmt oral, first-order abs, linear elim | Structure identification |
| a2_2cmt_iv_parallel_mm.csv | 2-cmt IV, parallel linear+MM elim | Compartment count + nonlinear CL |
| a3_transit_1cmt_linear.csv | Transit (n=3), 1-cmt, linear elim | Transit chain detection |
| a4_1cmt_oral_mm.csv | 1-cmt oral, MM elimination | Nonlinear clearance detection |
| a5_tmdd_qss.csv | TMDD quasi-steady-state (SC mAb) | TMDD vs. 2-cmt confusion |
| a6_1cmt_covariates.csv | 1-cmt oral + allometric WT + renal covariate | Covariate structure recovery |
| a7_2cmt_node_absorption.csv | 2-cmt + NODE saturable absorption | NODE shape recovery + surrogate fidelity |
| a8_1cmt_tvcl_covariate.csv | 1-cmt oral + time-varying CL + CRCL covariate | Time-varying clearance + covariate recovery |

## Reference Parameters

`reference_params.json` contains true parameter values, omega (IIV), and sigma for each scenario.
