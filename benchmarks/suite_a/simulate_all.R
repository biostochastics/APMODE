# SPDX-License-Identifier: GPL-2.0-or-later
# Benchmark Suite A: Simulated ground-truth PopPK datasets for A1-A8 scenarios.
#
# Usage:
#   Rscript simulate_all.R [output_dir] [n_replicates]
#
# When n_replicates > 1, output files are suffixed `_repNN`. Every
# `<scenario>[_repNN].csv` has a matching `<scenario>[_repNN]_eta.csv`
# with the per-subject eta draws for eta-recovery diagnostics.
#
# Conventions:
#   * sigma_prop / sigma_add are STANDARD DEVIATIONS on the data scale
#     (NONMEM's SIGMA is variance; square before comparing).
#   * Seeds are per-scenario, per-replicate: BASE_SEED + scenario_idx *
#     10000 + replicate_idx * 100 + salt (no cross-scenario stream coupling).
#   * The event table passed to rxSolve is a single-subject template that
#     rxSolve replicates via nSub. iCov scenarios (A6, A8) use an
#     expanded template with matching `id` column via build_et_icov().
#
# References:
#   PRD v0.3 §5; Boeckmann/Sheiner/Beal (1994) NONMEM Users Guide;
#   Gibiansky et al. (2008) J Pharmacokinet Pharmacodyn 35:573 (TMDD QSS);
#   Comets/Brendel/Mentré (2008) CMPB 90:154 (npde);
#   Duvnjak et al. (2024) CPT:PSP doi:10.1002/psp4.13213.

suppressMessages(library(rxode2))
suppressMessages(library(lotri))

# ============================================================
# Configuration
# ============================================================

args        <- commandArgs(trailingOnly = TRUE)
output_dir  <- if (length(args) >= 1) args[1] else "benchmarks/suite_a"
N_REPS      <- if (length(args) >= 2) as.integer(args[2]) else 1L
BASE_SEED   <- 20260413L

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# scenario_id → (name, n_subjects, lloq for BLQ; NULL = no BLQ)
scenario_config <- list(
  A1 = list(name = "a1_1cmt_oral_linear",       n = 50,  lloq = NULL,   idx = 1L),
  A2 = list(name = "a2_2cmt_iv_parallel_mm",    n = 50,  lloq = 0.02,   idx = 2L),
  A3 = list(name = "a3_transit_1cmt_linear",    n = 50,  lloq = NULL,   idx = 3L),
  A4 = list(name = "a4_1cmt_oral_mm",           n = 50,  lloq = 0.02,   idx = 4L),
  A5 = list(name = "a5_tmdd_qss",               n = 50,  lloq = 0.01,   idx = 5L),
  A6 = list(name = "a6_1cmt_covariates",        n = 100, lloq = NULL,   idx = 6L),
  A7 = list(name = "a7_2cmt_node_absorption",   n = 50,  lloq = NULL,   idx = 7L),
  A8 = list(name = "a8_1cmt_autoind_covariate", n = 100, lloq = NULL,   idx = 8L),
  A9 = list(name = "a9_erlang_absorption",          n = 50,  lloq = NULL, idx = 9L),
  A10 = list(name = "a10_parallel_first_order",      n = 50,  lloq = NULL, idx = 10L),
  A11 = list(name = "a11_mixed_first_zero_absorption", n = 50, lloq = NULL, idx = 11L),
  A12 = list(name = "a12_zero_order_absorption",     n = 50,  lloq = NULL, idx = 12L),
  A13 = list(name = "a13_sum_ig_absorption",         n = 80,  lloq = NULL, idx = 13L),
  A14 = list(name = "a14_3cmt_iv_bolus",             n = 50,  lloq = NULL, idx = 14L),
  A15 = list(name = "a15_tmdd_core_dosearms",        n = 60,  lloq = NULL, idx = 15L),
  A16 = list(name = "a16_time_varying_elim_unconfounded", n = 60, lloq = NULL, idx = 16L),
  A17 = list(name = "a17_block_iiv_additive_error",  n = 80,  lloq = NULL, idx = 17L),
  A18 = list(name = "a18_iov_occasions",             n = 60,  lloq = NULL, idx = 18L),
  A19 = list(name = "a19_maturation_covariate",      n = 100, lloq = NULL, idx = 19L),
  A20 = list(name = "a20_1cmt_oral_blq_elevated_lloq", n = 60, lloq = 0.12, idx = 20L),
  A21 = list(name = "a21_tmdd_qss_multi_analyte",    n = 60,  lloq = NULL, idx = 21L)
)

# ============================================================
# Helpers
# ============================================================

# Reproducible covariate generator; truncates WT at 40 kg (adult clinical floor).
generate_covariates <- function(n, seed, include_sex = TRUE,
                                include_renal = FALSE, include_crcl = FALSE) {
  set.seed(seed)
  wt <- pmax(rnorm(n, mean = 70, sd = 15), 40)
  cov_df <- data.frame(id = seq_len(n), WT = wt)
  if (include_sex)   cov_df$SEX   <- sample(c(0L, 1L), n, replace = TRUE)  # 0=F, 1=M
  if (include_renal) cov_df$RENAL <- as.integer(runif(n) < 0.3)
  if (include_crcl)  cov_df$CRCL  <- runif(n, min = 30, max = 150)
  cov_df
}

# Build NONMEM-format event table for a SINGLE subject template.
# rxSolve(mod, et, nSub=n, omega=...) replicates this template n times with
# sim.id 1..n. Do NOT pre-expand the event table across n subjects and then
# also pass nSub=n — rxode2 will multiply them (n×n simulations), producing
# physically impossible concentrations.
#
# `rate`: optional NONMEM RATE value for the dose row. rxode2's modeled
# infusion-duration mechanism (`dur(<cmt>) <- ...` in the model block, see
# A12/A11) only activates when the dosing record sets RATE = -2 ("duration
# is modeled, read dur(cmt) from the model") — a plain AMT record with no
# RATE is treated as an ordinary bolus and dur(cmt) is silently ignored.
# Defaults to NA (no RATE column emitted) so every pre-existing A1-A8 call
# site is byte-for-byte unaffected.
build_et <- function(n_id, dose_amt, times, dose_cmt, rate = NA_real_) {
  n_rows <- length(times) + 1L
  et <- data.frame(
    NMID = rep(0L, n_rows),                   # placeholder; replaced post-sim
    TIME = c(0, times),
    DV   = c(0, rep(NA_real_, length(times))),
    MDV  = c(1L, rep(0L, length(times))),
    EVID = c(1L, rep(0L, length(times))),
    AMT  = c(dose_amt, rep(0, length(times))),
    CMT  = rep(dose_cmt, n_rows)
  )
  if (!is.na(rate)) et$RATE <- c(rate, rep(0, length(times)))
  et
}

# Build a single-subject template with MULTIPLE dose rows (e.g. two parallel
# depots dosed simultaneously, or a multi-dose QD regimen) interleaved with
# observation rows, sorted by TIME (dose rows first on ties so build_
# nonmem_output's -EVID sort tiebreak convention is honored downstream too).
# `dose_df` needs columns TIME/AMT/CMT and may optionally carry RATE (see
# build_et's `rate` param docs above for when RATE=-2 is required).
build_et_multi <- function(dose_df, obs_times, obs_cmt) {
  if (!("RATE" %in% names(dose_df))) dose_df$RATE <- 0
  dose_rows <- data.frame(
    NMID = 0L, TIME = dose_df$TIME, DV = 0, MDV = 1L, EVID = 1L,
    AMT = dose_df$AMT, CMT = dose_df$CMT, RATE = dose_df$RATE
  )
  obs_rows <- data.frame(
    NMID = 0L, TIME = obs_times, DV = NA_real_, MDV = 0L, EVID = 0L,
    AMT = 0, CMT = obs_cmt, RATE = 0
  )
  et <- rbind(dose_rows, obs_rows)
  et[order(et$TIME, -et$EVID), ]
}

# For scenarios A6/A8 that use iCov, we DO need an expanded event table with
# `id` column (rxode2 iCov requires per-id events). This helper is scenario-
# specific and matches rxode2's expectation.
build_et_icov <- function(n_id, dose_amt, times, dose_cmt) {
  rows <- vector("list", n_id * (length(times) + 1L))
  k <- 1L
  for (id in seq_len(n_id)) {
    rows[[k]] <- data.frame(
      id = id, NMID = id, TIME = 0, DV = 0, MDV = 1L, EVID = 1L,
      AMT = dose_amt, CMT = dose_cmt
    )
    k <- k + 1L
    for (t in times) {
      rows[[k]] <- data.frame(
        id = id, NMID = id, TIME = t, DV = NA_real_, MDV = 0L, EVID = 0L,
        AMT = 0, CMT = dose_cmt
      )
      k <- k + 1L
    }
  }
  do.call(rbind, rows)
}

# Extract per-subject ETAs from an rxSolve object as a data.frame.
extract_etas <- function(sim, eta_names) {
  # rxSolve stores per-simulation parameter draws in $params
  p <- tryCatch(sim$params, error = function(e) NULL)
  if (is.null(p)) return(data.frame(NMID = integer(0)))
  id_col <- if ("sim.id" %in% names(p)) "sim.id" else if ("id" %in% names(p)) "id" else NULL
  if (is.null(id_col)) return(data.frame(NMID = integer(0)))
  eta_df <- data.frame(NMID = p[[id_col]])
  for (nm in eta_names) if (nm %in% names(p)) eta_df[[nm]] <- p[[nm]]
  eta_df
}

# Convert rxSolve output → NONMEM-style long-format with residual error,
# BLQ handling (M3), ETA preservation, and proper CMT labeling.
# `et` may be either the single-subject template (for nSub-replicated sims)
# or the expanded iCov event table (for A6/A8). Both are handled.
build_nonmem_output <- function(sim, et, n_id, covs, sigma_prop_sd,
                                sigma_add_sd = 0, lloq = NULL,
                                central_cmt = 2L, blq_seed = NULL) {
  if (!is.null(blq_seed)) set.seed(blq_seed)

  obs_times <- sort(unique(et$TIME[et$EVID == 0]))
  id_col <- if ("sim.id" %in% names(sim)) sim$sim.id else sim$id

  sim_df <- data.frame(NMID = as.integer(id_col), TIME = sim$time, CP = sim$cp)
  sim_df <- sim_df[round(sim_df$TIME, 6) %in% round(obs_times, 6), ]

  # One row per (subject, obs_time)
  obs_list <- vector("list", length(unique(sim_df$NMID)))
  k <- 1L
  for (id in sort(unique(sim_df$NMID))) {
    sub <- sim_df[sim_df$NMID == id, , drop = FALSE]
    sub <- sub[!duplicated(round(sub$TIME, 6)), , drop = FALSE]
    obs_list[[k]] <- sub
    k <- k + 1L
  }
  obs_df <- do.call(rbind, obs_list)

  # Residual error — sigma_prop_sd and sigma_add_sd are SDs on data scale
  n_obs <- nrow(obs_df)
  obs_df$DV <- obs_df$CP * (1 + rnorm(n_obs, 0, sigma_prop_sd)) +
               rnorm(n_obs, 0, sigma_add_sd)

  # BLQ handling
  blq_flag <- rep(0L, n_obs)
  if (!is.null(lloq)) {
    below <- obs_df$DV < lloq
    obs_df$DV[below] <- lloq / 2
    blq_flag[below] <- 1L
  } else {
    neg <- obs_df$DV < 0
    obs_df$DV[neg]  <- NA_real_
    blq_flag[neg]   <- NA_integer_
  }

  # Build dose rows for ALL subjects
  # If et has no `id` column, it is a single-subject template — replicate
  # EVERY dose row (there may be more than one, e.g. two parallel depots
  # dosed simultaneously, or a multi-dose QD regimen) across n_id subjects.
  # If et has an `id` column (iCov style, A6/A8/A15), dose rows are already
  # expanded one-or-more-per-subject — use them as-is.
  dose_rows <- et[et$EVID == 1, , drop = FALSE]
  has_rate <- "RATE" %in% names(dose_rows) && any(dose_rows$RATE != 0, na.rm = TRUE)
  if (!("id" %in% names(et))) {
    # Single-subject template — replicate every dose row for each subject
    dose_out <- do.call(rbind, lapply(seq_len(n_id), function(sid) {
      data.frame(
        NMID = sid,
        TIME = dose_rows$TIME,
        DV   = 0,
        MDV  = 1L, EVID = 1L,
        AMT  = dose_rows$AMT,
        CMT  = dose_rows$CMT,
        BLQ  = 0L
      )
    }))
    # RATE only needs to be non-zero on the modeled-duration dose row(s);
    # rxode2 requires RATE=-2 there for `dur(<cmt>) <- ...` to activate
    # (see build_et's docstring). Recycle the per-row RATE pattern once
    # per replicated subject block.
    if (has_rate) dose_out$RATE <- rep(dose_rows$RATE, n_id)
  } else {
    # Expanded (iCov) event table — one or more dose rows per subject already
    dose_out <- data.frame(
      NMID = if ("NMID" %in% names(dose_rows)) dose_rows$NMID else dose_rows$id,
      TIME = dose_rows$TIME, DV = 0, MDV = 1L, EVID = 1L,
      AMT  = dose_rows$AMT, CMT = dose_rows$CMT, BLQ = 0L
    )
    if (has_rate) dose_out$RATE <- dose_rows$RATE
  }

  obs_out <- data.frame(
    NMID = obs_df$NMID, TIME = obs_df$TIME, DV = obs_df$DV,
    MDV = ifelse(is.na(obs_df$DV), 1L, 0L),
    EVID = 0L, AMT = 0, CMT = central_cmt, BLQ = blq_flag
  )
  if (has_rate) obs_out$RATE <- 0

  combined <- rbind(dose_out, obs_out)

  # Merge covariates by NMID; strip covs$id to prevent leaks
  covs2 <- covs
  if ("id" %in% names(covs2)) covs2$id <- NULL
  covs2$NMID <- seq_len(n_id)
  combined <- merge(combined, covs2, by = "NMID", sort = FALSE)
  combined <- combined[order(combined$NMID, combined$TIME, -combined$EVID), ]
  rownames(combined) <- NULL
  combined
}

# Per-scenario sub-seed (never touches global stream state)
scn_seed <- function(scn_idx, rep_idx, salt) BASE_SEED + scn_idx * 10000L + rep_idx * 100L + salt

# ============================================================
# Scenario simulators
# ============================================================

sim_A1 <- function(n, seed_base) {
  # 1-cmt oral, first-order absorption, linear elimination
  mod <- rxode2({
    ka <- exp(lka + eta.ka)
    V  <- exp(lV  + eta.V)
    CL <- exp(lCL + eta.CL)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.5), lV = log(70), lCL = log(5))
  omega  <- lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.15, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A2 <- function(n, seed_base) {
  # 2-cmt IV, parallel linear + Michaelis-Menten elimination
  mod <- rxode2({
    V1   <- exp(lV1 + eta.V1)
    V2   <- exp(lV2)
    Q    <- exp(lQ  + eta.Q)
    CL   <- exp(lCL + eta.CL)
    Vmax <- exp(lVmax + eta.Vmax)
    Km   <- exp(lKm)
    d/dt(centr)  <- -CL / V1 * centr - Vmax * (centr / V1) / (Km + centr / V1) -
                    Q / V1 * centr + Q / V2 * periph
    d/dt(periph) <- Q / V1 * centr - Q / V2 * periph
    cp <- centr / V1
  })
  params <- c(lV1 = log(50), lV2 = log(80), lQ = log(10),
              lCL = log(3),  lVmax = log(100), lKm = log(10))
  omega <- lotri(eta.CL ~ 0.09, eta.V1 ~ 0.04, eta.Q ~ 0.04, eta.Vmax ~ 0.09)
  et    <- build_et(n, dose_amt = 500,
                    times = c(0.083, 0.25, 0.5, 1, 2, 4, 8, 12, 24), dose_cmt = 1L)
  covs  <- generate_covariates(n, seed_base + 1L)
  sim   <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df    <- build_nonmem_output(sim, et, n, covs,
                               sigma_prop_sd = 0.1, sigma_add_sd = 0.05,
                               lloq = 0.02, central_cmt = 1L,
                               blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.CL","eta.V1","eta.Q","eta.Vmax"))
  list(df = df, eta = eta_df)
}

sim_A3 <- function(n, seed_base) {
  # Transit (n_tr=3), 1-cmt, linear elimination
  mod <- rxode2({
    ktr <- exp(lktr + eta.ktr)
    ka  <- exp(lka  + eta.ka)
    V   <- exp(lV   + eta.V)
    CL  <- exp(lCL  + eta.CL)
    mtt <- (3 + 1) / ktr
    d/dt(depot) <- transit(3, mtt) - ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lktr = log(2), lka = log(1), lV = log(60), lCL = log(4))
  omega  <- lotri(eta.CL ~ 0.09, eta.V ~ 0.04,
                  eta.ktr ~ 0.09, eta.ka ~ 0.04)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.12, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.CL","eta.V","eta.ktr","eta.ka"))
  list(df = df, eta = eta_df)
}

sim_A4 <- function(n, seed_base) {
  # 1-cmt oral, Michaelis-Menten elimination
  mod <- rxode2({
    ka   <- exp(lka + eta.ka)
    V    <- exp(lV  + eta.V)
    Vmax <- exp(lVmax + eta.Vmax)
    Km   <- exp(lKm)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - Vmax * (centr / V) / (Km + centr / V)
    cp <- centr / V
  })
  params <- c(lka = log(1.2), lV = log(65), lVmax = log(80), lKm = log(8))
  omega  <- lotri(eta.Vmax ~ 0.09, eta.V ~ 0.04, eta.ka ~ 0.09)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.1, sigma_add_sd = 0.03,
                                lloq = 0.02, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.Vmax","eta.V","eta.ka"))
  list(df = df, eta = eta_df)
}

sim_A5 <- function(n, seed_base) {
  # TMDD quasi-steady-state (SC mAb)
  mod <- rxode2({
    ka   <- exp(lka + eta.ka)
    V    <- exp(lV  + eta.V)
    CL   <- exp(lCL + eta.CL)
    R0   <- exp(lR0   + eta.R0)
    KD   <- exp(lKD   + eta.KD)
    kint <- exp(lkint + eta.kint)
    d/dt(depot) <- -ka * depot
    Cfree <- centr / V
    tmdd_rate <- kint * R0 * Cfree / (KD + Cfree)
    d/dt(centr) <- ka * depot - CL * Cfree - tmdd_rate * V
    cp <- Cfree
  })
  params <- c(lka = log(0.02), lV = log(3.5), lCL = log(0.015),
              lR0 = log(10), lKD = log(1), lkint = log(0.03))
  omega  <- lotri(
    eta.ka   ~ 0.04, eta.V    ~ 0.04, eta.CL ~ 0.09,
    eta.R0   ~ 0.06, eta.KD   ~ 0.04, eta.kint ~ 0.06
  )
  times <- c(2, 6, 12, 18, 24, 72, 168, 336, 504, 672, 1008, 1344)
  et    <- build_et(n, dose_amt = 150, times = times, dose_cmt = 1L)
  covs  <- generate_covariates(n, seed_base + 1L)
  sim   <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df    <- build_nonmem_output(sim, et, n, covs,
                               sigma_prop_sd = 0.15, lloq = 0.01,
                               central_cmt = 2L, blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL","eta.R0","eta.KD","eta.kint"))
  list(df = df, eta = eta_df)
}

sim_A6 <- function(n, seed_base) {
  # 1-cmt oral + WT allometry + RENAL + SEX exp-effect on CL
  mod <- rxode2({
    ka <- exp(lka + eta.ka)
    CL <- exp(lCL + eta.CL) * (WT / 70)^0.75 * (1 - 0.4 * RENAL) * exp(theta_sex * SEX)
    V  <- exp(lV  + eta.V)  * (WT / 70)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.5), lV = log(70), lCL = log(5), theta_sex = 0.1)
  omega  <- lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)
  et     <- build_et_icov(n, dose_amt = 100,
                          times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L, include_renal = TRUE)
  icov   <- covs[, c("id","WT","RENAL","SEX")]
  sim    <- rxSolve(mod, params, et, omega = omega, iCov = icov,
                    nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.12, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  df$id  <- NULL
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A7 <- function(n, seed_base) {
  # 2-cmt + saturable (MM) absorption — NODE surrogate test
  mod <- rxode2({
    Vmax_abs <- exp(lVmax_abs + eta.Vmax_abs)
    Km_abs   <- exp(lKm_abs   + eta.Km_abs)
    V1       <- exp(lV1 + eta.V1)
    V2       <- exp(lV2)
    Q        <- exp(lQ  + eta.Q)
    CL       <- exp(lCL + eta.CL)
    d/dt(depot)  <- -Vmax_abs * depot / (Km_abs + depot)
    d/dt(centr)  <-  Vmax_abs * depot / (Km_abs + depot) - CL / V1 * centr -
                     Q / V1 * centr + Q / V2 * periph
    d/dt(periph) <-  Q / V1 * centr - Q / V2 * periph
    cp <- centr / V1
  })
  params <- c(lVmax_abs = log(50), lKm_abs = log(20),
              lV1 = log(50), lV2 = log(80), lQ = log(10), lCL = log(4))
  omega  <- lotri(
    eta.CL ~ 0.09, eta.V1 ~ 0.04, eta.Q ~ 0.04,
    eta.Vmax_abs ~ 0.06, eta.Km_abs ~ 0.06
  )
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.1, sigma_add_sd = 0.03,
                                central_cmt = 2L, blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.CL","eta.V1","eta.Q","eta.Vmax_abs","eta.Km_abs"))
  list(df = df, eta = eta_df)
}

sim_A8 <- function(n, seed_base) {
  # 1-cmt oral + CRCL^0.75 on CL + monotonic autoinduction (was mislabeled "diurnal")
  # CL(t, CRCL) = CL0 * (CRCL/90)^0.75 * exp(-delta * t / 24)
  mod <- rxode2({
    ka  <- exp(lka + eta.ka)
    CL0 <- exp(lCL0 + eta.CL)
    V   <- exp(lV + eta.V)
    CL  <- CL0 * (CRCL / 90)^theta_crcl * exp(-delta_autoind * t / 24)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = 0.6, lCL0 = 1.5, lV = 3.4,
              theta_crcl = 0.75, delta_autoind = 0.15)
  omega  <- lotri(eta.CL ~ 0.04, eta.V ~ 0.05, eta.ka ~ 0.09)
  times  <- c(0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 36, 48)
  et     <- build_et_icov(n, dose_amt = 200, times = times, dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L, include_crcl = TRUE)
  icov   <- covs[, c("id","CRCL")]
  sim    <- rxSolve(mod, params, et, omega = omega, iCov = icov,
                    nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.10, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  df$id  <- NULL
  eta_df <- extract_etas(sim, c("eta.CL","eta.V","eta.ka"))
  list(df = df, eta = eta_df)
}

sim_A9 <- function(n, seed_base) {
  # Erlang absorption (n=3 explicit chain, shared ktr, no terminal ka; ADR-0003 D2),
  # 1-cmt, linear elimination. Distinct from A3's rxode2 transit(n, mtt) (gamma
  # interpolation + terminal ka) -- Erlang lowers to an explicit n-compartment
  # chain with the last state feeding centr directly.
  mod <- rxode2({
    ktr <- exp(lktr + eta.ktr)
    V   <- exp(lV + eta.V)
    CL  <- exp(lCL + eta.CL)
    d/dt(E1) <- -ktr * E1
    d/dt(E2) <- ktr * E1 - ktr * E2
    d/dt(E3) <- ktr * E2 - ktr * E3
    d/dt(centr) <- ktr * E3 - CL / V * centr
    cp <- centr / V
  })
  params <- c(lktr = log(2.0), lV = log(65), lCL = log(4.5))
  omega  <- lotri(eta.ktr ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.12, central_cmt = 4L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ktr","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A10 <- function(n, seed_base) {
  # Parallel first-order absorption: two SIMULTANEOUS first-order depots
  # (fast ka1, slow ka2), bioavailability-fraction split via f(). Distinct
  # from A11's mixed first+zero-order (sequential mechanism mix). frac is a
  # fixed population value (no IIV) -- only ka1/ka2 carry BSV.
  mod <- rxode2({
    ka1  <- exp(lka1 + eta.ka1)
    ka2  <- exp(lka2 + eta.ka2)
    V    <- exp(lV + eta.V)
    CL   <- exp(lCL + eta.CL)
    frac <- 0.6
    d/dt(depot_fast) <- -ka1 * depot_fast
    d/dt(depot_slow) <- -ka2 * depot_slow
    f(depot_fast) <- frac
    f(depot_slow) <- 1 - frac
    d/dt(centr) <- ka1 * depot_fast + ka2 * depot_slow - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka1 = log(2.0), lka2 = log(0.3), lV = log(60), lCL = log(4.0))
  omega  <- lotri(eta.ka1 ~ 0.09, eta.ka2 ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)
  dose_df <- data.frame(TIME = c(0, 0), AMT = c(100, 100), CMT = c(1L, 2L))
  obs_times <- c(0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24)
  et     <- build_et_multi(dose_df, obs_times, obs_cmt = 3L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.10, central_cmt = 3L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka1","eta.ka2","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A11 <- function(n, seed_base) {
  # Mixed first-order + zero-order absorption: a first-order depot and a
  # SEPARATE zero-order depot (modeled duration via dur(depot_zo), RATE=-2
  # on that dose row), both feeding centr; f() splits bioavailability.
  # frac fixed (no IIV) for the same identifiability reason as A10.
  mod <- rxode2({
    ka    <- exp(lka + eta.ka)
    ldurp <- ldur + eta.dur
    durp  <- exp(ldurp)
    V     <- exp(lV + eta.V)
    CL    <- exp(lCL + eta.CL)
    frac  <- 0.55
    d/dt(depot_fo) <- -ka * depot_fo
    dur(depot_zo) <- durp
    d/dt(depot_zo) <- -depot_zo
    f(depot_fo) <- frac
    f(depot_zo) <- 1 - frac
    d/dt(centr) <- ka * depot_fo + depot_zo - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.5), ldur = log(3.0), lV = log(60), lCL = log(4.0))
  omega  <- lotri(eta.ka ~ 0.09, eta.dur ~ 0.04, eta.V ~ 0.04, eta.CL ~ 0.09)
  dose_df <- data.frame(TIME = c(0, 0), AMT = c(100, 100), CMT = c(1L, 2L),
                        RATE = c(0, -2))
  obs_times <- c(0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24)
  et     <- build_et_multi(dose_df, obs_times, obs_cmt = 3L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.10, central_cmt = 3L,
                                blq_seed = seed_base + 3L)
  df$RATE <- NULL
  eta_df <- extract_etas(sim, c("eta.ka","eta.dur","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A12 <- function(n, seed_base) {
  # Standalone zero-order absorption (e.g. matrix-controlled-release oral):
  # dose enters centr directly at a constant rate over `dur` hours via
  # rxode2's modeled-duration infusion (dur(centr) <- durp, RATE=-2 on the
  # dosing row). No depot compartment or explicit influx ODE term.
  mod <- rxode2({
    ldurp <- ldur + eta.dur
    durp  <- exp(ldurp)
    V     <- exp(lV + eta.V)
    CL    <- exp(lCL + eta.CL)
    dur(centr) <- durp
    d/dt(centr) <- -CL / V * centr
    cp <- centr / V
  })
  params <- c(ldur = log(4.0), lV = log(55), lCL = log(4.5))
  omega  <- lotri(eta.dur ~ 0.06, eta.V ~ 0.04, eta.CL ~ 0.09)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 3, 4, 5, 6, 8, 12, 24),
                     dose_cmt = 1L, rate = -2)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.12, central_cmt = 1L,
                                blq_seed = seed_base + 3L)
  df$RATE <- NULL
  eta_df <- extract_etas(sim, c("eta.dur","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A13 <- function(n, seed_base) {
  # Sum of two Inverse Gaussians (SumIG, k=2) absorption -- closed-form
  # input rate (Csajka 2005; Weiss & Wegner 2022). Single-dose only (v0.7
  # limitation, ADR-0003 D4). NOTE: the nlmixr2 emitter's own SumIG ODE
  # references the reserved event-column variable `amt` as a *persistent*
  # per-timepoint value, but rxode2 5.0.2 only exposes `amt` on the exact
  # dosing row (NA at every later observation time) -- verified empirically
  # in this environment; a literal port of the emitter's expression yields
  # NA concentrations everywhere but t=0. This ground-truth simulation
  # substitutes an explicit DOSE constant (valid since v0.7 is single-dose
  # only, so the dose amount is a known compile-time constant here) to
  # produce a physically meaningful absorption profile; this does not
  # change the structural form (the closed-form input-rate function),
  # only how the dose scalar is threaded into it. Per SumIG's own
  # identifiability note (ADR-0003 D5), disposition (CL/V) is kept FIXED
  # (no IIV) here -- only the absorption-shape parameter MT_1 carries BSV.
  mod <- rxode2({
    CL <- 4.0
    V  <- 40.0
    MT_1  <- exp(lMT_1 + eta.MT_1)
    delta_MT_2 <- 3.0
    MT_2  <- MT_1 + delta_MT_2
    RD2_1 <- 0.3
    RD2_2 <- 2.0
    weight_1 <- 0.55
    weight_2 <- 1 - weight_1
    DOSE <- 100
    t_safe <- ifelse(t > 1e-6, t, 1e-6)
    ig_1 <- sqrt(RD2_1 / (2 * 3.141592653589793 * t_safe^3)) *
            exp(-RD2_1 * (t_safe - MT_1)^2 / (2 * MT_1^2 * t_safe))
    ig_2 <- sqrt(RD2_2 / (2 * 3.141592653589793 * t_safe^3)) *
            exp(-RD2_2 * (t_safe - MT_2)^2 / (2 * MT_2^2 * t_safe))
    sumig_input <- weight_1 * ig_1 + weight_2 * ig_2
    d/dt(centr) <- DOSE * sumig_input - CL / V * centr
    cp <- centr / V
  })
  params <- c(lMT_1 = log(0.5))
  omega  <- lotri(eta.MT_1 ~ 0.04)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 24),
                     dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.15, central_cmt = 1L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.MT_1"))
  list(df = df, eta = eta_df)
}

sim_A14 <- function(n, seed_base) {
  # Three-compartment IV bolus, linear elimination. Dose routes directly to
  # centr (IVBolus -- no depot). Sampling extends to 120h with several
  # points beyond 24h so the deep third compartment (V3/Q3) is identifiable
  # and the fit cannot collapse to an apparent 2-cmt model.
  mod <- rxode2({
    V1 <- exp(lV1 + eta.V1)
    V2 <- exp(lV2)
    V3 <- exp(lV3)
    Q2 <- exp(lQ2 + eta.Q2)
    Q3 <- exp(lQ3 + eta.Q3)
    CL <- exp(lCL + eta.CL)
    d/dt(centr)   <- -CL / V1 * centr - Q2 / V1 * centr + Q2 / V2 * periph1 -
                      Q3 / V1 * centr + Q3 / V3 * periph2
    d/dt(periph1) <- Q2 / V1 * centr - Q2 / V2 * periph1
    d/dt(periph2) <- Q3 / V1 * centr - Q3 / V3 * periph2
    cp <- centr / V1
  })
  params <- c(lV1 = log(5), lV2 = log(15), lV3 = log(100),
              lQ2 = log(8), lQ3 = log(1.5), lCL = log(5))
  omega  <- lotri(eta.V1 ~ 0.04, eta.Q2 ~ 0.06, eta.Q3 ~ 0.06, eta.CL ~ 0.09)
  et     <- build_et(n, dose_amt = 250,
                     times = c(0.083, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120),
                     dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.15, central_cmt = 1L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.V1","eta.Q2","eta.Q3","eta.CL"))
  list(df = df, eta = eta_df)
}

build_et_icov_dosearms <- function(n_id, dose_arms, times, dose_cmt) {
  # Like build_et_icov, but each subject's dose AMT is drawn from
  # `dose_arms` (recycled/blocked so subjects are split evenly across
  # arms) instead of a single shared dose. Used for TMDD dose-ranging
  # designs where sub-saturating vs saturating exposure must be spanned
  # to identify the nonlinear (target-mediated) disposition parameters.
  n_arms <- length(dose_arms)
  arm_of <- rep(dose_arms, length.out = n_id)
  rows <- vector("list", n_id * (length(times) + 1L))
  k <- 1L
  for (id in seq_len(n_id)) {
    rows[[k]] <- data.frame(
      id = id, NMID = id, TIME = 0, DV = 0, MDV = 1L, EVID = 1L,
      AMT = arm_of[id], CMT = dose_cmt
    )
    k <- k + 1L
    for (tt in times) {
      rows[[k]] <- data.frame(
        id = id, NMID = id, TIME = tt, DV = NA_real_, MDV = 0L, EVID = 0L,
        AMT = 0, CMT = dose_cmt
      )
      k <- k + 1L
    }
  }
  do.call(rbind, rows)
}

sim_A15 <- function(n, seed_base) {
  # TMDD full binding model (Mager & Jusko 2001) -- distinct from A5's QSS
  # approximation. A 3-arm dose-ranging design (low/mid/high, ~20 subjects
  # per arm) spans sub-saturating to saturating target exposure so kon/koff
  # are identifiable from the shape of the nonlinear-clearance transition;
  # a single dose level cannot separate linear from target-mediated
  # clearance (Xen consensus review). Binding-kinetic parameters
  # (kon, koff, kint) carry no IIV -- only CL/V/R0/ka do -- consistent
  # with the "fix the binding kinetics" identifiability guidance.
  mod <- rxode2({
    ka   <- exp(lka + eta.ka)
    V    <- exp(lV + eta.V)
    R0   <- exp(lR0 + eta.R0)
    kon  <- 0.1
    koff <- 0.1
    kint <- 0.03
    CL   <- exp(lCL + eta.CL)
    kdeg <- koff
    ksyn <- kdeg * R0
    d/dt(depot) <- -ka * depot
    L <- centr / V
    d/dt(centr) <- ka * depot - CL / V * centr - kon * L * R * V + koff * RC * V
    d/dt(R)  <- ksyn - kdeg * R - kon * L * R + koff * RC
    d/dt(RC) <- kon * L * R - koff * RC - kint * RC
    R(0) <- R0
    cp <- centr / V
  })
  params <- c(lka = log(0.02), lV = log(3.0), lR0 = log(20), lCL = log(0.01))
  omega  <- lotri(eta.ka ~ 0.04, eta.V ~ 0.04, eta.R0 ~ 0.06, eta.CL ~ 0.09)
  times  <- c(1, 4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 672)
  et     <- build_et_icov_dosearms(n, dose_arms = c(10, 60, 400), times = times, dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.15, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  df$id  <- NULL
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.R0","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A16 <- function(n, seed_base) {
  # Time-varying (exponential-decay) elimination, UNCONFOUNDED: no covariate
  # is attached to CL, unlike A8's covariate-confounded autoinduction. QD
  # dosing over 14 days (repeated dosing, not single-dose) so the CL decay
  # is separable from distribution kinetics; kdecay chosen for ~35% decline
  # in CL by the end of the 336h window. kdecay is fixed (no IIV).
  mod <- rxode2({
    ka  <- exp(lka + eta.ka)
    V   <- exp(lV + eta.V)
    CL0 <- exp(lCL + eta.CL)
    kdecay <- 0.0015
    CL <- CL0 * exp(-kdecay * t)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.2), lV = log(50), lCL = log(5))
  omega  <- lotri(eta.ka ~ 0.06, eta.V ~ 0.04, eta.CL ~ 0.09)
  dose_times <- seq(0, 312, by = 24)
  dose_df <- data.frame(TIME = dose_times, AMT = rep(200, length(dose_times)),
                        CMT = rep(1L, length(dose_times)))
  obs_times <- c(0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 168, 192, 240, 264, 288,
                312.5, 313, 314, 316, 320, 324, 336)
  et     <- build_et_multi(dose_df, obs_times, obs_cmt = 2L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.10, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A17 <- function(n, seed_base) {
  # Block-structured IIV: genuine positive CL-V correlation (corr ~0.4,
  # var_CL=0.09 [~30% CV], var_V=0.04 [~20% CV], cov=corr*sqrt(var_CL*var_V)
  # = 0.024), combined with an Additive (not proportional) residual error
  # model. These are orthogonal DSL axes (variability structure vs.
  # observation error) so combining them in one spec does not confound
  # either one's estimability (Xen consensus review).
  mod <- rxode2({
    ka <- exp(lka + eta.ka)
    V  <- exp(lV + eta.V)
    CL <- exp(lCL + eta.CL)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.3), lV = log(12), lCL = log(4.5))
  omega  <- lotri(eta.CL + eta.V ~ c(0.09, 0.024, 0.04), eta.ka ~ 0.09)
  et     <- build_et(n, dose_amt = 80,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0, sigma_add_sd = 0.5,
                                central_cmt = 2L, blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A18 <- function(n, seed_base) {
  # Inter-occasion variability (IOV) on ka across 3 dosing occasions spaced
  # a week apart (negligible carryover given the ~5-10h elimination half-
  # life at these CL/V values). Uses rxode2's native nested-omega mechanism
  # (rxode2-nesting.Rmd): an id-level block for IIV and an occ-level block
  # for IOV. rxode2 requires the occasion-indexing event-table column to be
  # literally named `occ` (lowercase, verified empirically in this
  # environment — an `OCC` column is not recognized and errors with
  # "could not find 'occ' in data"); the output CSV renames it to `OCC`
  # to match DSLSpec's OccasionByDoseEpoch(column="OCC") declaration.
  mod <- rxode2({
    ka <- exp(lka + eta.ka + iov.ka)
    V  <- exp(lV + eta.V)
    CL <- exp(lCL + eta.CL)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.2), lV = log(55), lCL = log(4))
  omega  <- lotri(
    lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09) | id(nu = n),
    lotri(iov.ka ~ 0.06) | occ(nu = n * 3L)
  )
  occ_times <- c(0, 168, 336)
  obs_off   <- c(0.5, 1, 2, 4, 8, 12, 24)
  n_rows_per_subj <- length(occ_times) * (1L + length(obs_off))
  rows <- vector("list", n * n_rows_per_subj)
  k <- 1L
  for (id in seq_len(n)) {
    for (occ_idx in seq_along(occ_times)) {
      t0 <- occ_times[occ_idx]
      rows[[k]] <- data.frame(id = id, occ = occ_idx, TIME = t0, DV = 0,
                              MDV = 1L, EVID = 1L, AMT = 100, CMT = 1L)
      k <- k + 1L
      for (off in obs_off) {
        rows[[k]] <- data.frame(id = id, occ = occ_idx, TIME = t0 + off, DV = NA_real_,
                                MDV = 0L, EVID = 0L, AMT = 0, CMT = 2L)
        k <- k + 1L
      }
    }
  }
  et  <- do.call(rbind, rows)
  et  <- et[order(et$id, et$TIME, -et$EVID), ]
  sim <- rxSolve(mod, params, et, omega = omega, seed = seed_base + 2L)
  sim_df <- data.frame(NMID = as.integer(sim$id), TIME = sim$time, occ = sim$occ, CP = sim$cp)

  obs_mask <- et$EVID == 0
  obs_meta <- data.frame(NMID = et$id[obs_mask], TIME = et$TIME[obs_mask], occ = et$occ[obs_mask])
  obs_df <- merge(obs_meta, sim_df, by = c("NMID","TIME","occ"), sort = FALSE)
  obs_df <- obs_df[!duplicated(obs_df[, c("NMID","TIME","occ")]), ]

  set.seed(seed_base + 3L)
  n_obs <- nrow(obs_df)
  obs_df$DV <- obs_df$CP * (1 + rnorm(n_obs, 0, 0.10))
  neg <- obs_df$DV < 0
  obs_df$DV[neg] <- NA_real_

  dose_mask <- et$EVID == 1
  dose_out <- data.frame(
    NMID = et$id[dose_mask], TIME = et$TIME[dose_mask], OCC = et$occ[dose_mask],
    DV = 0, MDV = 1L, EVID = 1L, AMT = et$AMT[dose_mask], CMT = et$CMT[dose_mask], BLQ = 0L
  )
  obs_out <- data.frame(
    NMID = obs_df$NMID, TIME = obs_df$TIME, OCC = obs_df$occ, DV = obs_df$DV,
    MDV = ifelse(is.na(obs_df$DV), 1L, 0L), EVID = 0L, AMT = 0, CMT = 2L, BLQ = 0L
  )
  combined <- rbind(dose_out, obs_out)
  covs   <- generate_covariates(n, seed_base + 1L)
  covs$id <- NULL
  covs$NMID <- seq_len(n)
  combined <- merge(combined, covs, by = "NMID", sort = FALSE)
  combined <- combined[order(combined$NMID, combined$TIME, -combined$EVID), ]
  rownames(combined) <- NULL

  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = combined, eta = eta_df)
}

sim_A19 <- function(n, seed_base) {
  # Hill/sigmoid maturation-form covariate: CL(PNA) = CL0 * PNA^hill /
  # (PNA^hill + TM50^hill), postnatal age (PNA, weeks) covariate drawn from
  # uniform(2, 40) so the cohort meaningfully straddles TM50=15 (populating
  # the sigmoid's steep region, not just its flat asymptotes -- otherwise
  # TM50/hill are unidentifiable, per Xen consensus review).
  mod <- rxode2({
    ka  <- exp(lka + eta.ka)
    CL0 <- exp(lCL + eta.CL)
    V   <- exp(lV + eta.V)
    CL  <- CL0 * PNA^hill / (PNA^hill + TM50^hill)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.0), lCL = log(6.0), lV = log(25), hill = 3, TM50 = 15)
  omega  <- lotri(eta.ka ~ 0.09, eta.CL ~ 0.09, eta.V ~ 0.04)
  times  <- c(0.5, 1, 2, 4, 6, 8, 12, 24)
  et     <- build_et_icov(n, dose_amt = 50, times = times, dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  covs$PNA <- runif(n, min = 2, max = 40)
  icov   <- covs[, c("id","PNA")]
  sim    <- rxSolve(mod, params, et, omega = omega, iCov = icov,
                    nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.12, central_cmt = 2L,
                                blq_seed = seed_base + 3L)
  df$id  <- NULL
  eta_df <- extract_etas(sim, c("eta.ka","eta.CL","eta.V"))
  list(df = df, eta = eta_df)
}

sim_A20 <- function(n, seed_base) {
  # 1-cmt oral, linear elimination, with an ELEVATED LLOQ (vs. A2/A4/A5's
  # ~0.01-0.02) chosen for a ~15-30% BLQ fraction. Shared ground truth for
  # BOTH the BLQ_M3 and BLQ_M4 DSLSpec.observation variants (they differ
  # only in the emitted likelihood, not the raw M3-style data encoding used
  # here -- see scenario_a20a/scenario_a20b in suite_a.py). A faster CL
  # (shorter half-life) than A1 ensures a meaningful late-time censored
  # tail by 48h.
  mod <- rxode2({
    ka <- exp(lka + eta.ka)
    V  <- exp(lV + eta.V)
    CL <- exp(lCL + eta.CL)
    d/dt(depot) <- -ka * depot
    d/dt(centr) <- ka * depot - CL / V * centr
    cp <- centr / V
  })
  params <- c(lka = log(1.5), lV = log(70), lCL = log(6.0))
  omega  <- lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)
  et     <- build_et(n, dose_amt = 100,
                     times = c(0.5, 1, 2, 4, 6, 8, 12, 24, 36, 48), dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)
  df     <- build_nonmem_output(sim, et, n, covs,
                                sigma_prop_sd = 0.15, lloq = 0.12,
                                central_cmt = 2L, blq_seed = seed_base + 3L)
  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL"))
  list(df = df, eta = eta_df)
}

sim_A21 <- function(n, seed_base) {
  # TMDD quasi-steady-state (same structural values as A5) with a genuine
  # multi-analyte observation model: free-drug concentration (DVID=1, "cp")
  # and total-target concentration (DVID=2, "Rtot") sampled at the same
  # timepoints. This is the DSL's only multi-analyte-eligible pairing --
  # DSLSpec.known_prediction_variables() only exposes "C_target_total"
  # (the Rtot state) under TMDDQSS; there is no metabolite/parent-child
  # compartment topology in the DSL yet.
  mod <- rxode2({
    ka   <- exp(lka + eta.ka)
    V    <- exp(lV + eta.V)
    CL   <- exp(lCL + eta.CL)
    R0   <- exp(lR0 + eta.R0)
    KD   <- exp(lKD + eta.KD)
    kint <- exp(lkint + eta.kint)
    d/dt(depot) <- -ka * depot
    KSS <- KD
    Ctot <- Atot / V
    Cfree <- 0.5 * ((Ctot - Rtot - KSS) + sqrt((Ctot - Rtot - KSS)^2 + 4 * KSS * Ctot))
    Rfree <- Rtot * KSS / (KSS + Cfree)
    RC <- Ctot - Cfree
    kdeg <- kint
    ksyn <- kdeg * R0
    d/dt(Atot) <- ka * depot - CL * Cfree - kint * RC * V
    d/dt(Rtot) <- ksyn - kdeg * Rfree - kint * RC
    Atot(0) <- 0
    Rtot(0) <- R0
    cp <- Cfree
  })
  params <- c(lka = log(0.02), lV = log(3.5), lCL = log(0.015),
              lR0 = log(10), lKD = log(1), lkint = log(0.03))
  omega  <- lotri(eta.ka ~ 0.04, eta.V ~ 0.04, eta.CL ~ 0.09,
                  eta.R0 ~ 0.06, eta.KD ~ 0.04, eta.kint ~ 0.06)
  times  <- c(2, 6, 12, 18, 24, 72, 168, 336, 504, 672, 1008, 1344)
  et     <- build_et(n, dose_amt = 150, times = times, dose_cmt = 1L)
  covs   <- generate_covariates(n, seed_base + 1L)
  sim    <- rxSolve(mod, params, et, omega = omega, nSub = n, seed = seed_base + 2L)

  obs_times <- sort(unique(et$TIME[et$EVID == 0]))
  sim_df <- data.frame(NMID = as.integer(sim$sim.id), TIME = sim$time,
                       CP = sim$cp, RTOT = sim$Rtot)
  sim_df <- sim_df[round(sim_df$TIME, 6) %in% round(obs_times, 6), ]
  sim_df <- sim_df[!duplicated(sim_df[, c("NMID","TIME")]), ]

  set.seed(seed_base + 3L)
  n_obs <- nrow(sim_df)
  dv_drug   <- sim_df$CP   * (1 + rnorm(n_obs, 0, 0.15))
  dv_target <- sim_df$RTOT * (1 + rnorm(n_obs, 0, 0.12))
  dv_drug[dv_drug < 0]     <- NA_real_
  dv_target[dv_target < 0] <- NA_real_

  obs_drug <- data.frame(NMID = sim_df$NMID, TIME = sim_df$TIME, DV = dv_drug,
                         MDV = ifelse(is.na(dv_drug), 1L, 0L), EVID = 0L,
                         AMT = 0, CMT = 2L, BLQ = 0L, DVID = 1L)
  obs_target <- data.frame(NMID = sim_df$NMID, TIME = sim_df$TIME, DV = dv_target,
                           MDV = ifelse(is.na(dv_target), 1L, 0L), EVID = 0L,
                           AMT = 0, CMT = 2L, BLQ = 0L, DVID = 2L)

  dose_rows <- et[et$EVID == 1, , drop = FALSE]
  dose_out <- do.call(rbind, lapply(seq_len(n), function(sid) {
    data.frame(NMID = sid, TIME = dose_rows$TIME[1], DV = 0, MDV = 1L, EVID = 1L,
              AMT = dose_rows$AMT[1], CMT = dose_rows$CMT[1], BLQ = 0L, DVID = 0L)
  }))

  combined <- rbind(dose_out, obs_drug, obs_target)
  covs2 <- covs
  covs2$id <- NULL
  covs2$NMID <- seq_len(n)
  combined <- merge(combined, covs2, by = "NMID", sort = FALSE)
  combined <- combined[order(combined$NMID, combined$TIME, -combined$EVID, combined$DVID), ]
  rownames(combined) <- NULL

  eta_df <- extract_etas(sim, c("eta.ka","eta.V","eta.CL","eta.R0","eta.KD","eta.kint"))
  list(df = combined, eta = eta_df)
}

# ============================================================
# Main loop: run every scenario × replicate
# ============================================================

sim_dispatch <- list(
  A1 = sim_A1, A2 = sim_A2, A3 = sim_A3, A4 = sim_A4,
  A5 = sim_A5, A6 = sim_A6, A7 = sim_A7, A8 = sim_A8,
  A9 = sim_A9, A10 = sim_A10, A11 = sim_A11, A12 = sim_A12,
  A13 = sim_A13, A14 = sim_A14, A15 = sim_A15, A16 = sim_A16,
  A17 = sim_A17, A18 = sim_A18, A19 = sim_A19, A20 = sim_A20,
  A21 = sim_A21
)

message(sprintf("Suite A: %d scenarios × %d replicate(s) → %s",
                length(scenario_config), N_REPS, output_dir))

t0 <- Sys.time()

for (rep in seq_len(N_REPS)) {
  for (scn_key in names(scenario_config)) {
    cfg      <- scenario_config[[scn_key]]
    seed_base <- scn_seed(cfg$idx, rep, salt = 0L)
    result   <- sim_dispatch[[scn_key]](n = cfg$n, seed_base = seed_base)

    suffix   <- if (N_REPS == 1L) "" else sprintf("_rep%02d", rep)
    csv_path <- file.path(output_dir, sprintf("%s%s.csv", cfg$name, suffix))
    eta_path <- file.path(output_dir, sprintf("%s%s_eta.csv", cfg$name, suffix))

    write.csv(result$df,  csv_path, row.names = FALSE)
    write.csv(result$eta, eta_path, row.names = FALSE)

    cat(sprintf("  [rep %d/%d] %-30s  rows=%4d  Cmax=%.3f\n",
                rep, N_REPS, cfg$name,
                nrow(result$df),
                max(result$df$DV[result$df$EVID == 0 & !is.na(result$df$DV)])))
  }
}

# ============================================================
# Reference parameters (corrected)
# ============================================================

ref_params <- list(
  `_meta` = list(
    generated_at  = as.character(Sys.Date()),
    sigma_convention = "SD on data scale (not variance)",
    replicates_generated = N_REPS,
    seed_base = BASE_SEED
  ),
  A1 = list(
    ka = 1.5, V = 70, CL = 5,
    omega = list(ka = 0.09, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.15)
  ),
  A2 = list(
    V1 = 50, V2 = 80, Q = 10, CL = 3, Vmax = 100, Km = 10,
    omega = list(CL = 0.09, V1 = 0.04, Q = 0.04, Vmax = 0.09),
    sigma = list(prop = 0.1, add = 0.05)
  ),
  A3 = list(
    n = 3, ktr = 2, ka = 1, V = 60, CL = 4,
    omega = list(CL = 0.09, V = 0.04, ktr = 0.09, ka = 0.04),
    sigma = list(prop = 0.12)
  ),
  A4 = list(
    ka = 1.2, V = 65, Vmax = 80, Km = 8,
    omega = list(Vmax = 0.09, V = 0.04, ka = 0.09),
    sigma = list(prop = 0.1, add = 0.03),
    lloq = 0.02
  ),
  A5 = list(
    ka = 0.02, V = 3.5, R0 = 10, KD = 1, kint = 0.03, CL = 0.015,
    omega = list(ka = 0.04, V = 0.04, CL = 0.09,
                 R0 = 0.06, KD = 0.04, kint = 0.06),
    sigma = list(prop = 0.15),
    lloq = 0.01
  ),
  A6 = list(
    ka = 1.5, V = 70, CL = 5,
    covariates = list(
      WT_on_CL    = list(form = "power", exponent = 0.75, reference = 70),
      WT_on_V     = list(form = "power", exponent = 1.0,  reference = 70),
      RENAL_on_CL = list(form = "categorical_multiplicative", factor = 0.6,
                         prevalence = 0.3),
      SEX_on_CL   = list(form = "exp_theta", theta = 0.1,
                         encoding = "0=F, 1=M")
    ),
    omega = list(ka = 0.09, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.12)
  ),
  A7 = list(
    V1 = 50, V2 = 80, Q = 10, CL = 4,
    absorption = list(type = "saturable_mm", Vmax_abs = 50, Km_abs = 20),
    omega = list(CL = 0.09, V1 = 0.04, Q = 0.04,
                 Vmax_abs = 0.06, Km_abs = 0.06),
    sigma = list(prop = 0.1, add = 0.03)
  ),
  A8 = list(
    ka = 1.822, CL0 = 4.482, V = 29.964,
    covariates = list(
      CRCL_on_CL = list(form = "power", exponent = 0.75, reference = 90),
      time_on_CL = list(form = "monotonic_autoinduction",
                        rate = 0.15,
                        note = "CL(t) = CL0 * exp(-rate*t/24). Monotonic decay, NOT diurnal.")
    ),
    omega = list(CL = 0.04, V = 0.05, ka = 0.09),
    sigma = list(prop = 0.10),
    time_averaged_CL_over_48h_no_covariate = 3.872,   # corrected from 3.678
    static_target_bias_pct = -13.66,                  # corrected from -17.9
    note_on_bias = "APMODE has no autoinduction primitive; a static-CL fit will recover a time-averaged CL of 3.872 vs true CL0 = 4.482. Bias = -13.66% (not -17.9%). Use time-averaged target when benchmarking, not CL0."
  ),
  A9 = list(
    n = 3, ktr = 2.0, V = 65, CL = 4.5,
    omega = list(ktr = 0.09, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.12)
  ),
  A10 = list(
    ka1 = 2.0, ka2 = 0.3, frac = 0.6, V = 60, CL = 4.0,
    omega = list(ka1 = 0.09, ka2 = 0.09, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.10)
  ),
  A11 = list(
    ka = 1.5, dur = 3.0, frac = 0.55, V = 60, CL = 4.0,
    omega = list(ka = 0.09, dur = 0.04, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.10)
  ),
  A12 = list(
    dur = 4.0, V = 55, CL = 4.5,
    omega = list(dur = 0.06, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.12)
  ),
  A13 = list(
    CL = 4.0, V = 40.0, MT_1 = 0.5, MT_2 = 3.5, RD2_1 = 0.3, RD2_2 = 2.0,
    weight_1 = 0.55,
    omega = list(MT_1 = 0.04),
    sigma = list(prop = 0.15),
    note = "CL/V/RD2_1/RD2_2/weight_1 fixed (no IIV) per SumIG identifiability constraint (ADR-0003 D5); only MT_1 carries BSV. k=2 fixed (v0.7)."
  ),
  A14 = list(
    V1 = 5, V2 = 15, V3 = 100, Q2 = 8, Q3 = 1.5, CL = 5,
    omega = list(V1 = 0.04, Q2 = 0.06, Q3 = 0.06, CL = 0.09),
    sigma = list(prop = 0.15)
  ),
  A15 = list(
    ka = 0.02, V = 3.0, R0 = 20, kon = 0.1, koff = 0.1, kint = 0.03, CL = 0.01,
    dose_arms_mg = list(10, 60, 400),
    omega = list(ka = 0.04, V = 0.04, R0 = 0.06, CL = 0.09),
    sigma = list(prop = 0.15),
    note = "kon/koff/kint fixed (no IIV); 3 dose arms (~20 subjects each) span sub-saturating to saturating target engagement."
  ),
  A16 = list(
    ka = 1.2, V = 50, CL0 = 5, kdecay = 0.0015,
    dosing = "200 mg QD x 14 days (t=0..312h)",
    omega = list(ka = 0.06, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.10),
    note = "CL(t) = CL0*exp(-kdecay*t), unconfounded (no covariate). kdecay fixed (no IIV). ~35% CL decline over the 336h window."
  ),
  A17 = list(
    ka = 1.3, V = 12, CL = 4.5,
    omega = list(ka = 0.09, CL = 0.09, V = 0.04, CL_V_cov = 0.024, CL_V_corr = 0.4),
    sigma = list(add = 0.5),
    note = "Block-structured IIV on CL+V (genuine correlation) with an Additive (not proportional) residual error model."
  ),
  A18 = list(
    ka = 1.2, V = 55, CL = 4,
    occasions = 3, occasion_spacing_h = 168,
    omega = list(ka = 0.09, V = 0.04, CL = 0.09),
    iov = list(ka = 0.06),
    sigma = list(prop = 0.10),
    note = "IOV on ka across 3 occasions (occ column, week-spaced dosing)."
  ),
  A19 = list(
    ka = 1.0, CL0 = 6.0, V = 25, hill = 3, TM50 = 15,
    covariates = list(
      PNA_on_CL = list(form = "maturation", hill = 3, tm50 = 15,
                       note = "CL(PNA) = CL0 * PNA^hill / (PNA^hill + TM50^hill); PNA ~ U(2, 40) weeks")
    ),
    omega = list(ka = 0.09, CL = 0.09, V = 0.04),
    sigma = list(prop = 0.12)
  ),
  A20 = list(
    ka = 1.5, V = 70, CL = 6.0,
    omega = list(ka = 0.09, V = 0.04, CL = 0.09),
    sigma = list(prop = 0.15),
    lloq = 0.12,
    note = "Shared ground truth for BOTH BLQ_M3 and BLQ_M4 DSLSpec.observation (scenario_a20a/scenario_a20b)."
  ),
  A21 = list(
    ka = 0.02, V = 3.5, R0 = 10, KD = 1, kint = 0.03, CL = 0.015,
    omega = list(ka = 0.04, V = 0.04, CL = 0.09, R0 = 0.06, KD = 0.04, kint = 0.06),
    sigma = list(free_drug_prop = 0.15, total_target_prop = 0.12),
    note = "Multi-analyte: DVID=1 free drug (cp), DVID=2 total target (Rtot). Same structural values as A5 (TMDD QSS)."
  )
)

jsonlite::write_json(
  ref_params,
  file.path(output_dir, "reference_params.json"),
  pretty = TRUE, auto_unbox = TRUE, na = "null"
)

t1 <- Sys.time()
cat(sprintf("\nSuite A complete in %.1fs. Seed=%d, replicates=%d.\n",
            as.numeric(difftime(t1, t0, units = "secs")), BASE_SEED, N_REPS))
cat(sprintf("Reference parameters written: %s\n",
            file.path(output_dir, "reference_params.json")))
