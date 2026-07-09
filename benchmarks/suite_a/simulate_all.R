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
  A8 = list(name = "a8_1cmt_autoind_covariate", n = 100, lloq = NULL,   idx = 8L)
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
build_et <- function(n_id, dose_amt, times, dose_cmt) {
  n_rows <- length(times) + 1L
  data.frame(
    NMID = rep(0L, n_rows),                   # placeholder; replaced post-sim
    TIME = c(0, times),
    DV   = c(0, rep(NA_real_, length(times))),
    MDV  = c(1L, rep(0L, length(times))),
    EVID = c(1L, rep(0L, length(times))),
    AMT  = c(dose_amt, rep(0, length(times))),
    CMT  = rep(dose_cmt, n_rows)
  )
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
  # If et has one row per subject (iCov style), extract them.
  # If et is single-subject template, replicate across n_id subjects.
  dose_rows <- et[et$EVID == 1, , drop = FALSE]
  if (nrow(dose_rows) == 1L) {
    # Single-subject template — replicate for each simulated subject
    dose_out <- data.frame(
      NMID = seq_len(n_id),
      TIME = dose_rows$TIME[1],
      DV   = 0,
      MDV  = 1L, EVID = 1L,
      AMT  = dose_rows$AMT[1],
      CMT  = dose_rows$CMT[1],
      BLQ  = 0L
    )
  } else {
    # Expanded (iCov) event table — one dose row per subject already
    dose_out <- data.frame(
      NMID = if ("NMID" %in% names(dose_rows)) dose_rows$NMID else dose_rows$id,
      TIME = dose_rows$TIME, DV = 0, MDV = 1L, EVID = 1L,
      AMT  = dose_rows$AMT, CMT = dose_rows$CMT, BLQ = 0L
    )
  }

  obs_out <- data.frame(
    NMID = obs_df$NMID, TIME = obs_df$TIME, DV = obs_df$DV,
    MDV = ifelse(is.na(obs_df$DV), 1L, 0L),
    EVID = 0L, AMT = 0, CMT = central_cmt, BLQ = blq_flag
  )

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

# ============================================================
# Main loop: run every scenario × replicate
# ============================================================

sim_dispatch <- list(
  A1 = sim_A1, A2 = sim_A2, A3 = sim_A3, A4 = sim_A4,
  A5 = sim_A5, A6 = sim_A6, A7 = sim_A7, A8 = sim_A8
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
