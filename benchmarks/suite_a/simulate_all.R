# SPDX-License-Identifier: GPL-2.0-or-later
# Benchmark Suite A: Simulate datasets for A1-A4 scenarios using rxode2.
#
# Usage: Rscript simulate_all.R [output_dir]
# Default output: benchmarks/suite_a/
#
# Each scenario generates a NONMEM-style CSV with:
#   NMID, TIME, DV, MDV, EVID, AMT, CMT, WT, SEX
#
# Reference: PRD v0.3 §5, Suite A

library(rxode2)

# ----- Configuration -----
N_SUBJECTS <- 50
SEED <- 20260413
set.seed(SEED)
RNGkind("L'Ecuyer-CMRG")

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/suite_a"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Covariate generation (shared across scenarios)
generate_covariates <- function(n) {
  data.frame(
    WT = rnorm(n, mean = 70, sd = 15),
    SEX = sample(c("M", "F"), n, replace = TRUE)
  )
}

# Build a proper NONMEM-format output from rxSolve result and event table.
# rxSolve changes EVID internally; we reconstruct NONMEM EVID from the
# original event table structure.
build_nonmem_output <- function(sim, et, n_id, covs, sigma_prop, sigma_add = 0) {
  # Extract simulated concentrations — rxSolve returns many integration
  # steps. We only want the observation times from the event table.
  obs_times <- sort(unique(et$TIME[et$EVID == 0]))

  # rxode2 5.x uses 'id' when event table has 'id', 'sim.id' otherwise
  if ("sim.id" %in% names(sim)) {
    id_col <- sim$sim.id
  } else {
    id_col <- sim$id
  }
  sim_df <- data.frame(
    NMID = id_col,
    TIME = sim$time,
    CP   = sim$cp
  )
  # Filter to observation times only (with tolerance for floating point)
  sim_df <- sim_df[round(sim_df$TIME, 6) %in% round(obs_times, 6), ]

  # Build dose rows from original event table (EVID==1)
  dose_rows <- et[et$EVID == 1, , drop = FALSE]

  # For each subject, take exactly one simulated value per obs time
  obs_list <- list()
  for (id in sort(unique(sim_df$NMID))) {
    sub <- sim_df[sim_df$NMID == id, ]
    sub <- sub[!duplicated(round(sub$TIME, 6)), ]
    obs_list[[length(obs_list) + 1]] <- sub
  }
  obs_df <- do.call(rbind, obs_list)
  # Add residual error
  obs_df$DV <- obs_df$CP * (1 + rnorm(nrow(obs_df), 0, sigma_prop)) +
               rnorm(nrow(obs_df), 0, sigma_add)
  obs_df$DV[obs_df$DV < 0] <- 0  # censor negative concentrations

  # Combine dose + obs
  dose_out <- data.frame(
    NMID = dose_rows$NMID,
    TIME = dose_rows$TIME,
    DV   = 0,
    MDV  = 1L,
    EVID = 1L,
    AMT  = dose_rows$AMT,
    CMT  = dose_rows$CMT
  )
  obs_out <- data.frame(
    NMID = obs_df$NMID,
    TIME = obs_df$TIME,
    DV   = obs_df$DV,
    MDV  = 0L,
    EVID = 0L,
    AMT  = 0,
    CMT  = 1L
  )

  combined <- rbind(dose_out, obs_out)
  combined <- merge(combined,
                    data.frame(NMID = seq_len(n_id), covs),
                    by = "NMID")
  combined <- combined[order(combined$NMID, combined$TIME, -combined$EVID), ]
  combined
}

# Standard oral dosing protocol (for A1, A3, A4)
oral_event_table <- function(n_id) {
  times <- c(0.5, 1, 2, 4, 6, 8, 12, 24)
  dose_amt <- 100

  rows <- list()
  for (id in seq_len(n_id)) {
    rows[[length(rows) + 1]] <- data.frame(
      NMID = id, TIME = 0, DV = 0, MDV = 1, EVID = 1,
      AMT = dose_amt, CMT = 1
    )
    for (t in times) {
      rows[[length(rows) + 1]] <- data.frame(
        NMID = id, TIME = t, DV = NA_real_, MDV = 0, EVID = 0,
        AMT = 0, CMT = 1
      )
    }
  }
  do.call(rbind, rows)
}

# IV bolus dosing protocol (for A2)
iv_event_table <- function(n_id) {
  times <- c(0.083, 0.25, 0.5, 1, 2, 4, 8, 12, 24)
  dose_amt <- 500

  rows <- list()
  for (id in seq_len(n_id)) {
    rows[[length(rows) + 1]] <- data.frame(
      NMID = id, TIME = 0, DV = 0, MDV = 1, EVID = 1,
      AMT = dose_amt, CMT = 1
    )
    for (t in times) {
      rows[[length(rows) + 1]] <- data.frame(
        NMID = id, TIME = t, DV = NA_real_, MDV = 0, EVID = 0,
        AMT = 0, CMT = 1
      )
    }
  }
  do.call(rbind, rows)
}


# ===== Scenario A1: 1-cmt oral, first-order absorption, linear elim =====
cat("Simulating A1...\n")

a1_mod <- rxode2({
  ka <- exp(lka + eta.ka)
  V  <- exp(lV  + eta.V)
  CL <- exp(lCL + eta.CL)
  d/dt(depot) <- -ka * depot
  d/dt(centr) <- ka * depot - CL / V * centr
  cp <- centr / V
})

a1_params <- c(lka = log(1.5), lV = log(70), lCL = log(5))
a1_omega <- lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)

a1_et <- oral_event_table(N_SUBJECTS)
a1_covs <- generate_covariates(N_SUBJECTS)

a1_sim <- rxSolve(
  a1_mod, a1_params, a1_et,
  omega = a1_omega,
  nSub = N_SUBJECTS, seed = SEED
)

a1_out <- build_nonmem_output(a1_sim, a1_et, N_SUBJECTS, a1_covs,
                               sigma_prop = 0.15)
write.csv(a1_out, file.path(output_dir, "a1_1cmt_oral_linear.csv"),
          row.names = FALSE)
cat("  A1 done:", nrow(a1_out), "rows,", N_SUBJECTS, "subjects\n")


# ===== Scenario A2: 2-cmt IV, parallel linear + MM elimination =====
cat("Simulating A2...\n")

a2_mod <- rxode2({
  V1   <- exp(lV1 + eta.V1)
  V2   <- exp(lV2)
  Q    <- exp(lQ)
  CL   <- exp(lCL + eta.CL)
  Vmax <- exp(lVmax + eta.Vmax)
  Km   <- exp(lKm)
  d/dt(centr) <- -CL / V1 * centr - Vmax * (centr / V1) / (Km + centr / V1) - Q / V1 * centr + Q / V2 * periph
  d/dt(periph) <- Q / V1 * centr - Q / V2 * periph
  cp <- centr / V1
})

a2_params <- c(
  lV1 = log(50), lV2 = log(80), lQ = log(10),
  lCL = log(3), lVmax = log(100), lKm = log(10)
)
a2_omega <- lotri(eta.CL ~ 0.09, eta.V1 ~ 0.04, eta.Vmax ~ 0.09)

a2_et <- iv_event_table(N_SUBJECTS)
a2_covs <- generate_covariates(N_SUBJECTS)

a2_sim <- rxSolve(
  a2_mod, a2_params, a2_et,
  omega = a2_omega,
  nSub = N_SUBJECTS, seed = SEED + 1
)

a2_out <- build_nonmem_output(a2_sim, a2_et, N_SUBJECTS, a2_covs,
                               sigma_prop = 0.1, sigma_add = 0.5)
write.csv(a2_out, file.path(output_dir, "a2_2cmt_iv_parallel_mm.csv"),
          row.names = FALSE)
cat("  A2 done:", nrow(a2_out), "rows\n")


# ===== Scenario A3: Transit absorption (n=3), 1-cmt, linear elim =====
cat("Simulating A3...\n")

a3_mod <- rxode2({
  ktr <- exp(lktr + eta.ktr)
  ka  <- exp(lka)
  V   <- exp(lV + eta.V)
  CL  <- exp(lCL + eta.CL)
  mtt <- (3 + 1) / ktr
  d/dt(depot) <- transit(3, mtt) - ka * depot
  d/dt(centr) <- ka * depot - CL / V * centr
  cp <- centr / V
})

a3_params <- c(lktr = log(2), lka = log(1), lV = log(60), lCL = log(4))
a3_omega <- lotri(eta.CL ~ 0.09, eta.V ~ 0.04, eta.ktr ~ 0.09)

a3_et <- oral_event_table(N_SUBJECTS)
a3_covs <- generate_covariates(N_SUBJECTS)

a3_sim <- rxSolve(
  a3_mod, a3_params, a3_et,
  omega = a3_omega,
  nSub = N_SUBJECTS, seed = SEED + 2
)

a3_out <- build_nonmem_output(a3_sim, a3_et, N_SUBJECTS, a3_covs,
                               sigma_prop = 0.12)
write.csv(a3_out, file.path(output_dir, "a3_transit_1cmt_linear.csv"),
          row.names = FALSE)
cat("  A3 done:", nrow(a3_out), "rows\n")


# ===== Scenario A4: 1-cmt oral, Michaelis-Menten elimination =====
cat("Simulating A4...\n")

a4_mod <- rxode2({
  ka   <- exp(lka + eta.ka)
  V    <- exp(lV + eta.V)
  Vmax <- exp(lVmax + eta.Vmax)
  Km   <- exp(lKm)
  d/dt(depot) <- -ka * depot
  d/dt(centr) <- ka * depot - Vmax * (centr / V) / (Km + centr / V)
  cp <- centr / V
})

a4_params <- c(lka = log(1.2), lV = log(65), lVmax = log(80), lKm = log(8))
a4_omega <- lotri(eta.Vmax ~ 0.09, eta.V ~ 0.04, eta.ka ~ 0.09)

a4_et <- oral_event_table(N_SUBJECTS)
a4_covs <- generate_covariates(N_SUBJECTS)

a4_sim <- rxSolve(
  a4_mod, a4_params, a4_et,
  omega = a4_omega,
  nSub = N_SUBJECTS, seed = SEED + 3
)

a4_out <- build_nonmem_output(a4_sim, a4_et, N_SUBJECTS, a4_covs,
                               sigma_prop = 0.1, sigma_add = 0.3)
write.csv(a4_out, file.path(output_dir, "a4_1cmt_oral_mm.csv"),
          row.names = FALSE)
cat("  A4 done:", nrow(a4_out), "rows\n")


# ===== Scenario A5: TMDD quasi-steady-state (SC monoclonal antibody) =====
cat("Simulating A5...\n")

# TMDD QSS model (Gibiansky 2008). Target-mediated clearance produces
# nonlinear PK: dominates at low concentrations, saturates at high.
# This tests whether APMODE can distinguish TMDD from 2-cmt distribution.
a5_mod <- rxode2({
  ka   <- exp(lka + eta.ka)
  V    <- exp(lV + eta.V)
  CL   <- exp(lCL + eta.CL)
  R0   <- exp(lR0)
  KD   <- exp(lKD)
  kint <- exp(lkint)

  d/dt(depot) <- -ka * depot
  Cfree <- centr / V
  # QSS target-mediated elimination: kint * R0 * Cfree / (KD + Cfree)
  tmdd_rate <- kint * R0 * Cfree / (KD + Cfree)
  d/dt(centr) <- ka * depot - CL * Cfree - tmdd_rate * V
  cp <- Cfree
})

a5_params <- c(lka = log(0.02), lV = log(3.5), lCL = log(0.015),
               lR0 = log(10), lKD = log(1), lkint = log(0.03))
a5_omega <- lotri(eta.ka ~ 0.04, eta.V ~ 0.04, eta.CL ~ 0.09)

# SC mAb dosing: 150 mg SC, observations over 8 weeks (hours)
sc_event_table <- function(n_id) {
  # Observation at days 1,3,7,14,21,28,42,56 (in hours)
  times <- c(24, 72, 168, 336, 504, 672, 1008, 1344)
  dose_amt <- 150

  rows <- list()
  for (id in seq_len(n_id)) {
    rows[[length(rows) + 1]] <- data.frame(
      NMID = id, TIME = 0, DV = 0, MDV = 1, EVID = 1,
      AMT = dose_amt, CMT = 1
    )
    for (t in times) {
      rows[[length(rows) + 1]] <- data.frame(
        NMID = id, TIME = t, DV = NA_real_, MDV = 0, EVID = 0,
        AMT = 0, CMT = 1
      )
    }
  }
  do.call(rbind, rows)
}

a5_et <- sc_event_table(N_SUBJECTS)
a5_covs <- generate_covariates(N_SUBJECTS)

a5_sim <- rxSolve(
  a5_mod, a5_params, a5_et,
  omega = a5_omega,
  nSub = N_SUBJECTS, seed = SEED + 4
)

a5_out <- build_nonmem_output(a5_sim, a5_et, N_SUBJECTS, a5_covs,
                               sigma_prop = 0.15)
write.csv(a5_out, file.path(output_dir, "a5_tmdd_qss.csv"),
          row.names = FALSE)
cat("  A5 done:", nrow(a5_out), "rows\n")


# ===== Scenario A6: 1-cmt oral with covariate effects =====
cat("Simulating A6...\n")

# Allometric WT on CL (exponent 0.75) and V (exponent 1.0).
# Categorical RENAL impairment reduces CL by 40%.
a6_mod <- rxode2({
  ka   <- exp(lka + eta.ka)
  CL   <- exp(lCL + eta.CL) * (WT / 70)^0.75 * (1 - 0.4 * RENAL)
  V    <- exp(lV + eta.V) * (WT / 70)
  d/dt(depot) <- -ka * depot
  d/dt(centr) <- ka * depot - CL / V * centr
  cp <- centr / V
})

a6_params <- c(lka = log(1.5), lV = log(70), lCL = log(5))
a6_omega <- lotri(eta.ka ~ 0.09, eta.V ~ 0.04, eta.CL ~ 0.09)

a6_et <- oral_event_table(N_SUBJECTS)
# rxode2 5.x requires 'id' column for iCov
a6_et$id <- a6_et$NMID
# Generate covariates with RENAL indicator (~30% prevalence)
a6_covs <- generate_covariates(N_SUBJECTS)
a6_covs$RENAL <- as.integer(runif(N_SUBJECTS) < 0.3)
a6_covs$id <- seq_len(N_SUBJECTS)

a6_sim <- rxSolve(
  a6_mod, a6_params, a6_et,
  omega = a6_omega,
  iCov = a6_covs,
  nSub = N_SUBJECTS, seed = SEED + 5
)

# RENAL is already in a6_covs; build_nonmem_output merges all covariates
a6_out <- build_nonmem_output(a6_sim, a6_et, N_SUBJECTS, a6_covs,
                               sigma_prop = 0.12)
# Drop the 'id' column added for rxode2 iCov compatibility
a6_out$id <- NULL
write.csv(a6_out, file.path(output_dir, "a6_1cmt_covariates.csv"),
          row.names = FALSE)
cat("  A6 done:", nrow(a6_out), "rows\n")


# ===== Scenario A7: 2-cmt with nonlinear (saturable) absorption =====
cat("Simulating A7...\n")

# Ground truth: Michaelis-Menten absorption (depot → central).
# This is NOT representable by classical DSL absorption modules
# (first-order, zero-order, transit). The hybrid NODE should learn
# this nonlinear absorption rate function.
a7_mod <- rxode2({
  Vmax_abs <- exp(lVmax_abs)
  Km_abs   <- exp(lKm_abs)
  V1       <- exp(lV1 + eta.V1)
  V2       <- exp(lV2)
  Q        <- exp(lQ)
  CL       <- exp(lCL + eta.CL)

  # Saturable absorption: dA_depot/dt = -Vmax_abs * depot / (Km_abs + depot)
  d/dt(depot)  <- -Vmax_abs * depot / (Km_abs + depot)
  d/dt(centr)  <- Vmax_abs * depot / (Km_abs + depot) - CL / V1 * centr - Q / V1 * centr + Q / V2 * periph
  d/dt(periph) <- Q / V1 * centr - Q / V2 * periph
  cp <- centr / V1
})

a7_params <- c(lVmax_abs = log(50), lKm_abs = log(20),
               lV1 = log(50), lV2 = log(80), lQ = log(10), lCL = log(4))
a7_omega <- lotri(eta.CL ~ 0.09, eta.V1 ~ 0.04)

a7_et <- oral_event_table(N_SUBJECTS)
a7_covs <- generate_covariates(N_SUBJECTS)

a7_sim <- rxSolve(
  a7_mod, a7_params, a7_et,
  omega = a7_omega,
  nSub = N_SUBJECTS, seed = SEED + 6
)

a7_out <- build_nonmem_output(a7_sim, a7_et, N_SUBJECTS, a7_covs,
                               sigma_prop = 0.1, sigma_add = 0.3)
write.csv(a7_out, file.path(output_dir, "a7_2cmt_node_absorption.csv"),
          row.names = FALSE)
cat("  A7 done:", nrow(a7_out), "rows\n")


# ===== Scenario A8: 1-cmt oral, time-varying CL + CRCL covariate =====
cat("Simulating A8...\n")

# Ground truth: CL(t, CRCL) = CL0 * (CRCL/90)^theta_crcl *
#                              exp(-delta_diurnal * t / 24).
# Tests whether APMODE can detect static allometric-style renal scaling
# combined with a diurnal clearance rhythm. PRD §5 Suite A: nonstandard
# pharmacology scenarios. 60 subjects × 11 observations.
n_a8 <- 60

a8_mod <- rxode2({
  ka  <- exp(lka)
  CL0 <- exp(lCL0 + eta.CL)
  V   <- exp(lV + eta.V)
  CL  <- CL0 * (CRCL / 90)^theta_crcl * exp(-delta_diurnal * t / 24)
  d/dt(depot) <- -ka * depot
  d/dt(centr) <- ka * depot - CL / V * centr
  cp <- centr / V
})

a8_params <- c(lka = 0.6, lCL0 = 1.5, lV = 3.4,
               theta_crcl = 0.75, delta_diurnal = 0.15)
a8_omega <- lotri(eta.CL ~ 0.04, eta.V ~ 0.05)

# Custom event table: 11 observations per subject at c(0.25, 0.5, 1, 2,
# 4, 8, 12, 18, 24, 36, 48) h post a single 200 mg oral dose.
a8_event_table <- function(n_id) {
  times <- c(0.25, 0.5, 1, 2, 4, 8, 12, 18, 24, 36, 48)
  dose_amt <- 200

  rows <- list()
  for (id in seq_len(n_id)) {
    rows[[length(rows) + 1]] <- data.frame(
      NMID = id, TIME = 0, DV = 0, MDV = 1, EVID = 1,
      AMT = dose_amt, CMT = 1
    )
    for (t in times) {
      rows[[length(rows) + 1]] <- data.frame(
        NMID = id, TIME = t, DV = NA_real_, MDV = 0, EVID = 0,
        AMT = 0, CMT = 1
      )
    }
  }
  do.call(rbind, rows)
}

a8_et <- a8_event_table(n_a8)
a8_et$id <- a8_et$NMID

# Subject-level CRCL covariate: Uniform(30, 150) mL/min
a8_covs <- data.frame(
  id   = seq_len(n_a8),
  CRCL = runif(n_a8, min = 30, max = 150)
)

a8_sim <- rxSolve(
  a8_mod, a8_params, a8_et,
  omega = a8_omega,
  iCov = a8_covs,
  nSub = n_a8, seed = SEED + 7
)

a8_out <- build_nonmem_output(a8_sim, a8_et, n_a8, a8_covs,
                               sigma_prop = 0.10)
# Drop the auxiliary 'id' column added for rxode2 iCov dispatch
a8_out$id <- NULL
write.csv(a8_out, file.path(output_dir, "a8_1cmt_tvcl_covariate.csv"),
          row.names = FALSE)
cat("  A8 done:", nrow(a8_out), "rows,", n_a8, "subjects\n")


# ===== Reference Parameters =====
ref_params <- list(
  A1 = list(ka = 1.5, V = 70, CL = 5,
            omega = list(ka = 0.09, V = 0.04, CL = 0.09),
            sigma = list(prop = 0.15)),
  A2 = list(V1 = 50, V2 = 80, Q = 10, CL = 3, Vmax = 100, Km = 10,
            omega = list(CL = 0.09, V1 = 0.04, Vmax = 0.09),
            sigma = list(prop = 0.1, add = 0.5)),
  A3 = list(n = 3, ktr = 2, ka = 1, V = 60, CL = 4,
            omega = list(CL = 0.09, V = 0.04, ktr = 0.09),
            sigma = list(prop = 0.12)),
  A4 = list(ka = 1.2, V = 65, Vmax = 80, Km = 8,
            omega = list(Vmax = 0.09, V = 0.04, ka = 0.09),
            sigma = list(prop = 0.1, add = 0.3)),
  A5 = list(ka = 0.02, V = 3.5, R0 = 10, KD = 1, kint = 0.03, CL = 0.015,
            omega = list(ka = 0.04, V = 0.04, CL = 0.09),
            sigma = list(prop = 0.15)),
  A6 = list(ka = 1.5, V = 70, CL = 5,
            covariates = list(
              WT_on_CL = list(form = "power", exponent = 0.75, reference = 70),
              WT_on_V = list(form = "power", exponent = 1.0, reference = 70),
              RENAL_on_CL = list(form = "categorical", factor = 0.6)
            ),
            omega = list(ka = 0.09, V = 0.04, CL = 0.09),
            sigma = list(prop = 0.12)),
  A7 = list(V1 = 50, V2 = 80, Q = 10, CL = 4,
            absorption = list(type = "saturable_mm", Vmax_abs = 50, Km_abs = 20),
            omega = list(CL = 0.09, V1 = 0.04),
            sigma = list(prop = 0.1, add = 0.3)),
  A8 = list(ka = 1.822, CL0 = 4.482, V = 29.964,
            covariates = list(
              CRCL_on_CL = list(form = "power", exponent = 0.75, reference = 90),
              time_on_CL = list(form = "diurnal", rate = 0.15, period_hr = 24)
            ),
            omega = list(CL = 0.04, V = 0.05),
            sigma = list(prop = 0.10))
)

jsonlite::write_json(ref_params,
                     file.path(output_dir, "reference_params.json"),
                     pretty = TRUE, auto_unbox = TRUE)

cat("\nAll Suite A datasets generated in:", output_dir, "\n")
cat("Reference params written to: reference_params.json\n")
cat("Seed used:", SEED, "\n")
