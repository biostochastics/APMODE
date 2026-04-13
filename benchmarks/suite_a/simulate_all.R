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

# Standard oral dosing protocol (for A1, A3, A4)
oral_event_table <- function(n_id) {
  times <- c(0.5, 1, 2, 4, 6, 8, 12, 24)
  dose_amt <- 100

  rows <- list()
  for (id in seq_len(n_id)) {
    # Dose row
    rows[[length(rows) + 1]] <- data.frame(
      NMID = id, TIME = 0, DV = 0, MDV = 1, EVID = 1,
      AMT = dose_amt, CMT = 1
    )
    # Observation rows
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
      AMT = dose_amt, CMT = 1  # central compartment (rxode2: centr is CMT 1)
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

a1_out <- merge(
  data.frame(
    NMID = a1_sim$sim.id,
    TIME = a1_sim$time,
    DV = a1_sim$cp * (1 + rnorm(nrow(a1_sim), 0, 0.15)),
    MDV = ifelse(a1_sim$evid == 1, 1, 0),
    EVID = a1_sim$evid,
    AMT = ifelse(a1_sim$evid == 1, a1_sim$amt, 0),
    CMT = 1
  ),
  data.frame(NMID = seq_len(N_SUBJECTS), a1_covs),
  by = "NMID"
)
# Fix dose rows
a1_out$DV[a1_out$EVID == 1] <- 0
a1_out <- a1_out[order(a1_out$NMID, a1_out$TIME), ]
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

a2_out <- merge(
  data.frame(
    NMID = a2_sim$sim.id,
    TIME = a2_sim$time,
    DV = a2_sim$cp * (1 + rnorm(nrow(a2_sim), 0, 0.1)) +
         rnorm(nrow(a2_sim), 0, 0.5),
    MDV = ifelse(a2_sim$evid == 1, 1, 0),
    EVID = a2_sim$evid,
    AMT = ifelse(a2_sim$evid == 1, a2_sim$amt, 0),
    CMT = 1  # all central (centr is CMT 1 in rxode2)
  ),
  data.frame(NMID = seq_len(N_SUBJECTS), a2_covs),
  by = "NMID"
)
a2_out$DV[a2_out$EVID == 1] <- 0
a2_out <- a2_out[order(a2_out$NMID, a2_out$TIME), ]
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

a3_out <- merge(
  data.frame(
    NMID = a3_sim$sim.id,
    TIME = a3_sim$time,
    DV = a3_sim$cp * (1 + rnorm(nrow(a3_sim), 0, 0.12)),
    MDV = ifelse(a3_sim$evid == 1, 1, 0),
    EVID = a3_sim$evid,
    AMT = ifelse(a3_sim$evid == 1, a3_sim$amt, 0),
    CMT = 1
  ),
  data.frame(NMID = seq_len(N_SUBJECTS), a3_covs),
  by = "NMID"
)
a3_out$DV[a3_out$EVID == 1] <- 0
a3_out <- a3_out[order(a3_out$NMID, a3_out$TIME), ]
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

a4_out <- merge(
  data.frame(
    NMID = a4_sim$sim.id,
    TIME = a4_sim$time,
    DV = a4_sim$cp * (1 + rnorm(nrow(a4_sim), 0, 0.1)) +
         rnorm(nrow(a4_sim), 0, 0.3),
    MDV = ifelse(a4_sim$evid == 1, 1, 0),
    EVID = a4_sim$evid,
    AMT = ifelse(a4_sim$evid == 1, a4_sim$amt, 0),
    CMT = 1
  ),
  data.frame(NMID = seq_len(N_SUBJECTS), a4_covs),
  by = "NMID"
)
a4_out$DV[a4_out$EVID == 1] <- 0
a4_out <- a4_out[order(a4_out$NMID, a4_out$TIME), ]
write.csv(a4_out, file.path(output_dir, "a4_1cmt_oral_mm.csv"),
          row.names = FALSE)
cat("  A4 done:", nrow(a4_out), "rows\n")


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
            sigma = list(prop = 0.1, add = 0.3))
)

jsonlite::write_json(ref_params,
                     file.path(output_dir, "reference_params.json"),
                     pretty = TRUE, auto_unbox = TRUE)

cat("\nAll Suite A datasets generated in:", output_dir, "\n")
cat("Reference params written to: reference_params.json\n")
cat("Seed used:", SEED, "\n")
