# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare Schoemaker 2019 / ACOP 2016 benchmark grid from nlmixr2data.
#
# Usage: Rscript prepare.R [output_dir]
# Default output: benchmarks/datasets/nlmixr2data_schoemaker/
#
# Requires: nlmixr2data (CRAN)
#
# Extracts 12 datasets from the nlmixr2data package and writes them
# as canonical NONMEM-style CSVs. These are the exact datasets used
# in Schoemaker et al. (2019) CPT:PSP 8(12):923-930.
#
# Reference: Schoemaker R et al. (2019). "Performance of the SAEM and FOCEI
# Algorithms in the Open-Source, Nonlinear Mixed Effect Modeling Tool nlmixr."
# doi:10.1002/psp4.12471

library(nlmixr2data)

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/datasets/nlmixr2data_schoemaker"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# The 12 ACOP 2016 simulated datasets in nlmixr2data
datasets <- c(
  "Bolus_1CPT", "Bolus_1CPTMM", "Bolus_2CPT", "Bolus_2CPTMM",
  "Infusion_1CPT", "Infusion_1CPTMM", "Infusion_2CPT", "Infusion_2CPTMM",
  "Oral_1CPT", "Oral_1CPTMM", "Oral_2CPT", "Oral_2CPTMM"
)

# Canonical column mapping for APMODE ingestion
canonicalize <- function(df, dataset_name) {
  # nlmixr2data uses ID/TIME/DV/AMT/EVID/CMT — mostly NONMEM-compatible
  # Standardize column names to uppercase NONMEM convention
  col_map <- c(
    "ID" = "NMID", "id" = "NMID",
    "TIME" = "TIME", "time" = "TIME",
    "DV" = "DV", "dv" = "DV",
    "AMT" = "AMT", "amt" = "AMT",
    "EVID" = "EVID", "evid" = "EVID",
    "CMT" = "CMT", "cmt" = "CMT",
    "MDV" = "MDV", "mdv" = "MDV",
    "RATE" = "RATE", "rate" = "RATE",
    "SS" = "SS", "ss" = "SS",
    "II" = "II", "ii" = "II",
    "ADDL" = "ADDL", "addl" = "ADDL",
    "WT" = "WT", "wt" = "WT",
    "SEX" = "SEX", "sex" = "SEX",
    "DOSE" = "DOSE", "dose" = "DOSE"
  )

  # Rename columns that exist in the data
  for (old_name in names(col_map)) {
    if (old_name %in% names(df)) {
      names(df)[names(df) == old_name] <- col_map[old_name]
    }
  }

  # --- Normalize rxode2/nlmixr2data conventions to NONMEM standard ---

  # EVID: rxode2 uses EVID=2 for "other type" (reset/washout).
  # NONMEM standard: {0=obs, 1=dose, 2=other type, 3=reset, 4=reset+dose}.
  # Drop EVID=2 rows — they are internal rxode2 events not part of the PK data.
  if ("EVID" %in% names(df)) {
    df <- df[df$EVID %in% c(0, 1), ]
  }

  # EVID: rxode2 uses EVID=101 for bolus into depot compartment → normalize to 1
  if ("EVID" %in% names(df)) {
    df$EVID[df$EVID == 101] <- 1L
  }

  # SS: nlmixr2data uses SS=99 for "not applicable" → normalize to 0
  if ("SS" %in% names(df)) {
    df$SS[df$SS == 99] <- 0L
  }

  # Ensure MDV exists
  if (!"MDV" %in% names(df)) {
    df$MDV <- ifelse(!is.na(df$AMT) & df$AMT > 0, 1L, 0L)
  }

  # Ensure EVID exists
  if (!"EVID" %in% names(df)) {
    df$EVID <- ifelse(!is.na(df$AMT) & df$AMT > 0, 1L, 0L)
  }

  # Drop nlmixr2data internal columns not needed for APMODE
  drop_cols <- c("LNDV", "SD", "DOSE", "V", "CL", "KA", "V1", "V2", "Q",
                 "VMAX", "KM", "KA1")
  df <- df[, !(names(df) %in% drop_cols), drop = FALSE]

  df
}

for (ds_name in datasets) {
  cat("Processing", ds_name, "...\n")
  df <- get(ds_name, envir = asNamespace("nlmixr2data"))
  df <- canonicalize(df, ds_name)

  out_file <- file.path(output_dir, paste0(tolower(ds_name), ".csv"))
  write.csv(df, out_file, row.names = FALSE)
  cat("  Written:", out_file, "(", nrow(df), "rows,",
      length(unique(df$NMID)), "subjects )\n")
}

# Write dataset metadata as JSON
meta <- list()
for (ds_name in datasets) {
  df <- get(ds_name, envir = asNamespace("nlmixr2data"))
  id_col <- if ("ID" %in% names(df)) "ID" else "id"
  meta[[ds_name]] <- list(
    n_rows = nrow(df),
    n_subjects = length(unique(df[[id_col]])),
    columns = names(df)
  )
}
jsonlite::write_json(meta, file.path(output_dir, "metadata.json"),
                     pretty = TRUE, auto_unbox = TRUE)

cat("\nSchoemaker grid preparation complete.\n")
cat("Output directory:", output_dir, "\n")
cat("Datasets written:", length(datasets), "\n")
