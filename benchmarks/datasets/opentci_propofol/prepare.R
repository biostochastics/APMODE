# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare Eleveld propofol PK dataset from OpenTCI.
#
# Usage: Rscript prepare.R [output_dir]
#
# 1033 subjects, 15433 PK observations. IV infusion, 3-CMT.
# Covariates: age, weight, sex, opioid co-administration.
#
# Citation: Eleveld DJ et al. (2018). "Pharmacokinetic-pharmacodynamic
# model for propofol for broad application in anaesthesia and sedation."
# Br J Anaesth 120(5):942-959. doi:10.1016/j.bja.2018.01.018
#
# Source: https://opentci.org/ (free, no credentials required)
# License: Open access
#
# NOTE: The exact download URL may change. Check opentci.org for the
# current location of the Eleveld propofol dataset.

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/datasets/opentci_propofol"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# The Eleveld dataset is distributed as a CSV or can be extracted
# from the supplementary materials of the 2018 BJA paper.
# OpenTCI hosts it at a stable URL.

raw_dir <- file.path(output_dir, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

raw_file <- file.path(raw_dir, "eleveld_propofol_raw.csv")

if (!file.exists(raw_file)) {
  cat("Eleveld propofol dataset not found.\n")
  cat("Please download from https://opentci.org/ and save as:\n")
  cat("  ", raw_file, "\n")
  cat("\nAlternatively, extract from BJA 2018 supplementary materials.\n")
  cat("Skipping canonicalization.\n")
  quit(status = 0)
}

cat("Reading raw Eleveld dataset...\n")
df <- read.csv(raw_file)

# Canonicalize column names
canon_map <- c(
  "ID" = "NMID", "id" = "NMID",
  "TIME" = "TIME", "time" = "TIME",
  "DV" = "DV", "dv" = "DV",
  "AMT" = "AMT", "amt" = "AMT",
  "EVID" = "EVID", "evid" = "EVID",
  "CMT" = "CMT", "cmt" = "CMT",
  "MDV" = "MDV", "mdv" = "MDV",
  "RATE" = "RATE", "rate" = "RATE",
  "WT" = "WT", "wt" = "WT", "BW" = "WT",
  "AGE" = "AGE", "age" = "AGE",
  "SEX" = "SEX", "sex" = "SEX",
  "OPIOID" = "OPIOID", "opioid" = "OPIOID"
)

for (old in names(canon_map)) {
  if (old %in% names(df)) {
    names(df)[names(df) == old] <- canon_map[old]
  }
}

# Ensure MDV
if (!"MDV" %in% names(df)) {
  df$MDV <- ifelse(!is.na(df$AMT) & df$AMT > 0, 1L, 0L)
}

# Sort
df <- df[order(df$NMID, df$TIME, -df$EVID), ]

write.csv(df, file.path(output_dir, "eleveld_propofol.csv"), row.names = FALSE)
cat("Eleveld propofol dataset written:", nrow(df), "rows,",
    length(unique(df$NMID)), "subjects\n")
