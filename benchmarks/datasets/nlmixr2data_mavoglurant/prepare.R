# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare mavoglurant PK dataset from nlmixr2data.
#
# Usage: Rscript prepare.R [output_dir]
#
# ~222 subjects, ~2346 observations. Real 2-CMT mGluR5 antagonist (Novartis).
# This is the largest real dataset in nlmixr2data and ideal for stress-testing.

library(nlmixr2data)

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/datasets/nlmixr2data_mavoglurant"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- nlmixr2data::mavoglurant

# Canonicalize column names (nlmixr2data uses lowercase)
canon_map <- c(
  "id" = "NMID", "ID" = "NMID",
  "time" = "TIME", "TIME" = "TIME",
  "dv" = "DV", "DV" = "DV",
  "amt" = "AMT", "AMT" = "AMT",
  "evid" = "EVID", "EVID" = "EVID",
  "cmt" = "CMT", "CMT" = "CMT",
  "mdv" = "MDV", "MDV" = "MDV",
  "wt" = "WT", "WT" = "WT",
  "sex" = "SEX", "SEX" = "SEX",
  "age" = "AGE", "AGE" = "AGE",
  "rate" = "RATE", "RATE" = "RATE"
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

write.csv(df, file.path(output_dir, "mavoglurant.csv"), row.names = FALSE)
cat("Mavoglurant dataset written:", nrow(df), "rows,",
    length(unique(df$NMID)), "subjects\n")
