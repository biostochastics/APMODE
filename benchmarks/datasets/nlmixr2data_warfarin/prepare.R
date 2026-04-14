# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare warfarin PK dataset from nlmixr2data.
#
# Usage: Rscript prepare.R [output_dir]
#
# 32 subjects, PK + PCA (PD). For PK benchmarking we extract only
# the concentration observations (DV_TYPE == "cp" or CMT == 2).

library(nlmixr2data)

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/datasets/nlmixr2data_warfarin"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

df <- nlmixr2data::warfarin

# Canonicalize column names
names(df)[names(df) == "id"] <- "NMID"
names(df)[names(df) == "ID"] <- "NMID"
names(df)[names(df) == "time"] <- "TIME"
names(df)[names(df) == "TIME"] <- "TIME"
names(df)[names(df) == "dv"] <- "DV"
names(df)[names(df) == "DV"] <- "DV"
names(df)[names(df) == "amt"] <- "AMT"
names(df)[names(df) == "AMT"] <- "AMT"
names(df)[names(df) == "evid"] <- "EVID"
names(df)[names(df) == "EVID"] <- "EVID"
names(df)[names(df) == "cmt"] <- "CMT"
names(df)[names(df) == "CMT"] <- "CMT"
names(df)[names(df) == "mdv"] <- "MDV"
names(df)[names(df) == "MDV"] <- "MDV"
names(df)[names(df) == "wt"] <- "WT"
names(df)[names(df) == "WT"] <- "WT"
names(df)[names(df) == "age"] <- "AGE"
names(df)[names(df) == "AGE"] <- "AGE"
names(df)[names(df) == "sex"] <- "SEX"
names(df)[names(df) == "SEX"] <- "SEX"

# Ensure MDV column
if (!"MDV" %in% names(df)) {
  df$MDV <- ifelse(!is.na(df$AMT) & df$AMT > 0, 1L, 0L)
}

# Write full dataset (PK + PD)
write.csv(df, file.path(output_dir, "warfarin_full.csv"), row.names = FALSE)

# Write PK-only subset (CMT == 2 for concentration, or all if no CMT-based split)
if ("CMT" %in% names(df) && any(df$CMT == 2, na.rm = TRUE)) {
  pk_df <- df[df$CMT %in% c(1, 2) | df$EVID == 1, ]
} else {
  pk_df <- df
}
write.csv(pk_df, file.path(output_dir, "warfarin_pk.csv"), row.names = FALSE)

cat("Warfarin dataset written:", nrow(df), "rows (full),",
    nrow(pk_df), "rows (PK only),",
    length(unique(df$NMID)), "subjects\n")
