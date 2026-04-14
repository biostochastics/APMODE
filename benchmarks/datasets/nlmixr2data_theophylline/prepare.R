# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare theophylline PK dataset from R base + nlmixr2data.
#
# Usage: Rscript prepare.R [output_dir]
#
# Sources both R base Theoph (Upton) and nlmixr2data::theo_sd.
# The nlmixr2data version includes explicit dosing event rows (EVID=1).
#
# Reference: Boeckmann AJ, Sheiner LB, Beal SL (1994) NONMEM Users Guide.

library(nlmixr2data)

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/datasets/nlmixr2data_theophylline"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# --- nlmixr2data::theo_sd (NONMEM-style with dosing events) ---
df <- nlmixr2data::theo_sd

# Canonicalize columns
names(df)[names(df) == "ID"] <- "NMID"

# Normalize rxode2 EVID conventions:
# EVID=101 (rxode2 bolus into depot) -> EVID=1 (NONMEM standard dose)
df$EVID[df$EVID == 101] <- 1L

# Ensure standard columns exist
if (!"MDV" %in% names(df)) {
  df$MDV <- ifelse(df$EVID == 1, 1L, 0L)
}

write.csv(df, file.path(output_dir, "theophylline.csv"), row.names = FALSE)
cat("Theophylline dataset written:", nrow(df), "rows,",
    length(unique(df$NMID)), "subjects\n")

# --- R base Theoph (for cross-reference) ---
theoph <- datasets::Theoph
theoph$NMID <- as.integer(theoph$Subject)
theoph$EVID <- 0L
theoph$MDV <- 0L
theoph$AMT <- 0
theoph$CMT <- 1L

# Add dosing event rows
dose_rows <- data.frame(
  NMID = unique(theoph$NMID),
  Wt = theoph$Wt[match(unique(theoph$NMID), theoph$NMID)],
  Dose = theoph$Dose[match(unique(theoph$NMID), theoph$NMID)],
  Time = 0,
  conc = 0,
  Subject = unique(theoph$Subject),
  EVID = 1L,
  MDV = 1L,
  AMT = theoph$Dose[match(unique(theoph$NMID), theoph$NMID)] *
        theoph$Wt[match(unique(theoph$NMID), theoph$NMID)],
  CMT = 1L
)

theoph_full <- rbind(
  theoph[, c("NMID", "Wt", "Dose", "Time", "conc", "Subject", "EVID", "MDV", "AMT", "CMT")],
  dose_rows
)
theoph_full <- theoph_full[order(theoph_full$NMID, theoph_full$Time, -theoph_full$EVID), ]

# Rename to canonical
names(theoph_full)[names(theoph_full) == "Time"] <- "TIME"
names(theoph_full)[names(theoph_full) == "conc"] <- "DV"
names(theoph_full)[names(theoph_full) == "Wt"] <- "WT"
theoph_full$Subject <- NULL
theoph_full$Dose <- NULL

write.csv(theoph_full, file.path(output_dir, "theophylline_base.csv"),
          row.names = FALSE)
cat("Theophylline (R base) written:", nrow(theoph_full), "rows\n")
