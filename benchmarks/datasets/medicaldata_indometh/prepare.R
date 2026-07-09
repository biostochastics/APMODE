# SPDX-License-Identifier: GPL-2.0-or-later
# medicaldata::indometh + indo_rct
# Package license: MIT + file LICENSE (verified via DESCRIPTION 2026-07-08)
# Reference: Peter Higgins (medicaldata R package)

suppressMessages(library(medicaldata))
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "benchmarks/datasets/medicaldata_indometh"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

data(indometh, package = "medicaldata")
write.csv(indometh, file.path(out, "indometh.csv"), row.names = FALSE)
cat(sprintf("indometh: %d rows × %d cols\n", nrow(indometh), ncol(indometh)))

data(indo_rct, package = "medicaldata")
write.csv(indo_rct, file.path(out, "indo_rct.csv"), row.names = FALSE)
cat(sprintf("indo_rct: %d rows × %d cols\n", nrow(indo_rct), ncol(indo_rct)))
