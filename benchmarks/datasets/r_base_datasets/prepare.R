# SPDX-License-Identifier: GPL-2.0-or-later
# R base datasets — Theoph + Indometh
# License: GPL-2 (R core)
# References:
#   Kwan/Breault/Umbenhauer/McMahon/Duggan (1976) J Pharmacokinet Biopharm 4:255-280
#   Boeckmann/Sheiner/Beal (1994) NONMEM Users Guide

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) > 0) args[1] else "benchmarks/datasets/r_base_datasets"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

data(Theoph, package = "datasets")
write.csv(Theoph, file.path(out, "Theoph.csv"), row.names = FALSE)
cat(sprintf("Theoph: %d rows × %d cols × %d subjects\n",
            nrow(Theoph), ncol(Theoph), length(unique(Theoph$Subject))))

data(Indometh, package = "datasets")
write.csv(Indometh, file.path(out, "Indometh.csv"), row.names = FALSE)
cat(sprintf("Indometh: %d rows × %d cols × %d subjects\n",
            nrow(Indometh), ncol(Indometh), length(unique(Indometh$Subject))))
